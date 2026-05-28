// ============================================================
//  tracer_falling.metal — Falling-camera GPG ray tracer (MSL)
//
//  One thread = one pixel.
//  Threadgroup size: 16×16.
//  All arithmetic in float. Near-horizon pixels are refined by
//  the CPU pass in falling_renderer.cpp (double precision).
//
//  Physics reference: Lin & Soo 2009, arXiv:0905.3244
// ============================================================
#include <metal_stdlib>
using namespace metal;

// ── FallingCameraParams (mirrors FallingCameraParams_C in C++) ───────────────
struct FallingCameraParams {
    float e[4][4];
    float x[4];
    float M, a, Q, Lambda;
    float r_in, r_out, r_isco, r_escape, r_singularity, r_horizon;
    float disk_brightness, fov_h, h0, r_switch_factor;
    int   max_steps, width, height, pad[3];
};

// ── Flat 4×4 index helper ─────────────────────────────────────────────────────
#define G4(arr,r,c)         ((arr)[(r)*4+(c)])
// Flat 4×4×4 index helper (for Gamma)
#define G444(arr,mu,al,be)  ((arr)[(mu)*16+(al)*4+(be)])

// ── gpg_f_f: GPG gauge function f(r) ─────────────────────────────────────────
static float gpg_f_f(float M, float a, float Q, float Lambda, float r) {
    float R2 = r*r + a*a;
    float Q2 = Q*Q;
    if (Lambda >= 0.0f)
        return sqrt(R2 + Q2 + Lambda*a*a*a*a/3.0f);
    else
        return sqrt(R2*(1.0f - Lambda*r*r/3.0f) + Q2);
}

// ── gpg_covariant_f: g_μν in GPG coordinates (flat output, row-major) ────────
// gLL[mu*4+nu], float precision.
static void gpg_covariant_f(float M, float a, float Q, float Lambda,
                              float r, float theta, thread float* gLL)
{
    float R2   = r*r + a*a;
    float Q2   = Q*Q;
    float ct   = cos(theta), st = sin(theta);
    float cos2 = ct*ct, sin2 = st*st;
    float rho2 = r*r + a*a*cos2;
    float Xi   = 1.0f + Lambda*a*a/3.0f;
    float Xit  = 1.0f + Lambda*a*a*cos2/3.0f;
    float Del  = R2*(1.0f - Lambda*r*r/3.0f) - 2.0f*M*r + Q2;
    float f    = gpg_f_f(M, a, Q, Lambda, r);
    float f2   = f*f;

    float D    = f2 - a*a*Xit*sin2;
    float sqD  = sqrt(max(D, 0.0f));
    float sqFD = sqrt(max(f2 - Del, 0.0f));
    float sqXit= sqrt(max(Xit, 0.0f));
    float rho  = sqrt(rho2);

    float e0T  = sqD / (Xi * rho);
    float e0ph = (sqD > 1e-10f) ? a*sin2*(Xit*R2 - f2) / (Xi*rho*sqD) : 0.0f;
    float e1T  = sqFD / (Xi * rho);
    float e1r  = rho / max(f, 1e-10f);
    float e1ph = -a*sin2*sqFD / max(Xi*rho, 1e-10f);
    float e2th = rho / max(sqXit, 1e-10f);
    float e3T  = (sqD > 1e-10f) ? -a*rho*sqXit*st / (f*Xi*sqD) : 0.0f;
    float e3ph = (sqD > 1e-10f) ?  rho*R2*sqXit*st / (f*Xi*sqD) : 0.0f;

    for (int i=0;i<16;++i) gLL[i] = 0.0f;
    G4(gLL,0,0) = -e0T*e0T + e1T*e1T + e3T*e3T;
    G4(gLL,1,1) = e1r*e1r;
    G4(gLL,2,2) = e2th*e2th;
    G4(gLL,3,3) = -e0ph*e0ph + e1ph*e1ph + e3ph*e3ph;
    G4(gLL,0,1) = G4(gLL,1,0) = e1T*e1r;
    G4(gLL,0,3) = G4(gLL,3,0) = -e0T*e0ph + e1T*e1ph + e3T*e3ph;
    G4(gLL,1,3) = G4(gLL,3,1) = e1r*e1ph;
}

// ── cofactor3_f: 3×3 minor from a flat 4×4 matrix ────────────────────────────
static float cofactor3_f(thread float* m, int r0, int c0) {
    float sub[9]; int ri = 0;
    for (int i=0;i<4;++i) {
        if (i==r0) continue;
        int ci=0;
        for (int j=0;j<4;++j) {
            if (j==c0) continue;
            sub[ri*3+ci++] = m[i*4+j];
        }
        ++ri;
    }
    return sub[0]*(sub[4]*sub[8]-sub[5]*sub[7])
          -sub[1]*(sub[3]*sub[8]-sub[5]*sub[6])
          +sub[2]*(sub[3]*sub[7]-sub[4]*sub[6]);
}

// ── gpg_contravariant_f: g^μν via 4×4 cofactor inverse ───────────────────────
static void gpg_contravariant_f(float M, float a, float Q, float Lambda,
                                  float r, float theta, thread float* gUU)
{
    float gLL[16];
    gpg_covariant_f(M, a, Q, Lambda, r, theta, gLL);

    float det = 0.0f;
    for (int j=0;j<4;++j)
        det += gLL[j] * cofactor3_f(gLL,0,j) * ((j%2==0) ? 1.0f : -1.0f);
    float inv_det = (abs(det) > 1e-20f) ? 1.0f/det : 0.0f;

    for (int i=0;i<4;++i)
        for (int j=0;j<4;++j)
            gUU[i*4+j] = cofactor3_f(gLL,j,i)
                         * (((i+j)%2==0) ? 1.0f : -1.0f)
                         * inv_det;
}

// ── gpg_christoffel_f: Γ^μ_{αβ} via central finite differences ───────────────
// Gamma[mu*16 + al*4 + be] (flat 4×4×4).
// Step sizes larger than CPU to avoid float cancellation: hr=r*1e-4+1e-5, ht=1e-4.
static void gpg_christoffel_f(float M, float a, float Q, float Lambda,
                                float r, float theta,
                                thread float* Gamma)
{
    float hr = r * 1e-4f + 1e-5f;
    float ht = 1e-4f;

    float gp[16], gm[16], gtp[16], gtm[16];
    gpg_covariant_f(M, a, Q, Lambda, r+hr, theta,    gp);
    gpg_covariant_f(M, a, Q, Lambda, r-hr, theta,    gm);
    gpg_covariant_f(M, a, Q, Lambda, r,    theta+ht, gtp);
    gpg_covariant_f(M, a, Q, Lambda, r,    theta-ht, gtm);

    float dgr[16], dgt[16];
    for (int i=0;i<16;++i) {
        dgr[i] = (gp[i]  - gm[i])  / (2.0f*hr);
        dgt[i] = (gtp[i] - gtm[i]) / (2.0f*ht);
    }

    float gUU[16];
    gpg_contravariant_f(M, a, Q, Lambda, r, theta, gUU);

    for (int mu=0;mu<4;++mu)
        for (int al=0;al<4;++al)
            for (int be=0;be<4;++be) {
                float s = 0.0f;
                for (int nu=0;nu<4;++nu) {
                    // ∂_α g_νβ: coord=1→r, 2→θ, else 0 (stationary+axisymmetric)
                    float dg_al = (al==1) ? dgr[nu*4+be]
                                : (al==2) ? dgt[nu*4+be] : 0.0f;
                    float dg_be = (be==1) ? dgr[nu*4+al]
                                : (be==2) ? dgt[nu*4+al] : 0.0f;
                    float dg_nu = (nu==1) ? dgr[al*4+be]
                                : (nu==2) ? dgt[al*4+be] : 0.0f;
                    s += gUU[mu*4+nu] * (dg_al + dg_be - dg_nu);
                }
                G444(Gamma,mu,al,be) = 0.5f * s;
            }
}

// ── photon_step_f: single RK4 step for null geodesic ─────────────────────────
static void photon_step_f(float M, float a, float Q, float Lambda,
                           thread float* x, thread float* k, float dlam)
{
    float Gamma[64];
    float dx1[4],dk1[4], dx2[4],dk2[4], dx3[4],dk3[4], dx4[4],dk4[4];
    float xt[4], kt[4];

    // k1
    gpg_christoffel_f(M, a, Q, Lambda, x[1], x[2], Gamma);
    for (int mu=0;mu<4;++mu) {
        dx1[mu] = k[mu];
        float acc=0.0f;
        for (int al=0;al<4;++al) for (int be=0;be<4;++be)
            acc -= G444(Gamma,mu,al,be)*k[al]*k[be];
        dk1[mu] = acc;
    }
    // k2
    for (int i=0;i<4;++i){xt[i]=x[i]+0.5f*dlam*dx1[i]; kt[i]=k[i]+0.5f*dlam*dk1[i];}
    gpg_christoffel_f(M, a, Q, Lambda, xt[1], xt[2], Gamma);
    for (int mu=0;mu<4;++mu) {
        dx2[mu] = kt[mu];
        float acc=0.0f;
        for (int al=0;al<4;++al) for (int be=0;be<4;++be)
            acc -= G444(Gamma,mu,al,be)*kt[al]*kt[be];
        dk2[mu] = acc;
    }
    // k3
    for (int i=0;i<4;++i){xt[i]=x[i]+0.5f*dlam*dx2[i]; kt[i]=k[i]+0.5f*dlam*dk2[i];}
    gpg_christoffel_f(M, a, Q, Lambda, xt[1], xt[2], Gamma);
    for (int mu=0;mu<4;++mu) {
        dx3[mu] = kt[mu];
        float acc=0.0f;
        for (int al=0;al<4;++al) for (int be=0;be<4;++be)
            acc -= G444(Gamma,mu,al,be)*kt[al]*kt[be];
        dk3[mu] = acc;
    }
    // k4
    for (int i=0;i<4;++i){xt[i]=x[i]+dlam*dx3[i]; kt[i]=k[i]+dlam*dk3[i];}
    gpg_christoffel_f(M, a, Q, Lambda, xt[1], xt[2], Gamma);
    for (int mu=0;mu<4;++mu) {
        dx4[mu] = kt[mu];
        float acc=0.0f;
        for (int al=0;al<4;++al) for (int be=0;be<4;++be)
            acc -= G444(Gamma,mu,al,be)*kt[al]*kt[be];
        dk4[mu] = acc;
    }
    // Combine
    for (int i=0;i<4;++i) {
        x[i] += (dlam/6.0f)*(dx1[i]+2*dx2[i]+2*dx3[i]+dx4[i]);
        k[i] += (dlam/6.0f)*(dk1[i]+2*dk2[i]+2*dk3[i]+dk4[i]);
    }
}

// ── shade_pixel_f: Page-Thorne shading (mirrors shade_falling_pixel in C++) ──
static uchar4 shade_pixel_f(int outcome, float r_hit, float redshift,
                              float r_isco, float r_out, float disk_brightness)
{
    uint8_t R=0, G=0, B=0;
    if (outcome == 0) {
        R = G = B = 30;
    } else if (outcome == 1) {
        float lum = 0.0f;
        if (r_hit > r_isco && r_hit <= r_out) {
            float x_pt     = sqrt(r_isco / r_hit);
            lum = (1.0f - x_pt) / (r_hit * r_hit * r_hit);
            float r_peak   = 3.0f * r_isco;
            float x_peak   = sqrt(r_isco / r_peak);
            float lum_peak = (1.0f - x_peak) / (r_peak*r_peak*r_peak);
            if (lum_peak > 1e-30f) lum /= lum_peak;
            lum *= pow(max(redshift, 0.0f), 4.0f);
            lum  = min(lum * disk_brightness, 1.0f);
        }
        if (redshift > 1.0f) {
            R = (uint8_t)min(255.0f, 255.0f * lum);
            G = (uint8_t)min(255.0f, 210.0f * lum);
            B = (uint8_t)min(255.0f, 100.0f * lum);
        } else {
            R = (uint8_t)min(255.0f, 220.0f * lum);
            G = (uint8_t)min(255.0f, 100.0f * lum * redshift);
            B = 0;
        }
    }
    // outcome 2 (singularity), 3 (trapped) → black
    return uchar4(R, G, B, 255);
}

// ── photon_init_f: null photon k^μ for pixel (px,py) from tetrad ──────────────
static void photon_init_f(uint px, uint py,
                           constant FallingCameraParams& fp,
                           thread float* k_out)
{
    float W    = float(fp.width);
    float H    = float(fp.height);
    float fov_v= fp.fov_h * H / W;
    float alpha= fp.fov_h * (float(px) - W*0.5f) / (W - 1.0f);
    float beta = fov_v   * (H*0.5f - float(py))  / (H - 1.0f);

    float nx = sin(beta)*cos(alpha);
    float ny = sin(beta)*sin(alpha);
    float nz = cos(beta);

    // k^μ = e[0]^μ + nx·e[1]^μ + ny·e[2]^μ + nz·e[3]^μ
    float k[4];
    for (int mu=0;mu<4;++mu)
        k[mu] = fp.e[0][mu] + nx*fp.e[1][mu] + ny*fp.e[2][mu] + nz*fp.e[3][mu];

    // Enforce null: rescale k^T so g_μν k^μ k^ν = 0
    float gLL[16];
    gpg_covariant_f(fp.M, fp.a, fp.Q, fp.Lambda, fp.x[1], fp.x[2], gLL);
    float A = G4(gLL,0,0);
    float B = 0.0f, C = 0.0f;
    for (int mu=1;mu<4;++mu) B += 2.0f*G4(gLL,0,mu)*k[mu];
    for (int mu=1;mu<4;++mu)
        for (int nu=1;nu<4;++nu)
            C += G4(gLL,mu,nu)*k[mu]*k[nu];
    float disc = B*B - 4.0f*A*C;
    if (disc >= 0.0f) {
        float kT1 = (-B + sqrt(disc))/(2.0f*A);
        float kT2 = (-B - sqrt(disc))/(2.0f*A);
        k[0] = (kT1 > kT2) ? kT1 : kT2;
    }
    for (int i=0;i<4;++i) k_out[i] = k[i];
}

// ── trace_falling_pixel: main kernel — one thread per pixel ───────────────────
kernel void trace_falling_pixel(
    device uchar4*                   rgb_out  [[buffer(0)]],
    device float*                    rmin_out [[buffer(1)]],
    constant FallingCameraParams&    fp       [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(fp.width) || gid.y >= uint(fp.height)) return;

    // Initialise photon at camera position fp.x
    float k[4], x[4];
    for (int i=0;i<4;++i) x[i] = fp.x[i];
    photon_init_f(gid.x, gid.y, fp, k);

    float r_min      = x[1];
    float dlam       = fp.h0;
    int   outcome    = 3;       // trapped default
    float r_hit      = 0.0f;
    float redshift   = 1.0f;
    float prev_theta = x[2];
    float prev_r     = x[1];

    for (int step=0; step < fp.max_steps; ++step) {
        if (x[1] < r_min) r_min = x[1];

        // Escape
        if (x[1] >= fp.r_escape) {
            outcome = 0; r_hit = x[1]; break;
        }
        // Singularity
        if (x[1] < fp.r_singularity) {
            outcome = 2; break;
        }

        // Disk crossing: sign change of (θ − π/2) within [r_in, r_out]
        float cur_theta = x[2];
        if (x[1] >= fp.r_in && x[1] <= fp.r_out) {
            float dp = prev_theta - M_PI_F/2.0f;
            float dc = cur_theta  - M_PI_F/2.0f;
            if (dp * dc < 0.0f) {
                outcome = 1;
                float t = abs(dp) / (abs(dp) + abs(dc));
                r_hit = prev_r + t*(x[1] - prev_r);

                // Redshift g-factor: Keplerian Omega_K, g_μν at r_hit equatorial
                float gLL_d[16];
                gpg_covariant_f(fp.M, fp.a, fp.Q, fp.Lambda,
                                r_hit, M_PI_F/2.0f, gLL_d);
                float sqrtM = sqrt(fp.M);
                float Omega = sqrtM / (pow(r_hit, 1.5f) + fp.a * sqrtM);
                float N2 = -(G4(gLL_d,0,0)
                           + 2.0f*G4(gLL_d,0,3)*Omega
                           + G4(gLL_d,3,3)*Omega*Omega);
                if (N2 > 1e-20f) {
                    float ut_em  = 1.0f / sqrt(N2);
                    float uph_em = Omega * ut_em;
                    float k_low[4] = {0,0,0,0};
                    for (int mu=0;mu<4;++mu)
                        for (int nu=0;nu<4;++nu)
                            k_low[mu] += G4(gLL_d,mu,nu)*k[nu];
                    float k_u_emit = k_low[0]*ut_em + k_low[3]*uph_em;
                    float k_u_obs  = abs(k_low[0]);
                    if (abs(k_u_emit) > 1e-20f)
                        redshift = k_u_obs / abs(k_u_emit);
                }
                break;
            }
        }
        prev_theta = cur_theta;
        prev_r     = x[1];

        // Adaptive step: fine near horizon
        dlam = (x[1] < fp.r_horizon * 3.0f) ? 0.005f : fp.h0;
        photon_step_f(fp.M, fp.a, fp.Q, fp.Lambda, x, k, dlam);
    }

    uint idx = gid.y * uint(fp.width) + gid.x;
    rgb_out [idx] = shade_pixel_f(outcome, r_hit, redshift,
                                   fp.r_isco, fp.r_out, fp.disk_brightness);
    rmin_out[idx] = r_min;
}
