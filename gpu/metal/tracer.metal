// ============================================================
//  tracer.metal — KNdS geodesic ray-tracer (Metal / MSL)
//
//  One thread  =  one pixel.
//  Thread-group size: 16×16 (256 threads per group).
//
//  MSL is C++14-compatible, so most of our maths ports
//  directly — no classes/templates from C++ standard library.
// ============================================================
#include <metal_stdlib>
using namespace metal;

// ── KNdS parameters (passed as uniform) ─────────────────────
struct KNdSParams {
    float M;
    float a;
    float Q;
    float Lambda;
    float r_horizon;
    float r_isco;
    float r_disk_out;
};

// ── Camera uniform ────────────────────────────────────────────
struct CameraParams {
    float r_obs;
    float theta_obs;   // radians
    float fov_h;       // radians
    int   width;
    int   height;
};

// ── Pixel output ──────────────────────────────────────────────
struct PixelResult {
    uint8_t r, g, b, a;
};

// ── KNdS metric helpers ───────────────────────────────────────
static float Sigma(float r, float theta, float a) {
    float ct = cos(theta);
    return r*r + a*a*ct*ct;
}
static float Delta_r(float r, float M, float a, float Q, float L) {
    return (r*r + a*a)*(1.0f - L*r*r/3.0f) - 2.0f*M*r + Q*Q;
}
static float Delta_th(float theta, float a, float L) {
    float ct = cos(theta);
    return 1.0f + L*a*a*ct*ct/3.0f;
}
static float Xi(float a, float L) { return 1.0f + L*a*a/3.0f; }

static float keplerian_omega(float r, float M, float a, float Q, float L) {
    const float Meff = M - Q*Q/(2.0f*r) + L*a*r*r/3.0f;
    const float sq   = sqrt(max(Meff, 0.0f));
    const float den  = r*sqrt(r) + a*sq;
    return (abs(den) > 1e-12f) ? (sq/den) : 0.0f;
}

// Contravariant metric g^μν components in BL
static void gUU(float r, float theta,
                float M, float a, float Q, float L,
                thread float guu[4][4]) {
    const float sig  = Sigma(r, theta, a);
    const float dr   = Delta_r(r, M, a, Q, L);
    const float dth  = Delta_th(theta, a, L);
    const float xi   = Xi(a, L);
    const float xi2  = xi*xi;
    const float st   = sin(theta);
    const float st2  = st*st;
    const float r2a2 = r*r + a*a;
    const float pre  = sig*dr*dth;

    for (int i=0;i<4;i++) for (int j=0;j<4;j++) guu[i][j]=0.0f;

    guu[0][0] = -xi2*(dth*r2a2*r2a2 - dr*a*a*st2) / pre;
    guu[0][3] = guu[3][0] = a*xi2*(dr - dth*r2a2) / pre;
    guu[1][1] = dr/sig;
    guu[2][2] = dth/sig;
    if (st2 > 1e-10f)
        guu[3][3] = xi2*(dr - dth*a*a*st2) / (pre*st2);
}

// Hamiltonian H = 0.5 g^μν p_μ p_ν
static float hamiltonian(float r, float theta,
                         float pr, float pth,
                         float pt, float pphi,
                         float M, float a, float Q, float L) {
    float gu[4][4];
    gUU(r, theta, M, a, Q, L, gu);
    return 0.5f*(gu[0][0]*pt*pt + 2.0f*gu[0][3]*pt*pphi
               + gu[1][1]*pr*pr + gu[2][2]*pth*pth
               + gu[3][3]*pphi*pphi);
}

// RHS: dr/dλ, dθ/dλ, dp_r/dλ, dp_θ/dλ
static void geodesic_rhs(float r, float theta,
                          float pr, float pth,
                          float pt, float pphi,
                          float M, float a, float Q, float L,
                          thread float& dr_out, thread float& dth_out,
                          thread float& dpr_out, thread float& dpth_out) {
    float gu[4][4];
    gUU(r, theta, M, a, Q, L, gu);

    dr_out  = gu[1][1]*pr;
    dth_out = gu[2][2]*pth;

    const float er  = 1e-5f*(abs(r)+0.1f);
    const float eth = 1e-6f;

    dpr_out  = -(hamiltonian(r+er, theta, pr, pth, pt, pphi, M, a, Q, L)
               - hamiltonian(r-er, theta, pr, pth, pt, pphi, M, a, Q, L)) / (2.0f*er);

    dpth_out = -(hamiltonian(r, theta+eth, pr, pth, pt, pphi, M, a, Q, L)
               - hamiltonian(r, theta-eth, pr, pth, pt, pphi, M, a, Q, L)) / (2.0f*eth);
}

// RK4 step
static void rk4(thread float& r, thread float& theta,
                thread float& pr, thread float& pth,
                float pt, float pphi,
                float M, float a, float Q, float L,
                float dlam) {
    float dr1,dth1,dpr1,dpth1;
    float dr2,dth2,dpr2,dpth2;
    float dr3,dth3,dpr3,dpth3;
    float dr4,dth4,dpr4,dpth4;

    geodesic_rhs(r,               theta,               pr,               pth,               pt,pphi,M,a,Q,L, dr1,dth1,dpr1,dpth1);
    geodesic_rhs(r+.5f*dlam*dr1, theta+.5f*dlam*dth1, pr+.5f*dlam*dpr1, pth+.5f*dlam*dpth1, pt,pphi,M,a,Q,L, dr2,dth2,dpr2,dpth2);
    geodesic_rhs(r+.5f*dlam*dr2, theta+.5f*dlam*dth2, pr+.5f*dlam*dpr2, pth+.5f*dlam*dpth2, pt,pphi,M,a,Q,L, dr3,dth3,dpr3,dpth3);
    geodesic_rhs(r+    dlam*dr3, theta+    dlam*dth3, pr+    dlam*dpr3, pth+    dlam*dpth3, pt,pphi,M,a,Q,L, dr4,dth4,dpr4,dpth4);

    r     += dlam/6.0f*(dr1   +2.0f*dr2   +2.0f*dr3   +dr4);
    theta += dlam/6.0f*(dth1  +2.0f*dth2  +2.0f*dth3  +dth4);
    pr    += dlam/6.0f*(dpr1  +2.0f*dpr2  +2.0f*dpr3  +dpr4);
    pth   += dlam/6.0f*(dpth1 +2.0f*dpth2 +2.0f*dpth3 +dpth4);
}

// Covariant metric g_μν for initial conditions
static void gLL_BL(float r, float theta,
                   float M, float a, float Q, float L,
                   thread float gll[4][4]) {
    const float sig  = Sigma(r, theta, a);
    const float dr   = Delta_r(r, M, a, Q, L);
    const float dth  = Delta_th(theta, a, L);
    const float xi2  = Xi(a,L)*Xi(a,L);
    const float st   = sin(theta);
    const float st2  = st*st;
    const float r2a2 = r*r + a*a;
    const float pre  = sig*xi2;

    for (int i=0;i<4;i++) for (int j=0;j<4;j++) gll[i][j]=0.0f;

    gll[0][0] = (-dr + dth*a*a*st2) / pre;
    gll[0][3] = gll[3][0] = a*st2*(dr - dth*r2a2) / pre;
    gll[1][1] = sig/dr;
    gll[2][2] = sig/dth;
    gll[3][3] = st2*(dth*r2a2*r2a2 - dr*a*a*st2) / pre;
}

// ── Colour helpers ────────────────────────────────────────────
static float4 blackbody_rgb(float T) {
    T = clamp(T, 800.0f, 4e4f);
    float t = log10(T/800.0f) / log10(4e4f/800.0f);
    float R, G, B;
    if (t < 0.25f) {
        R=1.0f; G=t/0.25f*0.4f; B=0.0f;
    } else if (t < 0.50f) {
        float f=(t-0.25f)/0.25f; R=1.0f; G=0.4f+f*0.4f; B=f*0.3f;
    } else if (t < 0.75f) {
        float f=(t-0.50f)/0.25f; R=1.0f; G=0.8f+f*0.2f; B=0.3f+f*0.5f;
    } else {
        float f=(t-0.75f)/0.25f; R=1.0f-f*0.2f; G=1.0f; B=0.8f+f*0.2f;
    }
    return float4(clamp(R,0.0f,1.0f), clamp(G,0.0f,1.0f), clamp(B,0.0f,1.0f), 1.0f);
}

// ── Main compute kernel ───────────────────────────────────────
kernel void trace_pixel(
    device uint32_t*   output    [[buffer(0)]],
    constant KNdSParams& kp      [[buffer(1)]],
    constant CameraParams& cp    [[buffer(2)]],
    uint2              gid       [[thread_position_in_grid]])
{
    const int px = (int)gid.x;
    const int py = (int)gid.y;
    if (px >= cp.width || py >= cp.height) return;

    const float M = kp.M, a = kp.a, Q = kp.Q, L = kp.Lambda;

    // ── Pixel → (α, β) ───────────────────────────────────────
    const int span = max(cp.width - 1, 1);
    const float alpha = cp.fov_h*(float(px) - 0.5f*(cp.width-1))  / float(span);
    const float beta  = cp.fov_h*(0.5f*(cp.height-1) - float(py)) / float(span);

    // ── Initial conditions (approx. flat at large r) ──────────
    const float r0  = cp.r_obs;
    const float th0 = cp.theta_obs;

    float gll[4][4];
    gLL_BL(r0, th0, M, a, Q, L, gll);

    float gu[4][4];
    gUU(r0, th0, M, a, Q, L, gu);

    const float ca = cos(alpha), sa = sin(alpha);
    const float cb = cos(beta),  sb = sin(beta);

    const float sqrt_grr   = sqrt(abs(gll[1][1]));
    const float sqrt_gthth = sqrt(abs(gll[2][2]));
    const float sqrt_gphph = sqrt(abs(gll[3][3]));

    const float pUr   = -ca*cb / sqrt_grr;
    const float pUth  = -sb    / sqrt_gthth;
    const float pUphi =  sa*cb / sqrt_gphph;

    float pt   = gll[0][0]*1.0f + gll[0][3]*pUphi;
    float pr   = gll[1][1]*pUr;
    float pth  = gll[2][2]*pUth;
    float pphi = gll[3][0]*1.0f + gll[3][3]*pUphi;

    // Null condition: solve A pt² + B pt + C = 0
    const float A = gu[0][0];
    const float B = 2.0f*gu[0][3]*pphi;
    const float C = gu[1][1]*pr*pr + gu[2][2]*pth*pth + gu[3][3]*pphi*pphi;
    const float disc = B*B - 4.0f*A*C;
    if (disc >= 0.0f && abs(A) > 1e-15f) {
        const float sq  = sqrt(disc);
        const float pt1 = (-B - sq)/(2.0f*A);
        const float pt2 = (-B + sq)/(2.0f*A);
        pt = (pt1 < 0.0f) ? pt1 : pt2;
        if (pt > 0.0f) pt = min(pt1, pt2);
    }

    // ── Trace ─────────────────────────────────────────────────
    float r = r0, theta = th0;
    float dlam = 1.0f;
    float prev_r = r0;
    float prev_cos = cos(th0);

    uint32_t colour = 0xFF000000u;  // black (ABGR)

    for (int iter = 0; iter < 200000; ++iter) {
        // Adaptive step (simple: fixed tolerance)
        float r_h = r, th_h = theta, pr_h = pr, pth_h = pth;
        rk4(r_h, th_h, pr_h, pth_h, pt, pphi, M, a, Q, L, dlam);

        float r_f = r, th_f = theta, pr_f = pr, pth_f = pth;
        rk4(r_f, th_f, pr_f, pth_f, pt, pphi, M, a, Q, L, dlam*0.5f);
        rk4(r_f, th_f, pr_f, pth_f, pt, pphi, M, a, Q, L, dlam*0.5f);

        const float err = length(float4(r_h-r_f, th_h-th_f,
                                        pr_h-pr_f, pth_h-pth_f)) / 15.0f;
        const float tol = 1e-5f;
        if (err < tol || dlam < 1e-8f) {
            r = r_f; theta = th_f; pr = pr_f; pth = pth_f;
            float sc = (err > 1e-10f) ? 0.9f*pow(tol/err, 0.2f) : 2.0f;
            dlam = clamp(dlam*sc, 1e-8f, 50.0f);
        } else {
            dlam = clamp(dlam*0.9f*pow(tol/err, 0.25f), 1e-8f, dlam*0.5f);
            continue;
        }

        if (r < kp.r_horizon * 1.03f) break;
        if (r > cp.r_obs * 1.05f)     break;

        const float cos_th = cos(theta);
        if (prev_cos*cos_th <= 0.0f) {
            const float denom = prev_cos - cos_th;
            float w = (abs(denom) > 1e-8f) ? (prev_cos / denom) : 0.5f;
            w = clamp(w, 0.0f, 1.0f);
            const float r_hit = prev_r + w*(r - prev_r);
            if (!(r_hit >= kp.r_isco && r_hit <= kp.r_disk_out)) {
                prev_cos = cos_th;
                prev_r = r;
                continue;
            }

            // Disk hit
            const float Omega = keplerian_omega(r_hit, M, a, Q, L);
            const float b_ip  = pphi / (-pt);
            float gll2[4][4];
            gLL_BL(r_hit, M_PI_2_F, M, a, Q, L, gll2);
            const float d2 = -(gll2[0][0]+2.0f*gll2[0][3]*Omega+gll2[3][3]*Omega*Omega);
            const float dv = 1.0f - Omega*b_ip;
            float red = (d2 > 0.0f && abs(dv) > 1e-8f) ? sqrt(d2)/dv : 1.0f;
            red = clamp(red, 0.0f, 20.0f);

            const float T0 = 2e6f * pow(r_hit/(6.0f*M), -0.75f);
            const float T  = T0 * clamp(red, 0.1f, 10.0f);
            const float I  = clamp(pow(red, 4.0f)*pow(6.0f*M/r_hit, 3.0f), 0.0f, 2.5f);
            float4 c = blackbody_rgb(T) * I;
            c = clamp(c, 0.0f, 1.0f);

            // Pack as ABGR (Metal standard)
            colour = (0xFFu << 24)
                   | (uint32_t(c.b*255.0f) << 16)
                   | (uint32_t(c.g*255.0f) << 8)
                   |  uint32_t(c.r*255.0f);
            break;
        }
        prev_cos = cos_th;
        prev_r = r;
    }

    output[py * cp.width + px] = colour;
}
