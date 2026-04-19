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
    float phi_obs;     // radians
    float fov_h;       // radians
    int   width;
    int   height;
    int   chart;       // 0 = BL, 1 = KS
};

struct RenderParams {
    uint width;
    uint height;
    uint y_offset;
    uint tile_h;
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
    const float s    = (a < 0.0f) ? -1.0f : 1.0f;
    const float Meff = M - Q*Q/(2.0f*r) + L*a*r*r/3.0f;
    const float sq   = sqrt(max(Meff, 0.0f));
    const float den  = r*sqrt(r) + s*a*sq;
    return (abs(den) > 1e-12f) ? (s*sq/den) : 0.0f;
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
                          thread float& dphi_out,
                          thread float& dpr_out, thread float& dpth_out) {
    float gu[4][4];
    gUU(r, theta, M, a, Q, L, gu);

    dr_out  = gu[1][1]*pr;
    dth_out = gu[2][2]*pth;
    dphi_out = gu[3][0]*pt + gu[3][3]*pphi;

    const float er  = 1e-5f*(abs(r)+0.1f);
    const float eth = 1e-6f;

    dpr_out  = -(hamiltonian(r+er, theta, pr, pth, pt, pphi, M, a, Q, L)
               - hamiltonian(r-er, theta, pr, pth, pt, pphi, M, a, Q, L)) / (2.0f*er);

    dpth_out = -(hamiltonian(r, theta+eth, pr, pth, pt, pphi, M, a, Q, L)
               - hamiltonian(r, theta-eth, pr, pth, pt, pphi, M, a, Q, L)) / (2.0f*eth);
}

// RK4 step
static void rk4(thread float& r, thread float& theta, thread float& phi,
                thread float& pr, thread float& pth,
                float pt, float pphi,
                float M, float a, float Q, float L,
                float dlam) {
    float dr1,dth1,dph1,dpr1,dpth1;
    float dr2,dth2,dph2,dpr2,dpth2;
    float dr3,dth3,dph3,dpr3,dpth3;
    float dr4,dth4,dph4,dpr4,dpth4;

    geodesic_rhs(r,               theta,               pr,               pth,               pt,pphi,M,a,Q,L, dr1,dth1,dph1,dpr1,dpth1);
    geodesic_rhs(r+.5f*dlam*dr1, theta+.5f*dlam*dth1, pr+.5f*dlam*dpr1, pth+.5f*dlam*dpth1, pt,pphi,M,a,Q,L, dr2,dth2,dph2,dpr2,dpth2);
    geodesic_rhs(r+.5f*dlam*dr2, theta+.5f*dlam*dth2, pr+.5f*dlam*dpr2, pth+.5f*dlam*dpth2, pt,pphi,M,a,Q,L, dr3,dth3,dph3,dpr3,dpth3);
    geodesic_rhs(r+    dlam*dr3, theta+    dlam*dth3, pr+    dlam*dpr3, pth+    dlam*dpth3, pt,pphi,M,a,Q,L, dr4,dth4,dph4,dpr4,dpth4);

    r     += dlam/6.0f*(dr1   +2.0f*dr2   +2.0f*dr3   +dr4);
    theta += dlam/6.0f*(dth1  +2.0f*dth2  +2.0f*dth3  +dth4);
    phi   += dlam/6.0f*(dph1  +2.0f*dph2  +2.0f*dph3  +dph4);
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

// ── Kerr-Schild (Lambda=0) helpers ────────────────────────────
static float r_KS(float X, float Y, float Z, float a) {
    const float R2 = X*X + Y*Y + Z*Z;
    const float a2 = a*a;
    const float b  = R2 - a2;
    return sqrt(0.5f * (b + sqrt(b*b + 4.0f*a2*Z*Z)));
}

static void BL_to_KS_spatial(float r, float theta, float phi, float a,
                             thread float& X, thread float& Y, thread float& Z) {
    const float st = sin(theta), ct = cos(theta);
    const float sf = sin(phi),   cf = cos(phi);
    X = st * (r*cf + a*sf);
    Y = st * (r*sf - a*cf);
    Z = r * ct;
}

static void KS_to_BL_spatial(float X, float Y, float Z, float a,
                             thread float& r, thread float& theta, thread float& phi) {
    r = r_KS(X, Y, Z, a);
    const float Zr = (abs(r) > 1e-10f) ? (Z / r) : 1.0f;
    theta = acos(clamp(Zr, -1.0f, 1.0f));
    const float r2 = max(r*r, 1e-12f);
    const float st = sqrt(max(1.0f - (Z*Z)/r2, 0.0f));
    const float r2a2 = r*r + a*a;
    if (st > 1e-10f && r2a2 > 1e-10f) {
        const float cf = (X*r + Y*a) / (st * r2a2);
        const float sf = (Y*r - X*a) / (st * r2a2);
        phi = atan2(sf, cf);
    } else {
        phi = 0.0f;
    }
}

static void jacobian_BL_to_KS(float r, float theta, float phi, float a,
                              thread float J[3][3]) {
    const float st = sin(theta), ct = cos(theta);
    const float sf = sin(phi),   cf = cos(phi);
    const float rcfa = r*cf + a*sf;
    const float rsfa = r*sf - a*cf;

    // Columns: (dr, dtheta, dphi), rows: (X,Y,Z)
    J[0][0] = st * cf;
    J[1][0] = st * sf;
    J[2][0] = ct;

    J[0][1] = ct * rcfa;
    J[1][1] = ct * rsfa;
    J[2][1] = -r * st;

    J[0][2] = -st * rsfa;
    J[1][2] =  st * rcfa;
    J[2][2] = 0.0f;
}

static bool solve3x3(thread float A[3][3], thread float b[3], thread float x[3]) {
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        float best = abs(A[piv][col]);
        for (int r = col + 1; r < 3; ++r) {
            const float v = abs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-12f) return false;

        if (piv != col) {
            for (int k = col; k < 3; ++k) {
                const float tmp = A[col][k];
                A[col][k] = A[piv][k];
                A[piv][k] = tmp;
            }
            const float tb = b[col];
            b[col] = b[piv];
            b[piv] = tb;
        }

        const float inv = 1.0f / A[col][col];
        for (int k = col; k < 3; ++k) A[col][k] *= inv;
        b[col] *= inv;

        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const float f = A[r][col];
            for (int k = col; k < 3; ++k) A[r][k] -= f * A[col][k];
            b[r] -= f * b[col];
        }
    }
    x[0] = b[0]; x[1] = b[1]; x[2] = b[2];
    return true;
}

static bool BL_covector_to_KS(float r, float theta, float phi, float a,
                              float pr, float ptheta, float pphi,
                              thread float& pX, thread float& pY, thread float& pZ) {
    float J[3][3];
    jacobian_BL_to_KS(r, theta, phi, a, J);

    float A[3][3];
    // A = J^T, solve A * p_xyz = p_bl
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            A[j][i] = J[i][j];

    float b[3] = {pr, ptheta, pphi};
    float x[3];
    if (!solve3x3(A, b, x)) return false;
    pX = x[0]; pY = x[1]; pZ = x[2];
    return true;
}

static void KS_covector_to_BL(float r, float theta, float phi, float a,
                              float pX, float pY, float pZ,
                              thread float& pr, thread float& ptheta, thread float& pphi) {
    float J[3][3];
    jacobian_BL_to_KS(r, theta, phi, a, J);
    const float pxyz[3] = {pX, pY, pZ};
    pr = ptheta = pphi = 0.0f;
    for (int j = 0; j < 3; ++j) {
        float s = 0.0f;
        for (int i = 0; i < 3; ++i) s += J[i][j] * pxyz[i];
        if (j == 0) pr = s;
        if (j == 1) ptheta = s;
        if (j == 2) pphi = s;
    }
}

static void gUU_KS(float X, float Y, float Z, float M, float a, float Q,
                   thread float guu[4][4]) {
    const float r = r_KS(X, Y, Z, a);
    const float rr = max(r*r, 1e-12f);
    const float rho2 = rr + a*a*Z*Z/rr;
    const float H = (2.0f*M*r - Q*Q) / rho2;
    const float r2a2 = rr + a*a;

    // Ingoing KS null covector
    const float l0 = 1.0f;
    const float l1 = (r*X + a*Y) / r2a2;
    const float l2 = (r*Y - a*X) / r2a2;
    const float l3 = Z / max(r, 1e-6f);

    // l^a = eta^{ab} l_b
    const float lU[4] = {-l0, l1, l2, l3};
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) guu[i][j]=0.0f;
    guu[0][0]=-1.0f; guu[1][1]=1.0f; guu[2][2]=1.0f; guu[3][3]=1.0f;
    for (int mu=0; mu<4; ++mu)
        for (int nu=0; nu<4; ++nu)
            guu[mu][nu] -= H * lU[mu] * lU[nu];
}

static float hamiltonian_KS(float X, float Y, float Z,
                            float pT, float pX, float pY, float pZ,
                            float M, float a, float Q) {
    float guu[4][4];
    gUU_KS(X, Y, Z, M, a, Q, guu);
    return 0.5f * (
        guu[0][0]*pT*pT + 2.0f*guu[0][1]*pT*pX + 2.0f*guu[0][2]*pT*pY + 2.0f*guu[0][3]*pT*pZ +
        guu[1][1]*pX*pX + guu[2][2]*pY*pY + guu[3][3]*pZ*pZ +
        2.0f*guu[1][2]*pX*pY + 2.0f*guu[1][3]*pX*pZ + 2.0f*guu[2][3]*pY*pZ
    );
}

static void geodesic_rhs_KS(float X, float Y, float Z,
                            float pT, float pX, float pY, float pZ,
                            float M, float a, float Q,
                            thread float& dX, thread float& dY, thread float& dZ,
                            thread float& dpX, thread float& dpY, thread float& dpZ) {
    float guu[4][4];
    gUU_KS(X, Y, Z, M, a, Q, guu);
    dX = guu[1][0]*pT + guu[1][1]*pX + guu[1][2]*pY + guu[1][3]*pZ;
    dY = guu[2][0]*pT + guu[2][1]*pX + guu[2][2]*pY + guu[2][3]*pZ;
    dZ = guu[3][0]*pT + guu[3][1]*pX + guu[3][2]*pY + guu[3][3]*pZ;

    const float eX = 1e-5f*(abs(X)+0.1f);
    const float eY = 1e-5f*(abs(Y)+0.1f);
    const float eZ = 1e-5f*(abs(Z)+0.1f);
    dpX = -(hamiltonian_KS(X+eX, Y, Z, pT, pX, pY, pZ, M, a, Q)
          - hamiltonian_KS(X-eX, Y, Z, pT, pX, pY, pZ, M, a, Q)) / (2.0f*eX);
    dpY = -(hamiltonian_KS(X, Y+eY, Z, pT, pX, pY, pZ, M, a, Q)
          - hamiltonian_KS(X, Y-eY, Z, pT, pX, pY, pZ, M, a, Q)) / (2.0f*eY);
    dpZ = -(hamiltonian_KS(X, Y, Z+eZ, pT, pX, pY, pZ, M, a, Q)
          - hamiltonian_KS(X, Y, Z-eZ, pT, pX, pY, pZ, M, a, Q)) / (2.0f*eZ);
}

static void rk4_KS(thread float& X, thread float& Y, thread float& Z,
                   thread float& pX, thread float& pY, thread float& pZ,
                   float pT, float M, float a, float Q, float dlam) {
    float dX1,dY1,dZ1,dpX1,dpY1,dpZ1;
    float dX2,dY2,dZ2,dpX2,dpY2,dpZ2;
    float dX3,dY3,dZ3,dpX3,dpY3,dpZ3;
    float dX4,dY4,dZ4,dpX4,dpY4,dpZ4;

    geodesic_rhs_KS(X, Y, Z, pT, pX, pY, pZ, M, a, Q, dX1,dY1,dZ1,dpX1,dpY1,dpZ1);
    geodesic_rhs_KS(X+0.5f*dlam*dX1, Y+0.5f*dlam*dY1, Z+0.5f*dlam*dZ1,
                    pT, pX+0.5f*dlam*dpX1, pY+0.5f*dlam*dpY1, pZ+0.5f*dlam*dpZ1,
                    M, a, Q, dX2,dY2,dZ2,dpX2,dpY2,dpZ2);
    geodesic_rhs_KS(X+0.5f*dlam*dX2, Y+0.5f*dlam*dY2, Z+0.5f*dlam*dZ2,
                    pT, pX+0.5f*dlam*dpX2, pY+0.5f*dlam*dpY2, pZ+0.5f*dlam*dpZ2,
                    M, a, Q, dX3,dY3,dZ3,dpX3,dpY3,dpZ3);
    geodesic_rhs_KS(X+dlam*dX3, Y+dlam*dY3, Z+dlam*dZ3,
                    pT, pX+dlam*dpX3, pY+dlam*dpY3, pZ+dlam*dpZ3,
                    M, a, Q, dX4,dY4,dZ4,dpX4,dpY4,dpZ4);

    X  += dlam/6.0f*(dX1  +2.0f*dX2  +2.0f*dX3  +dX4);
    Y  += dlam/6.0f*(dY1  +2.0f*dY2  +2.0f*dY3  +dY4);
    Z  += dlam/6.0f*(dZ1  +2.0f*dZ2  +2.0f*dZ3  +dZ4);
    pX += dlam/6.0f*(dpX1 +2.0f*dpX2 +2.0f*dpX3 +dpX4);
    pY += dlam/6.0f*(dpY1 +2.0f*dpY2 +2.0f*dpY3 +dpY4);
    pZ += dlam/6.0f*(dpZ1 +2.0f*dpZ2 +2.0f*dpZ3 +dpZ4);
}

// ── Emissivity / tonemap helpers ──────────────────────────────
// Page-Thorne emissivity f(r) = (1 − √(r_isco/r)) / r³, normalised to 1
// at its peak r = 3·r_isco  (matches CPU disk_colour exactly)
static float page_thorne_norm(float r, float r_isco) {
    if (r <= r_isco) return 0.0f;
    float pt      = (1.0f - sqrt(r_isco / r)) / (r*r*r);
    float r_peak  = 3.0f * r_isco;
    float pt_peak = (1.0f - sqrt(r_isco / r_peak)) / (r_peak*r_peak*r_peak);
    return (pt_peak > 0.0f) ? (pt / pt_peak) : 0.0f;
}

// Reinhard tonemap with gamma  (exposure = 1.0 default, matches CPU ColorParams)
static float tonemap_ch(float x, float exposure, float gamma) {
    x = x * exposure;
    x = x / (1.0f + x);
    return pow(clamp(x, 0.0f, 1.0f), 1.0f / gamma);
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

static float4 sample_background(texture2d<float, access::sample> bg_tex,
                                sampler bg_samp,
                                float theta, float phi) {
    constexpr float PI_F     = 3.14159265358979323846f;
    constexpr float TWO_PI_F = 6.28318530717958647692f;
    float pw = fmod(phi, TWO_PI_F);
    if (pw < 0.0f) pw += TWO_PI_F;
    const float u = pw / TWO_PI_F;
    const float v = clamp(theta / PI_F, 0.0f, 1.0f);
    return bg_tex.sample(bg_samp, float2(u, v));
}

// ── Main compute kernel ───────────────────────────────────────
kernel void trace_pixel(
    device uint32_t*   output    [[buffer(0)]],
    constant KNdSParams& kp      [[buffer(1)]],
    constant CameraParams& cp    [[buffer(2)]],
    constant RenderParams& rp    [[buffer(3)]],
    texture2d<float, access::sample> bg_tex [[texture(0)]],
    sampler             bg_samp   [[sampler(0)]],
    uint2              gid       [[thread_position_in_grid]])
{
    const int px = (int)gid.x;
    const int py_local = (int)gid.y;
    const int width = (int)rp.width;
    const int height = (int)rp.height;
    if (px >= width || py_local >= (int)rp.tile_h) return;
    const int py = py_local + (int)rp.y_offset;

    const float M = kp.M, a = kp.a, Q = kp.Q, L = kp.Lambda;

    // ── Pixel → (α, β) ───────────────────────────────────────
    const int span = max(width - 1, 1);
    const float alpha = cp.fov_h*(float(px) - 0.5f*(width-1))  / float(span);
    const float beta  = cp.fov_h*(0.5f*(height-1) - float(py)) / float(span);

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
    const float pUphi = -sa*cb / sqrt_gphph;

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
    float r = r0, theta = th0, phi = cp.phi_obs;
    float dlam = 1.0f;
    float prev_r = r0;
    float prev_theta = th0;
    float prev_phi = phi;
    float prev_cos = cos(th0);

    uint32_t colour = 0xFF000000u;  // black (ABGR)

    // ── KS chart path (Lambda=0 only) ─────────────────────────
    const bool want_ks = (cp.chart == 1 && abs(L) <= 1e-8f);
    if (want_ks) {
        float X, Y, Z;
        BL_to_KS_spatial(r0, th0, cp.phi_obs, a, X, Y, Z);
        float pX, pY, pZ;
        bool ks_ok = BL_covector_to_KS(r0, th0, cp.phi_obs, a, pr, pth, pphi, pX, pY, pZ);

        float pT = pt;
        if (ks_ok) {
            float gu_ks[4][4];
            gUU_KS(X, Y, Z, M, a, Q, gu_ks);
            const float Aks = gu_ks[0][0];
            const float Bks = 2.0f * (gu_ks[0][1]*pX + gu_ks[0][2]*pY + gu_ks[0][3]*pZ);
            const float Cks = gu_ks[1][1]*pX*pX + gu_ks[2][2]*pY*pY + gu_ks[3][3]*pZ*pZ
                            + 2.0f*gu_ks[1][2]*pX*pY + 2.0f*gu_ks[1][3]*pX*pZ + 2.0f*gu_ks[2][3]*pY*pZ;
            const float dks = Bks*Bks - 4.0f*Aks*Cks;
            if (dks >= 0.0f && abs(Aks) > 1e-12f) {
                const float sq = sqrt(dks);
                const float pT1 = (-Bks - sq) / (2.0f*Aks);
                const float pT2 = (-Bks + sq) / (2.0f*Aks);
                pT = (pT1 < 0.0f) ? pT1 : pT2;
                if (pT > 0.0f) pT = min(pT1, pT2);
                ks_ok = isfinite(pT);
            } else {
                ks_ok = false;
            }
        }

        if (ks_ok) {
            float dlam_ks = 1.0f;
            float prevX = X, prevY = Y, prevZ = Z;
            float prevPX = pX, prevPY = pY, prevPZ = pZ;
            float prev_r_ks = r0;

            for (int iter = 0; iter < 60000; ++iter) {
                float X_h = X, Y_h = Y, Z_h = Z, pX_h = pX, pY_h = pY, pZ_h = pZ;
                rk4_KS(X_h, Y_h, Z_h, pX_h, pY_h, pZ_h, pT, M, a, Q, dlam_ks);

                float X_f = X, Y_f = Y, Z_f = Z, pX_f = pX, pY_f = pY, pZ_f = pZ;
                rk4_KS(X_f, Y_f, Z_f, pX_f, pY_f, pZ_f, pT, M, a, Q, dlam_ks*0.5f);
                rk4_KS(X_f, Y_f, Z_f, pX_f, pY_f, pZ_f, pT, M, a, Q, dlam_ks*0.5f);

                const float err = sqrt(
                    (X_h-X_f)*(X_h-X_f) + (Y_h-Y_f)*(Y_h-Y_f) + (Z_h-Z_f)*(Z_h-Z_f) +
                    (pX_h-pX_f)*(pX_h-pX_f) + (pY_h-pY_f)*(pY_h-pY_f) + (pZ_h-pZ_f)*(pZ_h-pZ_f)
                ) / 15.0f;
                const float tol = 2e-5f;
                if (err < tol || dlam_ks < 1e-6f) {
                    X = X_f; Y = Y_f; Z = Z_f; pX = pX_f; pY = pY_f; pZ = pZ_f;
                    const float sc = (err > 1e-10f) ? 0.9f*pow(tol/err, 0.2f) : 2.0f;
                    dlam_ks = clamp(dlam_ks*sc, 1e-6f, 50.0f);
                } else {
                    dlam_ks = clamp(dlam_ks*0.9f*pow(tol/err, 0.25f), 1e-6f, dlam_ks*0.5f);
                    continue;
                }

                if (!(isfinite(X) && isfinite(Y) && isfinite(Z) &&
                      isfinite(pX) && isfinite(pY) && isfinite(pZ))) {
                    break;
                }

                const float r_now = r_KS(X, Y, Z, a);
                if (r_now < kp.r_horizon * 1.03f) break;

                if (r_now > cp.r_obs * 1.05f) {
                    const float r_escape = cp.r_obs * 1.05f;
                    const float denom = r_now - prev_r_ks;
                    float w = (abs(denom) > 1e-8f) ? ((r_escape - prev_r_ks) / denom) : 1.0f;
                    w = clamp(w, 0.0f, 1.0f);
                    const float X_esc = prevX + w*(X - prevX);
                    const float Y_esc = prevY + w*(Y - prevY);
                    const float Z_esc = prevZ + w*(Z - prevZ);
                    float r_esc, th_esc, ph_esc;
                    KS_to_BL_spatial(X_esc, Y_esc, Z_esc, a, r_esc, th_esc, ph_esc);
                    float4 bgc = clamp(sample_background(bg_tex, bg_samp, th_esc, ph_esc), 0.0f, 1.0f);
                    colour = (0xFFu << 24)
                           | (uint32_t(bgc.b*255.0f) << 16)
                           | (uint32_t(bgc.g*255.0f) << 8)
                           |  uint32_t(bgc.r*255.0f);
                    break;
                }

                if (prevZ * Z <= 0.0f) {
                    const float denom = prevZ - Z;
                    float w = (abs(denom) > 1e-8f) ? (prevZ / denom) : 0.5f;
                    w = clamp(w, 0.0f, 1.0f);
                    const float Xh = prevX + w*(X - prevX);
                    const float Yh = prevY + w*(Y - prevY);
                    const float Zh = prevZ + w*(Z - prevZ);
                    const float pXh = prevPX + w*(pX - prevPX);
                    const float pYh = prevPY + w*(pY - prevPY);
                    const float pZh = prevPZ + w*(pZ - prevPZ);
                    float r_hit, th_hit, ph_hit;
                    KS_to_BL_spatial(Xh, Yh, Zh, a, r_hit, th_hit, ph_hit);
                    if (r_hit >= kp.r_isco && r_hit <= kp.r_disk_out) {
                        float pr_hit, pth_hit, pphi_hit;
                        KS_covector_to_BL(r_hit, th_hit, ph_hit, a, pXh, pYh, pZh,
                                          pr_hit, pth_hit, pphi_hit);
                        const float Omega = keplerian_omega(r_hit, M, a, Q, L);
                        const float b_ip  = pphi_hit / (-pT);
                        float gll2[4][4];
                        gLL_BL(r_hit, M_PI_2_F, M, a, Q, L, gll2);
                        const float d2 = -(gll2[0][0]+2.0f*gll2[0][3]*Omega+gll2[3][3]*Omega*Omega);
                        const float dv = 1.0f - Omega*b_ip;
                        float red = (d2 > 0.0f && abs(dv) > 1e-8f) ? sqrt(d2)/dv : 1.0f;
                        red = clamp(red, 0.0f, 20.0f);

                        const float T = 6500.0f * sqrt(6.0f*M/r_hit) * clamp(red, 0.2f, 5.0f);
                        float I = page_thorne_norm(r_hit, kp.r_isco);
                        I *= pow(clamp(red, 0.1f, 10.0f), 4.0f);
                        float4 bb = blackbody_rgb(T);
                        float4 c = float4(tonemap_ch(bb.r * I, 1.0f, 2.2f),
                                          tonemap_ch(bb.g * I, 1.0f, 2.2f),
                                          tonemap_ch(bb.b * I, 1.0f, 2.2f),
                                          1.0f);
                        colour = (0xFFu << 24)
                               | (uint32_t(c.b*255.0f) << 16)
                               | (uint32_t(c.g*255.0f) << 8)
                               |  uint32_t(c.r*255.0f);
                        break;
                    }
                }

                prevX = X; prevY = Y; prevZ = Z;
                prevPX = pX; prevPY = pY; prevPZ = pZ;
                prev_r_ks = r_now;
            }

            output[py * width + px] = colour;
            return;
        }
    }

    for (int iter = 0; iter < 60000; ++iter) {
        // Adaptive step (simple: fixed tolerance)
        float r_h = r, th_h = theta, ph_h = phi, pr_h = pr, pth_h = pth;
        rk4(r_h, th_h, ph_h, pr_h, pth_h, pt, pphi, M, a, Q, L, dlam);

        float r_f = r, th_f = theta, ph_f = phi, pr_f = pr, pth_f = pth;
        rk4(r_f, th_f, ph_f, pr_f, pth_f, pt, pphi, M, a, Q, L, dlam*0.5f);
        rk4(r_f, th_f, ph_f, pr_f, pth_f, pt, pphi, M, a, Q, L, dlam*0.5f);

        const float err = length(float4(r_h-r_f, th_h-th_f,
                                        pr_h-pr_f, pth_h-pth_f)) / 15.0f;
        const float tol = 2e-5f;
        if (err < tol || dlam < 1e-6f) {
            r = r_f; theta = th_f; phi = ph_f; pr = pr_f; pth = pth_f;
            float sc = (err > 1e-10f) ? 0.9f*pow(tol/err, 0.2f) : 2.0f;
            dlam = clamp(dlam*sc, 1e-6f, 50.0f);
        } else {
            dlam = clamp(dlam*0.9f*pow(tol/err, 0.25f), 1e-6f, dlam*0.5f);
            continue;
        }

        if (!(isfinite(r) && isfinite(theta) && isfinite(phi) &&
              isfinite(pr) && isfinite(pth))) {
            break;
        }

        if (r < kp.r_horizon * 1.03f) break;
        if (r > cp.r_obs * 1.05f) {
            const float r_escape = cp.r_obs * 1.05f;
            const float denom = r - prev_r;
            float w = (abs(denom) > 1e-8f) ? ((r_escape - prev_r) / denom) : 1.0f;
            w = clamp(w, 0.0f, 1.0f);
            const float th_esc = prev_theta + w*(theta - prev_theta);
            const float ph_esc = prev_phi   + w*(phi   - prev_phi);
            float4 bgc = clamp(sample_background(bg_tex, bg_samp, th_esc, ph_esc), 0.0f, 1.0f);
            colour = (0xFFu << 24)
                   | (uint32_t(bgc.b*255.0f) << 16)
                   | (uint32_t(bgc.g*255.0f) << 8)
                   |  uint32_t(bgc.r*255.0f);
            break;
        }

        const float cos_th = cos(theta);
        if (prev_cos*cos_th <= 0.0f) {
            const float denom = prev_cos - cos_th;
            float w = (abs(denom) > 1e-8f) ? (prev_cos / denom) : 0.5f;
            w = clamp(w, 0.0f, 1.0f);
            const float r_hit = prev_r + w*(r - prev_r);
            if (!(r_hit >= kp.r_isco && r_hit <= kp.r_disk_out)) {
                prev_cos = cos_th;
                prev_r = r;
                prev_theta = theta;
                prev_phi = phi;
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

            // Match CPU disk_colour():
            //   T = 6500 * sqrt(6M/r) * clamp(red,0.2,5)
            //   I = page_thorne_norm(r) * red^4
            //   output = tonemap(blackbody(T) * I, exposure=1, gamma=2.2)
            const float T = 6500.0f * sqrt(6.0f*M/r_hit) * clamp(red, 0.2f, 5.0f);
            float I = page_thorne_norm(r_hit, kp.r_isco);
            I *= pow(clamp(red, 0.1f, 10.0f), 4.0f);
            float4 bb = blackbody_rgb(T);
            float4 c = float4(tonemap_ch(bb.r * I, 1.0f, 2.2f),
                              tonemap_ch(bb.g * I, 1.0f, 2.2f),
                              tonemap_ch(bb.b * I, 1.0f, 2.2f),
                              1.0f);

            // Pack as ABGR (Metal standard)
            colour = (0xFFu << 24)
                   | (uint32_t(c.b*255.0f) << 16)
                   | (uint32_t(c.g*255.0f) << 8)
                   |  uint32_t(c.r*255.0f);
            break;
        }
        prev_cos = cos_th;
        prev_r = r;
        prev_theta = theta;
        prev_phi = phi;
    }

    output[py * width + px] = colour;
}
