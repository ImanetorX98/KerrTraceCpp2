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

// Metal uses float arithmetic: keep a conservative adaptive-step floor to avoid
// coarse near-horizon integration while still preventing denormal/no-progress loops.
constant float ADAPT_H_MIN = 1e-6f;
constant float ADAPT_H_MAX = 50.0f;
constant float ADAPT_ERR_TINY_DOPRI = 1e-12f;
constant float ADAPT_ERR_TINY_RK4 = 1e-10f;

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
    int   solver_mode; // 0 = standard, 1 = semi-analytic, 2 = elliptic-closed
    int   integrator_mode; // 0 = RK4-doubling, 1 = DOPRI5
    int   use_bundles; // 0 = single ray, 1 = ray-bundle (finite-difference proxy)
    int   metal_kernel_mode; // 0 = auto, 1 = unified(legacy), 2 = single, 3 = bundle
    int   intersection_mode; // 0 = linear, 1 = hermite
    int   elliptic_fallback_black; // 0 = normal fallback, 1 = force black on fallback rays
    int   anti_fireflies; // 0 = off, 1 = robust anti-fireflies filter (ray-bundle path)
    int   max_steps; // hard cap on adaptive integration iterations per ray
    float step_init; // initial affine step size
    float integrator_tol; // adaptive integrator tolerance
    float pixel_offset_x; // subpixel X offset in pixel units
    float pixel_offset_y; // subpixel Y offset in pixel units
};

struct RenderParams {
    uint width;
    uint height;
    uint x_offset;
    uint tile_w;
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
    const float s    = (a < 0.0f) ? 1.0f : -1.0f;
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

struct BLState {
    float r, theta, phi, pr, pth;
};

struct BLDeriv {
    float dr, dth, dphi, dpr, dpth;
};

static BLDeriv eval_bl_rhs_state(thread const BLState& s,
                                 float pt, float pphi,
                                 float M, float a, float Q, float L) {
    BLDeriv k{};
    geodesic_rhs(s.r, s.theta, s.pr, s.pth, pt, pphi, M, a, Q, L,
                 k.dr, k.dth, k.dphi, k.dpr, k.dpth);
    return k;
}

static bool dopri5_adaptive_bl(thread float& r, thread float& theta, thread float& phi,
                               thread float& pr, thread float& pth,
                               thread float& h,
                               float pt, float pphi,
                               float M, float a, float Q, float L,
                               float tol) {
    const BLState s0{r, theta, phi, pr, pth};
    const BLDeriv k1 = eval_bl_rhs_state(s0, pt, pphi, M, a, Q, L);

    BLState s2 = s0;
    s2.r     += h * (1.0f/5.0f) * k1.dr;
    s2.theta += h * (1.0f/5.0f) * k1.dth;
    s2.phi   += h * (1.0f/5.0f) * k1.dphi;
    s2.pr    += h * (1.0f/5.0f) * k1.dpr;
    s2.pth   += h * (1.0f/5.0f) * k1.dpth;
    const BLDeriv k2 = eval_bl_rhs_state(s2, pt, pphi, M, a, Q, L);

    BLState s3 = s0;
    s3.r     += h * (3.0f/40.0f*k1.dr   + 9.0f/40.0f*k2.dr);
    s3.theta += h * (3.0f/40.0f*k1.dth  + 9.0f/40.0f*k2.dth);
    s3.phi   += h * (3.0f/40.0f*k1.dphi + 9.0f/40.0f*k2.dphi);
    s3.pr    += h * (3.0f/40.0f*k1.dpr  + 9.0f/40.0f*k2.dpr);
    s3.pth   += h * (3.0f/40.0f*k1.dpth + 9.0f/40.0f*k2.dpth);
    const BLDeriv k3 = eval_bl_rhs_state(s3, pt, pphi, M, a, Q, L);

    BLState s4 = s0;
    s4.r     += h * (44.0f/45.0f*k1.dr   - 56.0f/15.0f*k2.dr   + 32.0f/9.0f*k3.dr);
    s4.theta += h * (44.0f/45.0f*k1.dth  - 56.0f/15.0f*k2.dth  + 32.0f/9.0f*k3.dth);
    s4.phi   += h * (44.0f/45.0f*k1.dphi - 56.0f/15.0f*k2.dphi + 32.0f/9.0f*k3.dphi);
    s4.pr    += h * (44.0f/45.0f*k1.dpr  - 56.0f/15.0f*k2.dpr  + 32.0f/9.0f*k3.dpr);
    s4.pth   += h * (44.0f/45.0f*k1.dpth - 56.0f/15.0f*k2.dpth + 32.0f/9.0f*k3.dpth);
    const BLDeriv k4 = eval_bl_rhs_state(s4, pt, pphi, M, a, Q, L);

    BLState s5 = s0;
    s5.r     += h * (19372.0f/6561.0f*k1.dr   - 25360.0f/2187.0f*k2.dr   + 64448.0f/6561.0f*k3.dr   - 212.0f/729.0f*k4.dr);
    s5.theta += h * (19372.0f/6561.0f*k1.dth  - 25360.0f/2187.0f*k2.dth  + 64448.0f/6561.0f*k3.dth  - 212.0f/729.0f*k4.dth);
    s5.phi   += h * (19372.0f/6561.0f*k1.dphi - 25360.0f/2187.0f*k2.dphi + 64448.0f/6561.0f*k3.dphi - 212.0f/729.0f*k4.dphi);
    s5.pr    += h * (19372.0f/6561.0f*k1.dpr  - 25360.0f/2187.0f*k2.dpr  + 64448.0f/6561.0f*k3.dpr  - 212.0f/729.0f*k4.dpr);
    s5.pth   += h * (19372.0f/6561.0f*k1.dpth - 25360.0f/2187.0f*k2.dpth + 64448.0f/6561.0f*k3.dpth - 212.0f/729.0f*k4.dpth);
    const BLDeriv k5 = eval_bl_rhs_state(s5, pt, pphi, M, a, Q, L);

    BLState s6 = s0;
    s6.r     += h * (9017.0f/3168.0f*k1.dr   - 355.0f/33.0f*k2.dr   + 46732.0f/5247.0f*k3.dr   + 49.0f/176.0f*k4.dr   - 5103.0f/18656.0f*k5.dr);
    s6.theta += h * (9017.0f/3168.0f*k1.dth  - 355.0f/33.0f*k2.dth  + 46732.0f/5247.0f*k3.dth  + 49.0f/176.0f*k4.dth  - 5103.0f/18656.0f*k5.dth);
    s6.phi   += h * (9017.0f/3168.0f*k1.dphi - 355.0f/33.0f*k2.dphi + 46732.0f/5247.0f*k3.dphi + 49.0f/176.0f*k4.dphi - 5103.0f/18656.0f*k5.dphi);
    s6.pr    += h * (9017.0f/3168.0f*k1.dpr  - 355.0f/33.0f*k2.dpr  + 46732.0f/5247.0f*k3.dpr  + 49.0f/176.0f*k4.dpr  - 5103.0f/18656.0f*k5.dpr);
    s6.pth   += h * (9017.0f/3168.0f*k1.dpth - 355.0f/33.0f*k2.dpth + 46732.0f/5247.0f*k3.dpth + 49.0f/176.0f*k4.dpth - 5103.0f/18656.0f*k5.dpth);
    const BLDeriv k6 = eval_bl_rhs_state(s6, pt, pphi, M, a, Q, L);

    BLState y5 = s0;
    y5.r     += h * (35.0f/384.0f*k1.dr   + 500.0f/1113.0f*k3.dr   + 125.0f/192.0f*k4.dr   - 2187.0f/6784.0f*k5.dr   + 11.0f/84.0f*k6.dr);
    y5.theta += h * (35.0f/384.0f*k1.dth  + 500.0f/1113.0f*k3.dth  + 125.0f/192.0f*k4.dth  - 2187.0f/6784.0f*k5.dth  + 11.0f/84.0f*k6.dth);
    y5.phi   += h * (35.0f/384.0f*k1.dphi + 500.0f/1113.0f*k3.dphi + 125.0f/192.0f*k4.dphi - 2187.0f/6784.0f*k5.dphi + 11.0f/84.0f*k6.dphi);
    y5.pr    += h * (35.0f/384.0f*k1.dpr  + 500.0f/1113.0f*k3.dpr  + 125.0f/192.0f*k4.dpr  - 2187.0f/6784.0f*k5.dpr  + 11.0f/84.0f*k6.dpr);
    y5.pth   += h * (35.0f/384.0f*k1.dpth + 500.0f/1113.0f*k3.dpth + 125.0f/192.0f*k4.dpth - 2187.0f/6784.0f*k5.dpth + 11.0f/84.0f*k6.dpth);

    const BLDeriv k7 = eval_bl_rhs_state(y5, pt, pphi, M, a, Q, L);

    BLState y4 = s0;
    y4.r     += h * (5179.0f/57600.0f*k1.dr   + 7571.0f/16695.0f*k3.dr   + 393.0f/640.0f*k4.dr   - 92097.0f/339200.0f*k5.dr   + 187.0f/2100.0f*k6.dr   + 1.0f/40.0f*k7.dr);
    y4.theta += h * (5179.0f/57600.0f*k1.dth  + 7571.0f/16695.0f*k3.dth  + 393.0f/640.0f*k4.dth  - 92097.0f/339200.0f*k5.dth  + 187.0f/2100.0f*k6.dth  + 1.0f/40.0f*k7.dth);
    y4.phi   += h * (5179.0f/57600.0f*k1.dphi + 7571.0f/16695.0f*k3.dphi + 393.0f/640.0f*k4.dphi - 92097.0f/339200.0f*k5.dphi + 187.0f/2100.0f*k6.dphi + 1.0f/40.0f*k7.dphi);
    y4.pr    += h * (5179.0f/57600.0f*k1.dpr  + 7571.0f/16695.0f*k3.dpr  + 393.0f/640.0f*k4.dpr  - 92097.0f/339200.0f*k5.dpr  + 187.0f/2100.0f*k6.dpr  + 1.0f/40.0f*k7.dpr);
    y4.pth   += h * (5179.0f/57600.0f*k1.dpth + 7571.0f/16695.0f*k3.dpth + 393.0f/640.0f*k4.dpth - 92097.0f/339200.0f*k5.dpth + 187.0f/2100.0f*k6.dpth + 1.0f/40.0f*k7.dpth);

    const float sr = tol * (1.0f + abs(s0.r));
    const float stheta = tol * (1.0f + abs(s0.theta));
    const float spr = tol * (1.0f + abs(s0.pr));
    const float spth = tol * (1.0f + abs(s0.pth));
    const float er0 = (y5.r - y4.r) / max(sr, 1e-12f);
    const float er1 = (y5.theta - y4.theta) / max(stheta, 1e-12f);
    const float er2 = (y5.pr - y4.pr) / max(spr, 1e-12f);
    const float er3 = (y5.pth - y4.pth) / max(spth, 1e-12f);
    const float err = sqrt((er0*er0 + er1*er1 + er2*er2 + er3*er3) * 0.25f);

    if (!isfinite(err)) {
        h = max(h * 0.5f, ADAPT_H_MIN);
        return false;
    }

    if (err <= 1.0f || h < ADAPT_H_MIN) {
        r = y5.r; theta = y5.theta; phi = y5.phi; pr = y5.pr; pth = y5.pth;
        const float fac = (err > ADAPT_ERR_TINY_DOPRI) ? 0.9f * pow(1.0f/err, 0.2f) : 2.0f;
        h = clamp(h * fac, ADAPT_H_MIN, ADAPT_H_MAX);
        return true;
    }

    h = clamp(h * 0.9f * pow(1.0f/err, 0.25f), ADAPT_H_MIN, h * 0.5f);
    return false;
}

static bool adaptive_step_bl(thread float& r, thread float& theta, thread float& phi,
                             thread float& pr, thread float& pth,
                             thread float& h,
                             float pt, float pphi,
                             float M, float a, float Q, float L,
                             int integrator_mode,
                             float tol) {
    if (integrator_mode == 1) {
        return dopri5_adaptive_bl(r, theta, phi, pr, pth, h, pt, pphi, M, a, Q, L, tol);
    }

    const float step_used = h;
    float r_h = r, th_h = theta, ph_h = phi, pr_h = pr, pth_h = pth;
    rk4(r_h, th_h, ph_h, pr_h, pth_h, pt, pphi, M, a, Q, L, step_used);

    float r_f = r, th_f = theta, ph_f = phi, pr_f = pr, pth_f = pth;
    rk4(r_f, th_f, ph_f, pr_f, pth_f, pt, pphi, M, a, Q, L, step_used*0.5f);
    rk4(r_f, th_f, ph_f, pr_f, pth_f, pt, pphi, M, a, Q, L, step_used*0.5f);

    const float err = length(float4(r_h-r_f, th_h-th_f, pr_h-pr_f, pth_h-pth_f)) / 15.0f;
    if (err < tol || h < ADAPT_H_MIN) {
        r = r_f; theta = th_f; phi = ph_f; pr = pr_f; pth = pth_f;
        const float sc = (err > ADAPT_ERR_TINY_RK4) ? 0.9f*pow(tol/err, 0.2f) : 2.0f;
        h = clamp(h*sc, ADAPT_H_MIN, ADAPT_H_MAX);
        return true;
    }

    h = clamp(h*0.9f*pow(tol/err, 0.25f), ADAPT_H_MIN, h*0.5f);
    return false;
}

// Event localization helpers (cubic Hermite + bisection).
static inline bool sign_change_f(float f0, float f1) {
    return ((f0 <= 0.0f && f1 >= 0.0f) || (f0 >= 0.0f && f1 <= 0.0f));
}

static float hermite_interp_f(float y0, float y1,
                              float dy0, float dy1,
                              float h, float alpha) {
    alpha = clamp(alpha, 0.0f, 1.0f);
    const float a2 = alpha * alpha;
    const float a3 = a2 * alpha;
    const float h00 =  2.0f*a3 - 3.0f*a2 + 1.0f;
    const float h10 =        a3 - 2.0f*a2 + alpha;
    const float h01 = -2.0f*a3 + 3.0f*a2;
    const float h11 =        a3 -        a2;
    return h00*y0 + h10*(h*dy0) + h01*y1 + h11*(h*dy1);
}

static float refine_event_alpha_hermite_f(float y0, float y1,
                                          float dy0, float dy1,
                                          float h, float target,
                                          int iterations) {
    const float f0 = y0 - target;
    const float f1 = y1 - target;
    float alpha = abs(f0) / (abs(f0) + abs(f1) + 1e-12f);
    alpha = clamp(alpha, 0.0f, 1.0f);

    if (!sign_change_f(f0, f1)) return alpha;

    float lo = 0.0f;
    float hi = 1.0f;
    for (int i = 0; i < iterations; ++i) {
        const float mid = 0.5f * (lo + hi);
        const float y_mid = hermite_interp_f(y0, y1, dy0, dy1, h, mid);
        const float f_mid = y_mid - target;
        if (sign_change_f(f0, f_mid)) hi = mid;
        else lo = mid;
    }
    return 0.5f * (lo + hi);
}

static bool first_event_alpha_hermite_f(float y0, float y1,
                                        float dy0, float dy1,
                                        float h, float target,
                                        thread float& alpha_out,
                                        int bins,
                                        int iterations) {
    bins = max(2, bins);
    const float f_eps = 1e-12f;

    float a0 = 0.0f;
    float f0 = y0 - target;
    if (abs(f0) <= f_eps) {
        alpha_out = 0.0f;
        return true;
    }

    for (int i = 1; i <= bins; ++i) {
        const float a1 = float(i) / float(bins);
        const float f1 = (i == bins)
            ? (y1 - target)
            : (hermite_interp_f(y0, y1, dy0, dy1, h, a1) - target);
        if (abs(f1) <= f_eps) {
            alpha_out = a1;
            return true;
        }
        if (sign_change_f(f0, f1)) {
            float lo = a0;
            float hi = a1;
            for (int k = 0; k < iterations; ++k) {
                const float mid = 0.5f * (lo + hi);
                const float fm = hermite_interp_f(y0, y1, dy0, dy1, h, mid) - target;
                if (sign_change_f(f0, fm)) hi = mid;
                else lo = mid;
            }
            alpha_out = 0.5f * (lo + hi);
            return true;
        }
        a0 = a1;
        f0 = f1;
    }
    return false;
}

static bool first_event_alpha_linear_f(float y0, float y1,
                                       float target,
                                       thread float& alpha_out) {
    const float f0 = y0 - target;
    const float f1 = y1 - target;
    if (!sign_change_f(f0, f1)) return false;
    const float denom = f0 - f1;
    float alpha = (abs(denom) > 1e-8f) ? (f0 / denom) : 0.5f;
    alpha_out = clamp(alpha, 0.0f, 1.0f);
    return true;
}

static bool first_event_alpha_mode_f(float y0, float y1,
                                     float dy0, float dy1,
                                     float h, float target,
                                     int mode,
                                     thread float& alpha_out) {
    if (mode == 0) return first_event_alpha_linear_f(y0, y1, target, alpha_out);
    if (first_event_alpha_hermite_f(y0, y1, dy0, dy1, h, target, alpha_out, 8, 7))
        return true;
    // Robust fallback: if Hermite misses a true sign-crossing, recover linearly.
    return first_event_alpha_linear_f(y0, y1, target, alpha_out);
}

static float interp_scalar_mode_f(float y0, float y1,
                                  float dy0, float dy1,
                                  float h, float alpha,
                                  int mode) {
    alpha = clamp(alpha, 0.0f, 1.0f);
    if (mode == 0) return y0 + alpha*(y1 - y0);
    return hermite_interp_f(y0, y1, dy0, dy1, h, alpha);
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
    X = st * (r*cf - a*sf);
    Y = st * (r*sf + a*cf);
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
    const float rcfa = r*cf - a*sf;
    const float rsfa = r*sf + a*cf;

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

struct KSState {
    float X, Y, Z, pX, pY, pZ;
};

struct KSDeriv {
    float dX, dY, dZ, dpX, dpY, dpZ;
};

static KSDeriv eval_ks_rhs_state(thread const KSState& s,
                                 float pT, float M, float a, float Q) {
    KSDeriv k{};
    geodesic_rhs_KS(s.X, s.Y, s.Z, pT, s.pX, s.pY, s.pZ, M, a, Q,
                    k.dX, k.dY, k.dZ, k.dpX, k.dpY, k.dpZ);
    return k;
}

static bool dopri5_adaptive_ks(thread float& X, thread float& Y, thread float& Z,
                               thread float& pX, thread float& pY, thread float& pZ,
                               thread float& h,
                               float pT, float M, float a, float Q,
                               float tol) {
    const KSState s0{X, Y, Z, pX, pY, pZ};
    const KSDeriv k1 = eval_ks_rhs_state(s0, pT, M, a, Q);

    KSState s2 = s0;
    s2.X  += h * (1.0f/5.0f) * k1.dX;  s2.Y  += h * (1.0f/5.0f) * k1.dY;  s2.Z  += h * (1.0f/5.0f) * k1.dZ;
    s2.pX += h * (1.0f/5.0f) * k1.dpX; s2.pY += h * (1.0f/5.0f) * k1.dpY; s2.pZ += h * (1.0f/5.0f) * k1.dpZ;
    const KSDeriv k2 = eval_ks_rhs_state(s2, pT, M, a, Q);

    KSState s3 = s0;
    s3.X  += h * (3.0f/40.0f*k1.dX  + 9.0f/40.0f*k2.dX);
    s3.Y  += h * (3.0f/40.0f*k1.dY  + 9.0f/40.0f*k2.dY);
    s3.Z  += h * (3.0f/40.0f*k1.dZ  + 9.0f/40.0f*k2.dZ);
    s3.pX += h * (3.0f/40.0f*k1.dpX + 9.0f/40.0f*k2.dpX);
    s3.pY += h * (3.0f/40.0f*k1.dpY + 9.0f/40.0f*k2.dpY);
    s3.pZ += h * (3.0f/40.0f*k1.dpZ + 9.0f/40.0f*k2.dpZ);
    const KSDeriv k3 = eval_ks_rhs_state(s3, pT, M, a, Q);

    KSState s4 = s0;
    s4.X  += h * (44.0f/45.0f*k1.dX  - 56.0f/15.0f*k2.dX  + 32.0f/9.0f*k3.dX);
    s4.Y  += h * (44.0f/45.0f*k1.dY  - 56.0f/15.0f*k2.dY  + 32.0f/9.0f*k3.dY);
    s4.Z  += h * (44.0f/45.0f*k1.dZ  - 56.0f/15.0f*k2.dZ  + 32.0f/9.0f*k3.dZ);
    s4.pX += h * (44.0f/45.0f*k1.dpX - 56.0f/15.0f*k2.dpX + 32.0f/9.0f*k3.dpX);
    s4.pY += h * (44.0f/45.0f*k1.dpY - 56.0f/15.0f*k2.dpY + 32.0f/9.0f*k3.dpY);
    s4.pZ += h * (44.0f/45.0f*k1.dpZ - 56.0f/15.0f*k2.dpZ + 32.0f/9.0f*k3.dpZ);
    const KSDeriv k4 = eval_ks_rhs_state(s4, pT, M, a, Q);

    KSState s5 = s0;
    s5.X  += h * (19372.0f/6561.0f*k1.dX  - 25360.0f/2187.0f*k2.dX  + 64448.0f/6561.0f*k3.dX  - 212.0f/729.0f*k4.dX);
    s5.Y  += h * (19372.0f/6561.0f*k1.dY  - 25360.0f/2187.0f*k2.dY  + 64448.0f/6561.0f*k3.dY  - 212.0f/729.0f*k4.dY);
    s5.Z  += h * (19372.0f/6561.0f*k1.dZ  - 25360.0f/2187.0f*k2.dZ  + 64448.0f/6561.0f*k3.dZ  - 212.0f/729.0f*k4.dZ);
    s5.pX += h * (19372.0f/6561.0f*k1.dpX - 25360.0f/2187.0f*k2.dpX + 64448.0f/6561.0f*k3.dpX - 212.0f/729.0f*k4.dpX);
    s5.pY += h * (19372.0f/6561.0f*k1.dpY - 25360.0f/2187.0f*k2.dpY + 64448.0f/6561.0f*k3.dpY - 212.0f/729.0f*k4.dpY);
    s5.pZ += h * (19372.0f/6561.0f*k1.dpZ - 25360.0f/2187.0f*k2.dpZ + 64448.0f/6561.0f*k3.dpZ - 212.0f/729.0f*k4.dpZ);
    const KSDeriv k5 = eval_ks_rhs_state(s5, pT, M, a, Q);

    KSState s6 = s0;
    s6.X  += h * (9017.0f/3168.0f*k1.dX  - 355.0f/33.0f*k2.dX  + 46732.0f/5247.0f*k3.dX  + 49.0f/176.0f*k4.dX  - 5103.0f/18656.0f*k5.dX);
    s6.Y  += h * (9017.0f/3168.0f*k1.dY  - 355.0f/33.0f*k2.dY  + 46732.0f/5247.0f*k3.dY  + 49.0f/176.0f*k4.dY  - 5103.0f/18656.0f*k5.dY);
    s6.Z  += h * (9017.0f/3168.0f*k1.dZ  - 355.0f/33.0f*k2.dZ  + 46732.0f/5247.0f*k3.dZ  + 49.0f/176.0f*k4.dZ  - 5103.0f/18656.0f*k5.dZ);
    s6.pX += h * (9017.0f/3168.0f*k1.dpX - 355.0f/33.0f*k2.dpX + 46732.0f/5247.0f*k3.dpX + 49.0f/176.0f*k4.dpX - 5103.0f/18656.0f*k5.dpX);
    s6.pY += h * (9017.0f/3168.0f*k1.dpY - 355.0f/33.0f*k2.dpY + 46732.0f/5247.0f*k3.dpY + 49.0f/176.0f*k4.dpY - 5103.0f/18656.0f*k5.dpY);
    s6.pZ += h * (9017.0f/3168.0f*k1.dpZ - 355.0f/33.0f*k2.dpZ + 46732.0f/5247.0f*k3.dpZ + 49.0f/176.0f*k4.dpZ - 5103.0f/18656.0f*k5.dpZ);
    const KSDeriv k6 = eval_ks_rhs_state(s6, pT, M, a, Q);

    KSState y5 = s0;
    y5.X  += h * (35.0f/384.0f*k1.dX  + 500.0f/1113.0f*k3.dX  + 125.0f/192.0f*k4.dX  - 2187.0f/6784.0f*k5.dX  + 11.0f/84.0f*k6.dX);
    y5.Y  += h * (35.0f/384.0f*k1.dY  + 500.0f/1113.0f*k3.dY  + 125.0f/192.0f*k4.dY  - 2187.0f/6784.0f*k5.dY  + 11.0f/84.0f*k6.dY);
    y5.Z  += h * (35.0f/384.0f*k1.dZ  + 500.0f/1113.0f*k3.dZ  + 125.0f/192.0f*k4.dZ  - 2187.0f/6784.0f*k5.dZ  + 11.0f/84.0f*k6.dZ);
    y5.pX += h * (35.0f/384.0f*k1.dpX + 500.0f/1113.0f*k3.dpX + 125.0f/192.0f*k4.dpX - 2187.0f/6784.0f*k5.dpX + 11.0f/84.0f*k6.dpX);
    y5.pY += h * (35.0f/384.0f*k1.dpY + 500.0f/1113.0f*k3.dpY + 125.0f/192.0f*k4.dpY - 2187.0f/6784.0f*k5.dpY + 11.0f/84.0f*k6.dpY);
    y5.pZ += h * (35.0f/384.0f*k1.dpZ + 500.0f/1113.0f*k3.dpZ + 125.0f/192.0f*k4.dpZ - 2187.0f/6784.0f*k5.dpZ + 11.0f/84.0f*k6.dpZ);

    const KSDeriv k7 = eval_ks_rhs_state(y5, pT, M, a, Q);

    KSState y4 = s0;
    y4.X  += h * (5179.0f/57600.0f*k1.dX  + 7571.0f/16695.0f*k3.dX  + 393.0f/640.0f*k4.dX  - 92097.0f/339200.0f*k5.dX  + 187.0f/2100.0f*k6.dX  + 1.0f/40.0f*k7.dX);
    y4.Y  += h * (5179.0f/57600.0f*k1.dY  + 7571.0f/16695.0f*k3.dY  + 393.0f/640.0f*k4.dY  - 92097.0f/339200.0f*k5.dY  + 187.0f/2100.0f*k6.dY  + 1.0f/40.0f*k7.dY);
    y4.Z  += h * (5179.0f/57600.0f*k1.dZ  + 7571.0f/16695.0f*k3.dZ  + 393.0f/640.0f*k4.dZ  - 92097.0f/339200.0f*k5.dZ  + 187.0f/2100.0f*k6.dZ  + 1.0f/40.0f*k7.dZ);
    y4.pX += h * (5179.0f/57600.0f*k1.dpX + 7571.0f/16695.0f*k3.dpX + 393.0f/640.0f*k4.dpX - 92097.0f/339200.0f*k5.dpX + 187.0f/2100.0f*k6.dpX + 1.0f/40.0f*k7.dpX);
    y4.pY += h * (5179.0f/57600.0f*k1.dpY + 7571.0f/16695.0f*k3.dpY + 393.0f/640.0f*k4.dpY - 92097.0f/339200.0f*k5.dpY + 187.0f/2100.0f*k6.dpY + 1.0f/40.0f*k7.dpY);
    y4.pZ += h * (5179.0f/57600.0f*k1.dpZ + 7571.0f/16695.0f*k3.dpZ + 393.0f/640.0f*k4.dpZ - 92097.0f/339200.0f*k5.dpZ + 187.0f/2100.0f*k6.dpZ + 1.0f/40.0f*k7.dpZ);

    const float sX = tol * (1.0f + abs(s0.X));
    const float sY = tol * (1.0f + abs(s0.Y));
    const float sZ = tol * (1.0f + abs(s0.Z));
    const float spX = tol * (1.0f + abs(s0.pX));
    const float spY = tol * (1.0f + abs(s0.pY));
    const float spZ = tol * (1.0f + abs(s0.pZ));
    const float er0 = (y5.X - y4.X) / max(sX, 1e-12f);
    const float er1 = (y5.Y - y4.Y) / max(sY, 1e-12f);
    const float er2 = (y5.Z - y4.Z) / max(sZ, 1e-12f);
    const float er3 = (y5.pX - y4.pX) / max(spX, 1e-12f);
    const float er4 = (y5.pY - y4.pY) / max(spY, 1e-12f);
    const float er5 = (y5.pZ - y4.pZ) / max(spZ, 1e-12f);
    const float err = sqrt((er0*er0 + er1*er1 + er2*er2 + er3*er3 + er4*er4 + er5*er5) / 6.0f);

    if (!isfinite(err)) {
        h = max(h * 0.5f, ADAPT_H_MIN);
        return false;
    }

    if (err <= 1.0f || h < ADAPT_H_MIN) {
        X = y5.X; Y = y5.Y; Z = y5.Z; pX = y5.pX; pY = y5.pY; pZ = y5.pZ;
        const float fac = (err > ADAPT_ERR_TINY_DOPRI) ? 0.9f * pow(1.0f/err, 0.2f) : 2.0f;
        h = clamp(h * fac, ADAPT_H_MIN, ADAPT_H_MAX);
        return true;
    }

    h = clamp(h * 0.9f * pow(1.0f/err, 0.25f), ADAPT_H_MIN, h * 0.5f);
    return false;
}

static bool adaptive_step_ks(thread float& X, thread float& Y, thread float& Z,
                             thread float& pX, thread float& pY, thread float& pZ,
                             thread float& h,
                             float pT, float M, float a, float Q,
                             int integrator_mode,
                             float tol) {
    if (integrator_mode == 1) {
        return dopri5_adaptive_ks(X, Y, Z, pX, pY, pZ, h, pT, M, a, Q, tol);
    }

    const float step_used = h;
    float X_h = X, Y_h = Y, Z_h = Z, pX_h = pX, pY_h = pY, pZ_h = pZ;
    rk4_KS(X_h, Y_h, Z_h, pX_h, pY_h, pZ_h, pT, M, a, Q, step_used);

    float X_f = X, Y_f = Y, Z_f = Z, pX_f = pX, pY_f = pY, pZ_f = pZ;
    rk4_KS(X_f, Y_f, Z_f, pX_f, pY_f, pZ_f, pT, M, a, Q, step_used*0.5f);
    rk4_KS(X_f, Y_f, Z_f, pX_f, pY_f, pZ_f, pT, M, a, Q, step_used*0.5f);

    const float err = sqrt(
        (X_h-X_f)*(X_h-X_f) + (Y_h-Y_f)*(Y_h-Y_f) + (Z_h-Z_f)*(Z_h-Z_f) +
        (pX_h-pX_f)*(pX_h-pX_f) + (pY_h-pY_f)*(pY_h-pY_f) + (pZ_h-pZ_f)*(pZ_h-pZ_f)
    ) / 15.0f;
    if (err < tol || h < ADAPT_H_MIN) {
        X = X_f; Y = Y_f; Z = Z_f; pX = pX_f; pY = pY_f; pZ = pZ_f;
        const float sc = (err > ADAPT_ERR_TINY_RK4) ? 0.9f*pow(tol/err, 0.2f) : 2.0f;
        h = clamp(h*sc, ADAPT_H_MIN, ADAPT_H_MAX);
        return true;
    }

    h = clamp(h*0.9f*pow(tol/err, 0.25f), ADAPT_H_MIN, h*0.5f);
    return false;
}

// ── Separable Kerr semi-analytic path (BL, Q=0, Lambda=0) ───
struct SeparableConsts {
    float M, a;
    float E;   // E = -p_t
    float Lz;  // p_phi
    float Qc;  // Carter constant
};

struct SeparableState {
    float r, theta, phi;
    int sgn_r, sgn_th;
};

static float kerr_sep_delta(thread const SeparableConsts& c, float r) {
    return r*r - 2.0f*c.M*r + c.a*c.a;
}

static float kerr_sep_R(thread const SeparableConsts& c, float r) {
    const float r2 = r*r;
    const float P = c.E*(r2 + c.a*c.a) - c.a*c.Lz;
    const float K = c.Qc + (c.Lz - c.a*c.E)*(c.Lz - c.a*c.E);
    return P*P - kerr_sep_delta(c, r) * K;
}

static float kerr_sep_Theta(thread const SeparableConsts& c, float theta) {
    const float st = sin(theta);
    const float ct = cos(theta);
    const float st2 = max(st*st, 1e-14f);
    return c.Qc - ct*ct * (c.Lz*c.Lz/st2 - c.a*c.a*c.E*c.E);
}

static bool kerr_sep_rhs(thread const SeparableConsts& c, thread SeparableState& s,
                         thread float& dr, thread float& dth, thread float& dphi) {
    const float st = sin(s.theta);
    const float ct = cos(s.theta);
    const float st2 = max(st*st, 1e-14f);
    const float sigma = max(s.r*s.r + c.a*c.a*ct*ct, 1e-14f);
    const float delta = kerr_sep_delta(c, s.r);
    const float P = c.E*(s.r*s.r + c.a*c.a) - c.a*c.Lz;
    const float R = kerr_sep_R(c, s.r);
    const float Th = kerr_sep_Theta(c, s.theta);

    dr = ((s.sgn_r >= 0) ? 1.0f : -1.0f) * sqrt(max(R, 0.0f)) / sigma;
    dth = ((s.sgn_th >= 0) ? 1.0f : -1.0f) * sqrt(max(Th, 0.0f)) / sigma;
    dphi = (c.Lz/st2 + c.a*(P/max(delta, 1e-14f) - c.E)) / sigma;
    return isfinite(dr) && isfinite(dth) && isfinite(dphi);
}

static bool rk4_step_separable_kerr(thread const SeparableConsts& c, thread SeparableState& s, float h) {
    float dr1,dth1,dphi1;
    float dr2,dth2,dphi2;
    float dr3,dth3,dphi3;
    float dr4,dth4,dphi4;

    SeparableState s0 = s;
    if (!kerr_sep_rhs(c, s0, dr1, dth1, dphi1)) return false;

    SeparableState s2 = s0;
    s2.r     = s0.r     + 0.5f*h*dr1;
    s2.theta = s0.theta + 0.5f*h*dth1;
    s2.phi   = s0.phi   + 0.5f*h*dphi1;
    if (!kerr_sep_rhs(c, s2, dr2, dth2, dphi2)) return false;

    SeparableState s3 = s0;
    s3.r     = s0.r     + 0.5f*h*dr2;
    s3.theta = s0.theta + 0.5f*h*dth2;
    s3.phi   = s0.phi   + 0.5f*h*dphi2;
    if (!kerr_sep_rhs(c, s3, dr3, dth3, dphi3)) return false;

    SeparableState s4 = s0;
    s4.r     = s0.r     + h*dr3;
    s4.theta = s0.theta + h*dth3;
    s4.phi   = s0.phi   + h*dphi3;
    if (!kerr_sep_rhs(c, s4, dr4, dth4, dphi4)) return false;

    s.r     += h/6.0f*(dr1   + 2.0f*dr2   + 2.0f*dr3   + dr4);
    s.theta += h/6.0f*(dth1  + 2.0f*dth2  + 2.0f*dth3  + dth4);
    s.phi   += h/6.0f*(dphi1 + 2.0f*dphi2 + 2.0f*dphi3 + dphi4);
    return isfinite(s.r) && isfinite(s.theta) && isfinite(s.phi);
}

static bool rk4_adaptive_separable_kerr(thread const SeparableConsts& c,
                                        thread SeparableState& s,
                                        thread float& h,
                                        float tol = 1e-6f) {
    const SeparableState s0 = s;

    SeparableState sA = s0;
    if (!rk4_step_separable_kerr(c, sA, h)) {
        h = max(h*0.5f, 1e-10f);
        return false;
    }

    SeparableState sB = s0;
    if (!rk4_step_separable_kerr(c, sB, 0.5f*h) ||
        !rk4_step_separable_kerr(c, sB, 0.5f*h)) {
        h = max(h*0.5f, 1e-10f);
        return false;
    }

    const float err = sqrt(
        (sA.r - sB.r)*(sA.r - sB.r) +
        (sA.theta - sB.theta)*(sA.theta - sB.theta) +
        (sA.phi - sB.phi)*(sA.phi - sB.phi)
    ) / 15.0f;

    if (!isfinite(err)) {
        h = max(h*0.5f, 1e-10f);
        return false;
    }

    if (err < tol || h < 1e-10f) {
        s = sB;
        const float sc = (err > 1e-14f) ? 0.9f * pow(tol/err, 0.2f) : 4.0f;
        h = clamp(h*sc, 1e-10f, 100.0f);
        return true;
    }

    h = max(h * 0.9f * pow(tol/err, 0.25f), 1e-10f);
    return false;
}

// ── Elliptic-closed helpers (GPU) ────────────────────────────
struct Cx { float re, im; };
static Cx cx_make(float re, float im) { Cx z{re, im}; return z; }
static Cx cx_add(Cx a, Cx b) { return cx_make(a.re + b.re, a.im + b.im); }
static Cx cx_sub(Cx a, Cx b) { return cx_make(a.re - b.re, a.im - b.im); }
static Cx cx_mul(Cx a, Cx b) {
    return cx_make(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re);
}
static Cx cx_add_scalar(Cx a, float s) { return cx_make(a.re + s, a.im); }
static float cx_abs(Cx a) { return sqrt(a.re*a.re + a.im*a.im); }
static Cx cx_div(Cx a, Cx b) {
    const float den = b.re*b.re + b.im*b.im;
    if (den < 1e-20f) return cx_make(NAN, NAN);
    return cx_make((a.re*b.re + a.im*b.im)/den, (a.im*b.re - a.re*b.im)/den);
}

static float carlson_rf(float x, float y, float z) {
    if (!(x >= 0.0f && y >= 0.0f && z >= 0.0f)) return NAN;
    if (!(isfinite(x) && isfinite(y) && isfinite(z))) return NAN;

    float xn = x, yn = y, zn = z;
    constexpr float ERR_TOL = 1e-6f;
    for (int it = 0; it < 48; ++it) {
        const float mu = (xn + yn + zn) / 3.0f;
        if (!(mu > 0.0f) || !isfinite(mu)) return NAN;

        const float X = 1.0f - xn / mu;
        const float Y = 1.0f - yn / mu;
        const float Z = 1.0f - zn / mu;
        const float eps = max(max(abs(X), abs(Y)), abs(Z));
        if (eps < ERR_TOL) {
            const float E2 = X*Y - Z*Z;
            const float E3 = X*Y*Z;
            const float corr =
                1.0f
                - E2 / 10.0f
                + E3 / 14.0f
                + (E2*E2) / 24.0f
                - (3.0f * E2 * E3) / 44.0f;
            return corr / sqrt(mu);
        }

        const float sx = sqrt(xn);
        const float sy = sqrt(yn);
        const float sz = sqrt(zn);
        const float lam = sx*sy + sx*sz + sy*sz;
        xn = 0.25f * (xn + lam);
        yn = 0.25f * (yn + lam);
        zn = 0.25f * (zn + lam);
    }
    return NAN;
}

static float ellint_F_incomplete(float phi, float m) {
    const float s = sin(phi);
    const float c = cos(phi);
    const float x = c*c;
    const float y = 1.0f - m*s*s;
    if (y <= 0.0f) return NAN;
    const float rf = carlson_rf(x, y, 1.0f);
    return s * rf;
}

static float ellint_K_complete(float m) {
    if (!(m >= 0.0f && m < 1.0f)) return NAN;
    return carlson_rf(0.0f, 1.0f - m, 1.0f);
}

static float jacobi_sn2_from_u(float u, float m, float Kc) {
    if (!(m >= 0.0f && m < 1.0f) || !(Kc > 0.0f) || !isfinite(u)) return NAN;
    float ur = fmod(u, 2.0f*Kc);
    if (ur < 0.0f) ur += 2.0f*Kc;
    if (ur > Kc) ur = 2.0f*Kc - ur;

    float lo = 0.0f, hi = 0.5f * M_PI_F;
    for (int it = 0; it < 40; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float Fm = ellint_F_incomplete(mid, m);
        if (!isfinite(Fm)) return NAN;
        if (Fm < ur) lo = mid; else hi = mid;
    }
    const float s = sin(0.5f * (lo + hi));
    return s*s;
}

static Cx eval_monic_quartic(float b0, float b1, float b2, float b3, Cx z) {
    Cx v = cx_add_scalar(z, b0);
    v = cx_add(cx_mul(v, z), cx_make(b1, 0.0f));
    v = cx_add(cx_mul(v, z), cx_make(b2, 0.0f));
    v = cx_add(cx_mul(v, z), cx_make(b3, 0.0f));
    return v;
}

static void quartic_roots_monic_complex(float b0, float b1, float b2, float b3,
                                        thread Cx roots_out[4]) {
    const float radius = 1.0f + max(max(abs(b0), abs(b1)), max(abs(b2), abs(b3)));
    Cx z[4] = {
        cx_make( radius, 0.0f),
        cx_make(0.0f,  radius),
        cx_make(-radius, 0.0f),
        cx_make(0.0f, -radius)
    };

    for (int it = 0; it < 96; ++it) {
        float max_delta = 0.0f;
        Cx zn[4];
        for (int i = 0; i < 4; ++i) {
            Cx den = cx_make(1.0f, 0.0f);
            for (int j = 0; j < 4; ++j) {
                if (i == j) continue;
                den = cx_mul(den, cx_sub(z[i], z[j]));
            }
            if (cx_abs(den) < 1e-16f) den = cx_make(1e-16f, 0.0f);
            const Cx pz = eval_monic_quartic(b0, b1, b2, b3, z[i]);
            const Cx delta = cx_div(pz, den);
            zn[i] = cx_sub(z[i], delta);
            max_delta = max(max_delta, cx_abs(delta));
        }
        for (int i = 0; i < 4; ++i) z[i] = zn[i];
        if (max_delta < 1e-6f) break;
    }

    for (int i = 0; i < 4; ++i) roots_out[i] = z[i];
}

static bool jacobi_sn_cn_from_u(float u, float m, float Kc,
                                thread float& sn_out,
                                thread float& cn_out) {
    if (!(m >= 0.0f && m < 1.0f) || !(Kc > 0.0f) || !isfinite(u)) return false;

    const float fourK = 4.0f * Kc;
    float ur = fmod(u, fourK);
    if (ur < 0.0f) ur += fourK;

    int quadrant = 0;
    if (ur > 2.0f * Kc) {
        ur -= 2.0f * Kc;
        quadrant = 2;
    }
    if (ur > Kc) {
        ur = 2.0f * Kc - ur;
        quadrant ^= 1;
    }

    float lo = 0.0f;
    float hi = 0.5f * M_PI_F;
    for (int it = 0; it < 42; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float Fm = ellint_F_incomplete(mid, m);
        if (!isfinite(Fm)) return false;
        if (Fm < ur) lo = mid; else hi = mid;
    }

    const float phi = 0.5f * (lo + hi);
    float sn = sin(phi);
    float cn = cos(phi);

    if ((quadrant & 1) != 0) cn = -cn;
    if ((quadrant & 2) != 0) sn = -sn;

    sn_out = sn;
    cn_out = cn;
    return true;
}

static bool jacobi_sc_from_u(float u, float m, float Kc,
                             thread float& sc_out) {
    float sn = 0.0f, cn = 0.0f;
    if (!jacobi_sn_cn_from_u(u, m, Kc, sn, cn)) return false;
    if (abs(cn) < 1e-8f) return false;
    sc_out = sn / cn;
    return true;
}

enum {
    ELLIPTIC_REGION_I_II_REAL4 = 0,
    ELLIPTIC_REGION_III_REAL2_COMPLEX2 = 1,
    ELLIPTIC_REGION_IV_COMPLEX4 = 2
};

struct EllipticRadialMap {
    int   radial_case = ELLIPTIC_REGION_I_II_REAL4;

    float r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f;
    float m = 0.0f;
    float Kc = 0.0f;
    float omega = 0.0f;
    float X0 = 0.0f;

    int phase_sign = 1;
    int tau_sign = 1;

    float A = 0.0f, B = 0.0f;
    float r_lo = 0.0f, r_hi = 0.0f;

    float a2 = 0.0f;
    float b1 = 0.0f;
    float g0 = 0.0f;
};

static float elliptic_radial_r_from_phase(thread const EllipticRadialMap& mp, float X) {
    if (mp.radial_case == ELLIPTIC_REGION_I_II_REAL4) {
        const float sn2 = jacobi_sn2_from_u(X, mp.m, mp.Kc);
        if (!isfinite(sn2)) return NAN;

        const float r24 = (mp.r2 - mp.r4);
        const float r14 = (mp.r1 - mp.r4);
        const float den = r24 - r14 * sn2;
        if (abs(den) < 1e-8f) return NAN;
        return (mp.r1 * r24 - mp.r2 * r14 * sn2) / den;
    }

    if (mp.radial_case == ELLIPTIC_REGION_III_REAL2_COMPLEX2) {
        float sn = 0.0f, cn = 0.0f;
        if (!jacobi_sn_cn_from_u(X, mp.m, mp.Kc, sn, cn)) return NAN;

        const float num = (mp.B * mp.r_hi - mp.A * mp.r_lo)
                        + (mp.B * mp.r_hi + mp.A * mp.r_lo) * cn;
        const float den = (mp.B - mp.A) + (mp.B + mp.A) * cn;
        if (abs(den) < 1e-8f) return NAN;
        return num / den;
    }

    float sc = 0.0f;
    if (!jacobi_sc_from_u(X, mp.m, mp.Kc, sc)) return NAN;
    const float den = 1.0f + mp.g0 * sc;
    if (abs(den) < 1e-8f) return NAN;
    return -mp.a2 * ((mp.g0 - sc) / den) - mp.b1;
}

static float conj_dist_cx(Cx a, Cx b) {
    return sqrt((a.re - b.re)*(a.re - b.re) + (a.im + b.im)*(a.im + b.im));
}

static void pair_params_cx(Cx zp, Cx zn,
                           thread float& b,
                           thread float& a) {
    b = 0.5f * (zp.re + zn.re);
    a = 0.5f * (abs(zp.im) + abs(zn.im));
}

static void align_initial_radial_sign(thread EllipticRadialMap& mp,
                                      float dr0,
                                      bool flip_phase,
                                      bool flip_tau) {
    if (abs(dr0) < 1e-10f) return;

    const float probe_tau = 1e-4f;
    const float Xp = float(mp.phase_sign) * mp.X0
                   + float(mp.tau_sign) * mp.omega * probe_tau;
    const float Xm = float(mp.phase_sign) * mp.X0
                   - float(mp.tau_sign) * mp.omega * probe_tau;
    const float rp = elliptic_radial_r_from_phase(mp, Xp);
    const float rm = elliptic_radial_r_from_phase(mp, Xm);
    if (!isfinite(rp) || !isfinite(rm)) return;

    const bool want_out = (dr0 >= 0.0f);
    const bool is_out = (rp >= rm);
    if (want_out == is_out) return;

    if (flip_phase) mp.phase_sign *= -1;
    if (flip_tau)   mp.tau_sign *= -1;
}

static bool init_elliptic_radial_map(thread const SeparableConsts& c,
                                     float r0, float dr0,
                                     thread EllipticRadialMap& out) {
    const float Kpot = c.Qc + (c.Lz - c.a*c.E)*(c.Lz - c.a*c.E);
    const float A0 = c.E*c.a*c.a - c.a*c.Lz;
    const float c4 = c.E*c.E;
    if (!(c4 > 1e-10f)) return false;
    const float c2 = 2.0f*c.E*A0 - Kpot;
    const float c1 = 2.0f*c.M*Kpot;
    const float c0 = A0*A0 - c.a*c.a*Kpot;

    Cx roots[4];
    quartic_roots_monic_complex(0.0f, c2/c4, c1/c4, c0/c4, roots);

    float real_roots[4];
    Cx complex_roots[4];
    int nreal = 0;
    int ncomplex = 0;

    for (int i = 0; i < 4; ++i) {
        if (!isfinite(roots[i].re) || !isfinite(roots[i].im)) return false;
        const float tol_im = 1e-4f * max(1.0f, abs(roots[i].re));
        if (abs(roots[i].im) <= tol_im) {
            if (nreal >= 4) return false;
            real_roots[nreal++] = roots[i].re;
        } else {
            if (ncomplex >= 4) return false;
            complex_roots[ncomplex++] = roots[i];
        }
    }

    if (nreal == 4) {
        for (int i = 0; i < 4; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                if (real_roots[j] > real_roots[i]) {
                    const float t = real_roots[i];
                    real_roots[i] = real_roots[j];
                    real_roots[j] = t;
                }
            }
        }

        out = EllipticRadialMap{};
        out.radial_case = ELLIPTIC_REGION_I_II_REAL4;
        out.r1 = real_roots[0];
        out.r2 = real_roots[1];
        out.r3 = real_roots[2];
        out.r4 = real_roots[3];

        const float den_omega = (out.r1 - out.r3) * (out.r2 - out.r4);
        if (!(den_omega > 0.0f)) return false;
        out.omega = 0.5f * sqrt(den_omega);

        const float k_den = (out.r2 - out.r4) * (out.r1 - out.r3);
        if (!(k_den > 0.0f)) return false;
        out.m = ((out.r2 - out.r3) * (out.r1 - out.r4)) / k_den;
        if (!(out.m >= 0.0f && out.m < 1.0f)) return false;
        out.Kc = ellint_K_complete(out.m);
        if (!isfinite(out.Kc) || !(out.Kc > 0.0f)) return false;

        const float den_obs = (r0 - out.r2);
        const float den_root = (out.r1 - out.r4);
        if (abs(den_obs) < 1e-8f || abs(den_root) < 1e-8f) return false;
        const float u0_raw = ((r0 - out.r1) / den_obs) * ((out.r2 - out.r4) / den_root);
        if (!isfinite(u0_raw)) return false;
        if (u0_raw < -1e-4f || u0_raw > 1.0f + 1e-4f) return false;
        const float u0 = clamp(u0_raw, 0.0f, 1.0f);

        out.X0 = ellint_F_incomplete(asin(sqrt(u0)), out.m);
        if (!isfinite(out.X0)) return false;

        const bool case1 = (r0 >= out.r3 && r0 <= out.r2);
        const int alpha = case1 ? -1 : +1;
        const int nu_r = (dr0 >= 0.0f) ? +1 : -1;
        out.phase_sign = alpha * nu_r;
        out.tau_sign = +1;
        align_initial_radial_sign(out, dr0, /*flip_phase=*/true, /*flip_tau=*/false);
        return true;
    }

    if (nreal == 2 && ncomplex == 2) {
        if (real_roots[0] > real_roots[1]) {
            const float t = real_roots[0];
            real_roots[0] = real_roots[1];
            real_roots[1] = t;
        }
        const float r_lo = real_roots[0];
        const float r_hi = real_roots[1];

        const float b1 = 0.5f * (complex_roots[0].re + complex_roots[1].re);
        const float a1 = 0.5f * (abs(complex_roots[0].im) + abs(complex_roots[1].im));
        if (!(a1 > 0.0f) || !isfinite(a1) || !isfinite(b1)) return false;

        const float A = sqrt(a1*a1 + (b1 - r_hi)*(b1 - r_hi));
        const float B = sqrt(a1*a1 + (b1 - r_lo)*(b1 - r_lo));
        if (!(A > 0.0f && B > 0.0f && B > A)) return false;

        const float r21 = r_hi - r_lo;
        const float k3 = (((A + B) * (A + B)) - r21 * r21) / (4.0f * A * B);
        if (!(k3 > 0.0f && k3 < 1.0f)) return false;
        const float K3 = ellint_K_complete(k3);
        if (!isfinite(K3) || !(K3 > 0.0f)) return false;

        const float x3_den = A*(r0 - r_lo) + B*(r0 - r_hi);
        if (abs(x3_den) < 1e-8f) return false;
        const float x3 = (A*(r0 - r_lo) - B*(r0 - r_hi)) / x3_den;
        if (!isfinite(x3)) return false;
        const float phi0 = acos(clamp(x3, -1.0f, 1.0f));
        const float F0 = ellint_F_incomplete(phi0, k3);
        if (!isfinite(F0)) return false;

        out = EllipticRadialMap{};
        out.radial_case = ELLIPTIC_REGION_III_REAL2_COMPLEX2;
        out.r1 = r_hi;
        out.r2 = r_lo;
        out.r_lo = r_lo;
        out.r_hi = r_hi;
        out.A = A;
        out.B = B;
        out.m = k3;
        out.Kc = K3;
        out.omega = sqrt(A * B);
        out.X0 = F0;
        out.phase_sign = (dr0 >= 0.0f) ? +1 : -1;
        out.tau_sign = +1;

        align_initial_radial_sign(out, dr0, /*flip_phase=*/true, /*flip_tau=*/false);
        return true;
    }

    if (nreal == 0 && ncomplex == 4) {
        Cx pos_im[2];
        Cx neg_im[2];
        int npos = 0;
        int nneg = 0;

        for (int i = 0; i < 4; ++i) {
            if (complex_roots[i].im >= 0.0f) {
                if (npos >= 2) return false;
                pos_im[npos++] = complex_roots[i];
            } else {
                if (nneg >= 2) return false;
                neg_im[nneg++] = complex_roots[i];
            }
        }
        if (npos != 2 || nneg != 2) return false;

        const int j0 = (conj_dist_cx(pos_im[0], neg_im[0]) <= conj_dist_cx(pos_im[0], neg_im[1])) ? 0 : 1;

        float bA = 0.0f, aA = 0.0f;
        float bB = 0.0f, aB = 0.0f;
        pair_params_cx(pos_im[0], neg_im[j0], bA, aA);
        pair_params_cx(pos_im[1], neg_im[1 - j0], bB, aB);
        if (!(aA > 0.0f && aB > 0.0f)) return false;

        float b1 = bA, a1 = aA;
        float b2 = bB, a2 = aB;
        if (bB > bA) {
            b1 = bB; a1 = aB;
            b2 = bA; a2 = aA;
        }
        if (!(b1 > b2)) return false;

        const float C = sqrt((a1 - a2)*(a1 - a2) + (b1 - b2)*(b1 - b2));
        const float D = sqrt((a1 + a2)*(a1 + a2) + (b1 - b2)*(b1 - b2));
        if (!(C > 0.0f && D > 0.0f)) return false;

        const float k4 = (4.0f * C * D) / ((C + D) * (C + D));
        if (!(k4 > 0.0f && k4 < 1.0f)) return false;
        const float K4 = ellint_K_complete(k4);
        if (!isfinite(K4) || !(K4 > 0.0f)) return false;

        const float g_num = 4.0f * a2 * a2 - (C - D) * (C - D);
        const float g_den = (C + D) * (C + D) - 4.0f * a2 * a2;
        if (!(g_num > 0.0f && g_den > 0.0f)) return false;
        const float g0 = sqrt(g_num / g_den);
        if (!(g0 > 0.0f) || !isfinite(g0)) return false;

        const float x4 = (r0 - b2) / max(a2, 1e-8f);
        const float phi0 = atan(x4) + atan(g0);
        const float F0 = ellint_F_incomplete(phi0, k4);
        if (!isfinite(F0)) return false;

        out = EllipticRadialMap{};
        out.radial_case = ELLIPTIC_REGION_IV_COMPLEX4;
        out.m = k4;
        out.Kc = K4;
        out.omega = 0.5f * (C + D);
        out.X0 = F0;
        out.phase_sign = +1;
        out.tau_sign = (dr0 >= 0.0f) ? +1 : -1;
        out.a2 = a2;
        out.b1 = b1;
        out.g0 = g0;

        align_initial_radial_sign(out, dr0, /*flip_phase=*/false, /*flip_tau=*/true);
        return true;
    }

    return false;
}

static float separable_radial_potential(thread const SeparableConsts& c, float r) {
    const float Kpot = c.Qc + (c.Lz - c.a*c.E) * (c.Lz - c.a*c.E);
    const float A0 = c.E * (r*r + c.a*c.a) - c.a * c.Lz;
    const float Delta = r*r - 2.0f*c.M*r + c.a*c.a;
    return A0*A0 - Delta*Kpot;
}

static bool first_equator_crossing_mino_time(thread const SeparableConsts& c,
                                             float theta0, float dtheta0,
                                             thread float& tau_first,
                                             thread float& tau_period) {
    const float A = c.a*c.a*c.E*c.E;
    if (!(A > 1e-8f)) return false;

    const float B = A - c.Lz*c.Lz - c.Qc;
    const float disc = B*B + 4.0f*A*c.Qc;
    if (!(disc >= 0.0f)) return false;
    const float sq = sqrt(disc);
    const float u_plus  = (B + sq) / (2.0f*A);
    const float u_minus = (B - sq) / (2.0f*A);
    if (!(u_plus > 0.0f && u_minus < 0.0f)) return false;

    // Dexter & Agol 2009 (M_- < 0 branch, Eq. 42):
    //   mu(tau) = mu_+ * cn(u, k),  k^2 = u_+ / (u_+ - u_-)
    const float m = u_plus / (u_plus - u_minus);
    if (!(m >= 0.0f && m < 1.0f)) return false;
    const float Kc = ellint_K_complete(m);
    if (!isfinite(Kc) || !(Kc > 0.0f)) return false;

    const float omega = sqrt(A * (u_plus - u_minus));
    if (!(omega > 0.0f) || !isfinite(omega)) return false;

    const float mu_plus = sqrt(u_plus);
    const float mu0 = cos(theta0);
    const float c0 = clamp(abs(mu0) / max(mu_plus, 1e-8f), 0.0f, 1.0f);
    const float phi0 = acos(c0);
    const float u_base = ellint_F_incomplete(phi0, m);
    if (!isfinite(u_base)) return false;

    tau_period = 4.0f * Kc / omega;
    if (!(tau_period > 0.0f) || !isfinite(tau_period)) return false;

    const bool toward_equator = (cos(theta0) * dtheta0) > 0.0f;
    const float u0_phase = toward_equator ? u_base : -u_base;
    tau_first = (Kc - u0_phase) / omega;
    if (tau_first < 0.0f) tau_first += tau_period;
    return isfinite(tau_first);
}

// Returns: 0=inconclusive, 1=horizon, 2=disk
static int elliptic_closed_first_hit(thread const SeparableConsts& c,
                                     float r0, float theta0, float dr0, float dtheta0,
                                     float r_horizon, float r_isco, float r_disk_out, float r_escape,
                                     thread float& r_hit_out) {
    float tau_first = 0.0f, tau_period = 0.0f;
    if (!first_equator_crossing_mino_time(c, theta0, dtheta0, tau_first, tau_period))
        return 0;
    (void)tau_period;

    EllipticRadialMap mp{};
    if (!init_elliptic_radial_map(c, r0, dr0, mp))
        return 0;

    // Keep GPU behaviour aligned with CPU elliptic-closed:
    // evaluate the radial map only at the first equatorial crossing.
    const float X = float(mp.phase_sign) * mp.X0
                  + float(mp.tau_sign) * mp.omega * tau_first;
    const float r_now = elliptic_radial_r_from_phase(mp, X);
    if (!isfinite(r_now)) return 0;
    const float R = separable_radial_potential(c, r_now);
    const float R_scale = max(1.0f, abs(c.E*c.E*r_now*r_now*r_now*r_now));
    if (R < -1e-6f * R_scale) return 0;
    // For Region I/II and III, require the crossing radius to stay in the
    // observer-accessible outer branch. Region IV has no real turning points.
    if (mp.radial_case != ELLIPTIC_REGION_IV_COMPLEX4 && r_now < mp.r1) return 0;
    if (r_now < r_horizon * 1.03f) return 0;
    if (r_now > r_escape) return 0;
    if (r_now >= r_isco && r_now <= r_disk_out) {
        r_hit_out = r_now;
        return 2;
    }
    return 0;
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

static inline float doppler_intensity_scale(float redshift) {
    const float red_c = clamp(redshift, 0.1f, 10.0f);
    const float receding_lift = 1.0f + 0.85f * clamp(1.0f - red_c, 0.0f, 1.0f);
    return pow(red_c, 4.0f) * receding_lift;
}

static inline float robust_disk_redshift(float r_hit,
                                         float pt_cov,
                                         float pphi_cov,
                                         float M, float a, float Q, float L) {
    const float pt_abs = max(abs(pt_cov), 1e-12f);
    const float Omega = keplerian_omega(r_hit, M, a, Q, L);
    const float b_ip  = -pphi_cov / pt_abs;
    float gll2[4][4];
    gLL_BL(r_hit, M_PI_2_F, M, a, Q, L, gll2);
    const float d2 = -(gll2[0][0] + 2.0f*gll2[0][3]*Omega + gll2[3][3]*Omega*Omega);
    if (!(d2 > 0.0f)) return 1.0f;
    // Avoid sign-flip seams from tiny interpolation errors around dv≈0.
    const float dv = abs(1.0f - Omega*b_ip);
    if (dv < 1e-8f) return 20.0f;
    const float red = sqrt(d2) / dv;
    if (!isfinite(red)) return 1.0f;
    return clamp(red, 0.0f, 20.0f);
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

struct RayTraceResultBL {
    int outcome;   // 0 = escape, 1 = disk, 2 = horizon
    float r_hit;
    float redshift;
    float theta_esc;
    float phi_esc;
    float phi_hit;
};

static float wrap_delta_phi(float dphi) {
    constexpr float PI_F = 3.14159265358979323846f;
    constexpr float TWO_PI_F = 6.28318530717958647692f;
    dphi = fmod(dphi + PI_F, TWO_PI_F);
    if (dphi < 0.0f) dphi += TWO_PI_F;
    return dphi - PI_F;
}

static float median3v(float a, float b, float c) {
    return a + b + c - min(a, min(b, c)) - max(a, max(b, c));
}

static float median5v(float a, float b, float c, float d, float e) {
    float v[5] = {a, b, c, d, e};
    for (int i = 1; i < 5; ++i) {
        const float key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            --j;
        }
        v[j + 1] = key;
    }
    return v[2];
}

static uint32_t pack_abgr(float r, float g, float b) {
    const uint32_t rr = uint32_t(clamp(r, 0.0f, 1.0f) * 255.0f);
    const uint32_t gg = uint32_t(clamp(g, 0.0f, 1.0f) * 255.0f);
    const uint32_t bb = uint32_t(clamp(b, 0.0f, 1.0f) * 255.0f);
    return (0xFFu << 24) | (bb << 16) | (gg << 8) | rr;
}

static uint32_t disk_color_abgr(float r_hit, float redshift, float magnif,
                                float M, float r_isco) {
    const float red = clamp(redshift, 0.0f, 20.0f);
    const float T = 6500.0f * sqrt(6.0f*M/r_hit) * clamp(red, 0.2f, 5.0f);
    float I = page_thorne_norm(r_hit, r_isco);
    I *= doppler_intensity_scale(red);
    I *= clamp(1.0f / max(magnif, 1e-12f), 0.05f, 5.0f);
    const float4 bb = blackbody_rgb(T);
    const float rr = tonemap_ch(bb.r * I, 1.0f, 2.2f);
    const float gg = tonemap_ch(bb.g * I, 1.0f, 2.2f);
    const float bbv = tonemap_ch(bb.b * I, 1.0f, 2.2f);
    return pack_abgr(rr, gg, bbv);
}

static RayTraceResultBL trace_standard_bl_from_angles(float alpha, float beta,
                                                      float M, float a, float Q, float L,
                                                      float r_obs, float theta_obs, float phi_obs,
                                                      float r_horizon, float r_isco, float r_disk_out,
                                                      int intersection_mode,
                                                      int integrator_mode,
                                                      int max_steps,
                                                      float step_init,
                                                      float integrator_tol) {
    RayTraceResultBL res{};
    res.outcome = 2;
    res.r_hit = 0.0f;
    res.redshift = 1.0f;
    res.theta_esc = theta_obs;
    res.phi_esc = phi_obs;
    res.phi_hit = phi_obs;

    float gll[4][4];
    gLL_BL(r_obs, theta_obs, M, a, Q, L, gll);

    float gu[4][4];
    gUU(r_obs, theta_obs, M, a, Q, L, gu);

    const float ca = cos(alpha), sa = sin(alpha);
    const float cb = cos(beta),  sb = sin(beta);

    const float gtt   = gll[0][0];
    const float gtphi = gll[0][3];
    const float grr   = gll[1][1];
    const float gthth = gll[2][2];
    const float gphph = gll[3][3];

    const float sqrt_grr   = sqrt(abs(grr));
    const float sqrt_gthth = sqrt(abs(gthth));
    const float denom_phi  = gphph - (gtphi*gtphi)/gtt;

    float pUt   = 1.0f;
    float pUr   = -ca*cb / max(sqrt_grr, 1e-14f);
    float pUth  = -sb    / max(sqrt_gthth, 1e-14f);
    float pUphi = -sa*cb / sqrt(max(abs(gphph), 1e-14f));

    const bool tetrad_ok =
        (gtt < -1e-14f) &&
        isfinite(denom_phi) && (denom_phi > 1e-14f) &&
        isfinite(sqrt_grr)  && (sqrt_grr  > 1e-14f) &&
        isfinite(sqrt_gthth)&& (sqrt_gthth> 1e-14f);
    if (tetrad_ok) {
        const float ut     = 1.0f / sqrt(-gtt);
        const float ephi_p = 1.0f / sqrt(denom_phi);
        const float ephi_t = -gtphi/gtt * ephi_p;
        const float n_r  = -ca * cb;
        const float n_th = -sb;
        const float n_ph = -sa * cb;
        pUt   = ut + n_ph * ephi_t;
        pUr   = n_r  / sqrt_grr;
        pUth  = n_th / sqrt_gthth;
        pUphi = n_ph * ephi_p;
    }

    float pt   = gtt* pUt + gtphi*pUphi;
    float pr   = grr* pUr;
    float pth  = gthth*pUth;
    float pphi = gtphi*pUt + gphph*pUphi;

    const float A = gu[0][0];
    const float B = 2.0f * gu[0][3] * pphi;
    const float C = gu[1][1]*pr*pr + gu[2][2]*pth*pth + gu[3][3]*pphi*pphi;
    const float disc = B*B - 4.0f*A*C;
    if (disc >= 0.0f && abs(A) > 1e-15f) {
        const float sq  = sqrt(disc);
        const float pt1 = (-B - sq)/(2.0f*A);
        const float pt2 = (-B + sq)/(2.0f*A);
        pt = (pt1 < 0.0f) ? pt1 : pt2;
        if (pt > 0.0f) pt = min(pt1, pt2);
    }

    float r = r_obs, theta = theta_obs, phi = phi_obs;
    float dlam = max(step_init, ADAPT_H_MIN);
    float prev_r = r_obs;
    float prev_theta = theta_obs;
    float prev_phi = phi_obs;
    float prev_pr = pr;
    float prev_pth = pth;
    const float r_escape = r_obs * 1.05f;

    const int iter_cap = max(max_steps, 1);
    for (int iter = 0; iter < iter_cap; ++iter) {
        const float step_used = dlam;
        if (!adaptive_step_bl(r, theta, phi, pr, pth, dlam, pt, pphi, M, a, Q, L,
                              integrator_mode, integrator_tol)) {
            continue;
        }

        if (!(isfinite(r) && isfinite(theta) && isfinite(phi) && isfinite(pr) && isfinite(pth))) {
            return res;
        }

        float best_alpha = 2.0f;
        int best_event = 0; // 0 none, 1 disk, 2 horizon, 3 escape

        const float q0 = prev_theta - M_PI_2_F;
        const float q1 = theta - M_PI_2_F;
        const bool maybe_equator = sign_change_f(q0, q1) ||
                                   (intersection_mode != 0 && min(abs(q0), abs(q1)) < 0.35f);
        if (maybe_equator) {
            float dr0=0.0f, dth0=0.0f, dphi0=0.0f, dpr0=0.0f, dpth0=0.0f;
            float dr1=0.0f, dth1=0.0f, dphi1=0.0f, dpr1=0.0f, dpth1=0.0f;
            geodesic_rhs(prev_r, prev_theta, prev_pr, prev_pth, pt, pphi, M, a, Q, L,
                         dr0, dth0, dphi0, dpr0, dpth0);
            geodesic_rhs(r, theta, pr, pth, pt, pphi, M, a, Q, L,
                         dr1, dth1, dphi1, dpr1, dpth1);

            float alpha = 0.0f;
            if (first_event_alpha_mode_f(
                    prev_theta, theta, dth0, dth1, step_used, M_PI_2_F,
                    intersection_mode, alpha)) {
                const float r_hit = interp_scalar_mode_f(
                    prev_r, r, dr0, dr1, step_used, alpha, intersection_mode);
                if (r_hit >= r_isco && r_hit <= r_disk_out) {
                    const float red = robust_disk_redshift(r_hit, pt, pphi, M, a, Q, L);

                    res.r_hit = r_hit;
                    res.redshift = red;
                    res.phi_hit = interp_scalar_mode_f(
                        prev_phi, phi, dphi0, dphi1, step_used, alpha, intersection_mode);
                    best_alpha = alpha;
                    best_event = 1;
                }
            }
        }

        const float rh_cut = r_horizon * 1.03f;
        const bool horizon_cross = ((prev_r > rh_cut) && (r <= rh_cut)) || (r <= rh_cut);
        if (horizon_cross) {
            const float denom_h = prev_r - r;
            float alpha_h = (abs(denom_h) > 1e-8f) ? ((prev_r - rh_cut) / denom_h) : 0.0f;
            alpha_h = clamp(alpha_h, 0.0f, 1.0f);
            if (alpha_h < best_alpha) {
                best_alpha = alpha_h;
                best_event = 2;
            }
        }

        const bool escape_cross = ((prev_r < r_escape) && (r >= r_escape)) || (r >= r_escape);
        if (escape_cross) {
            const float denom_e = r - prev_r;
            float alpha_e = (abs(denom_e) > 1e-8f) ? ((r_escape - prev_r) / denom_e) : 1.0f;
            alpha_e = clamp(alpha_e, 0.0f, 1.0f);
            if (alpha_e < best_alpha) {
                best_alpha = alpha_e;
                best_event = 3;
            }
        }

        if (best_event == 1) {
            res.outcome = 1;
            return res;
        }
        if (best_event == 2) {
            res.outcome = 2;
            return res;
        }
        if (best_event == 3) {
            res.outcome = 0;
            res.theta_esc = prev_theta + best_alpha*(theta - prev_theta);
            res.phi_esc = prev_phi + best_alpha*(phi - prev_phi);
            return res;
        }

        prev_r = r;
        prev_theta = theta;
        prev_phi = phi;
        prev_pr = pr;
        prev_pth = pth;
    }

    return res;
}

static RayTraceResultBL trace_standard_ks_from_angles(float alpha, float beta,
                                                      float M, float a, float Q, float L,
                                                      float r_obs, float theta_obs, float phi_obs,
                                                      float r_horizon, float r_isco, float r_disk_out,
                                                      int intersection_mode,
                                                      int integrator_mode,
                                                      int max_steps,
                                                      float step_init,
                                                      float integrator_tol) {
    RayTraceResultBL res{};
    res.outcome = 2;
    res.r_hit = 0.0f;
    res.redshift = 1.0f;
    res.theta_esc = theta_obs;
    res.phi_esc = phi_obs;
    res.phi_hit = phi_obs;

    float gll[4][4];
    gLL_BL(r_obs, theta_obs, M, a, Q, L, gll);

    float gu[4][4];
    gUU(r_obs, theta_obs, M, a, Q, L, gu);

    const float ca = cos(alpha), sa = sin(alpha);
    const float cb = cos(beta),  sb = sin(beta);

    const float gtt   = gll[0][0];
    const float gtphi = gll[0][3];
    const float grr   = gll[1][1];
    const float gthth = gll[2][2];
    const float gphph = gll[3][3];

    const float sqrt_grr   = sqrt(abs(grr));
    const float sqrt_gthth = sqrt(abs(gthth));
    const float denom_phi  = gphph - (gtphi*gtphi)/gtt;

    float pUt   = 1.0f;
    float pUr   = -ca*cb / max(sqrt_grr, 1e-14f);
    float pUth  = -sb    / max(sqrt_gthth, 1e-14f);
    float pUphi = -sa*cb / sqrt(max(abs(gphph), 1e-14f));

    const bool tetrad_ok =
        (gtt < -1e-14f) &&
        isfinite(denom_phi) && (denom_phi > 1e-14f) &&
        isfinite(sqrt_grr)  && (sqrt_grr  > 1e-14f) &&
        isfinite(sqrt_gthth)&& (sqrt_gthth> 1e-14f);
    if (tetrad_ok) {
        const float ut     = 1.0f / sqrt(-gtt);
        const float ephi_p = 1.0f / sqrt(denom_phi);
        const float ephi_t = -gtphi/gtt * ephi_p;
        const float n_r  = -ca * cb;
        const float n_th = -sb;
        const float n_ph = -sa * cb;
        pUt   = ut + n_ph * ephi_t;
        pUr   = n_r  / sqrt_grr;
        pUth  = n_th / sqrt_gthth;
        pUphi = n_ph * ephi_p;
    }

    float pt   = gtt* pUt + gtphi*pUphi;
    float pr   = grr* pUr;
    float pth  = gthth*pUth;
    float pphi = gtphi*pUt + gphph*pUphi;

    const float A = gu[0][0];
    const float B = 2.0f * gu[0][3] * pphi;
    const float C = gu[1][1]*pr*pr + gu[2][2]*pth*pth + gu[3][3]*pphi*pphi;
    const float disc = B*B - 4.0f*A*C;
    if (disc >= 0.0f && abs(A) > 1e-15f) {
        const float sq  = sqrt(disc);
        const float pt1 = (-B - sq)/(2.0f*A);
        const float pt2 = (-B + sq)/(2.0f*A);
        pt = (pt1 < 0.0f) ? pt1 : pt2;
        if (pt > 0.0f) pt = min(pt1, pt2);
    }

    float X, Y, Z;
    BL_to_KS_spatial(r_obs, theta_obs, phi_obs, a, X, Y, Z);
    float pX, pY, pZ;
    bool ks_ok = BL_covector_to_KS(r_obs, theta_obs, phi_obs, a, pr, pth, pphi, pX, pY, pZ);
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
    if (!ks_ok) return res;

    float dlam_ks = max(step_init, ADAPT_H_MIN);
    float prevX = X, prevY = Y, prevZ = Z;
    float prevPX = pX, prevPY = pY, prevPZ = pZ;
    float prev_r_ks = r_obs;
    const float rh_cut = r_horizon * 1.03f;
    const float r_escape = r_obs * 1.05f;

    const int iter_cap = max(max_steps, 1);
    for (int iter = 0; iter < iter_cap; ++iter) {
        const float step_used_ks = dlam_ks;
        if (!adaptive_step_ks(X, Y, Z, pX, pY, pZ, dlam_ks, pT, M, a, Q,
                              integrator_mode, integrator_tol)) {
            continue;
        }

        if (!(isfinite(X) && isfinite(Y) && isfinite(Z) &&
              isfinite(pX) && isfinite(pY) && isfinite(pZ))) {
            return res;
        }

        const float r_now = r_KS(X, Y, Z, a);
        float best_alpha = 2.0f;
        int best_event = 0; // 0 none, 1 disk, 2 horizon, 3 escape

        {
            const bool maybe_eq_ks = sign_change_f(prevZ, Z) ||
                                     (intersection_mode != 0 && min(abs(prevZ), abs(Z)) < 0.35f);
            float alpha_evt = 0.0f;
            bool hit_event = false;

            float dX0=0.0f, dY0=0.0f, dZ0=0.0f, dpX0=0.0f, dpY0=0.0f, dpZ0=0.0f;
            float dX1=0.0f, dY1=0.0f, dZ1=0.0f, dpX1=0.0f, dpY1=0.0f, dpZ1=0.0f;
            if (maybe_eq_ks) {
                geodesic_rhs_KS(prevX, prevY, prevZ, pT, prevPX, prevPY, prevPZ,
                                M, a, Q, dX0,dY0,dZ0,dpX0,dpY0,dpZ0);
                geodesic_rhs_KS(X, Y, Z, pT, pX, pY, pZ,
                                M, a, Q, dX1,dY1,dZ1,dpX1,dpY1,dpZ1);
                hit_event = first_event_alpha_mode_f(
                    prevZ, Z, dZ0, dZ1, step_used_ks, 0.0f,
                    intersection_mode, alpha_evt);
            }

            if (hit_event) {
                const float Xh = interp_scalar_mode_f(prevX, X, dX0, dX1, step_used_ks, alpha_evt, intersection_mode);
                const float Yh = interp_scalar_mode_f(prevY, Y, dY0, dY1, step_used_ks, alpha_evt, intersection_mode);
                const float Zh = interp_scalar_mode_f(prevZ, Z, dZ0, dZ1, step_used_ks, alpha_evt, intersection_mode);
                const float pXh = interp_scalar_mode_f(prevPX, pX, dpX0, dpX1, step_used_ks, alpha_evt, intersection_mode);
                const float pYh = interp_scalar_mode_f(prevPY, pY, dpY0, dpY1, step_used_ks, alpha_evt, intersection_mode);
                const float pZh = interp_scalar_mode_f(prevPZ, pZ, dpZ0, dpZ1, step_used_ks, alpha_evt, intersection_mode);
                float r_hit, th_hit, ph_hit;
                KS_to_BL_spatial(Xh, Yh, Zh, a, r_hit, th_hit, ph_hit);
                if (r_hit >= r_isco && r_hit <= r_disk_out) {
                    float pr_hit, pth_hit, pphi_hit;
                    KS_covector_to_BL(r_hit, th_hit, ph_hit, a, pXh, pYh, pZh,
                                      pr_hit, pth_hit, pphi_hit);
                    const float red = robust_disk_redshift(r_hit, pT, pphi_hit, M, a, Q, L);

                    res.r_hit = r_hit;
                    res.redshift = red;
                    res.phi_hit = ph_hit;
                    best_alpha = alpha_evt;
                    best_event = 1;
                }
            }
        }

        const bool horizon_cross = ((prev_r_ks > rh_cut) && (r_now <= rh_cut)) || (r_now <= rh_cut);
        if (horizon_cross) {
            const float denom_h = prev_r_ks - r_now;
            float alpha_h = (abs(denom_h) > 1e-8f) ? ((prev_r_ks - rh_cut) / denom_h) : 0.0f;
            alpha_h = clamp(alpha_h, 0.0f, 1.0f);
            if (alpha_h < best_alpha) {
                best_alpha = alpha_h;
                best_event = 2;
            }
        }

        const bool escape_cross = ((prev_r_ks < r_escape) && (r_now >= r_escape)) || (r_now >= r_escape);
        if (escape_cross) {
            const float denom_e = r_now - prev_r_ks;
            float alpha_e = (abs(denom_e) > 1e-8f) ? ((r_escape - prev_r_ks) / denom_e) : 1.0f;
            alpha_e = clamp(alpha_e, 0.0f, 1.0f);
            if (alpha_e < best_alpha) {
                best_alpha = alpha_e;
                best_event = 3;
            }
        }

        if (best_event == 1) {
            res.outcome = 1;
            return res;
        }
        if (best_event == 2) {
            res.outcome = 2;
            return res;
        }
        if (best_event == 3) {
            const float X_esc = prevX + best_alpha*(X - prevX);
            const float Y_esc = prevY + best_alpha*(Y - prevY);
            const float Z_esc = prevZ + best_alpha*(Z - prevZ);
            float r_esc, th_esc, ph_esc;
            KS_to_BL_spatial(X_esc, Y_esc, Z_esc, a, r_esc, th_esc, ph_esc);
            res.outcome = 0;
            res.theta_esc = th_esc;
            res.phi_esc = ph_esc;
            return res;
        }

        prevX = X; prevY = Y; prevZ = Z;
        prevPX = pX; prevPY = pY; prevPZ = pZ;
        prev_r_ks = r_now;
    }

    return res;
}

static RayTraceResultBL trace_standard_chart_from_angles(float alpha, float beta,
                                                         float M, float a, float Q, float L,
                                                         float r_obs, float theta_obs, float phi_obs,
                                                         float r_horizon, float r_isco, float r_disk_out,
                                                         int chart, int intersection_mode,
                                                         int integrator_mode,
                                                         int max_steps,
                                                         float step_init,
                                                         float integrator_tol) {
    if (chart == 1 && abs(L) <= 1e-8f) {
        return trace_standard_ks_from_angles(alpha, beta, M, a, Q, L,
                                             r_obs, theta_obs, phi_obs,
                                             r_horizon, r_isco, r_disk_out,
                                             intersection_mode, integrator_mode,
                                             max_steps, step_init, integrator_tol);
    }
    return trace_standard_bl_from_angles(alpha, beta, M, a, Q, L,
                                         r_obs, theta_obs, phi_obs,
                                         r_horizon, r_isco, r_disk_out,
                                         intersection_mode, integrator_mode,
                                         max_steps, step_init, integrator_tol);
}

// ── Main compute implementation (shared by all entry kernels) ─
static inline void trace_pixel_impl(
    device uint32_t*   output,
    KNdSParams         kp,
    CameraParams       cp,
    RenderParams       rp,
    texture2d<float, access::sample> bg_tex,
    sampler            bg_samp,
    uint2              gid)
{
    const int px_local = (int)gid.x;
    const int py_local = (int)gid.y;
    const int width = (int)rp.width;
    const int height = (int)rp.height;
    if (px_local >= (int)rp.tile_w || py_local >= (int)rp.tile_h) return;
    const int px = px_local + (int)rp.x_offset;
    const int py = py_local + (int)rp.y_offset;
    if (px >= width || py >= height) return;

    const float M = kp.M, a = kp.a, Q = kp.Q, L = kp.Lambda;

    // ── Pixel → (α, β) ───────────────────────────────────────
    const int span = max(width - 1, 1);
    const float pxf = float(px) + cp.pixel_offset_x;
    const float pyf = float(py) + cp.pixel_offset_y;
    const float alpha = cp.fov_h*(pxf - 0.5f*(width-1))  / float(span);
    const float beta  = cp.fov_h*(0.5f*(height-1) - pyf) / float(span);

    // ── Ray-bundle mode (GPU native, standard solver; BL and KS) ─
    const bool bundle_chart_ok = (cp.chart == 0) || (cp.chart == 1 && abs(L) <= 1e-8f);
    if (cp.use_bundles != 0 && cp.solver_mode == 0 && bundle_chart_ok) {
        uint32_t colour = 0xFF000000u;
        const RayTraceResultBL c = trace_standard_chart_from_angles(
            alpha, beta, M, a, Q, L,
            cp.r_obs, cp.theta_obs, cp.phi_obs,
            kp.r_horizon, kp.r_isco, kp.r_disk_out,
            cp.chart, cp.intersection_mode, cp.integrator_mode,
            cp.max_steps, cp.step_init, cp.integrator_tol);

        if (c.outcome == 0) {
            const float4 bgc = clamp(sample_background(bg_tex, bg_samp, c.theta_esc, c.phi_esc), 0.0f, 1.0f);
            colour = pack_abgr(bgc.r, bgc.g, bgc.b);
            output[py * width + px] = colour;
            return;
        }

        if (c.outcome == 1) {
            const bool anti_fireflies = (cp.anti_fireflies != 0);
            float shade_r_hit = c.r_hit;
            float shade_redshift = c.redshift;
            float magnif = 1.0f;
            const float eps = cp.fov_h / float(max(cp.width, 1)) * 0.5f;
            if (eps > 1e-8f) {
                const RayTraceResultBL ap = trace_standard_chart_from_angles(
                    alpha + eps, beta, M, a, Q, L,
                    cp.r_obs, cp.theta_obs, cp.phi_obs,
                    kp.r_horizon, kp.r_isco, kp.r_disk_out,
                    cp.chart, cp.intersection_mode, cp.integrator_mode,
                    cp.max_steps, cp.step_init, cp.integrator_tol);
                const RayTraceResultBL am = trace_standard_chart_from_angles(
                    alpha - eps, beta, M, a, Q, L,
                    cp.r_obs, cp.theta_obs, cp.phi_obs,
                    kp.r_horizon, kp.r_isco, kp.r_disk_out,
                    cp.chart, cp.intersection_mode, cp.integrator_mode,
                    cp.max_steps, cp.step_init, cp.integrator_tol);
                const RayTraceResultBL bp = trace_standard_chart_from_angles(
                    alpha, beta + eps, M, a, Q, L,
                    cp.r_obs, cp.theta_obs, cp.phi_obs,
                    kp.r_horizon, kp.r_isco, kp.r_disk_out,
                    cp.chart, cp.intersection_mode, cp.integrator_mode,
                    cp.max_steps, cp.step_init, cp.integrator_tol);
                const RayTraceResultBL bm = trace_standard_chart_from_angles(
                    alpha, beta - eps, M, a, Q, L,
                    cp.r_obs, cp.theta_obs, cp.phi_obs,
                    kp.r_horizon, kp.r_isco, kp.r_disk_out,
                    cp.chart, cp.intersection_mode, cp.integrator_mode,
                    cp.max_steps, cp.step_init, cp.integrator_tol);

                if (ap.outcome == 1 && am.outcome == 1 && bp.outcome == 1 && bm.outcome == 1) {
                    const float dr_da = (ap.r_hit - am.r_hit) / (2.0f * eps);
                    const float dr_db = (bp.r_hit - bm.r_hit) / (2.0f * eps);
                    const float dphi_da = wrap_delta_phi(ap.phi_hit - am.phi_hit) / (2.0f * eps);
                    const float dphi_db = wrap_delta_phi(bp.phi_hit - bm.phi_hit) / (2.0f * eps);
                    const float jac_center = abs(dr_da * dphi_db - dr_db * dphi_da);

                    if (!anti_fireflies) {
                        magnif = max(jac_center, 1e-12f);
                    } else {
                        const float dr_da_pp = (ap.r_hit - c.r_hit) / eps;
                        const float dr_db_pp = (bp.r_hit - c.r_hit) / eps;
                        const float dphi_da_pp = wrap_delta_phi(ap.phi_hit - c.phi_hit) / eps;
                        const float dphi_db_pp = wrap_delta_phi(bp.phi_hit - c.phi_hit) / eps;
                        const float jac_pp = abs(dr_da_pp * dphi_db_pp - dr_db_pp * dphi_da_pp);

                        const float dr_da_mm = (c.r_hit - am.r_hit) / eps;
                        const float dr_db_mm = (c.r_hit - bm.r_hit) / eps;
                        const float dphi_da_mm = wrap_delta_phi(c.phi_hit - am.phi_hit) / eps;
                        const float dphi_db_mm = wrap_delta_phi(c.phi_hit - bm.phi_hit) / eps;
                        const float jac_mm = abs(dr_da_mm * dphi_db_mm - dr_db_mm * dphi_da_mm);

                        const float jac_robust = median3v(jac_center, jac_pp, jac_mm);
                        const float r_med = median5v(c.r_hit, ap.r_hit, am.r_hit, bp.r_hit, bm.r_hit);
                        const float red_med = median5v(c.redshift, ap.redshift, am.redshift,
                                                       bp.redshift, bm.redshift);
                        const float r_dev = max(max(abs(ap.r_hit - c.r_hit), abs(am.r_hit - c.r_hit)),
                                                max(abs(bp.r_hit - c.r_hit), abs(bm.r_hit - c.r_hit)));
                        const float red_dev = max(max(abs(ap.redshift - c.redshift), abs(am.redshift - c.redshift)),
                                                  max(abs(bp.redshift - c.redshift), abs(bm.redshift - c.redshift)));
                        const float phi_dev = max(max(abs(wrap_delta_phi(ap.phi_hit - c.phi_hit)),
                                                      abs(wrap_delta_phi(am.phi_hit - c.phi_hit))),
                                                  max(abs(wrap_delta_phi(bp.phi_hit - c.phi_hit)),
                                                      abs(wrap_delta_phi(bm.phi_hit - c.phi_hit))));
                        const float r_tol = max(0.35f, 0.06f * max(c.r_hit, 1.0f));
                        const float red_tol = max(0.35f, 0.35f * max(abs(red_med), 1.0f));
                        const bool unstable =
                            !isfinite(jac_robust) || !isfinite(r_med) || !isfinite(red_med) ||
                            (jac_robust <= 1e-12f) ||
                            (r_dev > r_tol) ||
                            (red_dev > red_tol) ||
                            (phi_dev > 1.20f);

                        magnif = max(jac_robust, 1e-12f);
                        if (unstable) {
                            // At branch discontinuities we damp spikes instead of replacing hit data.
                            magnif = max(magnif, 2.0f);
                        }
                        if (isfinite(red_med)) {
                            const float red_cap = red_med + red_tol;
                            shade_redshift = min(shade_redshift, red_cap);
                        }
                    }
                }
            }

            colour = disk_color_abgr(shade_r_hit, shade_redshift, magnif, M, kp.r_isco);
            output[py * width + px] = colour;
            return;
        }

        output[py * width + px] = colour;
        return;
    }

    // ── Initial conditions (approx. flat at large r) ──────────
    const float r0  = cp.r_obs;
    const float th0 = cp.theta_obs;

    float gll[4][4];
    gLL_BL(r0, th0, M, a, Q, L, gll);

    float gu[4][4];
    gUU(r0, th0, M, a, Q, L, gu);

    const float ca = cos(alpha), sa = sin(alpha);
    const float cb = cos(beta),  sb = sin(beta);

    const float gtt   = gll[0][0];
    const float gtphi = gll[0][3];
    const float grr   = gll[1][1];
    const float gthth = gll[2][2];
    const float gphph = gll[3][3];

    const float sqrt_grr   = sqrt(abs(grr));
    const float sqrt_gthth = sqrt(abs(gthth));
    const float denom_phi  = gphph - (gtphi*gtphi)/gtt;

    float pUt   = 1.0f;
    float pUr   = -ca*cb / max(sqrt_grr, 1e-14f);
    float pUth  = -sb    / max(sqrt_gthth, 1e-14f);
    float pUphi = -sa*cb / sqrt(max(abs(gphph), 1e-14f));

    const bool tetrad_ok =
        (gtt < -1e-14f) &&
        isfinite(denom_phi) && (denom_phi > 1e-14f) &&
        isfinite(sqrt_grr)  && (sqrt_grr  > 1e-14f) &&
        isfinite(sqrt_gthth)&& (sqrt_gthth> 1e-14f);
    if (tetrad_ok) {
        const float ut     = 1.0f / sqrt(-gtt);
        const float ephi_p = 1.0f / sqrt(denom_phi);
        const float ephi_t = -gtphi/gtt * ephi_p;
        const float n_r  = -ca * cb;
        const float n_th = -sb;
        const float n_ph = -sa * cb;
        pUt   = ut + n_ph * ephi_t;
        pUr   = n_r  / sqrt_grr;
        pUth  = n_th / sqrt_gthth;
        pUphi = n_ph * ephi_p;
    }

    float pt   = gtt* pUt + gtphi*pUphi;
    float pr   = grr* pUr;
    float pth  = gthth*pUth;
    float pphi = gtphi*pUt + gphph*pUphi;

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
    float dlam = max(cp.step_init, ADAPT_H_MIN);
    float prev_r = r0;
    float prev_theta = th0;
    float prev_phi = phi;
    float prev_cos = cos(th0);

    uint32_t colour = 0xFF000000u;  // black (ABGR)
    const bool can_separable_kerr = (abs(Q) <= 1e-8f) && (abs(L) <= 1e-8f);
    const bool can_bl_separable = (cp.chart == 0) && can_separable_kerr;

    // ── Elliptic-closed path (Kerr, per-ray fallback) ────────
    if (cp.solver_mode == 2 && !can_separable_kerr && cp.elliptic_fallback_black != 0) {
        output[py * width + px] = colour;
        return;
    }
    if (cp.solver_mode == 2 && can_separable_kerr) {
        SeparableConsts sc{};
        sc.M = M;
        sc.a = a;
        sc.E = -pt;
        sc.Lz = pphi;
        const float st = sin(theta);
        const float ct = cos(theta);
        const float st2 = max(st*st, 1e-14f);
        sc.Qc = pth*pth + ct*ct * (sc.Lz*sc.Lz/st2 - sc.a*sc.a*sc.E*sc.E);

        float dr0=0.0f, dth0=0.0f, dphi0=0.0f, dpr0=0.0f, dpth0=0.0f;
        geodesic_rhs(r, theta, pr, pth, pt, pphi, M, a, Q, L, dr0, dth0, dphi0, dpr0, dpth0);

        float r_hit_ell = 0.0f;
        const int eout = elliptic_closed_first_hit(sc, r, theta, dr0, dth0,
                                                   kp.r_horizon, kp.r_isco, kp.r_disk_out,
                                                   cp.r_obs * 1.05f, r_hit_ell);
        if (eout == 1) {
            // horizon: keep black
            output[py * width + px] = colour;
            return;
        }
        if (eout == 2) {
            // Horizon-precedence safety check against chart-native standard tracer.
            const RayTraceResultBL verify = trace_standard_chart_from_angles(
                alpha, beta, M, a, Q, L,
                cp.r_obs, cp.theta_obs, cp.phi_obs,
                kp.r_horizon, kp.r_isco, kp.r_disk_out,
                cp.chart, cp.intersection_mode, cp.integrator_mode,
                cp.max_steps, cp.step_init, cp.integrator_tol);
            if (verify.outcome == 2) {
                output[py * width + px] = colour;
                return;
            }

            const float red = robust_disk_redshift(r_hit_ell, pt, pphi, M, a, Q, L);

            const float T = 6500.0f * sqrt(6.0f*M/r_hit_ell) * clamp(red, 0.2f, 5.0f);
            float I = page_thorne_norm(r_hit_ell, kp.r_isco);
            I *= doppler_intensity_scale(red);
            float4 bb = blackbody_rgb(T);
            float4 c = float4(tonemap_ch(bb.r * I, 1.0f, 2.2f),
                              tonemap_ch(bb.g * I, 1.0f, 2.2f),
                              tonemap_ch(bb.b * I, 1.0f, 2.2f),
                              1.0f);
            colour = (0xFFu << 24)
                   | (uint32_t(c.b*255.0f) << 16)
                   | (uint32_t(c.g*255.0f) << 8)
                   |  uint32_t(c.r*255.0f);
            output[py * width + px] = colour;
            return;
        }

        // Inconclusive for this ray: optionally force black, otherwise continue on
        // the chart-native standard integrator path.
        if (cp.elliptic_fallback_black != 0) {
            output[py * width + px] = colour;
            return;
        }
    }

    // ── Separable Kerr semi-analytic path (BL only) ──────────
    const bool want_semi_analytic =
        (cp.solver_mode == 1) &&
        can_bl_separable;
    if (want_semi_analytic) {
        SeparableConsts sc{};
        sc.M = M;
        sc.a = a;
        sc.E = -pt;
        sc.Lz = pphi;
        const float st = sin(theta);
        const float ct = cos(theta);
        const float st2 = max(st*st, 1e-14f);
        sc.Qc = pth*pth + ct*ct * (sc.Lz*sc.Lz/st2 - sc.a*sc.a*sc.E*sc.E);

        bool semi_ok = isfinite(sc.E) && isfinite(sc.Lz) && isfinite(sc.Qc) && (sc.E > 0.0f);
        if (semi_ok) {
            float dr0=0.0f, dth0=0.0f, dphi0=0.0f, dpr0=0.0f, dpth0=0.0f;
            geodesic_rhs(r, theta, pr, pth, pt, pphi, M, a, Q, L, dr0, dth0, dphi0, dpr0, dpth0);

            SeparableState s{
                r, theta, phi,
                (dr0 >= 0.0f) ? 1 : -1,
                (dth0 >= 0.0f) ? 1 : -1
            };

            float h = max(cp.step_init, ADAPT_H_MIN);
            float prev_r_sep = s.r;
            float prev_theta_sep = s.theta;
            float prev_phi_sep = s.phi;
            float prev_Rpot = kerr_sep_R(sc, s.r);
            float prev_Thpot = kerr_sep_Theta(sc, s.theta);
            constexpr float R_turn_eps = 1e-7f;
            constexpr float Th_turn_eps = 1e-9f;
            bool done = false;

            const int semi_iter_cap = max(cp.max_steps, 1);
            for (int iter = 0; iter < semi_iter_cap; ++iter) {
                SeparableState s_prev = s;
                const float prev_Rpot_step = prev_Rpot;
                const float prev_Thpot_step = prev_Thpot;
                float step_used = h;
                int rejects = 0;
                while (true) {
                    step_used = h;
                    if (rk4_adaptive_separable_kerr(sc, s, h, cp.integrator_tol)) break;
                    if (!isfinite(h) || ++rejects > 64) {
                        semi_ok = false;
                        break;
                    }
                }
                if (!semi_ok) break;

                // Keep theta in [0, pi] and flip polar branch at reflections.
                if (s.theta < 0.0f) {
                    s.theta = -s.theta;
                    s.sgn_th *= -1;
                } else if (s.theta > 3.14159265358979323846f) {
                    s.theta = 6.28318530717958647692f - s.theta;
                    s.sgn_th *= -1;
                }

                const float Rnow = kerr_sep_R(sc, s.r);
                const float Thnow = kerr_sep_Theta(sc, s.theta);
                if (Rnow <= R_turn_eps && prev_Rpot_step > R_turn_eps) s.sgn_r *= -1;
                if (Thnow <= Th_turn_eps && prev_Thpot_step > Th_turn_eps) s.sgn_th *= -1;

                if (s.r < kp.r_horizon * 1.03f) {
                    done = true; // black
                    break;
                }

                if (s.r > cp.r_obs * 1.05f) {
                    const float r_escape = cp.r_obs * 1.05f;
                    const float denom = s.r - prev_r_sep;
                    float w = (abs(denom) > 1e-8f) ? ((r_escape - prev_r_sep) / denom) : 1.0f;
                    w = clamp(w, 0.0f, 1.0f);
                    const float th_esc = prev_theta_sep + w*(s.theta - prev_theta_sep);
                    const float ph_esc = prev_phi_sep + w*(s.phi - prev_phi_sep);
                    float4 bgc = clamp(sample_background(bg_tex, bg_samp, th_esc, ph_esc), 0.0f, 1.0f);
                    colour = (0xFFu << 24)
                           | (uint32_t(bgc.b*255.0f) << 16)
                           | (uint32_t(bgc.g*255.0f) << 8)
                           |  uint32_t(bgc.r*255.0f);
                    done = true;
                    break;
                }

                const float q0 = s_prev.theta - M_PI_2_F;
                const float q1 = s.theta - M_PI_2_F;
                const bool maybe_equator_sep = sign_change_f(q0, q1) ||
                                               (min(abs(q0), abs(q1)) < 0.35f);
                if (maybe_equator_sep) {
                    float dr_prev=0.0f, dth_prev=0.0f, dphi_prev=0.0f;
                    float dr_now=0.0f, dth_now=0.0f, dphi_now=0.0f;
                    semi_ok = kerr_sep_rhs(sc, s_prev, dr_prev, dth_prev, dphi_prev) &&
                              kerr_sep_rhs(sc, s,      dr_now,  dth_now,  dphi_now);
                    if (!semi_ok) break;

                    float alpha = 0.0f;
                    if (first_event_alpha_hermite_f(
                            s_prev.theta, s.theta, dth_prev, dth_now, step_used, M_PI_2_F,
                            alpha, 8, 7)) {
                        const float r_hit = hermite_interp_f(
                            s_prev.r, s.r, dr_prev, dr_now, step_used, alpha);
                        if (r_hit >= kp.r_isco && r_hit <= kp.r_disk_out) {
                            const float red = robust_disk_redshift(r_hit, pt, pphi, M, a, Q, L);

                            const float T = 6500.0f * sqrt(6.0f*M/r_hit) * clamp(red, 0.2f, 5.0f);
                            float I = page_thorne_norm(r_hit, kp.r_isco);
                            I *= doppler_intensity_scale(red);
                            float4 bb = blackbody_rgb(T);
                            float4 c = float4(tonemap_ch(bb.r * I, 1.0f, 2.2f),
                                              tonemap_ch(bb.g * I, 1.0f, 2.2f),
                                              tonemap_ch(bb.b * I, 1.0f, 2.2f),
                                              1.0f);
                            colour = (0xFFu << 24)
                                   | (uint32_t(c.b*255.0f) << 16)
                                   | (uint32_t(c.g*255.0f) << 8)
                                   |  uint32_t(c.r*255.0f);
                            done = true;
                            break;
                        }
                    }
                }

                prev_r_sep = s.r;
                prev_theta_sep = s.theta;
                prev_phi_sep = s.phi;
                prev_Rpot = Rnow;
                prev_Thpot = Thnow;
            }

            if (semi_ok && done) {
                output[py * width + px] = colour;
                return;
            }
        }
    }

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
            float dlam_ks = max(cp.step_init, ADAPT_H_MIN);
            float prevX = X, prevY = Y, prevZ = Z;
            float prevPX = pX, prevPY = pY, prevPZ = pZ;
            float prev_r_ks = r0;

            const int ks_iter_cap = max(cp.max_steps, 1);
            for (int iter = 0; iter < ks_iter_cap; ++iter) {
                const float step_used_ks = dlam_ks;
                if (!adaptive_step_ks(X, Y, Z, pX, pY, pZ, dlam_ks, pT, M, a, Q,
                                      cp.integrator_mode, cp.integrator_tol)) {
                    continue;
                }

                if (!(isfinite(X) && isfinite(Y) && isfinite(Z) &&
                      isfinite(pX) && isfinite(pY) && isfinite(pZ))) {
                    break;
                }

                const float r_now = r_KS(X, Y, Z, a);
                const float rh_cut = kp.r_horizon * 1.03f;
                const float r_escape = cp.r_obs * 1.05f;
                float best_alpha = 2.0f;
                int best_event = 0; // 0 none, 1 disk, 2 horizon, 3 escape

                {
                    const bool maybe_eq_ks = sign_change_f(prevZ, Z) ||
                                             (cp.intersection_mode != 0 && min(abs(prevZ), abs(Z)) < 0.35f);
                    float alpha = 0.0f;
                    bool hit_event = false;

                    float dX0=0.0f, dY0=0.0f, dZ0=0.0f, dpX0=0.0f, dpY0=0.0f, dpZ0=0.0f;
                    float dX1=0.0f, dY1=0.0f, dZ1=0.0f, dpX1=0.0f, dpY1=0.0f, dpZ1=0.0f;
                    if (maybe_eq_ks) {
                        geodesic_rhs_KS(prevX, prevY, prevZ, pT, prevPX, prevPY, prevPZ,
                                        M, a, Q, dX0,dY0,dZ0,dpX0,dpY0,dpZ0);
                        geodesic_rhs_KS(X, Y, Z, pT, pX, pY, pZ,
                                        M, a, Q, dX1,dY1,dZ1,dpX1,dpY1,dpZ1);
                        hit_event = first_event_alpha_mode_f(
                            prevZ, Z, dZ0, dZ1, step_used_ks, 0.0f,
                            cp.intersection_mode, alpha);
                    }

                    if (hit_event) {
                        const float Xh = interp_scalar_mode_f(prevX, X, dX0, dX1, step_used_ks, alpha, cp.intersection_mode);
                        const float Yh = interp_scalar_mode_f(prevY, Y, dY0, dY1, step_used_ks, alpha, cp.intersection_mode);
                        const float Zh = interp_scalar_mode_f(prevZ, Z, dZ0, dZ1, step_used_ks, alpha, cp.intersection_mode);
                        const float pXh = interp_scalar_mode_f(prevPX, pX, dpX0, dpX1, step_used_ks, alpha, cp.intersection_mode);
                        const float pYh = interp_scalar_mode_f(prevPY, pY, dpY0, dpY1, step_used_ks, alpha, cp.intersection_mode);
                        const float pZh = interp_scalar_mode_f(prevPZ, pZ, dpZ0, dpZ1, step_used_ks, alpha, cp.intersection_mode);
                        float r_hit, th_hit, ph_hit;
                        KS_to_BL_spatial(Xh, Yh, Zh, a, r_hit, th_hit, ph_hit);
                        if (r_hit >= kp.r_isco && r_hit <= kp.r_disk_out) {
                            float pr_hit, pth_hit, pphi_hit;
                            KS_covector_to_BL(r_hit, th_hit, ph_hit, a, pXh, pYh, pZh,
                                              pr_hit, pth_hit, pphi_hit);
                            const float red = robust_disk_redshift(r_hit, pT, pphi_hit, M, a, Q, L);

                            const float T = 6500.0f * sqrt(6.0f*M/r_hit) * clamp(red, 0.2f, 5.0f);
                            float I = page_thorne_norm(r_hit, kp.r_isco);
                            I *= doppler_intensity_scale(red);
                            float4 bb = blackbody_rgb(T);
                            float4 c = float4(tonemap_ch(bb.r * I, 1.0f, 2.2f),
                                              tonemap_ch(bb.g * I, 1.0f, 2.2f),
                                              tonemap_ch(bb.b * I, 1.0f, 2.2f),
                                              1.0f);
                            colour = (0xFFu << 24)
                                   | (uint32_t(c.b*255.0f) << 16)
                                   | (uint32_t(c.g*255.0f) << 8)
                                   |  uint32_t(c.r*255.0f);
                            best_alpha = alpha;
                            best_event = 1;
                        }
                    }
                }

                const bool horizon_cross = ((prev_r_ks > rh_cut) && (r_now <= rh_cut)) || (r_now <= rh_cut);
                if (horizon_cross) {
                    const float denom_h = prev_r_ks - r_now;
                    float alpha_h = (abs(denom_h) > 1e-8f) ? ((prev_r_ks - rh_cut) / denom_h) : 0.0f;
                    alpha_h = clamp(alpha_h, 0.0f, 1.0f);
                    if (alpha_h < best_alpha) {
                        best_alpha = alpha_h;
                        best_event = 2;
                    }
                }

                const bool escape_cross = ((prev_r_ks < r_escape) && (r_now >= r_escape)) || (r_now >= r_escape);
                if (escape_cross) {
                    const float denom_e = r_now - prev_r_ks;
                    float alpha_e = (abs(denom_e) > 1e-8f) ? ((r_escape - prev_r_ks) / denom_e) : 1.0f;
                    alpha_e = clamp(alpha_e, 0.0f, 1.0f);
                    if (alpha_e < best_alpha) {
                        best_alpha = alpha_e;
                        best_event = 3;
                    }
                }

                if (best_event == 1) break;
                if (best_event == 2) break;
                if (best_event == 3) {
                    const float X_esc = prevX + best_alpha*(X - prevX);
                    const float Y_esc = prevY + best_alpha*(Y - prevY);
                    const float Z_esc = prevZ + best_alpha*(Z - prevZ);
                    float r_esc, th_esc, ph_esc;
                    KS_to_BL_spatial(X_esc, Y_esc, Z_esc, a, r_esc, th_esc, ph_esc);
                    float4 bgc = clamp(sample_background(bg_tex, bg_samp, th_esc, ph_esc), 0.0f, 1.0f);
                    colour = (0xFFu << 24)
                           | (uint32_t(bgc.b*255.0f) << 16)
                           | (uint32_t(bgc.g*255.0f) << 8)
                           |  uint32_t(bgc.r*255.0f);
                    break;
                }

                prevX = X; prevY = Y; prevZ = Z;
                prevPX = pX; prevPY = pY; prevPZ = pZ;
                prev_r_ks = r_now;
            }

            output[py * width + px] = colour;
            return;
        }
    }

    const int std_iter_cap = max(cp.max_steps, 1);
    for (int iter = 0; iter < std_iter_cap; ++iter) {
        if (!adaptive_step_bl(r, theta, phi, pr, pth, dlam, pt, pphi, M, a, Q, L,
                              cp.integrator_mode, cp.integrator_tol)) {
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
            const float red = robust_disk_redshift(r_hit, pt, pphi, M, a, Q, L);

            // Match CPU disk_colour():
            //   T = 6500 * sqrt(6M/r) * clamp(red,0.2,5)
            //   I = page_thorne_norm(r) * red^4
            //   output = tonemap(blackbody(T) * I, exposure=1, gamma=2.2)
            const float T = 6500.0f * sqrt(6.0f*M/r_hit) * clamp(red, 0.2f, 5.0f);
            float I = page_thorne_norm(r_hit, kp.r_isco);
            I *= doppler_intensity_scale(red);
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

// ── Kernel entrypoints (selectable) ──────────────────────────
kernel void trace_pixel(
    device uint32_t*   output    [[buffer(0)]],
    constant KNdSParams& kp      [[buffer(1)]],
    constant CameraParams& cp    [[buffer(2)]],
    constant RenderParams& rp    [[buffer(3)]],
    texture2d<float, access::sample> bg_tex [[texture(0)]],
    sampler             bg_samp   [[sampler(0)]],
    uint2               gid       [[thread_position_in_grid]])
{
    trace_pixel_impl(output, kp, cp, rp, bg_tex, bg_samp, gid);
}

kernel void trace_pixel_single(
    device uint32_t*   output    [[buffer(0)]],
    constant KNdSParams& kp      [[buffer(1)]],
    constant CameraParams& cp    [[buffer(2)]],
    constant RenderParams& rp    [[buffer(3)]],
    texture2d<float, access::sample> bg_tex [[texture(0)]],
    sampler             bg_samp   [[sampler(0)]],
    uint2               gid       [[thread_position_in_grid]])
{
    CameraParams local = cp;
    local.use_bundles = 0;
    trace_pixel_impl(output, kp, local, rp, bg_tex, bg_samp, gid);
}

kernel void trace_pixel_bundle(
    device uint32_t*   output    [[buffer(0)]],
    constant KNdSParams& kp      [[buffer(1)]],
    constant CameraParams& cp    [[buffer(2)]],
    constant RenderParams& rp    [[buffer(3)]],
    texture2d<float, access::sample> bg_tex [[texture(0)]],
    sampler             bg_samp   [[sampler(0)]],
    uint2               gid       [[thread_position_in_grid]])
{
    CameraParams local = cp;
    local.use_bundles = 1;
    // Bundle GPU path uses the requested chart when supported.
    local.solver_mode = 0;
    if (local.chart == 1 && abs(kp.Lambda) > 1e-8f) local.chart = 0;
    trace_pixel_impl(output, kp, local, rp, bg_tex, bg_samp, gid);
}
