#include "falling_renderer.hpp"
// Define stb_image_write implementation only when not linking with main.cpp.
// In the test binary (compiled with -DFALLING_RENDERER_STANDALONE), this
// ensures stbi_write_png is available.  In the main build, main.cpp already
// defines STB_IMAGE_WRITE_IMPLEMENTATION before including stb_image_write.h.
#ifdef FALLING_RENDERER_STANDALONE
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image_write.h"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <vector>

// ── Photon initialization ─────────────────────────────────────────────────────
void init_photon_k(const CameraState& cs, const double e[4][4],
                   const KNdSMetric& bh,
                   int px, int py, int W, int H, double fov_h,
                   double k[4])
{
    const double fov_v = fov_h * double(H) / double(W);
    const double alpha = fov_h * (px - W*0.5) / double(W - 1);
    const double beta  = fov_v * (H*0.5 - py) / double(H - 1);

    // Local direction in tetrad frame
    const double nx = std::sin(beta)*std::cos(alpha);
    const double ny = std::sin(beta)*std::sin(alpha);
    const double nz = std::cos(beta);

    // k^μ = e[0]^μ + nx·e[1]^μ + ny·e[2]^μ + nz·e[3]^μ
    for (int mu=0;mu<4;++mu)
        k[mu] = e[0][mu] + nx*e[1][mu] + ny*e[2][mu] + nz*e[3][mu];

    // Enforce null: rescale k^T so g_μν k^μ k^ν = 0
    // g_TT (k^T)² + 2 Σ_{μ≠T} g_Tμ k^T k^μ + Σ_{μ,ν≠T} g_μν k^μ k^ν = 0
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);

    const double A = gLL[0][0];
    double B = 0.0, C = 0.0;
    for (int mu=1;mu<4;++mu) B += 2.0*gLL[0][mu]*k[mu];
    for (int mu=1;mu<4;++mu) for (int nu=1;nu<4;++nu) C += gLL[mu][nu]*k[mu]*k[nu];
    const double disc = B*B - 4.0*A*C;
    if (disc >= 0.0) {
        const double kT1 = (-B + std::sqrt(disc))/(2.0*A);
        const double kT2 = (-B - std::sqrt(disc))/(2.0*A);
        // Pick future-directed root (larger value for our sign convention)
        k[0] = (kT1 > kT2) ? kT1 : kT2;
    }
}

// ── RK4 step for photon geodesic ──────────────────────────────────────────────
static void photon_step(const KNdSMetric& bh, double x[4], double kv[4],
                         double dlam)
{
    auto deriv = [&](const double* xi, const double* ki,
                     double* dxdl, double* dkdl) {
        double Gamma[4][4][4];
        gpg_christoffel(bh, xi[1], xi[2], Gamma);
        for (int mu=0;mu<4;++mu) {
            dxdl[mu] = ki[mu];
            double acc=0.0;
            for (int al=0;al<4;++al) for (int be=0;be<4;++be)
                acc -= Gamma[mu][al][be]*ki[al]*ki[be];
            dkdl[mu] = acc;
        }
    };

    double dx1[4],dk1[4], dx2[4],dk2[4], dx3[4],dk3[4], dx4[4],dk4[4];
    double xt[4], kt[4];

    deriv(x, kv, dx1, dk1);
    for (int i=0;i<4;++i){xt[i]=x[i]+0.5*dlam*dx1[i]; kt[i]=kv[i]+0.5*dlam*dk1[i];}
    deriv(xt, kt, dx2, dk2);
    for (int i=0;i<4;++i){xt[i]=x[i]+0.5*dlam*dx2[i]; kt[i]=kv[i]+0.5*dlam*dk2[i];}
    deriv(xt, kt, dx3, dk3);
    for (int i=0;i<4;++i){xt[i]=x[i]+dlam*dx3[i]; kt[i]=kv[i]+dlam*dk3[i];}
    deriv(xt, kt, dx4, dk4);

    for (int i=0;i<4;++i) {
        x[i]  += (dlam/6.0)*(dx1[i]+2*dx2[i]+2*dx3[i]+dx4[i]);
        kv[i] += (dlam/6.0)*(dk1[i]+2*dk2[i]+2*dk3[i]+dk4[i]);
    }
}

// ── Backward null geodesic trace ──────────────────────────────────────────────
FallingGeoPixel trace_photon_gpg(const double x0[4], const double k0[4],
                                  const KNdSMetric& bh,
                                  const FallingParams& fp,
                                  int max_steps, double h0, double /*tol*/)
{
    double x[4], k[4];
    for (int i=0;i<4;++i){x[i]=x0[i]; k[i]=k0[i];}

    const double r_sing = fp.r_singularity;
    const double r_esc  = fp.r_escape;
    const double r_in   = (fp.r_disk_in < 0) ? bh.r_isco() : fp.r_disk_in;
    const double r_out  = fp.r_disk_out;

    FallingGeoPixel pix{};
    pix.r_min    = float(x[1]);
    pix.redshift = 1.0f;

    double prev_theta = x[2];
    double dlam = h0;
    const double r_h = bh.r_horizon();

    // prev_x tracks position before each step for interpolation
    double prev_x[4];
    for (int i=0;i<4;++i) prev_x[i] = x[i];

    for (int step=0; step<max_steps; ++step) {
        if (x[1] < double(pix.r_min)) pix.r_min = float(x[1]);

        // Escape
        if (x[1] >= r_esc) {
            pix.outcome   = 0;
            pix.r_hit     = float(x[1]);
            pix.theta_esc = float(std::fmod(std::abs(x[2]), M_PI));
            {
                double phi_wrapped = std::fmod(x[3], 2.0*M_PI);
                if (phi_wrapped < 0.0) phi_wrapped += 2.0*M_PI;
                pix.phi_esc = float(phi_wrapped);
            }
            return pix;
        }
        // Singularity
        if (x[1] < r_sing) {
            pix.outcome = 2;
            pix.r_hit   = float(x[1]);
            return pix;
        }

        // Disk crossing: sign change of (θ − π/2) while r_in ≤ r ≤ r_out
        double cur_theta = x[2];
        if (x[1] >= r_in && x[1] <= r_out) {
            double d_prev = prev_theta - M_PI/2.0;
            double d_cur  = cur_theta  - M_PI/2.0;
            if (d_prev * d_cur < 0.0) {
                // Linear interpolation between prev_x and x for the crossing point
                double t_frac = std::abs(d_prev) / (std::abs(d_prev) + std::abs(d_cur));
                pix.outcome = 1;
                pix.r_hit   = float(prev_x[1] + t_frac*(x[1]-prev_x[1]));
                pix.phi_hit = float(prev_x[3] + t_frac*(x[3]-prev_x[3]));

                // Compute approximate redshift g-factor
                {
                    double gLL_d[4][4];
                    gpg_covariant(bh, double(pix.r_hit), M_PI/2.0, gLL_d);
                    const double Omega = bh.keplerian_omega(double(pix.r_hit));
                    const double N2 = -(gLL_d[0][0]
                                      + 2.0*gLL_d[0][3]*Omega
                                      + gLL_d[3][3]*Omega*Omega);
                    if (N2 > 1e-20) {
                        const double ut_em  = 1.0 / std::sqrt(N2);
                        const double uph_em = Omega * ut_em;
                        double k_low[4] = {0.0};
                        for (int mu=0;mu<4;++mu)
                            for (int nu=0;nu<4;++nu)
                                k_low[mu] += gLL_d[mu][nu] * k[nu];
                        const double k_u_emit = k_low[0]*ut_em + k_low[3]*uph_em;
                        const double k_u_obs  = std::abs(k_low[0]);  // coordinate energy approx
                        if (std::abs(k_u_emit) > 1e-20)
                            pix.redshift = float(k_u_obs / std::abs(k_u_emit));
                    }
                }

                return pix;
            }
        }
        prev_theta = cur_theta;

        // Bidirectional adaptive step: fine near horizon, reset when far
        if (x[1] < r_h * 3.0)
            dlam = 0.005;
        else
            dlam = h0;

        // Save position before stepping
        for (int i=0;i<4;++i) prev_x[i] = x[i];

        photon_step(bh, x, k, dlam);
    }

    pix.outcome = 3;  // trapped
    return pix;
}

// ── render_falling_frame ──────────────────────────────────────────────────────
void render_falling_frame(int frame_idx, int total_frames,
                           const FallingParams& fp,
                           const CameraState& cs_at_frame,
                           const std::string& out_path)
{
    const int W = fp.width, H = fp.height;
    std::vector<uint8_t> rgb(W * H * 3, 0);

    // Build local tetrad and apply HorizonFlip roll
    double e[4][4];
    build_tetrad(cs_at_frame, fp.bh, e);
    const double psi = horizon_flip_psi(cs_at_frame.x[1], fp.bh.r_horizon());
    apply_roll(e, psi);

    // Pre-compute disk ISCO for Phase B shading
    const double r_isco_val = fp.bh.r_isco();
    const double r_in  = (fp.r_disk_in < 0.0) ? r_isco_val : fp.r_disk_in;
    const double r_out = fp.r_disk_out;
    (void)r_in;  // used indirectly via FallingParams in trace_photon_gpg

    auto t0 = std::chrono::steady_clock::now();

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic,16)
#endif
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            double k[4];
            init_photon_k(cs_at_frame, e, fp.bh,
                          px, py, W, H, fp.fov_h, k);

            FallingGeoPixel pix = trace_photon_gpg(
                cs_at_frame.x, k, fp.bh, fp, 50000, 0.05, 1e-7);

            uint8_t R = 0, G = 0, B = 0;

            if (pix.outcome == 0) {
                // Escaped — dark background (Phase A); Phase C will add HDRI skybox
                R = G = B = 30;

            } else if (pix.outcome == 1) {
                // Disk hit — Phase B shading
                const double r_h  = double(pix.r_hit);
                const double g_rs = double(pix.redshift);

                // Page-Thorne flux profile: f(r) ∝ (1 - sqrt(r_isco/r)) / r³
                double lum = 0.0;
                if (r_h > r_isco_val && r_h <= r_out) {
                    const double x_pt = std::sqrt(r_isco_val / r_h);
                    lum = (1.0 - x_pt) / (r_h * r_h * r_h);
                    // Normalize by approximate peak at 3*r_isco
                    const double r_peak = 3.0 * r_isco_val;
                    const double x_peak = std::sqrt(r_isco_val / r_peak);
                    const double lum_peak = (1.0 - x_peak) / (r_peak*r_peak*r_peak);
                    if (lum_peak > 1e-30) lum /= lum_peak;
                    // Apply redshift: I_obs = g^4 * I_emit
                    lum *= std::pow(std::max(g_rs, 0.0), 4.0);
                    lum  = std::min(lum * fp.disk_brightness, 1.0);
                }

                // Color: blueshift side (g>1) → white-yellow; redshift side → orange-red
                const float fl = float(lum);
                if (g_rs > 1.0) {
                    R = uint8_t(std::min(255.0f, 255.0f * fl));
                    G = uint8_t(std::min(255.0f, 210.0f * fl));
                    B = uint8_t(std::min(255.0f, 100.0f * fl));
                } else {
                    R = uint8_t(std::min(255.0f, 220.0f * fl));
                    G = uint8_t(std::min(255.0f, 100.0f * fl * float(g_rs)));
                    B = 0;
                }
                // outcome 2 (singularity), 3 (trapped) → remain black
            }

            const int idx = (py * W + px) * 3;
            rgb[idx + 0] = R;
            rgb[idx + 1] = G;
            rgb[idx + 2] = B;
        }

        // Progress (approximate — not thread-safe but visually fine)
        if (py % std::max(1, H / 20) == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            int pct = (py * 100) / H;
            std::printf("[frame %04d/%04d] %d%% %.1fs elapsed\r",
                        frame_idx + 1, total_frames, pct, elapsed);
            std::fflush(stdout);
        }
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - t0).count();
    std::printf("[frame %04d/%04d] 100%% %.1fs elapsed\n",
                frame_idx + 1, total_frames, elapsed);

    stbi_write_png(out_path.c_str(), W, H, 3, rgb.data(), W * 3);
}
