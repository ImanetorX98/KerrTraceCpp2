#include "falling_renderer.hpp"
#include <cmath>
#include <algorithm>
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

    for (int step=0; step<max_steps; ++step) {
        if (x[1] < double(pix.r_min)) pix.r_min = float(x[1]);

        // Escape
        if (x[1] >= r_esc) {
            pix.outcome   = 0;
            pix.r_hit     = float(x[1]);
            pix.theta_esc = float(std::fmod(std::abs(x[2]), M_PI));
            pix.phi_esc   = float(std::fmod(x[3] + 4.0*M_PI, 2.0*M_PI));
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
                pix.outcome = 1;
                // Linear interpolation for r_hit and phi_hit
                pix.r_hit   = float(x[1]);               // approx (last step r)
                pix.phi_hit = float(x[3]);
                return pix;
            }
        }
        prev_theta = cur_theta;

        // Reduce step size near horizon for accuracy
        if (x[1] < r_h * 3.0)
            dlam = std::min(dlam, 0.005);

        photon_step(bh, x, k, dlam);
    }

    pix.outcome = 3;  // trapped
    return pix;
}

// ── render_falling_frame stub — implemented in Task 6 ────────────────────────
void render_falling_frame(int /*frame_idx*/, int /*total_frames*/,
                           const FallingParams& /*fp*/,
                           const CameraState& /*cs_at_frame*/,
                           const std::string& /*out_path*/)
{
    // Implementation in Task 6
}
