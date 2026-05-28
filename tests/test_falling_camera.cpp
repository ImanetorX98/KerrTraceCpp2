#include "falling_camera.hpp"
#include "knds_metric.hpp"
#include <cmath>
#include <iostream>

static bool approx(double a, double b, double tol=1e-8) {
    return std::abs(a-b) <= tol;
}

bool test_gpg_metric_inverse() {
    KNdSMetric bh(1.0, 0.9, 0.0, 0.0);
    double gLL[4][4], gUU[4][4];
    gpg_covariant(bh, 10.0, M_PI/2, gLL);
    gpg_contravariant(bh, 10.0, M_PI/2, gUU);
    for (int mu=0; mu<4; ++mu)
        for (int nu=0; nu<4; ++nu) {
            double s=0.0;
            for (int a=0; a<4; ++a) s += gUU[mu][a]*gLL[a][nu];
            double ex = (mu==nu)?1.0:0.0;
            if (!approx(s, ex, 1e-7)) {
                std::cerr << "FAIL inverse ["<<mu<<"]["<<nu<<"] = "<<s<<"\n";
                return false;
            }
        }
    return true;
}

bool test_gpg_flat_limit() {
    KNdSMetric bh(1.0, 0.5, 0.0, 0.0);
    double gLL[4][4];
    // GPG g_Tr ~ sqrt(2M/r); need r >> 2M to be near flat.
    // At r=1e7: sqrt(2/1e7) ~ 4.5e-4 < 1e-3.
    gpg_covariant(bh, 1e7, M_PI/2, gLL);
    if (std::abs(gLL[0][1]) > 1e-3) {
        std::cerr << "FAIL flat limit g_Tr = " << gLL[0][1] << "\n";
        return false;
    }
    return true;
}

bool test_worldline_init_normalized() {
    KNdSMetric bh(1.0, 0.9, 0.0, 0.0);
    FallingParams fp; fp.bh = bh; fp.r_start = 20.0; fp.E = 1.0; fp.L = 0.0; fp.Qc = 0.0;
    CameraState cs = init_worldline(fp);
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);
    double norm=0.0;
    for (int mu=0;mu<4;++mu)
        for (int nu=0;nu<4;++nu)
            norm += gLL[mu][nu]*cs.u[mu]*cs.u[nu];
    if (!approx(norm, -1.0, 1e-6)) {
        std::cerr << "FAIL norm = " << norm << "\n"; return false;
    }
    return true;
}

bool test_worldline_killing_energy() {
    KNdSMetric bh(1.0, 0.5, 0.0, 0.0);
    FallingParams fp; fp.bh = bh; fp.r_start = 15.0; fp.E = 1.2; fp.L = 0.0; fp.Qc = 0.0;
    CameraState cs = init_worldline(fp);
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);
    double u_T = 0.0;
    for (int nu=0;nu<4;++nu) u_T += gLL[0][nu]*cs.u[nu];
    if (!approx(u_T, -fp.E, 1e-6)) {
        std::cerr << "FAIL u_T = " << u_T << " expected " << -fp.E << "\n"; return false;
    }
    return true;
}

bool test_worldline_step_stays_normalized() {
    KNdSMetric bh(1.0, 0.9, 0.0, 0.0);
    FallingParams fp; fp.bh = bh; fp.r_start = 15.0; fp.E = 1.0; fp.L = 0.0; fp.Qc = 0.0;
    CameraState cs = init_worldline(fp);
    for (int i=0;i<10;++i) cs = step_worldline(cs, bh, 0.05);
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);
    double norm=0.0;
    for (int mu=0;mu<4;++mu)
        for (int nu=0;nu<4;++nu) norm += gLL[mu][nu]*cs.u[mu]*cs.u[nu];
    if (!approx(norm, -1.0, 1e-5)) {
        std::cerr<<"FAIL step norm="<<norm<<"\n"; return false;
    }
    return true;
}

bool test_worldline_r_decreases() {
    KNdSMetric bh(1.0, 0.5, 0.0, 0.0);
    FallingParams fp; fp.bh = bh; fp.r_start = 20.0; fp.E = 1.0; fp.L = 0.0; fp.Qc = 0.0;
    CameraState cs = init_worldline(fp);
    double r0 = cs.x[1];
    for (int i=0;i<20;++i) cs = step_worldline(cs, bh, 0.1);
    if (cs.x[1] >= r0) {
        std::cerr<<"FAIL r did not decrease: "<<r0<<" -> "<<cs.x[1]<<"\n"; return false;
    }
    return true;
}

bool test_tetrad_orthonormal() {
    KNdSMetric bh(1.0, 0.9, 0.0, 0.0);
    FallingParams fp; fp.bh = bh; fp.r_start = 5.0; fp.E = 1.0; fp.L = 0.0; fp.Qc = 0.0;
    CameraState cs = init_worldline(fp);
    double e[4][4];
    build_tetrad(cs, bh, e);
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);
    const double eta[4][4]={{-1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    for (int a=0;a<4;++a) for (int b=0;b<4;++b) {
        double s=0.0;
        for (int mu=0;mu<4;++mu) for (int nu=0;nu<4;++nu)
            s += gLL[mu][nu]*e[a][mu]*e[b][nu];
        if (!approx(s, eta[a][b], 1e-6)) {
            std::cerr<<"FAIL tetrad ["<<a<<"]["<<b<<"] = "<<s<<" expected "<<eta[a][b]<<"\n";
            return false;
        }
    }
    return true;
}

bool test_apply_roll() {
    // Build a trivial tetrad in flat space and verify roll by π/2 swaps ê_1↔ê_2
    // We test using a Schwarzschild BH (a=0) at r=100 (nearly flat)
    KNdSMetric bh(1.0, 0.0, 0.0, 0.0);
    FallingParams fp; fp.bh = bh; fp.r_start=100.0; fp.E=1.0;
    CameraState cs = init_worldline(fp);
    double e[4][4];
    build_tetrad(cs, bh, e);
    // Save e[1] and e[2] before roll
    double e1_before[4], e2_before[4];
    for (int mu=0;mu<4;++mu) { e1_before[mu]=e[1][mu]; e2_before[mu]=e[2][mu]; }
    double e0_before[4], e3_before[4];
    for (int mu=0;mu<4;++mu) { e0_before[mu]=e[0][mu]; e3_before[mu]=e[3][mu]; }
    // Roll by π/2: ê_1' = ê_2, ê_2' = -ê_1
    apply_roll(e, M_PI/2.0);
    for (int mu=0;mu<4;++mu) {
        if (!approx(e[1][mu], e2_before[mu], 1e-12)) {
            std::cerr<<"FAIL apply_roll e[1] after π/2, mu="<<mu<<"\n"; return false;
        }
        if (!approx(e[2][mu], -e1_before[mu], 1e-12)) {
            std::cerr<<"FAIL apply_roll e[2] after π/2, mu="<<mu<<"\n"; return false;
        }
        if (!approx(e[0][mu], e0_before[mu], 1e-12)) {
            std::cerr<<"FAIL apply_roll modified e[0], mu="<<mu<<"\n"; return false;
        }
        if (!approx(e[3][mu], e3_before[mu], 1e-12)) {
            std::cerr<<"FAIL apply_roll modified e[3], mu="<<mu<<"\n"; return false;
        }
    }
    return true;
}

bool test_horizon_flip_psi() {
    // Schwarzschild r_horizon = 2M
    double r_h = 2.0;
    // r_far = 4.0, r_near = 1.6
    // Far: ψ = π
    if (!approx(horizon_flip_psi(10.0, r_h), M_PI, 1e-12)) {
        std::cerr<<"FAIL flip_psi far\n"; return false;
    }
    // Inside: ψ = 0
    if (!approx(horizon_flip_psi(0.5, r_h), 0.0, 1e-12)) {
        std::cerr<<"FAIL flip_psi inside\n"; return false;
    }
    // At r_far exactly: ψ = π
    if (!approx(horizon_flip_psi(4.0, r_h), M_PI, 1e-12)) {
        std::cerr<<"FAIL flip_psi at r_far\n"; return false;
    }
    // At r_near exactly: ψ = 0
    if (!approx(horizon_flip_psi(1.6, r_h), 0.0, 1e-12)) {
        std::cerr<<"FAIL flip_psi at r_near\n"; return false;
    }
    // Midpoint t=0.5: smoothstep(0.5) = 0.5, ψ = π * 0.5 = π/2
    double r_mid = 1.6 + 0.5*(4.0 - 1.6); // = 2.8
    double psi_mid = horizon_flip_psi(r_mid, r_h);
    if (!approx(psi_mid, M_PI/2.0, 1e-10)) {
        std::cerr<<"FAIL flip_psi midpoint psi="<<psi_mid<<"\n"; return false;
    }
    return true;
}

int main() {
    int fail=0;
    if (!test_gpg_metric_inverse())  { std::cerr<<"FAIL test_gpg_metric_inverse\n";  ++fail; }
    else std::cout<<"PASS test_gpg_metric_inverse\n";
    if (!test_gpg_flat_limit())      { std::cerr<<"FAIL test_gpg_flat_limit\n";      ++fail; }
    else std::cout<<"PASS test_gpg_flat_limit\n";
    if (!test_worldline_init_normalized()) { std::cerr<<"FAIL worldline_normalized\n"; ++fail; }
    else std::cout<<"PASS test_worldline_init_normalized\n";
    if (!test_worldline_killing_energy())  { std::cerr<<"FAIL killing_energy\n";       ++fail; }
    else std::cout<<"PASS test_worldline_killing_energy\n";
    if (!test_worldline_step_stays_normalized()) { std::cerr<<"FAIL step_normalized\n"; ++fail; }
    else std::cout<<"PASS test_worldline_step_stays_normalized\n";
    if (!test_worldline_r_decreases())           { std::cerr<<"FAIL r_decreases\n";     ++fail; }
    else std::cout<<"PASS test_worldline_r_decreases\n";
    if (!test_tetrad_orthonormal()) { std::cerr<<"FAIL tetrad_orthonormal\n"; ++fail; }
    else std::cout<<"PASS test_tetrad_orthonormal\n";
    if (!test_apply_roll())         { std::cerr<<"FAIL apply_roll\n"; ++fail; }
    else std::cout<<"PASS test_apply_roll\n";
    if (!test_horizon_flip_psi())   { std::cerr<<"FAIL horizon_flip_psi\n"; ++fail; }
    else std::cout<<"PASS test_horizon_flip_psi\n";
    return fail;
}
