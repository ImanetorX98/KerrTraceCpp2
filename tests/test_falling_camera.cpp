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
    return fail;
}
