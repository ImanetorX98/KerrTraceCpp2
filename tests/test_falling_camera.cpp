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

int main() {
    int fail=0;
    if (!test_gpg_metric_inverse())  { std::cerr<<"FAIL test_gpg_metric_inverse\n";  ++fail; }
    else std::cout<<"PASS test_gpg_metric_inverse\n";
    if (!test_gpg_flat_limit())      { std::cerr<<"FAIL test_gpg_flat_limit\n";      ++fail; }
    else std::cout<<"PASS test_gpg_flat_limit\n";
    return fail;
}
