// The Metal semi-analytic (separable Kerr) path against the Metal standard BL
// path on the same frame: same image, and not much slower. Before 2026-11-16
// (audit A1) rays froze at turning points (dr = +-sqrt(max(R,0)) = 0 there)
// until the step cap, then fell back to BL: the image was right but a 720p
// frame took 3 minutes instead of 0.1 s.
#import <Metal/Metal.h>
#define KERRTRACE_NO_MAIN
#ifndef OUT_DIR
#define OUT_DIR "."
#endif
#include "../main.cpp"
#include <chrono>

int main() {
    @autoreleasepool {
        if (!MTLCreateSystemDefaultDevice()) return 77;
        constexpr int width = 160, height = 90;
        FrameParams frame; frame.a = .998; frame.theta = 80; frame.phi = 0;
        frame.r_obs = 60; frame.fov = 30; frame.disk_out = 12;
        BackgroundImage background;
        IntegratorControls controls;   // default cap: a stalled ray costs 500 000 steps
        ColorParams colors;
        colors.palette = DiskPalette::BLACKBODY;

        auto render = [&](RaySolverMode mode, double& seconds) {
            const auto t0 = std::chrono::steady_clock::now();
            auto img = render_image(width, height, frame, background, false, mode,
                CoordinateChart::BL, Integrator::RK4_DOUBLING, controls, 1, 1, 0, 0,
                IntersectionMode::HERMITE, 0, false, false, false, colors);
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return img;
        };
        double t_bl = 0, t_sep = 0;
        render(RaySolverMode::STANDARD, t_bl);   // warm-up: shader compilation
        auto bl = render(RaySolverMode::STANDARD, t_bl);
        auto sep = render(RaySolverMode::SEMI_ANALYTIC, t_sep);
        if (std::string(last_render_backend) != "gpu-metal") {
            std::cerr << "The Metal test must run on the GPU, not a CPU fallback\n";
            return 1;
        }
        int differ = 0;
        for (int i = 0; i < width * height; ++i) {
            const int d = std::max({std::abs(int(bl[i].r) - int(sep[i].r)),
                                    std::abs(int(bl[i].g) - int(sep[i].g)),
                                    std::abs(int(bl[i].b) - int(sep[i].b))});
            differ += d > 2;
        }
        std::cout << "Metal semi-analytic vs BL: " << differ << " of " << width * height
                  << " pixels differ by more than 2 levels; time " << t_sep << " s vs BL "
                  << t_bl << " s\n";
        int failures = 0;
        if (differ > width * height / 100) { std::cout << "[FAIL] image differs from BL\n"; ++failures; }
        if (t_sep > 5 * t_bl + 0.5) { std::cout << "[FAIL] rays stall: far slower than BL\n"; ++failures; }
        std::cout << (failures ? "FAILED" : "ALL PASSED") << '\n';
        return failures ? 1 : 0;
    }
}
