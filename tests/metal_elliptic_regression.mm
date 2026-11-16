// The Metal closed-form (elliptic) solver against the CPU, in FP32.
//   1. F(phi|m) over [-2pi, 2pi] and Jacobi sn, cn over four periods of u,
//      against the CPU's double versions;
//   2. every direct disk hit the shader claims on a frame (elliptic_closed_first_hit
//      == 2) is a disk hit of the CPU BL integrator, at the same radius.
// Before 2026-11-16 cn had the wrong sign for u in (2K, 4K) (only sn was
// flipped), which put Region III hits at r ~ 1.3 instead of ~ 2 above the shadow,
// and the FP32 quartic solver, started from a conjugate-symmetric set, could
// not split two close real roots and took Region I/II rays for Region III.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define KERRTRACE_NO_MAIN
#ifndef OUT_DIR
#define OUT_DIR "."
#endif
#include "../main.cpp"

struct RaySample { float M, a, E, Lz, Qc, r0, th0, dr0, dth0, rh, risco, rout, resc; };
struct JacobiSample { float u, m, Kc; };

static NSString* const kKernels = @R"(
struct RaySample { float M, a, E, Lz, Qc, r0, th0, dr0, dth0, rh, risco, rout, resc; };
struct JacobiSample { float u, m, Kc; };
kernel void test_jacobi(device const JacobiSample* in [[buffer(0)]], device float* out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    float sn = NAN, cn = NAN;
    jacobi_sn_cn_from_u(in[i].u, in[i].m, in[i].Kc, sn, cn);
    out[2*i] = sn; out[2*i+1] = cn;
}
kernel void test_F(device const JacobiSample* in [[buffer(0)]], device float* out [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    out[2*i] = ellint_F_incomplete(in[i].u, in[i].m);
    out[2*i+1] = 0.0f;
}
kernel void test_first_hit(device const RaySample* in [[buffer(0)]], device float* out [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    const RaySample p = in[i];
    SeparableConsts c{}; c.M = p.M; c.a = p.a; c.E = p.E; c.Lz = p.Lz; c.Qc = p.Qc;
    float r_hit = 0.0f;
    out[2*i] = float(elliptic_closed_first_hit(c, p.r0, p.th0, p.dr0, p.dth0,
                                               p.rh, p.risco, p.rout, p.resc, r_hit));
    out[2*i+1] = r_hit;
}
)";

template <class T>
static const float* run(id<MTLDevice> device, id<MTLLibrary> lib, NSString* name,
                        const std::vector<T>& in, id<MTLBuffer> __strong& output) {
    NSError* error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    id<MTLComputePipelineState> pipeline = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!pipeline) return nullptr;
    id<MTLBuffer> input = [device newBufferWithBytes:in.data() length:in.size() * sizeof(T)
                                             options:MTLResourceStorageModeShared];
    output = [device newBufferWithLength:in.size() * 2 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(in.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return nullptr;
    return static_cast<const float*>(output.contents);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 2) return 1;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 77;
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                     encoding:NSUTF8StringEncoding error:&error];
        if (!source) { std::cerr << [[error localizedDescription] UTF8String] << '\n'; return 1; }
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion2_4;
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
        else options.fastMathEnabled = NO;
        id<MTLLibrary> lib = [device newLibraryWithSource:[source stringByAppendingString:kKernels]
                                                  options:options error:&error];
        if (!lib) { std::cerr << [[error localizedDescription] UTF8String] << '\n'; return 1; }
        int failures = 0;

        std::vector<JacobiSample> amp;   // u holds the amplitude phi
        for (double m : {0.3, 0.8, 0.97})
            for (int k = -200; k <= 200; ++k) amp.push_back({float(k * 0.01 * 2 * M_PI), float(m), 0});
        id<MTLBuffer> amp_out = nil;
        const float* fv = run(device, lib, @"test_F", amp, amp_out);
        if (!fv) { std::cerr << "Metal F kernel failed\n"; return 1; }
        double f_worst = 0;
        for (size_t i = 0; i < amp.size(); ++i)
            f_worst = std::max(f_worst, std::abs(fv[2*i] - ellint_F_incomplete(amp[i].u, amp[i].m)));
        std::cout << "Metal F: " << amp.size() << " amplitudes over [-2pi, 2pi], max error " << f_worst << '\n';
        if (!(f_worst < 1e-4)) { std::cout << "[FAIL] Metal F(phi|m) differs from the CPU\n"; ++failures; }

        std::vector<JacobiSample> jac;
        for (double m : {0.3, 0.8, 0.97}) {
            const double K = ellint_K_complete(m);
            for (int k = -400; k <= 400; ++k)
                jac.push_back({float(k * 0.02 * K), float(m), float(K)});
        }
        id<MTLBuffer> jac_out = nil;
        const float* jv = run(device, lib, @"test_jacobi", jac, jac_out);
        if (!jv) { std::cerr << "Metal jacobi kernel failed\n"; return 1; }
        double jac_worst = 0;
        for (size_t i = 0; i < jac.size(); ++i) {
            double sn, cn;
            jacobi_sn_cn_from_u(jac[i].u, jac[i].m, jac[i].Kc, sn, cn);
            jac_worst = std::max({jac_worst, std::abs(jv[2*i] - sn), std::abs(jv[2*i+1] - cn)});
        }
        std::cout << "Metal sn/cn: " << jac.size() << " points over u in [-8K, 8K], max error "
                  << jac_worst << '\n';
        if (!(jac_worst < 1e-3)) { std::cout << "[FAIL] Metal sn/cn differ from the CPU\n"; ++failures; }

        // The default view at a = 0.998 (Region III photons hit the disk above the shadow).
        KNdSMetric g(1, 0.998, 0, 0);
        const double r_in = g.r_isco(), r_out = 12, r_esc = 63;
        Camera cam(60, 80, 0, 30, 160, 90);
        std::vector<RaySample> rays;
        std::vector<GeodesicState> states;
        for (int y = 0; y < 90; ++y) for (int x = 0; x < 160; ++x) {
            const auto s = cam.pixel_ray(x, y, g);
            SeparableKerrConsts c{};
            if (!init_separable_consts(s, g, c)) continue;
            double dr, dth, dpr, dpth;
            geodesic_rhs(g, s.r, s.theta, s.pr, s.ptheta, s.pt, s.pphi, dr, dth, dpr, dpth);
            rays.push_back({1, float(g.a), float(c.E), float(c.Lz), float(c.Qc), float(s.r),
                            float(s.theta), float(dr), float(dth), float(g.r_horizon()),
                            float(r_in), float(r_out), float(r_esc)});
            states.push_back(s);
        }
        id<MTLBuffer> hit_out = nil;
        const float* hv = run(device, lib, @"test_first_hit", rays, hit_out);
        if (!hv) { std::cerr << "Metal first-hit kernel failed\n"; return 1; }
        int direct = 0, not_disk = 0, far = 0;
        double worst = 0;
        for (size_t i = 0; i < rays.size(); ++i) {
            if (hv[2*i] != 2.0f) continue;
            ++direct;
            const TraceResult n = trace_single(states[i], g, r_in, r_out, r_esc, Integrator::DOPRI5);
            if (n.out != Outcome::DISK_HIT) { ++not_disk; continue; }
            const double err = std::abs(hv[2*i+1] - n.r) / n.r;
            worst = std::max(worst, err);
            if (err > 2e-3) ++far;
        }
        std::cout << "Metal direct disk hits: " << direct << ", " << not_disk
                  << " not disk hits for the CPU integrator, " << far
                  << " off by more than 0.2% (max " << worst << ")\n";
        if (direct == 0 || not_disk || far) {
            std::cout << "[FAIL] Metal closed-form hits disagree with the CPU integrator\n";
            ++failures;
        }
        std::cout << (failures ? "FAILED" : "ALL PASSED") << '\n';
        return failures ? 1 : 0;
    }
}
