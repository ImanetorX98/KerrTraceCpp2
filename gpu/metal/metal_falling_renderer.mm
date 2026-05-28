// metal_falling_renderer.mm — Objective-C++ Metal bridge for falling-camera renderer.
// Loads tracer_falling.metal at runtime, dispatches one thread per pixel.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_falling_renderer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

// ── Per-process singleton ─────────────────────────────────────────────────────
namespace {
struct FallingMetalCache {
    id<MTLDevice>               device = nil;
    id<MTLLibrary>              lib    = nil;
    id<MTLComputePipelineState> pso    = nil;
    id<MTLCommandQueue>         queue  = nil;
};

static FallingMetalCache g_fc;
static bool              g_fc_valid = false;

static bool ensure_falling_cache()
{
    if (g_fc_valid) return true;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return false;

    // Load tracer_falling.metal: try source path, then build-dir copy
    NSString* exeDir = [[[NSProcessInfo processInfo] arguments][0]
                         stringByDeletingLastPathComponent];
    NSString* src = nil;
    NSError*  err = nil;
    for (NSString* candidate in @[
        [exeDir stringByAppendingPathComponent:@"../gpu/metal/tracer_falling.metal"],
        [exeDir stringByAppendingPathComponent:@"tracer_falling.metal"],
        [[[NSBundle mainBundle] resourcePath]
             stringByAppendingPathComponent:@"tracer_falling.metal"]
    ]) {
        src = [NSString stringWithContentsOfFile:candidate
                                        encoding:NSUTF8StringEncoding
                                           error:&err];
        if (src) break;
    }
    if (!src) {
        fprintf(stderr, "[falling-metal] cannot load tracer_falling.metal\n");
        return false;
    }

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion2_4;
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeFast;
    } else {
        opts.fastMathEnabled = YES;
    }

    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "[falling-metal] compile error: %s\n",
                [err.localizedDescription UTF8String]);
        return false;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"trace_falling_pixel"];
    if (!fn) {
        fprintf(stderr, "[falling-metal] kernel 'trace_falling_pixel' not found\n");
        return false;
    }

    NSError* e2 = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&e2];
    if (!pso) {
        fprintf(stderr, "[falling-metal] PSO creation failed: %s\n",
                [e2.localizedDescription UTF8String]);
        return false;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return false;

    g_fc = { device, lib, pso, queue };
    g_fc_valid = true;
    return true;
}
} // namespace

// ── make_falling_metal_params ─────────────────────────────────────────────────
FallingCameraParams_C make_falling_metal_params(
    const FallingParams& fp,
    const CameraState&   cs,
    const double         e[4][4])
{
    FallingCameraParams_C c{};
    for (int a=0;a<4;++a)
        for (int mu=0;mu<4;++mu)
            c.e[a][mu] = float(e[a][mu]);
    for (int i=0;i<4;++i) c.x[i] = float(cs.x[i]);
    c.M      = float(fp.bh.M);
    c.a      = float(fp.bh.a);
    c.Q      = float(fp.bh.Q);
    c.Lambda = float(fp.bh.Lambda);
    const double r_isco_val = fp.bh.r_isco();
    c.r_in          = float((fp.r_disk_in < 0.0) ? r_isco_val : fp.r_disk_in);
    c.r_out         = float(fp.r_disk_out);
    c.r_isco        = float(r_isco_val);
    c.r_escape      = float(fp.r_escape);
    c.r_singularity = float(fp.r_singularity);
    c.r_horizon     = float(fp.bh.r_horizon());
    c.disk_brightness = float(fp.disk_brightness);
    c.fov_h         = float(fp.fov_h);
    c.h0            = 0.05f;
    c.r_switch_factor = float(fp.r_switch_factor);
    c.max_steps     = 20000;   // GPU cap (lower than CPU 50000; near-horizon pixels
                                // are refined by CPU pass anyway)
    c.width         = fp.width;
    c.height        = fp.height;
    c.pad[0] = c.pad[1] = c.pad[2] = 0;
    return c;
}

// ── metal_render_falling_frame ────────────────────────────────────────────────
bool metal_render_falling_frame(
    const FallingCameraParams_C& params,
    std::vector<uint8_t>&        rgb,
    std::vector<float>&          r_min)
{
    if (!ensure_falling_cache()) return false;

    auto& c = g_fc;
    const NSUInteger W    = (NSUInteger)params.width;
    const NSUInteger H    = (NSUInteger)params.height;
    const NSUInteger npix = W * H;

    id<MTLBuffer> rgbBuf  = [c.device
        newBufferWithLength:npix * 4
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> rminBuf = [c.device
        newBufferWithLength:npix * sizeof(float)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> cpBuf   = [c.device
        newBufferWithBytes:&params
        length:sizeof(params)
        options:MTLResourceStorageModeShared];
    if (!rgbBuf || !rminBuf || !cpBuf) return false;

    const auto t0 = std::chrono::steady_clock::now();

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pso];
    [enc setBuffer:rgbBuf  offset:0 atIndex:0];
    [enc setBuffer:rminBuf offset:0 atIndex:1];
    [enc setBuffer:cpBuf   offset:0 atIndex:2];

    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(((W+15)/16)*16, ((H+15)/16)*16, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[falling-metal] command buffer failed: %s\n",
                cmd.error ? [cmd.error.localizedDescription UTF8String] : "unknown");
        return false;
    }

    const auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1-t0).count();
    fprintf(stderr, "[falling-metal] GPU pass: %.2fs\n", elapsed);

    // Copy RGBA → RGB
    const uint8_t* src = (const uint8_t*)[rgbBuf contents];
    rgb.resize(npix * 3);
    for (NSUInteger i=0; i<npix; ++i) {
        rgb[i*3+0] = src[i*4+0];
        rgb[i*3+1] = src[i*4+1];
        rgb[i*3+2] = src[i*4+2];
    }
    r_min.resize(npix);
    std::memcpy(r_min.data(), [rminBuf contents], npix * sizeof(float));
    return true;
}
