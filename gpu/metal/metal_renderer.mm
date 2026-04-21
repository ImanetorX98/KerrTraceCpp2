// ============================================================
//  metal_renderer.mm — Objective-C++ Metal dispatch layer
//
//  Compiles only on macOS/iOS with Xcode.
//  Loads tracer.metal at runtime, dispatches one thread per pixel.
// ============================================================
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_renderer.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<uint32_t> metal_render(
    const KNdSParams_C&  kp,
    const CameraParams_C& cp,
    const uint8_t* bg_rgb,
    int bg_w,
    int bg_h)
{
    // ── Acquire GPU device ────────────────────────────────────
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
        throw std::runtime_error("No Metal-capable GPU found");

    // ── Compile the shader at runtime from source ─────────────
    NSError* err = nil;
    // Locate tracer.metal relative to the executable
    NSString* shaderPath = [[[NSBundle mainBundle] resourcePath]
                             stringByAppendingPathComponent:@"tracer.metal"];
    NSString* src = [NSString stringWithContentsOfFile:shaderPath
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
        // Fallback: look in the same directory as the binary
        NSString* exeDir = [[[NSProcessInfo processInfo]
                              arguments][0] stringByDeletingLastPathComponent];
        shaderPath = [exeDir stringByAppendingPathComponent:@"../gpu/metal/tracer.metal"];
        src = [NSString stringWithContentsOfFile:shaderPath
                                        encoding:NSUTF8StringEncoding
                                           error:&err];
    }
    if (!src)
        throw std::runtime_error(
            std::string("Cannot load tracer.metal: ") +
            [err.localizedDescription UTF8String]);

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion2_4;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib)
        throw std::runtime_error(
            std::string("Metal compile error: ") +
            [err.localizedDescription UTF8String]);

    NSString* kernelName = @"trace_pixel_single";
    switch (cp.metal_kernel_mode) {
        case 1:  kernelName = @"trace_pixel";        break; // unified legacy kernel
        case 2:  kernelName = @"trace_pixel_single"; break;
        case 3:  kernelName = @"trace_pixel_bundle"; break;
        default: kernelName = (cp.use_bundles != 0) ? @"trace_pixel_bundle"
                                                    : @"trace_pixel_single";
                 break;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:kernelName];
    if (!fn) {
        // Backward-compatible fallback for older shader builds.
        fn = [lib newFunctionWithName:@"trace_pixel"];
    }
    if (!fn)
        throw std::runtime_error("Cannot find Metal kernel entrypoint in tracer.metal");

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso)
        throw std::runtime_error(
            std::string("PSO creation failed: ") +
            [err.localizedDescription UTF8String]);

    // ── Buffers ───────────────────────────────────────────────
    const NSUInteger npix = cp.width * cp.height;
    id<MTLBuffer> outBuf = [device
        newBufferWithLength:npix * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];

    id<MTLBuffer> kpBuf  = [device
        newBufferWithBytes:&kp
        length:sizeof(kp)
        options:MTLResourceStorageModeShared];

    id<MTLBuffer> cpBuf  = [device
        newBufferWithBytes:&cp
        length:sizeof(cp)
        options:MTLResourceStorageModeShared];

    struct RenderParams_C {
        uint32_t width;
        uint32_t height;
        uint32_t x_offset;
        uint32_t tile_w;
        uint32_t y_offset;
        uint32_t tile_h;
    };

    // ── Background texture (RGB8 -> RGBA8, sampled in shader) ─
    const bool has_bg = (bg_rgb != nullptr && bg_w > 0 && bg_h > 0);
    const int tex_w = has_bg ? bg_w : 1;
    const int tex_h = has_bg ? bg_h : 1;
    std::vector<uint8_t> rgba((size_t)tex_w * (size_t)tex_h * 4u, 0u);
    if (has_bg) {
        for (size_t i = 0, n = (size_t)bg_w * (size_t)bg_h; i < n; ++i) {
            rgba[4*i + 0] = bg_rgb[3*i + 0];
            rgba[4*i + 1] = bg_rgb[3*i + 1];
            rgba[4*i + 2] = bg_rgb[3*i + 2];
            rgba[4*i + 3] = 255u;
        }
    } else {
        rgba[3] = 255u;
    }

    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:(NSUInteger)tex_w
                                                          height:(NSUInteger)tex_h
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> bgTex = [device newTextureWithDescriptor:td];
    if (!bgTex)
        throw std::runtime_error("Failed to create Metal background texture");

    MTLRegion region = MTLRegionMake2D(0, 0, tex_w, tex_h);
    [bgTex replaceRegion:region
             mipmapLevel:0
               withBytes:rgba.data()
             bytesPerRow:(NSUInteger)(tex_w * 4)];

    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeRepeat;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    id<MTLSamplerState> bgSamp = [device newSamplerStateWithDescriptor:sd];
    if (!bgSamp)
        throw std::runtime_error("Failed to create Metal sampler state");

    // ── Dispatch ──────────────────────────────────────────────
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue)
        throw std::runtime_error("Failed to create Metal command queue");

    // Tile in adaptive row/column slices to stay under GPU interactivity watchdog.
    // On failure we retry the same tile with smaller extents until it succeeds.
    //
    // Optional env overrides:
    //   KERR_METAL_TILE_ROWS=<1..64>
    //   KERR_METAL_TILE_COLS=<16..4096>
    // The shader uses x_offset/y_offset so each tile writes to the correct area.
    constexpr int TILE_ROWS_MAX = 64;
    constexpr int TILE_COLS_MAX = 4096;
    auto clamp_i = [](int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    };
    auto default_tile_rows = [&]() -> int {
        int rows = 16;
        if (cp.solver_mode != 0) rows = 8; // semi/elliptic are typically heavier
        if (cp.use_bundles != 0) rows = std::min(rows, 2);
        if (cp.width >= 2560 || cp.height >= 1440) rows = std::min(rows, 4);
        if (cp.width >= 3840 || cp.height >= 2160) rows = std::min(rows, 2);
        if (cp.use_bundles != 0 && (cp.width >= 2560 || cp.height >= 1440))
            rows = 1;
        return rows;
    };
    auto default_tile_cols = [&]() -> int {
        int cols = 1024;
        if (cp.use_bundles != 0) cols = 256;
        if (cp.width >= 3840) cols = std::min(cols, 512);
        if (cp.use_bundles != 0 && cp.width >= 2560) cols = std::min(cols, 128);
        return cols;
    };

    int tile_rows = default_tile_rows();
    int tile_cols = default_tile_cols();
    if (const char* env = std::getenv("KERR_METAL_TILE_ROWS")) {
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end != env && *end == '\0')
            tile_rows = clamp_i((int)parsed, 1, TILE_ROWS_MAX);
    }
    if (const char* env = std::getenv("KERR_METAL_TILE_COLS")) {
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end != env && *end == '\0')
            tile_cols = clamp_i((int)parsed, 16, TILE_COLS_MAX);
    }

    MTLSize tg = MTLSizeMake(16, 16, 1);
    for (int y0 = 0; y0 < cp.height; y0 += tile_rows) {
        const int base_rows = std::min(tile_rows, cp.height - y0);
        for (int x0 = 0; x0 < cp.width;) {
            int try_rows = base_rows;
            int try_cols = std::min(tile_cols, cp.width - x0);
            std::string last_error;
            bool submitted = false;

            while (!submitted) {
                const RenderParams_C rp{
                    (uint32_t)cp.width, (uint32_t)cp.height,
                    (uint32_t)x0, (uint32_t)try_cols,
                    (uint32_t)y0, (uint32_t)try_rows
                };
                id<MTLBuffer> rpBuf = [device
                    newBufferWithBytes:&rp
                    length:sizeof(rp)
                    options:MTLResourceStorageModeShared];

                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:outBuf  offset:0 atIndex:0];
                [enc setBuffer:kpBuf   offset:0 atIndex:1];
                [enc setBuffer:cpBuf   offset:0 atIndex:2];
                [enc setBuffer:rpBuf   offset:0 atIndex:3];
                [enc setTexture:bgTex atIndex:0];
                [enc setSamplerState:bgSamp atIndex:0];

                MTLSize grid = MTLSizeMake(
                    ((NSUInteger)try_cols + 15) / 16 * 16,
                    ((NSUInteger)try_rows + 15) / 16 * 16,
                    1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];

                if (cmd.status == MTLCommandBufferStatusCompleted) {
                    submitted = true;
                    x0 += try_cols;
                } else {
                    NSString* why = cmd.error ? cmd.error.localizedDescription : @"unknown Metal failure";
                    last_error = [why UTF8String];

                    if (try_rows > 1) {
                        try_rows = std::max(1, try_rows / 2);
                        continue;
                    }
                    if (try_cols > 16) {
                        try_cols = std::max(16, try_cols / 2);
                        continue;
                    }
                    throw std::runtime_error("Metal command failed: " + last_error);
                }
            }
        }
    }

    // ── Copy result ───────────────────────────────────────────
    std::vector<uint32_t> pixels(npix);
    std::memcpy(pixels.data(), [outBuf contents], npix * sizeof(uint32_t));
    return pixels;
}
