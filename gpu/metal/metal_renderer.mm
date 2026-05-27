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
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ── Per-process Metal singleton ───────────────────────────────────────────────
// Compiling tracer.metal takes ~60 s on first call. Cache device, library, and
// all three PSOs so subsequent calls within the same process are instant.
namespace {

struct MetalCache {
    id<MTLDevice>              device  = nil;
    id<MTLLibrary>             lib     = nil;
    id<MTLComputePipelineState> pso_single  = nil; // trace_pixel_single
    id<MTLComputePipelineState> pso_bundle  = nil; // trace_pixel_bundle
    id<MTLComputePipelineState> pso_unified = nil; // trace_pixel (legacy)
    id<MTLCommandQueue>        queue   = nil;
    bool                       precise = false;
};

static MetalCache g_cache;
static bool       g_cache_valid = false;

static void ensure_metal_cache(bool precise_math)
{
    if (g_cache_valid && g_cache.precise == precise_math) return;

    NSError* err = nil;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
        throw std::runtime_error("No Metal-capable GPU found");

    NSString* exeDir = [[[NSProcessInfo processInfo]
                          arguments][0] stringByDeletingLastPathComponent];
    NSString* shaderPath = [exeDir stringByAppendingPathComponent:@"../gpu/metal/tracer.metal"];
    NSString* src = [NSString stringWithContentsOfFile:shaderPath
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
        shaderPath = [[[NSBundle mainBundle] resourcePath]
                        stringByAppendingPathComponent:@"tracer.metal"];
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
    if (@available(macOS 15.0, *)) {
        opts.mathMode = precise_math ? MTLMathModeSafe : MTLMathModeFast;
    } else {
        opts.fastMathEnabled = precise_math ? NO : YES;
    }

    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib)
        throw std::runtime_error(
            std::string("Metal compile error: ") +
            [err.localizedDescription UTF8String]);

    auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) return nil;
        NSError* e2 = nil;
        id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&e2];
        return p; // nil on failure is handled at call-site
    };

    id<MTLComputePipelineState> pso_single  = make_pso(@"trace_pixel_single");
    id<MTLComputePipelineState> pso_bundle  = make_pso(@"trace_pixel_bundle");
    id<MTLComputePipelineState> pso_unified = make_pso(@"trace_pixel");

    if (!pso_single && !pso_unified)
        throw std::runtime_error("PSO creation failed: no usable kernel found in tracer.metal");

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue)
        throw std::runtime_error("Failed to create Metal command queue");

    g_cache.device      = device;
    g_cache.lib         = lib;
    g_cache.pso_single  = pso_single;
    g_cache.pso_bundle  = pso_bundle;
    g_cache.pso_unified = pso_unified;
    g_cache.queue       = queue;
    g_cache.precise     = precise_math;
    g_cache_valid       = true;
}

} // namespace

std::vector<uint32_t> metal_render(
    const KNdSParams_C&  kp,
    const CameraParams_C& cp,
    const uint8_t* bg_rgb,
    int bg_w,
    int bg_h)
{
    // ── Ensure cached device / compiled library / PSOs ────────
    bool precise_math = false;
    if (const char* env = std::getenv("KERR_METAL_PRECISE_MATH"))
        precise_math = (std::atoi(env) != 0);

    ensure_metal_cache(precise_math);

    id<MTLDevice>       device = g_cache.device;
    id<MTLCommandQueue> queue  = g_cache.queue;

    // ── Select PSO ────────────────────────────────────────────
    NSString* kernelName = @"trace_pixel_single";
    switch (cp.metal_kernel_mode) {
        case 1:  kernelName = @"trace_pixel";        break;
        case 2:  kernelName = @"trace_pixel_single"; break;
        case 3:  kernelName = @"trace_pixel_bundle"; break;
        default: kernelName = (cp.use_bundles != 0) ? @"trace_pixel_bundle"
                                                    : @"trace_pixel_single";
                 break;
    }

    id<MTLComputePipelineState> pso = nil;
    if      ([kernelName isEqualToString:@"trace_pixel_bundle"])  pso = g_cache.pso_bundle;
    else if ([kernelName isEqualToString:@"trace_pixel_single"])  pso = g_cache.pso_single;
    else                                                           pso = g_cache.pso_unified;
    // Fallback chain
    if (!pso) pso = g_cache.pso_single;
    if (!pso) pso = g_cache.pso_unified;
    if (!pso)
        throw std::runtime_error("No suitable Metal PSO available for requested kernel");

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
    // (queue is the cached g_cache.queue)

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

    // Build an explicit tile list, then split failed tiles.
    // This guarantees full image coverage even when retries shrink tile size.
    struct Tile {
        int x;
        int y;
        int w;
        int h;
    };
    std::vector<Tile> tiles;
    tiles.reserve(((cp.width + tile_cols - 1) / tile_cols) *
                  ((cp.height + tile_rows - 1) / tile_rows));
    for (int y0 = 0; y0 < cp.height; y0 += tile_rows) {
        const int h = std::min(tile_rows, cp.height - y0);
        for (int x0 = 0; x0 < cp.width; x0 += tile_cols) {
            const int w = std::min(tile_cols, cp.width - x0);
            tiles.push_back(Tile{x0, y0, w, h});
        }
    }

    const size_t total_tiles = tiles.size();
    size_t done_tiles = 0;
    const long long total_pixels = (long long)cp.width * cp.height;
    const auto t0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < tiles.size();) {
        const Tile t = tiles[i];
        const RenderParams_C rp{
            (uint32_t)cp.width, (uint32_t)cp.height,
            (uint32_t)t.x, (uint32_t)t.w,
            (uint32_t)t.y, (uint32_t)t.h
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
            ((NSUInteger)t.w + 15) / 16 * 16,
            ((NSUInteger)t.h + 15) / 16 * 16,
            1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status == MTLCommandBufferStatusCompleted) {
            ++i;
            ++done_tiles;
            // Emit progress in the same format the CPU renderer uses so the
            // server and frontend can track it.
            if (total_tiles > 0) {
                const int pct = (int)(done_tiles * 100 / total_tiles);
                const auto now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - t0).count();
                const double eta = (done_tiles < total_tiles && elapsed > 0.0)
                    ? elapsed * (double)(total_tiles - done_tiles) / (double)done_tiles
                    : 0.0;
                // Build a fixed-width bar (40 chars)
                const int bar_fill = (int)(pct * 40 / 100);
                char bar[41];
                for (int b = 0; b < 40; ++b) bar[b] = (b < bar_fill) ? '#' : '-';
                bar[40] = '\0';
                fprintf(stderr, "\r[%s] %3d%%  %.1fs elapsed, %.1fs ETA   ",
                        bar, pct, elapsed, eta);
                fflush(stderr);
                if (done_tiles == total_tiles) fprintf(stderr, "\n");
            }
            continue;
        }

        NSString* why = cmd.error ? cmd.error.localizedDescription : @"unknown Metal failure";
        const std::string last_error = [why UTF8String];

        if (t.h > 1) {
            const int h0 = t.h / 2;
            const int h1 = t.h - h0;
            tiles[i] = Tile{t.x, t.y, t.w, h0};
            tiles.insert(tiles.begin() + static_cast<std::ptrdiff_t>(i + 1),
                         Tile{t.x, t.y + h0, t.w, h1});
            continue;
        }
        if (t.w > 16) {
            const int w0 = std::max(16, t.w / 2);
            const int w1 = t.w - w0;
            tiles[i] = Tile{t.x, t.y, w0, t.h};
            if (w1 > 0) {
                tiles.insert(tiles.begin() + static_cast<std::ptrdiff_t>(i + 1),
                             Tile{t.x + w0, t.y, w1, t.h});
            }
            continue;
        }

        throw std::runtime_error("Metal command failed: " + last_error);
    }

    // ── Copy result ───────────────────────────────────────────
    std::vector<uint32_t> pixels(npix);
    std::memcpy(pixels.data(), [outBuf contents], npix * sizeof(uint32_t));
    return pixels;
}
