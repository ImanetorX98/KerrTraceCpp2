// ============================================================
//  metal_renderer.mm — Objective-C++ Metal dispatch layer
//
//  Compiles only on macOS/iOS with Xcode.
//  Loads tracer.metal at runtime, dispatches one thread per pixel.
// ============================================================
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_renderer.hpp"
#include <stdexcept>
#include <string>

std::vector<uint32_t> metal_render(
    const KNdSParams_C&  kp,
    const CameraParams_C& cp)
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

    id<MTLFunction>    fn  = [lib newFunctionWithName:@"trace_pixel"];
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

    // ── Dispatch ──────────────────────────────────────────────
    id<MTLCommandQueue>  queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmd   = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pso];
    [enc setBuffer:outBuf  offset:0 atIndex:0];
    [enc setBuffer:kpBuf   offset:0 atIndex:1];
    [enc setBuffer:cpBuf   offset:0 atIndex:2];

    // Thread-group 16×16; grid covers the full image
    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(
        ((NSUInteger)cp.width  + 15) / 16 * 16,
        ((NSUInteger)cp.height + 15) / 16 * 16,
        1);

    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // ── Copy result ───────────────────────────────────────────
    std::vector<uint32_t> pixels(npix);
    std::memcpy(pixels.data(), [outBuf contents], npix * sizeof(uint32_t));
    return pixels;
}
