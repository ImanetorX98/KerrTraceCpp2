#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 2) return 1;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { std::cout << "Metal device unavailable\n"; return 77; }
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
            encoding:NSUTF8StringEncoding error:&error];
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion2_4;
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
        else options.fastMathEnabled = NO;
        id<MTLLibrary> lib = source ? [device newLibraryWithSource:source options:options error:&error] : nil;
        if (!lib) { std::cerr << [[error localizedDescription] UTF8String] << '\n'; return 1; }
        for (NSString* name in @[@"trace_pixel", @"trace_pixel_single", @"trace_pixel_bundle"]) {
            id<MTLFunction> function = [lib newFunctionWithName:name];
            id<MTLComputePipelineState> pipeline = function ?
                [device newComputePipelineStateWithFunction:function error:&error] : nil;
            if (!pipeline) { std::cerr << "Cannot build pipeline " << [name UTF8String] << '\n'; return 1; }
        }
        std::cout << "Metal shader and all entrypoints compiled successfully\n";
    }
    return 0;
}
