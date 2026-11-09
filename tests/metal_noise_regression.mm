#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../renderer/common.hpp"
#include "../renderer/palettes.hpp"
#include <cstring>
#include <iostream>
#include <vector>

// The Interstellar disk texture (turbulence and soft edge) is value noise on an
// integer lattice. The lattice hash must give the same bits on the GPU as on the
// CPU, or the two backends draw different veins. The old sin hash failed this on
// ~80% of lattice points; the PCG hash must match on every one.
struct NoiseSample { float x,y,seed; int variant; };
static_assert(sizeof(NoiseSample)==16,"MSL input layout");

struct FbmVariant { int octaves,per_x,per_y; };
// The tilings the palettes actually use (soft mask, turbulence n1 and n2).
static const FbmVariant kVariants[]={{4,3,0},{4,2,0},{5,0,50},{4,0,157}};

int main(int argc,char** argv) {
    @autoreleasepool {
        if(argc!=2) return 1;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(!device) return 77;
        NSError* error=nil;
        NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
            encoding:NSUTF8StringEncoding error:&error];
        if(!source) {std::cerr<<[[error localizedDescription] UTF8String]<<'\n';return 1;}
        source=[source stringByAppendingString:@R"(
struct NoiseSample { float x,y,seed; int variant; };
kernel void test_noise(device const NoiseSample* samples [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    const NoiseSample s=samples[i];
    out[2*i]=lattice_hash_f(s.x,s.y,s.seed);
    const int oct[4]={4,4,5,4}, px[4]={3,2,0,0}, py[4]={0,0,50,157};
    const int v=s.variant;
    out[2*i+1]=fbm2d_tiled_f(s.x*0.173f,s.y*0.291f,s.seed,oct[v],px[v],py[v]);
}
)" ];
        MTLCompileOptions* options=[MTLCompileOptions new];
        options.languageVersion=MTLLanguageVersion2_4;
        if(@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
        else options.fastMathEnabled=NO;
        id<MTLLibrary> lib=[device newLibraryWithSource:source options:options error:&error];
        id<MTLFunction> fn=lib?[lib newFunctionWithName:@"test_noise"]:nil;
        id<MTLComputePipelineState> pipeline=fn?
            [device newComputePipelineStateWithFunction:fn error:&error]:nil;
        if(!pipeline) {std::cerr<<[[error localizedDescription] UTF8String]<<'\n';return 1;}

        // Lattice corners as the noise produces them: negative, wrapped and
        // large indices, and every seed the palettes pass (base + 17*octave).
        std::vector<NoiseSample> samples;
        int k=0;
        for(float seed : {11.f,28.f,29.f,41.f,58.f,73.f,92.f,109.f})
            for(int y=-40;y<=600;y+=7)
                for(int x=-300;x<=2600;x+=13)
                    samples.push_back(NoiseSample{float(x),float(y),seed,(k++)%4});

        id<MTLBuffer> input=[device newBufferWithBytes:samples.data()
            length:samples.size()*sizeof(NoiseSample) options:MTLResourceStorageModeShared];
        id<MTLBuffer> output=[device newBufferWithLength:samples.size()*2*sizeof(float)
            options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue=[device newCommandQueue];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if(!input || !output || !encoder) {std::cerr<<"Metal allocation failed\n";return 1;}
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(samples.size(),1,1)
            threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted) {std::cerr<<"GPU execution failed\n";return 1;}

        const float* values=static_cast<const float*>(output.contents);
        int hashFailures=0,fbmFailures=0;double fbmWorst=0;
        for(size_t i=0;i<samples.size();++i) {
            const auto& s=samples[i];
            const double expected=lattice_hash(s.x,s.y,s.seed);
            const float gpu=values[2*i];
            // Exact: 24-bit value, representable in float and double alike.
            if(double(gpu)!=expected) {
                if(hashFailures<8) std::cerr<<"hash mismatch x="<<s.x<<" y="<<s.y<<" seed="<<s.seed
                    <<" gpu="<<gpu<<" cpu="<<expected<<'\n';
                ++hashFailures;
            }
            // The fbm interpolates in float on the GPU: close, not bit-exact.
            const FbmVariant v=kVariants[s.variant];
            const double fbm=fbm2d_tiled(double(s.x*0.173f),double(s.y*0.291f),s.seed,
                                         v.octaves,v.per_x,v.per_y,false);
            const double err=std::abs(double(values[2*i+1])-fbm);
            fbmWorst=std::max(fbmWorst,err);
            if(!(err<=2e-5)) {
                if(fbmFailures<8) std::cerr<<"fbm mismatch x="<<s.x<<" y="<<s.y<<" seed="<<s.seed
                    <<" variant="<<s.variant<<" gpu="<<values[2*i+1]<<" cpu="<<fbm<<'\n';
                ++fbmFailures;
            }
        }
        std::cout<<"Metal lattice hash: "<<samples.size()<<" points, "<<hashFailures
                 <<" not bit-exact; fbm max error="<<fbmWorst<<" ("<<fbmFailures<<" over 2e-5)\n";
        return (hashFailures||fbmFailures)?1:0;
    }
}
