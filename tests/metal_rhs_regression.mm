#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../knds_metric.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

// Float inputs are also used by the double reference, so input quantization
// cannot masquerade as an error in the GPU's Hamiltonian derivatives.
struct Sample { float r,theta,pr,pth,pt,pphi,M,a,Q,L; };
static_assert(sizeof(Sample)==40,"MSL input layout");

int main(int argc,char** argv) {
    @autoreleasepool {
        if(argc!=2) return 1;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(!device) return 77;
        NSError* error=nil;
        NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
            encoding:NSUTF8StringEncoding error:&error];
        if(!source) {std::cerr<<[[error localizedDescription] UTF8String]<<'\n';return 1;}
        // Call the actual production function, not a copied CPU port.
        source=[source stringByAppendingString:@R"(
struct RHSSample { float r,theta,pr,pth,pt,pphi,M,a,Q,L; };
kernel void test_bl_rhs(device const RHSSample* samples [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    const RHSSample s=samples[i];
    float dr,dth,dphi,dpr,dpth;
    geodesic_rhs(s.r,s.theta,s.pr,s.pth,s.pt,s.pphi,s.M,s.a,s.Q,s.L,
                 dr,dth,dphi,dpr,dpth);
    out[5*i]=dr; out[5*i+1]=dth; out[5*i+2]=dphi;
    out[5*i+3]=dpr; out[5*i+4]=dpth;
}
)" ];
        MTLCompileOptions* options=[MTLCompileOptions new];
        options.languageVersion=MTLLanguageVersion2_4;
        if(@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
        else options.fastMathEnabled=NO;
        id<MTLLibrary> lib=[device newLibraryWithSource:source options:options error:&error];
        id<MTLFunction> fn=lib?[lib newFunctionWithName:@"test_bl_rhs"]:nil;
        id<MTLComputePipelineState> pipeline=fn?
            [device newComputePipelineStateWithFunction:fn error:&error]:nil;
        if(!pipeline) {std::cerr<<[[error localizedDescription] UTF8String]<<'\n';return 1;}
        std::vector<Sample> samples;
        for(int metric=0;metric<4;++metric)
            for(float r : {2.5f,4.f,12.f,40.f})
                for(float theta : {.03f,.3f,1.f,float(M_PI/2),2.6f,3.1f})
                    for(int momenta=0;momenta<3;++momenta) {
                        Sample s{r,theta,-.5f+.6f*momenta,-5.f+3.f*momenta,-1.f,
                                 -7.f+7.f*momenta,1.f,0.f,0.f,0.f};
                        if(metric==1) s.a=.7f;
                        if(metric==2) {s.a=-.7f;s.Q=.2f;s.L=1e-4f;}
                        if(metric==3) {s.a=.95f;s.Q=.1f;s.L=-1e-4f;}
                        samples.push_back(s); // off-null states occur at RK stages
                    }
        const size_t bytes=samples.size()*5*sizeof(float);
        id<MTLBuffer> input=[device newBufferWithBytes:samples.data()
            length:samples.size()*sizeof(Sample) options:MTLResourceStorageModeShared];
        id<MTLBuffer> output=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue=[device newCommandQueue];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if(!input || !output || !encoder) {std::cerr<<"Metal allocation failed\n";return 1;}
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(samples.size(),1,1)
            threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted) {
            std::cerr<<"GPU execution failed\n";return 1;
        }
        const float* values=static_cast<const float*>(output.contents);
        int failures=0;double worst=0;
        for(size_t i=0;i<samples.size();++i) {
            const auto& s=samples[i];
            KNdSMetric metric(s.M,s.a,s.Q,s.L);
            double gu[4][4];metric.contravariant_BL(s.r,s.theta,gu);
            auto H=[&](double r,double theta) {
                return metric.hamiltonian(r,theta,s.pr,s.pth,s.pt,s.pphi);
            };
            // Independent five-point differences in double precision, rather
            // than repeating the shader's analytic force expressions.
            auto derivative=[](auto f,double x,double h) {
                return (-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h);
            };
            const double expected[]={gu[1][1]*s.pr,gu[2][2]*s.pth,
                gu[3][0]*s.pt+gu[3][3]*s.pphi,
                -derivative([&](double r){return H(r,s.theta);},s.r,1e-4*s.r),
                -derivative([&](double th){return H(s.r,th);},s.theta,
                            1e-4*std::max(.01,std::abs(std::sin(double(s.theta)))))};
            for(int component=0;component<5;++component) {
                const double actual=values[5*i+component];
                const double scaled=std::abs(actual-expected[component])/(1+std::abs(expected[component]));
                worst=std::max(worst,scaled);
                if(!std::isfinite(actual) || scaled>3e-5) {
                    if(failures<8) std::cerr<<"RHS mismatch sample="<<i<<" component="<<component
                        <<" gpu="<<actual<<" reference="<<expected[component]<<'\n';
                    ++failures;
                }
            }
        }
        std::cout<<"Metal BL RHS: "<<samples.size()<<" states, "<<failures
                 <<" failed components; max scaled error="<<worst<<'\n';
        return failures?1:0;
    }
}
