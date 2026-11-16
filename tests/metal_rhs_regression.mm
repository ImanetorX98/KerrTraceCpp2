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
// Kerr-Schild Cartesian state (Lambda = 0 only, like the KS chart itself).
struct KSSample { float X,Y,Z,pT,pX,pY,pZ,M,a,Q; };
static_assert(sizeof(KSSample)==40,"MSL input layout");

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
struct KSRHSSample { float X,Y,Z,pT,pX,pY,pZ,M,a,Q; };
kernel void test_ks_rhs(device const KSRHSSample* samples [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    const KSRHSSample s=samples[i];
    float dX,dY,dZ,dpX,dpY,dpZ;
    geodesic_rhs_KS(s.X,s.Y,s.Z,s.pT,s.pX,s.pY,s.pZ,s.M,s.a,s.Q,
                    dX,dY,dZ,dpX,dpY,dpZ);
    out[6*i]=dX; out[6*i+1]=dY; out[6*i+2]=dZ;
    out[6*i+3]=dpX; out[6*i+4]=dpY; out[6*i+5]=dpZ;
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

        // ── KS chart: the force is -dH/dx in Kerr-Schild Cartesian coordinates ──
        id<MTLFunction> ksFn=[lib newFunctionWithName:@"test_ks_rhs"];
        id<MTLComputePipelineState> ksPipeline=ksFn?
            [device newComputePipelineStateWithFunction:ksFn error:&error]:nil;
        if(!ksPipeline) {std::cerr<<[[error localizedDescription] UTF8String]<<'\n';return 1;}
        std::vector<KSSample> ks;
        struct Spin { float a,Q,rmin; };
        for(const Spin sp : {Spin{0.f,0.f,2.2f},Spin{.7f,0.f,1.9f},Spin{.95f,.1f,1.5f},Spin{-.9f,0.f,1.6f}})
            for(float r : {sp.rmin,2.5f,4.f,12.f,40.f})
                for(float theta : {.03f,.3f,1.f,float(M_PI/2)-.02f,2.6f,3.1f})
                    for(float phi : {.4f,2.f})
                        for(int momenta=0;momenta<3;++momenta) {
                            double X,Y,Z;
                            KNdSMetric::BL_to_KS_spatial(r,theta,phi,sp.a,X,Y,Z);
                            // Off-null momenta as well: RK stages leave the null cone.
                            ks.push_back(KSSample{float(X),float(Y),float(Z),-1.f,
                                .3f-.4f*momenta,-.6f+.5f*momenta,.2f+.3f*momenta,1.f,sp.a,sp.Q});
                        }
        id<MTLBuffer> ksIn=[device newBufferWithBytes:ks.data()
            length:ks.size()*sizeof(KSSample) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ksOut=[device newBufferWithLength:ks.size()*6*sizeof(float)
            options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> ksCmd=[queue commandBuffer];
        id<MTLComputeCommandEncoder> ksEnc=[ksCmd computeCommandEncoder];
        if(!ksIn || !ksOut || !ksEnc) {std::cerr<<"Metal allocation failed\n";return 1;}
        [ksEnc setComputePipelineState:ksPipeline];
        [ksEnc setBuffer:ksIn offset:0 atIndex:0];
        [ksEnc setBuffer:ksOut offset:0 atIndex:1];
        [ksEnc dispatchThreads:MTLSizeMake(ks.size(),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [ksEnc endEncoding];[ksCmd commit];[ksCmd waitUntilCompleted];
        if(ksCmd.status!=MTLCommandBufferStatusCompleted) {std::cerr<<"GPU execution failed\n";return 1;}
        const float* ksValues=static_cast<const float*>(ksOut.contents);
        int ksFailures=0;double ksWorst=0;
        for(size_t i=0;i<ks.size();++i) {
            const auto& s=ks[i];
            KNdSMetric metric(s.M,s.a,s.Q,0.0);
            const double p[4]={s.pT,s.pX,s.pY,s.pZ};
            double gu[4][4];metric.contravariant_KS(0,s.X,s.Y,s.Z,true,gu);
            auto H=[&](double X,double Y,double Z){return metric.hamiltonian_KS(0,X,Y,Z,p,true);};
            auto derivative=[](auto f,double x,double h) {
                return (-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h);
            };
            const double hX=1e-4*(std::abs(double(s.X))+.1), hY=1e-4*(std::abs(double(s.Y))+.1),
                         hZ=1e-4*(std::abs(double(s.Z))+.1);
            double expected[6];
            for(int c=0;c<3;++c) expected[c]=gu[c+1][0]*p[0]+gu[c+1][1]*p[1]+gu[c+1][2]*p[2]+gu[c+1][3]*p[3];
            expected[3]=-derivative([&](double x){return H(x,s.Y,s.Z);},s.X,hX);
            expected[4]=-derivative([&](double y){return H(s.X,y,s.Z);},s.Y,hY);
            expected[5]=-derivative([&](double z){return H(s.X,s.Y,z);},s.Z,hZ);
            for(int c=0;c<6;++c) {
                const double actual=ksValues[6*i+c];
                const double scaled=std::abs(actual-expected[c])/(1+std::abs(expected[c]));
                ksWorst=std::max(ksWorst,scaled);
                if(!std::isfinite(actual) || scaled>3e-5) {
                    if(ksFailures<8) std::cerr<<"KS RHS mismatch sample="<<i<<" component="<<c
                        <<" gpu="<<actual<<" reference="<<expected[c]<<'\n';
                    ++ksFailures;
                }
            }
        }
        std::cout<<"Metal KS RHS: "<<ks.size()<<" states, "<<ksFailures
                 <<" failed components; max scaled error="<<ksWorst<<'\n';
        return (failures||ksFailures)?1:0;
    }
}
