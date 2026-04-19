// ============================================================
//  tracer.cu — CUDA geodesic ray-tracer (KNdS)
//
//  The metric / geodesic logic is ported from our C++ headers.
//  We use double precision throughout (matching the CPU path).
//
//  Compile:  nvcc -O3 -arch=sm_86 -std=c++17 tracer.cu -o tracer_cuda.o
//  (or let CMake handle it via enable_language(CUDA))
// ============================================================
#include "tracer.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <cstring>

// ── CUDA error helper ─────────────────────────────────────────
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(cudaGetErrorString(_e)); \
} while(0)

// ── Device-side KNdS metric ───────────────────────────────────
__device__ double d_Sigma(double r, double theta, double a) {
    double ct = cos(theta); return r*r + a*a*ct*ct;
}
__device__ double d_Delta_r(double r, double M, double a, double Q, double L) {
    return (r*r+a*a)*(1.0-L*r*r/3.0) - 2.0*M*r + Q*Q;
}
__device__ double d_Delta_th(double theta, double a, double L) {
    double ct = cos(theta); return 1.0 + L*a*a*ct*ct/3.0;
}
__device__ double d_Xi(double a, double L) { return 1.0 + L*a*a/3.0; }

__device__ double d_keplerian_omega(double r, double M, double a, double Q, double L) {
    const double Meff = M - Q*Q/(2.0*r) + L*a*r*r/3.0;
    const double sq   = sqrt(fmax(Meff, 0.0));
    const double den  = r*sqrt(r) + a*sq;
    return (fabs(den) > 1e-14) ? (sq/den) : 0.0;
}

__device__ void d_gUU(double r, double theta,
                      double M, double a, double Q, double L,
                      double gu[4][4]) {
    double sig  = d_Sigma(r, theta, a);
    double dr   = d_Delta_r(r, M, a, Q, L);
    double dth  = d_Delta_th(theta, a, L);
    double xi2  = d_Xi(a,L)*d_Xi(a,L);
    double st2  = sin(theta); st2 *= st2;
    double r2a2 = r*r + a*a;
    double pre  = sig*dr*dth;
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) gu[i][j]=0.0;
    gu[0][0] = -xi2*(dth*r2a2*r2a2 - dr*a*a*st2) / pre;
    gu[0][3] = gu[3][0] = a*xi2*(dr - dth*r2a2) / pre;
    gu[1][1] = dr/sig;
    gu[2][2] = dth/sig;
    if(st2 > 1e-14) gu[3][3] = xi2*(dr - dth*a*a*st2)/(pre*st2);
}

__device__ double d_H(double r, double theta,
                      double pr, double pth,
                      double pt, double pphi,
                      double M, double a, double Q, double L) {
    double gu[4][4]; d_gUU(r, theta, M, a, Q, L, gu);
    return 0.5*(gu[0][0]*pt*pt + 2.0*gu[0][3]*pt*pphi
              + gu[1][1]*pr*pr + gu[2][2]*pth*pth + gu[3][3]*pphi*pphi);
}

__device__ void d_rhs(double r, double theta, double pr, double pth,
                      double pt, double pphi, double M, double a, double Q, double L,
                      double& dr, double& dth, double& dpr, double& dpth) {
    double gu[4][4]; d_gUU(r, theta, M, a, Q, L, gu);
    dr  = gu[1][1]*pr;
    dth = gu[2][2]*pth;
    double er  = 1e-5*(fabs(r)+0.1);
    double eth = 1e-6;
    dpr  = -(d_H(r+er,theta,pr,pth,pt,pphi,M,a,Q,L)
           - d_H(r-er,theta,pr,pth,pt,pphi,M,a,Q,L))/(2.0*er);
    dpth = -(d_H(r,theta+eth,pr,pth,pt,pphi,M,a,Q,L)
           - d_H(r,theta-eth,pr,pth,pt,pphi,M,a,Q,L))/(2.0*eth);
}

__device__ void d_rk4(double& r, double& theta, double& pr, double& pth,
                      double pt, double pphi, double M, double a, double Q, double L,
                      double dlam) {
    double dr1,dth1,dpr1,dpth1, dr2,dth2,dpr2,dpth2;
    double dr3,dth3,dpr3,dpth3, dr4,dth4,dpr4,dpth4;
    d_rhs(r,           theta,           pr,           pth,           pt,pphi,M,a,Q,L,dr1,dth1,dpr1,dpth1);
    d_rhs(r+.5*dlam*dr1,theta+.5*dlam*dth1,pr+.5*dlam*dpr1,pth+.5*dlam*dpth1,pt,pphi,M,a,Q,L,dr2,dth2,dpr2,dpth2);
    d_rhs(r+.5*dlam*dr2,theta+.5*dlam*dth2,pr+.5*dlam*dpr2,pth+.5*dlam*dpth2,pt,pphi,M,a,Q,L,dr3,dth3,dpr3,dpth3);
    d_rhs(r+   dlam*dr3,theta+   dlam*dth3,pr+   dlam*dpr3,pth+   dlam*dpth3,pt,pphi,M,a,Q,L,dr4,dth4,dpr4,dpth4);
    r    +=dlam/6.0*(dr1  +2*dr2  +2*dr3  +dr4);
    theta+=dlam/6.0*(dth1 +2*dth2 +2*dth3 +dth4);
    pr   +=dlam/6.0*(dpr1 +2*dpr2 +2*dpr3 +dpr4);
    pth  +=dlam/6.0*(dpth1+2*dpth2+2*dpth3+dpth4);
}

__device__ void d_gLL(double r, double theta,
                      double M, double a, double Q, double L,
                      double gl[4][4]) {
    double sig  = d_Sigma(r, theta, a);
    double dr   = d_Delta_r(r, M, a, Q, L);
    double dth  = d_Delta_th(theta, a, L);
    double xi2  = d_Xi(a,L)*d_Xi(a,L);
    double st   = sin(theta), st2 = st*st;
    double r2a2 = r*r + a*a;
    double pre  = sig*xi2;
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) gl[i][j]=0.0;
    gl[0][0] = (-dr + dth*a*a*st2)/pre;
    gl[0][3] = gl[3][0] = a*st2*(dr - dth*r2a2)/pre;
    gl[1][1] = sig/dr; gl[2][2] = sig/dth;
    gl[3][3] = st2*(dth*r2a2*r2a2 - dr*a*a*st2)/pre;
}

// ── Colour ────────────────────────────────────────────────────
__device__ uchar4 d_blackbody(double T) {
    T = fmax(800.0, fmin(4e4, T));
    double t = log10(T/800.0)/log10(4e4/800.0);
    double R,G,B;
    if(t<0.25){R=1;G=t/0.25*0.4;B=0;}
    else if(t<0.5){double f=(t-0.25)/0.25;R=1;G=0.4+f*0.4;B=f*0.3;}
    else if(t<0.75){double f=(t-0.5)/0.25;R=1;G=0.8+f*0.2;B=0.3+f*0.5;}
    else{double f=(t-0.75)/0.25;R=1-f*0.2;G=1;B=0.8+f*0.2;}
    return {(unsigned char)(fmin(R,1.0)*255),
            (unsigned char)(fmin(G,1.0)*255),
            (unsigned char)(fmin(B,1.0)*255), 255};
}

// ── Main kernel (one thread = one pixel) ─────────────────────
__global__ void trace_kernel(uint32_t*              output,
                              const KNdSParams_CUDA  kp,
                              const CameraParams_CUDA cp) {
    const int px = blockIdx.x*blockDim.x + threadIdx.x;
    const int py = blockIdx.y*blockDim.y + threadIdx.y;
    if(px >= cp.width || py >= cp.height) return;

    double M=kp.M, a=kp.a, Q=kp.Q, L=kp.Lambda;

    int span = (cp.width > 1) ? (cp.width - 1) : 1;
    double alpha = cp.fov_h*(px - 0.5*(cp.width-1)) / span;
    double beta  = cp.fov_h*(0.5*(cp.height-1) - py) / span;

    double r0=cp.r_obs, th0=cp.theta_obs;
    double gl[4][4]; d_gLL(r0, th0, M, a, Q, L, gl);
    double gu[4][4]; d_gUU(r0, th0, M, a, Q, L, gu);

    double ca=cos(alpha),sa=sin(alpha),cb=cos(beta),sb=sin(beta);
    double pUr   =-ca*cb/sqrt(fabs(gl[1][1]));
    double pUth  =-sb    /sqrt(fabs(gl[2][2]));
    double pUphi = sa*cb /sqrt(fabs(gl[3][3]));
    double pt    = gl[0][0]+gl[0][3]*pUphi;
    double pr    = gl[1][1]*pUr;
    double pth   = gl[2][2]*pUth;
    double pphi  = gl[3][0]+gl[3][3]*pUphi;

    // Null correction
    double A=gu[0][0], B=2.0*gu[0][3]*pphi;
    double C=gu[1][1]*pr*pr+gu[2][2]*pth*pth+gu[3][3]*pphi*pphi;
    double disc=B*B-4.0*A*C;
    if(disc>=0.0 && fabs(A)>1e-15){
        double sq=sqrt(disc);
        double pt1=(-B-sq)/(2.0*A), pt2=(-B+sq)/(2.0*A);
        pt=(pt1<0.0)?pt1:pt2;
        if(pt>0.0) pt=fmin(pt1,pt2);
    }

    double r=r0, theta=th0, dlam=1.0;
    double prev_cos=cos(th0);
    uint32_t colour=0xFF000000u;

    for(int iter=0; iter<200000; ++iter) {
        // Step-doubling
        double rh=r,thh=theta,prh=pr,pthh=pth;
        d_rk4(rh,thh,prh,pthh,pt,pphi,M,a,Q,L,dlam);
        double rf=r,thf=theta,prf=pr,pthf=pth;
        d_rk4(rf,thf,prf,pthf,pt,pphi,M,a,Q,L,dlam*0.5);
        d_rk4(rf,thf,prf,pthf,pt,pphi,M,a,Q,L,dlam*0.5);
        double dr_=rh-rf,dth_=thh-thf,dpr_=prh-prf,dpth_=pthh-pthf;
        double err=sqrt(dr_*dr_+dth_*dth_+dpr_*dpr_+dpth_*dpth_)/15.0;
        const double tol=1e-7;
        if(err<tol||dlam<1e-10){
            r=rf;theta=thf;pr=prf;pth=pthf;
            double sc=(err>1e-12)?0.9*pow(tol/err,0.2):2.0;
            dlam=fmin(fmax(dlam*sc,1e-10),100.0);
        } else {
            dlam=fmax(dlam*0.9*pow(tol/err,0.25),1e-10);
            continue;
        }

        if(r < kp.r_horizon*1.03) break;
        if(r > cp.r_obs*1.05)     break;

        double cos_th=cos(theta);
        if(prev_cos*cos_th<=0.0 && r>=kp.r_isco && r<=kp.r_disk_out){
            double Omega=d_keplerian_omega(r, M, a, Q, L);
            double b_=pphi/(-pt);
            double gl2[4][4]; d_gLL(r, M_PI/2.0, M, a, Q, L, gl2);
            double d2=-(gl2[0][0]+2.0*gl2[0][3]*Omega+gl2[3][3]*Omega*Omega);
            double dv=1.0-Omega*b_;
            double red=(d2>0.0&&fabs(dv)>1e-10)?sqrt(d2)/dv:1.0;
            red=fmax(0.0,fmin(20.0,red));

            double T0=2e6*pow(r/(6.0*M),-0.75);
            double T =T0*fmax(0.1,fmin(10.0,red));
            double I =fmax(0.0,fmin(2.5,pow(red,4.0)*pow(6.0*M/r,3.0)));

            uchar4 c=d_blackbody(T);
            double R=fmin(c.x*I/255.0,1.0)*255;
            double G=fmin(c.y*I/255.0,1.0)*255;
            double B2=fmin(c.z*I/255.0,1.0)*255;
            colour=(0xFFu<<24)|((uint32_t)B2<<16)|((uint32_t)G<<8)|(uint32_t)R;
            break;
        }
        prev_cos=cos_th;
    }
    output[py*cp.width+px]=colour;
}

// ── Host-side launcher ────────────────────────────────────────
std::vector<uint32_t> cuda_render(
    const KNdSParams_CUDA&  kp,
    const CameraParams_CUDA& cp)
{
    const size_t npix = (size_t)cp.width * cp.height;

    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, npix * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_out, 0, npix * sizeof(uint32_t)));

    dim3 block(16, 16);
    dim3 grid(((unsigned)cp.width+15)/16, ((unsigned)cp.height+15)/16);
    trace_kernel<<<grid, block>>>(d_out, kp, cp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> pixels(npix);
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_out, npix*sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    return pixels;
}
