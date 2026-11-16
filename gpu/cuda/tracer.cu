// ============================================================
//  tracer.cu — CUDA geodesic ray-tracer (KNdS)
//
//  The metric / geodesic logic is ported from our C++ headers.
//  We use double precision throughout (matching the CPU path).
//
//  Compile (manual): nvcc -O3 -std=c++17 tracer.cu -o tracer_cuda.o
//  (recommended: let CMake handle CUDA architectures/toolchain)
// ============================================================
#include "tracer.cuh"
#ifndef KERRTRACE_CUDA_HOST_TEST
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#else
#define __device__ inline
#endif
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cstring>

#ifdef KERRTRACE_CUDA_HOST_TEST
using std::isfinite;
inline double rsqrt(double x) { return 1.0/std::sqrt(x); }
#endif

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
    (void)Q; (void)L;
    const double sign = (a < 0.0) ? -1.0 : 1.0;
    return sign * sqrt(M) / (r*sqrt(r) + fabs(a)*sqrt(M));
}

__device__ void d_gLL(double r, double theta,
                      double M, double a, double Q, double L,
                      double gl[4][4]);

__device__ double d_robust_disk_redshift(double r_hit,
                                         double pt_cov,
                                         double pphi_cov,
                                         double M, double a, double Q, double L, double observer_ut) {
    // Frequency-shift factor used by colour pipeline.
    constexpr double kRadicandFloor = 1.0e-8;
    constexpr double kDenomFloor  = 1.0e-8;
    constexpr double kRedMax  = 6.0;
    const double Omega = d_keplerian_omega(r_hit, M, a, Q, L);
    double gl2[4][4];
    d_gLL(r_hit, M_PI/2.0, M, a, Q, L, gl2);
    const double d2 = -(gl2[0][0] + 2.0*gl2[0][3]*Omega + gl2[3][3]*Omega*Omega);
    const double ut = rsqrt(fmax(d2, kRadicandFloor));
    const double denom = -(pt_cov * ut + pphi_cov * Omega * ut);
    const double denom_safe = fmax(denom, kDenomFloor);
    const double g_factor = (-pt_cov) * observer_ut / denom_safe;
    if (!isfinite(g_factor)) return 0.0;
    return fmax(0.0, fmin(kRedMax, g_factor));
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
    double er  = 1e-5*fmax(fabs(r),0.01);
    double eth = 1e-6;
    dpr  = -(d_H(r+er,theta,pr,pth,pt,pphi,M,a,Q,L)
           - d_H(r-er,theta,pr,pth,pt,pphi,M,a,Q,L))/(2.0*er);
    dpth = -(d_H(r,theta+eth,pr,pth,pt,pphi,M,a,Q,L)
           - d_H(r,theta-eth,pr,pth,pt,pphi,M,a,Q,L))/(2.0*eth);
}

__device__ double d_phi_velocity(double r, double th, double pt, double pp,
                                        double M, double a, double Q, double L) {
    double gu[4][4]; d_gUU(r,th,M,a,Q,L,gu);
    return gu[3][0]*pt + gu[3][3]*pp;
}

__device__ void d_rk4(double& r, double& theta, double& phi, double& pr, double& pth,
                      double pt, double pphi, double M, double a, double Q, double L,
                      double dlam) {
    double dr1,dth1,dpr1,dpth1, dr2,dth2,dpr2,dpth2;
    double dr3,dth3,dpr3,dpth3, dr4,dth4,dpr4,dpth4;
    d_rhs(r,           theta,           pr,           pth,           pt,pphi,M,a,Q,L,dr1,dth1,dpr1,dpth1);
    d_rhs(r+.5*dlam*dr1,theta+.5*dlam*dth1,pr+.5*dlam*dpr1,pth+.5*dlam*dpth1,pt,pphi,M,a,Q,L,dr2,dth2,dpr2,dpth2);
    d_rhs(r+.5*dlam*dr2,theta+.5*dlam*dth2,pr+.5*dlam*dpr2,pth+.5*dlam*dpth2,pt,pphi,M,a,Q,L,dr3,dth3,dpr3,dpth3);
    d_rhs(r+   dlam*dr3,theta+   dlam*dth3,pr+   dlam*dpr3,pth+   dlam*dpth3,pt,pphi,M,a,Q,L,dr4,dth4,dpr4,dpth4);
    phi += dlam/6.0 * (
        d_phi_velocity(r,theta,pt,pphi,M,a,Q,L)
        + 2.0*d_phi_velocity(r+.5*dlam*dr1,theta+.5*dlam*dth1,pt,pphi,M,a,Q,L)
        + 2.0*d_phi_velocity(r+.5*dlam*dr2,theta+.5*dlam*dth2,pt,pphi,M,a,Q,L)
        + d_phi_velocity(r+dlam*dr3,theta+dlam*dth3,pt,pphi,M,a,Q,L));
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

// ── Kerr-Schild (Lambda=0) helpers ───────────────────────────
__device__ double d_r_KS(double X, double Y, double Z, double a) {
    const double R2 = X*X + Y*Y + Z*Z;
    const double a2 = a*a;
    const double b  = R2 - a2;
    return sqrt(0.5*(b + sqrt(b*b + 4.0*a2*Z*Z)));
}

__device__ void d_BL_to_KS_spatial(double r, double theta, double phi, double a,
                                   double& X, double& Y, double& Z) {
    const double st = sin(theta), ct = cos(theta);
    const double sf = sin(phi),   cf = cos(phi);
    X = st * (r*cf - a*sf);
    Y = st * (r*sf + a*cf);
    Z = r * ct;
}

__device__ void d_KS_to_BL_spatial(double X, double Y, double Z, double a,
                                   double& r, double& theta, double& phi) {
    r = d_r_KS(X, Y, Z, a);
    const double Zr = (fabs(r) > 1e-12) ? (Z / r) : 1.0;
    theta = acos(fmax(-1.0, fmin(1.0, Zr)));
    const double r2 = fmax(r*r, 1e-16);
    const double st = sqrt(fmax(1.0 - (Z*Z)/r2, 0.0));
    const double r2a2 = r*r + a*a;
    if (st > 1e-12 && r2a2 > 1e-12) {
        const double cf = (X*r + Y*a) / (st * r2a2);
        const double sf = (Y*r - X*a) / (st * r2a2);
        phi = atan2(sf, cf);
    } else {
        phi = 0.0;
    }
}

__device__ void d_jacobian_BL_to_KS(double r, double theta, double phi, double a,
                                    double J[3][3]) {
    const double st = sin(theta), ct = cos(theta);
    const double sf = sin(phi),   cf = cos(phi);
    const double rcfa = r*cf - a*sf;
    const double rsfa = r*sf + a*cf;

    // Columns: (dr, dtheta, dphi), rows: (X,Y,Z)
    J[0][0] = st * cf;
    J[1][0] = st * sf;
    J[2][0] = ct;

    J[0][1] = ct * rcfa;
    J[1][1] = ct * rsfa;
    J[2][1] = -r * st;

    J[0][2] = -st * rsfa;
    J[1][2] =  st * rcfa;
    J[2][2] = 0.0;
}

__device__ bool d_solve3x3(double A[3][3], double b[3], double x[3]) {
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        double best = fabs(A[piv][col]);
        for (int r = col + 1; r < 3; ++r) {
            const double v = fabs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-16) return false;

        if (piv != col) {
            for (int k = col; k < 3; ++k) {
                const double tmp = A[col][k];
                A[col][k] = A[piv][k];
                A[piv][k] = tmp;
            }
            const double tb = b[col];
            b[col] = b[piv];
            b[piv] = tb;
        }

        const double inv = 1.0 / A[col][col];
        for (int k = col; k < 3; ++k) A[col][k] *= inv;
        b[col] *= inv;

        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = A[r][col];
            for (int k = col; k < 3; ++k) A[r][k] -= f * A[col][k];
            b[r] -= f * b[col];
        }
    }
    x[0] = b[0]; x[1] = b[1]; x[2] = b[2];
    return true;
}

__device__ bool d_BL_covector_to_KS(double r, double theta, double phi, double a,
                                    double pr, double ptheta, double pphi,
                                    double pt, double M, double Q,
                                    double& pX, double& pY, double& pZ) {
    double J[3][3];
    d_jacobian_BL_to_KS(r, theta, phi, a, J);

    double A[3][3];
    // A = J^T, solve A * p_xyz = p_bl
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            A[j][i] = J[i][j];

    const double delta = r*r + a*a - 2.0*M*r + Q*Q;
    const double shifted_pr = pr - ((2.0*M*r-Q*Q)*pt + a*pphi)/delta;
    double b[3] = {shifted_pr, ptheta, pphi};
    double x[3];
    if (!d_solve3x3(A, b, x)) return false;
    pX = x[0]; pY = x[1]; pZ = x[2];
    return true;
}

__device__ void d_KS_covector_to_BL(double r, double theta, double phi, double a,
                                    double pX, double pY, double pZ,
                                    double pt, double M, double Q,
                                    double& pr, double& ptheta, double& pphi) {
    double J[3][3];
    d_jacobian_BL_to_KS(r, theta, phi, a, J);
    const double pxyz[3] = {pX, pY, pZ};
    pr = ptheta = pphi = 0.0;
    for (int j = 0; j < 3; ++j) {
        double s = 0.0;
        for (int i = 0; i < 3; ++i) s += J[i][j] * pxyz[i];
        if (j == 0) pr = s;
        if (j == 1) ptheta = s;
        if (j == 2) pphi = s;
    }
    const double delta = r*r + a*a - 2.0*M*r + Q*Q;
    pr += ((2.0*M*r-Q*Q)*pt + a*pphi)/delta;
}

__device__ void d_gUU_KS(double X, double Y, double Z, double M, double a, double Q,
                         double guu[4][4]) {
    const double r = d_r_KS(X, Y, Z, a);
    const double rr = fmax(r*r, 1e-16);
    const double rho2 = rr + a*a*Z*Z/rr;
    const double H = (2.0*M*r - Q*Q) / rho2;
    const double r2a2 = rr + a*a;

    // Ingoing KS null covector
    const double l0 = 1.0;
    const double l1 = (r*X + a*Y) / r2a2;
    const double l2 = (r*Y - a*X) / r2a2;
    const double l3 = Z / fmax(r, 1e-10);

    const double lU[4] = {-l0, l1, l2, l3};
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) guu[i][j]=0.0;
    guu[0][0]=-1.0; guu[1][1]=1.0; guu[2][2]=1.0; guu[3][3]=1.0;
    for (int mu=0; mu<4; ++mu)
        for (int nu=0; nu<4; ++nu)
            guu[mu][nu] -= H * lU[mu] * lU[nu];
}

__device__ double d_H_KS(double X, double Y, double Z,
                         double pT, double pX, double pY, double pZ,
                         double M, double a, double Q) {
    double guu[4][4];
    d_gUU_KS(X, Y, Z, M, a, Q, guu);
    return 0.5 * (
        guu[0][0]*pT*pT + 2.0*guu[0][1]*pT*pX + 2.0*guu[0][2]*pT*pY + 2.0*guu[0][3]*pT*pZ +
        guu[1][1]*pX*pX + guu[2][2]*pY*pY + guu[3][3]*pZ*pZ +
        2.0*guu[1][2]*pX*pY + 2.0*guu[1][3]*pX*pZ + 2.0*guu[2][3]*pY*pZ
    );
}

__device__ void d_rhs_KS(double X, double Y, double Z,
                         double pT, double pX, double pY, double pZ,
                         double M, double a, double Q,
                         double& dX, double& dY, double& dZ,
                         double& dpX, double& dpY, double& dpZ) {
    double guu[4][4];
    d_gUU_KS(X, Y, Z, M, a, Q, guu);
    dX = guu[1][0]*pT + guu[1][1]*pX + guu[1][2]*pY + guu[1][3]*pZ;
    dY = guu[2][0]*pT + guu[2][1]*pX + guu[2][2]*pY + guu[2][3]*pZ;
    dZ = guu[3][0]*pT + guu[3][1]*pX + guu[3][2]*pY + guu[3][3]*pZ;

    const double eX = 1e-5*(fabs(X)+0.1);
    const double eY = 1e-5*(fabs(Y)+0.1);
    const double eZ = 1e-5*(fabs(Z)+0.1);
    dpX = -(d_H_KS(X+eX, Y, Z, pT, pX, pY, pZ, M, a, Q)
          - d_H_KS(X-eX, Y, Z, pT, pX, pY, pZ, M, a, Q)) / (2.0*eX);
    dpY = -(d_H_KS(X, Y+eY, Z, pT, pX, pY, pZ, M, a, Q)
          - d_H_KS(X, Y-eY, Z, pT, pX, pY, pZ, M, a, Q)) / (2.0*eY);
    dpZ = -(d_H_KS(X, Y, Z+eZ, pT, pX, pY, pZ, M, a, Q)
          - d_H_KS(X, Y, Z-eZ, pT, pX, pY, pZ, M, a, Q)) / (2.0*eZ);
}

__device__ void d_rk4_KS(double& X, double& Y, double& Z,
                         double& pX, double& pY, double& pZ,
                         double pT, double M, double a, double Q, double dlam) {
    double dX1,dY1,dZ1,dpX1,dpY1,dpZ1;
    double dX2,dY2,dZ2,dpX2,dpY2,dpZ2;
    double dX3,dY3,dZ3,dpX3,dpY3,dpZ3;
    double dX4,dY4,dZ4,dpX4,dpY4,dpZ4;
    d_rhs_KS(X, Y, Z, pT, pX, pY, pZ, M, a, Q, dX1,dY1,dZ1,dpX1,dpY1,dpZ1);
    d_rhs_KS(X+0.5*dlam*dX1, Y+0.5*dlam*dY1, Z+0.5*dlam*dZ1,
             pT, pX+0.5*dlam*dpX1, pY+0.5*dlam*dpY1, pZ+0.5*dlam*dpZ1,
             M, a, Q, dX2,dY2,dZ2,dpX2,dpY2,dpZ2);
    d_rhs_KS(X+0.5*dlam*dX2, Y+0.5*dlam*dY2, Z+0.5*dlam*dZ2,
             pT, pX+0.5*dlam*dpX2, pY+0.5*dlam*dpY2, pZ+0.5*dlam*dpZ2,
             M, a, Q, dX3,dY3,dZ3,dpX3,dpY3,dpZ3);
    d_rhs_KS(X+dlam*dX3, Y+dlam*dY3, Z+dlam*dZ3,
             pT, pX+dlam*dpX3, pY+dlam*dpY3, pZ+dlam*dpZ3,
             M, a, Q, dX4,dY4,dZ4,dpX4,dpY4,dpZ4);

    X  += dlam/6.0*(dX1  +2.0*dX2  +2.0*dX3  +dX4);
    Y  += dlam/6.0*(dY1  +2.0*dY2  +2.0*dY3  +dY4);
    Z  += dlam/6.0*(dZ1  +2.0*dZ2  +2.0*dZ3  +dZ4);
    pX += dlam/6.0*(dpX1 +2.0*dpX2 +2.0*dpX3 +dpX4);
    pY += dlam/6.0*(dpY1 +2.0*dpY2 +2.0*dpY3 +dpY4);
    pZ += dlam/6.0*(dpZ1 +2.0*dpZ2 +2.0*dpZ3 +dpZ4);
}

// The same function is exercised by CPU CI and called by the CUDA kernel.
// No shading here: CUDA hands geometry to the shared colorize_buffer().
__device__ double d_twist(double r, double a, double M, double Q) {
    if (fabs(a) < 1e-15) return 0.0;
    const double disc = M*M-a*a-Q*Q;
    if (disc > 1e-14*M*M) {
        const double root=sqrt(disc), rp=M+root, rm=M-root;
        return a/(2.0*root)*log(fabs((r-rp)/(r-rm)));
    }
    return -a/(r-M);
}

__device__ double d_hermite(double y0,double y1,double f0,double f1,double h,double t) {
    double t2=t*t,t3=t2*t;
    return (2*t3-3*t2+1)*y0+(t3-2*t2+t)*h*f0+(-2*t3+3*t2)*y1+(t3-t2)*h*f1;
}
__device__ double d_crossing(double y0,double y1,double f0,double f1,double h,int mode) {
    if (mode==0) return y0*y1<=0.0 ? fabs(y0)/(fabs(y0)+fabs(y1)+1e-12) : -1.0;
    double prev=y0,lo=0.0;
    if (fabs(prev)<=1e-12) return 0.0;
    for(int b=1;b<=8;++b) {
        double hi=double(b)/8.0,cur=d_hermite(y0,y1,f0,f1,h,hi);
        if(fabs(cur)<=1e-12) return hi;
        if(prev*cur<=0.0) {
            for(int k=0;k<8;++k) {
                double mid=(lo+hi)*.5,val=d_hermite(y0,y1,f0,f1,h,mid);
                if(prev*val<=0.0) hi=mid; else lo=mid;
            }
            return .5*(lo+hi);
        }
        lo=hi;prev=cur;
    }
    return -1.0;
}

__device__ GeoPixel d_trace_one(int px,int py,KNdSParams_CUDA kp,CameraParams_CUDA cp) {
    GeoPixel result{};
    result.redshift=1.0f; result.magnif=1.0f; result.coverage=1.0f;result._pad[1]=2;
    const double M=kp.M,a=kp.a,Q=kp.Q,L=kp.Lambda;
    const double span=cp.width>1?cp.width-1:1;
    const double alpha=cp.fov_h*(px+cp.pixel_offset_x-.5*(cp.width-1))/span;
    const double beta=cp.fov_h*(.5*(cp.height-1)-py-cp.pixel_offset_y)/span;
    double gl[4][4];d_gLL(cp.r_obs,cp.theta_obs,M,a,Q,L,gl);
    const double obs_ut=rsqrt(-gl[0][0]);
    const double nph=-sin(alpha)*cos(beta);
    const double ep=rsqrt(gl[3][3]-gl[0][3]*gl[0][3]/gl[0][0]);
    const double put=obs_ut-nph*gl[0][3]/gl[0][0]*ep;
    const double puphi=nph*ep;
    const double pt=gl[0][0]*put+gl[0][3]*puphi;
    const double pp=gl[3][0]*put+gl[3][3]*puphi;
    double r=cp.r_obs,th=cp.theta_obs,ph=cp.phi_obs;
    double pr=-cos(alpha)*cos(beta)*sqrt(gl[1][1]),pth=-sin(beta)*sqrt(gl[2][2]);
    double h=cp.step_init;
    const double tol=cp.tolerance,rh=kp.r_horizon*1.03,re=cp.r_obs*1.05;
    if(cp.chart==1) {
        double X,Y,Z,pX,pY,pZ;
        const double phi_ks=ph+d_twist(r,a,M,Q);
        d_BL_to_KS_spatial(r,th,phi_ks,a,X,Y,Z);
        if(!d_BL_covector_to_KS(r,th,phi_ks,a,pr,pth,pp,pt,M,Q,pX,pY,pZ)) return result;
        int rejects=0;
        for(int step=0;step<cp.max_steps;++step) {
            const double x0=X,y0=Y,z0=Z,px0=pX,py0=pY,pz0=pZ,r0=d_r_KS(X,Y,Z,a),used=h;
            double xh=X,yh=Y,zh=Z,pxh=pX,pyh=pY,pzh=pZ;
            d_rk4_KS(xh,yh,zh,pxh,pyh,pzh,pt,M,a,Q,h);
            double xf=X,yf=Y,zf=Z,pxf=pX,pyf=pY,pzf=pZ;
            d_rk4_KS(xf,yf,zf,pxf,pyf,pzf,pt,M,a,Q,h*.5);
            d_rk4_KS(xf,yf,zf,pxf,pyf,pzf,pt,M,a,Q,h*.5);
            const double err=sqrt((xh-xf)*(xh-xf)+(yh-yf)*(yh-yf)+(zh-zf)*(zh-zf)
                +(pxh-pxf)*(pxh-pxf)+(pyh-pyf)*(pyh-pyf)+(pzh-pzf)*(pzh-pzf))/15.0;
            if(!isfinite(err) || !(err<tol || h<1e-10)) {
                h=fmax(1e-10,isfinite(err)?fmin(h*.5,h*.9*pow(tol/err,.25)):h*.5);
                if(++rejects>64) break; --step;continue;
            }
            rejects=0;X=xf;Y=yf;Z=zf;pX=pxf;pY=pyf;pZ=pzf;
            h=fmax(1e-10,fmin(100.0,h*(err>1e-14?.9*pow(tol/err,.2):4.0)));
            double rnow=d_r_KS(X,Y,Z,a),event=2.0;int outcome=-1;
            if(rnow<=rh) {event=fmax(0.0,fmin(1.0,(r0-rh)/(r0-rnow)));outcome=2;}
            if(rnow>=re) {double t=fmax(0.0,fmin(1.0,(re-r0)/(rnow-r0)));if(t<event){event=t;outcome=0;}}
            if(z0*Z<=0.0 || fmin(fabs(z0),fabs(Z))<.35) {
                double dx0,dy0,dz0,dpx0,dpy0,dpz0,dx1,dy1,dz1,dpx1,dpy1,dpz1;
                d_rhs_KS(x0,y0,z0,pt,px0,py0,pz0,M,a,Q,dx0,dy0,dz0,dpx0,dpy0,dpz0);
                d_rhs_KS(X,Y,Z,pt,pX,pY,pZ,M,a,Q,dx1,dy1,dz1,dpx1,dpy1,dpz1);
                double t=d_crossing(z0,Z,dz0,dz1,used,cp.intersection_mode);
                if(t>=0.0 && t<event) {
                    double xx=d_hermite(x0,X,dx0,dx1,used,t),yy=d_hermite(y0,Y,dy0,dy1,used,t),zz=d_hermite(z0,Z,dz0,dz1,used,t);
                    if(cp.intersection_mode==0) {xx=x0+t*(X-x0);yy=y0+t*(Y-y0);zz=z0+t*(Z-z0);}
                    double rr,tt,ff;d_KS_to_BL_spatial(xx,yy,zz,a,rr,tt,ff);
                    if(rr>=kp.r_isco && rr<=kp.r_disk_out) {
                        result.outcome=1;result.r=float(rr);result.phi_disk=float(ff-d_twist(rr,a,M,Q));
                        result.redshift=float(d_robust_disk_redshift(rr,pt,pp,M,a,Q,L,obs_ut));return result;
                    }
                }
            }
            if(outcome>=0) {
                X=x0+event*(X-x0);Y=y0+event*(Y-y0);Z=z0+event*(Z-z0);
                result.outcome=uint8_t(outcome);break;
            }
        }
        d_KS_to_BL_spatial(X,Y,Z,a,r,th,ph);ph-=d_twist(r,a,M,Q);
    } else {
        int rejects=0;
        for(int step=0;step<cp.max_steps;++step) {
            double r0=r,t0=th,f0=ph,pr0=pr,pt0=pth,used=h;
            double ra=r,ta=th,fa=ph,pra=pr,pta=pth;
            d_rk4(ra,ta,fa,pra,pta,pt,pp,M,a,Q,L,h);
            double rb=r,tb=th,fb=ph,prb=pr,ptb=pth;
            d_rk4(rb,tb,fb,prb,ptb,pt,pp,M,a,Q,L,h*.5);
            d_rk4(rb,tb,fb,prb,ptb,pt,pp,M,a,Q,L,h*.5);
            double err=sqrt((ra-rb)*(ra-rb)+(ta-tb)*(ta-tb)+(pra-prb)*(pra-prb)+(pta-ptb)*(pta-ptb))/15.0;
            if(!isfinite(err) || !(err<tol || h<1e-10)) {
                h=fmax(1e-10,isfinite(err)?fmin(h*.5,h*.9*pow(tol/err,.25)):h*.5);
                if(++rejects>64) break;--step;continue;
            }
            rejects=0;r=rb;th=tb;ph=fb;pr=prb;pth=ptb;
            h=fmax(1e-10,fmin(100.0,h*(err>1e-14?.9*pow(tol/err,.2):4.0)));
            double event=2.0;int outcome=-1;
            if(r<=rh){event=fmax(0.0,fmin(1.0,(r0-rh)/(r0-r)));outcome=2;}
            if(r>=re){double t=fmax(0.0,fmin(1.0,(re-r0)/(r-r0)));if(t<event){event=t;outcome=0;}}
            if((t0-M_PI/2)*(th-M_PI/2)<=0.0 || fmin(fabs(t0-M_PI/2),fabs(th-M_PI/2))<.35) {
                double dr0,dt0,dpr0,dpt0,dr1,dt1,dpr1,dpt1;
                d_rhs(r0,t0,pr0,pt0,pt,pp,M,a,Q,L,dr0,dt0,dpr0,dpt0);
                d_rhs(r,th,pr,pth,pt,pp,M,a,Q,L,dr1,dt1,dpr1,dpt1);
                double t=d_crossing(t0-M_PI/2,th-M_PI/2,dt0,dt1,used,cp.intersection_mode);
                if(t>=0.0 && t<event) {
                    double rr=cp.intersection_mode?d_hermite(r0,r,dr0,dr1,used,t):r0+t*(r-r0);
                    if(rr>=kp.r_isco && rr<=kp.r_disk_out) {
                        result.outcome=1;result.r=float(rr);result.phi_disk=float(f0+t*(ph-f0));
                        result.redshift=float(d_robust_disk_redshift(rr,pt,pp,M,a,Q,L,obs_ut));return result;
                    }
                }
            }
            if(outcome>=0){r=r0+event*(r-r0);th=t0+event*(th-t0);ph=f0+event*(ph-f0);result.outcome=uint8_t(outcome);break;}
        }
    }
    result.r=float(r);result.theta_esc=float(th);result.phi_esc=float(ph);
    if(result.outcome==2)result.redshift=0.0f;
    return result;
}

#ifndef KERRTRACE_CUDA_HOST_TEST
__global__ void trace_kernel(GeoPixel* output,KNdSParams_CUDA kp,CameraParams_CUDA cp) {
    int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;
    if(px<cp.width && py<cp.height)output[py*cp.width+px]=d_trace_one(px,py,kp,cp);
}

std::vector<GeoPixel> cuda_trace(const KNdSParams_CUDA& kp,const CameraParams_CUDA& cp,bool require_fp64) {
    int id=0;CUDA_CHECK(cudaGetDevice(&id));
    cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,id));
    if(require_fp64 && !(prop.major>1 || (prop.major==1 && prop.minor>=3)))
        throw std::runtime_error("CUDA device does not support native FP64");
    const size_t count=checked_pixel_count(uint32_t(cp.width),uint32_t(cp.height));
    std::vector<GeoPixel> pixels(count);
    GeoPixel* output=nullptr;
    CUDA_CHECK(cudaMalloc(&output,count*sizeof(GeoPixel)));
    try {
        dim3 block(16,16),grid((cp.width+15)/16,(cp.height+15)/16);
        trace_kernel<<<grid,block>>>(output,kp,cp);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(pixels.data(),output,count*sizeof(GeoPixel),cudaMemcpyDeviceToHost));
    } catch (...) { cudaFree(output);throw; }
    CUDA_CHECK(cudaFree(output));return pixels;
}
#endif
