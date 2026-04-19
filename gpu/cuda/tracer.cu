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
    const double s    = (a < 0.0) ? -1.0 : 1.0;
    const double Meff = M - Q*Q/(2.0*r) + L*a*r*r/3.0;
    const double sq   = sqrt(fmax(Meff, 0.0));
    const double den  = r*sqrt(r) + s*a*sq;
    return (fabs(den) > 1e-14) ? (s*sq/den) : 0.0;
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
    X = st * (r*cf + a*sf);
    Y = st * (r*sf - a*cf);
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
    const double rcfa = r*cf + a*sf;
    const double rsfa = r*sf - a*cf;

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
                                    double& pX, double& pY, double& pZ) {
    double J[3][3];
    d_jacobian_BL_to_KS(r, theta, phi, a, J);

    double A[3][3];
    // A = J^T, solve A * p_xyz = p_bl
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            A[j][i] = J[i][j];

    double b[3] = {pr, ptheta, pphi};
    double x[3];
    if (!d_solve3x3(A, b, x)) return false;
    pX = x[0]; pY = x[1]; pZ = x[2];
    return true;
}

__device__ void d_KS_covector_to_BL(double r, double theta, double phi, double a,
                                    double pX, double pY, double pZ,
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
    double pUphi =-sa*cb /sqrt(fabs(gl[3][3]));
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
    double prev_r=r0;
    double prev_cos=cos(th0);
    uint32_t colour=0xFF000000u;

    // ── KS chart path (Lambda=0 only) ─────────────────────────
    const bool want_ks = (cp.chart == 1 && fabs(L) <= 1e-12);
    if (want_ks) {
        double X, Y, Z;
        d_BL_to_KS_spatial(r0, th0, cp.phi_obs, a, X, Y, Z);
        double pX, pY, pZ;
        bool ks_ok = d_BL_covector_to_KS(r0, th0, cp.phi_obs, a, pr, pth, pphi, pX, pY, pZ);

        double pT = pt;
        if (ks_ok) {
            double gu_ks[4][4];
            d_gUU_KS(X, Y, Z, M, a, Q, gu_ks);
            const double Aks = gu_ks[0][0];
            const double Bks = 2.0 * (gu_ks[0][1]*pX + gu_ks[0][2]*pY + gu_ks[0][3]*pZ);
            const double Cks = gu_ks[1][1]*pX*pX + gu_ks[2][2]*pY*pY + gu_ks[3][3]*pZ*pZ
                             + 2.0*gu_ks[1][2]*pX*pY + 2.0*gu_ks[1][3]*pX*pZ + 2.0*gu_ks[2][3]*pY*pZ;
            const double dks = Bks*Bks - 4.0*Aks*Cks;
            if (dks >= 0.0 && fabs(Aks) > 1e-15) {
                const double sq = sqrt(dks);
                const double pT1 = (-Bks - sq)/(2.0*Aks);
                const double pT2 = (-Bks + sq)/(2.0*Aks);
                pT = (pT1 < 0.0) ? pT1 : pT2;
                if (pT > 0.0) pT = fmin(pT1, pT2);
                ks_ok = isfinite(pT);
            } else {
                ks_ok = false;
            }
        }

        if (ks_ok) {
            double dlam_ks = 1.0;
            double prevX = X, prevY = Y, prevZ = Z;
            double prevPX = pX, prevPY = pY, prevPZ = pZ;

            for(int iter=0; iter<200000; ++iter) {
                // Step-doubling
                double Xh=X,Yh=Y,Zh=Z,pXh=pX,pYh=pY,pZh=pZ;
                d_rk4_KS(Xh,Yh,Zh,pXh,pYh,pZh,pT,M,a,Q,dlam_ks);
                double Xf=X,Yf=Y,Zf=Z,pXf=pX,pYf=pY,pZf=pZ;
                d_rk4_KS(Xf,Yf,Zf,pXf,pYf,pZf,pT,M,a,Q,dlam_ks*0.5);
                d_rk4_KS(Xf,Yf,Zf,pXf,pYf,pZf,pT,M,a,Q,dlam_ks*0.5);

                const double err = sqrt((Xh-Xf)*(Xh-Xf) + (Yh-Yf)*(Yh-Yf) + (Zh-Zf)*(Zh-Zf)
                                      + (pXh-pXf)*(pXh-pXf) + (pYh-pYf)*(pYh-pYf) + (pZh-pZf)*(pZh-pZf))/15.0;
                const double tol = 1e-7;
                if(err<tol||dlam_ks<1e-10){
                    X=Xf;Y=Yf;Z=Zf;pX=pXf;pY=pYf;pZ=pZf;
                    double sc=(err>1e-12)?0.9*pow(tol/err,0.2):2.0;
                    dlam_ks=fmin(fmax(dlam_ks*sc,1e-10),100.0);
                } else {
                    dlam_ks=fmax(dlam_ks*0.9*pow(tol/err,0.25),1e-10);
                    continue;
                }

                if (!(isfinite(X) && isfinite(Y) && isfinite(Z) &&
                      isfinite(pX) && isfinite(pY) && isfinite(pZ))) break;

                const double r_now = d_r_KS(X, Y, Z, a);
                if (r_now < kp.r_horizon*1.03) break;
                if (r_now > cp.r_obs*1.05) break;

                if(prevZ*Z<=0.0){
                    double denom = prevZ - Z;
                    double w = (fabs(denom) > 1e-12) ? (prevZ / denom) : 0.5;
                    w = fmax(0.0, fmin(1.0, w));
                    double Xhit = prevX + w*(X - prevX);
                    double Yhit = prevY + w*(Y - prevY);
                    double Zhit = prevZ + w*(Z - prevZ);
                    double pXhit = prevPX + w*(pX - prevPX);
                    double pYhit = prevPY + w*(pY - prevPY);
                    double pZhit = prevPZ + w*(pZ - prevPZ);

                    double r_hit, th_hit, ph_hit;
                    d_KS_to_BL_spatial(Xhit, Yhit, Zhit, a, r_hit, th_hit, ph_hit);
                    if(!(r_hit>=kp.r_isco && r_hit<=kp.r_disk_out)){
                        prevX = X; prevY = Y; prevZ = Z;
                        prevPX = pX; prevPY = pY; prevPZ = pZ;
                        continue;
                    }

                    double pr_hit, pth_hit, pphi_hit;
                    d_KS_covector_to_BL(r_hit, th_hit, ph_hit, a, pXhit, pYhit, pZhit,
                                        pr_hit, pth_hit, pphi_hit);
                    double Omega=d_keplerian_omega(r_hit, M, a, Q, L);
                    double b_=pphi_hit/(-pT);
                    double gl2[4][4]; d_gLL(r_hit, M_PI/2.0, M, a, Q, L, gl2);
                    double d2=-(gl2[0][0]+2.0*gl2[0][3]*Omega+gl2[3][3]*Omega*Omega);
                    double dv=1.0-Omega*b_;
                    double red=(d2>0.0&&fabs(dv)>1e-10)?sqrt(d2)/dv:1.0;
                    red=fmax(0.0,fmin(20.0,red));

                    double T0=2e6*pow(r_hit/(6.0*M),-0.75);
                    double T =T0*fmax(0.1,fmin(10.0,red));
                    double I =fmax(0.0,fmin(2.5,pow(red,4.0)*pow(6.0*M/r_hit,3.0)));

                    uchar4 c=d_blackbody(T);
                    double R=fmin(c.x*I/255.0,1.0)*255;
                    double G=fmin(c.y*I/255.0,1.0)*255;
                    double B2=fmin(c.z*I/255.0,1.0)*255;
                    colour=(0xFFu<<24)|((uint32_t)B2<<16)|((uint32_t)G<<8)|(uint32_t)R;
                    break;
                }

                prevX = X; prevY = Y; prevZ = Z;
                prevPX = pX; prevPY = pY; prevPZ = pZ;
            }
            output[py*cp.width+px]=colour;
            return;
        }
    }

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
        if(prev_cos*cos_th<=0.0){
            double denom = prev_cos - cos_th;
            double w = (fabs(denom) > 1e-12) ? (prev_cos / denom) : 0.5;
            w = fmax(0.0, fmin(1.0, w));
            double r_hit = prev_r + w*(r - prev_r);
            if(!(r_hit>=kp.r_isco && r_hit<=kp.r_disk_out)){
                prev_cos=cos_th;
                prev_r=r;
                continue;
            }

            double Omega=d_keplerian_omega(r_hit, M, a, Q, L);
            double b_=pphi/(-pt);
            double gl2[4][4]; d_gLL(r_hit, M_PI/2.0, M, a, Q, L, gl2);
            double d2=-(gl2[0][0]+2.0*gl2[0][3]*Omega+gl2[3][3]*Omega*Omega);
            double dv=1.0-Omega*b_;
            double red=(d2>0.0&&fabs(dv)>1e-10)?sqrt(d2)/dv:1.0;
            red=fmax(0.0,fmin(20.0,red));

            double T0=2e6*pow(r_hit/(6.0*M),-0.75);
            double T =T0*fmax(0.1,fmin(10.0,red));
            double I =fmax(0.0,fmin(2.5,pow(red,4.0)*pow(6.0*M/r_hit,3.0)));

            uchar4 c=d_blackbody(T);
            double R=fmin(c.x*I/255.0,1.0)*255;
            double G=fmin(c.y*I/255.0,1.0)*255;
            double B2=fmin(c.z*I/255.0,1.0)*255;
            colour=(0xFFu<<24)|((uint32_t)B2<<16)|((uint32_t)G<<8)|(uint32_t)R;
            break;
        }
        prev_cos=cos_th;
        prev_r=r;
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
