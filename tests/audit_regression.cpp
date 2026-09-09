// Exercises the actual renderer helpers and the actual CUDA tracing function.
// CUDA_HOST_TEST changes only execution qualifiers and omits the GPU launcher.
#define KERRTRACE_NO_MAIN
#ifndef OUT_DIR
#define OUT_DIR "."
#endif
#include "../main.cpp"
#define KERRTRACE_CUDA_HOST_TEST
#include "../gpu/cuda/tracer.cu"

#include <filesystem>

namespace {
int failures = 0;
void check(bool ok, const char* description) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << description << '\n';
    if (!ok) ++failures;
}
bool near(double x, double y, double tol=1e-10) {
    return std::isfinite(x) && std::isfinite(y) && std::abs(x-y)<tol;
}
void physics_tests() {
    for (double q : {0.0, 0.1}) for (double lambda : {0.0, 1e-4, -1e-4}) {
        KNdSMetric positive(1,.5,q,lambda), negative(1,-.5,q,lambda);
        const double rp=positive.r_isco(), rn=negative.r_isco();
        check(near(rp,rn,1e-5) && rp>3 && rp<6, "general ISCO respects corotating spin reflection");
    }
    KNdSMetric rn(1,0,1,0);
    check(near(rn.r_isco(),4.0,3e-4), "extremal Reissner-Nordstrom ISCO is 4M");
    KNdSMetric g(1,0,0,0);
    for (double observer : {10.,40.,1000.}) {
        Camera cam(observer,80,0,45,32,18);
        auto s=cam.angle_ray(0,0,g);
        const double frequency=-s.pt*g.static_observer_ut(s.r,s.theta);
        const double expected=std::sqrt(1.-3./8.)/std::sqrt(1.-2./observer);
        check(near(frequency,1.) && near(g.disk_frequency_shift(8,s.pt,s.pphi,frequency),expected),
              "redshift agrees with analytic finite-observer Schwarzschild value");
        check(near(d_robust_disk_redshift(8,s.pt,s.pphi,1,0,0,0,g.static_observer_ut(s.r,s.theta)),expected),
              "CUDA redshift agrees with independent Schwarzschild value");
    }
    ColorParams colors;colors.radial_profile=DiskRadialProfile::PHYSICAL_NT;
    const double isco=KNdSMetric(1,.5,0,0).r_isco();
    for(double radius : {5.,8.,12.}) {
        const double plus=disk_flux_raw(radius,isco,.5,colors)/disk_flux_reference(isco,12,.5,colors);
        const double minus=disk_flux_raw(radius,isco,-.5,colors)/disk_flux_reference(isco,12,-.5,colors);
        check(near(plus,minus),"normalized NT proxy is invariant under corotating spin reflection");
    }
    for(double spin : {0.,.5,-.5,.998}) {
        KNdSMetric k(1,spin,0,0);Camera cam(40,80,15,45,32,18);
        auto s=cam.angle_ray(.03,-.02,k);
        double x,y,z,r,th,ph,px,py,pz;
        const double phi=s.phi+d_twist(s.r,spin,1,0);
        d_BL_to_KS_spatial(s.r,s.theta,phi,spin,x,y,z);
        d_KS_to_BL_spatial(x,y,z,spin,r,th,ph);
        check(near(r,s.r) && near(th,s.theta) && near(ph,phi),"CUDA spatial embedding round-trip");
        bool converted=d_BL_covector_to_KS(s.r,s.theta,phi,spin,s.pr,s.ptheta,s.pphi,s.pt,1,0,px,py,pz);
        double p[]={s.pt,px,py,pz};
        check(converted && std::abs(k.hamiltonian_KS(0,x,y,z,p,true))<1e-10,
              "CUDA BL-to-KS transform preserves the null covector and Killing energy");
        double pr, ptheta, pphi;
        d_KS_covector_to_BL(s.r,s.theta,phi,spin,px,py,pz,s.pt,1,0,pr,ptheta,pphi);
        check(converted && near(pr,s.pr) && near(ptheta,s.ptheta) && near(pphi,s.pphi),
              "CUDA full covector transform round-trip");
        const double expected=k.disk_frequency_shift(8,s.pt,s.pphi,-s.pt*k.static_observer_ut(s.r,s.theta));
        check(near(d_robust_disk_redshift(8,s.pt,s.pphi,1,spin,0,0,k.static_observer_ut(s.r,s.theta)),expected),
              "CUDA redshift has the physical angular-momentum sign");
    }
}
void io_tests() {
    const auto dir=std::filesystem::temp_directory_path()/
        ("kerrtrace-audit-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    const auto path=(dir/"frame.kgeo").string();
    KGeoMeta meta{2,2,1,.5,0,0,4.233,4.233,12,80,0,40};
    std::vector<GeoPixel> pixels(4),loaded;KGeoMeta read{};
    pixels[0].outcome=1;pixels[0].r=8;pixels[0].redshift=.8;
    save_kgeo(path.c_str(),pixels,meta);
    check(load_kgeo(path.c_str(),loaded,read) && loaded.size()==4 && loaded[0].r==8,"KGEO valid round-trip");
    auto raw_header=[&](uint32_t w,uint32_t h) {
        meta.W=w;meta.H=h;
        std::ofstream f(path,std::ios::binary);f.write(KGEO_MAGIC,4);
        f.write(reinterpret_cast<const char*>(&KGEO_VERSION),4);
        f.write(reinterpret_cast<const char*>(&meta),sizeof(meta));
    };
    raw_header(65536,65536);
    check(!load_kgeo(path.c_str(),loaded,read) && loaded.size()==4,"KGEO rejects overflowing dimensions without changing output");
    raw_header(2,2);check(!load_kgeo(path.c_str(),loaded,read),"KGEO rejects truncated payload before allocation");
    raw_header(0,2);check(!load_kgeo(path.c_str(),loaded,read),"KGEO rejects zero dimensions");
    {std::ofstream f(path,std::ios::binary);f.write("KGEO",4);}
    check(!load_kgeo(path.c_str(),loaded,read),"KGEO rejects truncated header");
    meta.W=meta.H=2;
    save_kgeo(path.c_str(),pixels,meta);
    {std::ofstream f(path,std::ios::app|std::ios::binary);f.put('x');}
    check(!load_kgeo(path.c_str(),loaded,read),"KGEO rejects unexpected trailing records");
    bool failed=false;try{save_kgeo((dir/"missing/file.kgeo").string().c_str(),pixels,meta);}catch(const std::exception&){failed=true;}
    check(failed,"KGEO writing propagates filesystem errors");
    failed=false;try{std::vector<RGB> rgb(4);write_png((dir/"missing/frame.png").string().c_str(),rgb,2,2);}catch(const std::exception&){failed=true;}
    check(failed,"PNG writing propagates filesystem errors");
    std::filesystem::remove_all(dir);
}
void backend_tests() {
    ColorParams c;
    check(!metal_cpu_reason(c,false,false),"Metal accepts supported blackbody shading");
    for(auto palette : {DiskPalette::STRATIFIED,DiskPalette::INTERSTELLAR_NASA}) {
        c.palette=palette;check(metal_cpu_reason(c,false,false)!=nullptr,"Metal refuses unsupported palettes explicitly");
    }
    c=ColorParams{};
    check(metal_cpu_reason(c,true,false)!=nullptr,"Metal routes Jacobi footprint filtering to CPU");
    check(metal_cpu_reason(c,false,true)!=nullptr,"Metal does not silently drop requested KGEO export");
    c.planck_emission=true;check(metal_cpu_reason(c,false,false)!=nullptr,"Metal refuses unsupported emission controls");
    c=ColorParams{};
    check(!cuda_cpu_reason(c,false,RaySolverMode::STANDARD,Integrator::RK4_DOUBLING),"CUDA accepts geometry with shared CPU shading");
    check(cuda_cpu_reason(c,false,RaySolverMode::STANDARD,Integrator::DOPRI5)!=nullptr,"CUDA does not silently ignore DOPRI5");
}
void cuda_trace_tests() {
    constexpr int width=48,height=27;
    IntegratorControls controls;controls.max_steps=60000;controls.tol=1e-9;
    for(double spin : {0.,.5,-.5}) for(int chart : {0,1}) {
        KNdSMetric g(1,spin,0,0);Camera camera(40,80,15,45,width,height);
        KNdSParams_CUDA kp{1,spin,0,0,g.r_horizon(),g.r_isco(),12};
        CameraParams_CUDA cp{40,camera.theta_obs,camera.phi_obs,camera.fov_h,width,height,
            chart,60000,1,1,1e-9,.25,-.25};
        int different=0,common=0;double max_dr=0,max_g=0;
        for(int y=0;y<height;++y)for(int x=0;x<width;++x) {
            auto gpu=d_trace_one(x,y,kp,cp);
            auto s=camera.pixel_ray(x,y,g,.25,-.25);
            auto cpu=chart?trace_single_ks(s,g,kp.r_isco,12,42,Integrator::RK4_DOUBLING,controls,nullptr)
                          :trace_single(s,g,kp.r_isco,12,42,Integrator::RK4_DOUBLING,controls,nullptr);
            different+=int(gpu.outcome)!=int(cpu.out);
            if(gpu.outcome==1 && cpu.out==Outcome::DISK_HIT) {
                ++common;max_dr=std::max(max_dr,std::abs(gpu.r-cpu.r));
                max_g=std::max(max_g,std::abs(gpu.redshift-cpu.redshift));
            }
        }
        std::cout<<"CUDA host trace spin="<<spin<<" chart="<<chart<<" outcome differences="<<different
                 <<" common="<<common<<" max dr="<<max_dr<<" max dg="<<max_g<<'\n';
        check(different<=2 && common>150 && max_dr<.002 && max_g<.001,
              "actual CUDA tracing function agrees with CPU, including subpixel offsets");
    }
}
}
int main() {
    physics_tests();io_tests();backend_tests();cuda_trace_tests();
    return failures?1:0;
}
