#import <Metal/Metal.h>
#define KERRTRACE_NO_MAIN
#define OUT_DIR "."
#include "../main.cpp"

int main() {
    @autoreleasepool {
        if (!MTLCreateSystemDefaultDevice()) return 77;
        constexpr int width=96, height=54;
        FrameParams frame; frame.a=.5; frame.theta=80; frame.phi=15;
        frame.r_obs=40; frame.fov=45; frame.disk_out=12;
        BackgroundImage background;
        IntegratorControls controls; controls.max_steps=60000;
        int failures=0;
        for (auto chart : {CoordinateChart::BL, CoordinateChart::KS}) {
            for (bool custom : {false,true}) {
                ColorParams colors;
                if (custom) { colors.temp_scale=.65; colors.doppler_exp=2; }
                auto gpu=render_image(width,height,frame,background,false,RaySolverMode::STANDARD,
                    chart,Integrator::RK4_DOUBLING,controls,1,1,0,0,
                    IntersectionMode::HERMITE,0,false,false,false,colors);
                if (std::string(last_render_backend)!="gpu-metal") {
                    std::cerr << "The Metal test must run on the GPU, not a CPU fallback\n";
                    return 1;
                }
                KGeoMeta meta;
                auto geo=trace_geodesics(width,height,frame,false,RaySolverMode::STANDARD,
                    chart,Integrator::RK4_DOUBLING,controls,0,0,1,0,0,&colors,&meta);
                auto cpu=colorize_buffer(geo,width,height,colors,background,1,frame.a,
                    meta.r_isco,meta.r_disk_in,meta.r_disk_out);
                double total=0; int samples=0, large=0;
                for(int y=1;y<height-1;++y) for(int x=1;x<width-1;++x) {
                    int i=y*width+x;
                    // Exclude one pixel around hit/miss boundaries: FP32 can
                    // move a marginal ray across a boundary at this resolution.
                    bool interior=true;
                    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx)
                        interior=interior && geo[i+dy*width+dx].outcome==1;
                    if(!interior) continue;
                    double error=(std::abs(int(gpu[i].r)-cpu[i].r)
                        +std::abs(int(gpu[i].g)-cpu[i].g)+std::abs(int(gpu[i].b)-cpu[i].b))/3.;
                    total+=error; large+=error>20; ++samples;
                }
                const double mean=total/std::max(samples,1);
                std::cout << "Metal chart=" << int(chart) << " custom=" << custom
                    << " interior=" << samples << " RGB MAE=" << mean << " large=" << large << '\n';
                if(samples<500 || mean>5 || large>samples*.05) ++failures;
            }
        }
        return failures?1:0;
    }
}
