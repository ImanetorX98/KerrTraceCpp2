#import <Metal/Metal.h>
#define KERRTRACE_NO_MAIN
#define OUT_DIR "."
#include "../main.cpp"
#include "contour_metrics.hpp"
#include <map>

namespace {
struct Scene {
    std::string name;
    int width=640, height=360;
    CoordinateChart chart=CoordinateChart::BL;
    Integrator integrator=Integrator::RK4_DOUBLING;
    bool custom=true;
    double offset_x=0, offset_y=0;
};
struct Reference {
    KGeoMeta meta;
    std::vector<GeoPixel> normal, tight;
};
contour::Mask mask(const std::vector<RGB>& image,int width,int height,int threshold) {
    contour::Mask result{width,height,{}};
    result.pixels.reserve(image.size());
    for(const auto& pixel : image)
        result.pixels.push_back(std::max({pixel.r,pixel.g,pixel.b})>threshold);
    return result;
}
void save_matches(const std::filesystem::path& path,const contour::Comparison& result) {
    std::ofstream out(path);
    out<<"side,x,y,nearest_x,nearest_y,distance_px\n";
    for(const auto& match : result.matches)
        out<<(match.from_candidate?"candidate":"reference")<<','
           <<match.point.x<<','<<match.point.y<<','<<match.nearest.x<<','
           <<match.nearest.y<<','<<match.distance<<'\n';
    if(!out) throw std::runtime_error("Cannot write contour matches");
}
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            std::filesystem::path output="contour-output";
            bool extended=false; std::string only_case;
            for(int i=1;i<argc;++i) {
                const std::string arg=argv[i];
                if(arg=="--output-dir" && i+1<argc) output=argv[++i];
                else if(arg=="--case" && i+1<argc) only_case=argv[++i];
                else if(arg=="--extended") extended=true;
                else throw std::runtime_error("Usage: metal_contour_test [--output-dir DIR] [--case NAME] [--extended]");
            }
            if(!MTLCreateSystemDefaultDevice()) {
                std::cout<<"SKIP: no accessible Metal device\n";return 77;
            }
            std::vector<Scene> scenes={
                {"bl-custom"},
                {"bl-default",640,360,CoordinateChart::BL,Integrator::RK4_DOUBLING,false},
                {"ks-custom",640,360,CoordinateChart::KS},
                {"bl-dopri",640,360,CoordinateChart::BL,Integrator::DOPRI5}
            };
            if(extended) {
                scenes.push_back({"bl-jitter",640,360,CoordinateChart::BL,Integrator::RK4_DOUBLING,true,.25,-.25});
                scenes.push_back({"bl-1280",1280,720});
            }
            std::filesystem::create_directories(output);
            std::ofstream summary(output/"summary.csv");
            summary<<"scene,width,height,comparison,threshold,reference_edges,candidate_edges,max_px,p95_px,p99_px,mean_px,beyond_one,total,pass\n";
            std::map<std::string,Reference> references;
            FrameParams frame;frame.a=.5;frame.theta=80;frame.phi=0;
            frame.r_obs=40;frame.fov=45;frame.disk_out=12;
            BackgroundImage background;background.w=2;background.h=2;
            background.px.assign(12,0);
            int failed=0,checked=0,selected=0;
            for(const auto& scene : scenes) {
                if(!only_case.empty() && scene.name!=only_case) continue;
                ++selected;
                ColorParams colors;
                if(scene.custom) {colors.temp_scale=.65;colors.doppler_exp=2;}
                IntegratorControls controls;controls.max_steps=500000;controls.tol=1e-7;
                const std::string key=std::to_string(scene.width)+":"+std::to_string(int(scene.chart))+":"+
                    std::to_string(int(scene.integrator))+":"+std::to_string(scene.offset_x)+":"+std::to_string(scene.offset_y);
                std::cout<<"Scene "<<scene.name<<": actual Metal, CPU tol=1e-7 and 1e-10\n"<<std::flush;
                if(!references.count(key)) {
                    Reference ref;
                    ref.normal=trace_geodesics(scene.width,scene.height,frame,false,RaySolverMode::STANDARD,
                        scene.chart,scene.integrator,controls,scene.offset_x,scene.offset_y,1,0,0,&colors,&ref.meta);
                    auto tight=controls;tight.tol=1e-10;
                    ref.tight=trace_geodesics(scene.width,scene.height,frame,false,RaySolverMode::STANDARD,
                        scene.chart,scene.integrator,tight,scene.offset_x,scene.offset_y,1,0,0,&colors,nullptr);
                    references.emplace(key,std::move(ref));
                }
                const auto& ref=references.at(key);
                auto colorize=[&](const std::vector<GeoPixel>& geo) {
                    return colorize_buffer(geo,scene.width,scene.height,colors,background,1,frame.a,
                        ref.meta.r_isco,ref.meta.r_disk_in,ref.meta.r_disk_out);
                };
                const auto cpu=colorize(ref.normal), tight=colorize(ref.tight);
                const auto gpu=render_image(scene.width,scene.height,frame,background,false,RaySolverMode::STANDARD,
                    scene.chart,scene.integrator,controls,1,1,0,0,IntersectionMode::HERMITE,
                    0,false,false,false,colors,nullptr,false,scene.offset_x,scene.offset_y);
                if(std::string(last_render_backend)!="gpu-metal")
                    throw std::runtime_error("Contour test requires actual GPU execution; CPU fallback is a failure");
                write_png((output/(scene.name+"-cpu.png")).string().c_str(),cpu,scene.width,scene.height);
                write_png((output/(scene.name+"-cpu-tight.png")).string().c_str(),tight,scene.width,scene.height);
                write_png((output/(scene.name+"-metal.png")).string().c_str(),gpu,scene.width,scene.height);
                // Same angular ROI at both resolutions: upper arc and inner rim,
                // including the thin higher-order image. No edge erosion.
                const contour::Region region{int(.30*scene.width),1,int(.70*scene.width),int(.45*scene.height)};
                for(int threshold : {1,3,8}) {
                    const auto reference_mask=mask(tight,scene.width,scene.height,threshold);
                    for(bool metal : {false,true}) {
                        const std::string name=metal?"metal-vs-cpu":"cpu-convergence";
                        const auto result=contour::compare(reference_mask,
                            mask(metal?gpu:cpu,scene.width,scene.height,threshold),region);
                        const bool passed=result.within_one_pixel() && result.reference_points>=500;
                        ++checked;failed+=!passed;
                        summary<<scene.name<<','<<scene.width<<','<<scene.height<<','<<name<<','<<threshold<<','
                            <<result.reference_points<<','<<result.candidate_points<<','<<result.maximum<<','
                            <<result.p95<<','<<result.p99<<','<<result.mean<<','<<result.beyond_one<<','
                            <<result.matches.size()<<','<<int(passed)<<'\n';
                        save_matches(output/(scene.name+"-"+name+"-t"+std::to_string(threshold)+".csv"),result);
                        std::cout<<(passed?"[PASS] ":"[FAIL] ")<<scene.name<<' '<<name<<" threshold="<<threshold
                            <<" max="<<result.maximum<<"px p99="<<result.p99<<"px beyond 1px="
                            <<result.beyond_one<<'/'<<result.matches.size()<<'\n'<<std::flush;
                    }
                }
                summary.flush();
            }
            if(!selected) throw std::runtime_error("Unknown or disabled case name");
            if(!summary) throw std::runtime_error("Cannot write contour summary");
            std::cout<<"Contour checks: "<<checked-failed<<'/'<<checked<<" passed. Artifacts: "<<output<<'\n';
            return failed?1:0;
        } catch(const std::exception& error) {
            std::cerr<<"Contour test error: "<<error.what()<<'\n';return 1;
        }
    }
}
