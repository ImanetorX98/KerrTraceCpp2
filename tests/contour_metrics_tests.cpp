#include "contour_metrics.hpp"
#include <iostream>

int main() {
    using namespace contour;
    const Region region{1,1,126,94};
    Mask reference{128,96,std::vector<uint8_t>(128*96,0)};
    for(int y=24;y<=64;++y) for(int x=16;x<=111;++x)
        reference.pixels[y*128+x]=1;
    int failures=0;
    auto check=[&](bool ok,const char* label) {
        std::cout<<(ok?"[PASS] ":"[FAIL] ")<<label<<'\n'; failures+=!ok;
    };
    check(compare(reference,reference,region).maximum==0,"identical contours agree exactly");
    Mask shifted{128,96,std::vector<uint8_t>(128*96,0)};
    for(int y=24;y<=64;++y) for(int x=16;x<=111;++x)
        shifted.pixels[(y+1)*128+x+1]=1;
    check(compare(reference,shifted,region).within_one_pixel(),
          "one diagonal pixel of raster quantization is allowed");
    auto bump=reference;
    for(int x=58;x<=69;++x) for(int y=21;y<=23;++y) bump.pixels[y*128+x]=1;
    auto result=compare(reference,bump,region);
    check(!result.within_one_pixel() && result.maximum==3,
          "localized three-pixel bump fails even on an otherwise identical image");
    auto hole=reference;hole.pixels[44*128+64]=0;
    result=compare(reference,hole,region);
    check(!result.within_one_pixel() && result.maximum>10,
          "isolated interior hole creates a detectable spurious contour");
    Mask empty{128,96,std::vector<uint8_t>(128*96,0)};
    check(!compare(reference,empty,region).within_one_pixel(),"all-black candidate cannot pass");
    check(!compare(empty,empty,region).valid,"two empty masks are not evidence of agreement");
    check(boundary(reference,{40,35,80,55}).empty(),"ROI clipping creates no artificial boundary");
    Mask diagonal{128,96,std::vector<uint8_t>(128*96,0)};
    auto parallel=diagonal;
    for(int y=0;y<96;++y) for(int x=0;x<128;++x) {
        diagonal.pixels[y*128+x]=x>=y;
        parallel.pixels[y*128+x]=x>=y+2;
    }
    check(compare(diagonal,parallel,{40,40,80,80}).within_one_pixel(),
          "nearest neighbours outside the ROI prevent inflated endpoint errors");
    return failures?1:0;
}
