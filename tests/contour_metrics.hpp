#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace contour {
struct Mask {
    int width, height;
    std::vector<uint8_t> pixels;
    bool at(int x, int y) const { return pixels[size_t(y)*width+x] != 0; }
};
struct Region { int x0, y0, x1, y1; }; // inclusive pixel coordinates
struct Point { int x, y; };
struct Match {
    Point point, nearest;
    int distance;
    bool from_candidate;
};

inline std::vector<Point> boundary(const Mask& mask, Region region) {
    if (mask.width < 3 || mask.height < 3 ||
        mask.pixels.size() != size_t(mask.width)*size_t(mask.height))
        throw std::invalid_argument("Invalid contour mask");
    std::vector<Point> points;
    for (int y=std::max(1,region.y0); y<=std::min(mask.height-2,region.y1); ++y)
        for (int x=std::max(1,region.x0); x<=std::min(mask.width-2,region.x1); ++x)
            // Determine the real image boundary before applying the ROI. The
            // ROI perimeter must never become an invented edge of the disk.
            if (mask.at(x,y) && (!mask.at(x-1,y) || !mask.at(x+1,y) ||
                                !mask.at(x,y-1) || !mask.at(x,y+1)))
                points.push_back({x,y});
    return points;
}

struct Comparison {
    bool valid=false;
    size_t reference_points=0, candidate_points=0;
    int maximum=-1, p95=-1, p99=-1;
    double mean=0;
    size_t beyond_one=0;
    std::vector<Match> matches;
    bool within_one_pixel() const { return valid && maximum<=1; }
};

inline Comparison compare(const Mask& reference, const Mask& candidate, Region region) {
    if (reference.width!=candidate.width || reference.height!=candidate.height)
        throw std::invalid_argument("Contour images have different dimensions");
    const auto ref=boundary(reference,region), test=boundary(candidate,region);
    const Region full{1,1,reference.width-2,reference.height-2};
    const auto ref_full=boundary(reference,full), test_full=boundary(candidate,full);
    Comparison result;
    result.reference_points=ref.size(); result.candidate_points=test.size();
    // Empty contours are a failed/missing observation, never perfect agreement.
    if (ref.empty() || test.empty()) return result;
    result.valid=true; result.maximum=0;
    auto directed=[&](const std::vector<Point>& source,
                      const std::vector<Point>& target, bool from_candidate) {
        for (Point point : source) {
            int distance=std::numeric_limits<int>::max(); Point nearest{};
            for (Point other : target) {
                // Chebyshev distance: one native pixel in either axis, including
                // a diagonal neighbour. No smoothing and no percentile cutoff.
                const int d=std::max(std::abs(point.x-other.x),std::abs(point.y-other.y));
                if (d<distance) { distance=d; nearest=other; }
                if (distance==0) break;
            }
            result.matches.push_back({point,nearest,distance,from_candidate});
            result.maximum=std::max(result.maximum,distance);
            result.beyond_one += distance>1;
            result.mean += distance;
        }
    };
    // Only source points are restricted to the ROI. A nearest neighbour may
    // lie just outside it: clipping targets would inflate endpoint distances.
    directed(ref,test_full,false); directed(test,ref_full,true);
    result.mean/=result.matches.size();
    std::vector<int> distances;
    for (const auto& match : result.matches) distances.push_back(match.distance);
    std::sort(distances.begin(),distances.end());
    result.p95=distances[size_t(std::ceil(.95*distances.size()))-1];
    result.p99=distances[size_t(std::ceil(.99*distances.size()))-1];
    return result;
}
} // namespace contour
