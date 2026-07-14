// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "../../main/cpp/depth-colorizer.h"

#include <array>
#include <cassert>
#include <limits>

int main() {
    std::array<int32_t, 256> colors{};
    for (int i = 0; i < 256; ++i) colors[i] = 0xff000000 | i;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::array<float, 12> depth = {99, 99, 99, 99, 99, 1, 2, nan, 99, inf, -1, 4};
    std::array<int32_t, 6> pixels{};
    DepthRange range{};
    assert(colorize_depth(depth.data(), 4, 1, 1, 3, 2, pixels.data(), colors.data(), range));
    assert(range.min == 1 && range.max == 4);
    assert(pixels[0] == colors[255] && pixels[1] != 0 && pixels[2] == 0);
    assert(pixels[3] == 0 && pixels[4] == 0 && pixels[5] == colors[0]);

    const std::array<float, 4> constant = {3, nan, 0, 3};
    std::array<int32_t, 4> constant_pixels{};
    assert(colorize_depth(constant.data(), 2, 0, 0, 2, 2, constant_pixels.data(), colors.data(), range));
    assert(range.min == 3 && range.max == 3);
    assert(constant_pixels[0] == colors[0] && constant_pixels[1] == 0 &&
           constant_pixels[2] == 0 && constant_pixels[3] == colors[0]);

    const std::array<float, 4> invalid = {0, -1, nan, inf};
    assert(!colorize_depth(invalid.data(), 2, 0, 0, 2, 2, pixels.data(), colors.data(), range));
}
