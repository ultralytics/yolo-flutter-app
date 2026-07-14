// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

struct DepthRange {
    float min;
    float max;
};

inline bool colorize_depth(
        const float *depth,
        int depth_width,
        int left,
        int top,
        int width,
        int height,
        int32_t *pixels,
        const int32_t *colors,
        DepthRange &range) {
    float min_depth = FLT_MAX;
    float max_depth = -FLT_MAX;
    for (int y = 0; y < height; ++y) {
        const float *row = depth + (y + top) * depth_width + left;
        for (int x = 0; x < width; ++x) {
            const float value = row[x];
            if (std::isfinite(value) && value > 0.0f) {
                min_depth = std::min(min_depth, value);
                max_depth = std::max(max_depth, value);
            }
        }
    }
    if (min_depth == FLT_MAX) return false;

    constexpr int lut_size = 4096;
    int32_t logarithmic_colors[lut_size];
    const float linear_range = max_depth - min_depth;
    if (linear_range > 0.0f) {
        const float log_max = std::log(max_depth);
        const float log_range = std::max(log_max - std::log(min_depth), 1e-6f);
        for (int bin = 0; bin < lut_size; ++bin) {
            const float value = min_depth + linear_range * bin / (lut_size - 1);
            const int index = std::clamp(
                    static_cast<int>(std::lround((log_max - std::log(value)) / log_range * 255.0f)),
                    0,
                    255);
            logarithmic_colors[bin] = colors[index];
        }
        const float bin_scale = (lut_size - 1) / linear_range;
        for (int y = 0; y < height; ++y) {
            const float *row = depth + (y + top) * depth_width + left;
            int32_t *target = pixels + y * width;
            for (int x = 0; x < width; ++x) {
                const float value = row[x];
                target[x] = std::isfinite(value) && value > 0.0f
                        ? logarithmic_colors[std::clamp(
                                  static_cast<int>(std::lround((value - min_depth) * bin_scale)),
                                  0,
                                  lut_size - 1)]
                        : 0;
            }
        }
    } else {
        for (int y = 0; y < height; ++y) {
            const float *row = depth + (y + top) * depth_width + left;
            int32_t *target = pixels + y * width;
            for (int x = 0; x < width; ++x) {
                const float value = row[x];
                target[x] = std::isfinite(value) && value > 0.0f ? colors[0] : 0;
            }
        }
    }
    range = {min_depth, max_depth};
    return true;
}
