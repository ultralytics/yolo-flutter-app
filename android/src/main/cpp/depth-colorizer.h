// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

struct DepthRange {
    float min;
    float max;
};

// Linear-interpolated percentile (numpy default) of q in [0, 1]; reorders values.
inline float percentile(std::vector<float> &values, float q) {
    const float rank = q * (values.size() - 1);
    const size_t low = static_cast<size_t>(rank);
    std::nth_element(values.begin(), values.begin() + low, values.end());
    if (low + 1 == values.size()) return values[low];
    const float high = *std::min_element(values.begin() + low + 1, values.end());
    return values[low] + (rank - low) * (high - values[low]);
}

// Colors depth like ultralytics/utils/plotting.py colorize_depth (disparity mode): v = 1/d for valid d, normalized
// between the 2nd and 98th percentiles of ~64k evenly strided valid v (near = warm, the end of colors). Invalid pixels
// are opaque black. range receives the valid depth min/max.
inline bool colorize_depth(
        const float *depth,
        int size,
        int32_t *pixels,
        const int32_t *colors,
        DepthRange &range) {
    float min_depth = FLT_MAX;
    float max_depth = -FLT_MAX;
    int valid = 0;
    for (int i = 0; i < size; ++i) {
        const float value = depth[i];
        if (std::isfinite(value) && value > 0.0f) {
            min_depth = std::min(min_depth, value);
            max_depth = std::max(max_depth, value);
            ++valid;
        }
    }
    if (valid == 0) return false;

    const int stride = std::max(1, valid / 65536);
    std::vector<float> sample;
    sample.reserve(valid / stride + 1);
    for (int i = 0, skip = 0; i < size; ++i) {
        const float value = depth[i];
        if (std::isfinite(value) && value > 0.0f && skip-- == 0) {
            sample.push_back(1.0f / value);
            skip = stride - 1;
        }
    }
    const float lo = percentile(sample, 0.02f);
    const float scale = 255.0f / std::max(percentile(sample, 0.98f) - lo, 1e-6f);
    for (int i = 0; i < size; ++i) {
        const float value = depth[i];
        pixels[i] = std::isfinite(value) && value > 0.0f
                    ? colors[static_cast<int>(std::clamp((1.0f / value - lo) * scale, 0.0f, 255.0f))]
                    : static_cast<int32_t>(0xFF000000);
    }
    range = {min_depth, max_depth};
    return true;
}
