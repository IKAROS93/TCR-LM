#pragma once

#include <cstdint>
#include <cmath>

namespace tcrlm
{
namespace terrain_change
{
enum class Direction : std::uint8_t
{
    none = 0,
    excavation = 1,
    accumulation = 2
};

inline Direction classifyDirection(float heightGap,
                                   double upperMinusLowerTime,
                                   float minTimeSeparation,
                                   float accumulationThreshold,
                                   float excavationThreshold)
{
    if (std::fabs(upperMinusLowerTime) < minTimeSeparation)
        return Direction::none;
    if (upperMinusLowerTime > 0.0)
        return heightGap >= accumulationThreshold ? Direction::accumulation : Direction::none;
    return heightGap >= excavationThreshold ? Direction::excavation : Direction::none;
}

inline bool sameTypedFourNeighbor(int u0, int v0, std::uint8_t type0,
                                  int u1, int v1, std::uint8_t type1)
{
    return type0 == type1 && (std::abs(u0 - u1) + std::abs(v0 - v1) == 1);
}

inline float progressiveWeight(std::uint16_t completedConfirmations,
                               float dropPerConfirmation,
                               float minimumWeight)
{
    const float next = 1.0f - dropPerConfirmation *
        static_cast<float>(completedConfirmations + 1);
    return next < minimumWeight ? minimumWeight : (next > 1.0f ? 1.0f : next);
}

inline bool oppositeDirections(Direction first, Direction second)
{
    return (first == Direction::excavation && second == Direction::accumulation) ||
           (first == Direction::accumulation && second == Direction::excavation);
}
}
}
