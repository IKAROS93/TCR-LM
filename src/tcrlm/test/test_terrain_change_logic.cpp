#include <gtest/gtest.h>
#include "tcrlm/terrain_change_logic.h"

using tcrlm::terrain_change::Direction;
using tcrlm::terrain_change::classifyDirection;
using tcrlm::terrain_change::sameTypedFourNeighbor;
using tcrlm::terrain_change::progressiveWeight;
using tcrlm::terrain_change::oppositeDirections;

TEST(TerrainChangeLogic, DetectsAccumulationWhenUpperLayerIsNewer)
{
    EXPECT_EQ(Direction::accumulation, classifyDirection(0.30f, 5.0, 3.0f, 0.25f, 0.25f));
}

TEST(TerrainChangeLogic, DetectsExcavationWhenLowerLayerIsNewer)
{
    EXPECT_EQ(Direction::excavation, classifyDirection(0.30f, -5.0, 3.0f, 0.25f, 0.25f));
}

TEST(TerrainChangeLogic, AppliesIndependentThresholdsAndTimeGate)
{
    EXPECT_EQ(Direction::none, classifyDirection(0.30f, 5.0, 3.0f, 0.35f, 0.25f));
    EXPECT_EQ(Direction::none, classifyDirection(0.30f, -2.0, 3.0f, 0.25f, 0.25f));
}

TEST(TerrainChangeLogic, OppositeDirectionsAreNotConnectedIntoOneRegion)
{
    EXPECT_TRUE(sameTypedFourNeighbor(0, 0, 1, 1, 0, 1));
    EXPECT_FALSE(sameTypedFourNeighbor(0, 0, 1, 1, 0, 2));
}

TEST(TerrainChangeLogic, RepeatedConfirmationsProgressivelyReduceWeight)
{
    EXPECT_NEAR(0.9f, progressiveWeight(0, 0.1f, 0.1f), 1e-6f);
    EXPECT_NEAR(0.8f, progressiveWeight(1, 0.1f, 0.1f), 1e-6f);
}

TEST(TerrainChangeLogic, DirectionFlipRequiresFreshPersistentRegion)
{
    EXPECT_TRUE(oppositeDirections(Direction::excavation, Direction::accumulation));
    EXPECT_TRUE(oppositeDirections(Direction::accumulation, Direction::excavation));
    EXPECT_FALSE(oppositeDirections(Direction::excavation, Direction::excavation));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
