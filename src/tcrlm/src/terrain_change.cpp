#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "tcrlm/TerrainChangeSnapshot.h"
#include "tcrlm/TerrainChangeUpdateArray.h"
#include "tcrlm/terrain_change_logic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

typedef pcl::PointXYZI PointType;

struct GridIndex
{
    int u = 0;
    int v = 0;

    bool operator==(const GridIndex& other) const
    {
        return u == other.u && v == other.v;
    }
};

struct GridIndexHash
{
    std::size_t operator()(const GridIndex& index) const
    {
        const std::size_t h1 = std::hash<int>()(index.u);
        const std::size_t h2 = std::hash<int>()(index.v);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

struct CandidateKey
{
    GridIndex index;
    uint8_t change_type = tcrlm::TerrainChangeUpdate::CHANGE_NONE;

    bool operator==(const CandidateKey& other) const
    {
        return index == other.index && change_type == other.change_type;
    }
};

struct CandidateKeyHash
{
    std::size_t operator()(const CandidateKey& key) const
    {
        return GridIndexHash()(key.index) ^ (static_cast<std::size_t>(key.change_type) << 1);
    }
};

struct CandidateTrack
{
    int count = 0;
    uint32_t last_snapshot_id = 0;
};

struct PointSource
{
    const tcrlm::TerrainChangePoint* point = nullptr;
    float h = 0.0f;
    double time = 0.0;
};

struct Cell
{
    std::vector<PointSource> sources;
    std::vector<float> heights;
    float h_low = std::numeric_limits<float>::max();
    float h_high = -std::numeric_limits<float>::max();
};

struct CandidateCell
{
    GridIndex index;
    uint8_t change_type = tcrlm::TerrainChangeUpdate::CHANGE_NONE;
    uint32_t region_id = 0;
    float h_recent = 0.0f;
    float h_history = 0.0f;
};

struct Region
{
    uint8_t change_type = tcrlm::TerrainChangeUpdate::CHANGE_NONE;
    std::vector<CandidateCell> cells;
};

class TerrainChangeNode
{
public:
    TerrainChangeNode()
    {
        nh.param<std::string>("tcrlm/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("tcrlm/terrainChangeHeightAxis", terrainChangeHeightAxis, "z");
        nh.param<std::string>("tcrlm/terrainChangeDetectionMode", terrainChangeDetectionMode, "both");
        if (terrainChangeDetectionMode != "both" && terrainChangeDetectionMode != "excavation" &&
            terrainChangeDetectionMode != "accumulation")
            throw std::invalid_argument("terrainChangeDetectionMode must be both, excavation, or accumulation");
        nh.param<float>("tcrlm/terrainChangeGridResolution", terrainChangeGridResolution, 1.0);
        nh.param<int>("tcrlm/terrainChangeMinActiveKeyframes", terrainChangeMinActiveKeyframes, 5);
        nh.param<int>("tcrlm/terrainChangeMinCellPoints", terrainChangeMinCellPoints, 4);
        nh.param<float>("tcrlm/terrainChangeTauUp", terrainChangeTauUp, 0.35);
        nh.param<float>("tcrlm/terrainChangeTauDown", terrainChangeTauDown, 0.35);
        nh.param<int>("tcrlm/terrainChangeMinLayerPoints", terrainChangeMinLayerPoints, 5);
        nh.param<float>("tcrlm/terrainChangeLayerBand", terrainChangeLayerBand, 0.15);
        nh.param<float>("tcrlm/terrainChangeMinLayerTimeSeparation", terrainChangeMinLayerTimeSeparation, 3.0);
        nh.param<float>("tcrlm/terrainChangeWriteBand", terrainChangeWriteBand, 0.15);
        nh.param<float>("tcrlm/terrainChangeRecentBand", terrainChangeRecentBand, 0.25);
        nh.param<int>("tcrlm/terrainChangeMinRegionCells", terrainChangeMinRegionCells, 4);
        nh.param<float>("tcrlm/terrainChangeMinRegionArea", terrainChangeMinRegionArea, 1.0);
        nh.param<int>("tcrlm/terrainChangeRequiredConfirmations", terrainChangeRequiredConfirmations, 1);
        nh.param<float>("tcrlm/terrainChangeWeightDropPerConfirmation", terrainChangeWeightDropPerConfirmation, 0.01);
        nh.param<float>("tcrlm/terrainChangeMinFeatureWeight", terrainChangeMinFeatureWeight, 0.2);
        nh.param<bool>("tcrlm/terrainChangeLogSummary", terrainChangeLogSummary, false);
        nh.param<bool>("tcrlm/terrainChangePublishCandidateClouds", terrainChangePublishCandidateClouds, true);

        terrainChangeGridResolution = std::max(0.05f, terrainChangeGridResolution);
        terrainChangeRequiredConfirmations = std::max(1, terrainChangeRequiredConfirmations);
        terrainChangeMinFeatureWeight = std::min(1.0f, std::max(0.0f, terrainChangeMinFeatureWeight));
        terrainChangeWeightDropPerConfirmation = std::min(1.0f, std::max(0.0f, terrainChangeWeightDropPerConfirmation));

        subSnapshot = nh.subscribe("tcrlm/mapping/terrain_change_snapshot", 1,
                                   &TerrainChangeNode::snapshotHandler, this,
                                   ros::TransportHints().tcpNoDelay());
        pubUpdates = nh.advertise<tcrlm::TerrainChangeUpdateArray>("tcrlm/terrain_change/keyframe_updates", 1);
        pubStalePoints = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/terrain_change/changed_points_candidate", 1, true);

        ROS_WARN_STREAM("[terrain_change_node] async detector ready, axis=" << terrainChangeHeightAxis
                        << ", grid=" << terrainChangeGridResolution);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber subSnapshot;
    ros::Publisher pubUpdates;
    ros::Publisher pubStalePoints;

    std::string odometryFrame;
    std::string terrainChangeHeightAxis;
    std::string terrainChangeDetectionMode;
    float terrainChangeGridResolution = 1.0f;
    int terrainChangeMinActiveKeyframes = 5;
    int terrainChangeMinCellPoints = 4;
    float terrainChangeTauUp = 0.35f;
    float terrainChangeTauDown = 0.35f;
    int terrainChangeMinLayerPoints = 5;
    float terrainChangeLayerBand = 0.15f;
    float terrainChangeMinLayerTimeSeparation = 3.0f;
    float terrainChangeWriteBand = 0.15f;
    float terrainChangeRecentBand = 0.25f;
    int terrainChangeMinRegionCells = 4;
    float terrainChangeMinRegionArea = 1.0f;
    int terrainChangeRequiredConfirmations = 1;
    float terrainChangeWeightDropPerConfirmation = 0.01f;
    float terrainChangeMinFeatureWeight = 0.2f;
    bool terrainChangeLogSummary = false;
    bool terrainChangePublishCandidateClouds = true;

    std::unordered_map<CandidateKey, CandidateTrack, CandidateKeyHash> tracks;
    std::unordered_map<CandidateKey, CandidateCell, CandidateKeyHash> persistentCells;
    std::unordered_map<std::string, PointType> accumulatedStalePoints;
    uint32_t nextRegionId = 1;

    void heightCoordinates(const tcrlm::TerrainChangePoint& p, float& h, float& a, float& b) const
    {
        if (terrainChangeHeightAxis == "x")
        {
            h = p.x; a = p.y; b = p.z;
        }
        else if (terrainChangeHeightAxis == "-x")
        {
            h = -p.x; a = p.y; b = p.z;
        }
        else if (terrainChangeHeightAxis == "y")
        {
            h = p.y; a = p.x; b = p.z;
        }
        else if (terrainChangeHeightAxis == "-y")
        {
            h = -p.y; a = p.x; b = p.z;
        }
        else if (terrainChangeHeightAxis == "-z")
        {
            h = -p.z; a = p.x; b = p.y;
        }
        else
        {
            h = p.z; a = p.x; b = p.y;
        }
    }

    GridIndex gridIndexForPoint(const tcrlm::TerrainChangePoint& p, float* heightOut = nullptr) const
    {
        float h = 0.0f, a = 0.0f, b = 0.0f;
        heightCoordinates(p, h, a, b);
        if (heightOut)
            *heightOut = h;
        GridIndex index;
        index.u = static_cast<int>(std::floor(a / terrainChangeGridResolution));
        index.v = static_cast<int>(std::floor(b / terrainChangeGridResolution));
        return index;
    }

    double median(std::vector<double>& values) const
    {
        if (values.empty())
            return 0.0;
        const std::size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        double result = values[mid];
        if (values.size() % 2 == 0)
        {
            std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
            result = 0.5 * (result + values[mid - 1]);
        }
        return result;
    }

    std::vector<Region> buildRegions(const std::vector<CandidateCell>& cells) const
    {
        std::unordered_map<GridIndex, CandidateCell, GridIndexHash> cellMap;
        for (const auto& cell : cells)
            cellMap[cell.index] = cell;

        std::unordered_set<GridIndex, GridIndexHash> visited;
        std::vector<Region> regions;
        const int du[4] = {1, -1, 0, 0};
        const int dv[4] = {0, 0, 1, -1};

        for (const auto& item : cellMap)
        {
            if (visited.find(item.first) != visited.end())
                continue;

            Region region;
            region.change_type = item.second.change_type;
            std::queue<GridIndex> q;
            q.push(item.first);
            visited.insert(item.first);

            while (!q.empty())
            {
                const GridIndex cur = q.front();
                q.pop();
                region.cells.push_back(cellMap.at(cur));

                for (int k = 0; k < 4; ++k)
                {
                    GridIndex next;
                    next.u = cur.u + du[k];
                    next.v = cur.v + dv[k];
                    if (visited.find(next) != visited.end())
                        continue;
                    const auto nextIt = cellMap.find(next);
                    if (nextIt == cellMap.end())
                        continue;
                    if (!tcrlm::terrain_change::sameTypedFourNeighbor(
                            cur.u, cur.v, region.change_type,
                            next.u, next.v, nextIt->second.change_type))
                        continue;
                    visited.insert(next);
                    q.push(next);
                }
            }

            const float area = region.cells.size() * terrainChangeGridResolution * terrainChangeGridResolution;
            if ((int)region.cells.size() >= terrainChangeMinRegionCells && area >= terrainChangeMinRegionArea)
                regions.push_back(region);
        }
        return regions;
    }

    float suggestedWeight(const tcrlm::TerrainChangePoint& point) const
    {
        return tcrlm::terrain_change::progressiveWeight(
            point.confirmations, terrainChangeWeightDropPerConfirmation,
            terrainChangeMinFeatureWeight);
    }

    void snapshotHandler(const tcrlm::TerrainChangeSnapshotConstPtr& msg)
    {
        const auto wallStart = ros::WallTime::now();
        std::unordered_set<uint32_t> keyframeIds;
        std::unordered_map<GridIndex, Cell, GridIndexHash> grid;
        std::unordered_map<GridIndex, Cell, GridIndexHash> allGrid;
        int fullPoints = 0;
        int cornerPoints = 0;
        int surfPoints = 0;

        for (const auto& point : msg->points)
        {
            keyframeIds.insert(point.keyframe_id);
            if (point.cloud_type == tcrlm::TerrainChangePoint::CLOUD_FULL)
                fullPoints++;
            else if (point.cloud_type == tcrlm::TerrainChangePoint::CLOUD_CORNER)
                cornerPoints++;
            else if (point.cloud_type == tcrlm::TerrainChangePoint::CLOUD_SURF)
                surfPoints++;

            if (point.cloud_type != tcrlm::TerrainChangePoint::CLOUD_FULL)
                continue;

            float h = 0.0f;
            const GridIndex index = gridIndexForPoint(point, &h);
            PointSource source;
            source.point = &point;
            source.h = h;
            source.time = point.time;
            auto appendSource = [&](Cell& cell) {
                cell.sources.push_back(source);
                cell.heights.push_back(h);
                cell.h_low = std::min(cell.h_low, h);
                cell.h_high = std::max(cell.h_high, h);
            };
            appendSource(allGrid[index]);
            if (point.state != tcrlm::TerrainChangePoint::STALE_OLD)
                appendSource(grid[index]);
        }

        if ((int)keyframeIds.size() < terrainChangeMinActiveKeyframes)
            return;

        std::vector<CandidateCell> confirmedCells;
        int rawCandidates = 0;
        for (const auto& item : grid)
        {
            const Cell& cell = item.second;
            if ((int)cell.heights.size() < terrainChangeMinCellPoints)
                continue;

            const float heightGap = cell.h_high - cell.h_low;

            std::vector<double> lowerTimes;
            std::vector<double> upperTimes;
            for (const auto& source : cell.sources)
            {
                if (source.h <= cell.h_low + terrainChangeLayerBand)
                    lowerTimes.push_back(source.time);
                if (source.h >= cell.h_high - terrainChangeLayerBand)
                    upperTimes.push_back(source.time);
            }

            if ((int)lowerTimes.size() < terrainChangeMinLayerPoints ||
                (int)upperTimes.size() < terrainChangeMinLayerPoints)
                continue;

            const double lowerTime = median(lowerTimes);
            const double upperTime = median(upperTimes);
            const double layerTimeDiff = upperTime - lowerTime;
            const auto direction = tcrlm::terrain_change::classifyDirection(
                heightGap, layerTimeDiff, terrainChangeMinLayerTimeSeparation,
                terrainChangeTauUp, terrainChangeTauDown);
            if (direction == tcrlm::terrain_change::Direction::none)
                continue;
            const uint8_t changeType = static_cast<uint8_t>(direction);
            if ((terrainChangeDetectionMode == "excavation" &&
                 changeType != tcrlm::TerrainChangeUpdate::CHANGE_EXCAVATION) ||
                (terrainChangeDetectionMode == "accumulation" &&
                 changeType != tcrlm::TerrainChangeUpdate::CHANGE_ACCUMULATION))
                continue;

            rawCandidates++;
            CandidateKey key;
            key.index = item.first;
            key.change_type = changeType;
            CandidateTrack& track = tracks[key];
            if (track.last_snapshot_id + 1 == msg->snapshot_id)
                track.count++;
            else
                track.count = 1;
            track.last_snapshot_id = msg->snapshot_id;

            if (track.count < terrainChangeRequiredConfirmations)
                continue;

            CandidateCell candidate;
            candidate.index = item.first;
            candidate.change_type = changeType;
            candidate.h_recent = changeType == tcrlm::TerrainChangeUpdate::CHANGE_ACCUMULATION
                ? cell.h_high : cell.h_low;
            candidate.h_history = changeType == tcrlm::TerrainChangeUpdate::CHANGE_ACCUMULATION
                ? cell.h_low : cell.h_high;
            confirmedCells.push_back(candidate);
        }




        const uint32_t trackTtl = static_cast<uint32_t>(
            std::max(5, terrainChangeRequiredConfirmations * 3));
        for (auto it = tracks.begin(); it != tracks.end(); )
        {
            if (msg->snapshot_id > it->second.last_snapshot_id + trackTtl)
                it = tracks.erase(it);
            else
                ++it;
        }

        const std::vector<Region> regions = buildRegions(confirmedCells);
        std::unordered_map<GridIndex, CandidateCell, GridIndexHash> acceptedCells;
        int acceptedRegions = 0;
        for (const auto& region : regions)
        {
            int regionOldPointCount = 0;
            for (const auto& cell : region.cells)
            {
                auto gridIt = grid.find(cell.index);
                if (gridIt == grid.end())
                    continue;
                for (const auto& source : gridIt->second.sources)
                    if ((cell.change_type == tcrlm::TerrainChangeUpdate::CHANGE_EXCAVATION &&
                         source.h > cell.h_recent + terrainChangeWriteBand) ||
                        (cell.change_type == tcrlm::TerrainChangeUpdate::CHANGE_ACCUMULATION &&
                         source.h < cell.h_recent - terrainChangeWriteBand))
                        regionOldPointCount++;
            }
            const int minRegionOldPoints = std::max(terrainChangeMinLayerPoints,
                                                     (int)region.cells.size() * 2);
            if (regionOldPointCount < minRegionOldPoints)
                continue;

            const uint32_t regionId = nextRegionId++;
            for (auto cell : region.cells)
            {
                cell.region_id = regionId;
                acceptedCells[cell.index] = cell;
                CandidateKey key;
                key.index = cell.index;
                key.change_type = cell.change_type;
                CandidateKey opposite = key;
                opposite.change_type = cell.change_type == tcrlm::TerrainChangeUpdate::CHANGE_EXCAVATION
                    ? tcrlm::TerrainChangeUpdate::CHANGE_ACCUMULATION
                    : tcrlm::TerrainChangeUpdate::CHANGE_EXCAVATION;
                persistentCells.erase(opposite);
                persistentCells[key] = cell;
            }
            acceptedRegions++;
        }




        for (auto it = persistentCells.begin(); it != persistentCells.end(); )
        {
            const CandidateCell& cell = it->second;
            if (acceptedCells.find(cell.index) != acceptedCells.end())
            {
                ++it;
                continue;
            }
            const auto allIt = allGrid.find(cell.index);
            if (allIt == allGrid.end())
            {
                it = persistentCells.erase(it);
                continue;
            }
            int recentActive = 0;
            int staleOld = 0;
            for (const auto& source : allIt->second.sources)
            {
                if (source.point->state != tcrlm::TerrainChangePoint::STALE_OLD &&
                    std::fabs(source.h - cell.h_recent) <= terrainChangeRecentBand)
                    ++recentActive;
                if (source.point->state == tcrlm::TerrainChangePoint::STALE_OLD &&
                    ((cell.change_type == tcrlm::TerrainChangeUpdate::CHANGE_EXCAVATION &&
                      source.h > cell.h_recent + terrainChangeWriteBand) ||
                     (cell.change_type == tcrlm::TerrainChangeUpdate::CHANGE_ACCUMULATION &&
                      source.h < cell.h_recent - terrainChangeWriteBand)))
                    ++staleOld;
            }
            if (recentActive >= terrainChangeMinLayerPoints && staleOld > 0)
            {
                acceptedCells[cell.index] = cell;
                ++it;
            }
            else
            {
                it = persistentCells.erase(it);
            }
        }

        tcrlm::TerrainChangeUpdateArray updates;
        updates.header = msg->header;
        updates.snapshot_id = msg->snapshot_id;



        std::unordered_set<std::string> emitted;
        for (const auto& point : msg->points)
        {
            if (point.cloud_type != tcrlm::TerrainChangePoint::CLOUD_FULL)
                continue;

            float h = 0.0f;
            const GridIndex index = gridIndexForPoint(point, &h);
            const auto cellIt = acceptedCells.find(index);
            if (cellIt == acceptedCells.end())
                continue;
            const CandidateCell& cell = cellIt->second;
            const bool isOldSurface =
                (cell.change_type == tcrlm::TerrainChangeUpdate::CHANGE_EXCAVATION &&
                 h > cell.h_recent + terrainChangeWriteBand) ||
                (cell.change_type == tcrlm::TerrainChangeUpdate::CHANGE_ACCUMULATION &&
                 h < cell.h_recent - terrainChangeWriteBand);
            if (!isOldSurface)
                continue;
            const std::string key = std::to_string(point.keyframe_id) + ":" +
                std::to_string(point.cloud_type) + ":" + std::to_string(point.point_index);
            if (emitted.find(key) != emitted.end())
                continue;
            emitted.insert(key);

            tcrlm::TerrainChangeUpdate update;
            update.keyframe_id = point.keyframe_id;
            update.point_index = point.point_index;
            update.cloud_type = point.cloud_type;
            update.change_type = cell.change_type;
            update.region_id = cell.region_id;
            update.confirmation_delta = 1;
            update.suggested_weight = suggestedWeight(point);
            updates.updates.push_back(update);

            PointType out;
            out.x = point.x;
            out.y = point.y;
            out.z = point.z;
            out.intensity = update.suggested_weight;
            accumulatedStalePoints[key] = out;
        }

        pubUpdates.publish(updates);

        if (terrainChangePublishCandidateClouds)
        {
            pcl::PointCloud<PointType> candidateCloud;
            candidateCloud.reserve(accumulatedStalePoints.size());
            for (const auto& item : accumulatedStalePoints)
                candidateCloud.push_back(item.second);
            sensor_msgs::PointCloud2 staleMsg;
            pcl::toROSMsg(candidateCloud, staleMsg);
            staleMsg.header = msg->header;
            if (staleMsg.header.frame_id.empty())
                staleMsg.header.frame_id = odometryFrame;
            pubStalePoints.publish(staleMsg);
        }

        if (terrainChangeLogSummary)
        {
            const double elapsedMs = (ros::WallTime::now() - wallStart).toSec() * 1000.0;
            ROS_WARN_STREAM("[terrain_change_node] snapshot=" << msg->snapshot_id
                            << ", keyframes=" << keyframeIds.size()
                            << ", full=" << fullPoints
                            << ", corner=" << cornerPoints
                            << ", surf=" << surfPoints
                            << ", cells=" << grid.size()
                            << ", raw_candidates=" << rawCandidates
                            << ", confirmed_cells=" << confirmedCells.size()
                            << ", regions=" << regions.size()
                            << ", accepted_regions=" << acceptedRegions
                            << ", updates=" << updates.updates.size()
                            << ", candidate_points=" << accumulatedStalePoints.size()
                            << ", elapsed_ms=" << elapsedMs);
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tcrlm_terrain_change");
    TerrainChangeNode node;
    ros::spin();
    return 0;
}
