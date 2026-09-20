#include "utility.h"

namespace
{
constexpr int kRejectedFeatureLabel = 2;
}
#include "tcrlm/cloud_info.h"

#include <cstdint>
#include <cstring>
#include <unordered_map>

struct smoothness_t{
    float value;
    size_t ind;
};

struct by_value{
    bool operator()(smoothness_t const &left, smoothness_t const &right) {
        return left.value < right.value;
    }
};

struct FeaturePostProcessConfig
{
    bool enable = false;
    int postWindowCols = 10;
    int postWindowRings = 2;
    int trendWindowCols = 24;
    int minValidPerSide = 4;
    int missingRunThreshold = 3;
    int scanEdgeMargin = 20;
    float nearRangeThreshold = 15.0f;
    int densityBeforeWindowCols = 4;
    int zeroDensityWindowCols = 8;
    float gapDensityThreshold = 0.65f;
    float validDensityThreshold = 0.60f;
    float zeroDensityThreshold = 0.60f;
    int sameRingLabel1ColWindow = 12;
    int crossRingColWindow = 8;
    float rangeJumpAbs = 0.8f;
    float rangeJumpRel = 0.08f;
    float spacingJumpRatio = 2.0f;
    int verticalSupportColRadius = 2;
    int maskNeighborCols = 2;
    int maskNeighborRings = 1;
    int maskBoundaryCols = 4;
    int maskBoundaryRings = 1;
    float invalidSeedDensityThreshold = 0.05f;
    bool maskInvalidSeeds = true;
    bool columnNmsEnable = false;
    int columnNmsBinCols = 4;
    int columnNmsMaxCorners = 6;
    bool logSummary = true;
};

struct BevBalancedEdgeConfig
{
    bool enable = true;
    float gridSize = 2.0f;
    int maxPerCell = 4;
    int maxTotal = 0;
    float minRange = 0.0f;
    bool logSummary = true;
};

struct RangeImageIndex
{
    std::vector<int> pointRing;
    std::vector<int> ringStart;
    std::vector<int> ringEnd;
    std::vector<int> pointGrid;
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subPreprocessFilterMask;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;
    ros::Publisher pubRejectedFeaturePoints;
    ros::Publisher pubCorrectedCloud;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;
    pcl::PointCloud<PointType>::Ptr rejectedFeatureCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;
    Eigen::Affine3f levelingTransform = Eigen::Affine3f::Identity();

    tcrlm::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;
    std::deque<sensor_msgs::PointCloud2> preprocessFilterMaskQueue;
    bool usePreprocessFilterMask = false;
    std::string preprocessFilterMaskTopic = "/tcrlm/preprocess/cloud_filter_mask";
    double preprocessFilterMaskSyncTolerance = 0.05;
    int featureCloudInfoQueueSize = 1;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;
    FeaturePostProcessConfig featurePostCfg;
    BevBalancedEdgeConfig bevBalancedEdgeCfg;
    std::vector<char> featureInvalidSeedMask;
    std::vector<char> featureRejectMask;
    std::vector<char> preprocessFilterRejectedGrid;

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<tcrlm::cloud_info>("tcrlm/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        nh.param<bool>("tcrlm/usePreprocessFilterMask", usePreprocessFilterMask, false);
        nh.param<std::string>("tcrlm/preprocessFilterMaskTopic", preprocessFilterMaskTopic, "/tcrlm/preprocess/cloud_filter_mask");
        nh.param<double>("tcrlm/preprocessFilterMaskSyncTolerance", preprocessFilterMaskSyncTolerance, 0.05);
        nh.param<int>("tcrlm/featureCloudInfoQueueSize", featureCloudInfoQueueSize, 1);
        featureCloudInfoQueueSize = std::max(1, featureCloudInfoQueueSize);
        if (usePreprocessFilterMask)
        {
            subPreprocessFilterMask = nh.subscribe<sensor_msgs::PointCloud2>(
                preprocessFilterMaskTopic, 5, &FeatureExtraction::preprocessFilterMaskHandler, this, ros::TransportHints().tcpNoDelay());
            ROS_INFO_STREAM("Feature post-process will use preprocess filter mask topic: " << preprocessFilterMaskTopic);
        }

        pubLaserCloudInfo = nh.advertise<tcrlm::cloud_info> ("tcrlm/feature/cloud_info", featureCloudInfoQueueSize);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/feature/cloud_surface", 1);
        pubRejectedFeaturePoints = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/feature/cloud_rejected", 1);
        pubCorrectedCloud = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/feature/cloud_corrected", 1);

        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);
        featureInvalidSeedMask.resize(N_SCAN*Horizon_SCAN, 0);
        featureRejectMask.resize(N_SCAN*Horizon_SCAN, 0);
        preprocessFilterRejectedGrid.resize(N_SCAN*Horizon_SCAN, 0);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());
        rejectedFeatureCloud.reset(new pcl::PointCloud<PointType>());

        if (enableFeatureCloudLeveling)
        {
            const float roll = featureCloudLevelingRollDeg * static_cast<float>(M_PI) / 180.0f;
            const float pitch = featureCloudLevelingPitchDeg * static_cast<float>(M_PI) / 180.0f;
            const float yaw = featureCloudLevelingYawDeg * static_cast<float>(M_PI) / 180.0f;
            levelingTransform = pcl::getTransformation(0.0f, 0.0f, 0.0f, roll, pitch, yaw);
            ROS_INFO_STREAM("Sensor mount correction enabled with rpy_deg=("
                            << featureCloudLevelingRollDeg << ", "
                            << featureCloudLevelingPitchDeg << ", "
                            << featureCloudLevelingYawDeg << ")");
        }

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];

        loadFeaturePostProcessConfig();
        loadBevBalancedEdgeConfig();
    }

    template <typename T>
    void loadFeaturePostParam(const std::string& name,
                              T& value,
                              const T& defaultValue)
    {
        nh.param<T>("tcrlm/" + name, value, defaultValue);
    }

    void loadFeaturePostProcessConfig()
    {
        loadFeaturePostParam<bool>("featurePostProcessEnable", featurePostCfg.enable, false);
        loadFeaturePostParam<int>("featurePostProcessWindowCols", featurePostCfg.postWindowCols, 10);
        loadFeaturePostParam<int>("featurePostProcessWindowRings", featurePostCfg.postWindowRings, 2);
        loadFeaturePostParam<int>("featurePostProcessTrendCols", featurePostCfg.trendWindowCols, 24);
        loadFeaturePostParam<int>("featurePostProcessMinValidPerSide", featurePostCfg.minValidPerSide, 4);
        loadFeaturePostParam<int>("featurePostProcessMissingRunThreshold", featurePostCfg.missingRunThreshold, 3);
        loadFeaturePostParam<int>("featurePostProcessScanEdgeMargin", featurePostCfg.scanEdgeMargin, 20);
        loadFeaturePostParam<float>("featurePostProcessNearRangeThreshold", featurePostCfg.nearRangeThreshold, 15.0f);
        loadFeaturePostParam<int>("featurePostProcessDensityBeforeCols", featurePostCfg.densityBeforeWindowCols, 4);
        loadFeaturePostParam<int>("featurePostProcessZeroDensityCols", featurePostCfg.zeroDensityWindowCols, 8);
        loadFeaturePostParam<float>("featurePostProcessGapDensityThreshold", featurePostCfg.gapDensityThreshold, 0.65f);
        loadFeaturePostParam<float>("featurePostProcessValidDensityThreshold", featurePostCfg.validDensityThreshold, 0.60f);
        loadFeaturePostParam<float>("featurePostProcessZeroDensityThreshold", featurePostCfg.zeroDensityThreshold, 0.60f);
        loadFeaturePostParam<int>("featurePostProcessSameRingLabel1Cols", featurePostCfg.sameRingLabel1ColWindow, 12);
        loadFeaturePostParam<int>("featurePostProcessCrossRingCols", featurePostCfg.crossRingColWindow, 8);
        loadFeaturePostParam<float>("featurePostProcessRangeJumpAbs", featurePostCfg.rangeJumpAbs, 0.8f);
        loadFeaturePostParam<float>("featurePostProcessRangeJumpRel", featurePostCfg.rangeJumpRel, 0.08f);
        loadFeaturePostParam<float>("featurePostProcessSpacingJumpRatio", featurePostCfg.spacingJumpRatio, 2.0f);
        loadFeaturePostParam<int>("featurePostProcessVerticalSupportColRadius", featurePostCfg.verticalSupportColRadius, 2);
        nh.param<int>("tcrlm/featurePostProcessMaskNeighborCols", featurePostCfg.maskNeighborCols, 2);
        nh.param<int>("tcrlm/featurePostProcessMaskNeighborRings", featurePostCfg.maskNeighborRings, 1);
        nh.param<int>("tcrlm/featurePostProcessMaskBoundaryCols", featurePostCfg.maskBoundaryCols, 4);
        nh.param<int>("tcrlm/featurePostProcessMaskBoundaryRings", featurePostCfg.maskBoundaryRings, 1);
        nh.param<float>("tcrlm/featurePostProcessInvalidSeedDensityThreshold", featurePostCfg.invalidSeedDensityThreshold, 0.05f);
        nh.param<bool>("tcrlm/featurePostProcessMaskInvalidSeeds", featurePostCfg.maskInvalidSeeds, true);
        loadFeaturePostParam<bool>("featurePostProcessColumnNmsEnable", featurePostCfg.columnNmsEnable, false);
        loadFeaturePostParam<int>("featurePostProcessColumnNmsBinCols", featurePostCfg.columnNmsBinCols, 4);
        loadFeaturePostParam<int>("featurePostProcessMaxCornersPerColumnBin", featurePostCfg.columnNmsMaxCorners, 6);
        loadFeaturePostParam<bool>("featurePostProcessLogSummary", featurePostCfg.logSummary, true);

        featurePostCfg.postWindowCols = std::max(1, featurePostCfg.postWindowCols);
        featurePostCfg.postWindowRings = std::max(1, featurePostCfg.postWindowRings);
        featurePostCfg.trendWindowCols = std::max(1, featurePostCfg.trendWindowCols);
        featurePostCfg.minValidPerSide = std::max(1, featurePostCfg.minValidPerSide);
        featurePostCfg.missingRunThreshold = std::max(1, featurePostCfg.missingRunThreshold);
        featurePostCfg.scanEdgeMargin = std::max(0, featurePostCfg.scanEdgeMargin);
        featurePostCfg.densityBeforeWindowCols = std::max(1, featurePostCfg.densityBeforeWindowCols);
        featurePostCfg.zeroDensityWindowCols = std::max(1, featurePostCfg.zeroDensityWindowCols);
        featurePostCfg.verticalSupportColRadius = std::max(0, featurePostCfg.verticalSupportColRadius);
        featurePostCfg.maskNeighborCols = std::max(0, featurePostCfg.maskNeighborCols);
        featurePostCfg.maskNeighborRings = std::max(0, featurePostCfg.maskNeighborRings);
        featurePostCfg.maskBoundaryCols = std::max(0, featurePostCfg.maskBoundaryCols);
        featurePostCfg.maskBoundaryRings = std::max(0, featurePostCfg.maskBoundaryRings);
        featurePostCfg.invalidSeedDensityThreshold = std::max(0.0f, std::min(featurePostCfg.invalidSeedDensityThreshold, 1.0f));
        featurePostCfg.columnNmsBinCols = std::max(1, featurePostCfg.columnNmsBinCols);
        featurePostCfg.columnNmsMaxCorners = std::max(1, featurePostCfg.columnNmsMaxCorners);

        if (featurePostCfg.enable || featurePostCfg.columnNmsEnable)
        {
            ROS_INFO_STREAM("Feature post-process enabled with window_cols=" << featurePostCfg.postWindowCols
                            << ", window_rings=" << featurePostCfg.postWindowRings
                            << ", trend_cols=" << featurePostCfg.trendWindowCols
                            << ", mask_neighbor_cols=" << featurePostCfg.maskNeighborCols
                            << ", mask_neighbor_rings=" << featurePostCfg.maskNeighborRings
                            << ", mask_boundary_cols=" << featurePostCfg.maskBoundaryCols
                            << ", mask_boundary_rings=" << featurePostCfg.maskBoundaryRings
                            << ", invalid_seed_density=" << featurePostCfg.invalidSeedDensityThreshold
                            << ", column_nms=" << featurePostCfg.columnNmsEnable
                            << ", column_nms_bin_cols=" << featurePostCfg.columnNmsBinCols
                            << ", max_corners_per_bin=" << featurePostCfg.columnNmsMaxCorners);
        }
    }

    void loadBevBalancedEdgeConfig()
    {
        nh.param<bool>("tcrlm/bevBalancedEdgeEnable", bevBalancedEdgeCfg.enable, true);
        nh.param<float>("tcrlm/bevBalancedEdgeGridSize", bevBalancedEdgeCfg.gridSize, 2.0f);
        nh.param<int>("tcrlm/bevBalancedEdgeMaxPerCell", bevBalancedEdgeCfg.maxPerCell, 4);
        nh.param<int>("tcrlm/bevBalancedEdgeMaxTotal", bevBalancedEdgeCfg.maxTotal, 0);
        nh.param<float>("tcrlm/bevBalancedEdgeMinRange", bevBalancedEdgeCfg.minRange, 0.0f);
        nh.param<bool>("tcrlm/bevBalancedEdgeLogSummary", bevBalancedEdgeCfg.logSummary, true);

        bevBalancedEdgeCfg.gridSize = std::max(0.05f, bevBalancedEdgeCfg.gridSize);
        bevBalancedEdgeCfg.maxPerCell = std::max(1, bevBalancedEdgeCfg.maxPerCell);
        bevBalancedEdgeCfg.maxTotal = std::max(0, bevBalancedEdgeCfg.maxTotal);
        bevBalancedEdgeCfg.minRange = std::max(0.0f, bevBalancedEdgeCfg.minRange);
    }

    struct MaskFieldLayout
    {
        int xOffset = -1, yOffset = -1, zOffset = -1, intensityOffset = -1;
        int labelOffset = -1, ringOffset = -1, timeOffset = -1, timestampOffset = -1;
        int xType = -1, yType = -1, zType = -1, intensityType = -1;
        int labelType = -1, ringType = -1, timeType = -1, timestampType = -1;
        bool hasXyz = false, hasIntensity = false, hasLabel = false;
        bool hasRing = false, hasTime = false, hasTimestamp = false;
    };

    static bool findMaskField(const sensor_msgs::PointCloud2& msg, const std::string& name,
                              int& offset, int& datatype)
    {
        for (const auto& field : msg.fields)
            if (field.name == name)
            {
                offset = static_cast<int>(field.offset);
                datatype = static_cast<int>(field.datatype);
                return true;
            }
        return false;
    }

    static bool readMaskFieldAsDouble(const uint8_t* ptr, int datatype, double& out)
    {
#define TCRLM_READ_MASK_FIELD(TYPE) do { TYPE value{}; std::memcpy(&value, ptr, sizeof(value)); out = static_cast<double>(value); return std::isfinite(out); } while (false)
        switch (datatype)
        {
        case sensor_msgs::PointField::INT8: TCRLM_READ_MASK_FIELD(int8_t);
        case sensor_msgs::PointField::UINT8: TCRLM_READ_MASK_FIELD(uint8_t);
        case sensor_msgs::PointField::INT16: TCRLM_READ_MASK_FIELD(int16_t);
        case sensor_msgs::PointField::UINT16: TCRLM_READ_MASK_FIELD(uint16_t);
        case sensor_msgs::PointField::INT32: TCRLM_READ_MASK_FIELD(int32_t);
        case sensor_msgs::PointField::UINT32: TCRLM_READ_MASK_FIELD(uint32_t);
        case sensor_msgs::PointField::FLOAT32: TCRLM_READ_MASK_FIELD(float);
        case sensor_msgs::PointField::FLOAT64: TCRLM_READ_MASK_FIELD(double);
        default: return false;
        }
#undef TCRLM_READ_MASK_FIELD
    }

    static MaskFieldLayout buildMaskFieldLayout(const sensor_msgs::PointCloud2& msg)
    {
        MaskFieldLayout layout;
        layout.hasXyz = findMaskField(msg, "x", layout.xOffset, layout.xType) &&
                        findMaskField(msg, "y", layout.yOffset, layout.yType) &&
                        findMaskField(msg, "z", layout.zOffset, layout.zType);
        layout.hasIntensity = findMaskField(msg, "intensity", layout.intensityOffset, layout.intensityType);
        layout.hasLabel = findMaskField(msg, "label", layout.labelOffset, layout.labelType);
        layout.hasRing = findMaskField(msg, "ring", layout.ringOffset, layout.ringType);
        layout.hasTime = findMaskField(msg, "time", layout.timeOffset, layout.timeType) ||
                         findMaskField(msg, "t", layout.timeOffset, layout.timeType);
        layout.hasTimestamp = findMaskField(msg, "timestamp", layout.timestampOffset, layout.timestampType);
        return layout;
    }

    void preprocessFilterMaskHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn)
    {
        preprocessFilterMaskQueue.push_back(*msgIn);
        while (preprocessFilterMaskQueue.size() > 10)
            preprocessFilterMaskQueue.pop_front();
    }

    bool findSyncedPreprocessFilterMask(sensor_msgs::PointCloud2& maskMsg)
    {
        if (!usePreprocessFilterMask || preprocessFilterMaskQueue.empty())
            return false;
        const double targetStamp = cloudHeader.stamp.toSec();
        while (!preprocessFilterMaskQueue.empty() &&
               preprocessFilterMaskQueue.front().header.stamp.toSec() < targetStamp - preprocessFilterMaskSyncTolerance)
            preprocessFilterMaskQueue.pop_front();
        double bestDiff = std::numeric_limits<double>::max();
        int bestIdx = -1;
        for (int i = 0; i < static_cast<int>(preprocessFilterMaskQueue.size()); ++i)
        {
            const double diff = std::fabs(preprocessFilterMaskQueue[i].header.stamp.toSec() - targetStamp);
            if (diff < bestDiff) { bestDiff = diff; bestIdx = i; }
        }
        if (bestIdx < 0 || bestDiff > preprocessFilterMaskSyncTolerance)
            return false;
        maskMsg = preprocessFilterMaskQueue[bestIdx];
        return true;
    }

    void updatePreprocessFilterMaskGrid()
    {
        preprocessFilterRejectedGrid.assign(N_SCAN * Horizon_SCAN, 0);
        sensor_msgs::PointCloud2 maskMsg;
        if (!findSyncedPreprocessFilterMask(maskMsg))
            return;
        const std::size_t count = static_cast<std::size_t>(maskMsg.width) * maskMsg.height;
        if (count == 0 || maskMsg.point_step == 0 || maskMsg.data.size() < count * maskMsg.point_step)
            return;
        const MaskFieldLayout layout = buildMaskFieldLayout(maskMsg);
        if (!layout.hasXyz || !layout.hasRing || (!layout.hasLabel && !layout.hasIntensity))
            return;

        double firstTimestamp = 0.0;
        bool firstTimestampSet = false;
        if (layout.hasTimestamp && !layout.hasTime)
            for (std::size_t i = 0; i < count; ++i)
                if (readMaskFieldAsDouble(maskMsg.data.data() + i * maskMsg.point_step + layout.timestampOffset,
                                          layout.timestampType, firstTimestamp))
                { firstTimestampSet = true; break; }

        for (std::size_t i = 0; i < count; ++i)
        {
            const uint8_t* src = maskMsg.data.data() + i * maskMsg.point_step;
            double label = 0.0;
            if (!(layout.hasLabel ? readMaskFieldAsDouble(src + layout.labelOffset, layout.labelType, label)
                                  : readMaskFieldAsDouble(src + layout.intensityOffset, layout.intensityType, label)) || label < 0.5)
                continue;
            double ringValue = 0.0;
            if (!readMaskFieldAsDouble(src + layout.ringOffset, layout.ringType, ringValue))
                continue;
            const int row = static_cast<int>(std::llround(ringValue));
            if (row < 0 || row >= N_SCAN || row % downsampleRate != 0)
                continue;

            int col = -1;
            if (projectionColumnMode == "time" || projectionColumnMode == "relative_time" || projectionColumnMode == "timestamp")
            {
                double relativeTime = 0.0;
                if (layout.hasTime)
                    readMaskFieldAsDouble(src + layout.timeOffset, layout.timeType, relativeTime);
                else if (layout.hasTimestamp && firstTimestampSet)
                {
                    double timestamp = 0.0;
                    if (readMaskFieldAsDouble(src + layout.timestampOffset, layout.timestampType, timestamp))
                        relativeTime = std::max(0.0, timestamp - firstTimestamp);
                }
                const double period = projectionTimeScanPeriod > 0.0f ? projectionTimeScanPeriod : 0.1;
                relativeTime = std::max(0.0, std::min(relativeTime, period - 1e-6));
                col = static_cast<int>(std::floor(relativeTime / period * Horizon_SCAN));
            }
            else
            {
                double x = 0.0, y = 0.0;
                if (!readMaskFieldAsDouble(src + layout.xOffset, layout.xType, x) ||
                    !readMaskFieldAsDouble(src + layout.yOffset, layout.yType, y))
                    continue;
                const float angle = std::atan2(static_cast<float>(x), static_cast<float>(y)) * 180.0f / M_PI;
                col = -std::round((angle - 90.0f) / (360.0f / Horizon_SCAN)) + Horizon_SCAN / 2;
                if (col >= Horizon_SCAN) col -= Horizon_SCAN;
            }
            if (col >= 0 && col < Horizon_SCAN)
                preprocessFilterRejectedGrid[row * Horizon_SCAN + col] = 1;
        }
    }

    void laserCloudInfoHandler(const tcrlm::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn;
        cloudHeader = msgIn->header;
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud);

        updatePreprocessFilterMaskGrid();

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        if (featureInvalidSeedMask.size() < static_cast<std::size_t>(cloudSize))
            featureInvalidSeedMask.resize(cloudSize, 0);
        if (featureRejectMask.size() < static_cast<std::size_t>(cloudSize))
            featureRejectMask.resize(cloudSize, 0);
        for (int i = 0; i < cloudSize; ++i)
        {
            cloudCurvature[i] = 0.0f;
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            cloudSmoothness[i].value = 0.0f;
            cloudSmoothness[i].ind = i;
        }

        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];

            cloudCurvature[i] = diffRange*diffRange;


            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    RangeImageIndex buildRangeImageIndex() const
    {
        const int cloudSize = extractedCloud->points.size();
        RangeImageIndex index;
        index.pointRing.assign(cloudSize, -1);
        index.ringStart.assign(N_SCAN, -1);
        index.ringEnd.assign(N_SCAN, -1);
        index.pointGrid.assign(N_SCAN * Horizon_SCAN, -1);

        int ring = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (i > 0 && cloudInfo.pointColInd[i] < cloudInfo.pointColInd[i - 1] && ring + 1 < N_SCAN)
                ++ring;

            index.pointRing[i] = ring;
            if (ring >= 0 && ring < N_SCAN)
            {
                if (index.ringStart[ring] == -1)
                    index.ringStart[ring] = i;
                index.ringEnd[ring] = i;

                int col = cloudInfo.pointColInd[i];
                if (col >= 0 && col < Horizon_SCAN)
                    index.pointGrid[ring * Horizon_SCAN + col] = i;
            }
        }

        return index;
    }

    void postProcessFeatures(const RangeImageIndex& rangeIndex)
    {
        if (!featurePostCfg.enable && !featurePostCfg.columnNmsEnable)
            return;

        const int cloudSize = extractedCloud->points.size();
        if (cloudSize == 0)
            return;

        featureRejectMask.assign(cloudSize, 0);
        if (featureInvalidSeedMask.size() < static_cast<std::size_t>(cloudSize))
            featureInvalidSeedMask.resize(cloudSize, 0);

        auto gridPointIndex = [&](int ring, int col) -> int {
            if (ring < 0 || ring >= N_SCAN || col < 0 || col >= Horizon_SCAN)
                return -1;
            return rangeIndex.pointGrid[ring * Horizon_SCAN + col];
        };

        auto isObserved = [&](int ring, int col) -> bool {
            return gridPointIndex(ring, col) >= 0;
        };

        auto labelAt = [&](int ring, int col) -> int {
            int idx = gridPointIndex(ring, col);
            if (idx < 0 || idx >= cloudSize)
                return kRejectedFeatureLabel;
            return cloudLabel[idx];
        };

        auto preprocessRejectedAt = [&](int ring, int col) -> bool {
            if (ring >= 0 && ring < N_SCAN && col >= 0 && col < Horizon_SCAN)
            {
                const int gridIdx = ring * Horizon_SCAN + col;
                if (gridIdx >= 0 &&
                    gridIdx < static_cast<int>(preprocessFilterRejectedGrid.size()) &&
                    preprocessFilterRejectedGrid[gridIdx])
                {
                    return true;
                }
            }
            return false;
        };

        auto invalidSeedAt = [&](int ring, int col) -> bool {
            if (!featurePostCfg.maskInvalidSeeds)
                return false;
            if (preprocessRejectedAt(ring, col))
                return true;
            int idx = gridPointIndex(ring, col);
            return idx >= 0 && idx < cloudSize && featureInvalidSeedMask[idx];
        };

        auto rangeAt = [&](int ring, int col) -> float {
            int idx = gridPointIndex(ring, col);
            if (idx < 0)
                return 0.0f;
            return cloudInfo.pointRange[idx];
        };

        auto nearPreprocessFilterProjectionBoundary = [&](int centerRing, int centerCol) -> bool {
            if (centerRing < 0 || centerRing >= N_SCAN || centerCol < 0 || centerCol >= Horizon_SCAN)
                return false;
            for (int ring = std::max(0, centerRing - featurePostCfg.maskBoundaryRings);
                 ring <= std::min(N_SCAN - 1, centerRing + featurePostCfg.maskBoundaryRings);
                 ++ring)
            {
                for (int col = std::max(0, centerCol - featurePostCfg.maskBoundaryCols);
                     col <= std::min(Horizon_SCAN - 1, centerCol + featurePostCfg.maskBoundaryCols);
                     ++col)
                {
                    if (ring == centerRing && col == centerCol)
                        continue;
                    if (preprocessRejectedAt(ring, col))
                        return true;
                }
            }
            return false;
        };

        auto markRejectNeighborhood = [&](int centerIdx) {
            if (centerIdx < 0 || centerIdx >= cloudSize)
                return;
            const int centerRing = rangeIndex.pointRing[centerIdx];
            const int centerCol = cloudInfo.pointColInd[centerIdx];
            if (centerRing < 0 || centerRing >= N_SCAN || centerCol < 0 || centerCol >= Horizon_SCAN)
                return;

            auto markGridNeighborhood = [&](int sourceRing, int sourceCol) {
                for (int ring = std::max(0, sourceRing - featurePostCfg.maskNeighborRings);
                     ring <= std::min(N_SCAN - 1, sourceRing + featurePostCfg.maskNeighborRings);
                     ++ring)
                {
                    for (int col = std::max(0, sourceCol - featurePostCfg.maskNeighborCols);
                         col <= std::min(Horizon_SCAN - 1, sourceCol + featurePostCfg.maskNeighborCols);
                         ++col)
                    {
                        const int idx = gridPointIndex(ring, col);
                        if (idx >= 0 && idx < cloudSize)
                            featureRejectMask[idx] = 1;
                    }
                }
            };

            markGridNeighborhood(centerRing, centerCol);
        };

        auto longestFalseRun = [&](int ring, int colStart, int colEnd) -> std::pair<int, int> {
            int observedCount = 0;
            int best = 0;
            int current = 0;
            for (int col = colStart; col < colEnd; ++col)
            {
                if (isObserved(ring, col))
                {
                    ++observedCount;
                    current = 0;
                }
                else
                {
                    ++current;
                    best = std::max(best, current);
                }
            }
            return std::make_pair(observedCount, best);
        };

        auto rangeJumpExceeds = [&](float candidateRange, const std::vector<float>& otherRanges) -> bool {
            if (otherRanges.empty())
                return false;
            const float nearest = *std::min_element(otherRanges.begin(), otherRanges.end());
            const float jump = candidateRange - nearest;
            return jump > featurePostCfg.rangeJumpAbs &&
                   jump > featurePostCfg.rangeJumpRel * std::max(candidateRange, 1e-6f);
        };

        auto nearestObservedNeighborCol = [&](int ring, int col, int direction, int maxWindow) -> int {
            for (int step = 1; step <= maxWindow; ++step)
            {
                const int candidateCol = col + direction * step;
                if (candidateCol < 0 || candidateCol >= Horizon_SCAN)
                    break;
                if (isObserved(ring, candidateCol))
                    return candidateCol;
            }
            return -1;
        };

        auto countVerticalSupport = [&](int ring, int col, int ringDirection) -> int {
            int support = 0;
            for (int step = 1; step <= featurePostCfg.postWindowRings; ++step)
            {
                const int ringCandidate = ring + ringDirection * step;
                if (ringCandidate < 0 || ringCandidate >= N_SCAN)
                    break;
                const int colStart = std::max(0, col - featurePostCfg.verticalSupportColRadius);
                const int colEnd = std::min(Horizon_SCAN, col + featurePostCfg.verticalSupportColRadius + 1);
                for (int candidateCol = colStart; candidateCol < colEnd; ++candidateCol)
                {
                    if (isObserved(ringCandidate, candidateCol))
                        ++support;
                }
            }
            return support;
        };

        struct SidePatternInfo
        {
            std::string pattern;
            std::vector<float> label1Ranges;
        };

        auto classifySidePattern = [&](int ring, int col, int direction) -> SidePatternInfo {
            SidePatternInfo info;
            info.pattern = "edge_open";
            for (int step = 1; step <= featurePostCfg.trendWindowCols; ++step)
            {
                const int candidateCol = col + direction * step;
                if (candidateCol < 0 || candidateCol >= Horizon_SCAN)
                    break;

                if (isObserved(ring, candidateCol))
                {
                    if (labelAt(ring, candidateCol) == 1)
                        info.label1Ranges.push_back(rangeAt(ring, candidateCol));
                    const int nextCol = candidateCol + direction;
                    info.pattern = (nextCol >= 0 && nextCol < Horizon_SCAN && isObserved(ring, nextCol)) ? "supported" : "hole";
                    return info;
                }
            }

            return info;
        };

        struct SideTrendStats
        {
            int edgeOpen = 0;
            int hole = 0;
            int supported = 0;
            int occluded = 0;
        };

        auto analyzeMultiringSideTrend = [&](int ring, int col, int direction, float candidateRange) -> SideTrendStats {
            SideTrendStats stats;
            for (int ringCandidate = std::max(0, ring - featurePostCfg.postWindowRings);
                 ringCandidate <= std::min(N_SCAN - 1, ring + featurePostCfg.postWindowRings);
                 ++ringCandidate)
            {
                if (!isObserved(ringCandidate, col))
                    continue;

                const SidePatternInfo sideInfo = classifySidePattern(ringCandidate, col, direction);
                if (sideInfo.pattern == "edge_open")
                    ++stats.edgeOpen;
                else if (sideInfo.pattern == "hole")
                    ++stats.hole;
                else
                    ++stats.supported;

                if (rangeJumpExceeds(candidateRange, sideInfo.label1Ranges))
                    ++stats.occluded;
            }
            return stats;
        };

        struct GapDensityInfo
        {
            float zeroDensity = 0.0f;
            float gapDensity = 0.0f;
            float label1Density = 0.0f;
            float invalidSeedDensity = 0.0f;
            float validBeforeRatio = 0.0f;
        };

        auto computeGapDensity = [&](int ring, int col, int direction) -> GapDensityInfo {
            int zeroCount = 0;
            int gapCount = 0;
            int label1Count = 0;
            int invalidSeedCount = 0;
            int zeroTotalCount = 0;
            int validBeforeCount = 0;
            int validBeforeTotal = 0;

            for (int ringCandidate = std::max(0, ring - featurePostCfg.postWindowRings);
                 ringCandidate <= std::min(N_SCAN - 1, ring + featurePostCfg.postWindowRings);
                 ++ringCandidate)
            {
                for (int step = 1; step <= featurePostCfg.densityBeforeWindowCols; ++step)
                {
                    const int backCol = col - direction * step;
                    if (backCol < 0 || backCol >= Horizon_SCAN)
                        continue;
                    ++validBeforeTotal;
                    if (isObserved(ringCandidate, backCol))
                        ++validBeforeCount;
                }

                for (int step = 1; step <= featurePostCfg.zeroDensityWindowCols; ++step)
                {
                    const int candidateCol = col + direction * step;
                    if (candidateCol < 0 || candidateCol >= Horizon_SCAN)
                        continue;

                    ++zeroTotalCount;
                    const bool observed = isObserved(ringCandidate, candidateCol);
                    const bool invalidSeed = invalidSeedAt(ringCandidate, candidateCol);
                    if (!observed || invalidSeed)
                        ++gapCount;
                    if (invalidSeed)
                        ++invalidSeedCount;
                    if (!observed)
                        ++zeroCount;
                    else if (labelAt(ringCandidate, candidateCol) == 1)
                        ++label1Count;
                }
            }

            GapDensityInfo info;
            info.zeroDensity = static_cast<float>(zeroCount) / std::max(zeroTotalCount, 1);
            info.gapDensity = static_cast<float>(gapCount) / std::max(zeroTotalCount, 1);
            info.label1Density = static_cast<float>(label1Count) / std::max(zeroTotalCount, 1);
            info.invalidSeedDensity = static_cast<float>(invalidSeedCount) / std::max(zeroTotalCount, 1);
            info.validBeforeRatio = static_cast<float>(validBeforeCount) / std::max(validBeforeTotal, 1);
            return info;
        };

        struct CornerEvaluation
        {
            bool trusted = true;
            bool nearInvalidSeedBoundary = false;
        };

        auto evaluateCornerCandidate = [&](int ring, int col) -> CornerEvaluation {
            CornerEvaluation evaluation;
            const float candidateRange = rangeAt(ring, col);
            std::vector<std::string> rejectReasons;
            std::vector<std::string> softRejectReasons;

            const bool nearPreprocessBoundary = nearPreprocessFilterProjectionBoundary(ring, col);
            if (nearPreprocessBoundary)
                rejectReasons.push_back("near_invalid_seed_projection_boundary");

            const int leftStart = std::max(0, col - featurePostCfg.postWindowCols);
            const auto leftStats = longestFalseRun(ring, leftStart, col);
            const bool leftSparse = leftStart < col &&
                                    (leftStats.first < featurePostCfg.minValidPerSide ||
                                     leftStats.second >= featurePostCfg.missingRunThreshold);

            const int rightEnd = std::min(Horizon_SCAN, col + featurePostCfg.postWindowCols + 1);
            const auto rightStats = longestFalseRun(ring, col + 1, rightEnd);
            const bool rightSparse = (col + 1) < rightEnd &&
                                     (rightStats.first < featurePostCfg.minValidPerSide ||
                                      rightStats.second >= featurePostCfg.missingRunThreshold);

            const bool nearScanEdge = col < featurePostCfg.scanEdgeMargin ||
                                      col >= Horizon_SCAN - featurePostCfg.scanEdgeMargin;

            if (leftSparse)
            {
                const SideTrendStats leftTrend = analyzeMultiringSideTrend(ring, col, -1, candidateRange);
                if (leftTrend.occluded >= 1)
                    softRejectReasons.push_back("left_multiring_label1_occlusion");
                else if (leftTrend.hole >= 1)
                    softRejectReasons.push_back("left_internal_hole_boundary");
                else if (leftTrend.edgeOpen >= 2 && nearScanEdge)
                    rejectReasons.push_back("left_scan_boundary");
                else
                    softRejectReasons.push_back("left_sparse_boundary");

                const GapDensityInfo leftGap = computeGapDensity(ring, col, -1);
                if (leftGap.validBeforeRatio >= featurePostCfg.validDensityThreshold &&
                    leftGap.gapDensity >= featurePostCfg.gapDensityThreshold &&
                    leftGap.invalidSeedDensity >= featurePostCfg.invalidSeedDensityThreshold)
                {
                    rejectReasons.push_back("left_near_invalid_seed_boundary");
                }
                else if (candidateRange < featurePostCfg.nearRangeThreshold &&
                         leftGap.validBeforeRatio >= featurePostCfg.validDensityThreshold &&
                    leftGap.gapDensity >= featurePostCfg.gapDensityThreshold)
                {
                    if (leftGap.zeroDensity >= featurePostCfg.zeroDensityThreshold)
                        softRejectReasons.push_back("left_near_zero_density_boundary");
                    else if (leftGap.label1Density >= 0.2f)
                        softRejectReasons.push_back("left_near_label1_gap_boundary");
                    else
                        softRejectReasons.push_back("left_near_gap_boundary");
                }
            }

            if (rightSparse)
            {
                const SideTrendStats rightTrend = analyzeMultiringSideTrend(ring, col, 1, candidateRange);
                if (rightTrend.occluded >= 1)
                    softRejectReasons.push_back("right_multiring_label1_occlusion");
                else if (rightTrend.hole >= 1)
                    softRejectReasons.push_back("right_internal_hole_boundary");
                else if (rightTrend.edgeOpen >= 2 && nearScanEdge)
                    rejectReasons.push_back("right_scan_boundary");
                else
                    softRejectReasons.push_back("right_sparse_boundary");

                const GapDensityInfo rightGap = computeGapDensity(ring, col, 1);
                if (rightGap.validBeforeRatio >= featurePostCfg.validDensityThreshold &&
                    rightGap.gapDensity >= featurePostCfg.gapDensityThreshold &&
                    rightGap.invalidSeedDensity >= featurePostCfg.invalidSeedDensityThreshold)
                {
                    rejectReasons.push_back("right_near_invalid_seed_boundary");
                }
                else if (candidateRange < featurePostCfg.nearRangeThreshold &&
                         rightGap.validBeforeRatio >= featurePostCfg.validDensityThreshold &&
                    rightGap.gapDensity >= featurePostCfg.gapDensityThreshold)
                {
                    if (rightGap.zeroDensity >= featurePostCfg.zeroDensityThreshold)
                        softRejectReasons.push_back("right_near_zero_density_boundary");
                    else if (rightGap.label1Density >= 0.2f)
                        softRejectReasons.push_back("right_near_label1_gap_boundary");
                    else
                        softRejectReasons.push_back("right_near_gap_boundary");
                }
            }

            const int leftCol = nearestObservedNeighborCol(ring, col, -1, featurePostCfg.postWindowCols);
            const int rightCol = nearestObservedNeighborCol(ring, col, 1, featurePostCfg.postWindowCols);
            if (leftCol >= 0 && rightCol >= 0)
            {
                const PointType& candidatePoint = extractedCloud->points[gridPointIndex(ring, col)];
                const PointType& leftPoint = extractedCloud->points[gridPointIndex(ring, leftCol)];
                const PointType& rightPoint = extractedCloud->points[gridPointIndex(ring, rightCol)];
                const float leftDist = pointDistance(candidatePoint, leftPoint);
                const float rightDist = pointDistance(candidatePoint, rightPoint);
                const float minDist = std::min(leftDist, rightDist);
                const float maxDist = std::max(leftDist, rightDist);
                if (minDist > 1e-6f && maxDist / minDist > featurePostCfg.spacingJumpRatio)
                    softRejectReasons.push_back("spacing_jump_boundary");
            }

            const int supportUp = countVerticalSupport(ring, col, -1);
            const int supportDown = countVerticalSupport(ring, col, 1);
            if (std::min(supportUp, supportDown) == 0 &&
                std::max(supportUp, supportDown) >= featurePostCfg.minValidPerSide)
            {
                softRejectReasons.push_back("vertical_support_boundary");
            }

            std::vector<float> sameRingLabel1Ranges;
            const int sameRingStart = std::max(0, col - featurePostCfg.sameRingLabel1ColWindow);
            const int sameRingEnd = std::min(Horizon_SCAN, col + featurePostCfg.sameRingLabel1ColWindow + 1);
            for (int candidateCol = sameRingStart; candidateCol < sameRingEnd; ++candidateCol)
            {
                if (isObserved(ring, candidateCol) && labelAt(ring, candidateCol) == 1)
                    sameRingLabel1Ranges.push_back(rangeAt(ring, candidateCol));
            }
            if (rangeJumpExceeds(candidateRange, sameRingLabel1Ranges))
                softRejectReasons.push_back("same_ring_label1_jump");

            std::vector<float> crossRingRanges;
            const int ringStart = std::max(0, ring - featurePostCfg.postWindowRings);
            const int ringEnd = std::min(N_SCAN - 1, ring + featurePostCfg.postWindowRings);
            const int colStart = std::max(0, col - featurePostCfg.crossRingColWindow);
            const int colEnd = std::min(Horizon_SCAN - 1, col + featurePostCfg.crossRingColWindow);
            for (int ringCandidate = ringStart; ringCandidate <= ringEnd; ++ringCandidate)
            {
                for (int candidateCol = colStart; candidateCol <= colEnd; ++candidateCol)
                {
                    if (ringCandidate == ring && candidateCol == col)
                        continue;
                    if (isObserved(ringCandidate, candidateCol) && labelAt(ringCandidate, candidateCol) == 1)
                        crossRingRanges.push_back(rangeAt(ringCandidate, candidateCol));
                }
            }
            if (crossRingRanges.size() >= 2 && rangeJumpExceeds(candidateRange, crossRingRanges))
                softRejectReasons.push_back("cross_ring_label1_jump");

            const bool hasNearInvalidSeedGap =
                std::find(rejectReasons.begin(), rejectReasons.end(), "left_near_invalid_seed_boundary") != rejectReasons.end() ||
                std::find(rejectReasons.begin(), rejectReasons.end(), "right_near_invalid_seed_boundary") != rejectReasons.end();

            evaluation.nearInvalidSeedBoundary = hasNearInvalidSeedGap || nearPreprocessBoundary;
            evaluation.trusted = rejectReasons.empty();
            return evaluation;
        };

        std::vector<int> cornerIndices;
        cornerIndices.reserve(cloudSize / 10);
        for (int i = 0; i < cloudSize; ++i)
        {
            if (cloudLabel[i] == 1)
                cornerIndices.push_back(i);
        }

        int ruleRejectedCount = 0;
        int invalidSeedCount = 0;
        int maskedPointCount = 0;
        int maskedCornerCount = 0;
        int maskedSurfaceCandidateCount = 0;
        if (featurePostCfg.enable)
        {
            for (int idx : cornerIndices)
            {
                const int ring = rangeIndex.pointRing[idx];
                const int col = cloudInfo.pointColInd[idx];
                const CornerEvaluation evaluation = evaluateCornerCandidate(ring, col);
                if (!evaluation.trusted)
                {
                    markRejectNeighborhood(idx);
                    ++ruleRejectedCount;
                    if (evaluation.nearInvalidSeedBoundary)
                        ++invalidSeedCount;
                }
            }

            for (int idx = 0; idx < cloudSize; ++idx)
            {
                if (!featureRejectMask[idx])
                    continue;

                ++maskedPointCount;
                if (cloudLabel[idx] == 1)
                    ++maskedCornerCount;
                else if (cloudLabel[idx] <= 0)
                    ++maskedSurfaceCandidateCount;

                cloudLabel[idx] = kRejectedFeatureLabel;
            }
        }

        int columnNmsRejectedCount = 0;
        if (featurePostCfg.columnNmsEnable)
        {
            const int binCount = (Horizon_SCAN + featurePostCfg.columnNmsBinCols - 1) / featurePostCfg.columnNmsBinCols;
            std::vector<std::vector<int>> columnBins(binCount);
            for (int idx : cornerIndices)
            {
                if (cloudLabel[idx] != 1)
                    continue;
                const int col = cloudInfo.pointColInd[idx];
                if (col < 0 || col >= Horizon_SCAN)
                    continue;
                const int bin = std::min(binCount - 1, col / featurePostCfg.columnNmsBinCols);
                columnBins[bin].push_back(idx);
            }

            for (std::vector<int>& binIndices : columnBins)
            {
                if (static_cast<int>(binIndices.size()) <= featurePostCfg.columnNmsMaxCorners)
                    continue;

                std::sort(binIndices.begin(), binIndices.end(), [&](int lhs, int rhs) {
                    return cloudCurvature[lhs] > cloudCurvature[rhs];
                });

                for (std::size_t i = static_cast<std::size_t>(featurePostCfg.columnNmsMaxCorners);
                     i < binIndices.size(); ++i)
                {
                    if (cloudLabel[binIndices[i]] == 1)
                    {
                        cloudLabel[binIndices[i]] = kRejectedFeatureLabel;
                        if (binIndices[i] >= 0 && binIndices[i] < static_cast<int>(featureRejectMask.size()))
                            featureRejectMask[binIndices[i]] = 1;
                        ++columnNmsRejectedCount;
                    }
                }
            }
        }

        if (featurePostCfg.logSummary)
        {
            int keptCount = 0;
            for (int idx : cornerIndices)
            {
                if (cloudLabel[idx] == 1)
                    ++keptCount;
            }

            ROS_INFO_STREAM_THROTTLE(5.0, "Feature post-process kept " << keptCount
                                     << " / " << cornerIndices.size()
                                     << " corners, rejected_rule=" << ruleRejectedCount
                                     << ", invalid_seeds=" << invalidSeedCount
                                     << ", masked_points=" << maskedPointCount
                                     << " (corners=" << maskedCornerCount
                                     << ", surface_candidates=" << maskedSurfaceCandidateCount
                                     << "), column_nms=" << columnNmsRejectedCount);
        }
    }

    void rebuildFeatureClouds(const RangeImageIndex& rangeIndex)
    {
        cornerCloud->clear();
        surfaceCloud->clear();
        rejectedFeatureCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        const int cloudSize = extractedCloud->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            if (i < static_cast<int>(featureRejectMask.size()) && featureRejectMask[i])
            {
                PointType rejectedPoint = extractedCloud->points[i];
                rejectedPoint.intensity = 255.0f;
                rejectedFeatureCloud->push_back(rejectedPoint);
            }

            if (cloudLabel[i] == 1)
                cornerCloud->push_back(extractedCloud->points[i]);
        }

        for (int ring = 0; ring < N_SCAN; ++ring)
        {
            if (rangeIndex.ringStart[ring] < 0 || rangeIndex.ringEnd[ring] < rangeIndex.ringStart[ring])
                continue;

            surfaceCloudScan->clear();
            for (int idx = rangeIndex.ringStart[ring]; idx <= rangeIndex.ringEnd[ring]; ++idx)
            {
                if (cloudLabel[idx] <= 0)
                    surfaceCloudScan->push_back(extractedCloud->points[idx]);
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);
            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void balanceEdgeFeaturesInBev()
    {
        if (!bevBalancedEdgeCfg.enable)
            return;

        std::vector<int> edgeIndices;
        edgeIndices.reserve(extractedCloud->size());
        for (int idx = 0; idx < static_cast<int>(extractedCloud->size()); ++idx)
        {
            if (cloudLabel[idx] == 1)
                edgeIndices.push_back(idx);
        }

        std::sort(edgeIndices.begin(), edgeIndices.end(), [&](int lhs, int rhs) {
            return cloudCurvature[lhs] > cloudCurvature[rhs];
        });

        auto gridKey = [&](const PointType& point) -> std::int64_t {
            const int gx = static_cast<int>(std::floor(point.x / bevBalancedEdgeCfg.gridSize));
            const int gy = static_cast<int>(std::floor(point.y / bevBalancedEdgeCfg.gridSize));
            return (static_cast<std::int64_t>(gx) << 32) ^ static_cast<std::uint32_t>(gy);
        };

        std::unordered_map<std::int64_t, int> cellEdgeCount;
        int keptCount = 0;
        int rejectedByCell = 0;
        int rejectedByTotal = 0;
        int rejectedByRange = 0;

        for (int idx : edgeIndices)
        {
            const PointType& point = extractedCloud->points[idx];
            const float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (range < bevBalancedEdgeCfg.minRange)
            {
                cloudLabel[idx] = kRejectedFeatureLabel;
                ++rejectedByRange;
                continue;
            }
            if (bevBalancedEdgeCfg.maxTotal > 0 && keptCount >= bevBalancedEdgeCfg.maxTotal)
            {
                cloudLabel[idx] = kRejectedFeatureLabel;
                ++rejectedByTotal;
                continue;
            }

            int& count = cellEdgeCount[gridKey(point)];
            if (count >= bevBalancedEdgeCfg.maxPerCell)
            {
                cloudLabel[idx] = kRejectedFeatureLabel;
                ++rejectedByCell;
                continue;
            }
            ++count;
            ++keptCount;
        }

        if (bevBalancedEdgeCfg.logSummary)
        {
            ROS_INFO_STREAM_THROTTLE(5.0, "BEV-balanced edge selection kept "
                << keptCount << " / " << edgeIndices.size()
                << " candidates, rejected_cell=" << rejectedByCell
                << ", rejected_total=" << rejectedByTotal
                << ", rejected_range=" << rejectedByRange);
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        auto markInvalidSeed = [&](int idx) {
            if (idx < 0 || idx >= cloudSize)
                return;
            cloudNeighborPicked[idx] = 1;
            featureInvalidSeedMask[idx] = 1;
        };


        for (int i = 5; i < cloudSize - 6; ++i)
        {

            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            if (columnDiff < 10){

                if (depth1 - depth2 > 0.3){
                    markInvalidSeed(i - 5);
                    markInvalidSeed(i - 4);
                    markInvalidSeed(i - 3);
                    markInvalidSeed(i - 2);
                    markInvalidSeed(i - 1);
                    markInvalidSeed(i);
                }else if (depth2 - depth1 > 0.3){
                    markInvalidSeed(i + 1);
                    markInvalidSeed(i + 2);
                    markInvalidSeed(i + 3);
                    markInvalidSeed(i + 4);
                    markInvalidSeed(i + 5);
                    markInvalidSeed(i + 6);
                }
            }

            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                markInvalidSeed(i);
        }
    }

    void extractFeatures()
    {
        for (int i = 0; i < N_SCAN; i++)
        {
            for (int j = 0; j < scanRingSegments; j++)
            {
                int sp = (cloudInfo.startRingIndex[i] * (scanRingSegments - j) + cloudInfo.endRingIndex[i] * j) / scanRingSegments;
                int ep = (cloudInfo.startRingIndex[i] * (scanRingSegments - 1 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / scanRingSegments - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= maxEdgePerSegment){
                            cloudLabel[ind] = 1;
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
            }
        }

        const RangeImageIndex rangeIndex = buildRangeImageIndex();
        postProcessFeatures(rangeIndex);
        balanceEdgeFeaturesInBev();
        rebuildFeatureClouds(rangeIndex);
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        if (enableFeatureCloudLeveling)
        {
            pcl::transformPointCloud(*extractedCloud, *extractedCloud, levelingTransform);
            pcl::transformPointCloud(*cornerCloud, *cornerCloud, levelingTransform);
            pcl::transformPointCloud(*surfaceCloud, *surfaceCloud, levelingTransform);
        }


        freeCloudInfoMemory();
        cloudInfo.cloud_deskewed = publishCloud(pubCorrectedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);

        cloudInfo.cloud_corner  = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        publishCloud(pubRejectedFeaturePoints, rejectedFeatureCloud, cloudHeader.stamp, lidarFrame);

        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "tcrlm_feature_extraction");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");

    ros::spin();

    return 0;
}
