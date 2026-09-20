#include "utility.h"
#include "tcrlm/cloud_info.h"
#include <boost/filesystem.hpp>
#include "tcrlm/save_map.h"
#include "tcrlm/TerrainChangeSnapshot.h"
#include "tcrlm/TerrainChangeUpdateArray.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <unordered_map>
#include <unordered_set>

using namespace gtsam;

using symbol_shorthand::X;
using symbol_shorthand::V;
using symbol_shorthand::B;
using symbol_shorthand::G;


struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


enum TerrainPointState : uint8_t
{
    VALID_STABLE = 0,
    CHANGE_CANDIDATE = 1,
    CHANGED_NEW = 2,
    STALE_OLD = 3,
    REJECTED_NON_TERRAIN = 4
};

enum TerrainChangeType : uint8_t
{
    CHANGE_NONE = 0,
    CHANGE_EXCAVATION = 1,
    CHANGE_ACCUMULATION = 2,
    CHANGE_UNKNOWN = 3
};

enum TerrainChangeRole : uint8_t
{
    ROLE_NONE = 0,
    ROLE_RECENT_SURFACE = 1,
    ROLE_OLD_SURFACE = 2
};

enum TerrainFeatureType : uint8_t
{
    TERRAIN_POINT_CORNER = 0,
    TERRAIN_POINT_SURF = 1,
    TERRAIN_POINT_FULL = 2,
    TERRAIN_POINT_BOTH = 255
};


struct ConstraintMatchDiagnostics
{
    int corner_used = 0;
    int surf_used = 0;
    int corner_affected = 0;
    int surf_affected = 0;
    double corner_raw_abs_res_sum = 0.0;
    double surf_raw_abs_res_sum = 0.0;
    double corner_weighted_abs_res_sum = 0.0;
    double surf_weighted_abs_res_sum = 0.0;
    double affected_raw_abs_res_sum = 0.0;
    double affected_weighted_abs_res_sum = 0.0;
    double affected_weight_sum = 0.0;

    void reset()
    {
        corner_used = 0;
        surf_used = 0;
        corner_affected = 0;
        surf_affected = 0;
        corner_raw_abs_res_sum = 0.0;
        surf_raw_abs_res_sum = 0.0;
        corner_weighted_abs_res_sum = 0.0;
        surf_weighted_abs_res_sum = 0.0;
        affected_raw_abs_res_sum = 0.0;
        affected_weighted_abs_res_sum = 0.0;
        affected_weight_sum = 0.0;
    }

    int totalUsed() const
    {
        return corner_used + surf_used;
    }

    int totalAffected() const
    {
        return corner_affected + surf_affected;
    }
};

struct TerrainStaleFullSet
{
    pcl::PointCloud<PointType>::Ptr local_points;
    std::vector<uint32_t> region_ids;
    std::vector<uint8_t> change_types;

    TerrainStaleFullSet()
    {
        local_points.reset(new pcl::PointCloud<PointType>());
    }
};


class mapOptimization : public ParamServer
{

public:

    bool useTerrainChangeMaintenance() const
    {
        return enableTerrainChangeMaintenance;
    }

    bool useTerrainChangeMapMaintenance() const
    {
        return useTerrainChangeMaintenance() && terrainChangeApplyToMapping;
    }

    std::string normalizedTerrainChangeMappingMode() const
    {
        std::string mode = terrainChangeMappingMode;
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return mode;
    }

    bool useTerrainChangeMapDownweighting() const
    {
        return useTerrainChangeMapMaintenance();
    }


    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;
    ros::Publisher pubTerrainChangeAllPoints;
    ros::Publisher pubTerrainChangeSnapshot;

    ros::Publisher pubSLAMInfo;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subGNSSHeading;
    ros::Subscriber subImuPosePrior;
    ros::Subscriber subLoop;
    ros::Subscriber subTerrainChangeUpdates;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    std::deque<sensor_msgs::Imu> gnssHeadingQueue;
    std::deque<nav_msgs::Odometry> imuPosePriorQueue;
    tcrlm::cloud_info cloudInfo;
    int mappingCloudInfoQueueSize = 1;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> fullCloudKeyFrames;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
    pcl::PointCloud<PointType>::Ptr laserCloudFullResDS;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec;
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<float> cornerMatchRawResidualVec;
    std::vector<float> cornerMatchFeatureWeightVec;
    std::vector<PointType> laserCloudOriSurfVec;
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;
    std::vector<float> surfMatchRawResidualVec;
    std::vector<float> surfMatchFeatureWeightVec;
    ConstraintMatchDiagnostics currentConstraintMatchDiag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    std::vector<std::vector<uint8_t>> cornerPointState;
    std::vector<std::vector<uint8_t>> cornerChangeMask;
    std::vector<std::vector<uint8_t>> cornerChangeType;
    std::vector<std::vector<uint8_t>> cornerChangeRole;
    std::vector<std::vector<uint32_t>> cornerChangeRegionId;
    std::vector<std::vector<uint32_t>> cornerChangeEpoch;
    std::vector<std::vector<uint16_t>> cornerChangeConfirmations;

    std::vector<std::vector<uint8_t>> surfPointState;
    std::vector<std::vector<uint8_t>> surfChangeMask;
    std::vector<std::vector<uint8_t>> surfChangeType;
    std::vector<std::vector<uint8_t>> surfChangeRole;
    std::vector<std::vector<uint32_t>> surfChangeRegionId;
    std::vector<std::vector<uint32_t>> surfChangeEpoch;
    std::vector<std::vector<uint16_t>> surfChangeConfirmations;

    std::vector<std::vector<uint8_t>> fullPointState;
    std::vector<std::vector<uint8_t>> fullChangeMask;
    std::vector<std::vector<uint8_t>> fullChangeType;
    std::vector<std::vector<uint8_t>> fullChangeRole;
    std::vector<std::vector<uint32_t>> fullChangeRegionId;
    std::vector<std::vector<uint32_t>> fullChangeEpoch;
    std::vector<std::vector<uint16_t>> fullChangeConfirmations;

    int terrainChangeEpoch = 0;
    double terrainChangeStartTime = -1.0;
    double lastTerrainChangeUpdateTime = -1.0;
    uint32_t terrainChangeSnapshotId = 0;
    std::mutex mtxTerrainChangeUpdates;
    std::vector<tcrlm::TerrainChangeUpdate> pendingTerrainChangeUpdates;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterTerrainChangeInput;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;

    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;
    bool lastFrameEnoughFeatures = true;
    bool enableReverseMotionGuard = true;
    float reverseMotionGuardCosThreshold = -0.2f;
    float reverseMotionGuardMinDelta = 0.05f;
    bool currentReverseMotionSuspected = false;
    bool currentOptimizationRejected = false;
    float currentMotionDirCos = 1.0f;
    double currentSlamComputeMs = 0.0;
    double currentSlamCycleMs = 0.0;
    bool hasAcceptedMotionDir = false;
    float lastAcceptedMotionDirX = 0.0f;
    float lastAcceptedMotionDirY = 0.0f;
    float lastAcceptedMotionDirZ = 0.0f;
    bool hasAcceptedPose = false;
    float lastAcceptedX = 0.0f;
    float lastAcceptedY = 0.0f;
    float lastAcceptedZ = 0.0f;
    float initialGuessTransform[6] = {0};

    std::mutex imuPosePriorLock;
    bool useImuPosePrior = false;
    bool useImuPosePriorInLM = true;
    bool useImuPosePriorInGraph = false;
    bool useImuPosePriorPosition = true;
    bool useImuPosePriorRollPitch = false;
    bool useImuPosePriorYaw = true;
    bool debugImuPosePrior = false;
    double imuPosePriorMaxAge = 0.20;
    double imuPosePriorPositionSigma = 0.30;
    double imuPosePriorRollPitchSigma = 0.20;
    double imuPosePriorYawSigma = 0.05;
    double imuPosePriorLmScale = 5.0;
    double imuPosePriorGraphScale = 1.0;
    double imuPosePriorMaxPositionError = 10.0;
    double imuPosePriorMaxRollPitchError = M_PI;
    double imuPosePriorMaxYawError = M_PI;
    bool imuPosePriorAvailable = false;
    double imuPosePriorTime = -1.0;
    double imuPosePriorAge = 0.0;
    float imuPosePriorTransform[6] = {0};

    bool aLoopIsClosed = false;
    int gpsFactorCount = 0;
    double gpsFactorVarianceScale = 1.0;
    double gpsFactorMaxTimeDiffRuntime = 0.2;
    double gpsFactorMinDistanceRuntime = 0.01;
    double gpsFactorVarianceOverrideRuntime = -1.0;
    int gnssHeadingFactorCount = 0;
    bool gnssInitializationDone = false;
    gtsam::Pose3 gnssInitializationPose;
    gtsam::noiseModel::Diagonal::shared_ptr gnssInitializationNoise;
    double lastGnssHeadingFactorTime = -1.0;
    bool gnssHeadingAnchorReady = false;
    int lastGnssHeadingKeyIndex = -1;
    double lastGnssHeadingYaw = 0.0;
    double lastGnssHeadingYawVar = 0.0;
    map<int, int> loopIndexContainer;
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;


    mapOptimization()
    {
        if (normalizedTerrainChangeMappingMode() != "downweight")
        {
            ROS_FATAL_STREAM("TCR-LM only supports terrainChangeMappingMode=downweight; got '"
                             << terrainChangeMappingMode << "'.");
            throw std::invalid_argument("unsupported terrainChangeMappingMode");
        }
        if (enableTerrainChangeMaintenance &&
            (!terrainChangeWriteBackLabels || !terrainChangeApplyToMapping))
        {
            ROS_FATAL("TCR-LM terrain maintenance requires both "
                      "terrainChangeWriteBackLabels=true and "
                      "terrainChangeApplyToMapping=true.");
            throw std::invalid_argument("incomplete terrain-change maintenance configuration");
        }
        if (gpsTopic.empty())
        {
            if (waitForGnssInitialization)
            {
                ROS_WARN("GPS topic is empty, disabling waitForGnssInitialization.");
                waitForGnssInitialization = false;
            }
            if (enableGnssHeadingFactor)
            {
                ROS_WARN("GPS topic is empty, disabling GNSS heading factor.");
                enableGnssHeadingFactor = false;
            }
            if (!gnssHeadingTopic.empty())
            {
                ROS_WARN_STREAM("GPS topic is empty, clearing GNSS heading topic: " << gnssHeadingTopic);
                gnssHeadingTopic.clear();
            }
        }
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        nh.param<bool>("tcrlm/useImuPosePrior", useImuPosePrior, false);
        nh.param<bool>("tcrlm/useImuPosePriorInLM", useImuPosePriorInLM, true);
        nh.param<bool>("tcrlm/useImuPosePriorInGraph", useImuPosePriorInGraph, false);
        nh.param<bool>("tcrlm/useImuPosePriorPosition", useImuPosePriorPosition, true);
        nh.param<bool>("tcrlm/useImuPosePriorRollPitch", useImuPosePriorRollPitch, false);
        nh.param<bool>("tcrlm/useImuPosePriorYaw", useImuPosePriorYaw, true);
        nh.param<bool>("tcrlm/debugImuPosePrior", debugImuPosePrior, false);
        nh.param<double>("tcrlm/imuPosePriorMaxAge", imuPosePriorMaxAge, 0.20);
        nh.param<double>("tcrlm/imuPosePriorPositionSigma", imuPosePriorPositionSigma, 0.30);
        nh.param<double>("tcrlm/imuPosePriorRollPitchSigma", imuPosePriorRollPitchSigma, 0.20);
        nh.param<double>("tcrlm/imuPosePriorYawSigma", imuPosePriorYawSigma, 0.05);
        nh.param<double>("tcrlm/imuPosePriorLmScale", imuPosePriorLmScale, 5.0);
        nh.param<double>("tcrlm/imuPosePriorGraphScale", imuPosePriorGraphScale, 1.0);
        nh.param<double>("tcrlm/imuPosePriorMaxPositionError", imuPosePriorMaxPositionError, 10.0);
        nh.param<double>("tcrlm/imuPosePriorMaxRollPitchError", imuPosePriorMaxRollPitchError, M_PI);
        nh.param<double>("tcrlm/imuPosePriorMaxYawError", imuPosePriorMaxYawError, M_PI);
        imuPosePriorMaxAge = std::max(0.0, imuPosePriorMaxAge);
        imuPosePriorPositionSigma = std::max(1e-4, imuPosePriorPositionSigma);
        imuPosePriorRollPitchSigma = std::max(1e-4, imuPosePriorRollPitchSigma);
        imuPosePriorYawSigma = std::max(1e-4, imuPosePriorYawSigma);
        imuPosePriorLmScale = std::max(0.0, imuPosePriorLmScale);
        imuPosePriorGraphScale = std::max(0.0, imuPosePriorGraphScale);
        nh.param<int>("tcrlm/mappingCloudInfoQueueSize", mappingCloudInfoQueueSize, 200);
        mappingCloudInfoQueueSize = std::max(1, mappingCloudInfoQueueSize);

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("tcrlm/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("tcrlm/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("tcrlm/mapping/path", 1);

        subCloud = nh.subscribe<tcrlm::cloud_info>("tcrlm/feature/cloud_info", mappingCloudInfoQueueSize, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        if (!gpsTopic.empty())
            subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        nh.param<double>("tcrlm/gpsFactorVarianceScale", gpsFactorVarianceScale, 1.0);
        gpsFactorVarianceScale = std::max(1.0, gpsFactorVarianceScale);
        gpsFactorMaxTimeDiffRuntime = std::max(0.0f, gpsFactorMaxTimeDiff);
        gpsFactorMinDistanceRuntime = std::max(0.0f, gpsFactorMinDistance);
        gpsFactorVarianceOverrideRuntime = gpsFactorVarianceOverride;
        if (enableGpsFactor && gpsFactorVarianceScale > 1.0)
            ROS_WARN_STREAM("[GNSS] GPSFactor variance scale=" << gpsFactorVarianceScale);
        if (!gnssHeadingTopic.empty())
            subGNSSHeading = nh.subscribe<sensor_msgs::Imu>(gnssHeadingTopic, 200, &mapOptimization::gnssHeadingHandler, this, ros::TransportHints().tcpNoDelay());
        if (useImu && useImuPosePrior && (useImuPosePriorInLM || useImuPosePriorInGraph))
        {
            subImuPosePrior = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &mapOptimization::imuPosePriorHandler, this, ros::TransportHints().tcpNoDelay());
            ROS_WARN_STREAM("[mapOptimization] IMU pose prior enabled: topic=" << odomTopic+"_incremental"
                            << ", in_lm=" << useImuPosePriorInLM
                            << ", in_graph=" << useImuPosePriorInGraph
                            << ", pos=" << useImuPosePriorPosition
                            << ", rpy=" << useImuPosePriorRollPitch
                            << ", yaw=" << useImuPosePriorYaw
                            << ", sigmas(pos,rp,yaw)=(" << imuPosePriorPositionSigma
                            << ", " << imuPosePriorRollPitchSigma
                            << ", " << imuPosePriorYawSigma << ")"
                            << ", lm_scale=" << imuPosePriorLmScale);
        }
        if (loopClosureEnableFlag)
            subLoop = nh.subscribe<std_msgs::Float64MultiArray>(
                "tcrlm/loop_closure/detection", 1,
                &mapOptimization::loopInfoHandler, this,
                ros::TransportHints().tcpNoDelay());

        srvSaveMap  = nh.advertiseService("tcrlm/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/tcrlm/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("tcrlm/mapping/cloud_registered_raw", 1);

        pubSLAMInfo           = nh.advertise<tcrlm::cloud_info>("tcrlm/mapping/slam_info", 1);
        pubTerrainChangeAllPoints = nh.advertise<sensor_msgs::PointCloud2>(
            "tcrlm/terrain_change/changed_points", 1, true);
        pubTerrainChangeSnapshot = nh.advertise<tcrlm::TerrainChangeSnapshot>(
            "tcrlm/mapping/terrain_change_snapshot", 1);
        subTerrainChangeUpdates = nh.subscribe<tcrlm::TerrainChangeUpdateArray>(
            "tcrlm/terrain_change/keyframe_updates", 1,
            &mapOptimization::terrainChangeUpdateHandler, this,
            ros::TransportHints().tcpNoDelay());

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterTerrainChangeInput.setLeafSize(terrainChangeInputLeafSize,
                                                      terrainChangeInputLeafSize,
                                                      terrainChangeInputLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity);
        nh.param<bool>("tcrlm/enableReverseMotionGuard", enableReverseMotionGuard, true);
        nh.param<float>("tcrlm/reverseMotionGuardCosThreshold", reverseMotionGuardCosThreshold, -0.2f);
        nh.param<float>("tcrlm/reverseMotionGuardMinDelta", reverseMotionGuardMinDelta, 0.05f);

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudFullRes.reset(new pcl::PointCloud<PointType>());
        laserCloudFullResDS.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        cornerMatchRawResidualVec.resize(N_SCAN * Horizon_SCAN, 0.0f);
        cornerMatchFeatureWeightVec.resize(N_SCAN * Horizon_SCAN, 1.0f);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);
        surfMatchRawResidualVec.resize(N_SCAN * Horizon_SCAN, 0.0f);
        surfMatchFeatureWeightVec.resize(N_SCAN * Horizon_SCAN, 1.0f);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    void laserCloudInfoHandler(const tcrlm::cloud_infoConstPtr& msgIn)
    {
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();


        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudFullRes);
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            if (waitForGnssInitialization && cloudKeyPoses3D->points.empty() && !gnssInitializationDone)
            {
                if (!initializeFirstPoseFromGnss())
                    return;
            }

            timeLastProcessing = timeLaserInfoCur;
            const auto slamCycleWallStart = std::chrono::steady_clock::now();
            auto slamElapsedMs = [&slamCycleWallStart]() -> double {
                return std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - slamCycleWallStart).count();
            };

            updateImuPosePrior();
            updateInitialGuess();
            std::copy(std::begin(transformTobeMapped), std::end(transformTobeMapped), std::begin(initialGuessTransform));
            currentReverseMotionSuspected = false;
            currentOptimizationRejected = false;
            currentMotionDirCos = 1.0f;

            applyPendingTerrainChangeUpdates();
            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            guardReverseMotion();

            saveKeyFramesAndFactor();

            correctPoses();

            currentSlamComputeMs = slamElapsedMs();
            publishOdometry();

            publishFrames();
            currentSlamCycleMs = slamElapsedMs();
            if (logLioStatePerFrame)
            {
                ROS_WARN_STREAM(std::fixed << std::setprecision(3)
                    << "[SLAM timing] mode=tcrlm"
                    << " t=" << timeLaserInfoCur
                    << " keyframes=" << cloudKeyPoses6D->size()
                    << " compute_ms=" << currentSlamComputeMs
                    << " cycle_ms=" << currentSlamCycleMs);
            }
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        gpsQueue.push_back(*gpsMsg);
    }

    void imuPosePriorHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock(imuPosePriorLock);
        imuPosePriorQueue.push_back(*odometryMsg);
        while (imuPosePriorQueue.size() > 2000)
            imuPosePriorQueue.pop_front();
    }

    void updateImuPosePrior()
    {
        imuPosePriorAvailable = false;
        imuPosePriorTime = -1.0;
        imuPosePriorAge = 0.0;

        if (!useImu || !useImuPosePrior || (!useImuPosePriorInLM && !useImuPosePriorInGraph))
            return;

        nav_msgs::Odometry priorOdom;
        int latestIdx = -1;
        {
            std::lock_guard<std::mutex> lock(imuPosePriorLock);
            for (int i = 0; i < (int)imuPosePriorQueue.size(); ++i)
            {
                if (imuPosePriorQueue[i].header.stamp.toSec() <= timeLaserInfoCur)
                    latestIdx = i;
                else
                    break;
            }

            if (latestIdx < 0)
            {
                double bestAge = imuPosePriorMaxAge;
                for (int i = 0; i < (int)imuPosePriorQueue.size(); ++i)
                {
                    const double age = std::fabs(imuPosePriorQueue[i].header.stamp.toSec() - timeLaserInfoCur);
                    if (age <= bestAge)
                    {
                        bestAge = age;
                        latestIdx = i;
                    }
                }
            }

            if (latestIdx < 0)
            {
                if (debugImuPosePrior)
                {
                    ROS_WARN_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                        << "[IMU pose prior] no aligned " << odomTopic
                        << "_incremental within " << imuPosePriorMaxAge
                        << " s of lidar_t=" << timeLaserInfoCur
                        << ", queue=" << imuPosePriorQueue.size());
                }
                return;
            }

            priorOdom = imuPosePriorQueue[latestIdx];
            while (imuPosePriorQueue.size() > 1 && latestIdx > 0)
            {
                imuPosePriorQueue.pop_front();
                --latestIdx;
            }
        }

        imuPosePriorTime = priorOdom.header.stamp.toSec();
        imuPosePriorAge = std::fabs(imuPosePriorTime - timeLaserInfoCur);
        if (imuPosePriorAge > imuPosePriorMaxAge)
        {
            if (debugImuPosePrior)
            {
                ROS_WARN_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                    << "[IMU pose prior] stale IMU odom. imu_t=" << imuPosePriorTime
                    << ", lidar_t=" << timeLaserInfoCur
                    << ", age=" << imuPosePriorAge
                    << ", max_age=" << imuPosePriorMaxAge);
            }
            return;
        }

        tf::Quaternion q;
        tf::quaternionMsgToTF(priorOdom.pose.pose.orientation, q);
        if (q.length2() < 1e-6 || !std::isfinite(q.x()) || !std::isfinite(q.y()) ||
            !std::isfinite(q.z()) || !std::isfinite(q.w()))
        {
            if (debugImuPosePrior)
                ROS_WARN_STREAM_THROTTLE(1.0, "[IMU pose prior] invalid IMU odom quaternion.");
            return;
        }
        q.normalize();

        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        const double x = priorOdom.pose.pose.position.x;
        const double y = priorOdom.pose.pose.position.y;
        const double z = priorOdom.pose.pose.position.z;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
            !std::isfinite(roll) || !std::isfinite(pitch) || !std::isfinite(yaw))
        {
            if (debugImuPosePrior)
                ROS_WARN_STREAM_THROTTLE(1.0, "[IMU pose prior] invalid IMU odom pose.");
            return;
        }

        imuPosePriorTransform[0] = static_cast<float>(roll);
        imuPosePriorTransform[1] = static_cast<float>(pitch);
        imuPosePriorTransform[2] = static_cast<float>(yaw);
        imuPosePriorTransform[3] = static_cast<float>(x);
        imuPosePriorTransform[4] = static_cast<float>(y);
        imuPosePriorTransform[5] = static_cast<float>(z);
        imuPosePriorAvailable = true;
    }

    void gnssHeadingHandler(const sensor_msgs::Imu::ConstPtr& headingMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        gnssHeadingQueue.push_back(*headingMsg);
    }

    void terrainChangeUpdateHandler(const tcrlm::TerrainChangeUpdateArrayConstPtr& msg)
    {
        if (!terrainChangeWriteBackLabels)
            return;
        if (msg->updates.empty())
            return;

        std::lock_guard<std::mutex> lock(mtxTerrainChangeUpdates);
        pendingTerrainChangeUpdates.insert(pendingTerrainChangeUpdates.end(),
                                           msg->updates.begin(), msg->updates.end());
    }

    void applyPendingTerrainChangeUpdates()
    {
        if (!terrainChangeWriteBackLabels)
            return;

        std::vector<tcrlm::TerrainChangeUpdate> updates;
        {
            std::lock_guard<std::mutex> lock(mtxTerrainChangeUpdates);
            if (pendingTerrainChangeUpdates.empty())
                return;
            updates.swap(pendingTerrainChangeUpdates);
        }

        ++terrainChangeEpoch;
        std::unordered_map<long long, int> writeCounts;
        std::unordered_set<int> dirtyKeyframes;
        std::unordered_set<std::string> appliedKeys;
        std::unordered_map<int, TerrainStaleFullSet> staleFullPointsByKeyframe;
        int applied = 0;

        for (const auto& update : updates)
        {
            if (update.change_type != tcrlm::TerrainChangeUpdate::CHANGE_EXCAVATION &&
                update.change_type != tcrlm::TerrainChangeUpdate::CHANGE_ACCUMULATION)
                continue;

            const std::string key = std::to_string(update.keyframe_id) + ":" +
                std::to_string(update.cloud_type) + ":" + std::to_string(update.point_index);
            if (appliedKeys.find(key) != appliedKeys.end())
                continue;
            appliedKeys.insert(key);

            if (update.cloud_type != tcrlm::TerrainChangeUpdate::CLOUD_FULL)
                continue;

            const int keyframeId = static_cast<int>(update.keyframe_id);
            const int pointId = static_cast<int>(update.point_index);
            if (markTerrainPoint(keyframeId,
                                 pointId,
                                 TERRAIN_POINT_FULL,
                                 STALE_OLD,
                                 update.change_type,
                                 ROLE_OLD_SURFACE,
                                 update.region_id,
                                 writeCounts,
                                 dirtyKeyframes))
            {
                applied++;
                if (keyframeId >= 0 &&
                    keyframeId < (int)fullCloudKeyFrames.size() &&
                    pointId >= 0 &&
                    pointId < (int)fullCloudKeyFrames[keyframeId]->size())
                {
                    TerrainStaleFullSet& staleSet = staleFullPointsByKeyframe[keyframeId];
                    staleSet.local_points->push_back(fullCloudKeyFrames[keyframeId]->points[pointId]);
                    staleSet.region_ids.push_back(update.region_id);
                    staleSet.change_types.push_back(update.change_type);
                }
            }
        }

        applied += markFeaturePointsNearStaleFullPoints(staleFullPointsByKeyframe,
                                                          TERRAIN_POINT_CORNER,
                                                          writeCounts, dirtyKeyframes);
        applied += markFeaturePointsNearStaleFullPoints(staleFullPointsByKeyframe,
                                                          TERRAIN_POINT_SURF,
                                                          writeCounts, dirtyKeyframes);

        invalidateTerrainChangeDirtyKeyframes(dirtyKeyframes);
        if (applied > 0)
            publishAsyncTerrainChangedPointsFromState();

        if (terrainChangeLogSummary && !updates.empty())
        {
            ROS_WARN_STREAM("[terrain_change_async] received_updates=" << updates.size()
                            << ", applied=" << applied
                            << ", dirty_keyframes=" << dirtyKeyframes.size()
                            << ", epoch=" << terrainChangeEpoch);
        }
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    bool terrainPointStateAllowsSoftMap(uint8_t state) const
    {
        return state != REJECTED_NON_TERRAIN;
    }

    float terrainConfidenceForState(uint8_t state, uint16_t confirmations) const
    {
        if (state != STALE_OLD)
            return 1.0f;

        const float effectiveConfirmations = static_cast<float>(std::max<uint16_t>(confirmations, 1));
        const float weight = 1.0f - terrainChangeWeightDropPerConfirmation * effectiveConfirmations;
        return std::min(1.0f, std::max(terrainChangeMinFeatureWeight, weight));
    }

    float terrainFeatureWeightForState(uint8_t state, uint16_t confirmations) const
    {
        if (!useTerrainChangeMapDownweighting())
            return 1.0f;
        return terrainConfidenceForState(state, confirmations);
    }

    bool terrainPointStateAllowsDetection(uint8_t state) const
    {
        return state == VALID_STABLE || state == CHANGED_NEW;
    }

    uint8_t terrainPointStateAt(const std::vector<std::vector<uint8_t>>& states, int keyframeId, int pointId) const
    {
        if (keyframeId < 0 || keyframeId >= (int)states.size())
            return VALID_STABLE;
        if (pointId < 0 || pointId >= (int)states[keyframeId].size())
            return VALID_STABLE;
        return states[keyframeId][pointId];
    }

    uint16_t terrainPointConfirmationsAt(const std::vector<std::vector<uint16_t>>& confirmations,
                                         int keyframeId,
                                         int pointId) const
    {
        if (keyframeId < 0 || keyframeId >= (int)confirmations.size())
            return 0;
        if (pointId < 0 || pointId >= (int)confirmations[keyframeId].size())
            return 0;
        return confirmations[keyframeId][pointId];
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloudWithState(
        pcl::PointCloud<PointType>::Ptr cloudIn,
        const std::vector<uint8_t>& pointState,
        const std::vector<uint16_t>& pointConfirmations,
        PointTypePose* transformIn,
        bool forLocalMap)
    {
        if (!useTerrainChangeMapMaintenance() || pointState.size() != cloudIn->size())
            return transformPointCloud(cloudIn, transformIn);

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        cloudOut->reserve(cloudIn->size());

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z,
                                                           transformIn->roll, transformIn->pitch, transformIn->yaw);

        for (int i = 0; i < (int)cloudIn->size(); ++i)
        {
            float featureWeight = 1.0f;
            if (forLocalMap)
            {
                if (useTerrainChangeMapDownweighting())
                {
                    if (!terrainPointStateAllowsSoftMap(pointState[i]))
                        continue;
                    const uint16_t confirmations =
                        i < (int)pointConfirmations.size() ? pointConfirmations[i] : 0;
                    featureWeight = terrainFeatureWeightForState(pointState[i], confirmations);
                }
            }

            const auto &pointFrom = cloudIn->points[i];
            PointType pointOut;
            pointOut.x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            pointOut.y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            pointOut.z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            pointOut.intensity = forLocalMap && useTerrainChangeMapDownweighting() ? featureWeight : pointFrom.intensity;
            cloudOut->push_back(pointOut);
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformCornerKeyframeForMaintainedMap(int keyframeId, PointTypePose* transformIn)
    {
        if (keyframeId < 0 || keyframeId >= (int)cornerCloudKeyFrames.size())
            return pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
        if (keyframeId < (int)cornerPointState.size())
            return transformPointCloudWithState(cornerCloudKeyFrames[keyframeId], cornerPointState[keyframeId],
                                                keyframeId < (int)cornerChangeConfirmations.size() ? cornerChangeConfirmations[keyframeId] : std::vector<uint16_t>(),
                                                transformIn, true);
        return transformPointCloud(cornerCloudKeyFrames[keyframeId], transformIn);
    }

    pcl::PointCloud<PointType>::Ptr transformSurfKeyframeForMaintainedMap(int keyframeId, PointTypePose* transformIn)
    {
        if (keyframeId < 0 || keyframeId >= (int)surfCloudKeyFrames.size())
            return pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
        if (keyframeId < (int)surfPointState.size())
            return transformPointCloudWithState(surfCloudKeyFrames[keyframeId], surfPointState[keyframeId],
                                                keyframeId < (int)surfChangeConfirmations.size() ? surfChangeConfirmations[keyframeId] : std::vector<uint16_t>(),
                                                transformIn, true);
        return transformPointCloud(surfCloudKeyFrames[keyframeId], transformIn);
    }

    PointType transformTerrainPointToMap(const PointType& pointIn, const Eigen::Affine3f& transform) const
    {
        PointType pointOut;
        pointOut.x = transform(0,0) * pointIn.x + transform(0,1) * pointIn.y + transform(0,2) * pointIn.z + transform(0,3);
        pointOut.y = transform(1,0) * pointIn.x + transform(1,1) * pointIn.y + transform(1,2) * pointIn.z + transform(1,3);
        pointOut.z = transform(2,0) * pointIn.x + transform(2,1) * pointIn.y + transform(2,2) * pointIn.z + transform(2,3);
        pointOut.intensity = pointIn.intensity;
        return pointOut;
    }

    void initializeTerrainChangeStateForNewKeyframe(
        int keyframeId,
        const pcl::PointCloud<PointType>::Ptr& cornerCloud,
        const pcl::PointCloud<PointType>::Ptr& surfCloud,
        const pcl::PointCloud<PointType>::Ptr& fullCloud)
    {
        if (keyframeId < 0)
            return;

        const std::size_t requiredSize = static_cast<std::size_t>(keyframeId + 1);
        cornerPointState.resize(requiredSize);
        cornerChangeMask.resize(requiredSize);
        cornerChangeType.resize(requiredSize);
        cornerChangeRole.resize(requiredSize);
        cornerChangeRegionId.resize(requiredSize);
        cornerChangeEpoch.resize(requiredSize);
        cornerChangeConfirmations.resize(requiredSize);

        surfPointState.resize(requiredSize);
        surfChangeMask.resize(requiredSize);
        surfChangeType.resize(requiredSize);
        surfChangeRole.resize(requiredSize);
        surfChangeRegionId.resize(requiredSize);
        surfChangeEpoch.resize(requiredSize);
        surfChangeConfirmations.resize(requiredSize);

        fullPointState.resize(requiredSize);
        fullChangeMask.resize(requiredSize);
        fullChangeType.resize(requiredSize);
        fullChangeRole.resize(requiredSize);
        fullChangeRegionId.resize(requiredSize);
        fullChangeEpoch.resize(requiredSize);
        fullChangeConfirmations.resize(requiredSize);

        const std::size_t cornerSize = cornerCloud ? cornerCloud->size() : 0;
        cornerPointState[keyframeId].assign(cornerSize, VALID_STABLE);
        cornerChangeMask[keyframeId].assign(cornerSize, 0);
        cornerChangeType[keyframeId].assign(cornerSize, CHANGE_NONE);
        cornerChangeRole[keyframeId].assign(cornerSize, ROLE_NONE);
        cornerChangeRegionId[keyframeId].assign(cornerSize, 0);
        cornerChangeEpoch[keyframeId].assign(cornerSize, 0);
        cornerChangeConfirmations[keyframeId].assign(cornerSize, 0);

        const std::size_t surfSize = surfCloud ? surfCloud->size() : 0;
        surfPointState[keyframeId].assign(surfSize, VALID_STABLE);
        surfChangeMask[keyframeId].assign(surfSize, 0);
        surfChangeType[keyframeId].assign(surfSize, CHANGE_NONE);
        surfChangeRole[keyframeId].assign(surfSize, ROLE_NONE);
        surfChangeRegionId[keyframeId].assign(surfSize, 0);
        surfChangeEpoch[keyframeId].assign(surfSize, 0);
        surfChangeConfirmations[keyframeId].assign(surfSize, 0);

        const std::size_t fullSize = fullCloud ? fullCloud->size() : 0;
        fullPointState[keyframeId].assign(fullSize, VALID_STABLE);
        fullChangeMask[keyframeId].assign(fullSize, 0);
        fullChangeType[keyframeId].assign(fullSize, CHANGE_NONE);
        fullChangeRole[keyframeId].assign(fullSize, ROLE_NONE);
        fullChangeRegionId[keyframeId].assign(fullSize, 0);
        fullChangeEpoch[keyframeId].assign(fullSize, 0);
        fullChangeConfirmations[keyframeId].assign(fullSize, 0);
    }

    void appendTerrainSnapshotCloud(
        tcrlm::TerrainChangeSnapshot& snapshot,
        int keyframeId,
        uint8_t cloudType,
        const pcl::PointCloud<PointType>::Ptr& cloud,
        const std::vector<std::vector<uint8_t>>& states,
        const std::vector<std::vector<uint16_t>>& confirmations,
        const Eigen::Affine3f& transform)
    {
        if (!cloud)
            return;

        for (int pointId = 0; pointId < (int)cloud->size(); ++pointId)
        {
            const uint8_t state = terrainPointStateAt(states, keyframeId, pointId);
            if (cloudType == TERRAIN_POINT_FULL &&
                !terrainPointStateAllowsDetection(state) && state != STALE_OLD)
                continue;
            if ((cloudType == TERRAIN_POINT_CORNER || cloudType == TERRAIN_POINT_SURF) &&
                state == REJECTED_NON_TERRAIN)
                continue;

            PointType pointMap = transformTerrainPointToMap(cloud->points[pointId], transform);
            if (!std::isfinite(pointMap.x) || !std::isfinite(pointMap.y) || !std::isfinite(pointMap.z))
                continue;

            tcrlm::TerrainChangePoint point;
            point.keyframe_id = static_cast<uint32_t>(keyframeId);
            point.point_index = static_cast<uint32_t>(pointId);
            point.cloud_type = cloudType;
            point.x = pointMap.x;
            point.y = pointMap.y;
            point.z = pointMap.z;
            point.time = keyframeId < (int)cloudKeyPoses6D->size() ? cloudKeyPoses6D->points[keyframeId].time : 0.0;
            point.state = state;
            point.confirmations = terrainPointConfirmationsAt(confirmations, keyframeId, pointId);
            point.weight = terrainConfidenceForState(state, point.confirmations);
            snapshot.points.push_back(point);
        }
    }

    void publishTerrainChangeSnapshotIfNeeded(int currentKeyframeId)
    {
        if (!useTerrainChangeMaintenance())
            return;

        if (currentKeyframeId < 0 || currentKeyframeId >= (int)cloudKeyPoses3D->size())
            return;
        if (currentKeyframeId >= (int)cloudKeyPoses6D->size())
            return;

        const double currentTime = cloudKeyPoses6D->points[currentKeyframeId].time;
        if (terrainChangeStartTime < 0.0)
            terrainChangeStartTime = currentTime;

        if (currentTime - terrainChangeStartTime < terrainChangeDetectionInterval)
            return;

        if (lastTerrainChangeUpdateTime >= 0.0 &&
            currentTime - lastTerrainChangeUpdateTime < terrainChangeDetectionInterval)
            return;

        if (pubTerrainChangeSnapshot.getNumSubscribers() == 0)
        {
            ROS_WARN_STREAM_THROTTLE(5.0, "[terrain_change_async] waiting for terrainChangeNode subscriber.");
            return;
        }

        std::vector<int> activeKeyframes;
        if (!selectTerrainChangeActiveKeyframes(currentKeyframeId, activeKeyframes))
        {
            if (terrainChangeLogSummary)
            {
                ROS_WARN_STREAM_THROTTLE(2.0, "[terrain_change_async] waiting for active keyframes: active="
                    << activeKeyframes.size() << "/" << terrainChangeMinActiveKeyframes
                    << ", keyframe=" << currentKeyframeId);
            }
            return;
        }

        tcrlm::TerrainChangeSnapshot snapshot;
        snapshot.header.stamp = timeLaserInfoStamp;
        snapshot.header.frame_id = odometryFrame;
        snapshot.snapshot_id = ++terrainChangeSnapshotId;
        snapshot.current_keyframe_id = static_cast<uint32_t>(currentKeyframeId);

        for (int keyframeId : activeKeyframes)
        {
            if (keyframeId < 0 || keyframeId >= (int)cloudKeyPoses6D->size())
                continue;
            const Eigen::Affine3f transform = pclPointToAffine3f(cloudKeyPoses6D->points[keyframeId]);
            if (keyframeId < (int)fullCloudKeyFrames.size())
                appendTerrainSnapshotCloud(snapshot, keyframeId, TERRAIN_POINT_FULL,
                                           fullCloudKeyFrames[keyframeId],
                                           fullPointState, fullChangeConfirmations, transform);
            if (keyframeId < (int)cornerCloudKeyFrames.size())
                appendTerrainSnapshotCloud(snapshot, keyframeId, TERRAIN_POINT_CORNER,
                                           cornerCloudKeyFrames[keyframeId],
                                           cornerPointState, cornerChangeConfirmations, transform);
            if (keyframeId < (int)surfCloudKeyFrames.size())
                appendTerrainSnapshotCloud(snapshot, keyframeId, TERRAIN_POINT_SURF,
                                           surfCloudKeyFrames[keyframeId],
                                           surfPointState, surfChangeConfirmations, transform);
        }

        pubTerrainChangeSnapshot.publish(snapshot);
        lastTerrainChangeUpdateTime = currentTime;

        if (terrainChangeLogSummary)
        {
            ROS_WARN_STREAM("[terrain_change_async] publish snapshot=" << snapshot.snapshot_id
                            << ", keyframe=" << currentKeyframeId
                            << ", active=" << activeKeyframes.size()
                            << ", points=" << snapshot.points.size());
        }
    }

    long long terrainWriteCountKey(int keyframeId, uint8_t pointType) const
    {
        return (static_cast<long long>(keyframeId) << 8) | static_cast<long long>(pointType);
    }

    bool markTerrainPoint(
        int keyframeId,
        int pointId,
        uint8_t pointType,
        uint8_t pointState,
        uint8_t changeType,
        uint8_t changeRole,
        uint32_t regionId,
        std::unordered_map<long long, int>& writeCounts,
        std::unordered_set<int>& dirtyKeyframes)
    {
        if (keyframeId < 0 || pointId < 0)
            return false;

        std::vector<std::vector<uint8_t>>* states = nullptr;
        std::vector<std::vector<uint8_t>>* masks = nullptr;
        std::vector<std::vector<uint8_t>>* types = nullptr;
        std::vector<std::vector<uint8_t>>* roles = nullptr;
        std::vector<std::vector<uint32_t>>* regions = nullptr;
        std::vector<std::vector<uint32_t>>* epochs = nullptr;
        std::vector<std::vector<uint16_t>>* confirmations = nullptr;
        int cloudSize = 0;

        if (pointType == TERRAIN_POINT_CORNER)
        {
            if (keyframeId >= (int)cornerCloudKeyFrames.size())
                return false;
            cloudSize = cornerCloudKeyFrames[keyframeId]->size();
            states = &cornerPointState;
            masks = &cornerChangeMask;
            types = &cornerChangeType;
            roles = &cornerChangeRole;
            regions = &cornerChangeRegionId;
            epochs = &cornerChangeEpoch;
            confirmations = &cornerChangeConfirmations;
        }
        else if (pointType == TERRAIN_POINT_SURF)
        {
            if (keyframeId >= (int)surfCloudKeyFrames.size())
                return false;
            cloudSize = surfCloudKeyFrames[keyframeId]->size();
            states = &surfPointState;
            masks = &surfChangeMask;
            types = &surfChangeType;
            roles = &surfChangeRole;
            regions = &surfChangeRegionId;
            epochs = &surfChangeEpoch;
            confirmations = &surfChangeConfirmations;
        }
        else if (pointType == TERRAIN_POINT_FULL)
        {
            if (keyframeId >= (int)fullCloudKeyFrames.size())
                return false;
            cloudSize = fullCloudKeyFrames[keyframeId]->size();
            states = &fullPointState;
            masks = &fullChangeMask;
            types = &fullChangeType;
            roles = &fullChangeRole;
            regions = &fullChangeRegionId;
            epochs = &fullChangeEpoch;
            confirmations = &fullChangeConfirmations;
        }
        else
        {
            return false;
        }

        if (keyframeId >= (int)states->size() || pointId >= (int)(*states)[keyframeId].size())
            return false;
        if (keyframeId >= (int)confirmations->size() || pointId >= (int)(*confirmations)[keyframeId].size())
            return false;

        const bool labelChanged =
            (*states)[keyframeId][pointId] != pointState ||
            (*masks)[keyframeId][pointId] != 1 ||
            (*types)[keyframeId][pointId] != changeType ||
            (*roles)[keyframeId][pointId] != changeRole ||
            (*regions)[keyframeId][pointId] != regionId;

        const bool newConfirmationThisEpoch =
            keyframeId >= (int)epochs->size() ||
            pointId >= (int)(*epochs)[keyframeId].size() ||
            (*epochs)[keyframeId][pointId] != static_cast<uint32_t>(terrainChangeEpoch);

        if (!labelChanged && !newConfirmationThisEpoch)
            return false;

        float maxUpdateRatio = terrainChangeMaxUpdateRatioPerKeyframe;
        if (pointType == TERRAIN_POINT_CORNER || pointType == TERRAIN_POINT_SURF)
            maxUpdateRatio = terrainChangeMaxFeatureUpdateRatioPerKeyframe;

        if (maxUpdateRatio > 0.0f && maxUpdateRatio < 1.0f)
        {
            const int maxWrites = std::max(1, static_cast<int>(std::floor(cloudSize * maxUpdateRatio)));
            const long long countKey = terrainWriteCountKey(keyframeId, pointType);
            if (writeCounts[countKey] >= maxWrites)
                return false;
            writeCounts[countKey]++;
        }

        if (newConfirmationThisEpoch && (*confirmations)[keyframeId][pointId] < std::numeric_limits<uint16_t>::max())
            (*confirmations)[keyframeId][pointId]++;

        (*states)[keyframeId][pointId] = pointState;
        (*masks)[keyframeId][pointId] = 1;
        (*types)[keyframeId][pointId] = changeType;
        (*roles)[keyframeId][pointId] = changeRole;
        (*regions)[keyframeId][pointId] = regionId;
        (*epochs)[keyframeId][pointId] = static_cast<uint32_t>(terrainChangeEpoch);
        dirtyKeyframes.insert(keyframeId);
        return true;
    }


    int markFeaturePointsNearStaleFullPoints(
        const std::unordered_map<int, TerrainStaleFullSet>& staleFullPointsByKeyframe,
        uint8_t pointType,
        std::unordered_map<long long, int>& writeCounts,
        std::unordered_set<int>& dirtyKeyframes)
    {
        int updatedPoints = 0;
        const float syncRadius = std::max(0.05f,
            std::min(terrainChangeGridResolution * 0.5f, terrainChangeWriteBand * 3.0f));

        for (const auto& item : staleFullPointsByKeyframe)
        {
            const int keyframeId = item.first;
            const TerrainStaleFullSet& staleSet = item.second;
            if (!staleSet.local_points || staleSet.local_points->empty())
                continue;

            pcl::PointCloud<PointType>::Ptr featureCloud;
            if (pointType == TERRAIN_POINT_CORNER)
            {
                if (keyframeId < 0 || keyframeId >= (int)cornerCloudKeyFrames.size())
                    continue;
                featureCloud = cornerCloudKeyFrames[keyframeId];
            }
            else if (pointType == TERRAIN_POINT_SURF)
            {
                if (keyframeId < 0 || keyframeId >= (int)surfCloudKeyFrames.size())
                    continue;
                featureCloud = surfCloudKeyFrames[keyframeId];
            }
            else
            {
                continue;
            }

            pcl::KdTreeFLANN<PointType> kdtreeFeature;
            kdtreeFeature.setInputCloud(featureCloud);

            std::vector<int> featureIndices;
            std::vector<float> featureSqDists;
            for (int staleIndex = 0; staleIndex < (int)staleSet.local_points->size(); ++staleIndex)
            {
                if (staleIndex >= (int)staleSet.region_ids.size() ||
                    staleIndex >= (int)staleSet.change_types.size())
                    break;

                featureIndices.clear();
                featureSqDists.clear();
                if (kdtreeFeature.radiusSearch(staleSet.local_points->points[staleIndex],
                                               syncRadius,
                                               featureIndices,
                                               featureSqDists) <= 0)
                {
                    continue;
                }

                for (int pointId : featureIndices)
                {
                    if (pointId < 0 || pointId >= (int)featureCloud->size())
                        continue;

                    if (markTerrainPoint(keyframeId, pointId, pointType,
                                         STALE_OLD, staleSet.change_types[staleIndex], ROLE_OLD_SURFACE,
                                         staleSet.region_ids[staleIndex],
                                         writeCounts, dirtyKeyframes))
                    {
                        updatedPoints++;
                    }
                }
            }
        }

        return updatedPoints;
    }

    void invalidateTerrainChangeDirtyKeyframes(const std::unordered_set<int>& dirtyKeyframes)
    {
        if (!useTerrainChangeMapMaintenance())
            return;

        for (int keyframeId : dirtyKeyframes)
            laserCloudMapContainer.erase(keyframeId);
    }

    bool selectTerrainChangeActiveKeyframes(int currentKeyframeId, std::vector<int>& activeKeyframes) const
    {
        activeKeyframes.clear();

        if (currentKeyframeId < 0 || currentKeyframeId >= (int)cloudKeyPoses3D->size())
            return false;

        const PointType& center = cloudKeyPoses3D->points[currentKeyframeId];
        const double currentTime = currentKeyframeId < (int)cloudKeyPoses6D->size() ?
            cloudKeyPoses6D->points[currentKeyframeId].time : 0.0;
        std::unordered_set<int> activeSet;

        for (int key = 0; key <= currentKeyframeId && key < (int)cloudKeyPoses3D->size(); ++key)
        {
            if (pointDistance(cloudKeyPoses3D->points[key], center) <= surroundingKeyframeSearchRadius)
                activeSet.insert(key);
        }

        for (int key = currentKeyframeId; key >= 0 && key < (int)cloudKeyPoses6D->size(); --key)
        {
            if (currentTime - cloudKeyPoses6D->points[key].time < 10.0)
                activeSet.insert(key);
            else
                break;
        }

        activeKeyframes.assign(activeSet.begin(), activeSet.end());
        std::sort(activeKeyframes.begin(), activeKeyframes.end());
        return (int)activeKeyframes.size() >= terrainChangeMinActiveKeyframes;
    }


    void publishTerrainCloudAlways(const ros::Publisher& publisher,
                                   const pcl::PointCloud<PointType>::Ptr& cloud)
    {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.stamp = timeLaserInfoStamp;
        msg.header.frame_id = odometryFrame;
        publisher.publish(msg);
    }

    void publishAsyncTerrainChangedPointsFromState()
    {
        if (!terrainChangePublishCandidateClouds)
            return;

        pcl::PointCloud<PointType>::Ptr changedPoints(new pcl::PointCloud<PointType>());
        const int totalKeyframes = std::min((int)fullCloudKeyFrames.size(), (int)cloudKeyPoses6D->size());
        int beginKeyframe = 0;
        if (terrainChangeMaxVisualizationKeyframes > 0)
            beginKeyframe = std::max(0, totalKeyframes - terrainChangeMaxVisualizationKeyframes);

        for (int keyframeId = beginKeyframe; keyframeId < totalKeyframes; ++keyframeId)
        {
            if (keyframeId >= (int)fullPointState.size() ||
                keyframeId >= (int)fullChangeConfirmations.size() ||
                keyframeId >= (int)fullCloudKeyFrames.size())
                continue;

            const Eigen::Affine3f transform = pclPointToAffine3f(cloudKeyPoses6D->points[keyframeId]);
            const auto& fullCloud = fullCloudKeyFrames[keyframeId];
            const int pointCount = std::min((int)fullCloud->size(), (int)fullPointState[keyframeId].size());
            for (int pointId = 0; pointId < pointCount; ++pointId)
            {
                if (fullPointState[keyframeId][pointId] != STALE_OLD)
                    continue;

                PointType pointMap = transformTerrainPointToMap(fullCloud->points[pointId], transform);
                if (!std::isfinite(pointMap.x) || !std::isfinite(pointMap.y) || !std::isfinite(pointMap.z))
                    continue;

                const uint16_t confirmations =
                    pointId < (int)fullChangeConfirmations[keyframeId].size() ?
                    fullChangeConfirmations[keyframeId][pointId] : 1;
                pointMap.intensity = terrainConfidenceForState(STALE_OLD, confirmations);
                changedPoints->push_back(pointMap);
            }
        }

        publishTerrainCloudAlways(pubTerrainChangeAllPoints, changedPoints);

        if (terrainChangeLogSummary)
        {
            ROS_WARN_STREAM("[terrain_change_async] publish state stale cloud: stale_points="
                            << changedPoints->size()
                            << ", keyframes=" << (totalKeyframes - beginKeyframe));
        }
    }

    void collectChangedFeaturePoints(
        uint32_t regionId,
        uint8_t changeTypeFilter,
        uint8_t changeRoleFilter,
        uint8_t pointType,
        pcl::PointCloud<PointType>::Ptr& cloudOut,
        bool useVisualizationWindow = true)
    {
        const bool collectCorner = pointType == TERRAIN_POINT_CORNER || pointType == TERRAIN_POINT_BOTH;
        const bool collectSurf = pointType == TERRAIN_POINT_SURF || pointType == TERRAIN_POINT_BOTH;
        const bool collectFull = pointType == TERRAIN_POINT_FULL || pointType == TERRAIN_POINT_BOTH;
        const int totalKeyframes = cloudKeyPoses6D->size();
        int beginKeyframe = 0;
        if (useVisualizationWindow && terrainChangeMaxVisualizationKeyframes > 0)
            beginKeyframe = std::max(0, totalKeyframes - terrainChangeMaxVisualizationKeyframes);

        auto collectOneCloud = [&](int keyframeId,
                                   const pcl::PointCloud<PointType>::Ptr& cloud,
                                   const std::vector<std::vector<uint8_t>>& states,
                                   const std::vector<std::vector<uint16_t>>& confirmations,
                                   const std::vector<std::vector<uint8_t>>& masks,
                                   const std::vector<std::vector<uint8_t>>& types,
                                   const std::vector<std::vector<uint8_t>>& roles,
                                   const std::vector<std::vector<uint32_t>>& regions)
        {
            if (keyframeId < 0 || keyframeId >= totalKeyframes)
                return;
            if (keyframeId >= (int)states.size() || keyframeId >= (int)confirmations.size() ||
                keyframeId >= (int)masks.size() || keyframeId >= (int)types.size() ||
                keyframeId >= (int)roles.size() || keyframeId >= (int)regions.size())
                return;
            if ((int)states[keyframeId].size() != (int)cloud->size() ||
                (int)masks[keyframeId].size() != (int)cloud->size())
                return;

            const Eigen::Affine3f transform = pclPointToAffine3f(cloudKeyPoses6D->points[keyframeId]);
            for (int pointId = 0; pointId < (int)cloud->size(); ++pointId)
            {
                if (masks[keyframeId][pointId] == 0)
                    continue;
                if (regionId != 0 && regions[keyframeId][pointId] != regionId)
                    continue;
                if (changeTypeFilter != CHANGE_NONE && types[keyframeId][pointId] != changeTypeFilter)
                    continue;
                if (changeRoleFilter != ROLE_NONE && roles[keyframeId][pointId] != changeRoleFilter)
                    continue;

                PointType pointMap = transformTerrainPointToMap(cloud->points[pointId], transform);
                const uint8_t state = states[keyframeId][pointId];
                const uint16_t confirmationCount = pointId < (int)confirmations[keyframeId].size() ?
                    confirmations[keyframeId][pointId] : 0;
                pointMap.intensity = terrainConfidenceForState(state, confirmationCount);
                cloudOut->push_back(pointMap);
            }
        };

        for (int keyframeId = beginKeyframe; keyframeId < totalKeyframes; ++keyframeId)
        {
            if (collectCorner && keyframeId < (int)cornerCloudKeyFrames.size())
            {
                collectOneCloud(keyframeId, cornerCloudKeyFrames[keyframeId],
                                cornerPointState, cornerChangeConfirmations,
                                cornerChangeMask, cornerChangeType, cornerChangeRole, cornerChangeRegionId);
            }
            if (collectSurf && keyframeId < (int)surfCloudKeyFrames.size())
            {
                collectOneCloud(keyframeId, surfCloudKeyFrames[keyframeId],
                                surfPointState, surfChangeConfirmations,
                                surfChangeMask, surfChangeType, surfChangeRole, surfChangeRegionId);
            }
            if (collectFull && keyframeId < (int)fullCloudKeyFrames.size())
            {
                collectOneCloud(keyframeId, fullCloudKeyFrames[keyframeId],
                                fullPointState, fullChangeConfirmations,
                                fullChangeMask, fullChangeType, fullChangeRole, fullChangeRegionId);
            }
        }
    }


    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    template <typename MsgType>
    bool findNearestMsg(const std::deque<MsgType>& queue, double targetTime, double maxTimeDiff, MsgType& outMsg, std::size_t& outIndex)
    {
        bool found = false;
        double bestDiff = maxTimeDiff;
        for (std::size_t i = 0; i < queue.size(); ++i)
        {
            const double dt = std::fabs(queue[i].header.stamp.toSec() - targetTime);
            if (dt <= bestDiff)
            {
                bestDiff = dt;
                outMsg = queue[i];
                outIndex = i;
                found = true;
            }
        }
        return found;
    }

    template <typename MsgType>
    void dropMessagesBefore(std::deque<MsgType>& queue, std::size_t keepIndex)
    {
        if (queue.empty())
            return;

        const std::size_t eraseCount = std::min(keepIndex, queue.size());
        queue.erase(queue.begin(), queue.begin() + eraseCount);
    }

    double normalizeYaw(double yaw) const
    {
        return std::atan2(std::sin(yaw), std::cos(yaw));
    }

    double normalizeAngle(double angle) const
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    bool odometryToTransformArray(const nav_msgs::Odometry& odom, float transformOut[6]) const
    {
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        if (!std::isfinite(orientation.x()) || !std::isfinite(orientation.y()) ||
            !std::isfinite(orientation.z()) || !std::isfinite(orientation.w()) ||
            orientation.length2() <= 1e-12)
            return false;

        orientation.normalize();
        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        transformOut[0] = static_cast<float>(roll);
        transformOut[1] = static_cast<float>(pitch);
        transformOut[2] = static_cast<float>(yaw);
        transformOut[3] = odom.pose.pose.position.x;
        transformOut[4] = odom.pose.pose.position.y;
        transformOut[5] = odom.pose.pose.position.z;
        return std::isfinite(transformOut[0]) && std::isfinite(transformOut[1]) &&
               std::isfinite(transformOut[2]) && std::isfinite(transformOut[3]) &&
               std::isfinite(transformOut[4]) && std::isfinite(transformOut[5]);
    }

    double imuPosePriorError(int dim) const
    {
        const double error = static_cast<double>(imuPosePriorTransform[dim] - transformTobeMapped[dim]);
        return dim < 3 ? normalizeAngle(error) : error;
    }

    double imuPosePriorPositionErrorNorm() const
    {
        const double ex = imuPosePriorError(3);
        const double ey = imuPosePriorError(4);
        const double ez = imuPosePriorError(5);
        return std::sqrt(ex * ex + ey * ey + ez * ez);
    }

    double imuPosePriorRollPitchErrorNorm() const
    {
        const double er = imuPosePriorError(0);
        const double ep = imuPosePriorError(1);
        return std::sqrt(er * er + ep * ep);
    }

    int imuPosePriorLmRowCount() const
    {
        if (!useImu || !useImuPosePrior || !useImuPosePriorInLM ||
            !imuPosePriorAvailable || imuPosePriorLmScale <= 0.0)
            return 0;

        int rowCount = 0;
        if (useImuPosePriorRollPitch && imuPosePriorRollPitchErrorNorm() <= imuPosePriorMaxRollPitchError)
            rowCount += 2;
        if (useImuPosePriorYaw && std::abs(imuPosePriorError(2)) <= imuPosePriorMaxYawError)
            rowCount += 1;
        if (useImuPosePriorPosition && imuPosePriorPositionErrorNorm() <= imuPosePriorMaxPositionError)
            rowCount += 3;
        return rowCount;
    }

    int appendImuPosePriorLmRows(cv::Mat& matA, cv::Mat& matB, int rowStart) const
    {
        if (!useImu || !useImuPosePrior || !useImuPosePriorInLM ||
            !imuPosePriorAvailable || imuPosePriorLmScale <= 0.0)
            return 0;

        int row = rowStart;
        const double lmScale = std::max(1e-6, imuPosePriorLmScale);
        auto appendPriorRow = [&](int dim, double sigma)
        {
            const double safeSigma = std::max(1e-6, sigma);
            const float weight = static_cast<float>(lmScale / safeSigma);
            matA.at<float>(row, dim) = weight;
            matB.at<float>(row, 0) = static_cast<float>(imuPosePriorError(dim) * weight);
            ++row;
        };

        if (useImuPosePriorRollPitch && imuPosePriorRollPitchErrorNorm() <= imuPosePriorMaxRollPitchError)
        {
            appendPriorRow(0, imuPosePriorRollPitchSigma);
            appendPriorRow(1, imuPosePriorRollPitchSigma);
        }
        if (useImuPosePriorYaw && std::abs(imuPosePriorError(2)) <= imuPosePriorMaxYawError)
            appendPriorRow(2, imuPosePriorYawSigma);
        if (useImuPosePriorPosition && imuPosePriorPositionErrorNorm() <= imuPosePriorMaxPositionError)
        {
            appendPriorRow(3, imuPosePriorPositionSigma);
            appendPriorRow(4, imuPosePriorPositionSigma);
            appendPriorRow(5, imuPosePriorPositionSigma);
        }
        return row - rowStart;
    }

    double correctedGnssHeadingYaw(const sensor_msgs::Imu& headingMsg) const
    {
        tf::Quaternion headingQuat;
        tf::quaternionMsgToTF(headingMsg.orientation, headingQuat);
        double headingRoll = 0.0, headingPitch = 0.0, headingYaw = 0.0;
        tf::Matrix3x3(headingQuat).getRPY(headingRoll, headingPitch, headingYaw);
        const double headingOffsetRad = static_cast<double>(gnssHeadingOffsetDeg) * M_PI / 180.0;
        return normalizeYaw(headingYaw + headingOffsetRad);
    }

    std::string normalizedGnssHeadingFactorMode() const
    {
        std::string mode = gnssHeadingFactorMode;
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return mode;
    }

    std::string normalizedGpsElevationAxis() const
    {
        std::string axis = gpsElevationAxis;
        std::transform(axis.begin(), axis.end(), axis.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (axis == "x" || axis == "y" || axis == "z")
            return axis;
        return "z";
    }

    bool gpsMeasurementTooNoisy(float noiseX, float noiseY, float noiseZ, const std::string& elevationAxis) const
    {
        if (useGpsElevation)
            return noiseX > gpsCovThreshold || noiseY > gpsCovThreshold || noiseZ > gpsCovThreshold;
        if (elevationAxis == "x")
            return noiseY > gpsCovThreshold || noiseZ > gpsCovThreshold;
        if (elevationAxis == "y")
            return noiseX > gpsCovThreshold || noiseZ > gpsCovThreshold;
        return noiseX > gpsCovThreshold || noiseY > gpsCovThreshold;
    }

    bool gpsMeasurementIsOrigin(float gpsX, float gpsY, float gpsZ, const std::string& elevationAxis) const
    {
        if (useGpsElevation)
            return std::abs(gpsX) < 1e-6 && std::abs(gpsY) < 1e-6 && std::abs(gpsZ) < 1e-6;
        if (elevationAxis == "x")
            return std::abs(gpsY) < 1e-6 && std::abs(gpsZ) < 1e-6;
        if (elevationAxis == "y")
            return std::abs(gpsX) < 1e-6 && std::abs(gpsZ) < 1e-6;
        return std::abs(gpsX) < 1e-6 && std::abs(gpsY) < 1e-6;
    }

    double gpsFactorVariance(double measuredVariance) const
    {
        if (std::isfinite(gpsFactorVarianceOverrideRuntime) && gpsFactorVarianceOverrideRuntime > 0.0)
            return gpsFactorVarianceOverrideRuntime;
        return std::max(measuredVariance, 1.0) * gpsFactorVarianceScale;
    }

    bool initializeFirstPoseFromGnss()
    {
        nav_msgs::Odometry alignedGps;
        sensor_msgs::Imu alignedHeading;
        std::size_t gpsIndex = 0;
        std::size_t headingIndex = 0;
        const double initRoll = static_cast<double>(gnssInitializationRollDeg) * M_PI / 180.0;
        const double initPitch = static_cast<double>(gnssInitializationPitchDeg) * M_PI / 180.0;

        if (!findNearestMsg(gpsQueue, timeLaserInfoCur, gnssInitializationMaxTimeDiff, alignedGps, gpsIndex))
        {
            ROS_WARN_THROTTLE(2.0,
                              "Waiting for GNSS pose to initialize SAM. Need /odometry/gps aligned within %.3f s of lidar stamp.",
                              gnssInitializationMaxTimeDiff);
            return false;
        }

        if (!findNearestMsg(gnssHeadingQueue, timeLaserInfoCur, gnssInitializationMaxTimeDiff, alignedHeading, headingIndex))
        {
            ROS_WARN_THROTTLE(2.0,
                              "Waiting for GNSS heading to initialize SAM. Need %s aligned within %.3f s of lidar stamp.",
                              gnssHeadingTopic.c_str(), gnssInitializationMaxTimeDiff);
            return false;
        }

        dropMessagesBefore(gpsQueue, gpsIndex);
        dropMessagesBefore(gnssHeadingQueue, headingIndex);

        const double headingYaw = correctedGnssHeadingYaw(alignedHeading);

        const double gpsX = alignedGps.pose.pose.position.x;
        const double gpsY = alignedGps.pose.pose.position.y;
        const double gpsZ = alignedGps.pose.pose.position.z;

        transformTobeMapped[0] = static_cast<float>(initRoll);
        transformTobeMapped[1] = static_cast<float>(initPitch);
        transformTobeMapped[2] = static_cast<float>(headingYaw);
        transformTobeMapped[3] = static_cast<float>(gpsX);
        transformTobeMapped[4] = static_cast<float>(gpsY);
        transformTobeMapped[5] = static_cast<float>(gpsZ);

        const double yawVar = (alignedHeading.orientation_covariance[8] > 0.0 && std::isfinite(alignedHeading.orientation_covariance[8]))
                                  ? alignedHeading.orientation_covariance[8]
                                  : 0.03;
        const double levelVar = std::pow(2.0 * M_PI / 180.0, 2);
        const double posVarX = (alignedGps.pose.covariance[0] > 0.0 && std::isfinite(alignedGps.pose.covariance[0]))
                                   ? alignedGps.pose.covariance[0]
                                   : 0.5;
        const double posVarY = (alignedGps.pose.covariance[7] > 0.0 && std::isfinite(alignedGps.pose.covariance[7]))
                                   ? alignedGps.pose.covariance[7]
                                   : 0.5;
        const double posVarZ = (alignedGps.pose.covariance[14] > 0.0 && std::isfinite(alignedGps.pose.covariance[14]))
                                   ? alignedGps.pose.covariance[14]
                                   : 1.0;

        gnssInitializationPose = gtsam::Pose3(gtsam::Rot3::RzRyRx(initRoll, initPitch, headingYaw), gtsam::Point3(gpsX, gpsY, gpsZ));
        gnssInitializationNoise = noiseModel::Diagonal::Variances(
            (Vector(6) << levelVar, levelVar, yawVar, posVarX, posVarY, posVarZ).finished());
        gnssInitializationDone = true;
        gnssHeadingAnchorReady = true;
        lastGnssHeadingKeyIndex = 0;
        lastGnssHeadingYaw = headingYaw;
        lastGnssHeadingYawVar = yawVar;
        lastGnssHeadingFactorTime = alignedHeading.header.stamp.toSec();

        ROS_WARN_STREAM("[GNSS INIT] Initialized SAM world frame from aligned GNSS pose+yaw."
                        << " lidar_t=" << std::fixed << std::setprecision(3) << timeLaserInfoCur
                        << ", gps_t=" << alignedGps.header.stamp.toSec()
                        << ", heading_t=" << alignedHeading.header.stamp.toSec()
                        << ", dt_gps=" << std::fabs(alignedGps.header.stamp.toSec() - timeLaserInfoCur)
                        << ", dt_heading=" << std::fabs(alignedHeading.header.stamp.toSec() - timeLaserInfoCur)
                        << ", pose=(" << gpsX << ", " << gpsY << ", " << gpsZ << ")"
                        << ", rpy=(" << initRoll << ", " << initPitch << ", " << headingYaw << ")");
        return true;
    }

    bool findAlignedHeadingForCurrentScan(sensor_msgs::Imu& alignedHeading, std::size_t& headingIndex, double maxTimeDiff)
    {
        if (!findNearestMsg(gnssHeadingQueue, timeLaserInfoCur, maxTimeDiff, alignedHeading, headingIndex))
            return false;

        while (!gnssHeadingQueue.empty() && gnssHeadingQueue.front().header.stamp.toSec() < timeLaserInfoCur - maxTimeDiff)
            gnssHeadingQueue.pop_front();

        return true;
    }















    bool exportKeyframeDataset(const string& saveMapDirectory)
    {
      if (!exportKeyframeDatasetOnSave)
          return true;

      const string keyframeDirectory = saveMapDirectory + "/keyframes";
      const string fullDirectory = keyframeDirectory + "/full";
      const string cornerDirectory = keyframeDirectory + "/corner";
      const string surfDirectory = keyframeDirectory + "/surf";
      const string labelsDirectory = keyframeDirectory + "/labels";
      boost::system::error_code mkdirError;
      for (const string& directory : {fullDirectory, cornerDirectory, surfDirectory, labelsDirectory})
      {
          boost::filesystem::create_directories(directory, mkdirError);
          if (mkdirError || !boost::filesystem::is_directory(directory))
              break;
      }
      if (mkdirError)
      {
          ROS_ERROR("Failed to create keyframe export directory %s: %s",
                    keyframeDirectory.c_str(), mkdirError.message().c_str());
          return false;
      }

      std::ofstream posesFile(keyframeDirectory + "/poses.csv");
      if (!posesFile.is_open())
      {
          ROS_ERROR("Failed to write keyframe poses: %s", (keyframeDirectory + "/poses.csv").c_str());
          return false;
      }

      posesFile << "keyframe_id,time,x,y,z,roll,pitch,yaw,intensity,full_points,corner_points,surf_points\n";

      std::ofstream labelsFile(labelsDirectory + "/changed_points.csv");
      if (!labelsFile.is_open())
      {
          ROS_ERROR("Failed to write keyframe labels: %s", (labelsDirectory + "/changed_points.csv").c_str());
          return false;
      }
      labelsFile << "keyframe_id,cloud_type,point_id,state,change_mask,change_type,change_role,region_id,epoch,confirmation_count,feature_weight\n";

      std::ofstream readmeFile(keyframeDirectory + "/README.md");
      if (readmeFile.is_open())
      {
          readmeFile
              << "# Exported TCR-LM Keyframes\n\n"
              << "- `poses.csv`: keyframe poses in the map/odometry frame.\n"
              << "- `full/*.pcd`: full-resolution keyframe clouds in the local LiDAR frame.\n"
              << "- `corner/*.pcd`: corner feature keyframes in the local LiDAR frame.\n"
              << "- `surf/*.pcd`: surface feature keyframes in the local LiDAR frame.\n"
              << "- `labels/changed_points.csv`: non-stable point labels written by terrain change sensing.\n\n"
              << "Use `pcl::getTransformation(x, y, z, roll, pitch, yaw)` or the provided offline scripts to transform local keyframes into the map frame.\n";
      }

      int ret = 0;
      int exportedLabelCount = 0;
      const int totalKeyframes = cloudKeyPoses6D->size();
      for (int i = 0; i < totalKeyframes; ++i)
      {
          std::ostringstream name;
          name << std::setw(6) << std::setfill('0') << i << ".pcd";
          const string fileName = name.str();

          const int fullPoints = i < (int)fullCloudKeyFrames.size() ? fullCloudKeyFrames[i]->size() : 0;
          const int cornerPoints = i < (int)cornerCloudKeyFrames.size() ? cornerCloudKeyFrames[i]->size() : 0;
          const int surfPoints = i < (int)surfCloudKeyFrames.size() ? surfCloudKeyFrames[i]->size() : 0;

          if (i < (int)fullCloudKeyFrames.size())
              ret |= pcl::io::savePCDFileBinary(fullDirectory + "/" + fileName, *fullCloudKeyFrames[i]);
          if (i < (int)cornerCloudKeyFrames.size())
              ret |= pcl::io::savePCDFileBinary(cornerDirectory + "/" + fileName, *cornerCloudKeyFrames[i]);
          if (i < (int)surfCloudKeyFrames.size())
              ret |= pcl::io::savePCDFileBinary(surfDirectory + "/" + fileName, *surfCloudKeyFrames[i]);

          const PointTypePose& pose = cloudKeyPoses6D->points[i];
          posesFile << i << ","
                    << std::setprecision(16) << pose.time << ","
                    << pose.x << "," << pose.y << "," << pose.z << ","
                    << pose.roll << "," << pose.pitch << "," << pose.yaw << ","
                    << pose.intensity << ","
                    << fullPoints << "," << cornerPoints << "," << surfPoints << "\n";

          auto exportLabelsForCloud = [&](const std::string& cloudType,
                                          const std::vector<std::vector<uint8_t>>& states,
                                          const std::vector<std::vector<uint8_t>>& masks,
                                          const std::vector<std::vector<uint8_t>>& types,
                                          const std::vector<std::vector<uint8_t>>& roles,
                                          const std::vector<std::vector<uint32_t>>& regions,
                                          const std::vector<std::vector<uint32_t>>& epochs,
                                          const std::vector<std::vector<uint16_t>>& confirmations)
          {
              if (i >= (int)states.size() || i >= (int)masks.size() || i >= (int)types.size() ||
                  i >= (int)roles.size() || i >= (int)regions.size() || i >= (int)epochs.size() ||
                  i >= (int)confirmations.size())
                  return;

              const int labelCount = states[i].size();
              for (int pointId = 0; pointId < labelCount; ++pointId)
              {
                  const uint8_t mask = pointId < (int)masks[i].size() ? masks[i][pointId] : 0;
                  const uint8_t state = states[i][pointId];
                  const uint16_t confirmationCount = pointId < (int)confirmations[i].size() ?
                      confirmations[i][pointId] : 0;
                  const float featureWeight = terrainConfidenceForState(state, confirmationCount);
                  if (mask == 0 && state == VALID_STABLE)
                      continue;

                  labelsFile << i << ","
                             << cloudType << ","
                             << pointId << ","
                             << static_cast<int>(state) << ","
                             << static_cast<int>(mask) << ","
                             << static_cast<int>(pointId < (int)types[i].size() ? types[i][pointId] : CHANGE_NONE) << ","
                             << static_cast<int>(pointId < (int)roles[i].size() ? roles[i][pointId] : ROLE_NONE) << ","
                             << (pointId < (int)regions[i].size() ? regions[i][pointId] : 0) << ","
                             << (pointId < (int)epochs[i].size() ? epochs[i][pointId] : 0) << ","
                             << confirmationCount << ","
                             << featureWeight << "\n";
                  exportedLabelCount++;
              }
          };

          exportLabelsForCloud("full", fullPointState, fullChangeMask, fullChangeType,
                               fullChangeRole, fullChangeRegionId, fullChangeEpoch, fullChangeConfirmations);
          exportLabelsForCloud("corner", cornerPointState, cornerChangeMask, cornerChangeType,
                               cornerChangeRole, cornerChangeRegionId, cornerChangeEpoch, cornerChangeConfirmations);
          exportLabelsForCloud("surf", surfPointState, surfChangeMask, surfChangeType,
                               surfChangeRole, surfChangeRegionId, surfChangeEpoch, surfChangeConfirmations);

          cout << "\r" << std::flush << "Exporting keyframe dataset " << i + 1 << " of " << totalKeyframes << " ...";
      }
      cout << endl;

      if (ret != 0)
      {
          ROS_ERROR("Failed to export one or more keyframe PCD files under: %s", keyframeDirectory.c_str());
          return false;
      }

      ROS_WARN_STREAM("[keyframe_export] saved " << totalKeyframes
          << " keyframes to " << keyframeDirectory
          << ", changed_labels=" << exportedLabelCount);
      return true;
    }



    bool saveMapService(tcrlm::save_mapRequest& req, tcrlm::save_mapResponse& res)
    {
      string saveMapDirectory;
      auto resolveSavePath = [](const string& path) -> string
      {
          if (path.empty())
              return string();

          if (path.front() == '/')
              return path;

          const char* home = std::getenv("HOME");
          if (home == nullptr)
              return path;

          if (path.front() == '~')
              return string(home) + path.substr(1);

          return string(home) + "/" + path;
      };

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = resolveSavePath(savePCDDirectory);
      else saveMapDirectory = resolveSavePath(req.destination);
      const boost::filesystem::path requestedPath(saveMapDirectory);
      bool hasParentTraversal = false;
      for (const auto& component : requestedPath)
      {
          if (component == "..")
          {
              hasParentTraversal = true;
              break;
          }
      }
      boost::system::error_code pathError;
      const boost::filesystem::path currentPath = boost::filesystem::current_path(pathError).lexically_normal();
      const boost::filesystem::path normalizedPath =
          (requestedPath.is_absolute() ? requestedPath : currentPath / requestedPath).lexically_normal();
      const char* homeEnvironment = std::getenv("HOME");
      const boost::filesystem::path homePath = homeEnvironment == nullptr
          ? boost::filesystem::path() : boost::filesystem::path(homeEnvironment).lexically_normal();
      if (pathError || saveMapDirectory.empty() || hasParentTraversal ||
          normalizedPath.empty() || normalizedPath == normalizedPath.root_path() ||
          normalizedPath == boost::filesystem::path("/home") ||
          (!homePath.empty() && normalizedPath == homePath) ||
          normalizedPath == currentPath)
      {
          ROS_ERROR("Refusing unsafe map save destination: %s", saveMapDirectory.c_str());
          res.success = false;
          return false;
      }
      saveMapDirectory = normalizedPath.string();
      cout << "Save destination: " << saveMapDirectory << endl;


      boost::filesystem::create_directories(saveMapDirectory, pathError);
      if (pathError || !boost::filesystem::is_directory(saveMapDirectory))
      {
          ROS_ERROR("Failed to prepare save directory %s: %s",
                    saveMapDirectory.c_str(), pathError.message().c_str());
          res.success = false;
          return false;
      }

      int ret = 0;
      ret |= pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      ret |= pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      if (!exportKeyframeDataset(saveMapDirectory))
          ret = 1;

      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformCornerKeyframeForMaintainedMap(i, &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformSurfKeyframeForMaintainedMap(i, &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;


        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        ret |= pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);

        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        ret |= pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {

        ret |= pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);

        ret |= pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }


      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;

      ret |= pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      if (useTerrainChangeMaintenance())
      {
          pcl::PointCloud<PointType>::Ptr changedRecent(new pcl::PointCloud<PointType>());
          pcl::PointCloud<PointType>::Ptr changedOld(new pcl::PointCloud<PointType>());
          collectChangedFeaturePoints(0, CHANGE_NONE, ROLE_RECENT_SURFACE, TERRAIN_POINT_FULL, changedRecent, false);
          collectChangedFeaturePoints(0, CHANGE_NONE, ROLE_OLD_SURFACE, TERRAIN_POINT_FULL, changedOld, false);
          if (!changedRecent->empty())
              ret |= pcl::io::savePCDFileBinary(saveMapDirectory + "/TerrainChangeRecent.pcd", *changedRecent);
          if (!changedOld->empty())
              ret |= pcl::io::savePCDFileBinary(saveMapDirectory + "/TerrainChangeOld.pcd", *changedOld);
      }
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        tcrlm::save_mapRequest  req;
        tcrlm::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());


        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;

        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity);
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }


        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformCornerKeyframeForMaintainedMap(thisKeyInd, &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformSurfKeyframeForMaintainedMap(thisKeyInd, &cloudKeyPoses6D->points[thisKeyInd]);
        }

        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize);
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }












    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();


        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;


        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }


        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);


        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;


        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }


        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();

        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);

        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);


        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();


        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;


        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;


        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {

        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;


        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }


        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {

        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformCornerKeyframeForMaintainedMap(keyNear, &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformSurfKeyframeForMaintainedMap(keyNear, &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;


        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;

        visualization_msgs::MarkerArray markerArray;

        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3;
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;

        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }











    void updateInitialGuess()
    {

        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;

        if (cloudKeyPoses3D->points.empty())
        {
            if (waitForGnssInitialization && gnssInitializationDone)
            {
                lastImuTransformation = pcl::getTransformation(0, 0, 0,
                                                               cloudInfo.imuRollInit,
                                                               cloudInfo.imuPitchInit,
                                                               cloudInfo.imuYawInit);
                return;
            }

            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            return;
        }


        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomAvailable == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ,
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
                return;
            }
        }


        if (cloudInfo.imuAvailable == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;


        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }


        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {

        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())
            {

                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {

                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformCornerKeyframeForMaintainedMap(thisKeyInd, &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformSurfKeyframeForMaintainedMap(thisKeyInd, &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }

        }


        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();

        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();


        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;








        extractNearby();
    }

    void downsampleCurrentScan()
    {

        laserCloudFullResDS->clear();
        downSizeFilterTerrainChangeInput.setInputCloud(laserCloudFullRes);
        downSizeFilterTerrainChangeInput.filter(*laserCloudFullResDS);


        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    float terrainNeighborFeatureWeight(
        const pcl::PointCloud<PointType>::Ptr& mapCloud,
        const std::vector<int>& pointSearchInd) const
    {
        if (!useTerrainChangeMapDownweighting() || !mapCloud)
            return 1.0f;
        if (pointSearchInd.empty())
            return 1.0f;

        float weightSum = 0.0f;
        int weightCount = 0;
        for (int index : pointSearchInd)
        {
            if (index < 0 || index >= (int)mapCloud->size())
                continue;
            const float weight = mapCloud->points[index].intensity;
            if (!std::isfinite(weight))
                continue;
            weightSum += std::min(1.0f, std::max(terrainChangeMinFeatureWeight, weight));
            weightCount++;
        }
        if (weightCount <= 0)
            return 1.0f;
        return std::min(1.0f, std::max(terrainChangeMinFeatureWeight, weightSum / weightCount));
    }

    void cornerOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, linePlaneFittingKnn, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            if (static_cast<int>(pointSearchSqDis.size()) == linePlaneFittingKnn &&
                pointSearchSqDis[linePlaneFittingKnn - 1] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < linePlaneFittingKnn; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= linePlaneFittingKnn;
                cy /= linePlaneFittingKnn;
                cz /= linePlaneFittingKnn;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < linePlaneFittingKnn; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= linePlaneFittingKnn;
                a12 /= linePlaneFittingKnn;
                a13 /= linePlaneFittingKnn;
                a22 /= linePlaneFittingKnn;
                a23 /= linePlaneFittingKnn;
                a33 /= linePlaneFittingKnn;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        const float featureWeight = terrainNeighborFeatureWeight(laserCloudCornerFromMapDS, pointSearchInd);
                        const float coeffScale = std::sqrt(featureWeight);
                        coeff.x *= coeffScale;
                        coeff.y *= coeffScale;
                        coeff.z *= coeffScale;
                        coeff.intensity *= coeffScale;
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        cornerMatchRawResidualVec[i] = std::fabs(ld2);
                        cornerMatchFeatureWeightVec[i] = featureWeight;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, linePlaneFittingKnn, pointSearchInd, pointSearchSqDis);

            Eigen::MatrixXf matA0(linePlaneFittingKnn, 3);
            Eigen::VectorXf matB0(linePlaneFittingKnn);
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (static_cast<int>(pointSearchSqDis.size()) == linePlaneFittingKnn &&
                pointSearchSqDis[linePlaneFittingKnn - 1] < 1.0) {
                for (int j = 0; j < linePlaneFittingKnn; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < linePlaneFittingKnn; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        const float featureWeight = terrainNeighborFeatureWeight(laserCloudSurfFromMapDS, pointSearchInd);
                        const float coeffScale = std::sqrt(featureWeight);
                        coeff.x *= coeffScale;
                        coeff.y *= coeffScale;
                        coeff.z *= coeffScale;
                        coeff.intensity *= coeffScale;
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        surfMatchRawResidualVec[i] = std::fabs(pd2);
                        surfMatchFeatureWeightVec[i] = featureWeight;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        currentConstraintMatchDiag.reset();


        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
                currentConstraintMatchDiag.corner_used++;
                const double rawResidual = std::fabs(cornerMatchRawResidualVec[i]);
                const double weightedResidual = std::fabs(coeffSelCornerVec[i].intensity);
                const double featureWeight = std::min(1.0f, std::max(terrainChangeMinFeatureWeight, cornerMatchFeatureWeightVec[i]));
                currentConstraintMatchDiag.corner_raw_abs_res_sum += rawResidual;
                currentConstraintMatchDiag.corner_weighted_abs_res_sum += weightedResidual;
                if (featureWeight < 0.999)
                {
                    currentConstraintMatchDiag.corner_affected++;
                    currentConstraintMatchDiag.affected_raw_abs_res_sum += rawResidual;
                    currentConstraintMatchDiag.affected_weighted_abs_res_sum += weightedResidual;
                    currentConstraintMatchDiag.affected_weight_sum += featureWeight;
                }
            }
        }

        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
                currentConstraintMatchDiag.surf_used++;
                const double rawResidual = std::fabs(surfMatchRawResidualVec[i]);
                const double weightedResidual = std::fabs(coeffSelSurfVec[i].intensity);
                const double featureWeight = std::min(1.0f, std::max(terrainChangeMinFeatureWeight, surfMatchFeatureWeightVec[i]));
                currentConstraintMatchDiag.surf_raw_abs_res_sum += rawResidual;
                currentConstraintMatchDiag.surf_weighted_abs_res_sum += weightedResidual;
                if (featureWeight < 0.999)
                {
                    currentConstraintMatchDiag.surf_affected++;
                    currentConstraintMatchDiag.affected_raw_abs_res_sum += rawResidual;
                    currentConstraintMatchDiag.affected_weighted_abs_res_sum += weightedResidual;
                    currentConstraintMatchDiag.affected_weight_sum += featureWeight;
                }
            }
        }

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {










        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        const int priorResidualDim = imuPosePriorLmRowCount();
        const int residualNum = laserCloudSelNum + priorResidualDim;

        cv::Mat matA(residualNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, residualNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(residualNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {

            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;

            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;

            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        const int appendedImuPriorRows = appendImuPosePriorLmRows(matA, matB, laserCloudSelNum);
        if (debugImuPosePrior && iterCount == 0)
        {
            constexpr double radToDeg = 57.29577951308232;
            ROS_INFO_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                << "[IMU pose prior] lm_rows=" << appendedImuPriorRows
                << ", age=" << imuPosePriorAge
                << ", prior_p=(" << imuPosePriorTransform[3]
                << ", " << imuPosePriorTransform[4]
                << ", " << imuPosePriorTransform[5] << ")"
                << ", prior_rpy_deg=(" << imuPosePriorTransform[0] * radToDeg
                << ", " << imuPosePriorTransform[1] * radToDeg
                << ", " << imuPosePriorTransform[2] * radToDeg << ")"
                << ", err_p_norm=" << imuPosePriorPositionErrorNorm()
                << ", err_rpy_deg=(" << imuPosePriorError(0) * radToDeg
                << ", " << imuPosePriorError(1) * radToDeg
                << ", " << imuPosePriorError(2) * radToDeg << ")"
                << ", lm_scale=" << imuPosePriorLmScale);
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;
        }
        return false;
    }

    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        currentConstraintMatchDiag.reset();

        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            lastFrameEnoughFeatures = true;
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;
            }

            transformUpdate();
        } else {
            lastFrameEnoughFeatures = false;
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void guardReverseMotion()
    {
        currentReverseMotionSuspected = false;
        currentOptimizationRejected = false;
        currentMotionDirCos = 1.0f;

        if (!hasAcceptedPose)
            return;

        const float dx = transformTobeMapped[3] - lastAcceptedX;
        const float dy = transformTobeMapped[4] - lastAcceptedY;
        const float dz = transformTobeMapped[5] - lastAcceptedZ;
        const float deltaDist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (deltaDist <= reverseMotionGuardMinDelta)
            return;

        const float curDirX = dx / deltaDist;
        const float curDirY = dy / deltaDist;
        const float curDirZ = dz / deltaDist;

        if (hasAcceptedMotionDir)
        {
            currentMotionDirCos = curDirX * lastAcceptedMotionDirX +
                                  curDirY * lastAcceptedMotionDirY +
                                  curDirZ * lastAcceptedMotionDirZ;
            currentReverseMotionSuspected = currentMotionDirCos < reverseMotionGuardCosThreshold;
        }

        if (enableReverseMotionGuard && currentReverseMotionSuspected)
        {
            std::copy(std::begin(initialGuessTransform), std::end(initialGuessTransform), std::begin(transformTobeMapped));
            incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
            currentOptimizationRejected = true;
            ROS_WARN_STREAM(std::fixed << std::setprecision(3)
                << "[TCR-LM guard] Reverse motion suspected, fallback to updateInitialGuess."
                << " motion_dir_cos=" << currentMotionDirCos
                << " delta_dist=" << deltaDist);
        }
    }

    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;


                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;


                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            if (waitForGnssInitialization && gnssInitializationDone)
            {
                gtSAMgraph.add(PriorFactor<Pose3>(0, gnssInitializationPose, gnssInitializationNoise));
                initialEstimate.insert(0, gnssInitializationPose);
            }
            else
            {
                noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished());
                gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
                initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
            }
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        if (!enableGpsFactor)
            return;

        if (gpsQueue.empty())
            return;


        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {

            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 0.1)
                return;
        }


        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        static PointType lastGPSPoint;

        nav_msgs::Odometry thisGPS;
        std::size_t gpsIndex = 0;
        if (!findNearestMsg(gpsQueue, timeLaserInfoCur, gpsFactorMaxTimeDiffRuntime, thisGPS, gpsIndex))
        {
            while (!gpsQueue.empty() && gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - gpsFactorMaxTimeDiffRuntime)
                gpsQueue.pop_front();
            return;
        }
        dropMessagesBefore(gpsQueue, gpsIndex);
        if (!gpsQueue.empty())
            gpsQueue.pop_front();

        float noise_x = thisGPS.pose.covariance[0];
        float noise_y = thisGPS.pose.covariance[7];
        float noise_z = thisGPS.pose.covariance[14];

        float gps_x = thisGPS.pose.pose.position.x;
        float gps_y = thisGPS.pose.pose.position.y;
        float gps_z = thisGPS.pose.pose.position.z;
        std::string elevationAxis = "z";
        if (!useGpsElevation)
        {
            elevationAxis = normalizedGpsElevationAxis();
            if (elevationAxis == "x")
            {
                gps_x = transformTobeMapped[3];
                noise_x = 0.01;
            }
            else if (elevationAxis == "y")
            {
                gps_y = transformTobeMapped[4];
                noise_y = 0.01;
            }
            else
            {
                gps_z = transformTobeMapped[5];
                noise_z = 0.01;
            }
        }
        if (gpsMeasurementTooNoisy(noise_x, noise_y, noise_z, elevationAxis))
            return;

        if (gpsMeasurementIsOrigin(gps_x, gps_y, gps_z, elevationAxis))
            return;

        PointType curGPSPoint;
        curGPSPoint.x = gps_x;
        curGPSPoint.y = gps_y;
        curGPSPoint.z = gps_z;
        if (gpsFactorCount > 0 && pointDistance(curGPSPoint, lastGPSPoint) < gpsFactorMinDistanceRuntime)
            return;
        lastGPSPoint = curGPSPoint;

        gtsam::Vector Vector3(3);
        Vector3 << gpsFactorVariance(static_cast<double>(noise_x)),
                   gpsFactorVariance(static_cast<double>(noise_y)),
                   gpsFactorVariance(static_cast<double>(noise_z));
        noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
        gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
        gtSAMgraph.add(gps_factor);
        gpsFactorCount++;
        ROS_WARN_STREAM("[GNSS] GPSFactor added to SAM graph at keyframe "
                        << cloudKeyPoses3D->size()
                        << ", total=" << gpsFactorCount);
        ROS_INFO_STREAM("[mapOptimization] Added GPSFactor #" << gpsFactorCount
                        << " at keyframe " << cloudKeyPoses3D->size()
                        << ", gps=(" << gps_x << ", " << gps_y << ", " << gps_z << ")"
                        << ", noise=(" << noise_x << ", " << noise_y << ", " << noise_z << ")"
                        << ", factor_var=(" << Vector3(0) << ", " << Vector3(1) << ", " << Vector3(2) << ")"
                        << ", dt_gps=" << std::fabs(thisGPS.header.stamp.toSec() - timeLaserInfoCur)
                        << ", poseCov=(" << poseCovariance(3,3) << ", " << poseCovariance(4,4) << ")");

        aLoopIsClosed = true;
    }

    void addGNSSHeadingFactor()
    {
        if (!enableGnssHeadingFactor || gnssHeadingQueue.empty() || cloudKeyPoses3D->points.empty())
            return;

        const std::string headingMode = normalizedGnssHeadingFactorMode();
        if (headingMode == "off")
            return;

        sensor_msgs::Imu alignedHeading;
        std::size_t headingIndex = 0;
        if (!findAlignedHeadingForCurrentScan(alignedHeading, headingIndex, gnssHeadingFactorMaxTimeDiff))
            return;

        const double headingTime = alignedHeading.header.stamp.toSec();
        if (lastGnssHeadingFactorTime >= 0.0 && std::fabs(headingTime - lastGnssHeadingFactorTime) < 1e-3)
            return;

        const double headingYaw = correctedGnssHeadingYaw(alignedHeading);

        const double yawVar =
            (alignedHeading.orientation_covariance[8] > 0.0 && std::isfinite(alignedHeading.orientation_covariance[8]))
                ? alignedHeading.orientation_covariance[8]
                : static_cast<double>(gnssHeadingFactorStd) * static_cast<double>(gnssHeadingFactorStd);

        const int currentKeyIndex = static_cast<int>(cloudKeyPoses3D->size());
        if (headingMode == "absolute")
        {
            const double currentX = transformTobeMapped[3];
            const double currentY = transformTobeMapped[4];
            const double currentZ = transformTobeMapped[5];
            const double currentRoll = transformTobeMapped[0];
            const double currentPitch = transformTobeMapped[1];

            noiseModel::Diagonal::shared_ptr headingNoise = noiseModel::Diagonal::Variances(
                (Vector(6) << 1e6, 1e6, yawVar, 1e6, 1e6, 1e6).finished());
            gtsam::Pose3 headingPose(
                gtsam::Rot3::RzRyRx(currentRoll, currentPitch, headingYaw),
                gtsam::Point3(currentX, currentY, currentZ));
            gtSAMgraph.add(PriorFactor<Pose3>(currentKeyIndex, headingPose, headingNoise));
            gnssHeadingFactorCount++;
            lastGnssHeadingFactorTime = headingTime;

            ROS_WARN_STREAM("[GNSS] HeadingFactor added to SAM graph at keyframe "
                            << currentKeyIndex
                            << ", total=" << gnssHeadingFactorCount
                            << ", mode=absolute"
                            << ", yaw=" << headingYaw
                            << ", dt=" << std::fabs(headingTime - timeLaserInfoCur));
            return;
        }

        if (!gnssHeadingAnchorReady)
        {
            gnssHeadingAnchorReady = true;
            lastGnssHeadingKeyIndex = std::max(0, currentKeyIndex - 1);
            lastGnssHeadingYaw = headingYaw;
            lastGnssHeadingYawVar = yawVar;
            lastGnssHeadingFactorTime = headingTime;
            return;
        }

        if (currentKeyIndex <= lastGnssHeadingKeyIndex)
            return;

        const int previousKeyIndex = lastGnssHeadingKeyIndex;
        const double deltaYaw = normalizeYaw(headingYaw - lastGnssHeadingYaw);
        const double deltaYawVar = std::max(yawVar + lastGnssHeadingYawVar, 1e-6);

        noiseModel::Diagonal::shared_ptr headingNoise = noiseModel::Diagonal::Variances(
            (Vector(6) << 1e6, 1e6, deltaYawVar, 1e6, 1e6, 1e6).finished());
        gtsam::Pose3 headingDeltaPose(
            gtsam::Rot3::RzRyRx(0.0, 0.0, deltaYaw),
            gtsam::Point3(0.0, 0.0, 0.0));
        gtSAMgraph.add(BetweenFactor<Pose3>(previousKeyIndex, currentKeyIndex, headingDeltaPose, headingNoise));
        gnssHeadingFactorCount++;
        lastGnssHeadingKeyIndex = currentKeyIndex;
        lastGnssHeadingYaw = headingYaw;
        lastGnssHeadingYawVar = yawVar;
        lastGnssHeadingFactorTime = headingTime;

        ROS_WARN_STREAM("[GNSS] HeadingDeltaFactor added to SAM graph from keyframe "
                        << previousKeyIndex
                        << " to " << currentKeyIndex
                        << ", total=" << gnssHeadingFactorCount
                        << ", mode=relative"
                        << ", delta_yaw=" << deltaYaw
                        << ", dt=" << std::fabs(headingTime - timeLaserInfoCur));
    }

    void addImuPosePriorGraphFactor()
    {
        if (!useImu || !useImuPosePrior || !useImuPosePriorInGraph ||
            !imuPosePriorAvailable || imuPosePriorGraphScale <= 0.0)
            return;

        const int currentKeyIndex = static_cast<int>(cloudKeyPoses3D->size());
        const double roll = useImuPosePriorRollPitch ? imuPosePriorTransform[0] : transformTobeMapped[0];
        const double pitch = useImuPosePriorRollPitch ? imuPosePriorTransform[1] : transformTobeMapped[1];
        const double yaw = useImuPosePriorYaw ? imuPosePriorTransform[2] : transformTobeMapped[2];
        const double x = useImuPosePriorPosition ? imuPosePriorTransform[3] : transformTobeMapped[3];
        const double y = useImuPosePriorPosition ? imuPosePriorTransform[4] : transformTobeMapped[4];
        const double z = useImuPosePriorPosition ? imuPosePriorTransform[5] : transformTobeMapped[5];

        const double disabledVar = 1e6;
        const double scale2 = imuPosePriorGraphScale * imuPosePriorGraphScale;
        const double rpVar = (imuPosePriorRollPitchSigma * imuPosePriorRollPitchSigma) / scale2;
        const double yawVar = (imuPosePriorYawSigma * imuPosePriorYawSigma) / scale2;
        const double posVar = (imuPosePriorPositionSigma * imuPosePriorPositionSigma) / scale2;

        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances(
            (Vector(6) << (useImuPosePriorRollPitch ? rpVar : disabledVar),
                         (useImuPosePriorRollPitch ? rpVar : disabledVar),
                         (useImuPosePriorYaw ? yawVar : disabledVar),
                         (useImuPosePriorPosition ? posVar : disabledVar),
                         (useImuPosePriorPosition ? posVar : disabledVar),
                         (useImuPosePriorPosition ? posVar : disabledVar)).finished());
        gtsam::Pose3 priorPose(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        gtSAMgraph.add(PriorFactor<Pose3>(currentKeyIndex, priorPose, priorNoise));

        if (debugImuPosePrior)
        {
            ROS_WARN_STREAM_THROTTLE(1.0, "[IMU pose prior] graph PriorFactor keyframe="
                                    << currentKeyIndex
                                    << ", age=" << imuPosePriorAge
                                    << ", scale=" << imuPosePriorGraphScale);
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;


        addOdomFactor();


        addGPSFactor();


        addGNSSHeadingFactor();


        addImuPosePriorGraphFactor();


        addLoopFactor();





        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();


        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);



        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size();
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ;
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        if (logKeyframeAddition)
        {
            const std::size_t keyIndex = cloudKeyPoses6D->size() - 1;
            if (keyIndex == 0)
            {
                ROS_WARN_STREAM("[mapOptimization] Keyframe added #" << keyIndex
                                << " (first keyframe)"
                                << ", time=" << thisPose6D.time
                                << ", pose=(" << thisPose6D.x << ", " << thisPose6D.y << ", " << thisPose6D.z << ")");
            }
            else
            {
                const PointTypePose &prevPose = cloudKeyPoses6D->points[keyIndex - 1];
                const Eigen::Affine3f transPrev = pclPointToAffine3f(prevPose);
                const Eigen::Affine3f transCurr = pclPointToAffine3f(thisPose6D);
                const Eigen::Affine3f transBetween = transPrev.inverse() * transCurr;
                float dx, dy, dz, droll, dpitch, dyaw;
                pcl::getTranslationAndEulerAngles(transBetween, dx, dy, dz, droll, dpitch, dyaw);
                const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                const bool triggerDist = dist >= surroundingkeyframeAddingDistThreshold;
                const bool triggerAngle =
                    std::fabs(droll) >= surroundingkeyframeAddingAngleThreshold ||
                    std::fabs(dpitch) >= surroundingkeyframeAddingAngleThreshold ||
                    std::fabs(dyaw) >= surroundingkeyframeAddingAngleThreshold;
                const std::string trigger =
                    triggerDist && triggerAngle ? "distance+angle" :
                    triggerDist ? "distance" :
                    triggerAngle ? "angle" : "unknown";

                ROS_WARN_STREAM("[mapOptimization] Keyframe added #" << keyIndex
                                << ", dt=" << (thisPose6D.time - prevPose.time)
                                << " s, dist=" << dist
                                << " m (threshold=" << surroundingkeyframeAddingDistThreshold << ")"
                                << ", d_rpy_deg=(" << pcl::rad2deg(droll) << ", "
                                << pcl::rad2deg(dpitch) << ", " << pcl::rad2deg(dyaw) << ")"
                                << " (threshold_deg=" << pcl::rad2deg(surroundingkeyframeAddingAngleThreshold) << ")"
                                << ", trigger=" << trigger);
            }
        }




        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);


        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();


        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisFullKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        pcl::copyPointCloud(*laserCloudFullResDS,     *thisFullKeyFrame);


        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        fullCloudKeyFrames.push_back(thisFullKeyFrame);

        const int keyframeId = static_cast<int>(cloudKeyPoses6D->size()) - 1;
        initializeTerrainChangeStateForNewKeyframe(keyframeId, thisCornerKeyFrame, thisSurfKeyFrame, thisFullKeyFrame);
        publishTerrainChangeSnapshotIfNeeded(keyframeId);


        updatePath(thisPose6D);
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {

            laserCloudMapContainer.clear();

            globalPath.poses.clear();

            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {

        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);


        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, lidarFrame);
        br.sendTransform(trans_odom_to_lidar);


        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental;
        static Eigen::Affine3f increOdomAffine;
        float increX = transformTobeMapped[3];
        float increY = transformTobeMapped[4];
        float increZ = transformTobeMapped[5];
        float increRoll = transformTobeMapped[0];
        float increPitch = transformTobeMapped[1];
        float increYaw = transformTobeMapped[2];
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;


                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;


                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            increX = x;
            increY = y;
            increZ = z;
            increRoll = roll;
            increPitch = pitch;
            increYaw = yaw;
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);

        if (logLioStatePerFrame)
        {
            static bool hasLastLoggedPose = false;
            static double lastLoggedTime = 0.0;
            static float lastLoggedX = 0.0f;
            static float lastLoggedY = 0.0f;
            static float lastLoggedZ = 0.0f;
            static float lastLoggedYaw = 0.0f;

            const double dt = hasLastLoggedPose ? (timeLaserInfoCur - lastLoggedTime) : 0.0;
            const float dx = hasLastLoggedPose ? (transformTobeMapped[3] - lastLoggedX) : 0.0f;
            const float dy = hasLastLoggedPose ? (transformTobeMapped[4] - lastLoggedY) : 0.0f;
            const float dz = hasLastLoggedPose ? (transformTobeMapped[5] - lastLoggedZ) : 0.0f;
            const float deltaDist = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double speed = (hasLastLoggedPose && dt > 1e-6) ? (deltaDist / dt) : 0.0;
            float motionDirCos = currentMotionDirCos;
            bool reverseMotionSuspected = currentReverseMotionSuspected;

            float deltaYawRad = 0.0f;
            if (hasLastLoggedPose)
            {
                deltaYawRad = transformTobeMapped[2] - lastLoggedYaw;
                while (deltaYawRad > static_cast<float>(M_PI))
                    deltaYawRad -= static_cast<float>(2.0 * M_PI);
                while (deltaYawRad < static_cast<float>(-M_PI))
                    deltaYawRad += static_cast<float>(2.0 * M_PI);
            }
            const float deltaYawDeg = pcl::rad2deg(deltaYawRad);
            const bool suspiciousJump = hasLastLoggedPose &&
                ((dt > 1e-6 && speed > 8.0) || std::abs(deltaYawDeg) > 1.0f);
            const int matchUsed = currentConstraintMatchDiag.totalUsed();
            const int terrainAffected = currentConstraintMatchDiag.totalAffected();
            const double surfRawMean = currentConstraintMatchDiag.surf_used > 0 ?
                currentConstraintMatchDiag.surf_raw_abs_res_sum / currentConstraintMatchDiag.surf_used : 0.0;
            const double surfWeightedMean = currentConstraintMatchDiag.surf_used > 0 ?
                currentConstraintMatchDiag.surf_weighted_abs_res_sum / currentConstraintMatchDiag.surf_used : 0.0;
            const double cornerRawMean = currentConstraintMatchDiag.corner_used > 0 ?
                currentConstraintMatchDiag.corner_raw_abs_res_sum / currentConstraintMatchDiag.corner_used : 0.0;
            const double cornerWeightedMean = currentConstraintMatchDiag.corner_used > 0 ?
                currentConstraintMatchDiag.corner_weighted_abs_res_sum / currentConstraintMatchDiag.corner_used : 0.0;
            const double affectedRawMean = terrainAffected > 0 ?
                currentConstraintMatchDiag.affected_raw_abs_res_sum / terrainAffected : 0.0;
            const double affectedWeightedMean = terrainAffected > 0 ?
                currentConstraintMatchDiag.affected_weighted_abs_res_sum / terrainAffected : 0.0;
            const double affectedWeightMean = terrainAffected > 0 ?
                currentConstraintMatchDiag.affected_weight_sum / terrainAffected : 1.0;
            const double affectedRatio = matchUsed > 0 ?
                static_cast<double>(terrainAffected) / static_cast<double>(matchUsed) : 0.0;
            const char* status = "OK";
            if (!lastFrameEnoughFeatures)
                status = "FEATURE_WEAK";
            else if (isDegenerate)
                status = "DEGENERATE";
            else if (currentOptimizationRejected)
                status = "REVERSE_MOTION_REJECTED";
            else if (reverseMotionSuspected)
                status = "REVERSE_MOTION";
            else if (suspiciousJump)
                status = "SUSPICIOUS";

            ROS_WARN_STREAM(std::fixed << std::setprecision(3)
                << "[TCR-LM state] status=" << status
                << " t=" << timeLaserInfoCur
                << " dt=" << dt
                << " map_xyz=(" << transformTobeMapped[3] << "," << transformTobeMapped[4] << "," << transformTobeMapped[5] << ")"
                << " map_rpy_deg=(" << pcl::rad2deg(transformTobeMapped[0]) << ","
                << pcl::rad2deg(transformTobeMapped[1]) << ","
                << pcl::rad2deg(transformTobeMapped[2]) << ")"
                << " d_xyz=(" << dx << "," << dy << "," << dz << ")"
                << " d_dist=" << deltaDist
                << " speed=" << speed
                << " d_yaw_deg=" << deltaYawDeg
                << " motion_dir_cos=" << motionDirCos
                << " incr_xyz=(" << increX << "," << increY << "," << increZ << ")"
                << " incr_rpy_deg=(" << pcl::rad2deg(increRoll) << ","
                << pcl::rad2deg(increPitch) << ","
                << pcl::rad2deg(increYaw) << ")"
                << " edge=" << laserCloudCornerLastDSNum
                << " surf=" << laserCloudSurfLastDSNum
                << " corner_used_coeff=" << currentConstraintMatchDiag.corner_used
                << " surf_used_coeff=" << currentConstraintMatchDiag.surf_used
                << " match_used_coeff=" << matchUsed
                << " corner_raw_abs_res_mean=" << cornerRawMean
                << " corner_weighted_abs_res_mean=" << cornerWeightedMean
                << " surf_raw_abs_res_mean=" << surfRawMean
                << " surf_weighted_abs_res_mean=" << surfWeightedMean
                << " terrain_affected_coeff=" << terrainAffected
                << " terrain_affected_ratio=" << affectedRatio
                << " terrain_affected_weight_mean=" << affectedWeightMean
                << " terrain_affected_raw_abs_res_mean=" << affectedRawMean
                << " terrain_affected_weighted_abs_res_mean=" << affectedWeightedMean
                << " enough_features=" << (lastFrameEnoughFeatures ? "true" : "false")
                << " degenerate=" << (isDegenerate ? "true" : "false")
                << " reverse_motion_suspected=" << (reverseMotionSuspected ? "true" : "false")
                << " reverse_motion_rejected=" << (currentOptimizationRejected ? "true" : "false")
                << " suspicious_jump=" << (suspiciousJump ? "true" : "false"));
            hasLastLoggedPose = true;
            lastLoggedTime = timeLaserInfoCur;
            lastLoggedX = transformTobeMapped[3];
            lastLoggedY = transformTobeMapped[4];
            lastLoggedZ = transformTobeMapped[5];
            lastLoggedYaw = transformTobeMapped[2];
        }

        if (hasAcceptedPose)
        {
            const float acceptedDx = transformTobeMapped[3] - lastAcceptedX;
            const float acceptedDy = transformTobeMapped[4] - lastAcceptedY;
            const float acceptedDz = transformTobeMapped[5] - lastAcceptedZ;
            const float acceptedDist = std::sqrt(acceptedDx * acceptedDx + acceptedDy * acceptedDy + acceptedDz * acceptedDz);
            if (acceptedDist > reverseMotionGuardMinDelta)
            {
                hasAcceptedMotionDir = true;
                lastAcceptedMotionDirX = acceptedDx / acceptedDist;
                lastAcceptedMotionDirY = acceptedDy / acceptedDist;
                lastAcceptedMotionDirZ = acceptedDz / acceptedDist;
            }
        }
        hasAcceptedPose = true;
        lastAcceptedX = transformTobeMapped[3];
        lastAcceptedY = transformTobeMapped[4];
        lastAcceptedZ = transformTobeMapped[5];
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);

        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);

        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }

        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }

        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }

        static std::size_t lastSLAMInfoPubSize = std::numeric_limits<std::size_t>::max();
        if (pubSLAMInfo.getNumSubscribers() != 0)
        {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size())
            {
                tcrlm::cloud_info slamInfo;
                slamInfo.header.stamp = timeLaserInfoStamp;
                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                *cloudOut += *laserCloudCornerLastDS;
                *cloudOut += *laserCloudSurfLastDS;
                slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut, timeLaserInfoStamp, lidarFrame);
                slamInfo.key_frame_poses = publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp, odometryFrame);
                pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
                *localMapOut += *laserCloudCornerFromMapDS;
                *localMapOut += *laserCloudSurfFromMapDS;
                slamInfo.key_frame_map = publishCloud(ros::Publisher(), localMapOut, timeLaserInfoStamp, odometryFrame);
                pubSLAMInfo.publish(slamInfo);
                lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "tcrlm_mapping");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
