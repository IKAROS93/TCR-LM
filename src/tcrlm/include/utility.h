#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER, LIVOX };

class ParamServer
{
public:

    ros::NodeHandle nh;

    std::string robot_id;


    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string imuInitialGuessTopic;
    string gpsTopic;
    string gnssHeadingTopic;


    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;


    bool useImuHeadingInitialization;
    bool useGpsElevation;
    string gpsElevationAxis;
    bool enableGpsFactor;
    float gpsCovThreshold;
    float poseCovThreshold;
    float gpsFactorMaxTimeDiff;
    float gpsFactorMinDistance;
    float gpsFactorVarianceOverride;
    bool waitForGnssInitialization;
    float gnssInitializationMaxTimeDiff;
    float gnssInitializationRollDeg;
    float gnssInitializationPitchDeg;
    bool enableGnssHeadingFactor;
    string gnssHeadingFactorMode;
    float gnssHeadingFactorMaxTimeDiff;
    float gnssHeadingFactorStd;
    float gnssHeadingOffsetDeg;
    bool enableFeatureCloudLeveling;
    float featureCloudLevelingRollDeg;
    float featureCloudLevelingPitchDeg;
    float featureCloudLevelingYawDeg;


    bool savePCD;
    string savePCDDirectory;
    bool exportKeyframeDatasetOnSave;


    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;
    string projectionColumnMode;
    float projectionTimeScanPeriod;


    bool useImu;
    bool useImuOrientation;
    bool useImuDeskew;
    bool useImuRPYInFrontend;
    bool useImuInitialGuess;
    bool useLidarOdomDeskew;
    bool debugImuInitialGuess;
    bool debugImuDeskew;
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    float imuInitialGuessWeight;
    float imuInitialGuessMaxAge;
    float imuInitialGuessMaxPositionError;
    float imuInitialGuessMaxRotationError;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    vector<double> imuGyrScaleV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Vector3d imuGyrScale;
    Eigen::Quaterniond extQRPY;


    float edgeThreshold;
    float surfThreshold;
    int scanRingSegments;
    int maxEdgePerSegment;
    int linePlaneFittingKnn;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;


    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float terrainChangeInputLeafSize;

    float z_tollerance;
    float rotation_tollerance;


    int numberOfCores;
    double mappingProcessInterval;
    bool logLioStatePerFrame;


    float surroundingkeyframeAddingDistThreshold;
    float surroundingkeyframeAddingAngleThreshold;
    bool logKeyframeAddition;
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;


    bool enableTerrainChangeMaintenance;
    bool terrainChangeWriteBackLabels;
    bool terrainChangeApplyToMapping;
    string terrainChangeMappingMode;
    float terrainChangeWeightDropPerConfirmation;
    float terrainChangeMinFeatureWeight;
    string terrainChangeHeightAxis;
    float terrainChangeGridResolution;
    float terrainChangeDetectionInterval;
    int terrainChangeMinActiveKeyframes;
    int terrainChangeMinLayerPoints;
    float terrainChangeLayerBand;
    float terrainChangeMinLayerTimeSeparation;
    int terrainChangeMinCellPoints;
    float terrainChangeTauUp;
    float terrainChangeTauDown;
    float terrainChangeWriteBand;
    float terrainChangeRecentBand;
    int terrainChangeMinRegionCells;
    float terrainChangeMinRegionArea;
    int terrainChangeRequiredConfirmations;
    float terrainChangeMaxUpdateRatioPerKeyframe;
    float terrainChangeMaxFeatureUpdateRatioPerKeyframe;
    int terrainChangeMaxVisualizationKeyframes;
    bool terrainChangePublishCandidateClouds;
    bool terrainChangeLogSummary;


    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;


    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    ParamServer()
    {
        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh.param<std::string>("tcrlm/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("tcrlm/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("tcrlm/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>("tcrlm/imuInitialGuessTopic", imuInitialGuessTopic, "");
        nh.param<std::string>("tcrlm/gpsTopic", gpsTopic, "odometry/gps");
        nh.param<std::string>("tcrlm/gnssHeadingTopic", gnssHeadingTopic, "/gnss_heading");

        nh.param<std::string>("tcrlm/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("tcrlm/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("tcrlm/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("tcrlm/mapFrame", mapFrame, "map");

        nh.param<bool>("tcrlm/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>("tcrlm/useGpsElevation", useGpsElevation, false);
        nh.param<std::string>("tcrlm/gpsElevationAxis", gpsElevationAxis, "z");
        nh.param<bool>("tcrlm/enableGpsFactor", enableGpsFactor, true);
        nh.param<float>("tcrlm/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("tcrlm/poseCovThreshold", poseCovThreshold, 25.0);
        nh.param<float>("tcrlm/gpsFactorMaxTimeDiff", gpsFactorMaxTimeDiff, 0.2);
        nh.param<float>("tcrlm/gpsFactorMinDistance", gpsFactorMinDistance, 0.01);
        nh.param<float>("tcrlm/gpsFactorVarianceOverride", gpsFactorVarianceOverride, -1.0);
        nh.param<bool>("tcrlm/waitForGnssInitialization", waitForGnssInitialization, false);
        nh.param<float>("tcrlm/gnssInitializationMaxTimeDiff", gnssInitializationMaxTimeDiff, 0.2);
        nh.param<float>("tcrlm/gnssInitializationRollDeg", gnssInitializationRollDeg, 0.0);
        nh.param<float>("tcrlm/gnssInitializationPitchDeg", gnssInitializationPitchDeg, 0.0);
        nh.param<bool>("tcrlm/enableGnssHeadingFactor", enableGnssHeadingFactor, false);
        nh.param<std::string>("tcrlm/gnssHeadingFactorMode", gnssHeadingFactorMode, "relative");
        nh.param<float>("tcrlm/gnssHeadingFactorMaxTimeDiff", gnssHeadingFactorMaxTimeDiff, 0.2);
        nh.param<float>("tcrlm/gnssHeadingFactorStd", gnssHeadingFactorStd, 0.1);
        nh.param<float>("tcrlm/gnssHeadingOffsetDeg", gnssHeadingOffsetDeg, 0.0);
        nh.param<bool>("tcrlm/enableFeatureCloudLeveling", enableFeatureCloudLeveling, false);
        nh.param<float>("tcrlm/featureCloudLevelingRollDeg", featureCloudLevelingRollDeg, 0.0);
        nh.param<float>("tcrlm/featureCloudLevelingPitchDeg", featureCloudLevelingPitchDeg, 0.0);
        nh.param<float>("tcrlm/featureCloudLevelingYawDeg", featureCloudLevelingYawDeg, 0.0);

        nh.param<bool>("tcrlm/savePCD", savePCD, false);
        nh.param<std::string>("tcrlm/savePCDDirectory", savePCDDirectory, "/tmp/tcrlm_map/");
        nh.param<bool>("tcrlm/exportKeyframeDatasetOnSave", exportKeyframeDatasetOnSave, false);

        std::string sensorStr;
        nh.param<std::string>("tcrlm/sensor", sensorStr, "");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "livox")
        {
            sensor = SensorType::LIVOX;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Invalid sensor type (must be either 'velodyne' or 'ouster' or 'livox'): " << sensorStr);
            ros::shutdown();
        }

        nh.param<int>("tcrlm/N_SCAN", N_SCAN, 16);
        nh.param<int>("tcrlm/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<int>("tcrlm/downsampleRate", downsampleRate, 1);
        nh.param<float>("tcrlm/lidarMinRange", lidarMinRange, 1.0);
        nh.param<float>("tcrlm/lidarMaxRange", lidarMaxRange, 1000.0);
        nh.param<std::string>("tcrlm/projectionColumnMode", projectionColumnMode, "azimuth");
        nh.param<float>("tcrlm/projectionTimeScanPeriod", projectionTimeScanPeriod, 0.1);

        nh.param<bool>("tcrlm/useImu", useImu, true);
        nh.param<bool>("tcrlm/useImuOrientation", useImuOrientation, true);
        nh.param<bool>("tcrlm/useImuDeskew", useImuDeskew, false);
        nh.param<bool>("tcrlm/useImuRPYInFrontend", useImuRPYInFrontend, false);
        nh.param<bool>("tcrlm/useImuInitialGuess", useImuInitialGuess, false);
        nh.param<bool>("tcrlm/useLidarOdomDeskew", useLidarOdomDeskew, true);
        nh.param<bool>("tcrlm/debugImuInitialGuess", debugImuInitialGuess, false);
        nh.param<bool>("tcrlm/debugImuDeskew", debugImuDeskew, false);
        nh.param<float>("tcrlm/imuAccNoise", imuAccNoise, 0.01);
        nh.param<float>("tcrlm/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>("tcrlm/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>("tcrlm/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>("tcrlm/imuGravity", imuGravity, 9.80511);
        nh.param<float>("tcrlm/imuRPYWeight", imuRPYWeight, 0.01);
        nh.param<float>("tcrlm/imuInitialGuessWeight", imuInitialGuessWeight, 0.0);
        imuInitialGuessWeight = std::max(0.0f, std::min(1.0f, imuInitialGuessWeight));
        nh.param<float>("tcrlm/imuInitialGuessMaxAge", imuInitialGuessMaxAge, 0.20);
        nh.param<float>("tcrlm/imuInitialGuessMaxPositionError", imuInitialGuessMaxPositionError, 1.0);
        nh.param<float>("tcrlm/imuInitialGuessMaxRotationError", imuInitialGuessMaxRotationError, 0.35);
        imuInitialGuessMaxAge = std::max(0.0f, imuInitialGuessMaxAge);
        imuInitialGuessMaxPositionError = std::max(0.0f, imuInitialGuessMaxPositionError);
        imuInitialGuessMaxRotationError = std::max(0.0f, imuInitialGuessMaxRotationError);
        nh.param<vector<double>>("tcrlm/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>("tcrlm/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>("tcrlm/extrinsicTrans", extTransV, vector<double>());
        nh.param<vector<double>>("tcrlm/imuGyrScale", imuGyrScaleV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        imuGyrScale = Eigen::Vector3d::Ones();
        if (imuGyrScaleV.size() == 3)
            imuGyrScale = Eigen::Vector3d(imuGyrScaleV[0], imuGyrScaleV[1], imuGyrScaleV[2]);
        extQRPY = Eigen::Quaterniond(extRPY).inverse();

        nh.param<float>("tcrlm/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>("tcrlm/surfThreshold", surfThreshold, 0.1);
        nh.param<int>("tcrlm/scanRingSegments", scanRingSegments, 6);
        nh.param<int>("tcrlm/maxEdgePerSegment", maxEdgePerSegment, 20);
        nh.param<int>("tcrlm/linePlaneFittingKnn", linePlaneFittingKnn, 5);
        scanRingSegments = std::max(1, scanRingSegments);
        maxEdgePerSegment = std::max(1, maxEdgePerSegment);
        linePlaneFittingKnn = std::max(3, linePlaneFittingKnn);
        nh.param<int>("tcrlm/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>("tcrlm/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>("tcrlm/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>("tcrlm/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("tcrlm/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);
        nh.param<float>("tcrlm/terrainChangeInputLeafSize", terrainChangeInputLeafSize, 0.2);

        nh.param<float>("tcrlm/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>("tcrlm/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>("tcrlm/numberOfCores", numberOfCores, 2);
        nh.param<double>("tcrlm/mappingProcessInterval", mappingProcessInterval, 0.15);
        nh.param<bool>("tcrlm/logLioStatePerFrame", logLioStatePerFrame, false);

        nh.param<float>("tcrlm/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 0.1);
        nh.param<float>("tcrlm/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<bool>("tcrlm/logKeyframeAddition", logKeyframeAddition, false);
        nh.param<float>("tcrlm/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("tcrlm/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh.param<bool>("tcrlm/enableTerrainChangeMaintenance", enableTerrainChangeMaintenance, false);
        nh.param<bool>("tcrlm/terrainChangeWriteBackLabels", terrainChangeWriteBackLabels, false);
        nh.param<bool>("tcrlm/terrainChangeApplyToMapping", terrainChangeApplyToMapping, false);
        nh.param<std::string>("tcrlm/terrainChangeMappingMode", terrainChangeMappingMode, "downweight");
        nh.param<float>("tcrlm/terrainChangeWeightDropPerConfirmation", terrainChangeWeightDropPerConfirmation, 0.01);
        nh.param<float>("tcrlm/terrainChangeMinFeatureWeight", terrainChangeMinFeatureWeight, 0.2);
        nh.param<std::string>("tcrlm/terrainChangeHeightAxis", terrainChangeHeightAxis, "z");
        nh.param<float>("tcrlm/terrainChangeGridResolution", terrainChangeGridResolution, 1.0);
        nh.param<float>("tcrlm/terrainChangeDetectionInterval", terrainChangeDetectionInterval, 10.0);
        nh.param<int>("tcrlm/terrainChangeMinActiveKeyframes", terrainChangeMinActiveKeyframes, 5);
        nh.param<int>("tcrlm/terrainChangeMinLayerPoints", terrainChangeMinLayerPoints, 5);
        nh.param<float>("tcrlm/terrainChangeLayerBand", terrainChangeLayerBand, 0.15);
        nh.param<float>("tcrlm/terrainChangeMinLayerTimeSeparation", terrainChangeMinLayerTimeSeparation, 3.0);
        nh.param<int>("tcrlm/terrainChangeMinCellPoints", terrainChangeMinCellPoints, 4);
        nh.param<float>("tcrlm/terrainChangeTauUp", terrainChangeTauUp, 0.35);
        nh.param<float>("tcrlm/terrainChangeTauDown", terrainChangeTauDown, 0.35);
        nh.param<float>("tcrlm/terrainChangeWriteBand", terrainChangeWriteBand, 0.15);
        nh.param<float>("tcrlm/terrainChangeRecentBand", terrainChangeRecentBand, 0.25);
        nh.param<int>("tcrlm/terrainChangeMinRegionCells", terrainChangeMinRegionCells, 4);
        nh.param<float>("tcrlm/terrainChangeMinRegionArea", terrainChangeMinRegionArea, 1.0);
        nh.param<int>("tcrlm/terrainChangeRequiredConfirmations", terrainChangeRequiredConfirmations, 1);
        nh.param<float>("tcrlm/terrainChangeMaxUpdateRatioPerKeyframe", terrainChangeMaxUpdateRatioPerKeyframe, 0.4);
        nh.param<float>("tcrlm/terrainChangeMaxFeatureUpdateRatioPerKeyframe", terrainChangeMaxFeatureUpdateRatioPerKeyframe, 0.005);
        nh.param<int>("tcrlm/terrainChangeMaxVisualizationKeyframes", terrainChangeMaxVisualizationKeyframes, 300);
        nh.param<bool>("tcrlm/terrainChangePublishCandidateClouds", terrainChangePublishCandidateClouds, true);
        nh.param<bool>("tcrlm/terrainChangeLogSummary", terrainChangeLogSummary, false);

        nh.param<bool>("tcrlm/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("tcrlm/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("tcrlm/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("tcrlm/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("tcrlm/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("tcrlm/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("tcrlm/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>("tcrlm/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("tcrlm/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("tcrlm/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
    }

    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;

        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();

        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        gyr = gyr.cwiseProduct(imuGyrScale);
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();


        Eigen::Quaterniond q_final;
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        if (!useImuOrientation || q_from.norm() < 0.1)
        {
            static double yaw_from_gyro = 0.0;
            static double last_imu_time = -1.0;

            double imu_time = imu_in.header.stamp.toSec();
            if (last_imu_time > 0.0 && imu_time > last_imu_time)
            {
                yaw_from_gyro += gyr.z() * (imu_time - last_imu_time);
                yaw_from_gyro = atan2(sin(yaw_from_gyro), cos(yaw_from_gyro));
            }
            last_imu_time = imu_time;


            double roll = atan2(acc.y(), acc.z());
            double pitch = atan2(-acc.x(), sqrt(acc.y() * acc.y() + acc.z() * acc.z()));
            tf::Quaternion q_fallback = tf::createQuaternionFromRPY(roll, pitch, yaw_from_gyro);
            q_final = Eigen::Quaterniond(q_fallback.w(), q_fallback.x(), q_fallback.y(), q_fallback.z());

            if (!useImuOrientation)
                ROS_WARN_ONCE("IMU orientation usage is disabled, using acc+gyro fallback quaternion.");
            else
                ROS_WARN_ONCE("IMU orientation is invalid, using acc+gyro fallback quaternion.");
        }
        else
        {
            q_final = q_from * extQRPY;
        }
        if (!std::isfinite(q_final.x()) || !std::isfinite(q_final.y()) ||
            !std::isfinite(q_final.z()) || !std::isfinite(q_final.w()) ||
            q_final.norm() < 0.1)
        {
            ROS_ERROR("Invalid quaternion after fallback, shutting down.");
            ros::shutdown();
        }
        q_final.normalize();

        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        return imu_out;
    }
};

template<typename T>
sensor_msgs::PointCloud2 publishCloud(const ros::Publisher& thisPub, const T& thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub.getNumSubscribers() != 0)
        thisPub.publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

#endif
