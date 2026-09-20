#include "utility.h"
#include "tcrlm/cloud_info.h"

#include <cstdint>

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    std::uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    std::uint32_t t;
    std::uint16_t reflectivity;
    std::uint8_t ring;
    std::uint16_t noise;
    std::uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint32_t, t, t) (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring) (std::uint16_t, noise, noise) (std::uint32_t, range, range)
)

struct HesaiPointXYZIRTD
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    std::uint16_t ring;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(HesaiPointXYZIRTD,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint16_t, ring, ring) (double, timestamp, timestamp)
)


using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;

    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;
    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;
    ros::Subscriber subImuOdom;
    std::deque<nav_msgs::Odometry> imuOdomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<HesaiPointXYZIRTD>::Ptr tmpHesaiCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool imuDeskewAvailable;
    int imuPointerCur;
    double imuTime[queueLength];
    double imuRotX[queueLength];
    double imuRotY[queueLength];
    double imuRotZ[queueLength];

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;
    float odomIncreRoll;
    float odomIncrePitch;
    float odomIncreYaw;
    double odomIncreTime;

    tcrlm::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;
    bool useTimeColumnProjection;


public:
    ImageProjection():
    deskewFlag(0)
    {
        if (useImu && (useImuDeskew || useImuRPYInFrontend))
        {
            subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
            ROS_WARN_STREAM("IMU deskew frontend subscriber enabled: topic=" << imuTopic
                            << ", useImuDeskew=" << useImuDeskew
                            << ", useImuRPYInFrontend=" << useImuRPYInFrontend);
        }
        subOdom = nh.subscribe<nav_msgs::Odometry>("tcrlm/mapping/odometry", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        if (useImu && useImuInitialGuess)
        {
            const std::string guessTopic = imuInitialGuessTopic.empty() ? odomTopic+"_incremental" : imuInitialGuessTopic;
            subImuOdom = nh.subscribe<nav_msgs::Odometry>(guessTopic, 2000, &ImageProjection::imuOdometryHandler, this, ros::TransportHints().tcpNoDelay());
            ROS_WARN_STREAM("IMU initial guess blending enabled: topic=" << guessTopic
                            << ", weight=" << imuInitialGuessWeight);
        }
        if (debugImuInitialGuess)
        {
            const std::string guessTopic = imuInitialGuessTopic.empty() ? odomTopic+"_incremental" : imuInitialGuessTopic;
            ROS_WARN_STREAM("IMU initial guess diagnostics enabled: useImu=" << useImu
                            << ", useImuInitialGuess=" << useImuInitialGuess
                            << ", weight=" << imuInitialGuessWeight
                            << ", imu_odom_topic=" << guessTopic);
        }
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("tcrlm/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<tcrlm::cloud_info> ("tcrlm/deskew/cloud_info", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

        useTimeColumnProjection = (projectionColumnMode == "time" ||
                                   projectionColumnMode == "relative_time" ||
                                   projectionColumnMode == "timestamp");
        if (!useTimeColumnProjection && projectionColumnMode != "azimuth")
        {
            ROS_WARN_STREAM("Unknown tcrlm/projectionColumnMode='" << projectionColumnMode
                            << "', falling back to azimuth projection.");
        }
        ROS_INFO_STREAM("Image projection column mode: "
                        << (useTimeColumnProjection ? "time" : "azimuth")
                        << ", projectionTimeScanPeriod=" << projectionTimeScanPeriod);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        tmpHesaiCloudIn.reset(new pcl::PointCloud<HesaiPointXYZIRTD>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        firstPointFlag = true;
        imuDeskewAvailable = false;
        imuPointerCur = 0;
        odomDeskewFlag = false;
        odomIncreX = 0.0f;
        odomIncreY = 0.0f;
        odomIncreZ = 0.0f;
        odomIncreRoll = 0.0f;
        odomIncrePitch = 0.0f;
        odomIncreYaw = 0.0f;
        odomIncreTime = 0.0;

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock2(odoLock);
        imuQueue.push_back(thisImu);
        while (imuQueue.size() > queueLength)
            imuQueue.pop_front();
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        imuOdomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        if (!cachePointCloud(laserCloudMsg))
            return;

        if (!deskewInfo())
            return;

        projectPointCloud();

        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {

        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;


        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();

        bool hasTimeField = false;
        bool hasTField = false;
        bool hasTimestampField = false;
        for (const auto &field : currentCloudMsg.fields)
        {
            if (field.name == "time")
                hasTimeField = true;
            else if (field.name == "t")
                hasTField = true;
            else if (field.name == "timestamp")
                hasTimestampField = true;
        }

        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            if (hasTimestampField && !hasTimeField && !hasTField)
            {

                pcl::moveFromROSMsg(currentCloudMsg, *tmpHesaiCloudIn);
                laserCloudIn->points.resize(tmpHesaiCloudIn->size());
                laserCloudIn->is_dense = tmpHesaiCloudIn->is_dense;

                double firstStamp = tmpHesaiCloudIn->points.empty() ? 0.0 : tmpHesaiCloudIn->points.front().timestamp;
                for (size_t i = 0; i < tmpHesaiCloudIn->size(); ++i)
                {
                    const auto &src = tmpHesaiCloudIn->points[i];
                    auto &dst = laserCloudIn->points[i];
                    dst.x = src.x;
                    dst.y = src.y;
                    dst.z = src.z;
                    dst.intensity = src.intensity;
                    dst.ring = src.ring;
                    double relTime = src.timestamp - firstStamp;
                    if (relTime < 0.0)
                        relTime = 0.0;
                    dst.time = static_cast<float>(relTime);
                }
            }
            else
            {
                pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
            }
        }
        else if (sensor == SensorType::OUSTER)
        {

            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }


        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        double maxRelTime = 0.0;
        for (const auto &point : laserCloudIn->points)
        {
            if (std::isfinite(point.time))
                maxRelTime = std::max(maxRelTime, static_cast<double>(point.time));
        }
        timeScanEnd = timeScanCur + maxRelTime;


        if (laserCloudIn->is_dense == false)
        {
            const size_t before_size = laserCloudIn->size();
            std::vector<int> valid_indices;
            pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, valid_indices);
            laserCloudIn->is_dense = true;
            if (laserCloudIn->size() < before_size)
            {
                ROS_WARN_THROTTLE(2.0, "Input cloud is non-dense. Filtered invalid points: %zu -> %zu",
                                  before_size, laserCloudIn->size());
            }

            if (laserCloudIn->empty())
            {
                ROS_WARN_THROTTLE(2.0, "All points were invalid after filtering.");
                return false;
            }
        }


        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }


        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t" || field.name == "timestamp")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp field (time/t/timestamp) not available, deskew disabled and drift will increase.");
        }

        return true;
    }

    bool deskewInfo()
    {
        cloudInfo.imuAvailable = false;
        cloudInfo.imuRollInit = 0;
        cloudInfo.imuPitchInit = 0;
        cloudInfo.imuYawInit = 0;

        std::lock_guard<std::mutex> lock2(odoLock);

        imuDeskewInfo();
        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;
        cloudInfo.imuRollInit = 0;
        cloudInfo.imuPitchInit = 0;
        cloudInfo.imuYawInit = 0;
        imuDeskewAvailable = false;
        imuPointerCur = 0;

        if (!useImu || (!useImuDeskew && !useImuRPYInFrontend))
            return;

        if (imuQueue.empty())
        {
            if (debugImuDeskew)
                ROS_WARN_STREAM_THROTTLE(1.0, "[IMU deskew diag] no raw IMU messages. topic=" << imuTopic);
            return;
        }

        while (imuQueue.size() > 1 && ROS_TIME(&imuQueue[1]) < timeScanCur - 0.01)
            imuQueue.pop_front();

        if (imuQueue.empty())
            return;

        const double firstImuTime = ROS_TIME(&imuQueue.front());
        const double lastImuTime = ROS_TIME(&imuQueue.back());
        if (firstImuTime > timeScanCur)
        {
            if (debugImuDeskew)
            {
                ROS_WARN_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                    << "[IMU deskew diag] waiting for IMU before scan start, first_imu="
                    << firstImuTime << ", scan_start=" << timeScanCur);
            }
            return;
        }

        bool hasInitRPY = false;
        double initRoll = 0.0;
        double initPitch = 0.0;
        double initYaw = 0.0;
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            const double currentImuTime = ROS_TIME(&thisImuMsg);
            if (currentImuTime > timeScanCur)
                break;

            double roll, pitch, yaw;
            imuRPY2rosRPY(&thisImuMsg, &roll, &pitch, &yaw);
            if (std::isfinite(roll) && std::isfinite(pitch) && std::isfinite(yaw))
            {
                initRoll = roll;
                initPitch = pitch;
                initYaw = yaw;
                hasInitRPY = true;
            }
        }

        if (useImuRPYInFrontend && hasInitRPY)
        {
            cloudInfo.imuAvailable = true;
            cloudInfo.imuRollInit = initRoll;
            cloudInfo.imuPitchInit = initPitch;
            cloudInfo.imuYawInit = initYaw;
        }

        if (!useImuDeskew)
        {
            if (debugImuDeskew)
            {
                ROS_INFO_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                    << "[IMU deskew diag] deskew disabled, rpy_frontend="
                    << cloudInfo.imuAvailable << ", imu_queue=" << imuQueue.size());
            }
            return;
        }

        if (lastImuTime < timeScanEnd)
        {
            if (debugImuDeskew)
            {
                ROS_WARN_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                    << "[IMU deskew diag] insufficient IMU coverage, last_imu="
                    << lastImuTime << ", scan_end=" << timeScanEnd
                    << ", will_try_lidar_odom_deskew=1");
            }
            return;
        }

        int imuCount = 0;
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            const double currentImuTime = ROS_TIME(&thisImuMsg);
            if (currentImuTime > timeScanEnd + 0.01)
                break;
            if (imuCount >= queueLength)
                break;

            imuTime[imuCount] = currentImuTime;
            if (imuCount == 0)
            {
                imuRotX[0] = 0.0;
                imuRotY[0] = 0.0;
                imuRotZ[0] = 0.0;
            }
            else
            {
                double angularX, angularY, angularZ;
                imuAngular2rosAngular(&thisImuMsg, &angularX, &angularY, &angularZ);
                double timeDiff = currentImuTime - imuTime[imuCount - 1];
                if (!std::isfinite(timeDiff) || timeDiff < 0.0)
                    timeDiff = 0.0;

                imuRotX[imuCount] = imuRotX[imuCount - 1] + angularX * timeDiff;
                imuRotY[imuCount] = imuRotY[imuCount - 1] + angularY * timeDiff;
                imuRotZ[imuCount] = imuRotZ[imuCount - 1] + angularZ * timeDiff;
            }

            ++imuCount;
        }

        if (imuCount <= 1)
        {
            if (debugImuDeskew)
                ROS_WARN_STREAM_THROTTLE(1.0, "[IMU deskew diag] too few IMU samples for deskew: " << imuCount);
            return;
        }

        imuPointerCur = imuCount - 1;
        imuDeskewAvailable = true;

        if (debugImuDeskew)
        {
            constexpr double radToDeg = 57.29577951308232;
            ROS_INFO_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                << "[IMU deskew diag] available=1"
                << ", rpy_frontend=" << cloudInfo.imuAvailable
                << ", scan=(" << timeScanCur << " -> " << timeScanEnd << ")"
                << ", imu_span=(" << imuTime[0] << " -> " << imuTime[imuPointerCur] << ")"
                << ", imu_count=" << imuCount
                << ", init_rpy_deg=(" << initRoll * radToDeg
                << ", " << initPitch * radToDeg
                << ", " << initYaw * radToDeg << ")"
                << ", integ_rpy_deg=(" << imuRotX[imuPointerCur] * radToDeg
                << ", " << imuRotY[imuPointerCur] * radToDeg
                << ", " << imuRotZ[imuPointerCur] * radToDeg << ")");
        }
    }

    bool findLatestOdomBefore(const std::deque<nav_msgs::Odometry>& queue, double stamp,
                              nav_msgs::Odometry& odom, int& index) const
    {
        index = -1;
        for (int i = 0; i < (int)queue.size(); ++i)
        {
            if (queue[i].header.stamp.toSec() <= stamp)
                index = i;
            else
                break;
        }

        if (index < 0)
            return false;

        odom = queue[index];
        return true;
    }

    bool normalizedOdomQuaternion(const nav_msgs::Odometry& odom, tf::Quaternion& quat) const
    {
        tf::quaternionMsgToTF(odom.pose.pose.orientation, quat);
        if (!std::isfinite(quat.x()) || !std::isfinite(quat.y()) ||
            !std::isfinite(quat.z()) || !std::isfinite(quat.w()) ||
            quat.length2() <= 1e-12)
            return false;

        quat.normalize();
        return true;
    }

    bool setInitialGuessFromOdom(const nav_msgs::Odometry& odom)
    {
        tf::Quaternion orientation;
        if (!normalizedOdomQuaternion(odom, orientation))
            return false;

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        cloudInfo.initialGuessX = odom.pose.pose.position.x;
        cloudInfo.initialGuessY = odom.pose.pose.position.y;
        cloudInfo.initialGuessZ = odom.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;
        return true;
    }

    bool setBlendedInitialGuess(const nav_msgs::Odometry& lidarOdom,
                                const nav_msgs::Odometry& imuOdom,
                                float imuWeight)
    {
        tf::Quaternion lidarQuat;
        tf::Quaternion imuQuat;
        if (!normalizedOdomQuaternion(lidarOdom, lidarQuat) ||
            !normalizedOdomQuaternion(imuOdom, imuQuat))
            return false;

        const double dx = imuOdom.pose.pose.position.x - lidarOdom.pose.pose.position.x;
        const double dy = imuOdom.pose.pose.position.y - lidarOdom.pose.pose.position.y;
        const double dz = imuOdom.pose.pose.position.z - lidarOdom.pose.pose.position.z;
        const double positionError = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (imuInitialGuessMaxPositionError > 0.0f &&
            positionError > imuInitialGuessMaxPositionError)
        {
            ROS_WARN_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                << "[IMU initial guess diag] reject IMU blend by position gate."
                << " err=" << positionError
                << ", max=" << imuInitialGuessMaxPositionError);
            return false;
        }

        double quatDot = lidarQuat.x() * imuQuat.x() + lidarQuat.y() * imuQuat.y() +
                         lidarQuat.z() * imuQuat.z() + lidarQuat.w() * imuQuat.w();
        quatDot = std::fabs(quatDot);
        if (quatDot > 1.0)
            quatDot = 1.0;
        const double rotationError = 2.0 * std::acos(quatDot);
        if (imuInitialGuessMaxRotationError > 0.0f &&
            rotationError > imuInitialGuessMaxRotationError)
        {
            constexpr double radToDeg = 57.29577951308232;
            ROS_WARN_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                << "[IMU initial guess diag] reject IMU blend by rotation gate."
                << " err_deg=" << rotationError * radToDeg
                << ", max_deg=" << imuInitialGuessMaxRotationError * radToDeg);
            return false;
        }

        const double lidarWeight = 1.0 - imuWeight;
        cloudInfo.initialGuessX = lidarWeight * lidarOdom.pose.pose.position.x + imuWeight * imuOdom.pose.pose.position.x;
        cloudInfo.initialGuessY = lidarWeight * lidarOdom.pose.pose.position.y + imuWeight * imuOdom.pose.pose.position.y;
        cloudInfo.initialGuessZ = lidarWeight * lidarOdom.pose.pose.position.z + imuWeight * imuOdom.pose.pose.position.z;

        tf::Quaternion blendedQuat = lidarQuat.slerp(imuQuat, imuWeight);
        blendedQuat.normalize();
        double roll, pitch, yaw;
        tf::Matrix3x3(blendedQuat).getRPY(roll, pitch, yaw);
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;
        return true;
    }

    string odomDebugString(const nav_msgs::Odometry& odom, double referenceTime) const
    {
        tf::Quaternion orientation;
        if (!normalizedOdomQuaternion(odom, orientation))
            return "invalid_quaternion";

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        const double odomTime = odom.header.stamp.toSec();
        const double age = referenceTime - odomTime;
        constexpr double radToDeg = 57.29577951308232;

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3)
           << "t=" << odomTime
           << ", age=" << age << "s"
           << ", p=(" << odom.pose.pose.position.x
           << ", " << odom.pose.pose.position.y
           << ", " << odom.pose.pose.position.z << ")"
           << ", rpy_deg=(" << roll * radToDeg
           << ", " << pitch * radToDeg
           << ", " << yaw * radToDeg << ")";
        return ss.str();
    }

    void logInitialGuessDiagnostics(const string& source,
                                    bool hasLidarOdom,
                                    const nav_msgs::Odometry& latestOdomMsg,
                                    int latestOdomIdx,
                                    bool hasImuOdom,
                                    const nav_msgs::Odometry& latestImuOdomMsg,
                                    int latestImuOdomIdx)
    {
        if (!debugImuInitialGuess)
            return;

        constexpr double radToDeg = 57.29577951308232;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3)
           << "[IMU initial guess diag] source=" << source
           << ", odomAvailable=" << cloudInfo.odomAvailable
           << ", useImu=" << useImu
           << ", useImuInitialGuess=" << useImuInitialGuess
           << ", weight=" << imuInitialGuessWeight
           << ", scan=(" << timeScanCur << " -> " << timeScanEnd << ")"
           << ", lidar_queue=" << odomQueue.size()
           << ", imu_queue=" << imuOdomQueue.size();

        if (hasLidarOdom)
            ss << ", lidar_idx=" << latestOdomIdx << ", lidar={" << odomDebugString(latestOdomMsg, timeScanCur) << "}";
        else
            ss << ", lidar=none";

        if (hasImuOdom)
            ss << ", imu_idx=" << latestImuOdomIdx << ", imu={" << odomDebugString(latestImuOdomMsg, timeScanCur) << "}";
        else
            ss << ", imu=none";

        ss << ", guess_p=(" << cloudInfo.initialGuessX
           << ", " << cloudInfo.initialGuessY
           << ", " << cloudInfo.initialGuessZ << ")"
           << ", guess_rpy_deg=(" << cloudInfo.initialGuessRoll * radToDeg
           << ", " << cloudInfo.initialGuessPitch * radToDeg
           << ", " << cloudInfo.initialGuessYaw * radToDeg << ")";

        ROS_INFO_STREAM_THROTTLE(1.0, ss.str());
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;
        odomDeskewFlag = false;
        odomIncreX = 0.0f;
        odomIncreY = 0.0f;
        odomIncreZ = 0.0f;
        odomIncreRoll = 0.0f;
        odomIncrePitch = 0.0f;
        odomIncreYaw = 0.0f;
        odomIncreTime = 0.0;

        int latestOdomIdx = -1;
        nav_msgs::Odometry latestOdomMsg;
        const bool hasLidarOdom = findLatestOdomBefore(odomQueue, timeScanCur, latestOdomMsg, latestOdomIdx);

        string initialGuessSource = "none";
        if (hasLidarOdom && setInitialGuessFromOdom(latestOdomMsg))
        {
            cloudInfo.odomAvailable = true;
            initialGuessSource = "lidar";
        }
        else if (hasLidarOdom)
        {
            initialGuessSource = "lidar_invalid_quaternion";
        }

        int latestImuOdomIdx = -1;
        nav_msgs::Odometry latestImuOdomMsg;
        bool hasImuOdom = false;
        if (useImu && useImuInitialGuess && imuInitialGuessWeight > 0.0f)
        {
            hasImuOdom = findLatestOdomBefore(imuOdomQueue, timeScanCur, latestImuOdomMsg, latestImuOdomIdx);

            if (hasImuOdom)
            {
                bool initialGuessSet = false;
                const double imuGuessAge = timeScanCur - latestImuOdomMsg.header.stamp.toSec();
                if (imuInitialGuessMaxAge > 0.0f && imuGuessAge > imuInitialGuessMaxAge)
                {
                    initialGuessSource = "imu_stale";
                    ROS_WARN_STREAM_THROTTLE(1.0, std::fixed << std::setprecision(3)
                        << "[IMU initial guess diag] reject stale IMU odom."
                        << " age=" << imuGuessAge
                        << ", max_age=" << imuInitialGuessMaxAge);
                }
                else if (hasLidarOdom && imuInitialGuessWeight < 1.0f)
                {
                    initialGuessSet = setBlendedInitialGuess(latestOdomMsg, latestImuOdomMsg, imuInitialGuessWeight);
                    initialGuessSource = initialGuessSet ? "lidar_imu_blend" : "lidar_imu_blend_rejected";
                }
                else
                {
                    initialGuessSet = setInitialGuessFromOdom(latestImuOdomMsg);
                    initialGuessSource = initialGuessSet ? "imu" : "imu_invalid_quaternion";
                }

                if (initialGuessSet)
                {
                    cloudInfo.odomAvailable = true;
                }

                while (imuOdomQueue.size() > 1 && latestImuOdomIdx > 0)
                {
                    imuOdomQueue.pop_front();
                    --latestImuOdomIdx;
                }
            }
            else
            {
                initialGuessSource += "_no_imu_odom";
            }
        }

        logInitialGuessDiagnostics(initialGuessSource,
                                   hasLidarOdom, latestOdomMsg, latestOdomIdx,
                                   hasImuOdom, latestImuOdomMsg, latestImuOdomIdx);

        if (!useLidarOdomDeskew)
            return;


        if (!hasLidarOdom || latestOdomIdx < 1)
            return;

        nav_msgs::Odometry prevOdomMsg = odomQueue[latestOdomIdx - 1];
        const double prevOdomTime = prevOdomMsg.header.stamp.toSec();
        const double latestOdomTime = latestOdomMsg.header.stamp.toSec();
        odomIncreTime = latestOdomTime - prevOdomTime;
        if (!std::isfinite(odomIncreTime) || odomIncreTime <= 1e-3)
            return;

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(prevOdomMsg.pose.pose.orientation, orientation);
        double rollPrev, pitchPrev, yawPrev;
        tf::Matrix3x3(orientation).getRPY(rollPrev, pitchPrev, yawPrev);
        Eigen::Affine3f transBegin = pcl::getTransformation(prevOdomMsg.pose.pose.position.x,
                                                            prevOdomMsg.pose.pose.position.y,
                                                            prevOdomMsg.pose.pose.position.z,
                                                            rollPrev, pitchPrev, yawPrev);

        tf::quaternionMsgToTF(latestOdomMsg.pose.pose.orientation, orientation);
        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(latestOdomMsg.pose.pose.position.x,
                                                          latestOdomMsg.pose.pose.position.y,
                                                          latestOdomMsg.pose.pose.position.z,
                                                          roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ,
                                          odomIncreRoll, odomIncrePitch, odomIncreYaw);

        odomDeskewFlag = true;

        while (odomQueue.size() > 2 && latestOdomIdx > 1)
        {
            odomQueue.pop_front();
            --latestOdomIdx;
        }
    }

    void findRotation(double relTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        if (imuDeskewAvailable)
        {
            const double pointTime = timeScanCur + relTime;

            if (pointTime <= imuTime[0])
            {
                *rotXCur = imuRotX[0];
                *rotYCur = imuRotY[0];
                *rotZCur = imuRotZ[0];
                return;
            }

            if (pointTime >= imuTime[imuPointerCur])
            {
                *rotXCur = imuRotX[imuPointerCur];
                *rotYCur = imuRotY[imuPointerCur];
                *rotZCur = imuRotZ[imuPointerCur];
                return;
            }

            int imuPointerFront = 1;
            while (imuPointerFront <= imuPointerCur && pointTime > imuTime[imuPointerFront])
                ++imuPointerFront;

            const int imuPointerBack = imuPointerFront - 1;
            const double timeDiff = imuTime[imuPointerFront] - imuTime[imuPointerBack];
            const double ratioFront = timeDiff <= 0.0 ? 0.0 : (pointTime - imuTime[imuPointerBack]) / timeDiff;
            const double ratioBack = 1.0 - ratioFront;

            *rotXCur = static_cast<float>(imuRotX[imuPointerBack] * ratioBack + imuRotX[imuPointerFront] * ratioFront);
            *rotYCur = static_cast<float>(imuRotY[imuPointerBack] * ratioBack + imuRotY[imuPointerFront] * ratioFront);
            *rotZCur = static_cast<float>(imuRotZ[imuPointerBack] * ratioBack + imuRotZ[imuPointerFront] * ratioFront);
            return;
        }

        if (odomDeskewFlag == false || odomIncreTime <= 0.0)
            return;

        const float ratio = static_cast<float>(relTime / odomIncreTime);
        *rotXCur = ratio * odomIncreRoll;
        *rotYCur = ratio * odomIncrePitch;
        *rotZCur = ratio * odomIncreYaw;
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        if (odomDeskewFlag == false || odomIncreTime <= 0.0)
            return;

        const float ratio = static_cast<float>(relTime / odomIncreTime);

        *posXCur = ratio * odomIncreX;
        *posYCur = ratio * odomIncreY;
        *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || (imuDeskewAvailable == false && odomDeskewFlag == false))
            return *point;

        float rotXCur, rotYCur, rotZCur;
        findRotation(relTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }


        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();

        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;
            if (useTimeColumnProjection)
            {
                const float scanPeriod = projectionTimeScanPeriod > 0.0f
                    ? projectionTimeScanPeriod
                    : std::max(1e-3, timeScanEnd - timeScanCur);
                float relTime = laserCloudIn->points[i].time;
                if (!std::isfinite(relTime))
                    relTime = 0.0f;
                relTime = std::max(0.0f, std::min(relTime, scanPeriod - 1e-6f));
                columnIdn = static_cast<int>(std::floor(relTime / scanPeriod * Horizon_SCAN));
            }
            else if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    void cloudExtraction()
    {
        int count = 0;

        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {

                    cloudInfo.pointColInd[count] = j;

                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);

                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);

                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }

    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tcrlm_range_image_projection_and_deskew");

    ImageProjection IP;

    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();

    return 0;
}
