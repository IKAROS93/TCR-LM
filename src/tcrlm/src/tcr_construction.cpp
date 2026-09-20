#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

class TcrConstructionNode
{
public:
    TcrConstructionNode()
    {
        ros::NodeHandle pnh("~");
        pnh.param<std::string>("input_topic", input_topic_, "/lidar_points");
        pnh.param<std::string>("output_topic", output_topic_, "/tcrlm/preprocess/cloud_filter");
        pnh.param<bool>("publish_labeled_mask", publish_labeled_mask_, false);
        pnh.param<std::string>("labeled_mask_topic", labeled_mask_topic_, "/tcrlm/preprocess/cloud_filter_mask");
        pnh.param<bool>("drop_nan", drop_nan_, true);
        pnh.param<int>("input_queue_size", input_queue_size_, 100);
        pnh.param<bool>("log_timing", log_timing_, false);
        pnh.param<std::string>("height_axis", height_axis_name_, std::string("y"));
        pnh.param<double>("output_rotation_roll_deg", output_rotation_roll_deg_, 0.0);
        pnh.param<double>("output_rotation_pitch_deg", output_rotation_pitch_deg_, 0.0);
        pnh.param<double>("output_rotation_yaw_deg", output_rotation_yaw_deg_, 0.0);
        pnh.param<std::string>("output_frame_id", output_frame_id_, std::string(""));

        pnh.param<double>("intensity_threshold", intensity_threshold_, 0.0);
        pnh.param<double>("cloth_resolution_m", cloth_resolution_m_, 1.0);
        pnh.param<double>("time_step", time_step_, 0.1);
        pnh.param<int>("rigidness", rigidness_, 2);
        pnh.param<bool>("steep_slope_fit", steep_slope_fit_, false);
        pnh.param<double>("classification_threshold_m", classification_threshold_m_, 0.5);
        pnh.param<double>("postprocess_height_threshold_m", postprocess_height_threshold_m_, 0.3);
        pnh.param<double>("cloth_buffer_m", cloth_buffer_m_, 2.0);
        pnh.param<int>("max_iterations", max_iterations_, 100);
        pnh.param<double>("convergence_eps_m", convergence_eps_m_, 0.01);
        pnh.param<double>("nonground_dilation_m", nonground_dilation_m_, 0.0);
        pnh.param<std::string>("nonground_dilation_mode", nonground_dilation_mode_, std::string("grid_2d"));
        pnh.param<double>("nonground_seed_grid_resolution_m", nonground_seed_grid_resolution_m_, 0.5);
        pnh.param<int>("nonground_dilation_min_seed_points", nonground_dilation_min_seed_points_, 50);
        pnh.param<int>("nonground_dilation_min_seed_cells", nonground_dilation_min_seed_cells_, 10);
        pnh.param<double>("candidate_voxel_resolution_m", candidate_voxel_resolution_m_, 0.5);
        pnh.param<bool>("export_labeled_txt", export_labeled_txt_, false);
        pnh.param<std::string>("export_labeled_txt_dir", export_labeled_txt_dir_, std::string(""));

        if (export_labeled_txt_)
        {
            initializeExportDirectory();
        }
        configureHeightAxis();
        configureOutputRotation();

        sub_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>(
            input_topic_, input_queue_size_, &TcrConstructionNode::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 20);
        if (publish_labeled_mask_)
        {
            pub_labeled_mask_ = nh_.advertise<sensor_msgs::PointCloud2>(labeled_mask_topic_, 5);
        }

        ROS_INFO_STREAM("==== TCR CONSTRUCTION MODE: CSF ====");
        ROS_INFO_STREAM("tcrlm_tcr_construction: " << input_topic_ << " -> " << output_topic_
                        << ", intensity_threshold=" << intensity_threshold_
                        << ", cloth_resolution_m=" << cloth_resolution_m_
                        << ", time_step=" << time_step_
                        << ", rigidness=" << rigidness_
                        << ", height_axis=" << height_axis_name_
                        << ", output_rotation_rpy_deg=(" << output_rotation_roll_deg_
                        << ", " << output_rotation_pitch_deg_
                        << ", " << output_rotation_yaw_deg_ << ")"
                        << ", classification_threshold_m=" << classification_threshold_m_
                        << ", nonground_dilation_m=" << nonground_dilation_m_
                        << ", nonground_dilation_mode=" << nonground_dilation_mode_
                        << ", candidate_voxel_resolution_m=" << candidate_voxel_resolution_m_
                        << ", export_labeled_txt=" << export_labeled_txt_
                        << ", export_labeled_txt_dir=" << export_labeled_txt_dir_);
    }

private:
    struct ParsedPoint
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float intensity = 0.0f;
        std::size_t source_index = 0;
    };

    enum class HeightAxis
    {
        X,
        Y,
        Z,
        NEG_X,
        NEG_Y,
        NEG_Z
    };

    struct CloudFieldLayout
    {
        int x_offset = -1;
        int y_offset = -1;
        int z_offset = -1;
        int intensity_offset = -1;
        int x_type = -1;
        int y_type = -1;
        int z_type = -1;
        int intensity_type = -1;
        int ring_offset = -1;
        int ring_type = -1;
        bool has_xyz = false;
        bool has_intensity = false;
        bool has_ring = false;
    };

    struct GridData
    {
        Eigen::Vector2f origin_xz = Eigen::Vector2f::Zero();
        int width = 0;
        int height = 0;
        std::vector<float> grid_x;
        std::vector<float> grid_z;
    };

    static bool findField(const sensor_msgs::PointCloud2 &msg, const std::string &field_name, int &offset, int &datatype)
    {
        for (const auto &field : msg.fields)
        {
            if (field.name == field_name)
            {
                offset = static_cast<int>(field.offset);
                datatype = static_cast<int>(field.datatype);
                return true;
            }
        }
        return false;
    }

    static bool readAsDouble(const uint8_t *ptr, int datatype, double &out)
    {
        switch (datatype)
        {
        case sensor_msgs::PointField::INT8:
        {
            int8_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            out = static_cast<double>(v);
            return true;
        }
        case sensor_msgs::PointField::UINT8:
        {
            uint8_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            out = static_cast<double>(v);
            return true;
        }
        case sensor_msgs::PointField::INT16:
        {
            int16_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            out = static_cast<double>(v);
            return true;
        }
        case sensor_msgs::PointField::UINT16:
        {
            uint16_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            out = static_cast<double>(v);
            return true;
        }
        case sensor_msgs::PointField::INT32:
        {
            int32_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            out = static_cast<double>(v);
            return true;
        }
        case sensor_msgs::PointField::UINT32:
        {
            uint32_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            out = static_cast<double>(v);
            return true;
        }
        case sensor_msgs::PointField::FLOAT32:
        {
            float v = 0.0f;
            std::memcpy(&v, ptr, sizeof(v));
            out = static_cast<double>(v);
            return std::isfinite(v);
        }
        case sensor_msgs::PointField::FLOAT64:
        {
            double v = 0.0;
            std::memcpy(&v, ptr, sizeof(v));
            out = v;
            return std::isfinite(v);
        }
        default:
            return false;
        }
    }

    static bool writeFromDouble(uint8_t *ptr, int datatype, double value)
    {
        switch (datatype)
        {
        case sensor_msgs::PointField::FLOAT32:
        {
            const float v = static_cast<float>(value);
            std::memcpy(ptr, &v, sizeof(v));
            return true;
        }
        case sensor_msgs::PointField::FLOAT64:
        {
            const double v = value;
            std::memcpy(ptr, &v, sizeof(v));
            return true;
        }
        default:
            return false;
        }
    }

    static CloudFieldLayout buildFieldLayout(const sensor_msgs::PointCloud2 &msg)
    {
        CloudFieldLayout layout;
        layout.has_xyz = findField(msg, "x", layout.x_offset, layout.x_type) &&
                         findField(msg, "y", layout.y_offset, layout.y_type) &&
                         findField(msg, "z", layout.z_offset, layout.z_type);
        layout.has_intensity = findField(msg, "intensity", layout.intensity_offset, layout.intensity_type);
        layout.has_ring = findField(msg, "ring", layout.ring_offset, layout.ring_type);
        return layout;
    }

    static bool ensureDirectoryExists(const std::string &dir_path)
    {
        if (dir_path.empty())
        {
            return false;
        }

        std::size_t pos = 0;
        while (pos < dir_path.size() && dir_path[pos] == '/')
        {
            ++pos;
        }

        std::string current = dir_path[0] == '/' ? "/" : "";
        while (pos <= dir_path.size())
        {
            const std::size_t next_sep = dir_path.find('/', pos);
            const std::string part = dir_path.substr(pos, next_sep == std::string::npos ? std::string::npos : next_sep - pos);
            if (!part.empty())
            {
                if (!current.empty() && current.back() != '/')
                {
                    current += "/";
                }
                current += part;

                struct stat st;
                if (stat(current.c_str(), &st) != 0)
                {
                    if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
                    {
                        return false;
                    }
                }
                else if (!S_ISDIR(st.st_mode))
                {
                    return false;
                }
            }

            if (next_sep == std::string::npos)
            {
                break;
            }
            pos = next_sep + 1;
        }

        return true;
    }

    void initializeExportDirectory()
    {
        if (export_labeled_txt_dir_.empty())
        {
            export_labeled_txt_dir_ = "/tmp/tcrlm_tcr_labels";
        }

        if (!ensureDirectoryExists(export_labeled_txt_dir_))
        {
            throw std::runtime_error("Failed to create labeled txt export directory: " + export_labeled_txt_dir_);
        }
    }

    void exportLabeledTxt(const sensor_msgs::PointCloud2 &msg,
                          const CloudFieldLayout &layout,
                          const std::vector<char> &keep_original) const
    {
        if (!export_labeled_txt_)
        {
            return;
        }

        const std::size_t point_count = static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
        std::ostringstream path_builder;
        path_builder << export_labeled_txt_dir_ << "/frame_" << std::setw(6) << std::setfill('0') << export_frame_index_++ << ".txt";
        const std::string file_path = path_builder.str();

        std::ofstream out(file_path.c_str());
        if (!out.is_open())
        {
            ROS_ERROR_STREAM_THROTTLE(2.0, "tcrlm_tcr_construction failed to open labeled txt file: " << file_path);
            return;
        }

        out << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < point_count; ++i)
        {
            const uint8_t *src = msg.data.data() + i * msg.point_step;
            double x = 0.0, y = 0.0, z = 0.0;
            if (!readAsDouble(src + layout.x_offset, layout.x_type, x) ||
                !readAsDouble(src + layout.y_offset, layout.y_type, y) ||
                !readAsDouble(src + layout.z_offset, layout.z_type, z))
            {
                continue;
            }

            int ring = -1;
            if (layout.has_ring)
            {
                double ring_value = 0.0;
                if (readAsDouble(src + layout.ring_offset, layout.ring_type, ring_value))
                {
                    ring = static_cast<int>(std::llround(ring_value));
                }
            }

            const int label = keep_original[i] ? 0 : 1;
            out << x << ' ' << y << ' ' << z << ' ' << ring << ' ' << label << '\n';
        }
    }

    static sensor_msgs::PointCloud2 makeEmptyOutput(const sensor_msgs::PointCloud2 &msg, bool is_dense)
    {
        sensor_msgs::PointCloud2 out = msg;
        out.height = 1;
        out.width = 0;
        out.row_step = 0;
        out.data.clear();
        out.is_dense = is_dense;
        return out;
    }

    static float computeQuantileSorted(const std::vector<float> &sorted_values, double q)
    {
        if (sorted_values.empty())
        {
            return std::numeric_limits<float>::quiet_NaN();
        }
        const double clamped_q = std::max(0.0, std::min(1.0, q));
        const double pos = clamped_q * static_cast<double>(sorted_values.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
        if (lo == hi)
        {
            return sorted_values[lo];
        }
        const double alpha = pos - static_cast<double>(lo);
        return static_cast<float>((1.0 - alpha) * static_cast<double>(sorted_values[lo]) +
                                  alpha * static_cast<double>(sorted_values[hi]));
    }

    static std::uint64_t packCellKey(int u, int v)
    {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)) << 32) |
               static_cast<std::uint32_t>(u);
    }

    void configureHeightAxis()
    {
        std::string axis = height_axis_name_;
        std::transform(axis.begin(), axis.end(), axis.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (axis == "x")
            height_axis_ = HeightAxis::X;
        else if (axis == "y")
            height_axis_ = HeightAxis::Y;
        else if (axis == "z")
            height_axis_ = HeightAxis::Z;
        else if (axis == "-x")
            height_axis_ = HeightAxis::NEG_X;
        else if (axis == "-y")
            height_axis_ = HeightAxis::NEG_Y;
        else if (axis == "-z")
            height_axis_ = HeightAxis::NEG_Z;
        else
        {
            ROS_WARN_STREAM("Unsupported CSF height_axis '" << height_axis_name_ << "', falling back to 'y'.");
            height_axis_name_ = "y";
            height_axis_ = HeightAxis::Y;
        }
    }

    void configureOutputRotation()
    {
        const double deg_to_rad = M_PI / 180.0;
        const double roll = output_rotation_roll_deg_ * deg_to_rad;
        const double pitch = output_rotation_pitch_deg_ * deg_to_rad;
        const double yaw = output_rotation_yaw_deg_ * deg_to_rad;

        const double cr = std::cos(roll);
        const double sr = std::sin(roll);
        const double cp = std::cos(pitch);
        const double sp = std::sin(pitch);
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);

        output_rotation_[0][0] = cy * cp;
        output_rotation_[0][1] = cy * sp * sr - sy * cr;
        output_rotation_[0][2] = cy * sp * cr + sy * sr;
        output_rotation_[1][0] = sy * cp;
        output_rotation_[1][1] = sy * sp * sr + cy * cr;
        output_rotation_[1][2] = sy * sp * cr - cy * sr;
        output_rotation_[2][0] = -sp;
        output_rotation_[2][1] = cp * sr;
        output_rotation_[2][2] = cp * cr;

        output_rotation_enabled_ =
            std::fabs(output_rotation_roll_deg_) > 1e-9 ||
            std::fabs(output_rotation_pitch_deg_) > 1e-9 ||
            std::fabs(output_rotation_yaw_deg_) > 1e-9;
    }

    void projectToCsfFrame(double raw_x, double raw_y, double raw_z, float &grid_x, float &height, float &grid_z) const
    {
        switch (height_axis_)
        {
        case HeightAxis::X:
            grid_x = static_cast<float>(raw_y);
            height = static_cast<float>(raw_x);
            grid_z = static_cast<float>(raw_z);
            break;
        case HeightAxis::NEG_X:
            grid_x = static_cast<float>(raw_y);
            height = static_cast<float>(-raw_x);
            grid_z = static_cast<float>(raw_z);
            break;
        case HeightAxis::Y:
            grid_x = static_cast<float>(raw_x);
            height = static_cast<float>(raw_y);
            grid_z = static_cast<float>(raw_z);
            break;
        case HeightAxis::NEG_Y:
            grid_x = static_cast<float>(raw_x);
            height = static_cast<float>(-raw_y);
            grid_z = static_cast<float>(raw_z);
            break;
        case HeightAxis::Z:
            grid_x = static_cast<float>(raw_x);
            height = static_cast<float>(raw_z);
            grid_z = static_cast<float>(raw_y);
            break;
        case HeightAxis::NEG_Z:
            grid_x = static_cast<float>(raw_x);
            height = static_cast<float>(-raw_z);
            grid_z = static_cast<float>(raw_y);
            break;
        }
    }

    std::vector<ParsedPoint> extractPoints(const sensor_msgs::PointCloud2 &msg, const CloudFieldLayout &layout) const
    {
        std::vector<ParsedPoint> points;
        const std::size_t point_count = static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
        points.reserve(point_count);

        for (std::size_t i = 0; i < point_count; ++i)
        {
            const uint8_t *src = msg.data.data() + i * msg.point_step;
            double x = 0.0, y = 0.0, z = 0.0, intensity = 0.0;
            const bool ok_x = readAsDouble(src + layout.x_offset, layout.x_type, x);
            const bool ok_y = readAsDouble(src + layout.y_offset, layout.y_type, y);
            const bool ok_z = readAsDouble(src + layout.z_offset, layout.z_type, z);
            const bool ok_intensity = readAsDouble(src + layout.intensity_offset, layout.intensity_type, intensity);
            if (!ok_x || !ok_y || !ok_z || !ok_intensity)
            {
                continue;
            }
            if (drop_nan_ && (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)))
            {
                continue;
            }
            if (intensity < intensity_threshold_)
            {
                continue;
            }

            ParsedPoint pt;
            projectToCsfFrame(x, y, z, pt.x, pt.y, pt.z);
            pt.intensity = static_cast<float>(intensity);
            pt.source_index = i;
            points.push_back(pt);
        }

        return points;
    }

    GridData buildClothGrid(const std::vector<ParsedPoint> &points) const
    {
        GridData grid;
        if (points.empty())
        {
            return grid;
        }

        float min_x = points.front().x;
        float min_z = points.front().z;
        float max_x = points.front().x;
        float max_z = points.front().z;
        for (const auto &pt : points)
        {
            min_x = std::min(min_x, pt.x);
            min_z = std::min(min_z, pt.z);
            max_x = std::max(max_x, pt.x);
            max_z = std::max(max_z, pt.z);
        }

        grid.origin_xz = Eigen::Vector2f(min_x, min_z);
        grid.width = static_cast<int>(std::floor((max_x - min_x) / cloth_resolution_m_)) + 1;
        grid.height = static_cast<int>(std::floor((max_z - min_z) / cloth_resolution_m_)) + 1;
        grid.grid_x.resize(static_cast<std::size_t>(grid.width));
        grid.grid_z.resize(static_cast<std::size_t>(grid.height));
        for (int u = 0; u < grid.width; ++u)
        {
            grid.grid_x[static_cast<std::size_t>(u)] = min_x + static_cast<float>(u) * static_cast<float>(cloth_resolution_m_);
        }
        for (int v = 0; v < grid.height; ++v)
        {
            grid.grid_z[static_cast<std::size_t>(v)] = min_z + static_cast<float>(v) * static_cast<float>(cloth_resolution_m_);
        }
        return grid;
    }

    std::vector<float> nearestIntersectionHeight(const std::vector<ParsedPoint> &points, const GridData &grid) const
    {
        std::vector<float> intersection(static_cast<std::size_t>(grid.width * grid.height), 0.0f);
        pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>());
        cloud->reserve(points.size());
        for (const auto &pt : points)
        {
            pcl::PointXY xy;
            xy.x = pt.x;
            xy.y = pt.z;
            cloud->push_back(xy);
        }

        pcl::KdTreeFLANN<pcl::PointXY> tree;
        tree.setInputCloud(cloud);
        std::vector<int> nearest_idx(1, 0);
        std::vector<float> nearest_dist(1, 0.0f);
        for (int v = 0; v < grid.height; ++v)
        {
            for (int u = 0; u < grid.width; ++u)
            {
                pcl::PointXY query;
                query.x = grid.grid_x[static_cast<std::size_t>(u)];
                query.y = grid.grid_z[static_cast<std::size_t>(v)];
                tree.nearestKSearch(query, 1, nearest_idx, nearest_dist);
                intersection[static_cast<std::size_t>(v * grid.width + u)] =
                    -points[static_cast<std::size_t>(nearest_idx.front())].y;
            }
        }
        return intersection;
    }

    void applyInternalConstraints(std::vector<float> &cloth_height,
                                  const std::vector<char> &movable_mask,
                                  int width,
                                  int height) const
    {
        if (rigidness_ <= 0 || width <= 0 || height <= 0)
        {
            return;
        }

        for (int iter = 0; iter < rigidness_; ++iter)
        {
            std::vector<float> delta(cloth_height.size(), 0.0f);
            std::vector<int> count(cloth_height.size(), 0);

            for (int v = 0; v < height; ++v)
            {
                for (int u = 0; u + 1 < width; ++u)
                {
                    const int lhs = v * width + u;
                    const int rhs = lhs + 1;
                    if (!movable_mask[static_cast<std::size_t>(lhs)] && !movable_mask[static_cast<std::size_t>(rhs)])
                    {
                        continue;
                    }
                    const float diff = cloth_height[static_cast<std::size_t>(rhs)] - cloth_height[static_cast<std::size_t>(lhs)];
                    if (movable_mask[static_cast<std::size_t>(lhs)])
                    {
                        delta[static_cast<std::size_t>(lhs)] += 0.5f * diff;
                        count[static_cast<std::size_t>(lhs)] += 1;
                    }
                    if (movable_mask[static_cast<std::size_t>(rhs)])
                    {
                        delta[static_cast<std::size_t>(rhs)] -= 0.5f * diff;
                        count[static_cast<std::size_t>(rhs)] += 1;
                    }
                }
            }

            for (int v = 0; v + 1 < height; ++v)
            {
                for (int u = 0; u < width; ++u)
                {
                    const int top = v * width + u;
                    const int bottom = (v + 1) * width + u;
                    if (!movable_mask[static_cast<std::size_t>(top)] && !movable_mask[static_cast<std::size_t>(bottom)])
                    {
                        continue;
                    }
                    const float diff = cloth_height[static_cast<std::size_t>(bottom)] - cloth_height[static_cast<std::size_t>(top)];
                    if (movable_mask[static_cast<std::size_t>(top)])
                    {
                        delta[static_cast<std::size_t>(top)] += 0.5f * diff;
                        count[static_cast<std::size_t>(top)] += 1;
                    }
                    if (movable_mask[static_cast<std::size_t>(bottom)])
                    {
                        delta[static_cast<std::size_t>(bottom)] -= 0.5f * diff;
                        count[static_cast<std::size_t>(bottom)] += 1;
                    }
                }
            }

            for (std::size_t i = 0; i < cloth_height.size(); ++i)
            {
                if (movable_mask[i] && count[i] > 0)
                {
                    cloth_height[i] += delta[i] / static_cast<float>(count[i]);
                }
            }
        }
    }

    void postprocessSteepSlopes(std::vector<float> &cloth_height,
                                std::vector<char> &movable_mask,
                                const std::vector<float> &intersection_height,
                                int width,
                                int height) const
    {
        if (width <= 0 || height <= 0)
        {
            return;
        }

        const int dv4[4] = {-1, 1, 0, 0};
        const int du4[4] = {0, 0, -1, 1};
        std::vector<char> visited(movable_mask.size(), 0);

        for (int start_v = 0; start_v < height; ++start_v)
        {
            for (int start_u = 0; start_u < width; ++start_u)
            {
                const int start_idx = start_v * width + start_u;
                if (!movable_mask[static_cast<std::size_t>(start_idx)] || visited[static_cast<std::size_t>(start_idx)])
                {
                    continue;
                }

                std::vector<int> component;
                std::deque<int> queue;
                queue.push_back(start_idx);
                visited[static_cast<std::size_t>(start_idx)] = 1;

                while (!queue.empty())
                {
                    const int idx = queue.front();
                    queue.pop_front();
                    component.push_back(idx);
                    const int v = idx / width;
                    const int u = idx % width;
                    for (int k = 0; k < 4; ++k)
                    {
                        const int nv = v + dv4[k];
                        const int nu = u + du4[k];
                        if (nv < 0 || nv >= height || nu < 0 || nu >= width)
                        {
                            continue;
                        }
                        const int nidx = nv * width + nu;
                        if (visited[static_cast<std::size_t>(nidx)] || !movable_mask[static_cast<std::size_t>(nidx)])
                        {
                            continue;
                        }
                        visited[static_cast<std::size_t>(nidx)] = 1;
                        queue.push_back(nidx);
                    }
                }

                std::deque<int> bfs_queue;
                std::vector<char> queued(movable_mask.size(), 0);
                for (const int idx : component)
                {
                    const int v = idx / width;
                    const int u = idx % width;
                    bool has_unmovable_neighbor = false;
                    for (int k = 0; k < 4; ++k)
                    {
                        const int nv = v + dv4[k];
                        const int nu = u + du4[k];
                        if (nv < 0 || nv >= height || nu < 0 || nu >= width)
                        {
                            continue;
                        }
                        const int nidx = nv * width + nu;
                        if (!movable_mask[static_cast<std::size_t>(nidx)])
                        {
                            has_unmovable_neighbor = true;
                            break;
                        }
                    }
                    if (has_unmovable_neighbor)
                    {
                        bfs_queue.push_back(idx);
                        queued[static_cast<std::size_t>(idx)] = 1;
                    }
                }

                while (!bfs_queue.empty())
                {
                    const int idx = bfs_queue.front();
                    bfs_queue.pop_front();
                    if (!movable_mask[static_cast<std::size_t>(idx)])
                    {
                        continue;
                    }

                    const int v = idx / width;
                    const int u = idx % width;
                    bool should_lock = false;
                    for (int k = 0; k < 4; ++k)
                    {
                        const int nv = v + dv4[k];
                        const int nu = u + du4[k];
                        if (nv < 0 || nv >= height || nu < 0 || nu >= width)
                        {
                            continue;
                        }
                        const int nidx = nv * width + nu;
                        if (movable_mask[static_cast<std::size_t>(nidx)])
                        {
                            continue;
                        }
                        const float height_gap = std::fabs(intersection_height[static_cast<std::size_t>(idx)] -
                                                           intersection_height[static_cast<std::size_t>(nidx)]);
                        if (height_gap <= static_cast<float>(postprocess_height_threshold_m_))
                        {
                            should_lock = true;
                            break;
                        }
                    }

                    if (should_lock)
                    {
                        cloth_height[static_cast<std::size_t>(idx)] = intersection_height[static_cast<std::size_t>(idx)];
                        movable_mask[static_cast<std::size_t>(idx)] = 0;
                    }

                    for (int k = 0; k < 4; ++k)
                    {
                        const int nv = v + dv4[k];
                        const int nu = u + du4[k];
                        if (nv < 0 || nv >= height || nu < 0 || nu >= width)
                        {
                            continue;
                        }
                        const int nidx = nv * width + nu;
                        if (movable_mask[static_cast<std::size_t>(nidx)] && !queued[static_cast<std::size_t>(nidx)])
                        {
                            bfs_queue.push_back(nidx);
                            queued[static_cast<std::size_t>(nidx)] = 1;
                        }
                    }
                }
            }
        }
    }

    std::vector<float> interpolateClothHeight(const std::vector<float> &cloth_height,
                                              const GridData &grid,
                                              const std::vector<ParsedPoint> &points) const
    {
        std::vector<float> interpolated(points.size(), 0.0f);
        if (grid.width <= 0 || grid.height <= 0)
        {
            return interpolated;
        }

        for (std::size_t i = 0; i < points.size(); ++i)
        {
            const float u = (points[i].x - grid.origin_xz.x()) / static_cast<float>(cloth_resolution_m_);
            const float v = (points[i].z - grid.origin_xz.y()) / static_cast<float>(cloth_resolution_m_);
            int u0 = static_cast<int>(std::floor(u));
            int v0 = static_cast<int>(std::floor(v));
            u0 = std::max(0, std::min(u0, grid.width - 1));
            v0 = std::max(0, std::min(v0, grid.height - 1));
            const int u1 = std::max(0, std::min(u0 + 1, grid.width - 1));
            const int v1 = std::max(0, std::min(v0 + 1, grid.height - 1));

            const float du = std::max(0.0f, std::min(u - static_cast<float>(u0), 1.0f));
            const float dv = std::max(0.0f, std::min(v - static_cast<float>(v0), 1.0f));

            const float h00 = cloth_height[static_cast<std::size_t>(v0 * grid.width + u0)];
            const float h10 = cloth_height[static_cast<std::size_t>(v0 * grid.width + u1)];
            const float h01 = cloth_height[static_cast<std::size_t>(v1 * grid.width + u0)];
            const float h11 = cloth_height[static_cast<std::size_t>(v1 * grid.width + u1)];

            const float top = h00 * (1.0f - du) + h10 * du;
            const float bottom = h01 * (1.0f - du) + h11 * du;
            interpolated[i] = top * (1.0f - dv) + bottom * dv;
        }
        return interpolated;
    }

    std::vector<std::vector<int>> getConnectedComponents(const std::vector<char> &occupancy, int width, int height) const
    {
        std::vector<std::vector<int>> components;
        if (width <= 0 || height <= 0)
        {
            return components;
        }

        const int dv8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        const int du8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        std::vector<char> visited(occupancy.size(), 0);
        for (int v = 0; v < height; ++v)
        {
            for (int u = 0; u < width; ++u)
            {
                const int idx = v * width + u;
                if (!occupancy[static_cast<std::size_t>(idx)] || visited[static_cast<std::size_t>(idx)])
                {
                    continue;
                }
                std::vector<int> component;
                std::deque<int> queue;
                queue.push_back(idx);
                visited[static_cast<std::size_t>(idx)] = 1;
                while (!queue.empty())
                {
                    const int cur = queue.front();
                    queue.pop_front();
                    component.push_back(cur);
                    const int cv = cur / width;
                    const int cu = cur % width;
                    for (int k = 0; k < 8; ++k)
                    {
                        const int nv = cv + dv8[k];
                        const int nu = cu + du8[k];
                        if (nv < 0 || nv >= height || nu < 0 || nu >= width)
                        {
                            continue;
                        }
                        const int nidx = nv * width + nu;
                        if (visited[static_cast<std::size_t>(nidx)] || !occupancy[static_cast<std::size_t>(nidx)])
                        {
                            continue;
                        }
                        visited[static_cast<std::size_t>(nidx)] = 1;
                        queue.push_back(nidx);
                    }
                }
                components.push_back(component);
            }
        }
        return components;
    }

    std::vector<char> applyNongroundDilation3DKnn(const std::vector<ParsedPoint> &points,
                                                  const std::vector<char> &raw_ground_mask,
                                                  const std::vector<char> &seed_mask) const
    {
        std::vector<char> ground_mask = raw_ground_mask;
        pcl::PointCloud<pcl::PointXYZ>::Ptr seed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        std::vector<int> ground_indices;
        for (int i = 0; i < static_cast<int>(points.size()); ++i)
        {
            if (seed_mask[static_cast<std::size_t>(i)])
            {
                seed_cloud->push_back(pcl::PointXYZ(points[static_cast<std::size_t>(i)].x,
                                                   points[static_cast<std::size_t>(i)].y,
                                                   points[static_cast<std::size_t>(i)].z));
            }
            if (raw_ground_mask[static_cast<std::size_t>(i)])
            {
                ground_indices.push_back(i);
            }
        }

        if (seed_cloud->empty())
        {
            return ground_mask;
        }

        pcl::KdTreeFLANN<pcl::PointXYZ> tree;
        tree.setInputCloud(seed_cloud);
        std::vector<int> nearest_idx(1, 0);
        std::vector<float> nearest_sq_dist(1, 0.0f);
        const float threshold_sq = static_cast<float>(nonground_dilation_m_ * nonground_dilation_m_);
        for (const int idx : ground_indices)
        {
            pcl::PointXYZ query(points[static_cast<std::size_t>(idx)].x,
                                points[static_cast<std::size_t>(idx)].y,
                                points[static_cast<std::size_t>(idx)].z);
            if (tree.nearestKSearch(query, 1, nearest_idx, nearest_sq_dist) > 0 &&
                nearest_sq_dist.front() <= threshold_sq)
            {
                ground_mask[static_cast<std::size_t>(idx)] = 0;
            }
        }
        return ground_mask;
    }

    std::vector<char> applyNongroundDilation2DGrid(const std::vector<ParsedPoint> &points,
                                                   const std::vector<char> &raw_ground_mask,
                                                   const std::vector<char> &seed_mask,
                                                   float min_x,
                                                   float min_z,
                                                   int width,
                                                   int height) const
    {
        std::vector<char> ground_mask = raw_ground_mask;
        std::vector<int> ground_indices;
        std::vector<char> seed_occupancy(static_cast<std::size_t>(width * height), 0);
        for (int i = 0; i < static_cast<int>(points.size()); ++i)
        {
            if (seed_mask[static_cast<std::size_t>(i)])
            {
                const int u = static_cast<int>(std::floor((points[static_cast<std::size_t>(i)].x - min_x) / nonground_seed_grid_resolution_m_));
                const int v = static_cast<int>(std::floor((points[static_cast<std::size_t>(i)].z - min_z) / nonground_seed_grid_resolution_m_));
                if (u >= 0 && u < width && v >= 0 && v < height)
                {
                    seed_occupancy[static_cast<std::size_t>(v * width + u)] = 1;
                }
            }
            if (raw_ground_mask[static_cast<std::size_t>(i)])
            {
                ground_indices.push_back(i);
            }
        }

        bool has_seed = false;
        for (const char occupied : seed_occupancy)
        {
            if (occupied)
            {
                has_seed = true;
                break;
            }
        }
        if (!has_seed)
        {
            return ground_mask;
        }

        const int radius_cells = std::max(1, static_cast<int>(std::ceil(nonground_dilation_m_ / nonground_seed_grid_resolution_m_)));
        std::vector<std::pair<int, int>> neighbor_offsets;
        neighbor_offsets.reserve(static_cast<std::size_t>((2 * radius_cells + 1) * (2 * radius_cells + 1)));
        const float threshold_sq = static_cast<float>(nonground_dilation_m_ * nonground_dilation_m_);
        for (int dv = -radius_cells; dv <= radius_cells; ++dv)
        {
            for (int du = -radius_cells; du <= radius_cells; ++du)
            {
                const float dx = static_cast<float>(du) * static_cast<float>(nonground_seed_grid_resolution_m_);
                const float dz = static_cast<float>(dv) * static_cast<float>(nonground_seed_grid_resolution_m_);
                if (dx * dx + dz * dz <= threshold_sq)
                {
                    neighbor_offsets.emplace_back(du, dv);
                }
            }
        }

        for (const int idx : ground_indices)
        {
            const auto &pt = points[static_cast<std::size_t>(idx)];
            const int u = static_cast<int>(std::floor((pt.x - min_x) / nonground_seed_grid_resolution_m_));
            const int v = static_cast<int>(std::floor((pt.z - min_z) / nonground_seed_grid_resolution_m_));
            bool near_seed = false;
            for (const auto &offset : neighbor_offsets)
            {
                const int nu = u + offset.first;
                const int nv = v + offset.second;
                if (nu < 0 || nu >= width || nv < 0 || nv >= height)
                {
                    continue;
                }
                if (seed_occupancy[static_cast<std::size_t>(nv * width + nu)])
                {
                    near_seed = true;
                    break;
                }
            }
            if (near_seed)
            {
                ground_mask[static_cast<std::size_t>(idx)] = 0;
            }
        }

        return ground_mask;
    }

    std::vector<int> getBevCandidateGroundIndices(const std::vector<ParsedPoint> &points,
                                                  const std::vector<char> &raw_ground_mask,
                                                  const std::vector<char> &seed_mask,
                                                  float min_x,
                                                  float min_z,
                                                  int width,
                                                  int height) const
    {
        std::vector<int> candidates;
        if (width <= 0 || height <= 0 || nonground_seed_grid_resolution_m_ <= 0.0)
        {
            return candidates;
        }

        std::vector<char> seed_occupancy(static_cast<std::size_t>(width * height), 0);
        for (int i = 0; i < static_cast<int>(points.size()); ++i)
        {
            if (!seed_mask[static_cast<std::size_t>(i)])
            {
                continue;
            }
            const int u = static_cast<int>(std::floor((points[static_cast<std::size_t>(i)].x - min_x) /
                                                      nonground_seed_grid_resolution_m_));
            const int v = static_cast<int>(std::floor((points[static_cast<std::size_t>(i)].z - min_z) /
                                                      nonground_seed_grid_resolution_m_));
            if (u >= 0 && u < width && v >= 0 && v < height)
            {
                seed_occupancy[static_cast<std::size_t>(v * width + u)] = 1;
            }
        }

        bool has_seed = false;
        for (const char occupied : seed_occupancy)
        {
            if (occupied)
            {
                has_seed = true;
                break;
            }
        }
        if (!has_seed)
        {
            return candidates;
        }

        const double candidate_radius_m =
            nonground_dilation_m_ + std::sqrt(2.0) * nonground_seed_grid_resolution_m_;
        const int radius_cells =
            std::max(1, static_cast<int>(std::ceil(candidate_radius_m / nonground_seed_grid_resolution_m_)));
        const float threshold_sq = static_cast<float>(candidate_radius_m * candidate_radius_m);
        std::vector<std::pair<int, int>> neighbor_offsets;
        neighbor_offsets.reserve(static_cast<std::size_t>((2 * radius_cells + 1) * (2 * radius_cells + 1)));
        for (int dv = -radius_cells; dv <= radius_cells; ++dv)
        {
            for (int du = -radius_cells; du <= radius_cells; ++du)
            {
                const float dx = static_cast<float>(du) * static_cast<float>(nonground_seed_grid_resolution_m_);
                const float dz = static_cast<float>(dv) * static_cast<float>(nonground_seed_grid_resolution_m_);
                if (dx * dx + dz * dz <= threshold_sq)
                {
                    neighbor_offsets.emplace_back(du, dv);
                }
            }
        }

        for (int i = 0; i < static_cast<int>(points.size()); ++i)
        {
            if (!raw_ground_mask[static_cast<std::size_t>(i)])
            {
                continue;
            }
            const auto &pt = points[static_cast<std::size_t>(i)];
            const int u = static_cast<int>(std::floor((pt.x - min_x) / nonground_seed_grid_resolution_m_));
            const int v = static_cast<int>(std::floor((pt.z - min_z) / nonground_seed_grid_resolution_m_));
            for (const auto &offset : neighbor_offsets)
            {
                const int nu = u + offset.first;
                const int nv = v + offset.second;
                if (nu < 0 || nu >= width || nv < 0 || nv >= height)
                {
                    continue;
                }
                if (seed_occupancy[static_cast<std::size_t>(nv * width + nu)])
                {
                    candidates.push_back(i);
                    break;
                }
            }
        }
        return candidates;
    }

    struct VoxelKey
    {
        int x = 0;
        int y = 0;
        int z = 0;

        bool operator==(const VoxelKey &other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash
    {
        std::size_t operator()(const VoxelKey &key) const
        {
            std::size_t h = static_cast<std::size_t>(key.x) * 73856093u;
            h ^= static_cast<std::size_t>(key.y) * 19349663u;
            h ^= static_cast<std::size_t>(key.z) * 83492791u;
            return h;
        }
    };

    VoxelKey voxelKeyForPoint(const ParsedPoint &pt, double voxel_res) const
    {
        return VoxelKey{
            static_cast<int>(std::floor(pt.x / voxel_res)),
            static_cast<int>(std::floor(pt.y / voxel_res)),
            static_cast<int>(std::floor(pt.z / voxel_res))};
    }

    std::vector<char> applyBevCandidate3DVoxel(const std::vector<ParsedPoint> &points,
                                               const std::vector<char> &raw_ground_mask,
                                               const std::vector<char> &seed_mask,
                                               float min_x,
                                               float min_z,
                                               int width,
                                               int height) const
    {
        std::vector<char> ground_mask = raw_ground_mask;
        const std::vector<int> candidate_indices =
            getBevCandidateGroundIndices(points, raw_ground_mask, seed_mask, min_x, min_z, width, height);
        if (candidate_indices.empty())
        {
            return ground_mask;
        }

        double voxel_res = candidate_voxel_resolution_m_ > 0.0 ? candidate_voxel_resolution_m_
                                                               : nonground_dilation_m_;
        if (voxel_res <= 0.0)
        {
            return ground_mask;
        }

        std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> voxel_map;
        voxel_map.reserve(seed_mask.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i)
        {
            if (!seed_mask[static_cast<std::size_t>(i)])
            {
                continue;
            }
            voxel_map[voxelKeyForPoint(points[static_cast<std::size_t>(i)], voxel_res)].push_back(i);
        }
        if (voxel_map.empty())
        {
            return ground_mask;
        }

        const int radius_cells = std::max(1, static_cast<int>(std::ceil(nonground_dilation_m_ / voxel_res)));
        const float threshold_sq = static_cast<float>(nonground_dilation_m_ * nonground_dilation_m_);
        for (const int idx : candidate_indices)
        {
            const auto &query = points[static_cast<std::size_t>(idx)];
            const VoxelKey base = voxelKeyForPoint(query, voxel_res);
            bool remove = false;
            for (int dx = -radius_cells; dx <= radius_cells && !remove; ++dx)
            {
                for (int dy = -radius_cells; dy <= radius_cells && !remove; ++dy)
                {
                    for (int dz = -radius_cells; dz <= radius_cells; ++dz)
                    {
                        const VoxelKey key{base.x + dx, base.y + dy, base.z + dz};
                        const auto it = voxel_map.find(key);
                        if (it == voxel_map.end())
                        {
                            continue;
                        }
                        for (const int seed_idx : it->second)
                        {
                            const auto &seed = points[static_cast<std::size_t>(seed_idx)];
                            const float ddx = seed.x - query.x;
                            const float ddy = seed.y - query.y;
                            const float ddz = seed.z - query.z;
                            if (ddx * ddx + ddy * ddy + ddz * ddz <= threshold_sq)
                            {
                                remove = true;
                                break;
                            }
                        }
                        if (remove)
                        {
                            break;
                        }
                    }
                }
            }
            if (remove)
            {
                ground_mask[static_cast<std::size_t>(idx)] = 0;
            }
        }
        return ground_mask;
    }

    std::vector<char> applyNongroundDilation(const std::vector<ParsedPoint> &points,
                                             const std::vector<char> &raw_ground_mask) const
    {
        std::vector<char> ground_mask = raw_ground_mask;
        if (points.empty() || nonground_dilation_m_ <= 0.0)
        {
            return ground_mask;
        }

        std::vector<int> nonground_indices;
        nonground_indices.reserve(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i)
        {
            if (!raw_ground_mask[static_cast<std::size_t>(i)])
            {
                nonground_indices.push_back(i);
            }
        }
        if (nonground_indices.empty())
        {
            return ground_mask;
        }

        float min_x = points[static_cast<std::size_t>(nonground_indices.front())].x;
        float min_z = points[static_cast<std::size_t>(nonground_indices.front())].z;
        float max_x = min_x;
        float max_z = min_z;
        for (const int idx : nonground_indices)
        {
            min_x = std::min(min_x, points[static_cast<std::size_t>(idx)].x);
            min_z = std::min(min_z, points[static_cast<std::size_t>(idx)].z);
            max_x = std::max(max_x, points[static_cast<std::size_t>(idx)].x);
            max_z = std::max(max_z, points[static_cast<std::size_t>(idx)].z);
        }

        const int width = static_cast<int>(std::floor((max_x - min_x) / nonground_seed_grid_resolution_m_)) + 1;
        const int height = static_cast<int>(std::floor((max_z - min_z) / nonground_seed_grid_resolution_m_)) + 1;
        std::vector<char> occupancy(static_cast<std::size_t>(width * height), 0);
        std::vector<int> point_linear_idx(nonground_indices.size(), 0);
        for (std::size_t local_i = 0; local_i < nonground_indices.size(); ++local_i)
        {
            const auto &pt = points[static_cast<std::size_t>(nonground_indices[local_i])];
            const int u = static_cast<int>(std::floor((pt.x - min_x) / nonground_seed_grid_resolution_m_));
            const int v = static_cast<int>(std::floor((pt.z - min_z) / nonground_seed_grid_resolution_m_));
            const int linear_idx = v * width + u;
            occupancy[static_cast<std::size_t>(linear_idx)] = 1;
            point_linear_idx[local_i] = linear_idx;
        }

        const std::vector<std::vector<int>> components = getConnectedComponents(occupancy, width, height);
        std::vector<char> seed_mask(points.size(), 0);
        for (const auto &component : components)
        {
            std::vector<char> in_component(occupancy.size(), 0);
            for (const int linear_idx : component)
            {
                in_component[static_cast<std::size_t>(linear_idx)] = 1;
            }
            int point_count = 0;
            for (const int linear_idx : point_linear_idx)
            {
                point_count += in_component[static_cast<std::size_t>(linear_idx)] ? 1 : 0;
            }
            const bool use_as_seed =
                point_count >= nonground_dilation_min_seed_points_ ||
                static_cast<int>(component.size()) >= nonground_dilation_min_seed_cells_;
            if (!use_as_seed)
            {
                continue;
            }
            for (std::size_t local_i = 0; local_i < nonground_indices.size(); ++local_i)
            {
                if (in_component[static_cast<std::size_t>(point_linear_idx[local_i])])
                {
                    seed_mask[static_cast<std::size_t>(nonground_indices[local_i])] = 1;
                }
            }
        }

        if (nonground_dilation_mode_ == "knn_3d")
        {
            return applyNongroundDilation3DKnn(points, raw_ground_mask, seed_mask);
        }
        if (nonground_dilation_mode_ == "bev_candidate_voxel_3d")
        {
            return applyBevCandidate3DVoxel(points, raw_ground_mask, seed_mask, min_x, min_z, width, height);
        }
        return applyNongroundDilation2DGrid(points, raw_ground_mask, seed_mask, min_x, min_z, width, height);
    }

    sensor_msgs::PointCloud2 assembleOutput(const sensor_msgs::PointCloud2 &msg,
                                            const CloudFieldLayout &layout,
                                            const std::vector<char> &keep_original) const
    {
        sensor_msgs::PointCloud2 out = msg;
        if (!output_frame_id_.empty())
        {
            out.header.frame_id = output_frame_id_;
        }
        const std::size_t point_count = static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
        out.data.resize(point_count * static_cast<std::size_t>(msg.point_step));

        std::size_t kept = 0;
        for (std::size_t i = 0; i < point_count; ++i)
        {
            if (!keep_original[i])
            {
                continue;
            }
            const uint8_t *src = msg.data.data() + i * msg.point_step;
            uint8_t *dst = out.data.data() + kept * msg.point_step;
            std::memcpy(dst, src, msg.point_step);
            if (output_rotation_enabled_)
            {
                double x = 0.0, y = 0.0, z = 0.0;
                if (readAsDouble(dst + layout.x_offset, layout.x_type, x) &&
                    readAsDouble(dst + layout.y_offset, layout.y_type, y) &&
                    readAsDouble(dst + layout.z_offset, layout.z_type, z))
                {
                    const double rx = output_rotation_[0][0] * x + output_rotation_[0][1] * y + output_rotation_[0][2] * z;
                    const double ry = output_rotation_[1][0] * x + output_rotation_[1][1] * y + output_rotation_[1][2] * z;
                    const double rz = output_rotation_[2][0] * x + output_rotation_[2][1] * y + output_rotation_[2][2] * z;
                    writeFromDouble(dst + layout.x_offset, layout.x_type, rx);
                    writeFromDouble(dst + layout.y_offset, layout.y_type, ry);
                    writeFromDouble(dst + layout.z_offset, layout.z_type, rz);
                }
            }
            ++kept;
        }

        out.height = 1;
        out.width = static_cast<uint32_t>(kept);
        out.row_step = static_cast<uint32_t>(kept * msg.point_step);
        out.data.resize(kept * static_cast<std::size_t>(msg.point_step));
        out.is_dense = drop_nan_;
        return out;
    }

    sensor_msgs::PointCloud2 assembleLabeledMaskCloud(const sensor_msgs::PointCloud2 &msg,
                                                      const CloudFieldLayout &layout,
                                                      const std::vector<char> &keep_original) const
    {
        sensor_msgs::PointCloud2 out = msg;
        if (!output_frame_id_.empty())
        {
            out.header.frame_id = output_frame_id_;
        }
        out.height = 1;
        const uint32_t input_point_step = msg.point_step;
        out.fields = msg.fields;
        sensor_msgs::PointField label_field;
        label_field.name = "label";
        label_field.offset = input_point_step;
        label_field.datatype = sensor_msgs::PointField::FLOAT32;
        label_field.count = 1;
        out.fields.push_back(label_field);
        out.point_step = input_point_step + sizeof(float);
        const std::size_t point_count = static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
        out.data.resize(point_count * static_cast<std::size_t>(out.point_step));

        std::size_t valid_count = 0;
        for (std::size_t i = 0; i < point_count; ++i)
        {
            const uint8_t *src = msg.data.data() + i * msg.point_step;
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            if (!readAsDouble(src + layout.x_offset, layout.x_type, x) ||
                !readAsDouble(src + layout.y_offset, layout.y_type, y) ||
                !readAsDouble(src + layout.z_offset, layout.z_type, z))
            {
                continue;
            }

            if (output_rotation_enabled_)
            {
                const double rx = output_rotation_[0][0] * x + output_rotation_[0][1] * y + output_rotation_[0][2] * z;
                const double ry = output_rotation_[1][0] * x + output_rotation_[1][1] * y + output_rotation_[1][2] * z;
                const double rz = output_rotation_[2][0] * x + output_rotation_[2][1] * y + output_rotation_[2][2] * z;
                x = rx;
                y = ry;
                z = rz;
            }

            const float label = (i < keep_original.size() && keep_original[i]) ? 0.0f : 1.0f;
            uint8_t *dst = out.data.data() + valid_count * out.point_step;
            std::memcpy(dst, src, input_point_step);
            writeFromDouble(dst + layout.x_offset, layout.x_type, x);
            writeFromDouble(dst + layout.y_offset, layout.y_type, y);
            writeFromDouble(dst + layout.z_offset, layout.z_type, z);
            if (layout.has_intensity)
                writeFromDouble(dst + layout.intensity_offset, layout.intensity_type, label);
            std::memcpy(dst + input_point_step, &label, sizeof(float));
            ++valid_count;
        }

        out.width = static_cast<uint32_t>(valid_count);
        out.row_step = static_cast<uint32_t>(valid_count * out.point_step);
        out.data.resize(valid_count * static_cast<std::size_t>(out.point_step));
        return out;
    }

    void cloudHandler(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        const auto t_start = std::chrono::steady_clock::now();
        const std::size_t point_count = static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height);
        if (point_count == 0 || msg->point_step == 0)
        {
            pub_cloud_.publish(*msg);
            return;
        }

        const std::size_t expected_size = point_count * static_cast<std::size_t>(msg->point_step);
        if (msg->data.size() < expected_size)
        {
            ROS_WARN_THROTTLE(2.0, "tcrlm_tcr_construction input data size mismatch, pass-through.");
            pub_cloud_.publish(*msg);
            return;
        }

        const CloudFieldLayout layout = buildFieldLayout(*msg);
        if (!layout.has_xyz || !layout.has_intensity)
        {
            ROS_WARN_THROTTLE(2.0, "tcrlm_tcr_construction cannot find x/y/z/intensity fields, pass-through.");
            pub_cloud_.publish(*msg);
            return;
        }

        std::vector<ParsedPoint> filtered_points = extractPoints(*msg, layout);
        if (filtered_points.empty())
        {
            pub_cloud_.publish(makeEmptyOutput(*msg, drop_nan_));
            return;
        }

        const GridData grid = buildClothGrid(filtered_points);
        std::vector<float> intersection_height = nearestIntersectionHeight(filtered_points, grid);

        const float initial_height = *std::max_element(intersection_height.begin(), intersection_height.end()) +
                                     static_cast<float>(cloth_buffer_m_);
        std::vector<float> cloth_height(intersection_height.size(), initial_height);
        std::vector<float> previous_height = cloth_height;
        std::vector<char> movable_mask(intersection_height.size(), 1);
        const float gravity_term = -static_cast<float>(time_step_ * time_step_);

        int iterations = 0;
        float max_height_delta = 0.0f;
        for (int iteration_idx = 0; iteration_idx < max_iterations_; ++iteration_idx)
        {
            const std::vector<float> before_step = cloth_height;
            for (std::size_t i = 0; i < cloth_height.size(); ++i)
            {
                if (movable_mask[i])
                {
                    cloth_height[i] = 2.0f * cloth_height[i] - previous_height[i] + gravity_term;
                }
            }
            previous_height = before_step;

            for (std::size_t i = 0; i < cloth_height.size(); ++i)
            {
                if (movable_mask[i] && cloth_height[i] <= intersection_height[i])
                {
                    cloth_height[i] = intersection_height[i];
                    movable_mask[i] = 0;
                    previous_height[i] = cloth_height[i];
                }
            }

            applyInternalConstraints(cloth_height, movable_mask, grid.width, grid.height);

            for (std::size_t i = 0; i < cloth_height.size(); ++i)
            {
                if (movable_mask[i] && cloth_height[i] <= intersection_height[i])
                {
                    cloth_height[i] = intersection_height[i];
                    movable_mask[i] = 0;
                    previous_height[i] = cloth_height[i];
                }
            }

            max_height_delta = 0.0f;
            for (std::size_t i = 0; i < cloth_height.size(); ++i)
            {
                max_height_delta = std::max(max_height_delta, std::fabs(cloth_height[i] - before_step[i]));
            }
            iterations = iteration_idx + 1;
            if (max_height_delta <= static_cast<float>(convergence_eps_m_))
            {
                break;
            }
        }

        if (steep_slope_fit_ && !cloth_height.empty())
        {
            postprocessSteepSlopes(cloth_height, movable_mask, intersection_height, grid.width, grid.height);
        }

        const std::vector<float> point_cloth_height_inverted = interpolateClothHeight(cloth_height, grid, filtered_points);
        std::vector<char> raw_ground_mask(filtered_points.size(), 0);
        for (std::size_t i = 0; i < filtered_points.size(); ++i)
        {
            const float point_ground_height = -point_cloth_height_inverted[i];
            const float residual = filtered_points[i].y - point_ground_height;
            raw_ground_mask[i] = std::fabs(residual) <= static_cast<float>(classification_threshold_m_) ? 1 : 0;
        }

        const std::vector<char> ground_mask = applyNongroundDilation(filtered_points, raw_ground_mask);
        std::vector<char> keep_original(point_count, 0);
        for (std::size_t i = 0; i < filtered_points.size(); ++i)
        {
            if (ground_mask[i])
            {
                keep_original[filtered_points[i].source_index] = 1;
            }
        }

        exportLabeledTxt(*msg, layout, keep_original);

        sensor_msgs::PointCloud2 out = assembleOutput(*msg, layout, keep_original);
        pub_cloud_.publish(out);
        if (publish_labeled_mask_)
        {
            pub_labeled_mask_.publish(assembleLabeledMaskCloud(*msg, layout, keep_original));
        }

        if (log_timing_)
        {
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start).count();
            const double kept_ratio =
                point_count > 0 ? static_cast<double>(out.width * out.height) / static_cast<double>(point_count) : 0.0;
            ROS_INFO_STREAM("tcrlm_tcr_construction timing: input=" << point_count
                            << ", filtered=" << filtered_points.size()
                            << ", output=" << out.width * out.height
                            << ", kept_ratio=" << kept_ratio
                            << ", iterations=" << iterations
                            << ", max_height_delta_m=" << max_height_delta
                            << ", elapsed_ms=" << elapsed_ms);
        }
    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_labeled_mask_;
    bool publish_labeled_mask_ = false;

    std::string input_topic_;
    std::string output_topic_;
    std::string labeled_mask_topic_;
    std::string height_axis_name_ = "y";
    HeightAxis height_axis_ = HeightAxis::Y;
    std::string output_frame_id_;
    double output_rotation_roll_deg_ = 0.0;
    double output_rotation_pitch_deg_ = 0.0;
    double output_rotation_yaw_deg_ = 0.0;
    bool output_rotation_enabled_ = false;
    double output_rotation_[3][3] = {{1.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0},
                                     {0.0, 0.0, 1.0}};

    bool drop_nan_ = true;
    int input_queue_size_ = 100;
    bool log_timing_ = false;

    double intensity_threshold_ = 0.0;
    double cloth_resolution_m_ = 1.0;
    double time_step_ = 0.1;
    int rigidness_ = 2;
    bool steep_slope_fit_ = false;
    double classification_threshold_m_ = 0.5;
    double postprocess_height_threshold_m_ = 0.3;
    double cloth_buffer_m_ = 2.0;
    int max_iterations_ = 100;
    double convergence_eps_m_ = 0.01;
    double nonground_dilation_m_ = 0.0;
    std::string nonground_dilation_mode_ = "grid_2d";
    double nonground_seed_grid_resolution_m_ = 0.5;
    int nonground_dilation_min_seed_points_ = 50;
    int nonground_dilation_min_seed_cells_ = 10;
    double candidate_voxel_resolution_m_ = 0.5;
    bool export_labeled_txt_ = false;
    std::string export_labeled_txt_dir_;
    mutable std::size_t export_frame_index_ = 0;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tcrlm_tcr_construction");
    TcrConstructionNode node;
    ros::spin();
    return 0;
}
