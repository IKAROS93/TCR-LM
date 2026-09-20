# TCR-LM

**TCR-LM: Terrain Confidence Region-Guided LiDAR Mapping for Open-Pit Mine Loading Operations**

TCR-LM is a ROS 1 LiDAR mapping system designed for open-pit mine shoveling and loading operations. It uses relatively stable terrain observations as the primary constraints for pose estimation and integrates terrain confidence region construction, motion compensation, spatially balanced feature selection, scan-to-map optimization, and bidirectional terrain-change maintenance. These components improve mapping stability under dynamic occlusion, dust interference, equipment vibration, and local terrain changes.

Project owner and maintainer: [Ziyu Zhao](mailto:ikaros.yuki.93@gmail.com)

## Validated Environment

The repository has been built, unit-tested, and validated through end-to-end ROS bag playback in the following software environment:

| Component | Version |
|---|---|
| Operating system | Ubuntu 20.04.3 LTS |
| ROS | Noetic |
| GCC | 9.4.0 |
| CMake | 4.2.3 |
| PCL | 1.10.0 |
| OpenCV | 4.2.0 |
| GTSAM | 4.0.3 |
| Boost | 1.71.0 |
| C++ standard | C++14 |

## Build and Test

Install GTSAM, PCL, OpenCV, and Boost:

```bash
sudo apt update
sudo apt install libgtsam-dev libpcl-dev libopencv-dev libboost-filesystem-dev
```

From the repository root, install the remaining ROS dependencies and build the catkin workspace:

```bash
cd /path/to/TCR-LM
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys GTSAM
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -j2
source devel/setup.bash
```

Run the unit tests:

```bash
catkin_make run_tests_tcrlm -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -j2
catkin_test_results --verbose
```

A successful test run should report all six terrain-change logic tests as passed, with no errors or failures from `catkin_test_results`.

## Dataset

The accompanying data constitute a CARLA-based simulation dataset for open-pit mine shoveling and loading operations. The dataset supports the mapping comparisons under complex operating conditions, terrain-change and keyframe-map maintenance evaluation, and continuous-mapping experiments under degraded GNSS availability reported in the paper.

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1iC8Vz8RGOFY0eBW7DhEvG9BPspIIVBCB?usp=drive_link).

After downloading and extracting the dataset, place the six ROS bag files directly in `src/tcrlm/datasets/`:

```text
src/tcrlm/datasets/
├── compare_bag1.bag
├── compare_bag2.bag
├── compare_bag3.bag
├── compare_bag4.bag
├── update_bag1.bag
└── gnss_lost_bag1.bag
```

| File | Duration | LiDAR frames | Experimental purpose |
|---|---:|---:|---|
| `compare_bag1.bag` | 37.14 s | 371 | Mapping comparison under shovel slewing, dust interference, and vibration |
| `compare_bag2.bag` | 32.05 s | 320 | Mapping comparison as a haul truck departs the operating area after loading |
| `compare_bag3.bag` | 33.02 s | 331 | Mapping comparison as a haul truck reverses into the loading area and parks |
| `compare_bag4.bag` | 60.98 s | 610 | Comprehensive mapping comparison under multiple coupled disturbances |
| `update_bag1.bag` | 18.34 s | 183 | Evaluation of terrain-change detection and keyframe-map maintenance |
| `gnss_lost_bag1.bag` | 73.78 s | 738 | Continuous-mapping evaluation during a long detour with degraded GNSS availability |

## Quick Start

From the repository root, source ROS and the current catkin workspace:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

### 1. Mapping Comparison Sequences

```bash
roslaunch tcrlm tcrlm_compare.launch bag_id:=1
```

Set `bag_id` to `1`, `2`, `3`, or `4` to run `compare_bag1.bag` through `compare_bag4.bag`, respectively. This launch enables terrain confidence region construction and disables keyframe terrain-change maintenance by default.

### 2. Terrain-Change and Map-Maintenance Sequence

```bash
roslaunch tcrlm tcrlm_keyframe_update.launch
```

This launch runs `update_bag1.bag` by default and enables terrain-change detection and keyframe-map maintenance.

### 3. Long-Distance Mapping without GNSS

```bash
roslaunch tcrlm tcrlm_gnss_lost.launch
```

This launch runs `gnss_lost_bag1.bag` by default. The default configuration does not add GNSS position or heading constraints to the factor graph and keeps terrain-change detection disabled, allowing continuous mapping without GNSS assistance to be evaluated.

Each launch file automatically loads the corresponding parameters and starts the required TCR-LM nodes, RViz, and ROS bag player.

## Citation

If you use this project in your research, please cite:

> Ziyu Zhao et al 2026 *Meas. Sci. Technol.* in press. https://doi.org/10.1088/1361-6501/aea88d

```bibtex
@article{zhao2026tcrlm,
  title   = {{TCR-LM}: Terrain Confidence Region-Guided {LiDAR} Mapping for Open-Pit Mine Loading Operations},
  author  = {Zhao, Ziyu and Hong, Xiang and Wang, Zhuo and Bi, Lin},
  journal = {Measurement Science and Technology},
  year    = {2026},
  note    = {In press},
  doi     = {10.1088/1361-6501/aea88d},
  url     = {https://doi.org/10.1088/1361-6501/aea88d}
}
```
