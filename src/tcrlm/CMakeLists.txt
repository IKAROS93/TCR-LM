cmake_minimum_required(VERSION 3.0.2)
project(tcrlm)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  pcl_conversions
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
)

find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
set(_tcrlm_multiarch_cmake "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake")
if(EXISTS "/usr/lib/cmake/eigen3/Eigen3Config.cmake" AND EXISTS "/usr/include/eigen3/Eigen/Core")
  set(Eigen3_DIR "/usr/lib/cmake/eigen3" CACHE PATH "Eigen config used by TCR-LM" FORCE)
  set(EIGEN_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Eigen headers used by TCR-LM" FORCE)
endif()
if(EXISTS "${_tcrlm_multiarch_cmake}/pcl/PCLConfig.cmake")
  set(PCL_DIR "${_tcrlm_multiarch_cmake}/pcl" CACHE PATH "PCL config used by TCR-LM" FORCE)
endif()
if(EXISTS "${_tcrlm_multiarch_cmake}/opencv4/OpenCVConfig.cmake")
  set(OpenCV_DIR "${_tcrlm_multiarch_cmake}/opencv4" CACHE PATH "OpenCV config used by TCR-LM" FORCE)
endif()
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
  TerrainChangePoint.msg
  TerrainChangeSnapshot.msg
  TerrainChangeUpdate.msg
  TerrainChangeUpdateArray.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL GTSAM Boost

  CATKIN_DEPENDS
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
  visualization_msgs
)
include_directories(
	include
  ${GTSAM_INCLUDE_DIR}
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)
link_directories(
	include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)
add_executable(${PROJECT_NAME}_range_image_projection_and_deskew src/range_image_projection_and_deskew.cpp)
add_dependencies(${PROJECT_NAME}_range_image_projection_and_deskew ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_range_image_projection_and_deskew ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
add_executable(${PROJECT_NAME}_feature_extraction src/feature_extraction.cpp)
add_dependencies(${PROJECT_NAME}_feature_extraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_feature_extraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
add_executable(${PROJECT_NAME}_mapping src/mapping.cpp)
add_dependencies(${PROJECT_NAME}_mapping ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mapping PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam)
add_executable(${PROJECT_NAME}_terrain_change src/terrain_change.cpp)
add_dependencies(${PROJECT_NAME}_terrain_change ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_terrain_change ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_executable(${PROJECT_NAME}_tcr_construction src/tcr_construction.cpp)
target_link_libraries(${PROJECT_NAME}_tcr_construction ${catkin_LIBRARIES} ${PCL_LIBRARIES})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_terrain_change_logic test/test_terrain_change_logic.cpp)
endif()

install(TARGETS
  ${PROJECT_NAME}_tcr_construction
  ${PROJECT_NAME}_range_image_projection_and_deskew
  ${PROJECT_NAME}_feature_extraction
  ${PROJECT_NAME}_mapping
  ${PROJECT_NAME}_terrain_change
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY config launch rviz datasets DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES include/utility.h DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
