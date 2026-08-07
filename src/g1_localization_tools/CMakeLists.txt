cmake_minimum_required(VERSION 3.8)
project(g1_localization_tools)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

ament_python_install_package(${PROJECT_NAME})

add_executable(pcd_xyzi_tool src/pcd_xyzi_tool.cpp)
target_compile_features(pcd_xyzi_tool PRIVATE cxx_std_17)
target_include_directories(pcd_xyzi_tool PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcd_xyzi_tool ${PCL_LIBRARIES})
target_compile_definitions(pcd_xyzi_tool PRIVATE ${PCL_DEFINITIONS})

install(TARGETS pcd_xyzi_tool DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  bin/bag_audit
  bin/g1_robot_visualizer
  bin/localization_evaluator
  bin/map_publisher
  bin/policy_odom_bridge
  bin/quantitative_analysis
  bin/reference_visualizer
  bin/trajectory_overlay
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
