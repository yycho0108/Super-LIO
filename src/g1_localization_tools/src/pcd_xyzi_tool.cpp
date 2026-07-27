#include <pcl/PCLPointCloud2.h>
#include <pcl/common/io.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

namespace {

bool load_cloud(const std::string &path, pcl::PCLPointCloud2 &blob) {
  const auto extension = std::filesystem::path(path).extension().string();
  if (extension == ".pcd") {
    return pcl::io::loadPCDFile(path, blob) == 0;
  }
  if (extension == ".ply") {
    return pcl::io::loadPLYFile(path, blob) == 0;
  }
  std::cerr << "Unsupported extension: " << extension << '\n';
  return false;
}

void print_fields(const pcl::PCLPointCloud2 &blob) {
  std::cout << "fields:";
  for (const auto &field : blob.fields) {
    std::cout << ' ' << field.name << "(offset=" << field.offset
              << ",datatype=" << static_cast<int>(field.datatype)
              << ",count=" << field.count << ')';
  }
  std::cout << '\n';
}

int inspect(const std::string &path, pcl::PointCloud<pcl::PointXYZ> *xyz_out = nullptr) {
  pcl::PCLPointCloud2 blob;
  if (!load_cloud(path, blob)) {
    std::cerr << "PCL failed to load: " << path << '\n';
    return 2;
  }

  pcl::PointCloud<pcl::PointXYZ> xyz;
  pcl::fromPCLPointCloud2(blob, xyz);
  if (xyz.empty()) {
    std::cerr << "PCL loaded zero points: " << path << '\n';
    return 3;
  }

  float min_x = std::numeric_limits<float>::infinity();
  float min_y = min_x;
  float min_z = min_x;
  float max_x = -min_x;
  float max_y = -min_x;
  float max_z = -min_x;
  std::size_t nonfinite = 0;
  for (const auto &point : xyz) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
      ++nonfinite;
      continue;
    }
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    min_z = std::min(min_z, point.z);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
    max_z = std::max(max_z, point.z);
  }

  std::cout << "PCL load success: " << path << '\n';
  std::cout << "points=" << xyz.size() << " width=" << blob.width
            << " height=" << blob.height << " point_step=" << blob.point_step
            << " row_step=" << blob.row_step << '\n';
  print_fields(blob);
  std::cout << "bounds_min=" << min_x << ',' << min_y << ',' << min_z << '\n';
  std::cout << "bounds_max=" << max_x << ',' << max_y << ',' << max_z << '\n';
  std::cout << "nonfinite=" << nonfinite << '\n';

  if (xyz_out != nullptr) {
    *xyz_out = std::move(xyz);
  }
  return nonfinite == 0 ? 0 : 4;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 3 && std::string(argv[1]) == "inspect") {
    return inspect(argv[2]);
  }
  if (argc != 4 || std::string(argv[1]) != "convert") {
    std::cerr << "Usage:\n  pcd_xyzi_tool inspect INPUT.{pcd,ply}\n"
              << "  pcd_xyzi_tool convert INPUT.{pcd,ply} OUTPUT.pcd\n";
    return 1;
  }

  pcl::PointCloud<pcl::PointXYZ> xyz;
  const int load_result = inspect(argv[2], &xyz);
  if (load_result != 0) {
    return load_result;
  }

  pcl::PointCloud<pcl::PointXYZI> xyzi;
  xyzi.reserve(xyz.size());
  for (const auto &point : xyz) {
    pcl::PointXYZI output;
    output.x = point.x;
    output.y = point.y;
    output.z = point.z;
    output.intensity = 0.0F;
    xyzi.push_back(output);
  }
  xyzi.width = static_cast<std::uint32_t>(xyzi.size());
  xyzi.height = 1;
  xyzi.is_dense = true;

  if (pcl::io::savePCDFileBinary(argv[3], xyzi) != 0) {
    std::cerr << "Failed to save XYZI PCD: " << argv[3] << '\n';
    return 5;
  }

  pcl::PointCloud<pcl::PointXYZI> verified;
  if (pcl::io::loadPCDFile(argv[3], verified) != 0 ||
      verified.size() != xyzi.size()) {
    std::cerr << "Saved file failed PointXYZI reload verification\n";
    return 6;
  }
  std::cout << "PointXYZI conversion success: " << argv[3]
            << " points=" << verified.size() << '\n';
  return 0;
}
