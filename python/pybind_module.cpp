#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "superansac.h"
#include "camera/types.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "termination/types.h"
#include "local_optimization/types.h"
#include "inlier_selectors/types.h"
#include "utils/types.h"
#include "settings.h"

namespace py = pybind11;

// Declaration of the external function
std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateHomography(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    superansac::RANSACSettings &settings_); // The RANSAC settings
    
std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateFundamentalMatrix(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    superansac::RANSACSettings &settings_); // The RANSAC settings

std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateEssentialMatrix(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const Eigen::Matrix3d &kIntrinsicsSource_, // The intrinsic matrix of the source camera
    const Eigen::Matrix3d &kIntrinsicsDestination_, // The intrinsic matrix of the destination camera
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    superansac::RANSACSettings &settings_); // The RANSAC settings
    
std::tuple<Eigen::Matrix4d, std::vector<size_t>, double, size_t> estimateRigidTransform(
    const DataMatrix& kCorrespondences_, // The 3D-3D point correspondences
    const std::vector<double>& kBoundingBoxSizes_, // Bounding box sizes (x1, y1, z1, x2, y2, z2)
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    superansac::RANSACSettings &settings_); // The RANSAC settings
    
std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<size_t>, double, size_t> estimateAbsolutePose(
    const DataMatrix& kCorrespondences_, // The 2D-3D point correspondences
    const superansac::camera::CameraType &kCameraType_, // The type of the camera 
    const std::vector<double>& kCameraParams_, // The camera parameters
    const std::vector<double>& kBoundingBox_, // The bounding box dimensions (image width, image height, X, Y, Z)
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    superansac::RANSACSettings &settings_); // The RANSAC settings

PYBIND11_MODULE(pysuperansac, m) {
    m.doc() = "Python bindings for the RANSAC C++ library using pybind11";

    // Expose the sampler types to Python
    py::enum_<superansac::scoring::ScoringType>(m, "ScoringType")
        .value("RANSAC", superansac::scoring::ScoringType::RANSAC)
        .value("MSAC", superansac::scoring::ScoringType::MSAC)
        .value("MAGSAC", superansac::scoring::ScoringType::MAGSAC)
        .value("MINPRAN", superansac::scoring::ScoringType::MINPRAN)
        .value("ACRANSAC", superansac::scoring::ScoringType::ACRANSAC)
        .value("GAU", superansac::scoring::ScoringType::GAU)
        .value("ML", superansac::scoring::ScoringType::ML)
        .value("Grid", superansac::scoring::ScoringType::GRID)
        .export_values();

    // Expose the sampler types to Python
    py::enum_<superansac::samplers::SamplerType>(m, "SamplerType")
        .value("Uniform", superansac::samplers::SamplerType::Uniform)
        .value("PROSAC", superansac::samplers::SamplerType::PROSAC)
        .value("NAPSAC", superansac::samplers::SamplerType::NAPSAC)
        .value("ProgressiveNAPSAC", superansac::samplers::SamplerType::ProgressiveNAPSAC)
        .value("ImportanceSampler", superansac::samplers::SamplerType::ImportanceSampler)
        .value("ARSampler", superansac::samplers::SamplerType::ARSampler)
        .value("Exhaustive", superansac::samplers::SamplerType::Exhaustive)
        .export_values();

    // Expose the LO types to Python
    py::enum_<superansac::local_optimization::LocalOptimizationType>(m, "LocalOptimizationType")
        .value("Nothing", superansac::local_optimization::LocalOptimizationType::None)
        .value("LSQ", superansac::local_optimization::LocalOptimizationType::LSQ)
        .value("IteratedLSQ", superansac::local_optimization::LocalOptimizationType::IRLS)
        .value("NestedRANSAC", superansac::local_optimization::LocalOptimizationType::NestedRANSAC)
        .value("GCRANSAC", superansac::local_optimization::LocalOptimizationType::GCRANSAC)
        .value("IteratedLMEDS", superansac::local_optimization::LocalOptimizationType::IteratedLMEDS)
        .export_values();

    // Expose the Termination types to Python
    py::enum_<superansac::termination::TerminationType>(m, "TerminationType")
        .value("RANSAC", superansac::termination::TerminationType::RANSAC)
        .export_values();

    // Expose the Inlier Selector types to Python
    py::enum_<superansac::inlier_selector::InlierSelectorType>(m, "InlierSelectorType")
        .value("Nothing", superansac::inlier_selector::InlierSelectorType::None)
        .value("SpacePartitioning", superansac::inlier_selector::InlierSelectorType::SpacePartitioningRANSAC)
        .export_values();

    // Expose the Neighborhood types to Python
    py::enum_<superansac::neighborhood::NeighborhoodType>(m, "NeighborhoodType")
        .value("Grid", superansac::neighborhood::NeighborhoodType::Grid)
        .value("FLANN_KNN", superansac::neighborhood::NeighborhoodType::FLANN_KNN)
        .value("FLANN_Radius", superansac::neighborhood::NeighborhoodType::FLANN_Radius)
        .value("BruteForce", superansac::neighborhood::NeighborhoodType::BruteForce)
        .export_values();

    // Expose the Camera types to Python
    py::enum_<superansac::camera::CameraType>(m, "CameraType")
        .value("SimpleRadial", superansac::camera::CameraType::SimpleRadial)
        .value("SimplePinhole", superansac::camera::CameraType::SimplePinhole)
        .export_values();

    // Expose the AR sampler settings to Python
    py::class_<superansac::ARSamplerSettings>(m, "ARSamplerSettings")
        .def(py::init<>())
        .def_readwrite("estimator_variance", &superansac::ARSamplerSettings::estimatorVariance)
        .def_readwrite("randomness", &superansac::ARSamplerSettings::randomness);

    py::class_<superansac::LocalOptimizationSettings>(m, "LocalOptimizationSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &superansac::LocalOptimizationSettings::maxIterations)
        .def_readwrite("graph_cut_number", &superansac::LocalOptimizationSettings::graphCutNumber)
        .def_readwrite("sample_size_multiplier", &superansac::LocalOptimizationSettings::sampleSizeMultiplier)
        .def_readwrite("spatial_coherence_weight", &superansac::LocalOptimizationSettings::spatialCoherenceWeight);

    py::class_<superansac::NeighborhoodSettings>(m, "NeighborhoodSettings")
        .def(py::init<>())
        .def_readwrite("neighborhood_size", &superansac::NeighborhoodSettings::neighborhoodSize)
        .def_readwrite("neighborhood_grid_density", &superansac::NeighborhoodSettings::neighborhoodGridDensity)
        .def_readwrite("nearest_neighbor_number", &superansac::NeighborhoodSettings::nearestNeighborNumber);

    // Expose the RANSAC settings to Python
    py::class_<superansac::RANSACSettings>(m, "RANSACSettings")
        .def(py::init<>())
        .def_readwrite("min_iterations", &superansac::RANSACSettings::minIterations)
        .def_readwrite("max_iterations", &superansac::RANSACSettings::maxIterations)
        .def_readwrite("inlier_threshold", &superansac::RANSACSettings::inlierThreshold)
        .def_readwrite("confidence", &superansac::RANSACSettings::confidence)
        .def_readwrite("scoring", &superansac::RANSACSettings::scoring)
        .def_readwrite("sampler", &superansac::RANSACSettings::sampler)
        .def_readwrite("neighborhood", &superansac::RANSACSettings::neighborhood)
        .def_readwrite("inlier_selector", &superansac::RANSACSettings::inlierSelector)
        .def_readwrite("local_optimization", &superansac::RANSACSettings::localOptimization)
        .def_readwrite("final_optimization", &superansac::RANSACSettings::finalOptimization)
        .def_readwrite("termination_criterion", &superansac::RANSACSettings::terminationCriterion)
        .def_readwrite("ar_sampler_settings", &superansac::RANSACSettings::arSamplerSettings)
        .def_readwrite("local_optimization_settings", &superansac::RANSACSettings::localOptimizationSettings)
        .def_readwrite("final_optimization_settings", &superansac::RANSACSettings::finalOptimizationSettings)
        .def_readwrite("neighborhood_settings", &superansac::RANSACSettings::neighborhoodSettings);
    
    // Expose the function to Python
    m.def("estimateHomography",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> correspondences,
           const std::vector<double>& image_sizes,
           const std::vector<double>& probabilities,
           superansac::RANSACSettings& config) {
  
            auto buf = correspondences.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Input must be a 2D array");
  
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(
                static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1]);
  
            return estimateHomography(mat, probabilities, image_sizes, config);
        },
        "A function that performs homography estimation from point correspondences.",
        py::arg("correspondences"),
        py::arg("image_sizes"),
        py::arg("probabilities") = std::vector<double>(),
        py::arg("config") = superansac::RANSACSettings());
        
    // Expose the function to Python
    m.def("estimateFundamentalMatrix",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> correspondences,
           const std::vector<double>& image_sizes,
           const std::vector<double>& probabilities,
           superansac::RANSACSettings& config) {
  
            auto buf = correspondences.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Input must be a 2D array");
  
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(
                static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1]);
  
            return estimateFundamentalMatrix(mat, probabilities, image_sizes, config);
        },
        "A function that performs fundamental matrix estimation from point correspondences.",
        py::arg("correspondences"),
        py::arg("image_sizes"),
        py::arg("probabilities") = std::vector<double>(),
        py::arg("config") = superansac::RANSACSettings());
        
    // Expose the function to Python
    m.def("estimateEssentialMatrix",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> correspondences,
           py::array_t<double, py::array::c_style | py::array::forcecast> intrinsics_src,
           py::array_t<double, py::array::c_style | py::array::forcecast> intrinsics_dst,
           const std::vector<double>& image_sizes,
           const std::vector<double>& probabilities,
           superansac::RANSACSettings& config) {
  
            auto buf = correspondences.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Input must be a 2D array");
  
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(
                static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1]);

            auto intr_src_buf = intrinsics_src.request();
            if (intr_src_buf.ndim != 2 || intr_src_buf.shape[0] != 3 || intr_src_buf.shape[1] != 3)
                throw std::runtime_error("intrinsics_src must be a 3x3 matrix");

            auto intr_dst_buf = intrinsics_dst.request();
            if (intr_dst_buf.ndim != 2 || intr_dst_buf.shape[0] != 3 || intr_dst_buf.shape[1] != 3)
                throw std::runtime_error("intrinsics_dst must be a 3x3 matrix");

            Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> intrinsics_src_mat(
                static_cast<double*>(intr_src_buf.ptr));
            Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> intrinsics_dst_mat(
                static_cast<double*>(intr_dst_buf.ptr));
  
            return estimateEssentialMatrix(mat, intrinsics_src_mat, intrinsics_dst_mat, probabilities, image_sizes, config);
        },
        "A function that performs essential matrix estimation from point correspondences.",
        py::arg("correspondences"),
        py::arg("intrinsics_src"),
        py::arg("intrinsics_dst"),
        py::arg("image_sizes"),
        py::arg("probabilities") = std::vector<double>(),
        py::arg("config") = superansac::RANSACSettings());
        
    // Expose the function to Python
    m.def("estimateRigidTransform",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> correspondences,
           const std::vector<double>& bounding_box_sizes,
           const std::vector<double>& probabilities,
           superansac::RANSACSettings& config) {

            auto buf = correspondences.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Input must be a 2D array");

            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(
                                                                                                         static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1]);

            return estimateRigidTransform(mat, bounding_box_sizes, probabilities, config);
        },
        "A function that performs 6D rigid transformation estimation from 3D-3D point correspondences.",
        py::arg("correspondences"),
        py::arg("bounding_box_sizes"),
        py::arg("probabilities") = std::vector<double>(),
        py::arg("config") = superansac::RANSACSettings());
        
    // Expose the function to Python
    m.def("estimateAbsolutePose",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> correspondences,
           const superansac::camera::CameraType& camera_type,
           const std::vector<double>& camera_params,
           const std::vector<double>& bounding_box,
           const std::vector<double>& probabilities,
           superansac::RANSACSettings& config) {
  
            auto buf = correspondences.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Input must be a 2D array");
  
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(
                static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1]);
  
            return estimateAbsolutePose(mat, camera_type, camera_params, bounding_box, probabilities, config);
        },
        "A function that performs absolute camera pose estimation from 2D-3D point correspondences.",
        py::arg("correspondences"),
        py::arg("camera_type"),
        py::arg("camera_params"),
        py::arg("bounding_box"),
        py::arg("probabilities") = std::vector<double>(),
        py::arg("config") = superansac::RANSACSettings());
}
