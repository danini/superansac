#include <Eigen/Dense>

#include "superansac.h"

#include "estimators/solver_homography_four_point.h"
#include "estimators/estimator_homography.h"

#include "estimators/solver_essential_matrix_five_point_nister.h"
#include "estimators/solver_essential_matrix_bundle_adjustment.h"
#include "estimators/estimator_essential_matrix.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_fundamental_matrix_bundle_adjustment.h"
#include "estimators/estimator_fundamental_matrix.h"

#include "estimators/solver_rigid_transform_proscrutes.h"
#include "estimators/estimator_rigid_transformation.h"

#include "estimators/solver_p3p_lambda_twist.h"
#include "estimators/solver_pnp_bundle_adjustment.h"
#include "estimators/estimator_absolute_pose.h"

#include "estimators/numerical_optimizer/types.h"

#include "superansac.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "local_optimization/types.h"
#include "termination/types.h"
#include "neighborhood/types.h"
#include "inlier_selectors/types.h"
#include "utils/types.h"
#include "utils/utils_point_correspondence.h"
#include "camera/types.h"

// Function to initialize the neighborhood graph
template <size_t _DimensionNumber>
void initializeNeighborhood(
    const DataMatrix& kCorrespondences_, // The point correspondences
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> &neighborhoodGraph_, // The neighborhood graph
    const superansac::neighborhood::NeighborhoodType kNeighborhoodType_, // The type of the neighborhood
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    const superansac::RANSACSettings &kSettings_) // The RANSAC settings
{
    // Create the neighborhood graph
    neighborhoodGraph_ = superansac::neighborhood::createNeighborhoodGraph<_DimensionNumber>(kNeighborhoodType_);
    // Initialize the neighborhood graph if the neighborhood is grid
    if (kNeighborhoodType_ == superansac::neighborhood::NeighborhoodType::Grid) 
    {
        // Check if the image sizes have the correct number of elements
        if (kImageSizes_.size() != _DimensionNumber)
            throw std::invalid_argument("The image sizes must have " + std::to_string(_DimensionNumber) + " elements.");

        // Cast the neighborhood graph to the grid neighborhood graph
        auto gridNeighborhoodGraph = 
            dynamic_cast<superansac::neighborhood::GridNeighborhoodGraph<_DimensionNumber> *>(neighborhoodGraph_.get());
        // Initialize the neighborhood graph
        const auto &kCellNumber = kSettings_.neighborhoodSettings.neighborhoodGridDensity;
        std::vector<double> kCellSizes(_DimensionNumber);
        for (size_t i = 0; i < _DimensionNumber; i++)
        {
            kCellSizes[i] = kImageSizes_[i] / kCellNumber;
            if (kCellSizes[i] < 1.0)
                throw std::invalid_argument("The cell size is too small (< 1px). Try setting a smaller neighborhood size (in grid it acts as the number of cells along an axis).");
        }

        gridNeighborhoodGraph->setCellSizes(
            kCellSizes, // The sizes of the cells in each dimension
            kCellNumber); // The number of cells in each dimension
    } else if (kNeighborhoodType_ == superansac::neighborhood::NeighborhoodType::FLANN_KNN)
    {
        // Cast the neighborhood graph to the FLANN neighborhood graph
        auto flannNeighborhoodGraph = 
            dynamic_cast<superansac::neighborhood::FlannNeighborhoodGraph<_DimensionNumber, 0> *>(neighborhoodGraph_.get());
        // Initialize the neighborhood graph
        flannNeighborhoodGraph->setNearestNeighborNumber(kSettings_.neighborhoodSettings.nearestNeighborNumber); 

    } else if (kNeighborhoodType_ == superansac::neighborhood::NeighborhoodType::FLANN_Radius)
    {
        // Cast the neighborhood graph to the FLANN neighborhood graph
        auto flannNeighborhoodGraph = 
            dynamic_cast<superansac::neighborhood::FlannNeighborhoodGraph<_DimensionNumber, 1> *>(neighborhoodGraph_.get());
        // Initialize the neighborhood graph
        flannNeighborhoodGraph->setRadius(kSettings_.neighborhoodSettings.neighborhoodSize); 

    }
    neighborhoodGraph_->initialize(&kCorrespondences_);
}

template <size_t _DimensionNumber>
void initializeLocalOptimizer(
    const DataMatrix& kCorrespondences_, // The point correspondences
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> &localOptimizer_,
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> &neighborhoodGraph_, // The neighborhood graph
    const superansac::neighborhood::NeighborhoodType kNeighborhoodType_, // The type of the neighborhood
    const superansac::local_optimization::LocalOptimizationType kLocalOptimizationType_, // The type of the neighborhood
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    const superansac::RANSACSettings &kSettings_, // The RANSAC settings
    const superansac::LocalOptimizationSettings &kLOSettings_, // The RANSAC settings
    const superansac::models::Types &kModelType_,
    const bool kFinalOptimization_ = false) 
{
    if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::None)
        return;

    if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::GCRANSAC)
    {
        // Initialize the neighborhood graph if needed
        if (neighborhoodGraph_ == nullptr)
            initializeNeighborhood<_DimensionNumber>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph_, // The neighborhood graph
                kNeighborhoodType_, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                kSettings_); // The RANSAC settings
        // Set the neighborhood graph to the local optimizer
        auto gcransacLocalOptimizer = dynamic_cast<superansac::local_optimization::GraphCutRANSACOptimizer *>(localOptimizer_.get());
        gcransacLocalOptimizer->setNeighborhood(neighborhoodGraph_.get());
        gcransacLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations);
        gcransacLocalOptimizer->setGraphCutNumber(kLOSettings_.graphCutNumber);
        gcransacLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier);
        gcransacLocalOptimizer->setSpatialCoherenceWeight(kLOSettings_.spatialCoherenceWeight);
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::IRLS)
    {
        // Set the neighborhood graph to the local optimizer
        auto irlsLocalOptimizer = dynamic_cast<superansac::local_optimization::IRLSOptimizer *>(localOptimizer_.get());
        irlsLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations);       
        if (kFinalOptimization_ || 
            kModelType_ == superansac::models::Types::Homography || 
            kModelType_ == superansac::models::Types::RigidTransformation || 
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix)
            irlsLocalOptimizer->setUseInliers(true); 
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::LSQ)
    {
        // Set the neighborhood graph to the local optimizer
        auto lsqLocalOptimizer = dynamic_cast<superansac::local_optimization::LeastSquaresOptimizer *>(localOptimizer_.get());
        if (kFinalOptimization_ || 
            kModelType_ == superansac::models::Types::Homography || 
            kModelType_ == superansac::models::Types::RigidTransformation || 
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix)
            lsqLocalOptimizer->setUseInliers(true);
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::NestedRANSAC)
    {
        // Set the neighborhood graph to the local optimizer
        auto nestedRansacLocalOptimizer = dynamic_cast<superansac::local_optimization::NestedRANSACOptimizer *>(localOptimizer_.get());
        nestedRansacLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations); 
        nestedRansacLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier); 
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::IteratedLMEDS)
    {
        auto iteratedLMEDSLocalOptimizer = dynamic_cast<superansac::local_optimization::IteratedLMEDSOptimizer *>(localOptimizer_.get());
        iteratedLMEDSLocalOptimizer->setModelType(kModelType_); 
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::CrossValidation)
    {
        auto crossValidationLocalOptimizer = dynamic_cast<superansac::local_optimization::CrossValidationOptimizer *>(localOptimizer_.get());
        if (kFinalOptimization_ || 
            kModelType_ == superansac::models::Types::Homography || 
            kModelType_ == superansac::models::Types::RigidTransformation || 
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix)
            crossValidationLocalOptimizer->setUseInliers(true);
        
        //crossValidationLocalOptimizer->setRepetitions(kLOSettings_.maxIterations);
        //crossValidationLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier);
    } 
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<size_t>, double, size_t> estimateAbsolutePose(
    const DataMatrix& kCorrespondences_, // The 2D-3D point correspondences
    const superansac::camera::CameraType &kCameraType_, // The type of the camera 
    const std::vector<double>& kCameraParams_, // The camera parameters
    const std::vector<double>& kBoundingBox_, // The bounding box dimensions (image width, image height, X, Y, Z)
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 5) 
        throw std::invalid_argument("The input matrix must have 5 columns (x1, y1, x2, y2, z2).");
    if (kCorrespondences_.rows() < 3) 
        throw std::invalid_argument("The input matrix must have at least 3 rows.");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");

    std::unique_ptr<superansac::camera::AbstractCamera> currentCamera =
        superansac::camera::createCamera(kCameraType_, kCameraParams_);
    std::unique_ptr<superansac::camera::AbstractCamera> onlyScaledCamera =
        superansac::camera::createIdentityCamera(kCameraType_);
    onlyScaledCamera->rescale(currentCamera->focalLength());

    // Normalize the point correspondences
    DataMatrix normalizedCorrespondences;
    currentCamera->fromPixelToImageCoordinates(kCorrespondences_, normalizedCorrespondences);
        
    // Normalize the threshold
    settings_.inlierThreshold = 
        currentCamera->normalizeThreshold(settings_.inlierThreshold);

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::AbsolutePoseEstimator> estimator = 
        std::unique_ptr<superansac::estimator::AbsolutePoseEstimator>(new superansac::estimator::AbsolutePoseEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::P3PLambdaTwistSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::PnPBundleAdjustmentSolver());
    dynamic_cast<superansac::estimator::solver::PnPBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver())->setCamera(onlyScaledCamera.get());
    auto &solverOptions = dynamic_cast<superansac::estimator::solver::PnPBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver())->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    solverOptions.loss_scale = settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<5>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<5> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kBoundingBox_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<5>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kBoundingBox_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<5>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
    } else if (kScoring == superansac::scoring::ScoringType::ACRANSAC) // Initialize the scoring object if the scoring is ACRANSAC
        throw std::invalid_argument("The ACRANSAC scoring is not implemented for the absolute pose estimation.");

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector = 
            superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<5>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kBoundingBox_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::AbsolutePose,
            false);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<5>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kBoundingBox_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::AbsolutePose,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // Run the robust estimator
    robustEstimator.run(normalizedCorrespondences);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix3d rotation = robustEstimator.getBestModel().getData().block<3, 3>(0, 0).eval();
    Eigen::Vector3d translation = robustEstimator.getBestModel().getData().block<3, 1>(0, 3).eval();

    // Return the best model with the inliers and the score
    return std::make_tuple(rotation, 
        translation,
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());
}

std::tuple<Eigen::Matrix4d, std::vector<size_t>, double, size_t> estimateRigidTransform(
    const DataMatrix& kCorrespondences_, // The 3D-3D point correspondences
    const std::vector<double>& kBoundingBoxSizes_, // Bounding box sizes (x1, y1, z1, x2, y2, z2)
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 6) 
        throw std::invalid_argument("The input matrix must have 6 columns (x1, y1, z1, x2, y2, z2).");
    if (kCorrespondences_.rows() < 3) 
        throw std::invalid_argument("The input matrix must have at least 3 rows.");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");

    // Normalize the point correspondences
    /*DataMatrix normalizedCorrespondences;
    Eigen::Matrix4d normalizingTransformSource, normalizingTransformDestination;
    normalize3D3DPointCorrespondences(
        kCorrespondences_,
        normalizedCorrespondences,
        normalizingTransformSource,
        normalizingTransformDestination);   

    std::cout << normalizingTransformSource << std::endl;
    std::cout << normalizingTransformDestination << std::endl;
        
    const double kScale = 
        0.5 * (normalizingTransformSource(0, 0) + normalizingTransformDestination(0, 0));
    settings_.inlierThreshold *= kScale;*/

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::RigidTransformationEstimator> estimator = 
        std::unique_ptr<superansac::estimator::RigidTransformationEstimator>(new superansac::estimator::RigidTransformationEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::RigidTransformProscrutesSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::RigidTransformProscrutesSolver());

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<6>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<6> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kBoundingBoxSizes_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<6>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kBoundingBoxSizes_, // Bounding box sizes (x1, y1, z1, x2, y2, z2)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<6>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::GRID) // Set the neighborhood structure if the scoring is GRID
    {
        // Check whether the neighborhood graph is already initialized
        superansac::neighborhood::GridNeighborhoodGraph<6> *gridNeighborhoodGraph;
        if (neighborhoodGraph == nullptr)
            // Initialize the neighborhood graph
            initializeNeighborhood<6>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kBoundingBoxSizes_, // Bounding box sizes (x1, y1, z1, x2, y2, z2)
                settings_); // The RANSAC settings
        else if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid) // Check whether the provided neighborhood type is grid
            throw std::invalid_argument("The neighborhood graph is already initialized, but the neighborhood type is not grid.");
        // Set the neighborhood graph
        dynamic_cast<superansac::scoring::GridScoring<6> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
    } else if (kScoring == superansac::scoring::ScoringType::ACRANSAC) // Initialize the scoring object if the scoring is ACRANSAC
        throw std::invalid_argument("The scoring type is not supported.");

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector = 
            superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<6>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kBoundingBoxSizes_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::RigidTransformation,
            false);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<6>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kBoundingBoxSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::RigidTransformation,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // Run the robust estimator
    robustEstimator.run(kCorrespondences_);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix4d::Identity(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix4d rigidTransform = robustEstimator.getBestModel().getData();

    // Transform the estimated fundamental matrix back to the not normalized space
    //rigidTransform = normalizingTransformDestination.transpose() * rigidTransform * normalizingTransformSource;

    // Return the best model with the inliers and the score
    return std::make_tuple(rigidTransform, 
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());    
}

std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateFundamentalMatrix(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 4) 
        throw std::invalid_argument("The input matrix must have 4 columns (x1, y1, x2, y2).");
    if (kCorrespondences_.rows() < 8) 
        throw std::invalid_argument("The input matrix must have at least 8 rows.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");
    if (kPointProbabilities_.size() > 0 && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have either the same number of elements as the number of correspondences or none.");

    // Normalize the point correspondences
    DataMatrix normalizedCorrespondences;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    normalize2D2DPointCorrespondences(
        kCorrespondences_,
        normalizedCorrespondences,
        normalizingTransformSource,
        normalizingTransformDestination);   
        
    const double kScale = 
        0.5 * (normalizingTransformSource(0, 0) + normalizingTransformDestination(0, 0));
    settings_.inlierThreshold *= kScale;

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::FundamentalMatrixEstimator> estimator = 
        std::unique_ptr<superansac::estimator::FundamentalMatrixEstimator>(new superansac::estimator::FundamentalMatrixEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::FundamentalMatrixSevenPointSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver());
    superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver * nonminimalSolver = 
        dynamic_cast<superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver());
    if (kPointProbabilities_.size() > 0)
        nonminimalSolver->setWeights(&kPointProbabilities_);
    auto &solverOptions = nonminimalSolver->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    solverOptions.loss_scale = settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<4>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<4> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kImageSizes_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<4>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<4>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
       scorer->setImageSize(kImageSizes_[0], kImageSizes_[1], kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::GRID) // Set the neighborhood structure if the scoring is GRID
    {
        // Check whether the neighborhood graph is already initialized
        superansac::neighborhood::GridNeighborhoodGraph<4> *gridNeighborhoodGraph;
        if (neighborhoodGraph == nullptr)
            // Initialize the neighborhood graph
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
        else if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid) // Check whether the provided neighborhood type is grid
            throw std::invalid_argument("The neighborhood graph is already initialized, but the neighborhood type is not grid.");
        // Set the neighborhood graph
        dynamic_cast<superansac::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
    }

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector = 
            superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Create the space partitioning inlier selector object
    std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector;
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    { 
        // Check whether the scoring is RANSAC, MSAC, or MAGSAC++
        if (kScoring != superansac::scoring::ScoringType::RANSAC &&
            kScoring != superansac::scoring::ScoringType::MSAC &&
            kScoring != superansac::scoring::ScoringType::MAGSAC)
            throw std::invalid_argument("The space partitioning inlier selector can only be used with RANSAC, MSAC, or MAGSAC++ scoring.");

        // Check if the neighborhood is grid
        if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid)
            throw std::invalid_argument("The space partitioning inlier selector can only be used with grid neighborhood.");

        // Initialize the neighborhood graph if needed
        if (neighborhoodGraph == nullptr)
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
                
        // Create the inlier selector
        inlierSelector = 
            superansac::inlier_selector::createInlierSelector(superansac::inlier_selector::InlierSelectorType::SpacePartitioningRANSAC);
        // Initialize the inlier selector
        superansac::inlier_selector::SpacePartitioningRANSAC *spacePartitioningRANSAC = 
            reinterpret_cast<superansac::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get());
        spacePartitioningRANSAC->initialize(
            neighborhoodGraph.get(), 
            superansac::models::Types::FundamentalMatrix);
        spacePartitioningRANSAC->setNormalizers(
            normalizingTransformSource(0, 0), normalizingTransformSource(0, 2), normalizingTransformSource(1, 2),
            normalizingTransformDestination(0, 0), normalizingTransformDestination(0, 2), normalizingTransformDestination(1, 2));
        // Set the inlier selector to the robust estimator
        robustEstimator.setInlierSelector(inlierSelector.get());
    }

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::FundamentalMatrix,
            false);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::FundamentalMatrix,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // Run the robust estimator
    robustEstimator.run(normalizedCorrespondences);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix3d fundamentalMatrix = robustEstimator.getBestModel().getData();

    // Transform the estimated fundamental matrix back to the not normalized space
    fundamentalMatrix = normalizingTransformDestination.transpose() * fundamentalMatrix * normalizingTransformSource;
    fundamentalMatrix.normalize();

    // Return the best model with the inliers and the score
    return std::make_tuple(fundamentalMatrix, 
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());
}

std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateEssentialMatrix(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const Eigen::Matrix3d &kIntrinsicsSource_, // The intrinsic matrix of the source camera
    const Eigen::Matrix3d &kIntrinsicsDestination_, // The intrinsic matrix of the destination camera
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 4) 
        throw std::invalid_argument("The input matrix must have 4 columns (x1, y1, x2, y2).");
    if (kCorrespondences_.rows() < 6) 
        throw std::invalid_argument("The input matrix must have at least 6 rows.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");
    if (kPointProbabilities_.size() > 0 && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have either the same number of elements as the number of correspondences or none.");

    // Normalize the point correspondences
    DataMatrix normalizedCorrespondences;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    normalizePointsByIntrinsics(
        kCorrespondences_,
        kIntrinsicsSource_,
        kIntrinsicsDestination_,
        normalizedCorrespondences);
        
    const double kScale = 
        0.25 * (kIntrinsicsSource_(0, 0) + kIntrinsicsSource_(1, 1) + kIntrinsicsDestination_(0, 0) + kIntrinsicsDestination_(1, 1));
    settings_.inlierThreshold /= kScale;

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::EssentialMatrixEstimator> estimator = 
        std::unique_ptr<superansac::estimator::EssentialMatrixEstimator>(new superansac::estimator::EssentialMatrixEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::EssentialMatrixFivePointNisterSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::EssentialMatrixBundleAdjustmentSolver());
    superansac::estimator::solver::EssentialMatrixBundleAdjustmentSolver * nonminimalSolver = 
        dynamic_cast<superansac::estimator::solver::EssentialMatrixBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver());
    if (kPointProbabilities_.size() > 0)
        nonminimalSolver->setWeights(&kPointProbabilities_);
    auto &solverOptions = nonminimalSolver->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    solverOptions.loss_scale = settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<4>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<4> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kImageSizes_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<4>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<4>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
       scorer->setImageSize(kImageSizes_[0], kImageSizes_[1], kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::GRID) // Set the neighborhood structure if the scoring is GRID
    {
        // Check whether the neighborhood graph is already initialized
        superansac::neighborhood::GridNeighborhoodGraph<4> *gridNeighborhoodGraph;
        if (neighborhoodGraph == nullptr)
            // Initialize the neighborhood graph
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
        else if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid) // Check whether the provided neighborhood type is grid
            throw std::invalid_argument("The neighborhood graph is already initialized, but the neighborhood type is not grid.");
        // Set the neighborhood graph
        dynamic_cast<superansac::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
    }

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector = 
            superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::EssentialMatrix,
            false);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::EssentialMatrix,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // Run the robust estimator
    robustEstimator.run(normalizedCorrespondences);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix3d essentialMatrix = robustEstimator.getBestModel().getData();

    // Return the best model with the inliers and the score
    return std::make_tuple(essentialMatrix, 
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());
}

std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateHomography(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 4) 
        throw std::invalid_argument("The input matrix must have 4 columns (x1, y1, x2, y2).");
    if (kCorrespondences_.rows() < 4) 
        throw std::invalid_argument("The input matrix must have at least 4 rows.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::HomographyEstimator> estimator = 
        std::unique_ptr<superansac::estimator::HomographyEstimator>(new superansac::estimator::HomographyEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::HomographyFourPointSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::HomographyFourPointSolver());

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<4>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<4> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kImageSizes_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<4>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<4>(kScoring);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
       scorer->setImageSize(kImageSizes_[0], kImageSizes_[1], kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::GRID) // Set the neighborhood structure if the scoring is GRID
    {
        // Check whether the neighborhood graph is already initialized
        superansac::neighborhood::GridNeighborhoodGraph<4> *gridNeighborhoodGraph;
        if (neighborhoodGraph == nullptr)
            // Initialize the neighborhood graph
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
        else if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid) // Check whether the provided neighborhood type is grid
            throw std::invalid_argument("The neighborhood graph is already initialized, but the neighborhood type is not grid.");
        // Set the neighborhood graph
        dynamic_cast<superansac::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
        dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector = 
            superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::Homography);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::Homography);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // Run the robust estimator
    robustEstimator.run(kCorrespondences_);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Return the best model with the inliers and the score
    return std::make_tuple(robustEstimator.getBestModel().getData(), 
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());
}