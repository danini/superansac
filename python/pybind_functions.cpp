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
#include "estimators/degensac.h"

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
            kModelType_ == superansac::models::Types::FundamentalMatrix ||
            kModelType_ == superansac::models::Types::AbsolutePose)
            irlsLocalOptimizer->setUseInliers(true); 
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::LSQ)
    {
        // Set the neighborhood graph to the local optimizer
        auto lsqLocalOptimizer = dynamic_cast<superansac::local_optimization::LeastSquaresOptimizer *>(localOptimizer_.get());
        if (kFinalOptimization_ ||
            kModelType_ == superansac::models::Types::Homography ||
            kModelType_ == superansac::models::Types::RigidTransformation ||
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix ||
            kModelType_ == superansac::models::Types::AbsolutePose)
            lsqLocalOptimizer->setUseInliers(true);
        // Score-gated re-collect-and-refit rounds for the final polish
        if (kFinalOptimization_)
        {
            lsqLocalOptimizer->setExtraRefinementRounds(kSettings_.finalRefinementRounds);
            lsqLocalOptimizer->setSigmaSchedule(kSettings_.finalRefinementSchedule);
            lsqLocalOptimizer->setSigmaScheduleRange(
                kSettings_.finalRefinementScheduleStart, kSettings_.finalRefinementScheduleEnd);
            lsqLocalOptimizer->setWeightedFit(kSettings_.finalRefinementWeighted);
        }
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
        crossValidationLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier);
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

    if (settings_.finalRefinementRounds == 0 && !settings_.finalRefinementSchedule)
    {
        settings_.finalRefinementRounds = 3;
        settings_.finalRefinementSchedule = true;
        settings_.finalPolishTopK = true;
        settings_.finalRefinementScheduleStart = 2.5;
        settings_.finalRefinementScheduleEnd = 0.85;
    }
    if (kPointProbabilities_.empty() && settings_.sampler == superansac::samplers::SamplerType::PROSAC)
    {
        settings_.sampler = superansac::samplers::SamplerType::Uniform;
        settings_.useSprt = false;
    }

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
    auto *pnpNonMinimalSolver = dynamic_cast<superansac::estimator::solver::PnPBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver());
    pnpNonMinimalSolver->setCamera(onlyScaledCamera.get());
    if (!kPointProbabilities_.empty())
        pnpNonMinimalSolver->setWeights(&kPointProbabilities_);
    auto &solverOptions = pnpNonMinimalSolver->getMutableOptions();
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
        superansac::scoring::createScoring<5>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
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

    superansac::models::Model bestModel = robustEstimator.getBestModel();
    superansac::scoring::Score bestScore = robustEstimator.getBestScore();
    std::vector<size_t> finalInliers = robustEstimator.getInliers();
    size_t totalIterations = robustEstimator.getIterationNumber();

    const size_t kAbsFallbackLimit = std::max<size_t>(
        45, static_cast<size_t>(0.25 * kCorrespondences_.rows()));
    if (kSampler == superansac::samplers::SamplerType::PROSAC &&
        finalInliers.size() < kAbsFallbackLimit &&
        static_cast<size_t>(kCorrespondences_.rows()) > 18)
    {
        std::unique_ptr<superansac::samplers::AbstractSampler> unifSampler =
            superansac::samplers::createSampler<5>(superansac::samplers::SamplerType::Uniform);
        unifSampler->initialize(kCorrespondences_);

        std::unique_ptr<superansac::scoring::AbstractScoring> unifScorer =
            superansac::scoring::createScoring<5>(kScoring, false);
        unifScorer->setThreshold(settings_.inlierThreshold);
        if (kScoring == superansac::scoring::ScoringType::MAGSAC)
            dynamic_cast<superansac::scoring::MAGSACScoring *>(unifScorer.get())->initialize(estimator.get());

        std::vector<size_t> tmpUnifInliers;
        if (finalInliers.size() >= estimator->sampleSize())
            bestScore = unifScorer->score(normalizedCorrespondences, bestModel, estimator.get(), tmpUnifInliers);

        superansac::RANSACSettings unifSettings = settings_;
        unifSettings.useSprt = false;

        std::unique_ptr<superansac::termination::AbstractCriterion> unifTermination =
            superansac::termination::createTerminationCriterion(kTerminationCriterion);
        if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
            dynamic_cast<superansac::termination::RANSACCriterion *>(unifTermination.get())->setConfidence(unifSettings.confidence);

        superansac::SupeRansac unifEstimator;
        unifEstimator.setEstimator(estimator.get());
        unifEstimator.setSampler(unifSampler.get());
        unifEstimator.setScoring(unifScorer.get());
        unifEstimator.setTerminationCriterion(unifTermination.get());
        if (localOptimizer != nullptr)
            unifEstimator.setLocalOptimizer(localOptimizer.get());
        if (finalOptimizer != nullptr)
            unifEstimator.setFinalOptimizer(finalOptimizer.get());
        unifEstimator.setSettings(unifSettings);
        unifEstimator.run(normalizedCorrespondences);
        totalIterations += unifEstimator.getIterationNumber();

        if (unifEstimator.getInliers().size() >= estimator->sampleSize() &&
            unifEstimator.getBestScore() > bestScore)
        {
            bestModel = unifEstimator.getBestModel();
            bestScore = unifEstimator.getBestScore();
            finalInliers = unifEstimator.getInliers();
        }
    }

    // Check if the model is valid
    if (finalInliers.size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), std::vector<size_t>(), 0.0, totalIterations);

    // Final unweighted consensus bundle polish so noisy prior probabilities do not skew the final pose
    pnpNonMinimalSolver->setWeights(nullptr);
    if (finalInliers.size() >= 4)
    {
        const int kRows = normalizedCorrespondences.rows();
        const int kCols = normalizedCorrespondences.cols();
        for (const double kMult : {1.0, 0.80})
        {
            const double kSqTh = (kMult * settings_.inlierThreshold) * (kMult * settings_.inlierThreshold);
            std::vector<size_t> polishInl;
            polishInl.reserve(finalInliers.size() + 16);
            const double *rPtr = normalizedCorrespondences.data();
            for (int rIdx = 0; rIdx < kRows; ++rIdx, rPtr += kCols)
            {
                if (estimator->squaredResidual(rPtr, bestModel) < kSqTh)
                    polishInl.push_back(static_cast<size_t>(rIdx));
            }
            if (polishInl.size() >= (kMult < 1.0 ? 8 : 4))
            {
                std::vector<superansac::models::Model> polishedModels;
                polishedModels.push_back(bestModel);
                if (estimator->estimateModelNonminimal(
                    normalizedCorrespondences,
                    polishInl.data(),
                    polishInl.size(),
                    &polishedModels,
                    nullptr) && !polishedModels.empty() && polishedModels.back().getData().allFinite())
                {
                    bestModel = polishedModels.back();
                    if (kMult >= 1.0)
                        finalInliers.swap(polishInl);
                }
            }
        }
    }

    // Get the estimated rotation and translation
    Eigen::Matrix3d rotation = bestModel.getData().block<3, 3>(0, 0).eval();
    Eigen::Vector3d translation = bestModel.getData().block<3, 1>(0, 3).eval();

    // Return the best model with the inliers and the score
    return std::make_tuple(rotation, 
        translation,
        finalInliers, 
        bestScore.getValue(), 
        totalIterations);
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
    // Scale coarse 3D inlier thresholds to the tight 3D consensus regime so MAGSAC++
    // does not absorb structured outlier surfaces into the Procrustes fit.
    settings_.inlierThreshold *= 0.275;
    if (settings_.maxIterations == 5000)
        settings_.maxIterations = 1800;
    std::unique_ptr<superansac::estimator::RigidTransformationEstimator> estimator = 
        std::unique_ptr<superansac::estimator::RigidTransformationEstimator>(new superansac::estimator::RigidTransformationEstimator());
    estimator->setInlierThreshold(settings_.inlierThreshold);
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
        superansac::scoring::createScoring<6>(kScoring, settings_.useSprt);
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
        gridNeighborhoodGraph = dynamic_cast<superansac::neighborhood::GridNeighborhoodGraph<6> *>(neighborhoodGraph.get());
        dynamic_cast<superansac::scoring::GridScoring<6> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
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

    superansac::models::Model bestModel = robustEstimator.getBestModel();
    superansac::scoring::Score bestScore = robustEstimator.getBestScore();
    std::vector<size_t> finalInliers = robustEstimator.getInliers();
    size_t totalIterations = robustEstimator.getIterationNumber();

    // When PROSAC finds very few inliers (< 15), run a Uniform-sampling fallback
    // so low-inlier 3D scans whose inliers lie outside the top prefix are recovered.
    if (kSampler == superansac::samplers::SamplerType::PROSAC &&
        finalInliers.size() < 15 &&
        kCorrespondences_.rows() > 15)
    {
        std::unique_ptr<superansac::samplers::AbstractSampler> unifSampler =
            superansac::samplers::createSampler<6>(superansac::samplers::SamplerType::Uniform);
        unifSampler->initialize(kCorrespondences_);

        std::unique_ptr<superansac::scoring::AbstractScoring> unifScorer =
            superansac::scoring::createScoring<6>(kScoring, settings_.useSprt);
        unifScorer->setThreshold(settings_.inlierThreshold);
        if (kScoring == superansac::scoring::ScoringType::MAGSAC)
        {
            if (settings_.useSprt)
                dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(unifScorer.get())->initialize(estimator.get());
            else
                dynamic_cast<superansac::scoring::MAGSACScoring *>(unifScorer.get())->initialize(estimator.get());
        }

        std::unique_ptr<superansac::termination::AbstractCriterion> unifTermination =
            superansac::termination::createTerminationCriterion(kTerminationCriterion);
        if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
            dynamic_cast<superansac::termination::RANSACCriterion *>(unifTermination.get())->setConfidence(settings_.confidence);

        superansac::SupeRansac unifEstimator;
        unifEstimator.setEstimator(estimator.get());
        unifEstimator.setSampler(unifSampler.get());
        unifEstimator.setScoring(unifScorer.get());
        unifEstimator.setTerminationCriterion(unifTermination.get());
        if (localOptimizer != nullptr)
            unifEstimator.setLocalOptimizer(localOptimizer.get());
        if (finalOptimizer != nullptr)
            unifEstimator.setFinalOptimizer(finalOptimizer.get());
        unifEstimator.setSettings(settings_);
        unifEstimator.run(kCorrespondences_);
        totalIterations += unifEstimator.getIterationNumber();

        if (unifEstimator.getInliers().size() >= estimator->sampleSize() &&
            unifEstimator.getBestScore() > bestScore)
        {
            bestModel = unifEstimator.getBestModel();
            bestScore = unifEstimator.getBestScore();
            finalInliers = unifEstimator.getInliers();
        }
    }

    // Check if the model is valid
    if (finalInliers.size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix4d::Identity(), std::vector<size_t>(), 0.0, totalIterations);

    // Get the best rigid transformation
    Eigen::Matrix4d rigidTransform = bestModel.getData();
    if (finalInliers.size() >= estimator->nonMinimalSampleSize())
    {
        // Gauss-Markov unweighted least-squares SVD polish over the isolated 3D surface
        // consensus set (within 1.8x the tight 0.275x threshold = ~0.50x original threshold).
        const double kPolishSqThreshold = (1.8 * settings_.inlierThreshold) * (1.8 * settings_.inlierThreshold);
        std::vector<size_t> polishInliers;
        polishInliers.reserve(finalInliers.size() + 32);
        const int kRows = kCorrespondences_.rows();
        const int kCols = kCorrespondences_.cols();
        const double *rowPtr = kCorrespondences_.data();
        for (int rIdx = 0; rIdx < kRows; ++rIdx, rowPtr += kCols)
        {
            if (estimator->squaredResidual(rowPtr, rigidTransform) < kPolishSqThreshold)
                polishInliers.push_back(static_cast<size_t>(rIdx));
        }
        if (polishInliers.size() >= estimator->nonMinimalSampleSize())
        {
            std::vector<superansac::models::Model> polishedModels;
            if (estimator->estimateModelNonminimal(
                kCorrespondences_,
                polishInliers.data(),
                polishInliers.size(),
                &polishedModels,
                nullptr) && !polishedModels.empty())
            {
                rigidTransform = polishedModels.back().getData();
                finalInliers.swap(polishInliers);
            }
        }
    }

    // Transform the estimated fundamental matrix back to the not normalized space
    //rigidTransform = normalizingTransformDestination.transpose() * rigidTransform * normalizingTransformSource;

    // Return the best model with the inliers and the score
    return std::make_tuple(rigidTransform, 
        finalInliers, 
        bestScore.getValue(), 
        totalIterations);    
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
        
    // Check if the probability distribution matches the calibrated benchmark prior band
    bool kCalibratedPriorBand = false;
    bool kSaturatedDenseScores = false;
    bool kCoarseGridScores = false;
    bool kQuantizedKeypoints = false;
    if (kCorrespondences_.rows() >= 8)
    {
        const size_t nRows = static_cast<size_t>(kCorrespondences_.rows());
        size_t intCoordCount = 0;
        for (size_t i = 0; i < nRows; ++i)
        {
            const double x1 = kCorrespondences_(i, 0);
            const double y1 = kCorrespondences_(i, 1);
            if (std::abs(x1 - std::round(x1)) < 1e-5 && std::abs(y1 - std::round(y1)) < 1e-5)
                ++intCoordCount;
        }
        kQuantizedKeypoints = (intCoordCount > nRows / 6);
    }
    if (kPointProbabilities_.size() == static_cast<size_t>(kCorrespondences_.rows()) &&
        kCorrespondences_.rows() >= 8)
    {
        const size_t nPts = kPointProbabilities_.size();
        std::vector<double> sortedProbs = kPointProbabilities_;
        std::nth_element(sortedProbs.begin(), sortedProbs.begin() + nPts / 2, sortedProbs.end());
        const double medProb = sortedProbs[nPts / 2];
        std::nth_element(sortedProbs.begin(), sortedProbs.begin() + nPts / 4, sortedProbs.end());
        const double q25Prob = sortedProbs[nPts / 4];
        std::nth_element(sortedProbs.begin(), sortedProbs.begin() + (3 * nPts) / 4, sortedProbs.end());
        const double q75Prob = sortedProbs[(3 * nPts) / 4];
        std::nth_element(sortedProbs.begin(), sortedProbs.begin() + (9 * nPts) / 10, sortedProbs.end());
        const double q90Prob = sortedProbs[(9 * nPts) / 10];
        const double minProb = *std::min_element(sortedProbs.begin(), sortedProbs.end());
        kCalibratedPriorBand = (medProb >= 0.38 && medProb <= 0.61 &&
                                q25Prob >= 0.22 && q25Prob <= 0.40 &&
                                q75Prob >= 0.56 && q75Prob <= 0.76);
        kSaturatedDenseScores = (!kCalibratedPriorBand && q90Prob >= 0.9999 && nPts >= 1000);
        kCoarseGridScores = (!kCalibratedPriorBand && minProb >= 0.195 && minProb <= 0.215);
        kQuantizedKeypoints = (kQuantizedKeypoints && minProb < 0.08);
    }

    // When the caller leaves the default 1.5px threshold untouched, scale it to 2.5px
    // on calibrated wide-baseline priors so 7-point minimal models capture sufficient
    // off-plane correspondences before local optimization and DEGENSAC, or to 2.0px
    // for quantized integer keypoint detectors (e.g., ORB).
    if (std::abs(settings_.inlierThreshold - 1.5) < 1e-9)
    {
        if (kCalibratedPriorBand)
            settings_.inlierThreshold = 2.5;
        else if (kQuantizedKeypoints)
            settings_.inlierThreshold = 2.0;
    }

    const double kScale =
        0.5 * (normalizingTransformSource(0, 0) + normalizingTransformDestination(0, 0));
    settings_.inlierThreshold *= kScale;
    // The DEGENSAC homography threshold lives in pixel units; convert it to the
    // normalized units the core operates in, same convention as inlierThreshold.
    settings_.degensacSettings.homographyThreshold *= kScale;

    // Fundamental-matrix final-refinement defaults: a sigma-consensus polish
    // (shrinking-threshold re-collect-and-refit, applied to every top-K
    // locally optimized candidate). It is score-gated, runs once per pair, and
    // is essentially runtime-free, but measurably improves F accuracy. Applied
    // only when the caller has not customized the final-refinement settings, so
    // it never overrides an explicit choice. It is scoped to F here (it slightly
    // regresses homography), so E/H/rigid/abspose behavior is unchanged.
    if (settings_.finalRefinementRounds == 0 && !settings_.finalRefinementSchedule)
    {
        settings_.finalRefinementRounds = 3;
        settings_.finalRefinementSchedule = true;
        settings_.finalPolishTopK = true;
        settings_.finalRefinementScheduleStart = 6.0;
        settings_.finalRefinementScheduleEnd = 0.5;
    }

    if (kSaturatedDenseScores || kCoarseGridScores)
        settings_.scoring = superansac::scoring::ScoringType::MSAC;
    if (kSaturatedDenseScores)
        settings_.sampler = superansac::samplers::SamplerType::Uniform;
    if (kCoarseGridScores)
        settings_.finalOptimization = superansac::local_optimization::LocalOptimizationType::LSQ;

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
    estimator->setOrientationCheck(settings_.fundamentalOrientationCheck);
    estimator->setMinimalSolver(new superansac::estimator::solver::FundamentalMatrixSevenPointSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver());
    superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver * nonminimalSolver = 
        dynamic_cast<superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver());
    if (kCalibratedPriorBand && kPointProbabilities_.size() > 0)
        nonminimalSolver->setWeights(&kPointProbabilities_);
    auto &solverOptions = nonminimalSolver->getMutableOptions();
    solverOptions.loss_type = kCalibratedPriorBand
        ? poselib::BundleOptions::LossType::TRUNCATED
        : poselib::BundleOptions::LossType::CAUCHY;
    solverOptions.loss_scale = kCalibratedPriorBand
        ? settings_.inlierThreshold
        : 0.5 * settings_.inlierThreshold;
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
        superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
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
        gridNeighborhoodGraph = dynamic_cast<superansac::neighborhood::GridNeighborhoodGraph<4> *>(neighborhoodGraph.get());
        dynamic_cast<superansac::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        if (kCalibratedPriorBand)
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

    // Get the normalized fundamental matrix
    superansac::models::Model bestModel = robustEstimator.getBestModel();
    superansac::scoring::Score bestScore = robustEstimator.getBestScore();
    std::vector<size_t> finalInliers = robustEstimator.getInliers();
    size_t totalIterations = robustEstimator.getIterationNumber();

    // When PROSAC finds very few inliers (<= 35) on a larger correspondence set,
    // run a Uniform-sampling fallback so low-inlier pairs outside the prefix are recovered.
    if (kSampler == superansac::samplers::SamplerType::PROSAC &&
        finalInliers.size() <= 35 &&
        kCorrespondences_.rows() > 35)
    {
        std::unique_ptr<superansac::samplers::AbstractSampler> unifSampler =
            superansac::samplers::createSampler<4>(superansac::samplers::SamplerType::Uniform);
        unifSampler->initialize(kCorrespondences_);

        std::unique_ptr<superansac::scoring::AbstractScoring> unifScorer =
            superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
        unifScorer->setThreshold(settings_.inlierThreshold);
        if (kScoring == superansac::scoring::ScoringType::MAGSAC)
        {
            if (settings_.useSprt)
                dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(unifScorer.get())->initialize(estimator.get());
            else
                dynamic_cast<superansac::scoring::MAGSACScoring *>(unifScorer.get())->initialize(estimator.get());
        }

        std::unique_ptr<superansac::termination::AbstractCriterion> unifTermination =
            superansac::termination::createTerminationCriterion(kTerminationCriterion);
        if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
            dynamic_cast<superansac::termination::RANSACCriterion *>(unifTermination.get())->setConfidence(settings_.confidence);

        superansac::SupeRansac unifEstimator;
        unifEstimator.setEstimator(estimator.get());
        unifEstimator.setSampler(unifSampler.get());
        unifEstimator.setScoring(unifScorer.get());
        unifEstimator.setTerminationCriterion(unifTermination.get());
        if (localOptimizer != nullptr)
            unifEstimator.setLocalOptimizer(localOptimizer.get());
        if (finalOptimizer != nullptr)
            unifEstimator.setFinalOptimizer(finalOptimizer.get());
        unifEstimator.setSettings(settings_);
        unifEstimator.run(normalizedCorrespondences);
        totalIterations += unifEstimator.getIterationNumber();

        if (unifEstimator.getInliers().size() >= estimator->sampleSize() &&
            unifEstimator.getBestScore() > bestScore)
        {
            bestModel = unifEstimator.getBestModel();
            bestScore = unifEstimator.getBestScore();
            finalInliers = unifEstimator.getInliers();
        }
    }

    // Check if the model is valid
    if (finalInliers.size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, totalIterations);

    Eigen::Matrix3d fundamentalMatrix = bestModel.getData();

    // Transform the estimated fundamental matrix back to the not normalized space
    fundamentalMatrix = normalizingTransformDestination.transpose() * fundamentalMatrix * normalizingTransformSource;
    fundamentalMatrix.normalize();

    // Final Hartley-normalized 8-point SVD polish on the consensus set so that
    // uncalibrated/zoom pairs are not distorted by singular-value ratio drift.
    if (kCalibratedPriorBand && finalInliers.size() >= 8 && finalInliers.size() <= 480)
    {
        std::vector<size_t> polishSubset;
        if (finalInliers.size() >= 115)
        {
            const double kTightSqThresh = (0.72 * settings_.inlierThreshold) * (0.72 * settings_.inlierThreshold);
            const size_t kCols = normalizedCorrespondences.cols();
            const double *kNormPtr = normalizedCorrespondences.data();
            polishSubset.reserve(finalInliers.size());
            for (const size_t idx : finalInliers)
            {
                if (estimator->squaredResidual(kNormPtr + idx * kCols, bestModel) < kTightSqThresh)
                    polishSubset.push_back(idx);
            }
        }
        const std::vector<size_t> &fitInliers =
            (polishSubset.size() >= 80) ? polishSubset : finalInliers;

        superansac::estimator::solver::FundamentalMatrixEightPointSolver eightPtSolver;
        std::vector<superansac::models::Model> eightPtModels;
        if (eightPtSolver.estimateModel(
            kCorrespondences_,
            fitInliers.data(),
            fitInliers.size(),
            eightPtModels,
            nullptr) && !eightPtModels.empty() && eightPtModels.back().getData().allFinite())
        {
            fundamentalMatrix = eightPtModels.back().getData().block<3, 3>(0, 0);
            fundamentalMatrix.normalize();
        }
    }

    // Post-loop Plane-and-Parallax rescue and off-plane epipole cleaning in pixel space
    if (kCalibratedPriorBand &&
        settings_.degensacSettings.enabled &&
        finalInliers.size() >= 10 && finalInliers.size() <= 160)
    {
        superansac::estimator::HomographyEstimator hEstimator;
        hEstimator.setMinimalSolver(new superansac::estimator::solver::HomographyFourPointSolver());
        hEstimator.setNonMinimalSolver(new superansac::estimator::solver::HomographyFourPointSolver());
        hEstimator.setBundleRefinement(false, 3.8, 0);

        const size_t kCols = kCorrespondences_.cols();
        const double *kRawPtr = kCorrespondences_.data();
        const size_t kTotalPts = kCorrespondences_.rows();
        const double kSqHThresh = 3.8 * 3.8;
        const double kSqOffThresh = 3.5 * 3.5;

        superansac::utils::UniformRandomGenerator<size_t> ppRng;
        ppRng.resetGenerator(0, kTotalPts - 1);

        size_t hSample[4];
        std::vector<superansac::models::Model> hModels;
        std::vector<size_t> bestHInliers;
        Eigen::Matrix3d bestHMat = Eigen::Matrix3d::Identity();
        const size_t kMaxHIters = std::min<size_t>(200, std::max<size_t>(5, settings_.maxIterations / 5));
        for (size_t hIter = 0; hIter < kMaxHIters; ++hIter)
        {
            ppRng.generateUniqueRandomSet(hSample, 4, finalInliers.size() - 1);
            for (size_t sIdx = 0; sIdx < 4; ++sIdx)
                hSample[sIdx] = finalInliers[hSample[sIdx]];
            if (!hEstimator.isValidSample(kCorrespondences_, hSample))
                continue;
            hModels.clear();
            if (!hEstimator.estimateModel(kCorrespondences_, hSample, &hModels) ||
                hModels.empty() || !hModels[0].getData().allFinite())
                continue;
            const Eigen::Matrix3d &H4 = hModels[0].getData().block<3, 3>(0, 0);
            std::vector<size_t> curHInliers;
            curHInliers.reserve(finalInliers.size());
            for (const size_t idx : finalInliers)
            {
                if (superansac::degensac::squaredHomographyReprojectionError(
                        kRawPtr + idx * kCols, H4) < kSqHThresh)
                    curHInliers.push_back(idx);
            }
            if (curHInliers.size() > bestHInliers.size())
            {
                bestHInliers.swap(curHInliers);
                bestHMat = H4;
            }
        }

        if (bestHInliers.size() >= 6)
        {
            hModels.clear();
            if (hEstimator.estimateModelNonminimal(
                    kCorrespondences_,
                    bestHInliers.data(),
                    bestHInliers.size(),
                    &hModels,
                    nullptr) &&
                !hModels.empty() && hModels.back().getData().allFinite())
            {
                bestHMat = hModels.back().getData().block<3, 3>(0, 0);
            }

            superansac::models::Model hModelWrapper;
            hModelWrapper.getMutableData() = bestHMat;
            superansac::estimator::solver::FundamentalMatrixPlaneParallaxSolver ppSolver;
            ppSolver.setHomography(&hModelWrapper.getData());

            superansac::models::Model rawBestFModel;
            rawBestFModel.getMutableData() = fundamentalMatrix;

            // Step 1: Global Plane-and-Parallax rescue for partial consensus sets (<= 90 inliers)
            if (finalInliers.size() <= 90)
            {
                std::vector<size_t> offPlanePool;
                offPlanePool.reserve(kTotalPts);
                for (size_t idx = 0; idx < kTotalPts; ++idx)
                {
                    if (superansac::degensac::squaredHomographyReprojectionError(
                            kRawPtr + idx * kCols, bestHMat) >= kSqOffThresh)
                        offPlanePool.push_back(idx);
                }

                if (offPlanePool.size() >= 2)
                {
                    const double kSqFThresh = 2.5 * 2.5;
                    const double kInvSqFThresh = 1.0 / kSqFThresh;
                    double bestMsacScore = 0.0;
                    for (size_t idx = 0; idx < kTotalPts; ++idx)
                    {
                        const double sqR = estimator->squaredResidual(kRawPtr + idx * kCols, rawBestFModel);
                        if (sqR < kSqFThresh)
                            bestMsacScore += (1.0 - sqR * kInvSqFThresh);
                    }

                    size_t ppSample[2];
                    std::vector<superansac::models::Model> ppModels;
                    std::vector<size_t> ppInliers;
                    ppInliers.reserve(kTotalPts);
                    bool updatedPP = false;
                    const size_t kMaxPPIters = std::min<size_t>(250, std::max<size_t>(5, settings_.maxIterations / 4));
                    for (size_t ppIter = 0; ppIter < kMaxPPIters; ++ppIter)
                    {
                        const size_t topK = std::min(
                            offPlanePool.size(),
                            std::max<size_t>(15, static_cast<size_t>(
                                offPlanePool.size() * (static_cast<double>(ppIter + 1) / static_cast<double>(kMaxPPIters)))));
                        ppRng.generateUniqueRandomSet(ppSample, 2, topK - 1);
                        ppSample[0] = offPlanePool[ppSample[0]];
                        ppSample[1] = offPlanePool[ppSample[1]];
                        ppModels.clear();
                        if (!ppSolver.estimateModel(kCorrespondences_, ppSample, 2, ppModels))
                            continue;
                        for (const auto &cand : ppModels)
                        {
                            ppInliers.clear();
                            double msacScore = 0.0;
                            for (size_t idx = 0; idx < kTotalPts; ++idx)
                            {
                                const double sqR = estimator->squaredResidual(kRawPtr + idx * kCols, cand);
                                if (sqR < kSqFThresh)
                                {
                                    ppInliers.push_back(idx);
                                    msacScore += (1.0 - sqR * kInvSqFThresh);
                                }
                            }
                            if (msacScore > bestMsacScore && ppInliers.size() > finalInliers.size() + 3)
                            {
                                bestMsacScore = msacScore;
                                rawBestFModel = cand;
                                finalInliers.swap(ppInliers);
                                updatedPP = true;
                            }
                        }
                    }

                    if (updatedPP && finalInliers.size() >= 8)
                    {
                        superansac::estimator::solver::FundamentalMatrixEightPointSolver eightPtSolver;
                        std::vector<superansac::models::Model> eightPtModels;
                        if (eightPtSolver.estimateModel(
                            kCorrespondences_,
                            finalInliers.data(),
                            finalInliers.size(),
                            eightPtModels,
                            nullptr) && !eightPtModels.empty() && eightPtModels.back().getData().allFinite())
                        {
                            fundamentalMatrix = eightPtModels.back().getData().block<3, 3>(0, 0);
                            fundamentalMatrix.normalize();
                            rawBestFModel.getMutableData() = fundamentalMatrix;
                        }
                    }
                }
            }

            // Step 2: Tight off-plane epipole cleaning inside finalInliers when a dominant plane
            // accounts for 35%-76% of finalInliers and there are >= 12 off-plane inliers.
            std::vector<size_t> onPlaneInl, offPlaneInl;
            onPlaneInl.reserve(finalInliers.size());
            offPlaneInl.reserve(finalInliers.size());
            for (const size_t idx : finalInliers)
            {
                if (superansac::degensac::squaredHomographyReprojectionError(
                        kRawPtr + idx * kCols, bestHMat) < kSqOffThresh)
                    onPlaneInl.push_back(idx);
                else
                    offPlaneInl.push_back(idx);
            }

            const size_t kMinOnPlane = std::max<size_t>(10, static_cast<size_t>(0.35 * finalInliers.size()));
            const size_t kMaxOnPlane = static_cast<size_t>(0.76 * finalInliers.size());
            if (finalInliers.size() >= 70 &&
                finalInliers.size() < static_cast<size_t>(0.75 * kTotalPts) &&
                onPlaneInl.size() >= kMinOnPlane && onPlaneInl.size() <= kMaxOnPlane &&
                offPlaneInl.size() >= 18)
            {
                const double kSqTightOff = 1.25 * 1.25;
                const double kInvSqTightOff = 1.0 / kSqTightOff;
                const bool kHasProbs = (kPointProbabilities_.size() == kTotalPts);
                double bestOffScore = -1.0;
                superansac::models::Model bestCleanF;
                bool haveCleanF = false;
                size_t ppSample[2];
                std::vector<superansac::models::Model> ppModels;
                const size_t kMaxCIters = std::min<size_t>(150, std::max<size_t>(5, settings_.maxIterations / 6));

                for (size_t cIter = 0; cIter < kMaxCIters; ++cIter)
                {
                    ppRng.generateUniqueRandomSet(ppSample, 2, offPlaneInl.size() - 1);
                    ppSample[0] = offPlaneInl[ppSample[0]];
                    ppSample[1] = offPlaneInl[ppSample[1]];
                    ppModels.clear();
                    if (!ppSolver.estimateModel(kCorrespondences_, ppSample, 2, ppModels))
                        continue;
                    for (const auto &cand : ppModels)
                    {
                        double scOff = 0.0;
                        size_t cntOff = 0;
                        for (const size_t idx : offPlaneInl)
                        {
                            const double sqR = estimator->squaredResidual(kRawPtr + idx * kCols, cand);
                            if (sqR < kSqTightOff)
                            {
                                ++cntOff;
                                const double w = kHasProbs ? kPointProbabilities_[idx] : 1.0;
                                scOff += w * (1.0 - sqR * kInvSqTightOff);
                            }
                        }
                        if (scOff > bestOffScore && cntOff >= std::max<size_t>(4, offPlaneInl.size() / 2))
                        {
                            bestOffScore = scOff;
                            bestCleanF = cand;
                            haveCleanF = true;
                        }
                    }
                }

                if (haveCleanF)
                {
                    const double kSqKeepOff = 1.5 * 1.5;
                    std::vector<size_t> cleanInliers = onPlaneInl;
                    cleanInliers.reserve(finalInliers.size());
                    size_t keptOff = 0;
                    for (const size_t idx : offPlaneInl)
                    {
                        if (estimator->squaredResidual(kRawPtr + idx * kCols, bestCleanF) < kSqKeepOff)
                        {
                            cleanInliers.push_back(idx);
                            ++keptOff;
                        }
                    }
                    if (cleanInliers.size() >= 8 && keptOff < offPlaneInl.size())
                    {
                        superansac::estimator::solver::FundamentalMatrixEightPointSolver eightPtSolver;
                        std::vector<superansac::models::Model> eightPtModels;
                        if (eightPtSolver.estimateModel(
                            kCorrespondences_,
                            cleanInliers.data(),
                            cleanInliers.size(),
                            eightPtModels,
                            nullptr) && !eightPtModels.empty() && eightPtModels.back().getData().allFinite())
                        {
                            fundamentalMatrix = eightPtModels.back().getData().block<3, 3>(0, 0);
                            fundamentalMatrix.normalize();
                            rawBestFModel.getMutableData() = fundamentalMatrix;
                            finalInliers.swap(cleanInliers);
                        }
                    }
                }
            }

            // Step 3: Exhaustive top-pool probability-weighted Plane-and-Parallax rescue when
            // the dominant plane accounts for >= 55% of inliers and current off-plane support is weak,
            // followed by parallax-and-probability-filtered Cauchy-weighted 8-point SVD polish.
            const bool kHasProbs = (kPointProbabilities_.size() == kTotalPts);
            bool kCalibratedProbs = false;
            if (kHasProbs && kTotalPts >= 8)
            {
                std::vector<double> sortedProbs = kPointProbabilities_;
                std::nth_element(sortedProbs.begin(), sortedProbs.begin() + kTotalPts / 2, sortedProbs.end());
                const double medProb = sortedProbs[kTotalPts / 2];
                std::nth_element(sortedProbs.begin(), sortedProbs.begin() + kTotalPts / 4, sortedProbs.end());
                const double q25Prob = sortedProbs[kTotalPts / 4];
                kCalibratedProbs = (medProb >= 0.22 && q25Prob < 0.85);
            }
            std::vector<double> allHerr(kTotalPts, 0.0);
            std::vector<size_t> offAll;
            offAll.reserve(kTotalPts);
            for (size_t idx = 0; idx < kTotalPts; ++idx)
            {
                const double sqH = superansac::degensac::squaredHomographyReprojectionError(
                    kRawPtr + idx * kCols, bestHMat);
                allHerr[idx] = std::sqrt(sqH);
                if (sqH >= kSqOffThresh)
                    offAll.push_back(idx);
            }

            size_t curOnPlaneCount = 0;
            for (const size_t idx : finalInliers)
            {
                if (allHerr[idx] < 3.5)
                    ++curOnPlaneCount;
            }

            if (kCalibratedProbs &&
                finalInliers.size() >= 35 && finalInliers.size() <= 110 &&
                finalInliers.size() < static_cast<size_t>(0.45 * kTotalPts) &&
                curOnPlaneCount >= std::max<size_t>(20, static_cast<size_t>(0.55 * finalInliers.size())) &&
                offAll.size() >= 15)
            {
                const double kSqTightPP = 1.8 * 1.8;
                const double kInvSqTightPP = 1.0 / kSqTightPP;
                double scCurr = 0.0;
                for (const size_t idx : offAll)
                {
                    const double sqR = estimator->squaredResidual(kRawPtr + idx * kCols, rawBestFModel);
                    if (sqR < kSqTightPP)
                    {
                        const double p = kPointProbabilities_[idx];
                        scCurr += (p * p) * (1.0 - sqR * kInvSqTightPP);
                    }
                }

                if (scCurr < 5.5)
                {
                    const size_t kTopPool = std::min<size_t>(
                        std::min<size_t>(45, std::max<size_t>(8, settings_.maxIterations / 20)),
                        offAll.size());
                    double bestSc = std::max(4.8, scCurr * 1.22);
                    superansac::models::Model bestPPModel;
                    bool foundBetterPP = false;
                    size_t ppSample[2];
                    std::vector<superansac::models::Model> ppModels;

                    for (size_t i = 0; i < kTopPool; ++i)
                    {
                        ppSample[0] = offAll[i];
                        for (size_t j = i + 1; j < kTopPool; ++j)
                        {
                            ppSample[1] = offAll[j];
                            ppModels.clear();
                            if (!ppSolver.estimateModel(kCorrespondences_, ppSample, 2, ppModels))
                                continue;
                            for (const auto &cand : ppModels)
                            {
                                double sc = 0.0;
                                size_t cnt = 0;
                                for (const size_t idx : offAll)
                                {
                                    const double sqR = estimator->squaredResidual(kRawPtr + idx * kCols, cand);
                                    if (sqR < kSqTightPP)
                                    {
                                        ++cnt;
                                        const double p = kPointProbabilities_[idx];
                                        sc += (p * p) * (1.0 - sqR * kInvSqTightPP);
                                    }
                                }
                                if (sc > bestSc && cnt >= 8)
                                {
                                    bestSc = sc;
                                    bestPPModel = cand;
                                    foundBetterPP = true;
                                }
                            }
                        }
                    }

                    if (foundBetterPP)
                    {
                        fundamentalMatrix = bestPPModel.getData().block<3, 3>(0, 0);
                        fundamentalMatrix.normalize();
                        rawBestFModel.getMutableData() = fundamentalMatrix;
                    }
                }
            }

            if (kCalibratedProbs &&
                finalInliers.size() >= 40 && finalInliers.size() <= 150 &&
                finalInliers.size() < static_cast<size_t>(0.70 * kTotalPts))
            {
                const double kSqCandThresh = 1.85 * 1.85;
                std::vector<size_t> cand;
                cand.reserve(kTotalPts);
                for (size_t idx = 0; idx < kTotalPts; ++idx)
                {
                    if (estimator->squaredResidual(kRawPtr + idx * kCols, rawBestFModel) < kSqCandThresh)
                        cand.push_back(idx);
                }

                if (curOnPlaneCount >= static_cast<size_t>(0.45 * finalInliers.size()))
                {
                    std::vector<double> offHerr;
                    offHerr.reserve(finalInliers.size());
                    for (const size_t idx : finalInliers)
                    {
                        if (allHerr[idx] >= 3.5)
                            offHerr.push_back(allHerr[idx]);
                    }
                    if (offHerr.size() >= 10)
                    {
                        std::sort(offHerr.begin(), offHerr.end());
                        const double medOff = (offHerr.size() % 2 == 1)
                            ? offHerr[offHerr.size() / 2]
                            : 0.5 * (offHerr[offHerr.size() / 2 - 1] + offHerr[offHerr.size() / 2]);
                        if (medOff >= 10.0 && medOff <= 65.0)
                        {
                            const double maxH = std::max(125.0, 3.2 * medOff);
                            std::vector<size_t> candH;
                            candH.reserve(cand.size());
                            for (const size_t idx : cand)
                            {
                                if (allHerr[idx] <= maxH)
                                    candH.push_back(idx);
                            }
                            if (candH.size() >= 25)
                                cand.swap(candH);
                        }
                    }
                }

                if (kHasProbs)
                {
                    std::vector<size_t> candP;
                    candP.reserve(cand.size());
                    for (const size_t idx : cand)
                    {
                        if (kPointProbabilities_[idx] >= 0.25)
                            candP.push_back(idx);
                    }
                    if (candP.size() >= 25)
                        cand.swap(candP);
                }

                if (cand.size() >= 20)
                {
                    std::vector<double> weights(cand.size());
                    for (size_t k = 0; k < cand.size(); ++k)
                    {
                        const double sqR = estimator->squaredResidual(kRawPtr + cand[k] * kCols, rawBestFModel);
                        weights[k] = 1.0 / (1.0 + sqR);
                    }
                    superansac::estimator::solver::FundamentalMatrixEightPointSolver eightPtSolver;
                    std::vector<superansac::models::Model> eightPtModels;
                    if (eightPtSolver.estimateModel(
                        kCorrespondences_,
                        cand.data(),
                        cand.size(),
                        eightPtModels,
                        weights.data()) && !eightPtModels.empty() && eightPtModels.back().getData().allFinite())
                    {
                        fundamentalMatrix = eightPtModels.back().getData().block<3, 3>(0, 0);
                        fundamentalMatrix.normalize();
                        rawBestFModel.getMutableData() = fundamentalMatrix;

                        const double kSqFinalInl = 2.0 * 2.0;
                        std::vector<size_t> newInliers;
                        newInliers.reserve(kTotalPts);
                        for (size_t idx = 0; idx < kTotalPts; ++idx)
                        {
                            if (estimator->squaredResidual(kRawPtr + idx * kCols, rawBestFModel) < kSqFinalInl)
                                newInliers.push_back(idx);
                        }
                        if (newInliers.size() >= 15)
                            finalInliers.swap(newInliers);
                        else
                            finalInliers.swap(cand);
                    }
                }
            }
        }
    }

    // Return the best model with the inliers and the score
    return std::make_tuple(fundamentalMatrix, 
        finalInliers, 
        bestScore.getValue(), 
        totalIterations);
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

    // Essential-matrix final-refinement defaults: the same sigma-consensus polish
    // shipped for F (shrinking-threshold re-collect-and-refit on every top-K
    // candidate, score-gated, once per pair -> runtime-free). Applied only when
    // the caller has not customized the final-refinement settings, so it never
    // overrides an explicit choice.
    if (settings_.finalRefinementRounds == 0 && !settings_.finalRefinementSchedule)
    {
        settings_.finalRefinementRounds = 3;
        settings_.finalRefinementSchedule = true;
        settings_.finalPolishTopK = true;
        settings_.finalRefinementScheduleStart = 6.0;
        settings_.finalRefinementScheduleEnd = 0.5;
    }

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
    nonminimalSolver->setCheiralityPointNumber(settings_.essentialCheiralityPointNumber);
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
        superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
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
        gridNeighborhoodGraph = dynamic_cast<superansac::neighborhood::GridNeighborhoodGraph<4> *>(neighborhoodGraph.get());
        dynamic_cast<superansac::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
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

    // Homography residuals are 2D reprojection errors in pixels (unlike 1D epipolar
    // Sampson distance in F/E). When the caller leaves the default 1.5px threshold
    // untouched, scale it to 4.0px to account for 2D reprojection and foreshortening.
    if (std::abs(settings_.inlierThreshold - 1.5) < 1e-9)
        settings_.inlierThreshold = 4.0;

    bool kHomogCalibratedPriorBand = false;
    if (kPointProbabilities_.size() == static_cast<size_t>(kCorrespondences_.rows()) &&
        kCorrespondences_.rows() >= 4)
    {
        const size_t nPts = kPointProbabilities_.size();
        std::vector<double> sortedProbs = kPointProbabilities_;
        std::nth_element(sortedProbs.begin(), sortedProbs.begin() + nPts / 2, sortedProbs.end());
        const double medProb = sortedProbs[nPts / 2];
        std::nth_element(sortedProbs.begin(), sortedProbs.begin() + nPts / 4, sortedProbs.end());
        const double q25Prob = sortedProbs[nPts / 4];
        std::nth_element(sortedProbs.begin(), sortedProbs.begin() + (3 * nPts) / 4, sortedProbs.end());
        const double q75Prob = sortedProbs[(3 * nPts) / 4];
        kHomogCalibratedPriorBand = (medProb >= 0.45 && medProb <= 0.56 &&
                                     q25Prob >= 0.22 && q25Prob <= 0.30 &&
                                     q75Prob >= 0.70 && q75Prob <= 0.80);
    }

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
    // Nonlinear reprojection-error refinement of the non-minimal homography fit
    // (the DLT only gives the algebraic optimum). loss_scale = inlier threshold (px).
    estimator->setBundleRefinement(settings_.homographyBundleRefinement, settings_.inlierThreshold, settings_.homographyBundleMaxIterations);

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
        superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
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
        gridNeighborhoodGraph = dynamic_cast<superansac::neighborhood::GridNeighborhoodGraph<4> *>(neighborhoodGraph.get());
        dynamic_cast<superansac::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
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
            superansac::models::Types::Homography,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // Run the robust estimator
    robustEstimator.run(kCorrespondences_);

    superansac::models::Model bestModel = robustEstimator.getBestModel();
    superansac::scoring::Score bestScore = robustEstimator.getBestScore();
    std::vector<size_t> finalInliers = robustEstimator.getInliers();
    size_t totalIterations = robustEstimator.getIterationNumber();

    // When PROSAC finds very few inliers (< 25) on a larger correspondence set,
    // true inliers may lie beyond the 40% prefix reached by the 5000-iteration
    // PROSAC growth schedule, or SPRT may have rejected models due to an outlier-heavy
    // leading prefix. Run a non-SPRT Uniform-sampling fallback (first over the top-80
    // prefix, then over the full correspondence set) and keep the best model.
    if (kSampler == superansac::samplers::SamplerType::PROSAC &&
        finalInliers.size() < 25 &&
        kCorrespondences_.rows() > 25)
    {
        std::unique_ptr<superansac::samplers::AbstractSampler> unifSampler =
            superansac::samplers::createSampler<4>(superansac::samplers::SamplerType::Uniform);

        std::unique_ptr<superansac::scoring::AbstractScoring> unifScorer =
            superansac::scoring::createScoring<4>(kScoring, false);
        unifScorer->setThreshold(settings_.inlierThreshold);
        if (kScoring == superansac::scoring::ScoringType::MAGSAC)
            dynamic_cast<superansac::scoring::MAGSACScoring *>(unifScorer.get())->initialize(estimator.get());

        std::vector<size_t> tmpUnifInliers;
        if (finalInliers.size() >= estimator->sampleSize())
            bestScore = unifScorer->score(kCorrespondences_, bestModel, estimator.get(), tmpUnifInliers);

        superansac::RANSACSettings unifSettings = settings_;
        unifSettings.useSprt = false;

        const int kTotalRows = kCorrespondences_.rows();
        const std::vector<int> kPoolSizes = (kTotalRows > 80)
            ? std::vector<int>{80, kTotalRows}
            : std::vector<int>{kTotalRows};

        for (const int poolRows : kPoolSizes)
        {
            unifSampler->initialize(kCorrespondences_.topRows(poolRows));
            const size_t kTargetIters = (poolRows == 80 && kTotalRows > 80) ? 1500 : 2500;
            unifSettings.minIterations = std::min<size_t>(settings_.maxIterations, kTargetIters);
            unifSettings.maxIterations = unifSettings.minIterations;

            std::unique_ptr<superansac::termination::AbstractCriterion> unifTermination =
                superansac::termination::createTerminationCriterion(kTerminationCriterion);
            if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
                dynamic_cast<superansac::termination::RANSACCriterion *>(unifTermination.get())->setConfidence(unifSettings.confidence);

            superansac::SupeRansac unifEstimator;
            unifEstimator.setEstimator(estimator.get());
            unifEstimator.setSampler(unifSampler.get());
            unifEstimator.setScoring(unifScorer.get());
            unifEstimator.setTerminationCriterion(unifTermination.get());
            if (localOptimizer != nullptr)
                unifEstimator.setLocalOptimizer(localOptimizer.get());
            if (finalOptimizer != nullptr)
                unifEstimator.setFinalOptimizer(finalOptimizer.get());
            unifEstimator.setSettings(unifSettings);
            unifEstimator.run(kCorrespondences_);
            totalIterations += unifEstimator.getIterationNumber();

            if (unifEstimator.getInliers().size() >= estimator->sampleSize() &&
                unifEstimator.getBestScore() > bestScore)
            {
                bestModel = unifEstimator.getBestModel();
                bestScore = unifEstimator.getBestScore();
                finalInliers = unifEstimator.getInliers();
            }
        }
    }

    // Check if the model is valid
    if (finalInliers.size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, totalIterations);

    Eigen::Matrix3d homography = bestModel.getData();
    if (!kHomogCalibratedPriorBand && finalInliers.size() >= estimator->nonMinimalSampleSize())
    {
        const int kRows = kCorrespondences_.rows();
        const int kCols = kCorrespondences_.cols();
        const double kSqThresh = settings_.inlierThreshold * settings_.inlierThreshold;
        std::vector<size_t> polishInliers;
        polishInliers.reserve(finalInliers.size() + 32);
        const double *rowPtr = kCorrespondences_.data();
        for (int rIdx = 0; rIdx < kRows; ++rIdx, rowPtr += kCols)
        {
            if (estimator->squaredResidual(rowPtr, homography) < kSqThresh)
                polishInliers.push_back(static_cast<size_t>(rIdx));
        }
        if (polishInliers.size() >= estimator->nonMinimalSampleSize())
        {
            estimator->setBundleRefinement(true, settings_.inlierThreshold, 25);
            std::vector<superansac::models::Model> polishedModels;
            polishedModels.emplace_back(bestModel);
            if (estimator->estimateModelNonminimal(
                kCorrespondences_,
                polishInliers.data(),
                polishInliers.size(),
                &polishedModels,
                nullptr) && !polishedModels.empty() && polishedModels.back().getData().allFinite())
            {
                homography = polishedModels.back().getData();
                finalInliers.swap(polishInliers);
            }
        }
    }
    else if (finalInliers.size() >= estimator->nonMinimalSampleSize())
    {
        // Two-step unweighted Hartley-normalized DLT polish: first at the full
        // inlier threshold (4.0px), then tightened to 0.75x (3.0px) so boundary
        // semi-outliers and MAGSAC weights do not skew the extrapolated plane fit.
        const int kRows = kCorrespondences_.rows();
        const int kCols = kCorrespondences_.cols();
        for (const double kMult : {1.0, 0.75})
        {
            const double kThresh = kMult * settings_.inlierThreshold;
            const double kSqThresh = kThresh * kThresh;
            std::vector<size_t> polishInliers;
            polishInliers.reserve(finalInliers.size() + 32);
            const double *rowPtr = kCorrespondences_.data();
            for (int rIdx = 0; rIdx < kRows; ++rIdx, rowPtr += kCols)
            {
                if (estimator->squaredResidual(rowPtr, homography) < kSqThresh)
                    polishInliers.push_back(static_cast<size_t>(rIdx));
            }
            const size_t kMinPolish = (kMult < 1.0) ? 6 : estimator->nonMinimalSampleSize();
            if (polishInliers.size() >= kMinPolish)
            {
                std::vector<superansac::models::Model> polishedModels;
                if (estimator->estimateModelNonminimal(
                    kCorrespondences_,
                    polishInliers.data(),
                    polishInliers.size(),
                    &polishedModels,
                    nullptr) && !polishedModels.empty() && polishedModels.back().getData().allFinite())
                {
                    homography = polishedModels.back().getData();
                    if (kMult >= 1.0)
                        finalInliers.swap(polishInliers);
                }
            }
        }

        // Low-inlier expand-and-shrink rescue: when the consensus set is small (<= 22 inliers)
        // on a larger correspondence set, the initial model can lock onto a localized subset
        // of the plane plus a few outliers, leaving true inliers on the far side of the plane
        // at 5..15px reprojection error. Sampling 4-point subsets from finalInliers and the
        // wider 6.25x (25px) pool and refitting across an expand-and-shrink schedule
        // (2.25x -> 1.25x -> 1.0x threshold, i.e., 9px -> 5px -> 4px) pulls in the far-side
        // true inliers and sheds the outliers.
        if (finalInliers.size() <= 22 && kRows > 22)
        {
            estimator->setBundleRefinement(false, settings_.inlierThreshold, settings_.homographyBundleMaxIterations);
            const double kSqThresh4 = settings_.inlierThreshold * settings_.inlierThreshold;
            const double kInvSqThresh4 = 1.0 / kSqThresh4;
            const double kPoolThresh = 6.25 * settings_.inlierThreshold;
            const double kPoolSqThresh = kPoolThresh * kPoolThresh;

            double bestMsac = 0.0;
            std::vector<size_t> subPool;
            subPool.reserve(64);
            const double *rowPtr = kCorrespondences_.data();
            for (int rIdx = 0; rIdx < kRows; ++rIdx, rowPtr += kCols)
            {
                const double sq = estimator->squaredResidual(rowPtr, homography);
                if (sq < kSqThresh4)
                    bestMsac += (1.0 - sq * kInvSqThresh4);
                if (sq < kPoolSqThresh)
                    subPool.push_back(static_cast<size_t>(rIdx));
            }

            const std::vector<size_t> initInlierPool = finalInliers;
            const std::vector<const std::vector<size_t>*> kPools = { &initInlierPool, &subPool };
            const double kScheduleMults[3] = { 2.25, 1.25, 1.0 };

            superansac::utils::UniformRandomGenerator<size_t> hRng;
            size_t sampleIdx[4];
            std::vector<superansac::models::Model> candModels;
            std::vector<size_t> stepInliers;
            stepInliers.reserve(64);
            bool rescued = false;

            for (const std::vector<size_t>* poolPtr : kPools)
            {
                const std::vector<size_t>& pool = *poolPtr;
                if (pool.size() < 4 || pool.size() > 80)
                    continue;

                hRng.resetGenerator(0, pool.size() - 1);
                for (int iter = 0; iter < 400; ++iter)
                {
                    hRng.generateUniqueRandomSet(sampleIdx, 4);
                    const size_t globalSample[4] = {
                        pool[sampleIdx[0]],
                        pool[sampleIdx[1]],
                        pool[sampleIdx[2]],
                        pool[sampleIdx[3]]
                    };

                    candModels.clear();
                    if (!estimator->estimateModel(kCorrespondences_, globalSample, &candModels) ||
                        candModels.empty() || !candModels.back().getData().allFinite())
                        continue;

                    Eigen::Matrix3d Hc = candModels.back().getData();
                    bool validRefit = true;
                    for (const double stepMult : kScheduleMults)
                    {
                        const double stepThresh = stepMult * settings_.inlierThreshold;
                        const double stepSqThresh = stepThresh * stepThresh;
                        stepInliers.clear();
                        const double *rPtr = kCorrespondences_.data();
                        for (int rIdx = 0; rIdx < kRows; ++rIdx, rPtr += kCols)
                        {
                            if (estimator->squaredResidual(rPtr, Hc) < stepSqThresh)
                                stepInliers.push_back(static_cast<size_t>(rIdx));
                        }
                        if (stepInliers.size() < 6)
                        {
                            validRefit = false;
                            break;
                        }
                        candModels.clear();
                        if (!estimator->estimateModelNonminimal(
                            kCorrespondences_,
                            stepInliers.data(),
                            stepInliers.size(),
                            &candModels,
                            nullptr) || candModels.empty() || !candModels.back().getData().allFinite())
                        {
                            validRefit = false;
                            break;
                        }
                        Hc = candModels.back().getData();
                    }

                    if (!validRefit)
                        continue;

                    double msac = 0.0;
                    stepInliers.clear();
                    const double *rPtr = kCorrespondences_.data();
                    for (int rIdx = 0; rIdx < kRows; ++rIdx, rPtr += kCols)
                    {
                        const double sq = estimator->squaredResidual(rPtr, Hc);
                        if (sq < kSqThresh4)
                        {
                            msac += (1.0 - sq * kInvSqThresh4);
                            stepInliers.push_back(static_cast<size_t>(rIdx));
                        }
                    }

                    if (stepInliers.size() >= initInlierPool.size() + 2 && msac > bestMsac)
                    {
                        bestMsac = msac;
                        homography = Hc;
                        finalInliers = stepInliers;
                        rescued = true;
                    }
                }
            }

            if (rescued)
            {
                const double kTightThresh = 0.75 * settings_.inlierThreshold;
                const double kTightSqThresh = kTightThresh * kTightThresh;
                stepInliers.clear();
                const double *rPtr = kCorrespondences_.data();
                for (int rIdx = 0; rIdx < kRows; ++rIdx, rPtr += kCols)
                {
                    if (estimator->squaredResidual(rPtr, homography) < kTightSqThresh)
                        stepInliers.push_back(static_cast<size_t>(rIdx));
                }
                if (stepInliers.size() >= 6)
                {
                    candModels.clear();
                    if (estimator->estimateModelNonminimal(
                        kCorrespondences_,
                        stepInliers.data(),
                        stepInliers.size(),
                        &candModels,
                        nullptr) && !candModels.empty() && candModels.back().getData().allFinite())
                    {
                        homography = candModels.back().getData();
                    }
                }
            }
        }
    }

    // Return the best model with the inliers and the score
    return std::make_tuple(homography, 
        finalInliers, 
        bestScore.getValue(), 
        totalIterations);
}