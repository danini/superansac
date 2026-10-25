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
            kCellSizes[i] = std::max(1e-6, kImageSizes_[i] / static_cast<double>(kCellNumber));
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
        else if (kLOSettings_.refinementRounds > 0)
        {
            // In-loop LO with a widened collection schedule (LO-RANSAC style)
            lsqLocalOptimizer->setExtraRefinementRounds(kLOSettings_.refinementRounds);
            lsqLocalOptimizer->setSigmaSchedule(true);
            lsqLocalOptimizer->setSigmaScheduleRange(kLOSettings_.scheduleStart, kLOSettings_.scheduleEnd);
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

// ---------------------------------------------------------------------------------------------
// Final robust bundle polish shared by F / E / H / absolute pose.
//
// After RANSAC and the (score-gated) local optimizers, re-collect the inliers of the best model
// with the inlier threshold and run a robust LM refinement (PoseLib's bundle routines) on them,
// for settings.finalBundleRounds rounds: Cauchy loss, MAGSAC++ sigma-consensus weighted least
// squares, or weighted rounds followed by a Cauchy round (settings.finalBundleWeighting; auto is
// resolved per problem in the estimate* functions). The refined model is accepted unless its consensus
// set collapses (fewer than 80% of the previous inliers), in which case the previous model is kept.
// Everything here is result-dependent and identical for all inputs.
// ---------------------------------------------------------------------------------------------
namespace
{
    inline void collectInliers(
        const DataMatrix &kData_,
        const superansac::estimator::Estimator *kEstimator_,
        const superansac::models::Model &kModel_,
        const double kSquaredThreshold_,
        std::vector<size_t> &inliers_)
    {
        const size_t kRows = kData_.rows();
        const size_t kCols = kData_.cols();
        const double *row = kData_.data();
        inliers_.clear();
        inliers_.reserve(kRows);
        for (size_t i = 0; i < kRows; ++i, row += kCols)
            if (kEstimator_->squaredResidual(row, kModel_) < kSquaredThreshold_)
                inliers_.push_back(i);
    }

    inline poselib::BundleOptions finalBundleOptions(const superansac::RANSACSettings &kSettings_)
    {
        poselib::BundleOptions opt;
        // Unweighted rounds use the Cauchy loss; rounds that receive sigma-consensus weights switch to
        // weighted least squares inside the refine lambdas (see withWeights()).
        opt.loss_type = poselib::BundleOptions::LossType::CAUCHY;
        opt.loss_scale = kSettings_.finalBundleLossScale * kSettings_.inlierThreshold;
        opt.max_iterations = kSettings_.finalBundleMaxIterations;
        return opt;
    }

    // Options for one refinement round: weighted rounds solve a weighted least-squares problem.
    inline poselib::BundleOptions withWeights(const poselib::BundleOptions &kOpt_, const std::vector<double> &kWeights_)
    {
        poselib::BundleOptions opt = kOpt_;
        if (!kWeights_.empty())
            opt.loss_type = poselib::BundleOptions::LossType::TRIVIAL;
        return opt;
    }

    // Generic driver: refineFn_(inlierIndices, modelInOut, weights) -> bool (empty weights = unweighted round).
    // Round r collects the fit set with a threshold multiplier that shrinks geometrically from
    // settings.finalBundleScheduleStart to 1.0 (LO-RANSAC style iterative re-collection), so that
    // true inliers excluded by a locally tight fit can be re-admitted before the final robust fit.
    template <typename RefineFn>
    void runFinalBundle(
        const DataMatrix &kData_,
        const superansac::estimator::Estimator *kEstimator_,
        const superansac::RANSACSettings &kSettings_,
        const size_t kMinInliers_,
        superansac::models::Model &model_,
        std::vector<size_t> &inliers_,
        RefineFn refineFn_)
    {
        const size_t kRounds = kSettings_.finalBundleRounds;
        if (kRounds == 0)
            return;
        const double kSqTh = kSettings_.inlierThreshold * kSettings_.inlierThreshold;
        const double kStart = std::max(1.0, kSettings_.finalBundleScheduleStart);
        std::vector<size_t> current, fitSet, next;
        std::vector<double> weights;
        collectInliers(kData_, kEstimator_, model_, kSqTh, current);
        if (current.size() < kMinInliers_)
            return;
        const size_t kInitialInliers = current.size();
        for (size_t round = 0; round < kRounds; ++round)
        {
            const double kMult = (kRounds <= 1) ? 1.0
                : kStart * std::pow(1.0 / kStart, static_cast<double>(round) / static_cast<double>(kRounds - 1));
            collectInliers(kData_, kEstimator_, model_, kSqTh * kMult * kMult, fitSet);
            if (fitSet.size() < kMinInliers_)
                break;
            weights.clear();
            // 1: sigma-consensus weights in every round; 2: in every round but the last (which is an unweighted Cauchy fit)
            const bool kWeighted = kSettings_.finalBundleWeighting == 1 ||
                (kSettings_.finalBundleWeighting == 2 && round + 1 < kRounds);
            if (kWeighted)
            {
                // sigma-consensus (MAGSAC++) weights of the fit set at this round's scale
                superansac::scoring::MAGSACScoring magsac;
                magsac.setThreshold(kSettings_.inlierThreshold * kMult);
                magsac.initialize(kEstimator_);
                magsac.getWeights(kData_, model_, kEstimator_, weights, &fitSet);
            }
            superansac::models::Model candidate = model_;
            if (!refineFn_(fitSet, candidate, weights) || !candidate.getData().allFinite())
                break;
            collectInliers(kData_, kEstimator_, candidate, kSqTh, next);
            if (next.size() * 5 < kInitialInliers * 4 || next.size() < kMinInliers_)
                break; // diverged: keep the previous model
            model_ = candidate;
            current.swap(next);
        }
        inliers_ = current;
    }
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<size_t>, double, size_t> estimateAbsolutePose(

    const DataMatrix& kCorrespondences_, // The 2D-3D point correspondences
    const superansac::camera::CameraType &kCameraType_, // The type of the camera 
    const std::vector<double>& kCameraParams_, // The camera parameters
    const std::vector<double>& kBoundingBox_, // The bounding box dimensions (image width, image height, X, Y, Z)
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const superansac::RANSACSettings &kSettingsIn_) // The RANSAC settings (copied: the caller's object is never modified)
{
    superansac::RANSACSettings settings_ = kSettingsIn_;
    if (settings_.finalBundleWeighting < 0) // auto
        settings_.finalBundleWeighting = 1;
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
    if (!kPointProbabilities_.empty() && settings_.nonminimalConfidenceWeights)
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
    {
        auto *prosacSampler = dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get());
        prosacSampler->setSampleSize(estimator->sampleSize());
        if (settings_.prosacConvergenceMultiplier > 0.0)
            // PROSAC reaches uniform sampling after (multiplier x maxIterations) samples
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200,
                static_cast<size_t>(settings_.prosacConvergenceMultiplier * static_cast<double>(settings_.maxIterations))));
        else if (settings_.maxIterations < 1000)
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200, settings_.maxIterations * 10));
    }
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
    if (settings_.lowInlierUniformFallback && kSampler == superansac::samplers::SamplerType::PROSAC &&
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

    // Final robust bundle polish: PoseLib's absolute-pose LM (Cauchy on the reprojection error) on the re-collected inliers.
    pnpNonMinimalSolver->setWeights(nullptr);
    if (settings_.finalBundleRounds > 0)
    {
        const poselib::BundleOptions kOpt = finalBundleOptions(settings_);
        runFinalBundle(normalizedCorrespondences, estimator.get(), settings_, 4, bestModel, finalInliers,
            [&](const std::vector<size_t> &inl, superansac::models::Model &model, const std::vector<double> &w) -> bool
            {
                std::vector<Eigen::Vector2d> x(inl.size());
                std::vector<Eigen::Vector3d> X(inl.size());
                for (size_t i = 0; i < inl.size(); ++i)
                {
                    const size_t idx = inl[i];
                    x[i] = Eigen::Vector2d(normalizedCorrespondences(idx, 0), normalizedCorrespondences(idx, 1));
                    X[i] = Eigen::Vector3d(normalizedCorrespondences(idx, 2), normalizedCorrespondences(idx, 3), normalizedCorrespondences(idx, 4));
                }
                poselib::CameraPose pose;
                const Eigen::Matrix3d R = model.getData().block<3, 3>(0, 0);
                pose.q = poselib::rotmat_to_quat(R);
                pose.t = model.getData().block<3, 1>(0, 3);
                poselib::bundle_adjust(x, X, &pose, withWeights(kOpt, w), w);
                model.getMutableData().block<3, 3>(0, 0) = pose.R();
                model.getMutableData().block<3, 1>(0, 3) = pose.t;
                return true;
            });
    }
    else if (finalInliers.size() >= 4)
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
    const superansac::RANSACSettings &kSettingsIn_) // The RANSAC settings (copied: the caller's object is never modified)
{
    superansac::RANSACSettings settings_ = kSettingsIn_;
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
    {
        auto *prosacSampler = dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get());
        prosacSampler->setSampleSize(estimator->sampleSize());
        if (settings_.prosacConvergenceMultiplier > 0.0)
            // PROSAC reaches uniform sampling after (multiplier x maxIterations) samples
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200,
                static_cast<size_t>(settings_.prosacConvergenceMultiplier * static_cast<double>(settings_.maxIterations))));
        else if (settings_.maxIterations < 1000)
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200, settings_.maxIterations * 10));
    }
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
    if (settings_.lowInlierUniformFallback && kSampler == superansac::samplers::SamplerType::PROSAC &&
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
    const superansac::RANSACSettings &kSettingsIn_) // The RANSAC settings (copied: the caller's object is never modified)
{
    superansac::RANSACSettings settings_ = kSettingsIn_;
    if (settings_.finalBundleWeighting < 0) // auto: sigma-consensus weighted rounds + final unweighted Cauchy round
        settings_.finalBundleWeighting = 2;
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
    auto &solverOptions = nonminimalSolver->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::CAUCHY;
    solverOptions.loss_scale = 0.5 * settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<4>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
    {
        auto *prosacSampler = dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get());
        prosacSampler->setSampleSize(estimator->sampleSize());
        if (settings_.prosacConvergenceMultiplier > 0.0)
            // PROSAC reaches uniform sampling after (multiplier x maxIterations) samples
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200,
                static_cast<size_t>(settings_.prosacConvergenceMultiplier * static_cast<double>(settings_.maxIterations))));
        else if (settings_.maxIterations < 1000)
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200, settings_.maxIterations * 10));
    }
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
    if (settings_.lowInlierUniformFallback && kSampler == superansac::samplers::SamplerType::PROSAC &&
        settings_.maxIterations >= 500 &&
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

        superansac::RANSACSettings unifSettings = settings_;
        if (settings_.maxIterations < 5000)
        {
            unifSettings.minIterations = std::max<size_t>(15, settings_.minIterations / 2);
            unifSettings.maxIterations = std::max<size_t>(15, settings_.maxIterations / 2);
        }

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
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, totalIterations);

    // Final robust bundle polish in the normalized frame (PoseLib's 7-DoF F refinement, Cauchy loss).
    if (settings_.finalBundleRounds > 0)
    {
        const poselib::BundleOptions kOpt = finalBundleOptions(settings_);
        runFinalBundle(normalizedCorrespondences, estimator.get(), settings_, 8, bestModel, finalInliers,
            [&](const std::vector<size_t> &inl, superansac::models::Model &model, const std::vector<double> &w) -> bool
            {
                std::vector<Eigen::Vector2d> x1(inl.size()), x2(inl.size());
                for (size_t i = 0; i < inl.size(); ++i)
                {
                    const size_t idx = inl[i];
                    x1[i] = Eigen::Vector2d(normalizedCorrespondences(idx, 0), normalizedCorrespondences(idx, 1));
                    x2[i] = Eigen::Vector2d(normalizedCorrespondences(idx, 2), normalizedCorrespondences(idx, 3));
                }
                Eigen::Matrix3d F = model.getData().block<3, 3>(0, 0);
                poselib::refine_fundamental(x1, x2, &F, withWeights(kOpt, w), w);
                F.normalize();
                model.getMutableData() = F;
                return true;
            });
    }

    Eigen::Matrix3d fundamentalMatrix = bestModel.getData();

    // Transform the estimated fundamental matrix back to the not normalized space
    fundamentalMatrix = normalizingTransformDestination.transpose() * fundamentalMatrix * normalizingTransformSource;
    fundamentalMatrix.normalize();

    // Post-loop Plane-and-Parallax rescue and off-plane epipole cleaning in pixel space
    if (settings_.fundamentalPlaneParallaxPostRefinement &&
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
            const bool kCalibratedProbs = kHasProbs;
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
    const superansac::RANSACSettings &kSettingsIn_) // The RANSAC settings (copied: the caller's object is never modified)
{
    superansac::RANSACSettings settings_ = kSettingsIn_;
    if (settings_.finalBundleWeighting < 0) // auto
        settings_.finalBundleWeighting = 1;
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
    if (kPointProbabilities_.size() > 0 && settings_.nonminimalConfidenceWeights)
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
    {
        auto *prosacSampler = dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get());
        prosacSampler->setSampleSize(estimator->sampleSize());
        if (settings_.prosacConvergenceMultiplier > 0.0)
            // PROSAC reaches uniform sampling after (multiplier x maxIterations) samples
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200,
                static_cast<size_t>(settings_.prosacConvergenceMultiplier * static_cast<double>(settings_.maxIterations))));
        else if (settings_.maxIterations < 1000)
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200, settings_.maxIterations * 10));
    }
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

    superansac::models::Model bestModel = robustEstimator.getBestModel();
    superansac::scoring::Score bestScore = robustEstimator.getBestScore();
    std::vector<size_t> finalInliers = robustEstimator.getInliers();
    size_t totalIterations = robustEstimator.getIterationNumber();

    // When PROSAC finds very few inliers (<= 45) on a larger correspondence set,
    // run a Uniform-sampling fallback so low-inlier or cluttered-prefix pairs are recovered.
    if (settings_.lowInlierUniformFallback && kSampler == superansac::samplers::SamplerType::PROSAC &&
        settings_.maxIterations >= 500 &&
        finalInliers.size() <= 45 &&
        kCorrespondences_.rows() > 45)
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

        superansac::RANSACSettings unifSettings = settings_;
        if (settings_.maxIterations < 5000)
        {
            unifSettings.minIterations = std::max<size_t>(15, settings_.minIterations / 2);
            unifSettings.maxIterations = std::max<size_t>(15, settings_.maxIterations / 2);
        }

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
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, totalIterations);

    // Final robust bundle polish: refine (R, t) on the re-collected inliers in normalized image coordinates.
    if (settings_.finalBundleRounds > 0)
    {
        const poselib::BundleOptions kOpt = finalBundleOptions(settings_);
        runFinalBundle(normalizedCorrespondences, estimator.get(), settings_, 8, bestModel, finalInliers,
            [&](const std::vector<size_t> &inl, superansac::models::Model &model, const std::vector<double> &w) -> bool
            {
                const Eigen::Matrix3d E = model.getData().block<3, 3>(0, 0);
                std::vector<Eigen::Vector2d> x1(inl.size()), x2(inl.size());
                std::vector<Eigen::Vector3d> b1(inl.size()), b2(inl.size());
                for (size_t i = 0; i < inl.size(); ++i)
                {
                    const size_t idx = inl[i];
                    x1[i] = Eigen::Vector2d(normalizedCorrespondences(idx, 0), normalizedCorrespondences(idx, 1));
                    x2[i] = Eigen::Vector2d(normalizedCorrespondences(idx, 2), normalizedCorrespondences(idx, 3));
                    b1[i] = x1[i].homogeneous().normalized();
                    b2[i] = x2[i].homogeneous().normalized();
                }
                // Disambiguate the four decompositions by a cheirality vote over the inliers.
                poselib::CameraPoseVector poses;
                poselib::motion_from_essential(E, {}, {}, &poses);
                int best = -1;
                size_t bestVotes = 0;
                for (size_t p = 0; p < poses.size(); ++p)
                {
                    size_t votes = 0;
                    for (size_t i = 0; i < inl.size(); ++i)
                        votes += poselib::check_cheirality(poses[p], b1[i], b2[i]) ? 1 : 0;
                    if (votes > bestVotes)
                    {
                        bestVotes = votes;
                        best = static_cast<int>(p);
                    }
                }
                if (best < 0)
                    return false;
                poselib::CameraPose pose = poses[best];
                poselib::refine_relpose(x1, x2, &pose, withWeights(kOpt, w), w);
                Eigen::Matrix3d refined;
                poselib::essential_from_motion(pose, &refined);
                model.getMutableData() = refined;
                return true;
            });
    }

    // Get the normalized essential matrix
    Eigen::Matrix3d essentialMatrix = bestModel.getData();

    // Return the best model with the inliers and the score
    return std::make_tuple(essentialMatrix, 
        finalInliers, 
        bestScore.getValue(), 
        totalIterations);
}

std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateHomography(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    const superansac::RANSACSettings &kSettingsIn_) // The RANSAC settings (copied: the caller's object is never modified)
{
    superansac::RANSACSettings settings_ = kSettingsIn_;
    if (settings_.finalBundleWeighting < 0) // auto
        settings_.finalBundleWeighting = 0;
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
    {
        auto *prosacSampler = dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get());
        prosacSampler->setSampleSize(estimator->sampleSize());
        if (settings_.prosacConvergenceMultiplier > 0.0)
            // PROSAC reaches uniform sampling after (multiplier x maxIterations) samples
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200,
                static_cast<size_t>(settings_.prosacConvergenceMultiplier * static_cast<double>(settings_.maxIterations))));
        else if (settings_.maxIterations < 1000)
            prosacSampler->setRansacConvergenceIterations(std::max<size_t>(200, settings_.maxIterations * 10));
    }
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
    if (settings_.lowInlierUniformFallback && kSampler == superansac::samplers::SamplerType::PROSAC &&
        settings_.maxIterations >= 500 &&
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
            const size_t kTargetIters = (poolRows == 80 && kTotalRows > 80)
                ? (1500 * std::min<size_t>(settings_.maxIterations, 5000)) / 5000
                : (2500 * std::min<size_t>(settings_.maxIterations, 5000)) / 5000;
            unifSettings.minIterations = std::max<size_t>(15, std::min<size_t>(settings_.maxIterations, kTargetIters));
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
    if (settings_.finalBundleRounds > 0)
    {
        // Final robust bundle polish: PoseLib's homography LM (Cauchy on the transfer error) on the re-collected inliers.
        const poselib::BundleOptions kOpt = finalBundleOptions(settings_);
        runFinalBundle(kCorrespondences_, estimator.get(), settings_, estimator->nonMinimalSampleSize(), bestModel, finalInliers,
            [&](const std::vector<size_t> &inl, superansac::models::Model &model, const std::vector<double> &w) -> bool
            {
                std::vector<Eigen::Vector2d> x1(inl.size()), x2(inl.size());
                for (size_t i = 0; i < inl.size(); ++i)
                {
                    const size_t idx = inl[i];
                    x1[i] = Eigen::Vector2d(kCorrespondences_(idx, 0), kCorrespondences_(idx, 1));
                    x2[i] = Eigen::Vector2d(kCorrespondences_(idx, 2), kCorrespondences_(idx, 3));
                }
                Eigen::Matrix3d H = model.getData().block<3, 3>(0, 0);
                poselib::refine_homography(x1, x2, &H, withWeights(kOpt, w), w);
                if (std::abs(H(2, 2)) > 1e-12)
                    H /= H(2, 2);
                model.getMutableData() = H;
                return true;
            });
        homography = bestModel.getData();
    }
    else if (finalInliers.size() >= estimator->nonMinimalSampleSize())
    {
        // Legacy polish: one DLT + short LM refit on the re-collected inliers.
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

    // Return the best model with the inliers and the score
    return std::make_tuple(homography, 
        finalInliers, 
        bestScore.getValue(), 
        totalIterations);
}