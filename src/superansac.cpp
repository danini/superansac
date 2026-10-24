#include "superansac.h"

#include "estimators/degensac.h"
#include "estimators/estimator_fundamental_matrix.h"
#include "estimators/estimator_rigid_transformation.h"

#include <iostream>

namespace superansac {

SupeRansac::SupeRansac() :
    sampler(nullptr),
    estimator(nullptr),
    scoring(nullptr),
    localOptimizer(nullptr),
    finalOptimizer(nullptr),
    terminationCriterion(nullptr),
    inlierSelector(nullptr),
    currentSample(nullptr)
{
    // Reserve memory for inliers to avoid reallocations
    inliers.reserve(10000);
    tmpInliers.reserve(10000);
}

SupeRansac::~SupeRansac()
{
    // Clean up current sample if it exists
    if (currentSample != nullptr)
    {
        delete[] currentSample;
        currentSample = nullptr;
    }
}

void SupeRansac::run(const DataMatrix &kData_)
{
    // Check if the sampler is set
    if (sampler == nullptr)
        throw std::runtime_error("The sampler has not been set.");

    // Initialize the variables
    const double &kThreshold_ = scoring->getThreshold();
    bool isModelUpdated = false;  // Single declaration, initialized
    bool immediateTermination = false;
    const size_t kStrickIterationLimit = settings.maxIterations;
    int unsuccessfulModelGenerations;
    size_t pointNumber;
    iterationNumber = 0; // The iteration number
    minIterations = settings.minIterations; // The minimum number of iterations
    maxIterations = kStrickIterationLimit; // The maximum number of iterations
    bestScore = scoring::Score(); // The best score
    bestModel = models::Model(); // Reset the best model
    inliers.clear(); // Reset the best inliers
    tmpInliers.clear();
    currentModels.clear(); // Clearing the current models

    // Reserve memory for the inliers if needed (already reserved in constructor)
    if (inliers.capacity() < kData_.rows())
        inliers.reserve(kData_.rows());
    if (tmpInliers.capacity() < kData_.rows())
        tmpInliers.reserve(kData_.rows());
    std::vector<const std::vector<size_t>*> potentialInlierSets; // The potential inlier sets from the inlier selector
    std::vector<std::tuple<models::Model, std::vector<size_t>, scoring::Score>> bestModels;
    candidateModels.clear();

    // Current sample
    const size_t kSampleSize = estimator->sampleSize();

    // Allocate memory for the current sample
    if (currentSample != nullptr)
        delete[] currentSample;
    currentSample = new size_t[kSampleSize];

    // Fundamental-matrix-specific validation/degeneracy handling
    const estimator::FundamentalMatrixEstimator *kFundamentalEstimator =
        dynamic_cast<const estimator::FundamentalMatrixEstimator *>(estimator);
    // DEGENSAC (dominant-plane degeneracy handling) applies only to
    // fundamental-matrix estimation from 7-point minimal samples.
    const bool kUseDegensac = settings.degensacSettings.enabled &&
        kSampleSize == 7 &&
        kFundamentalEstimator != nullptr;
    const double kSquaredHomographyThreshold =
        settings.degensacSettings.homographyThreshold * settings.degensacSettings.homographyThreshold;
    // Run-scoped RNG (fixed seed) keeps the pipeline deterministic.
    utils::UniformRandomGenerator<size_t> degensacRng;
    // Symmetric-epipolar validation of new-best models: a stable F must keep
    // most of its Sampson inliers under the (less degeneracy-prone) symmetric
    // epipolar distance as well.
    const bool kUseSymmetricValidation =
        settings.fundamentalSymmetricValidation && kFundamentalEstimator != nullptr;
    const estimator::RigidTransformationEstimator *kRigidEstimator =
        dynamic_cast<const estimator::RigidTransformationEstimator *>(estimator);
    const double kSquaredThreshold = kThreshold_ * kThreshold_;
    const double kFourSquaredThreshold = 4.0 * kSquaredThreshold;

    // Iterate until the maximum number of iterations is reached and the minimum number of iterations is exceeded
    while (iterationNumber < maxIterations || iterationNumber < minIterations)
    {
        unsuccessfulModelGenerations = -1; 
        currentModels.clear();

        // Select a minimal sample and estimate the implied model parameters if possible.
        // If, after a certain number of sample selections, there is no success, terminate.
        while (++unsuccessfulModelGenerations < 100)
        {
            // Sample minimal set
            if (!sampler->sample(kData_, // Data matrix
                kSampleSize, // Selected minimal sample 
                currentSample)) // Sample indices
            {
                sampler->update(
                    currentSample,
                    kSampleSize,
                    iterationNumber,
                    0.0);
                continue;
            }

            // Check if the sample is valid
            if (!estimator->isValidSample(kData_, // Data matrix
                currentSample)) // The sample
            {
                sampler->update(
                    currentSample,
                    kSampleSize,
                    iterationNumber,
                    0.0);
                continue;
            }

            // Remove the previous models
            currentModels.clear();

            // Estimate the model
            if (!estimator->estimateModel(kData_, // Data matrix
                currentSample, // Selected minimal sample
                &currentModels)) // Estimated models
            {
                sampler->update(
                    currentSample,
                    kSampleSize,
                    iterationNumber,
                    0.0);
                continue;
            }

            // If no continue statement was executed, the model was successfully estimated
            break;
        }

        // Iterate through the models
        isModelUpdated = false;  // Reset flag for this iteration
        for (auto &model : currentModels) 
        {
            // Check if the model is valid
            if (!estimator->isValidModel(model, // The model parameters
				kData_, // All data points
				currentSample, // The indices of the selected points
				kThreshold_, // The inlier-outlier threshold
				isModelUpdated)) // The model is updated
                continue;

            // Select potential inliers if needed
            tmpInliers.clear();
            double tmpScore;
            if (settings.inlierSelector == inlier_selector::InlierSelectorType::SpacePartitioningRANSAC)
            {
                potentialInlierSets.clear();
                inlierSelector->run(kData_, // Data matrix
                    model, // The model
                    scoring, // The inlier-outlier threshold
                    potentialInlierSets, // The inliers
                    pointNumber);
                    
                if (pointNumber < bestScore.getInlierNumber())
                    continue;

                // Score the model considering the potential inliers
                currentScore = scoring->score(kData_, // Data matrix
                    model, // The model to be scored
                    estimator, // Estimator
                    tmpInliers, // Inlier indices
                    true, // Store inliers
                    bestScore, // The best score
                    &potentialInlierSets); // The potential inliers
                    
            } else 
                // Score the model
                currentScore = scoring->score(kData_, // Data matrix
                    model, // The model to be scored
                    estimator, // Estimator
                    tmpInliers, // Inlier indices
                    true, // Store inliers
                    bestScore); // The best score

            if (bestScore < currentScore)
            {
                // Validate the would-be best model with the symmetric epipolar
                // distance: require at least half of its Sampson inliers (and at
                // least a minimal sample) to also pass the symmetric test.
                if (kUseSymmetricValidation)
                {
                    const size_t kRequired = std::max(kSampleSize, tmpInliers.size() / 2);
                    size_t passing = 0;
                    const auto &kDescriptor = model.getData();
                    for (const size_t &idx : tmpInliers)
                        if (kFundamentalEstimator->squaredSymmetricEpipolarDistance(
                                kData_.row(idx).data(), kDescriptor) < kSquaredThreshold &&
                            ++passing >= kRequired)
                            break;
                    if (passing < kRequired)
                        continue; // Reject this model; keep the previous best
                }

                // 3D surface concentration (halo) check for rigid transformation:
                // a genuine surface alignment concentrates inliers inside r < tau with
                // a sparse outer shell [tau, 2*tau), whereas a false mode inside a
                // diffuse 3D outlier cloud has more points in [tau, 2*tau) than [0, tau).
                if (kRigidEstimator != nullptr && tmpInliers.size() >= 12)
                {
                    const auto &kDescriptor = model.getData();
                    const size_t kMaxHalo = (tmpInliers.size() * 2) / 3;
                    size_t haloCount = 0;
                    const int kRows = kData_.rows();
                    const int kCols = kData_.cols();
                    const double *rowPtr = kData_.data();
                    for (int rIdx = 0; rIdx < kRows; ++rIdx, rowPtr += kCols)
                    {
                        const double sqRes = kRigidEstimator->squaredResidual(rowPtr, kDescriptor);
                        if (sqRes >= kSquaredThreshold && sqRes < kFourSquaredThreshold)
                        {
                            if (++haloCount > kMaxHalo)
                                break;
                        }
                    }
                    if (haloCount > kMaxHalo)
                        continue;
                }

                // DEGENSAC: test the candidate minimal-sample model for
                // dominant-plane degeneracy before accepting it as the new
                // best. A degenerate candidate is either replaced by a
                // plane-and-parallax re-estimation that beats the best score,
                // or rejected entirely (keeping it would lock the search onto
                // the dominant plane).
                if (kUseDegensac)
                {
                    if (!degensac::applyDegensac(kData_,
                            estimator,
                            scoring,
                            currentSample,
                            kSquaredHomographyThreshold,
                            settings.degensacSettings.innerRansacIterations,
                            bestScore,
                            degensacRng,
                            model,
                            currentScore,
                            tmpInliers))
                        continue; // Reject the H-degenerate candidate
                    if (!(bestScore < currentScore))
                        continue; // The replacement does not beat the best model
                }

                // Update the best model
                bestScore = currentScore;
                bestModel = model;
                inliers.swap(tmpInliers);  // Swap keeps tmpInliers' capacity for reuse
                isModelUpdated = true;

                bestModels.emplace_back(std::make_tuple(bestModel, inliers, bestScore));
            }
        }

        // Perform local optimization inside the loop if enabled
        if (settings.localOptimizationInsideTheLoop && isModelUpdated)
        {
            // Perform local optimization if needed
            if (localOptimizer != nullptr)
            {
                tmpInliers.clear();
                localOptimizer->run(kData_, // Data matrix
                    inliers, // Inliers
                    bestModel, // The best model
                    bestScore, // The score of the best model
                    estimator, // Estimator
                    scoring, // Scoring object
                    locallyOptimizedModel, // The locally optimized model
                    currentScore, // The score of the current model
                    tmpInliers); // The inliers of the estimated model

                if (bestScore < currentScore)
                {
                    // Update the best model
                    bestScore = currentScore;
                    bestModel = locallyOptimizedModel;
                    inliers.swap(tmpInliers);  // Swap keeps tmpInliers' capacity for reuse
                }
            }

            // Update the termination criterion
            terminationCriterion->check(
                kData_, // Data matrix
                bestScore, // The score of the best model
                kSampleSize, // The sample size
                maxIterations, // The iteration number
                immediateTermination); // Immediate termination flag

            if (immediateTermination)
                break;

            // Update the maximum number of iterations
            if (maxIterations > kStrickIterationLimit)
                maxIterations = kStrickIterationLimit;
        }

        // Update the sampler
        sampler->update(
            currentSample,
            kSampleSize,
            iterationNumber,
            0.0);

        // Increase the iteration number
        iterationNumber++;

        if (settings.useSprt)
            updateSprt(scoring, isModelUpdated, bestScore, iterationNumber, kData_.rows());
    }

    if (localOptimizer != nullptr)
    {
        // Scratch for the optional per-candidate final polish
        models::Model polishedModel;
        scoring::Score polishedScore;
        std::vector<size_t> polishedInliers;
        const bool kPolishTopK = settings.finalPolishTopK && finalOptimizer != nullptr;

        const int kLastIdxToCheck =
            bestModels.size() >= settings.topKForLocalOptimization ?
                bestModels.size() - settings.topKForLocalOptimization : 0;
        for (int idx = bestModels.size() - 1; idx >= kLastIdxToCheck; --idx)
        {
            const auto &[currModel, currInliers, currScore] = bestModels[idx];

            tmpInliers.clear();
            localOptimizer->run(kData_, // Data matrix
                currInliers, // Inliers
                currModel, // The best model
                currScore, // The score of the best model
                estimator, // Estimator
                scoring, // Scoring object
                locallyOptimizedModel, // The locally optimized model
                currentScore, // The score of the current model
                tmpInliers); // The inliers of the estimated model

            // Optionally polish every locally optimized candidate with the
            // final optimizer BEFORE the best-model comparison, so a candidate
            // that overtakes the others only after refinement can still win.
            if (kPolishTopK && tmpInliers.size() > kSampleSize)
            {
                polishedInliers.clear();
                finalOptimizer->run(kData_,
                    tmpInliers,
                    locallyOptimizedModel,
                    currentScore,
                    estimator,
                    scoring,
                    polishedModel,
                    polishedScore,
                    polishedInliers);

                if (polishedScore.getInlierNumber() > kSampleSize && currentScore < polishedScore)
                {
                    locallyOptimizedModel = polishedModel;
                    currentScore = polishedScore;
                    tmpInliers.swap(polishedInliers);
                }
            }

            if (tmpInliers.size() > kSampleSize && locallyOptimizedModel.getData().allFinite())
                candidateModels.emplace_back(locallyOptimizedModel);

            if (bestScore < currentScore)
            {
                if (kRigidEstimator != nullptr && tmpInliers.size() >= 12)
                {
                    const auto &kDescriptor = locallyOptimizedModel.getData();
                    const size_t kMaxHalo = tmpInliers.size() / 2;
                    size_t haloCount = 0;
                    const int kRows = kData_.rows();
                    const int kCols = kData_.cols();
                    const double *rowPtr = kData_.data();
                    for (int rIdx = 0; rIdx < kRows; ++rIdx, rowPtr += kCols)
                    {
                        const double sqRes = kRigidEstimator->squaredResidual(rowPtr, kDescriptor);
                        if (sqRes >= kSquaredThreshold && sqRes < kFourSquaredThreshold)
                        {
                            if (++haloCount > kMaxHalo)
                                break;
                        }
                    }
                    if (haloCount > kMaxHalo)
                        continue;
                }

                // Update the best model
                bestScore = currentScore;
                bestModel = locallyOptimizedModel;
                inliers.swap(tmpInliers);  // Swap keeps tmpInliers' capacity for reuse
            }
        }
    }

    // Perform final optimization if needed
    if (finalOptimizer != nullptr && inliers.size() > kSampleSize)
    {
        tmpInliers.clear();
        finalOptimizer->run(kData_, // Data matrix
            inliers, // Inliers
            bestModel, // The best model
            bestScore, // The score of the best model
            estimator, // Estimator
            scoring, // Scoring object
            locallyOptimizedModel, // The locally optimized model
            currentScore, // The score of the current model
            tmpInliers); // The inliers of the estimated model

        // Update the best model (check both inlier count AND score value)
        if (currentScore.getInlierNumber() > kSampleSize && bestScore < currentScore)
        {
            bestScore = currentScore;
            bestModel = locallyOptimizedModel;
            inliers.swap(tmpInliers);  // Swap keeps tmpInliers' capacity for reuse
        }
    }

    // Clean up
    delete[] currentSample;
    currentSample = nullptr;
}

void SupeRansac::updateSprt(
    scoring::AbstractScoring *scoring_, 
    const bool kIsModelUpdated_, 
    const scoring::Score &kBestScore_, 
    const size_t kIterationNumber_, 
    const size_t kPointNumber_)
{
    const static scoring::Score kEmptyScore = scoring::Score();
    if (settings.scoring == scoring::ScoringType::MAGSAC)
    {
        if (kIsModelUpdated_) 
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scoring_)->updateSPRTParameters(kBestScore_, kIterationNumber_, kPointNumber_);
        else
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scoring_)->updateSPRTParameters(kEmptyScore, kIterationNumber_, kPointNumber_);
    } else
    {
        // TODO: implement
    }
}

/*
    Setters and getters
*/
// Set the scoring object
void SupeRansac::setScoring(scoring::AbstractScoring *scoring_)
{
    scoring = scoring_;
}

// Return a constant pointer to the scoring object
const scoring::AbstractScoring *SupeRansac::getScoring() const
{
    return scoring;
}

// Return a mutable pointer to the scoring object
scoring::AbstractScoring *SupeRansac::getMutableScoring()
{
    return scoring;
}

// Set the sampler
void SupeRansac::setSampler(samplers::AbstractSampler *sampler_)
{
    sampler = sampler_;
}

// Return a constant pointer to the sampler
const samplers::AbstractSampler *SupeRansac::getSampler() const
{
    return sampler;
}

// Return a mutable pointer to the sampler
samplers::AbstractSampler *SupeRansac::getMutableSampler()
{
    return sampler;
}

// Set the settings
void SupeRansac::setSettings(const RANSACSettings &kSettings_)
{
    settings = kSettings_;
}

// Return the settings
const RANSACSettings &SupeRansac::getSettings() const
{
    return settings;
}

// Return a mutable reference to the settings
RANSACSettings &SupeRansac::getMutableSettings()
{
    return settings;
}

// Set the estimator
void SupeRansac::setEstimator(estimator::Estimator *estimator_)
{
    estimator = estimator_;
}

// Return a constant pointer to the estimator
const estimator::Estimator *SupeRansac::getEstimator() const
{
    return estimator;
}

// Return a mutable pointer to the estimator
estimator::Estimator *SupeRansac::getMutableEstimator()
{
    return estimator;
}

// Get the best model
const models::Model &SupeRansac::getBestModel() const
{
    return bestModel;
}

// Get the inliers of the best model
const std::vector<size_t> &SupeRansac::getInliers() const
{
    return inliers;
}

// Get the score of the best model
const scoring::Score &SupeRansac::getBestScore() const
{
    return bestScore;
}

const std::vector<models::Model> &SupeRansac::getCandidateModels() const
{
    return candidateModels;
}

// Get the number of iterations
size_t SupeRansac::getIterationNumber() const
{
    return iterationNumber;
}

// Set the local optimization object
void SupeRansac::setLocalOptimizer(local_optimization::LocalOptimizer *localOptimizer_)
{
    localOptimizer = localOptimizer_;
}

// Return a constant pointer to the local optimization object
const local_optimization::LocalOptimizer *SupeRansac::getLocalOptimizer() const
{
    return localOptimizer;
}

// Return a mutable pointer to the local optimization object
local_optimization::LocalOptimizer *SupeRansac::getMutableLocalOptimizer()
{
    return localOptimizer;
}

// Set the local optimization object
void SupeRansac::setFinalOptimizer(local_optimization::LocalOptimizer *finalOptimizer_)
{
    finalOptimizer = finalOptimizer_;
}

// Return a constant pointer to the local optimization object
const local_optimization::LocalOptimizer *SupeRansac::getFinalOptimizer() const
{
    return finalOptimizer;
}

// Return a mutable pointer to the local optimization object
local_optimization::LocalOptimizer *SupeRansac::getMutableFinalOptimizer()
{
    return finalOptimizer;
}

// Set the termination criterion object
void SupeRansac::setTerminationCriterion(termination::AbstractCriterion *terminationCriterion_)
{
    terminationCriterion = terminationCriterion_;
}

// Return a constant pointer to the termination criterion object
const termination::AbstractCriterion *SupeRansac::getTerminationCriterion() const
{
    return terminationCriterion;
}

// Return a mutable pointer to the termination criterion object
termination::AbstractCriterion *SupeRansac::getMutableTerminationCriterion()
{
    return terminationCriterion;
}

// Create the space partitioning inlier selector object
void SupeRansac::setInlierSelector(inlier_selector::AbstractInlierSelector *inlierSelector_)
{
    inlierSelector = inlierSelector_;
}

// Return a constant pointer to the space partitioning inlier selector object
const inlier_selector::AbstractInlierSelector *SupeRansac::getInlierSelector() const
{
    return inlierSelector;
}

// Return a mutable pointer to the space partitioning inlier selector object
inlier_selector::AbstractInlierSelector *SupeRansac::getMutableInlierSelector()
{
    return inlierSelector;
}
}