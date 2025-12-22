#include "superansac.h"

#include <iostream>

namespace superansac {

SupeRansac::SupeRansac() :
    currentSample(nullptr),
    localOptimizer(nullptr),
    finalOptimizer(nullptr)
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
    currentModels.clear(); // Clearing the current models

    // Reserve memory for the inliers if needed (already reserved in constructor)
    if (inliers.capacity() < kData_.rows())
        inliers.reserve(kData_.rows());
    if (tmpInliers.capacity() < kData_.rows())
        tmpInliers.reserve(kData_.rows());
    std::vector<const std::vector<size_t>*> potentialInlierSets; // The potential inlier sets from the inlier selector
    std::vector<std::tuple<models::Model, std::vector<size_t>, scoring::Score>> bestModels;

    // Current sample
    const size_t kSampleSize = estimator->sampleSize();

    // Allocate memory for the current sample
    if (currentSample != nullptr)
        delete[] currentSample;
    currentSample = new size_t[kSampleSize];

    // Iterate until the maximum number of iterations is reached and the minimum number of iterations is exceeded
    while (iterationNumber < maxIterations || iterationNumber < minIterations)
    {
        unsuccessfulModelGenerations = -1; 

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
                // Update the best model
                bestScore = currentScore;
                bestModel = model;
                inliers.swap(tmpInliers);
                isModelUpdated = true;
                bestModels.emplace_back(std::make_tuple(model, inliers, currentScore));
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
                    inliers.swap(tmpInliers);
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

            if (bestScore < currentScore)
            {
                // Update the best model
                bestScore = currentScore;
                bestModel = locallyOptimizedModel;
                inliers.swap(tmpInliers);
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

        // Update the best model
        if (currentScore.getInlierNumber() > kSampleSize)
        {
            bestScore = currentScore;
            bestModel = locallyOptimizedModel;
            inliers.swap(tmpInliers);
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