// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <vector>
#include <Eigen/Core>
#include "abstract_local_optimizer.h"
#include "../utils/types.h"

namespace superansac
{
	namespace local_optimization
	{
		// This class implements a local optimization strategy based on cross-validation.
        // The core idea is to generate weights for the inliers of a given model.
        // These weights are derived by repeatedly building a new model from a bootstrap
        // sample of the inliers and evaluating its performance (error) against the
        // entire inlier set. Points that consistently yield low errors across many
        // bootstrap-trained models are considered more reliable and receive higher weights.
        // Finally, a single model is refit using a weighted least-squares approach
        // with all inliers and their calculated weights.
        class CrossValidationOptimizer : public LocalOptimizer
        {
        protected:
            size_t repetitions; // The number of bootstrap repetitions to perform.
			double sampleSizeMultiplier; // Multiplier for the sample size used in bootstrap sampling
			bool useInliers; // Whether to use inliers for the optimization

        public:
            CrossValidationOptimizer() : CrossValidationOptimizer(100, 0.5)
			{
			}
			
            CrossValidationOptimizer(const size_t repetitions_, const double sampleSizeMultiplier_)
				: repetitions(repetitions_), sampleSizeMultiplier(sampleSizeMultiplier_), useInliers(false)
			{
			}

            ~CrossValidationOptimizer() {}

            // Set the number of repetitions for the cross-validation process.
            void setRepetitions(const size_t repetitions_)
            {
                repetitions = repetitions_;
            }
			
			// Set the sample size multiplier for the bootstrap sampling.
            void setSampleSizeMultiplier(const size_t sampleSizeMultiplier_)
			{
				sampleSizeMultiplier = sampleSizeMultiplier_;
			}

			// Set whether to use inliers for the optimization.
			void setUseInliers(const bool kUseInliers_)
			{
				useInliers = kUseInliers_;
			}

            // The main function to run the cross-validation and weighted refitting.
            void run(const DataMatrix &kData_,                   // All data points
                     const std::vector<size_t> &kInliers_,      // Inlier indices from the initial model
                     const models::Model &kModel_,              // The initial model
                     const scoring::Score &kScore_,             // The score of the initial model
                     const estimator::Estimator *kEstimator_,   // The model estimator
                    scoring::AbstractScoring *kScoring_, // The scoring function
                     models::Model &estimatedModel_,            // Output: the refined model
                     scoring::Score &estimatedScore_,           // Output: the score of the refined model
                     std::vector<size_t> &estimatedInliers_) const // Output: inliers of the refined model
            {
                // Initialize outputs with the initial model and score.
                // They will be updated only if a better model is found.
                estimatedModel_ = kModel_;
                estimatedScore_ = kScore_;
                estimatedInliers_ = kInliers_;

                const size_t kInlierCount = kInliers_.size();
                const size_t kMinimalSampleSize = kEstimator_->sampleSize();
				const double kThreshold = kScoring_->getThreshold();

                // Not enough inliers to build a model, so we can't proceed.
                if (kInlierCount < kMinimalSampleSize)
                    return;

                // Determine the size of each bootstrap sample.
                const size_t kSampleSize = 
					std::max(kMinimalSampleSize, static_cast<size_t>(sampleSizeMultiplier * kInlierCount));
                std::vector<double> kAccumulatedScores(kInlierCount, 0.0);

                // Setup for random sampling with replacement.
                std::random_device randomDevice;
                std::mt19937 random_generator(randomDevice());
                std::uniform_int_distribution<size_t> distribution(0, kInlierCount - 1);

                // --- Cross-Validation Loop ---
                // In each repetition, a model is estimated from a random subset of inliers
                // and then used to score all inliers.
				std::vector<size_t> currentSample(kSampleSize);
				std::vector<models::Model> bootstrapModels;
				DataMatrix bootstrapData;
				bootstrapModels.reserve(1); // Reserve space for one model.

				if (useInliers)
					bootstrapData.resize(kSampleSize, kData_.cols());

                for (size_t i = 0; i < repetitions; ++i)
                {
                    // 1. Create a bootstrap sample from the inliers.
					for (size_t j = 0; j < kSampleSize; ++j)
						currentSample[j] = kInliers_[distribution(random_generator)];
							
					if (useInliers)
						for (size_t j = 0; j < kSampleSize; ++j)
							bootstrapData.row(j) = kData_.row(currentSample[j]);
						
                    // 2. Estimate a model from the bootstrap sample.
					bootstrapModels.clear();
                    
					if (useInliers)
						kEstimator_->estimateModelNonminimal(
								bootstrapData, 
								nullptr, // No sample provided, use all points in bootstrapData
								kSampleSize,
								&bootstrapModels);
					else
						kEstimator_->estimateModelNonminimal(
								kData_, 
								&currentSample[0],
								kSampleSize,
								&bootstrapModels);

					if (bootstrapModels.empty())
						continue; // No model was estimated, try next repetition.

					// 3. Get the current model from the bootstrap sample.
                    const models::Model &kCurrentModel = 
						bootstrapModels[0];

                    // 4. Calculate and accumulate scores for all original inliers.
                    for (size_t j = 0; j < kInlierCount; ++j)
                    {
                        const double kError = 
							kEstimator_->residual(kData_.row(kInliers_[j]), kCurrentModel);
                        const double kScore = 
							std::max(0.0, 1.0 - kError / kThreshold);
                        kAccumulatedScores[j] += kScore;
                    }
                }

                // --- Weighted Least-Squares Fitting ---
                // 5. Calculate final weights for each inlier.
                std::vector<double> weights(kInlierCount);
                for (size_t j = 0; j < kInlierCount; ++j)
                    weights[j] = kAccumulatedScores[j] / repetitions;
                    
                // 6. Refit the model using all inliers and their calculated weights.
                DataMatrix finalData;
                std::vector<models::Model> finalModels;
				if (useInliers)
                {
					finalData.resize(kInlierCount, kData_.cols());
                    for (size_t j = 0; j < kInlierCount; ++j)
                        finalData.row(j) = kData_.row(kInliers_[j]);

                    if (!kEstimator_->estimateModelNonminimal(finalData,        // All data points
                                                        nullptr, // The original inlier indices
                                                        kInlierCount,
                                                        &finalModels, // Output models
                                                        &weights[0]))  // The calculated weights
                        return; // Weighted fitting failed.
                } else if (!kEstimator_->estimateModelNonminimal(kData_,        // All data points
                                                          &kInliers_[0], // The original inlier indices
                                                          kInlierCount,
                                                          &finalModels, // Output models
                                                          &weights[0]))  // The calculated weights
                    return; // Weighted fitting failed.

                // 7. Find the best model among the results and update if it's better than the initial one.
                std::vector<size_t> currentInliers;
                currentInliers.reserve(kData_.rows());
                
                for (const auto &model : finalModels)
                {
                    currentInliers.clear();
                    scoring::Score currentScore = 
						kScoring_->score(kData_, model, kEstimator_, currentInliers);

                    if (currentScore > estimatedScore_)
                    {
                        estimatedScore_ = currentScore;
                        estimatedModel_ = model;
                        estimatedInliers_.swap(currentInliers);
                    }
                }
            }
        };
	}
}  // namespace gcransac