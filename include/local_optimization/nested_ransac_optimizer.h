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
#include "../samplers/uniform_random_sampler.h"
#include "../utils/types.h"

namespace superansac
{
	namespace local_optimization
	{
		// Templated class for estimating a model for RANSAC. This class is purely a
		// virtual class and should be implemented for the specific task that RANSAC is
		// being used for. Two methods must be implemented: estimateModel and residual. All
		// other methods are optional, but will likely enhance the quality of the RANSAC
		// output.
		class NestedRANSACOptimizer : public LocalOptimizer
		{
		protected:
			size_t maxIterations;
			size_t sampleSizeMultiplier;

		public:
			NestedRANSACOptimizer() : maxIterations(50), 
				sampleSizeMultiplier(7)
			{

			}

			~NestedRANSACOptimizer() {}

			// Set the maximum number of iterations
			void setMaxIterations(const size_t maxIterations_)
			{
				maxIterations = maxIterations_;
			}

			// Set the sample size multiplier
			void setSampleSizeMultiplier(const size_t sampleSizeMultiplier_)
			{
				sampleSizeMultiplier = sampleSizeMultiplier_;
			}
			
			// The function for estimating the model parameters from the data points.
			void run(const DataMatrix &kData_, // The data points
				const std::vector<size_t> &kInliers_, // The inliers of the previously estimated model
				const models::Model &kModel_, // The previously estimated model
				const scoring::Score &kScore_, // The of the previously estimated model
				const estimator::Estimator *kEstimator_, // The estimator used for the model estimation
				scoring::AbstractScoring *kScoring_, // The scoring object used for the model estimation
				models::Model &estimatedModel_, // The estimated model
				scoring::Score &estimatedScore_, // The score of the estimated model
				std::vector<size_t> &estimatedInliers_) const // The inliers of the estimated model
			{
				// The invalid score
				static const scoring::Score kInvalidScore = scoring::Score();

				// Initialize the estimated model and score
				estimatedModel_ = kModel_;

				// The size of the non-minimal samples
				const size_t kNonMinimalSampleSize = sampleSizeMultiplier * kEstimator_->sampleSize();
				size_t currentSampleSize;

				// The currently estimated models - reserve to avoid reallocations
				std::vector<models::Model> currentlyEstimatedModels;
				currentlyEstimatedModels.reserve(10); // Reserve space for typical number of models
				scoring::Score currentScore = kInvalidScore;
				std::vector<size_t> currentInliers;
				currentInliers.reserve(kData_.rows());

				// Use vector instead of raw pointer for automatic memory management
				std::vector<size_t> currentSample(kNonMinimalSampleSize);

				// The sampler used for selecting minimal samples - initialize once outside loop
				samplers::UniformRandomSampler sampler;

				// Calculate the score of the estimated model
				estimatedScore_ = kScoring_->score(kData_, estimatedModel_, kEstimator_, estimatedInliers_);
				sampler.initialize(estimatedInliers_.size() - 1);
				bool isModelUpdated;

				const size_t kMinSampleSize = kEstimator_->sampleSize();

				// The inner RANSAC loop
				for (size_t iteration = 0; iteration < maxIterations; ++iteration)
				{
					isModelUpdated = false;

					// Calculate the current sample size
					const size_t inlierCount = estimatedInliers_.size();
					currentSampleSize = std::min(inlierCount - 1, kNonMinimalSampleSize);

					// Break if the sample size is too small
					if (currentSampleSize < kMinSampleSize)
						break;

					// Check if the sample size is equal to the number of inliers
					if (currentSampleSize == inlierCount)
					{
						// Copy the inliers to the current sample (use memcpy or std::copy for speed)
						std::copy(estimatedInliers_.begin(),
						         estimatedInliers_.begin() + currentSampleSize,
						         currentSample.begin());
					} else
					{
						// Sample minimal set
						if (!sampler.sample(inlierCount, // Data matrix
							currentSampleSize, // Selected minimal sample
							currentSample.data())) // Sample indices
							continue;

						// Use pointer arithmetic for better performance
						const size_t* inlierData = estimatedInliers_.data();
						for (size_t sampleIdx = 0; sampleIdx < currentSampleSize; ++sampleIdx)
							currentSample[sampleIdx] = inlierData[currentSample[sampleIdx]];
					}

					// Remove the previous models
					currentlyEstimatedModels.clear();

					// Estimate the model
					if (!kEstimator_->estimateModelNonminimal(
						kData_,  // The data points
						currentSample.data(), // Selected minimal sample
						currentSampleSize, // The size of the minimal sample
						&currentlyEstimatedModels, // The estimated models
						nullptr)) // The indices of the inliers
						continue;

					// Calculate the scoring of the estimated model
					for (const auto &model : currentlyEstimatedModels)
					{
						// Calculate the score of the estimated model
						currentInliers.clear();
						currentScore = kScoring_->score(kData_, model, kEstimator_, currentInliers);

						// Check if the current model is better than the previous one
						if (currentScore > estimatedScore_)
						{
							// Update the estimated model
							isModelUpdated = true;
							estimatedModel_ = model;
							estimatedScore_ = currentScore;
							currentInliers.swap(estimatedInliers_);
							// Only reinitialize sampler if inlier count changed significantly
							sampler.initialize(estimatedInliers_.size() - 1);
						}
					}

					// Update SPRT parameters (only if model was updated to avoid unnecessary work)
					if (isModelUpdated)
						kScoring_->updateSPRTParameters(estimatedScore_, -1, kData_.rows());
					else
						kScoring_->updateSPRTParameters(scoring::Score(), -1, kData_.rows());
				}

				// No cleanup needed - vector handles memory automatically
			}

		};
	}
}  // namespace gcransac