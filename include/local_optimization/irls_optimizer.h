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
		// Templated class for estimating a model for RANSAC. This class is purely a
		// virtual class and should be implemented for the specific task that RANSAC is
		// being used for. Two methods must be implemented: estimateModel and residual. All
		// other methods are optional, but will likely enhance the quality of the RANSAC
		// output.
		class IRLSOptimizer : public LocalOptimizer
		{
		protected:
			size_t maxIterations;
			bool useInliers;

		public:
			IRLSOptimizer() : maxIterations(100), useInliers(false) 
			{

			}

			~IRLSOptimizer() {}

			// Set the maximum number of iterations
			void setMaxIterations(const size_t maxIterations_)
			{
				maxIterations = maxIterations_;
			}

			void setUseInliers(const bool kUseInliers_)
			{
				useInliers = kUseInliers_;
			}

			// The function for estimating the model parameters from the data points.
			void run(const DataMatrix &kData_, // The data points
				const std::vector<size_t> &kInliers_, // The inliers of the previously estimated model
				const models::Model &kModel_, // The previously estimated model 
				const scoring::Score &kScore_, // The of the previously estimated model
				const estimator::Estimator *kEstimator_, // The estimator used for the model estimation
				const scoring::AbstractScoring *kScoring_, // The scoring object used for the model estimation
				models::Model &estimatedModel_, // The estimated model
				scoring::Score &estimatedScore_, // The score of the estimated model
				std::vector<size_t> &estimatedInliers_) const // The inliers of the estimated model
			{
				// The invalid score
				static const scoring::Score kInvalidScore = scoring::Score();

				// The estimated models
				std::vector<models::Model> estimatedModels;
				scoring::Score currentScore = kInvalidScore;
				std::vector<double> weights(kData_.rows());

				// Initialize the estimated model and score
				estimatedModel_ = kModel_;
				estimatedScore_ = kScore_;
				
				// Clear the estimated inliers
				estimatedInliers_.clear();
				estimatedInliers_.reserve(kData_.rows());

				// Temp inliers for selecting the best model
				std::vector<size_t> tmpInliers;
				tmpInliers.reserve(kData_.rows());

				// A flag indicating if the model has been updated
				bool updated = false;

				// Calculate the score of the estimated model
				estimatedScore_ = kScoring_->score(kData_, estimatedModel_, kEstimator_, estimatedInliers_);

				for (size_t iteration = 0; iteration < maxIterations; ++iteration)
				{
					// If the model has not been updated in the previous iteration, break
					updated = false; 
					
					// Calculate the MSAC weights for the data points
					kScoring_->getWeights(kData_, estimatedModel_, kEstimator_, weights, &estimatedInliers_);
					
					estimatedModels.clear();

					if (useInliers)
					{
						if (kInliers_.size() > 0)
						{
							// Estimate the model using the inliers
							if (!kEstimator_->estimateModelNonminimal(
								kData_,  // The data points
								&kInliers_[0], 
								kInliers_.size(),
								&estimatedModels,
								nullptr))
							{
								estimatedScore_ = kInvalidScore;
								return;
							}
						}
						else
						{
							// Calculate the score of the estimated model
							currentScore = kScoring_->score(kData_, kModel_, kEstimator_, estimatedInliers_);

							// Estimate the model using the inliers
							if (!kEstimator_->estimateModelNonminimal(
								kData_,  // The data points
								&estimatedInliers_[0], 
								estimatedInliers_.size(),
								&estimatedModels,
								nullptr))
							{
								estimatedScore_ = kInvalidScore;
								return;
							}
						}
					} else if (!kEstimator_->estimateModelNonminimal( // Estimate the model using the inliers
						kData_,  // The data points
						&estimatedInliers_[0], 
						estimatedInliers_.size(),
						&estimatedModels,
						&weights[0]))
						break;

					// Calculate the scoring of the estimated model
					for (const auto &model : estimatedModels)
					{
						// Calculate the score of the estimated model
						tmpInliers.clear();
						currentScore = kScoring_->score(kData_, model, kEstimator_, tmpInliers);

						// Check if the current model is better than the previous one
						if (currentScore > estimatedScore_)
						{
							// Update the estimated model
							estimatedModel_ = model;
							estimatedScore_ = currentScore;
							tmpInliers.swap(estimatedInliers_);
							updated = true;
						}
					}

					// If the model has not been updated in the previous iteration, break
					if (!updated)
						break;
				}
			}

		};
	}
}  // namespace gcransac