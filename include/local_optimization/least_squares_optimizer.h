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
		class LeastSquaresOptimizer : public LocalOptimizer
		{
		protected:
			bool useInliers;

		public:
			LeastSquaresOptimizer() : useInliers(false) {}
			~LeastSquaresOptimizer() {}

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
				static const scoring::Score kInvalidScore = scoring::Score();
				// The estimated models
				std::vector<models::Model> estimatedModels;
				scoring::Score currentScore;

				estimatedModels.emplace_back(kModel_);

				if (useInliers)
				{
					//std::cout << " - " << kInliers_.size() << std::endl;
					//std::cout << " - " << kScore_.getValue() << std::endl;
					if (kInliers_.size() > 0)
					{
						/*std::cout << "FO" << std::endl;
						std::cout << estimatedModels[0].getData() << std::endl;
						std::cout << kInliers_.size() << std::endl;

						for (const auto &inlier : kInliers_)
							std::cout << inlier << " ";
						std::cout << std::endl;*/

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

						//std::cout << estimatedModels[0].getData() << std::endl;
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
					//std::cout << " - " << estimatedScore_.getValue() << std::endl;
				} else if (!kEstimator_->estimateModelNonminimal(
					kData_,  // The data points
					nullptr, 
					kData_.rows(),
					&estimatedModels,
					nullptr))
				{
					estimatedScore_ = kInvalidScore;
					return;
				}

				// Clear the estimated inliers
				estimatedInliers_.clear();

				// Temp inliers for selecting the best model
				std::vector<size_t> tmpInliers;
				tmpInliers.reserve(kData_.rows());

				//std::cout << " - " << estimatedModels.size() << std::endl;

				// Calculate the scoring of the estimated model
				for (const auto &model : estimatedModels)
				{
					// Calculate the score of the estimated model
					tmpInliers.clear();
					currentScore = kScoring_->score(kData_, model, kEstimator_, tmpInliers);

					/*if (useInliers)
					{
						std::cout << " - " << estimatedScore_.getInlierNumber() << std::endl;
						std::cout << " - " << estimatedScore_.getValue() << std::endl;
						std::cout << " - " << currentScore.getInlierNumber() << std::endl;
						std::cout << " - " << currentScore.getValue() << std::endl;
						std::cout << " - " << model.getData() << std::endl;
					}*/

					// Check if the current model is better than the previous one
					if (useInliers || estimatedScore_ < currentScore)
					{
						//std::cout << " - " << currentScore.getInlierNumber() << std::endl;
						//std::cout << " - " << currentScore.getValue() << std::endl;
						// Update the estimated model
						estimatedModel_ = model;
						estimatedScore_ = currentScore;
						tmpInliers.swap(estimatedInliers_);
						//std::cout << " - " << estimatedModel_.getData() << std::endl;
					}
				}
			}

		};
	}
}  // namespace gcransac