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
			// Number of additional score-gated refit rounds: after the first fit,
			// the inliers are re-collected from the refined model and the fit is
			// repeated; a round is kept only if the score improves.
			size_t extraRefinementRounds = 0;
			// When enabled, the polish rounds collect the fitting set with a
			// shrinking threshold schedule (2.0x, 1.0x, 0.5x of the scoring
			// threshold) instead of reusing the scoring inliers -- a wider first
			// collection pulls in borderline points, the narrow last round
			// sharpens the fit (sigma-consensus style). Acceptance stays
			// score-gated with the unchanged pipeline scoring.
			bool sigmaSchedule = false;
			double sigmaScheduleStart = 3.0, // Widest collection multiplier (first round)
				sigmaScheduleEnd = 0.5; // Narrowest collection multiplier (last round)
			// When set, the refinement rounds pass the scoring's per-point
			// confidences (e.g. MAGSAC marginalized weights) to the solver, so
			// the bundle adjustment becomes a confidence-weighted IRLS fit.
			bool weightedFit = false;

		public:
			LeastSquaresOptimizer() : useInliers(false) {}
			~LeastSquaresOptimizer() {}

			void setUseInliers(const bool kUseInliers_)
			{
				useInliers = kUseInliers_;
			}

			void setExtraRefinementRounds(const size_t kRounds_)
			{
				extraRefinementRounds = kRounds_;
			}

			void setSigmaSchedule(const bool kSigmaSchedule_)
			{
				sigmaSchedule = kSigmaSchedule_;
			}

			void setSigmaScheduleRange(const double kStart_, const double kEnd_)
			{
				sigmaScheduleStart = kStart_;
				sigmaScheduleEnd = kEnd_;
			}

			void setWeightedFit(const bool kWeightedFit_)
			{
				weightedFit = kWeightedFit_;
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
				static const scoring::Score kInvalidScore = scoring::Score();

				// Pre-reserve estimated models to avoid reallocations
				std::vector<models::Model> estimatedModels;
				estimatedModels.reserve(10);
				estimatedModels.emplace_back(kModel_);

				estimatedModel_ = kModel_;
				estimatedScore_ = kScore_;
				estimatedInliers_ = kInliers_;

				scoring::Score currentScore;
				std::vector<double> initWeights;

				if (useInliers)
				{
					if (kInliers_.size() > 0)
					{
						const double *wPtr = nullptr;
						if (weightedFit)
						{
							kScoring_->getWeights(kData_, kModel_, kEstimator_, initWeights, &kInliers_);
							wPtr = initWeights.data();
						}
						// Estimate the model using the inliers
						kEstimator_->estimateModelNonminimal(
							kData_,  // The data points
							kInliers_.data(),  // Use .data() instead of &[0]
							kInliers_.size(),
							&estimatedModels,
							wPtr);
					}
					else
					{
						// Calculate the score of the estimated model
						currentScore = kScoring_->score(kData_, kModel_, kEstimator_, estimatedInliers_);
						estimatedScore_ = currentScore;

						const double *wPtr = nullptr;
						if (weightedFit)
						{
							kScoring_->getWeights(kData_, kModel_, kEstimator_, initWeights, &estimatedInliers_);
							wPtr = initWeights.data();
						}
						// Estimate the model using the inliers
						kEstimator_->estimateModelNonminimal(
							kData_,  // The data points
							estimatedInliers_.data(),  // Use .data() instead of &[0]
							estimatedInliers_.size(),
							&estimatedModels,
							wPtr);
					}
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

				// Temp inliers for selecting the best model - reserve outside loop
				std::vector<size_t> tmpInliers;
				tmpInliers.reserve(kData_.rows());

				// Calculate the scoring of the estimated model
				for (const auto &model : estimatedModels)
				{
					// Calculate the score of the estimated model
					tmpInliers.clear();
					currentScore = kScoring_->score(kData_, model, kEstimator_, tmpInliers);

					// Check if the current model is better than the previous one
					if (estimatedScore_ < currentScore)
					{
						// Update the estimated model
						estimatedModel_ = model;
						estimatedScore_ = currentScore;
						tmpInliers.swap(estimatedInliers_);
					}
				}

				// Optional score-gated polish: re-collect the inliers from the
				// refined model and refit; keep a round only if it improves the
				// score. Runs once per estimation, so the cost is negligible.
				std::vector<size_t> scheduleInliers;
				std::vector<double> scheduleResiduals;
				std::vector<double> fitWeights;
				for (size_t round = 0; round < extraRefinementRounds; ++round)
				{
					// The set the refit uses: the scoring inliers, or -- with the
					// sigma schedule -- the points within a shrinking multiple of
					// the scoring threshold w.r.t. the current model.
					const std::vector<size_t> *fitSet = &estimatedInliers_;
					if (sigmaSchedule)
					{
						// Geometric shrink of the collection threshold from
						// sigmaScheduleStart*sigma down to sigmaScheduleEnd*sigma
						// across the rounds (sigma-consensus style: a wide first
						// collection that narrows to sharpen the fit).
						const double kMultiplier = (extraRefinementRounds <= 1)
							? sigmaScheduleEnd
							: sigmaScheduleStart * std::pow(
								sigmaScheduleEnd / sigmaScheduleStart,
								static_cast<double>(round) / (extraRefinementRounds - 1));
						const double kThreshold = kScoring_->getThreshold() * kMultiplier;
						const double kSquaredThreshold = kThreshold * kThreshold;
						const size_t kPointNumber = kData_.rows();
						scheduleResiduals.resize(kPointNumber);
						kEstimator_->squaredResiduals(
							kData_, estimatedModel_, 0, kPointNumber, scheduleResiduals.data());
						scheduleInliers.clear();
						for (size_t i = 0; i < kPointNumber; ++i)
							if (scheduleResiduals[i] < kSquaredThreshold)
								scheduleInliers.emplace_back(i);
						if (scheduleInliers.size() > kEstimator_->nonMinimalSampleSize())
							fitSet = &scheduleInliers;
					}

					if (fitSet->size() <= kEstimator_->nonMinimalSampleSize())
						break;

					// Confidence-weighted refit: feed the scoring's per-point
					// weights (e.g. MAGSAC marginalized confidences) for the fit
					// set into the solver's LM bundle adjustment.
					const double *weightsPtr = nullptr;
					if (weightedFit)
					{
						kScoring_->getWeights(kData_, estimatedModel_, kEstimator_, fitWeights, fitSet);
						weightsPtr = fitWeights.data();
					}

					std::vector<models::Model> refinedModels;
					refinedModels.emplace_back(estimatedModel_);
					if (!kEstimator_->estimateModelNonminimal(
						kData_,
						fitSet->data(),
						fitSet->size(),
						&refinedModels,
						weightsPtr))
						break;

					bool improved = false;
					for (const auto &model : refinedModels)
					{
						tmpInliers.clear();
						currentScore = kScoring_->score(kData_, model, kEstimator_, tmpInliers);
						if (estimatedScore_ < currentScore)
						{
							estimatedModel_ = model;
							estimatedScore_ = currentScore;
							tmpInliers.swap(estimatedInliers_);
							improved = true;
						}
					}
					// With the sigma schedule, later (narrower) rounds may still
					// improve after a failed wide round, so keep going.
					if (!improved && !sigmaSchedule)
						break;
				}
			}

		};
	}
}  // namespace gcransac