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

#include "solver_fundamental_matrix_eight_point.h"
#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

#include "numerical_optimizer/bundle.h"
#include "numerical_optimizer/camera_pose.h"
#include "numerical_optimizer/essential.h"


namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class EssentialMatrixBundleAdjustmentSolver : public AbstractSolver
			{
			protected:
				poselib::BundleOptions options;
				size_t pointNumberForCheiralityCheck;
				const std::vector<double> *pointWeights;

			public:
				EssentialMatrixBundleAdjustmentSolver(poselib::BundleOptions kOptions_ = poselib::BundleOptions())
					: 	options(kOptions_),
						pointNumberForCheiralityCheck(1),
						pointWeights(nullptr)
				{
				}

				~EssentialMatrixBundleAdjustmentSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				bool returnMultipleModels() const override
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				size_t maximumSolutions() const override
				{
					return 1;
				}
				
				// The minimum number of points required for the estimation
				size_t sampleSize() const override
				{
					return 6;
				}

				poselib::BundleOptions &getMutableOptions()
				{
					return options;
				}

				void setWeights(const std::vector<double> *pointWeights_)
				{
					pointWeights = pointWeights_;
				}

				void setCheiralityPointNumber(const size_t kPointNumber_)
				{
					pointNumberForCheiralityCheck = kPointNumber_;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				FORCE_INLINE bool estimateModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_ = nullptr) const override; // The weight for each point
			};

			FORCE_INLINE bool EssentialMatrixBundleAdjustmentSolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{				
				// Check if we have enough points for the bundle adjustment
				if (kSampleNumber_ < sampleSize())
					return false;

				// The point correspondences. Thread-local scratch buffers: this solver
				// runs in the inner loops of the local optimizers, so reusing the
				// buffers avoids per-call heap allocations. The contents are fully
				// overwritten below.
				static thread_local std::vector<Eigen::Vector2d> x1;
				static thread_local std::vector<Eigen::Vector2d> x2;
				static thread_local std::vector<double> weights;
				x1.resize(kSampleNumber_);
				x2.resize(kSampleNumber_);
				weights.clear();
				// Per-constraint confidences fed into the relative-pose LM bundle
				// adjustment. The kWeights_ array passed by the local optimizers
				// (e.g. the MAGSAC marginalized weights aligned with the sample
				// order) takes precedence; the member pointWeights is the legacy
				// fallback. Previously kWeights_ was silently ignored, so the LM
				// always ran with uniform per-point weights.
				const bool useArgWeights = kWeights_ != nullptr;
				const bool useMemberWeights = !useArgWeights && pointWeights != nullptr;
				if (useArgWeights || useMemberWeights)
					weights.resize(kSampleNumber_);

				// Filling the point correspondences if the sample is not provided
				if (kSample_ == nullptr)
				{
					// Filling the point correspondences
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						x1[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 0), kData_(pointIdx, 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 2), kData_(pointIdx, 3));
						if (useArgWeights)
							weights[pointIdx] = kWeights_[pointIdx];
						else if (useMemberWeights)
							weights[pointIdx] = (*pointWeights)[pointIdx];
					}
				} else // Filling the point correspondences if the sample is provided
				{
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						const size_t &idx = kSample_[pointIdx];
						x1[pointIdx] = Eigen::Vector2d(kData_(idx, 0), kData_(idx, 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(idx, 2), kData_(idx, 3));
						if (useArgWeights)
							weights[pointIdx] = kWeights_[pointIdx];
						else if (useMemberWeights)
							weights[pointIdx] = (*pointWeights)[idx];
					}
				}
				
				// Estimating the essential matrix using the five-point and/or weighted eight-point
				// algorithms if no model is provided
				{
					// Initializing the five-point solver
					EssentialMatrixFivePointNisterSolver fivePointSolver;
					std::vector<models::Model> fivePtModels;
					if (fivePointSolver.estimateModel(kData_, kSample_, kSampleNumber_, fivePtModels, kWeights_))
					{
						for (auto &m : fivePtModels)
							models_.push_back(std::move(m));
					}

					if (kSampleNumber_ >= 8)
					{
						FundamentalMatrixEightPointSolver eightPointSolver;
						std::vector<models::Model> eightPtModels;
						if (eightPointSolver.estimateModel(kData_, kSample_, kSampleNumber_, eightPtModels, kWeights_) && !eightPtModels.empty())
						{
							Eigen::Matrix3d E8 = eightPtModels[0].getData().block<3, 3>(0, 0);
							Eigen::JacobiSVD<Eigen::Matrix3d> svd(E8, Eigen::ComputeFullU | Eigen::ComputeFullV);
							Eigen::Matrix3d U = svd.matrixU();
							Eigen::Matrix3d V = svd.matrixV();
							if (U.determinant() < 0) U.col(2) = -U.col(2);
							if (V.determinant() < 0) V.col(2) = -V.col(2);
							Eigen::Vector3d s(1.0, 1.0, 0.0);
							models::Model m8;
							m8.getMutableData() = U * s.asDiagonal() * V.transpose();
							models_.push_back(m8);
						}
					}

					// If the estimation failed, return false
					if (models_.size() == 0)
						return false;
				}
				
				// The options for the bundle adjustment
				poselib::BundleOptions tmpOptions = options;
				// If the sample is provided, we use a more robust loss function. This typically runs in the end of the robust estimation
				if (kSample_ != nullptr) 
				{
					tmpOptions.loss_scale = 0.5 * options.loss_scale;
					tmpOptions.max_iterations = 50;
					tmpOptions.loss_type = poselib::BundleOptions::LossType::CAUCHY;
				}

				poselib::BundleOptions tightOptions = tmpOptions;
				tightOptions.loss_scale = 0.25 * options.loss_scale;
				tightOptions.max_iterations = 25;
				
				// Build the (normalized) bearing vectors used to disambiguate the
				// four essential-matrix decompositions by cheirality. Use up to 64
				// sample points for reliable majority voting.
				const size_t kPointNumberForCheck = std::max<size_t>(1,
					std::min<size_t>(std::max<size_t>(pointNumberForCheiralityCheck, 64), kSampleNumber_));
				static thread_local std::vector<Eigen::Vector3d> x1CheiralityCheck, x2CheiralityCheck;
				x1CheiralityCheck.resize(kPointNumberForCheck);
				x2CheiralityCheck.resize(kPointNumberForCheck);
				for (size_t idx = 0; idx < kPointNumberForCheck; idx++)
				{
					const size_t& pointIdx = kSample_ == nullptr ? idx : kSample_[idx];
					x1CheiralityCheck[idx] = Eigen::Vector3d(kData_(pointIdx, 0), kData_(pointIdx, 1), 1);
					x2CheiralityCheck[idx] = Eigen::Vector3d(kData_(pointIdx, 2), kData_(pointIdx, 3), 1);
					x1CheiralityCheck[idx].normalize();
					x2CheiralityCheck[idx].normalize();
				}

				// Empty filter so motion_from_essential returns all four candidate poses.
				static const std::vector<Eigen::Vector3d> kEmptyCheck;

				struct CandidatePose
				{
					poselib::CameraPose pose;
					double weightedVotes;
					double initialCost;
				};
				std::vector<CandidatePose> candidates;
				candidates.reserve(models_.size() * 2);

				const double scaleSq = std::max(1e-12, tmpOptions.loss_scale * tmpOptions.loss_scale);

				// First pass: evaluate all candidate models and their 4 pose decompositions cheaply
				// using weighted cheirality + Sampson residual cost, so we only run expensive LM
				// bundle adjustment on the top candidate poses across all models.
				for (auto& model : models_)
				{
					const Eigen::Matrix3d E = model.getData().block<3, 3>(0, 0);
					if (!E.allFinite())
						continue;

					double sampsonCost = 0.0;
					for (size_t i = 0; i < kPointNumberForCheck; ++i)
					{
						const Eigen::Vector3d p1(x1[i](0), x1[i](1), 1.0);
						const Eigen::Vector3d p2(x2[i](0), x2[i](1), 1.0);
						const Eigen::Vector3d Ep1 = E * p1;
						const Eigen::Vector3d Etp2 = E.transpose() * p2;
						const double r = p2.dot(Ep1);
						const double denom = Ep1(0) * Ep1(0) + Ep1(1) * Ep1(1) + Etp2(0) * Etp2(0) + Etp2(1) * Etp2(1) + 1e-15;
						const double sqErr = (r * r) / denom;
						const double w = weights.empty() ? 1.0 : weights[i];
						sampsonCost += w * std::log(1.0 + sqErr / scaleSq);
					}

					poselib::CameraPoseVector poses;
					poselib::motion_from_essential(
						E,
						kEmptyCheck, kEmptyCheck,
						&poses);

					double maxModelVotes = 0.0;
					int bestPoseIdx = -1;
					for (size_t pIdx = 0; pIdx < poses.size(); ++pIdx)
					{
						double votes = 0.0;
						for (size_t i = 0; i < kPointNumberForCheck; ++i)
						{
							if (poselib::check_cheirality(poses[pIdx], x1CheiralityCheck[i], x2CheiralityCheck[i]))
							{
								votes += (weights.empty() ? 1.0 : weights[i]);
							}
						}
						if (votes > maxModelVotes)
						{
							maxModelVotes = votes;
							bestPoseIdx = static_cast<int>(pIdx);
						}
					}

					if (bestPoseIdx >= 0 && maxModelVotes > 0.0)
					{
						candidates.push_back({poses[bestPoseIdx], maxModelVotes, sampsonCost});
					}
				}

				if (candidates.empty())
					return false;

				std::sort(candidates.begin(), candidates.end(), [](const CandidatePose &a, const CandidatePose &b) {
					if (std::abs(a.weightedVotes - b.weightedVotes) > 1e-6)
						return a.weightedVotes > b.weightedVotes;
					return a.initialCost < b.initialCost;
				});

				const size_t kMaxRefinements = std::min<size_t>(2, candidates.size());
				double bestCost = std::numeric_limits<double>::max();
				poselib::CameraPose bestPose;
				bool haveBestPose = false;

				for (size_t cIdx = 0; cIdx < kMaxRefinements; ++cIdx)
				{
					poselib::CameraPose pose = candidates[cIdx].pose;
					poselib::BundleStats stats = poselib::refine_relpose(
						x1,
						x2,
						&pose,
						tmpOptions,
						weights);
					stats = poselib::refine_relpose(
						x1,
						x2,
						&pose,
						tightOptions,
						weights);

					if (!haveBestPose || stats.cost < bestCost)
					{
						bestCost = stats.cost;
						bestPose = pose;
						haveBestPose = true;
					}
				}
				
				// Composing the essential matrix from the pose
				if (haveBestPose)
				{
					Eigen::Matrix3d essentialMatrix;
					poselib::essential_from_motion(bestPose, &essentialMatrix);

					// Adding the essential matrix as the estimated models.
					models_.resize(1);
					models_[0].getMutableData() = essentialMatrix;
				}

				return models_.size() > 0;
			}
		}
	}
}