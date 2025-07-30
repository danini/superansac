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

				// The point correspondences
				std::vector<Eigen::Vector2d> x1(kSampleNumber_); 
				std::vector<Eigen::Vector2d> x2(kSampleNumber_); 
				std::vector<double> weights;
				if (pointWeights != nullptr)
					weights.resize(kSampleNumber_);

				// Filling the point correspondences if the sample is not provided
				if (kSample_ == nullptr)
				{
					// Filling the point correspondences
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						x1[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 0), kData_(pointIdx, 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 2), kData_(pointIdx, 3));
						if (pointWeights != nullptr)
							weights[pointIdx] = (*pointWeights)[pointIdx];
					}
				} else // Filling the point correspondences if the sample is provided
				{
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						const size_t &idx = kSample_[pointIdx];
						x1[pointIdx] = Eigen::Vector2d(kData_(idx, 0), kData_(idx, 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(idx, 2), kData_(idx, 3));
						if (pointWeights != nullptr)
							weights[pointIdx] = (*pointWeights)[idx];
					}
				}
				
				// Estimating the essential matrix using the five-point algorithm if no model is provided
				if (models_.size() == 0)
				{
					// Initializing the five-point solver
					EssentialMatrixFivePointNisterSolver fivePointSolver;
					// Estimating the essential matrix
					fivePointSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);

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
					tmpOptions.max_iterations = 100;
					tmpOptions.loss_type = poselib::BundleOptions::LossType::CAUCHY;
				}
				
				// Select the first point in the sample to be used for the cheirality check
				const size_t kPointNumberForCheck = std::min(pointNumberForCheiralityCheck, kSampleNumber_);
				std::vector<Eigen::Vector3d> x1CheiralityCheck(kPointNumberForCheck), x2CheiralityCheck(kPointNumberForCheck);
				for (size_t idx = 0; idx < kPointNumberForCheck; idx++)
				{
					const size_t& pointIdx = kSample_ == nullptr ? idx : kSample_[idx];
					x1CheiralityCheck[idx] = Eigen::Vector3d(kData_(pointIdx, 0), kData_(pointIdx, 1), 1);
					x2CheiralityCheck[idx] = Eigen::Vector3d(kData_(pointIdx, 2), kData_(pointIdx, 3), 1);
					x1CheiralityCheck[idx].normalize();
					x2CheiralityCheck[idx].normalize();
				}

				// The pose with the lowest cost
				double bestCost = std::numeric_limits<double>::max();
				poselib::CameraPose bestPose;

				// Iterating through the potential models.
				for (auto& model : models_)
				{
					// Decompose the essential matrix to camera poses
					poselib::CameraPoseVector poses;
					
					poselib::motion_from_essential(
						model.getData().block<3, 3>(0, 0), // The essential matrix
						x1CheiralityCheck, x2CheiralityCheck, // The point correspondence used for the cheirality check
						&poses); // The decomposed poses

					// Iterating through the potential poses and optimizing each
					for (auto& pose : poses)
					{
						// Perform the bundle adjustment
						poselib::BundleStats stats;
						poselib::refine_relpose(
							x1, 
							x2, 
							&pose,
							tmpOptions,
							weights);

						if (stats.cost < bestCost)
						{
							bestCost = stats.cost;
							bestPose = pose;
						}
					}
				}
				
				// Composing the essential matrix from the pose
				if (bestCost < std::numeric_limits<double>::max())
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