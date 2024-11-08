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
#include "numerical_optimizer/essential.h"


namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class FundamentalMatrixBundleAdjustmentSolver : public AbstractSolver
			{
			protected:
				poselib::BundleOptions options;

			public:
				FundamentalMatrixBundleAdjustmentSolver(poselib::BundleOptions kOptions_ = poselib::BundleOptions())
					: options(kOptions_)
				{
				}

				~FundamentalMatrixBundleAdjustmentSolver()
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
					return 8;
				}

				poselib::BundleOptions &getMutableOptions()
				{
					return options;
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

			FORCE_INLINE bool FundamentalMatrixBundleAdjustmentSolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{				
				/*std::cout << "BA" << std::endl;

				std::cout << kSampleNumber_ << std::endl;
				std::cout << kData_.rows() << std::endl;
				std::cout << models_[0].getData() << std::endl;*/

				// Check if we have enough points for the bundle adjustment
				if (kSampleNumber_ < sampleSize())
					return false;

				std::vector<Eigen::Vector2d> x1(kSampleNumber_); 
				std::vector<Eigen::Vector2d> x2(kSampleNumber_); 

				if (kSample_ == nullptr)
				{
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						x1[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 0), kData_(pointIdx, 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 2), kData_(pointIdx, 3));
					}
				} else
				{
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						//std::cout << kSample_[pointIdx] << " ";
						x1[pointIdx] = Eigen::Vector2d(kData_(kSample_[pointIdx], 0), kData_(kSample_[pointIdx], 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(kSample_[pointIdx], 2), kData_(kSample_[pointIdx], 3));
					}
					//std::cout << std::endl;
				}
				
				if (models_.size() == 0)
				{
					FundamentalMatrixEightPointSolver eightPointSolver;
					eightPointSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);

					if (models_.size() == 0)
						return false;
				}

				/*std::cout << options.max_iterations << std::endl;
				std::cout << options.loss_type << std::endl;
				std::cout << options.loss_scale << std::endl;
				std::cout << options.gradient_tol << std::endl;
				std::cout << options.step_tol << std::endl;
				std::cout << options.initial_lambda << std::endl;
				std::cout << options.min_lambda << std::endl;
				std::cout << options.max_lambda << std::endl;
				std::cout << options.verbose << std::endl;
				std::cout << options.loss_scale << std::endl;*/

				poselib::BundleOptions tmpOptions = options;
				if (kSample_ != nullptr)
				{
					tmpOptions.loss_scale = 0.5 * options.loss_scale;
					tmpOptions.max_iterations = 100;
					tmpOptions.loss_type = poselib::BundleOptions::LossType::CAUCHY;
				}

				for (auto& model : models_)
				{
					// Get the fundamental matrix
					Eigen::Matrix3d fundamentalMatrix = model.getMutableData().block<3, 3>(0, 0).eval();

					// Perform the bundle adjustment
					poselib::BundleStats stats;
					poselib::refine_fundamental(
						x1, 
						x2, 
						&fundamentalMatrix,
						tmpOptions);

					/*if (kSample_ != nullptr)
					{
						std::cout << tmpOptions.max_iterations << std::endl;
						std::cout << tmpOptions.loss_type << std::endl;
						std::cout << tmpOptions.loss_scale << std::endl;
						std::cout << tmpOptions.gradient_tol << std::endl;
						std::cout << tmpOptions.step_tol << std::endl;
						std::cout << tmpOptions.initial_lambda << std::endl;
						std::cout << tmpOptions.min_lambda << std::endl;
						std::cout << tmpOptions.max_lambda << std::endl;
						std::cout << tmpOptions.verbose << std::endl;
						std::cout << fundamentalMatrix << std::endl;
						std::cout << "----------------" << std::endl;
					}*/

					// Update the model
					model.getMutableData().block<3, 3>(0, 0) = fundamentalMatrix;
				}

				return true;
			}
		}
	}
}