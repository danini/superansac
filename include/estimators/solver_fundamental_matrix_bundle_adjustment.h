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
				const std::vector<double> *pointWeights;

			public:
				FundamentalMatrixBundleAdjustmentSolver(poselib::BundleOptions kOptions_ = poselib::BundleOptions())
					: options(kOptions_), 
					pointWeights(nullptr)
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

			FORCE_INLINE bool FundamentalMatrixBundleAdjustmentSolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{				
				// Check if we have enough points for the bundle adjustment
				if (kSampleNumber_ < sampleSize())
					return false;

				std::vector<Eigen::Vector2d> x1(kSampleNumber_); 
				std::vector<Eigen::Vector2d> x2(kSampleNumber_); 
				std::vector<double> weights;
				bool useWeights = (pointWeights != nullptr && pointWeights->size() == kSampleNumber_);
				if (useWeights)
					weights.resize(kSampleNumber_);

				if (kSample_ == nullptr)
				{
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						x1[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 0), kData_(pointIdx, 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 2), kData_(pointIdx, 3));
						if (useWeights)
							weights[pointIdx] = pointWeights->at(pointIdx);
					}
				} else
				{
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						const size_t &idx = kSample_[pointIdx];
						x1[pointIdx] = Eigen::Vector2d(kData_(idx, 0), kData_(idx, 1));
						x2[pointIdx] = Eigen::Vector2d(kData_(idx, 2), kData_(idx, 3));
						if (useWeights)
							weights[pointIdx] = pointWeights->at(idx);
					}
				}
				
				if (models_.size() == 0)
				{
					FundamentalMatrixEightPointSolver eightPointSolver;
					eightPointSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);

					if (models_.size() == 0)
						return false;
				}

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
						tmpOptions,
						weights);
					
					if (stats.cost > stats.initial_cost)
					{
						std::cout <<std::endl<<stats.iterations;
						std::cout <<std::endl<<stats.initial_cost;
						std::cout <<std::endl<<stats.cost;
						std::cout <<std::endl<<stats.lambda;
						std::cout <<std::endl<<stats.invalid_steps;
						std::cout <<std::endl<<stats.step_norm;
						std::cout <<std::endl<<stats.grad_norm;

						while (1);
					}

					// Update the model
					model.getMutableData().block<3, 3>(0, 0) = fundamentalMatrix;
				}

				return true;
			}
		}
	}
}