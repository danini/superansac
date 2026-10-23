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

#include <algorithm>
#include <vector>
#include <unsupported/Eigen/Polynomials>

#include "abstract_solver.h"

namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class RigidTransformProscrutesSolver : public AbstractSolver
			{
			public:
				RigidTransformProscrutesSolver()
				{
				}

				~RigidTransformProscrutesSolver()
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
					return 3;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				FORCE_INLINE bool estimateModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_ = nullptr) const override; // The weight for each point
                    

			protected:
				FORCE_INLINE bool estimateMinimalModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_) const; // The weight for each point
			};

			FORCE_INLINE bool RigidTransformProscrutesSolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{				
				if (kSampleNumber_ < 3)
					return false;

				auto solveWeightedProcrustes = [&](const double *weights, Eigen::Matrix3d &R_out, Eigen::Vector3d &t_out) -> bool
				{
					double centroid0x = 0.0, centroid0y = 0.0, centroid0z = 0.0;
					double centroid1x = 0.0, centroid1y = 0.0, centroid1z = 0.0;
					double weightSum = 0.0;

					for (size_t i = 0; i < kSampleNumber_; ++i)
					{
						const size_t idx = (kSample_ == nullptr) ? i : kSample_[i];
						const double w = (weights == nullptr) ? 1.0 : weights[i];
						if (w <= 0.0)
							continue;

						centroid0x += w * kData_(idx, 0);
						centroid0y += w * kData_(idx, 1);
						centroid0z += w * kData_(idx, 2);
						centroid1x += w * kData_(idx, 3);
						centroid1y += w * kData_(idx, 4);
						centroid1z += w * kData_(idx, 5);
						weightSum += w;
					}

					if (weightSum <= 1e-12)
						return false;

					const double invWeightSum = 1.0 / weightSum;
					centroid0x *= invWeightSum;
					centroid0y *= invWeightSum;
					centroid0z *= invWeightSum;
					centroid1x *= invWeightSum;
					centroid1y *= invWeightSum;
					centroid1z *= invWeightSum;

					Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
					for (size_t i = 0; i < kSampleNumber_; ++i)
					{
						const size_t idx = (kSample_ == nullptr) ? i : kSample_[i];
						const double w = (weights == nullptr) ? 1.0 : weights[i];
						if (w <= 0.0)
							continue;

						const double dx0 = kData_(idx, 0) - centroid0x;
						const double dy0 = kData_(idx, 1) - centroid0y;
						const double dz0 = kData_(idx, 2) - centroid0z;
						const double dx1 = kData_(idx, 3) - centroid1x;
						const double dy1 = kData_(idx, 4) - centroid1y;
						const double dz1 = kData_(idx, 5) - centroid1z;

						covariance(0, 0) += w * dx0 * dx1;
						covariance(0, 1) += w * dx0 * dy1;
						covariance(0, 2) += w * dx0 * dz1;
						covariance(1, 0) += w * dy0 * dx1;
						covariance(1, 1) += w * dy0 * dy1;
						covariance(1, 2) += w * dy0 * dz1;
						covariance(2, 0) += w * dz0 * dx1;
						covariance(2, 1) += w * dz0 * dy1;
						covariance(2, 2) += w * dz0 * dz1;
					}

					covariance *= invWeightSum;
					if (covariance.hasNaN())
						return false;

					Eigen::JacobiSVD<Eigen::Matrix3d> svd(
						covariance,
						Eigen::ComputeFullV | Eigen::ComputeFullU);

					Eigen::Matrix3d V = svd.matrixV();
					const Eigen::Matrix3d &U = svd.matrixU();
					R_out = V * U.transpose();

					if (R_out.determinant() < 0.0)
					{
						V.col(2) = -V.col(2);
						R_out = V * U.transpose();
					}

					t_out(0) = -R_out(0, 0) * centroid0x - R_out(0, 1) * centroid0y - R_out(0, 2) * centroid0z + centroid1x;
					t_out(1) = -R_out(1, 0) * centroid0x - R_out(1, 1) * centroid0y - R_out(1, 2) * centroid0z + centroid1y;
					t_out(2) = -R_out(2, 0) * centroid0x - R_out(2, 1) * centroid0y - R_out(2, 2) * centroid0z + centroid1z;
					return true;
				};

				Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
				Eigen::Vector3d t = Eigen::Vector3d::Zero();
				if (!solveWeightedProcrustes(kWeights_, R, t))
					return false;
				
				models::Model model;
				auto &modelData = model.getMutableData();
				modelData.resize(4, 4);
				modelData << R(0, 0), R(1, 0), R(2, 0), 0,
					R(0, 1), R(1, 1), R(2, 1), 0,
					R(0, 2), R(1, 2), R(2, 2), 0,
					t(0), t(1), t(2), 1;
				models_.push_back(model);
				return true;
			}
		}
	}
}