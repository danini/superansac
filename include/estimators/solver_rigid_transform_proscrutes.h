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
				Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
				Eigen::Vector3d t = Eigen::Vector3d::Zero();
				double weight = 1.0;

				// Calculate the center of gravity for both point clouds
				double centroid0x = 0.0,
					centroid0y = 0.0,
					centroid0z = 0.0,
					centroid1x = 0.0,
					centroid1y = 0.0,
					centroid1z = 0.0;

				for (size_t i = 0; i < kSampleNumber_; i++)
				{
					const size_t idx =
						kSample_ == nullptr ? i : kSample_[i];

					const double
						&x0 = kData_(idx, 0),
						&y0 = kData_(idx, 1),
						&z0 = kData_(idx, 2),
						&x1 = kData_(idx, 3),
						&y1 = kData_(idx, 4),
						&z1 = kData_(idx, 5);

					centroid0x += x0;
					centroid0y += y0;
					centroid0z += z0;
					centroid1x += x1;
					centroid1y += y1;
					centroid1z += z1;
				}
				
				centroid0x /= kSampleNumber_;
				centroid0y /= kSampleNumber_;
				centroid0z /= kSampleNumber_;
				centroid1x /= kSampleNumber_;
				centroid1y /= kSampleNumber_;
				centroid1z /= kSampleNumber_;
				
				Eigen::MatrixXd coefficients0(3, kSampleNumber_);
				Eigen::MatrixXd coefficients1(3, kSampleNumber_);

				double avgDistance0 = 0.0,
					avgDistance1 = 0.0;

				for (size_t i = 0; i < kSampleNumber_; i++)
				{
					const size_t idx =
						kSample_ == nullptr ? i : kSample_[i];
					if (kWeights_ != nullptr)
						weight = kWeights_[idx];

					const double
						&x0 = kData_(idx, 0),
						&y0 = kData_(idx, 1),
						&z0 = kData_(idx, 2),
						&x1 = kData_(idx, 3),
						&y1 = kData_(idx, 4),
						&z1 = kData_(idx, 5);

					coefficients0(0, i) = x0 - centroid0x;
					coefficients0(1, i) = y0 - centroid0y;
					coefficients0(2, i) = z0 - centroid0z;
					coefficients1(0, i) = x1 - centroid1x;
					coefficients1(1, i) = y1 - centroid1y;
					coefficients1(2, i) = z1 - centroid1z;

					avgDistance0 +=
						std::sqrt(coefficients0(0, i) * coefficients0(0, i) +
							coefficients0(1, i) * coefficients0(1, i) +
							coefficients0(2, i) * coefficients0(2, i));

					avgDistance1 +=
						std::sqrt(coefficients1(0, i) * coefficients1(0, i) +
							coefficients1(1, i) * coefficients1(1, i) +
							coefficients1(2, i) * coefficients1(2, i));

					coefficients0(0, i) *= weight;
					coefficients0(1, i) *= weight;
					coefficients0(2, i) *= weight;
					coefficients1(0, i) *= weight;
					coefficients1(1, i) *= weight;
					coefficients1(2, i) *= weight;
					 
				}

				avgDistance0 /= kSampleNumber_;
				avgDistance1 /= kSampleNumber_;

				static const double sqrt_3 = std::sqrt(3.0);
				const double ratio0 = sqrt_3 / avgDistance0,
					ratio1 = sqrt_3 / avgDistance1;

				coefficients0 *= ratio0;
				coefficients1 *= ratio1;
				
				Eigen::MatrixXd covariance =
					coefficients0 * coefficients1.transpose();
				
				if (covariance.hasNaN())
					return false;

				// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
				// the solution is linear subspace of dimensionality 2.
				// => use the last two singular std::vectors as a basis of the space
				// (according to SVD properties)
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					// Theoretically, it would be faster to apply SVD only to matrix coefficients, but
					// multiplication is faster than SVD in the Eigen library. Therefore, it is faster
					// to apply SVD to a smaller matrix.
					covariance,
					Eigen::ComputeFullV | Eigen::ComputeFullU);

				R = svd.matrixV() * svd.matrixU().transpose();

				if (R.determinant() < 0)
				{
					Eigen::MatrixXd V = svd.matrixV();
					V.col(2) = -V.col(2);
					R = V * svd.matrixU().transpose();
				}

				t(0) = -R(0, 0) * centroid0x - R(0, 1) * centroid0y - R(0, 2) * centroid0z + centroid1x;
				t(1) = -R(1, 0) * centroid0x - R(1, 1) * centroid0y - R(1, 2) * centroid0z + centroid1y;
				t(2) = -R(2, 0) * centroid0x - R(2, 1) * centroid0y - R(2, 2) * centroid0z + centroid1z;
				
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