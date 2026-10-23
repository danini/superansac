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

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Eigen>

#include "abstract_estimator.h"
#include "../models/model.h"
#include "../utils/types.h"

#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_eight_point.h"

namespace superansac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		class RigidTransformationEstimator : public Estimator
		{
		protected:
			double inlierThreshold = 0.05;

		public:
			RigidTransformationEstimator() {}
			~RigidTransformationEstimator() {}

			void setInlierThreshold(const double kThreshold_)
			{
				inlierThreshold = kThreshold_;
			}
            
			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			bool isWeightingApplicable() const override
            {
                return true;
            }
			
            double multError() const
			{
				return 1.0;
			}

            double logAlpha0(size_t w, size_t h, double scalingFactor = 0.5) const
			{
				return log(1.0 / (w * h * scalingFactor));
			}

			// Degrees of freedom for the MAGSAC++ scoring
			size_t getDegreesOfFreedom() const
			{
				return 2;
			}
			
			// Estimating the model from a minimal sample
			FORCE_INLINE bool estimateModel(
				const DataMatrix& kData_, // The data points
				const size_t *kSample_, // The sample usd for the estimation
				std::vector<models::Model>* models_) const override // The estimated model parameters
			{
				const size_t kSampleSize = sampleSize();

				// Estimate the model parameters by the minimal solver
				minimalSolver->estimateModel(kData_, // The data points
					kSample_, // The sample used for the estimation
					kSampleSize, // The size of a minimal sample
					*models_); // The estimated model parameters

				// The estimation was successfull if at least one model is kept
				return models_->size() > 0;
			}

			// Estimating the model from a non-minimal sample
			FORCE_INLINE bool estimateModelNonminimal(
                const DataMatrix& kData_, // The data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t &kSampleNumber_, // The size of a minimal sample
				std::vector<models::Model>* models_,
				const double *kWeights_ = nullptr) const override // The estimated model parameters
			{
				// Return of there are not enough points for the estimation
				if (kSampleNumber_ < nonMinimalSampleSize())
					return false;

				if (!nonMinimalSolver->estimateModel(kData_,
					kSample_,
					kSampleNumber_,
					*models_,
					kWeights_))
					return false;
				return true;
			}

			FORCE_INLINE double squaredResidual(const double* point_,
				const models::Model& model_) const override
			{
				return squaredResidual(point_, model_.getData());
			}

			FORCE_INLINE double squaredResidual(
				const double* kPoint_,
				const ModelMatrix& kDescriptor_) const
			{
				const double &x1 = kPoint_[0];
				const double &y1 = kPoint_[1];
				const double &z1 = kPoint_[2];
				const double &x2 = kPoint_[3];
				const double &y2 = kPoint_[4];
				const double &z2 = kPoint_[5];

				const double t1 = kDescriptor_(0, 0) * x1 + kDescriptor_(1, 0) * y1 + kDescriptor_(2, 0) * z1 + kDescriptor_(3, 0);
				const double t2 = kDescriptor_(0, 1) * x1 + kDescriptor_(1, 1) * y1 + kDescriptor_(2, 1) * z1 + kDescriptor_(3, 1);
				const double t3 = kDescriptor_(0, 2) * x1 + kDescriptor_(1, 2) * y1 + kDescriptor_(2, 2) * z1 + kDescriptor_(3, 2);
				
				const double dx = x2 - t1;
				const double dy = y2 - t2;
				const double dz = z2 - t3;

				return dx * dx + dy * dy + dz * dz;
			}

			FORCE_INLINE double residual(const double* point_,
				const models::Model& model_) const override
			{
				return residual(point_, model_.getData());
			}

			FORCE_INLINE double residual(const double* point_,
				const ModelMatrix& descriptor_) const
			{
				return sqrt(squaredResidual(point_, descriptor_));
			}

			// Batched squared residuals: the descriptor is loaded once and the
			// per-point arithmetic (identical to squaredResidual) runs in a tight,
			// non-virtual loop over the contiguous row-major rows.
			void squaredResiduals(
				const DataMatrix& kData_,
				const models::Model& kModel_,
				const size_t kStartRow_,
				const size_t kCount_,
				double* kOut_) const override
			{
				const ModelMatrix& kDescriptor = kModel_.getData();
				const size_t kCols = kData_.cols();
				const double* row = kData_.data() + kStartRow_ * kCols;
				for (size_t i = 0; i < kCount_; ++i, row += kCols)
					kOut_[i] = squaredResidual(row, kDescriptor);
			}

			// Validate the model by checking the number of inlier with symmetric epipolar distance
			// instead of Sampson distance. In general, Sampson distance is more accurate but less
			// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
			// every so-far-the-best model is checked if it has enough inlier with symmetric
			// epipolar distance as well. 
			FORCE_INLINE bool isValidModel(models::Model& model_,
				const DataMatrix& kData_,
				const size_t* kMinimalSample_,
				const double kThreshold_,
				bool& modelUpdated_) const override
			{
				constexpr double kProperRotationThreshold = 1e-2;

				// Extract rotation matrix from the model
				const auto &kTransformation =  model_.getData(); 
				const auto &kRotationMatrix = kTransformation.block<3, 3>(0, 0);

				// Calculate the determinant of the rotation matrix
				double det = kRotationMatrix(0, 0) * (kRotationMatrix(1, 1) * kRotationMatrix(2, 2) - kRotationMatrix(1, 2) * kRotationMatrix(2, 1))
						- kRotationMatrix(0, 1) * (kRotationMatrix(1, 0) * kRotationMatrix(2, 2) - kRotationMatrix(1, 2) * kRotationMatrix(2, 0))
						+ kRotationMatrix(0, 2) * (kRotationMatrix(1, 0) * kRotationMatrix(2, 1) - kRotationMatrix(1, 1) * kRotationMatrix(2, 0));

				// Check if the determinant is close to +1 (indicating a proper rotation without mirroring)
				if (std::abs(det - 1.0) > kProperRotationThreshold) 
					return false; // Invalid model due to mirroring or improper rotation

				if (kMinimalSample_ != nullptr)
				{
					const double maxSqRes = kThreshold_ * kThreshold_;
					for (size_t i = 0; i < 3; ++i)
					{
						if (squaredResidual(kData_.row(kMinimalSample_[i]).data(), kTransformation) > maxSqRes)
							return false;
					}
				}
				return true;
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			FORCE_INLINE bool isValidSample(
				const DataMatrix& kData_, // All data points
				const size_t *kSample_) const override // The indices of the selected points
			{
				// Check whether the selected points are collinear
				constexpr double kCollinearityThreshold = 1e-6;
				
				// Extract the points based on indices in kSample_
				const auto& p1 = kData_.row(kSample_[0]);
				const auto& p2 = kData_.row(kSample_[1]);
				const auto& p3 = kData_.row(kSample_[2]);

				// Calculate vectors in the first domain (x1, y1, z1)
				double v1x1 = p2[0] - p1[0], 
					v1y1 = p2[1] - p1[1], 
					v1z1 = p2[2] - p1[2];
				double v2x1 = p3[0] - p1[0], 
					v2y1 = p3[1] - p1[1], 
					v2z1 = p3[2] - p1[2];

				// Calculate vectors in the second domain (x2, y2, z2)
				double v1x2 = p2[3] - p1[3], v1y2 = p2[4] - p1[4], v1z2 = p2[5] - p1[5];
				double v2x2 = p3[3] - p1[3], v2y2 = p3[4] - p1[4], v2z2 = p3[5] - p1[5];

				// Isometric triangle edge-length congruence check: under any rigid transformation
				// where all 3 sample points have residual <= inlierThreshold, each pairwise distance
				// is preserved within 2 * inlierThreshold by the triangle inequality.
				const double maxEdgeDiff = 2.0 * inlierThreshold;
				const double d1_12 = std::sqrt(v1x1 * v1x1 + v1y1 * v1y1 + v1z1 * v1z1);
				const double d2_12 = std::sqrt(v1x2 * v1x2 + v1y2 * v1y2 + v1z2 * v1z2);
				if (std::abs(d1_12 - d2_12) > maxEdgeDiff)
					return false;

				const double d1_13 = std::sqrt(v2x1 * v2x1 + v2y1 * v2y1 + v2z1 * v2z1);
				const double d2_13 = std::sqrt(v2x2 * v2x2 + v2y2 * v2y2 + v2z2 * v2z2);
				if (std::abs(d1_13 - d2_13) > maxEdgeDiff)
					return false;

				const double v3x1 = p3[0] - p2[0], v3y1 = p3[1] - p2[1], v3z1 = p3[2] - p2[2];
				const double v3x2 = p3[3] - p2[3], v3y2 = p3[4] - p2[4], v3z2 = p3[5] - p2[5];
				const double d1_23 = std::sqrt(v3x1 * v3x1 + v3y1 * v3y1 + v3z1 * v3z1);
				const double d2_23 = std::sqrt(v3x2 * v3x2 + v3y2 * v3y2 + v3z2 * v3z2);
				if (std::abs(d1_23 - d2_23) > maxEdgeDiff)
					return false;
				
				// Cross product in the first domain
				double cross1_x = v1y1 * v2z1 - v1z1 * v2y1;
				double cross1_y = v1z1 * v2x1 - v1x1 * v2z1;
				double cross1_z = v1x1 * v2y1 - v1y1 * v2x1;
				
				// Check if cross product is near zero for collinearity in the first domain
				if (std::abs(cross1_x) < kCollinearityThreshold && std::abs(cross1_y) < kCollinearityThreshold && std::abs(cross1_z) < kCollinearityThreshold) {
					return false; // Collinear in the first domain
				}
				
				// Cross product in the second domain
				double cross2_x = v1y2 * v2z2 - v1z2 * v2y2;
				double cross2_y = v1z2 * v2x2 - v1x2 * v2z2;
				double cross2_z = v1x2 * v2y2 - v1y2 * v2x2;
				
				// Check if cross product is near zero for collinearity in the second domain
				if (std::abs(cross2_x) < kCollinearityThreshold && std::abs(cross2_y) < kCollinearityThreshold && std::abs(cross2_z) < kCollinearityThreshold) {
					return false; // Collinear in the second domain
				}

				return true; // Points are not collinear in either domain
			}
		};
	}
}