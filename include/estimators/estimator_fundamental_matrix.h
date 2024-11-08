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
		class FundamentalMatrixEstimator : public Estimator
		{
		public:
			FundamentalMatrixEstimator() {}
			~FundamentalMatrixEstimator() {}
            
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
				static const size_t kSampleSize = sampleSize();

				// Estimate the model parameters by the minimal solver
				minimalSolver->estimateModel(kData_, // The data points
					kSample_, // The sample used for the estimation
					kSampleSize, // The size of a minimal sample
					*models_); // The estimated model parameters

				// Orientation constraint check 
				/*for (short modelIdx = models_->size() - 1; modelIdx >= 0; --modelIdx)
					if (!isOrientationValid(models_->at(modelIdx).getData(),
						kData_,
						kSample_,
						kSampleSize))
						models_->erase(models_->begin() + modelIdx);*/

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

			FORCE_INLINE double squaredResidual(const DataMatrix& point_,
				const models::Model& model_) const override
			{
				return squaredResidual(point_, model_.getData());
			}

			FORCE_INLINE double squaredResidual(
				const DataMatrix& kPoint_,
				const Eigen::MatrixXd& kDescriptor_) const
			{
				const double squaredResidual = squaredSampsonDistance(kPoint_, kDescriptor_);
				
				

				return squaredResidual;
			}

			// The symmetric epipolar distance between a point correspondence and an essential matrix
			FORCE_INLINE double squaredSymmetricEpipolarDistance(
				const DataMatrix& kPoint_,
				const Eigen::MatrixXd& kDescriptor_) const
			{
				// Use const references to avoid copying
				const double &x1 = kPoint_(0),
							&y1 = kPoint_(1),
							&x2 = kPoint_(2),
							&y2 = kPoint_(3);

				const double 
					&e11 = kDescriptor_(0, 0),
					&e12 = kDescriptor_(0, 1),
					&e13 = kDescriptor_(0, 2),
					&e21 = kDescriptor_(1, 0),
					&e22 = kDescriptor_(1, 1),
					&e23 = kDescriptor_(1, 2);

				const double rxc = e11 * x2 + e21 * y2 + kDescriptor_(2, 0);
				const double ryc = e12 * x2 + e22 * y2 + kDescriptor_(2, 1);
				const double rwc = e13 * x2 + e23 * y2 + kDescriptor_(2, 2);
				const double r = (x1 * rxc + y1 * ryc + rwc);
				const double rx = e11 * x1 + e12 * y1 + e13;
				const double ry = e21 * x1 + e22 * y1 + e23;
				const double a = rxc * rxc + ryc * ryc;
				const double b = rx * rx + ry * ry;

				return r * r * (a + b) / (a * b);
			}

			// The sampson distance between a point_ correspondence and an essential matrix
			FORCE_INLINE double squaredSampsonDistance(
				const DataMatrix& kPoint_,
				const Eigen::MatrixXd& kDescriptor_) const
			{
				const double E0_0 = kDescriptor_(0, 0), E0_1 = kDescriptor_(0, 1), E0_2 = kDescriptor_(0, 2);
				const double E1_0 = kDescriptor_(1, 0), E1_1 = kDescriptor_(1, 1), E1_2 = kDescriptor_(1, 2);
				const double E2_0 = kDescriptor_(2, 0), E2_1 = kDescriptor_(2, 1), E2_2 = kDescriptor_(2, 2);

				const double x1_0 = kPoint_(0), x1_1 = kPoint_(1);
				const double x2_0 = kPoint_(2), x2_1 = kPoint_(3);

				const double Ex1_0 = E0_0 * x1_0 + E0_1 * x1_1 + E0_2;
				const double Ex1_1 = E1_0 * x1_0 + E1_1 * x1_1 + E1_2;
				const double Ex1_2 = E2_0 * x1_0 + E2_1 * x1_1 + E2_2;

				const double Ex2_0 = E0_0 * x2_0 + E1_0 * x2_1 + E2_0;
				const double Ex2_1 = E0_1 * x2_0 + E1_1 * x2_1 + E2_1;
				// const double Ex2_2 = E0_2 * x2_0 + E1_2 * x2_1 + E2_2;

				const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;
				const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
				const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
				const double r2 = C * C / (Cx + Cy);

				return r2;
			}

			FORCE_INLINE double residual(const DataMatrix& point_,
				const models::Model& model_) const override
			{
				return residual(point_, model_.getData());
			}

			FORCE_INLINE double residual(const DataMatrix& point_,
				const DataMatrix& descriptor_) const
			{
				return sqrt(squaredResidual(point_, descriptor_));
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
				return true;
				// Validate the model by checking the number of inlier with symmetric epipolar distance
				// instead of Sampson distance. In general, Sampson distance is more accurate but less
				// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
				// every so-far-the-best model is checked if it has enough inlier with symmetric
				/*bool passed = false;
				size_t inlierNumber = 0; // Number of inlier if using symmetric epipolar distance
				const Eigen::Matrix3d &descriptor = model_.descriptor; // The decriptor of the current model
				constexpr size_t sample_size = sampleSize(); // Size of a minimal sample
				// Minimum number of inliers which should be inlier as well when using symmetric epipolar distance instead of Sampson distance
				const size_t minimum_inlier_number =
					MAX(sample_size, inliers_.size() * minimum_inlier_ratio_in_validity_check);
				// Squared inlier-outlier threshold
				const double squared_threshold = threshold_ * threshold_;

				// Iterate through the inliers_ determined by Sampson distance
				for (const auto &idx : inliers_)
					// Calculate the residual using symmetric epipolar distance and check if
					// it is smaller than the threshold_.
					if (squaredSymmetricEpipolarDistance(data_.row(idx), descriptor) < squared_threshold)
						// Increase the inlier number and terminate if enough inliers_ have been found.
						if (++inlierNumber >= minimum_inlier_number)
						{
							passed = true;
							break;
						}

				// If the fundamental matrix has not passed the symmetric epipolar tests,
				// terminate.
				if (!passed)
					return false;*/
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			FORCE_INLINE bool isValidSample(
				const DataMatrix& kData_, // All data points
				const size_t *kSample_) const override // The indices of the selected points
			{
				return true;
			}

			/************** Oriented epipolar constraints ******************/
			FORCE_INLINE void getEpipole(
				Eigen::Vector3d &epipole_, // The epipole 
				const Eigen::Matrix3d &kFundamentalMatrix_) const
			{
				constexpr double epsilon = 1.9984e-15;
				epipole_ = kFundamentalMatrix_.row(0).cross(kFundamentalMatrix_.row(2));

				for (auto i = 0; i < 3; i++)
					if ((epipole_(i) > epsilon) ||
						(epipole_(i) < -epsilon))
						return;
				epipole_ = kFundamentalMatrix_.row(1).cross(kFundamentalMatrix_.row(2));
			}

			FORCE_INLINE double getOrientationSignum(
				const Eigen::Matrix3d &kFundamentalMatrix_,
				const Eigen::Vector3d &kEpipole_,
				const DataMatrix &kPoint_) const
			{
				double signum1 = kFundamentalMatrix_(0, 0) * kPoint_(2) + kFundamentalMatrix_(1, 0) * kPoint_(3) + kFundamentalMatrix_(2, 0),
					signum2 = kEpipole_(1) - kEpipole_(2) * kPoint_(1);
				return signum1 * signum2;
			}

			FORCE_INLINE int isOrientationValid(
				const Eigen::Matrix3d &kFundamentalMatrix_, // The fundamental matrix
				const DataMatrix &kData_, // The data points
				const size_t *kSample_, // The sample used for the estimation
				size_t kSampleSize_) const // The size of the sample
			{
				Eigen::Vector3d epipole; // The epipole in the second image
				getEpipole(epipole, kFundamentalMatrix_);

				double signum1, signum2;

				// The sample is null pointer, the method is applied to normalized data_
				if (kSample_ == nullptr)
				{
					// Get the sign of orientation of the first point in the sample
					signum2 = getOrientationSignum(kFundamentalMatrix_, epipole, kData_.row(0));
					for (size_t i = 1; i < kSampleSize_; i++)
					{
						// Get the sign of orientation of the i-th point_ in the sample
						signum1 = getOrientationSignum(kFundamentalMatrix_, epipole, kData_.row(i));
						// The signs should be equal, otherwise, the fundamental matrix is invalid
						if (signum2 * signum1 < 0)
							return false;
					}
				}
				else
				{
					// Get the sign of orientation of the first point_ in the sample
					signum2 = getOrientationSignum(kFundamentalMatrix_, epipole, kData_.row(kSample_[0]));
					for (size_t i = 1; i < kSampleSize_; i++)
					{
						// Get the sign of orientation of the i-th point_ in the sample
						signum1 = getOrientationSignum(kFundamentalMatrix_, epipole, kData_.row(kSample_[i]));
						// The signs should be equal, otherwise, the fundamental matrix is invalid
						if (signum2 * signum1 < 0)
							return false;
					}
				}
				return true;
			}
		};
	}
}