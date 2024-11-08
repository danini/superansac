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

#include "solver_homography_four_point.h"

namespace superansac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		class AbsolutePoseEstimator : public Estimator
		{
		public:
			AbsolutePoseEstimator() {}
			~AbsolutePoseEstimator() {}
            
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
				return minimalSolver->estimateModel(kData_, // The data points
					kSample_, // The sample used for the estimation
					sampleSize(), // The size of a minimal sample
					*models_); // The estimated model parameters
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

				// The PnP algorithm
				if (!nonMinimalSolver->estimateModel(kData_,
					kSample_,
					kSampleNumber_,
					*models_,
					kWeights_))
					return false;
				return models_->size();
			}

			FORCE_INLINE double squaredResidual(const DataMatrix& point_,
				const models::Model& model_) const override
			{
				return squaredResidual(point_, model_.getData());
			}

			FORCE_INLINE double squaredResidual(const DataMatrix& point_,
				const DataMatrix& descriptor_) const
			{
				const double 
					&u = point_(0),
					&v = point_(1),
					&x = point_(2),
					&y = point_(3),
					&z = point_(4);

				const double 
					&r11 = descriptor_(0, 0),
					&r12 = descriptor_(0, 1),
					&r13 = descriptor_(0, 2),
					&r21 = descriptor_(1, 0),
					&r22 = descriptor_(1, 1),
					&r23 = descriptor_(1, 2),
					&r31 = descriptor_(2, 0),
					&r32 = descriptor_(2, 1),
					&r33 = descriptor_(2, 2),
					&tx = descriptor_(0, 3),
					&ty = descriptor_(1, 3),
					&tz = descriptor_(2, 3);
				
				const double px = r11 * x + r12 * y + r13 * z + tx,
					py = r21 * x + r22 * y + r23 * z + ty,
					pz = r31 * x + r32 * y + r33 * z + tz;

				const double pu = px / pz,
					pv = py / pz;	

				const double du = pu - u,
					dv = pv - v;
				
				return du * du + dv * dv;
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

			// Enable a quick check to see if the model is valid. This can be a geometric
			// check or some other verification of the model structure.
			FORCE_INLINE bool isValidModel(models::Model& model_,
				const DataMatrix& kData_,
				const size_t* kMinimalSample_,
				const double kThreshold_,
				bool& modelUpdated_) const override
			{ 
				// Calculate the determinant of the homography
				/*const double kDeterminant =
					model_.getData().determinant();

				// Check if the homography has a small determinant.
				constexpr double kMinimumDeterminant = 1e-2;
				if (abs(kDeterminant) < kMinimumDeterminant)
					return false;*/
				return true;
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			FORCE_INLINE bool isValidSample(
				const DataMatrix& kData_, // All data points
				const size_t *kSample_) const override // The indices of the selected points
			{
				/*if (sampleSize() < 4)
					return true;

				// Check oriented constraints
				Eigen::Vector3d p, q;

				// Use references to avoid repeated dereferencing
				const double *a = &kData_(kSample_[0], 0);
				const double *b = &kData_(kSample_[1], 0);
				const double *c = &kData_(kSample_[2], 0);
				const double *d = &kData_(kSample_[3], 0);

				crossProduct(p, a, b, 1);
				crossProduct(q, a + 2, b + 2, 1);

				// Use cached values and simplify conditions
				const double p_c = p[0] * c[0] + p[1] * c[1] + p[2];
				const double q_c = q[0] * c[2] + q[1] * c[3] + q[2];
				const double p_d = p[0] * d[0] + p[1] * d[1] + p[2];
				const double q_d = q[0] * d[2] + q[1] * d[3] + q[2];

				if (p_c * q_c < 0 || p_d * q_d < 0)
					return false;

				crossProduct(p, c, d, 1);
				crossProduct(q, c + 2, d + 2, 1);

				const double p_a = p[0] * a[0] + p[1] * a[1] + p[2];
				const double q_a = q[0] * a[2] + q[1] * a[3] + q[2];
				const double p_b = p[0] * b[0] + p[1] * b[1] + p[2];
				const double q_b = q[0] * b[2] + q[1] * b[3] + q[2];

				if (p_a * q_a < 0 || p_b * q_b < 0)
					return false;*/

				return true;
			}
		};
	}
}