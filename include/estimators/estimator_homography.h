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
#include "numerical_optimizer/bundle.h"

namespace superansac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		class HomographyEstimator : public Estimator
		{
		protected:
			// Optional nonlinear (LM) refinement of the non-minimal homography
			// estimate, minimizing the robust reprojection error -- the homography
			// analog of the bundle adjustment F/E already run. The DLT solver only
			// gives the algebraic (linear) optimum.
			bool useBundleRefinement = false;
			double bundleLossScale = 1.0; // Robust loss scale (= inlier threshold, in pixels)
			size_t bundleMaxIterations = 100; // LM iterations for the refinement

		public:
			HomographyEstimator() {}
			~HomographyEstimator() {}

			void setBundleRefinement(const bool kEnabled_, const double kLossScale_, const size_t kMaxIterations_ = 100)
			{
				useBundleRefinement = kEnabled_;
				bundleLossScale = kLossScale_;
				bundleMaxIterations = kMaxIterations_;
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
				static thread_local DataMatrix normalizedPoints;
				normalizedPoints.resize(sampleSize(), kData_.cols());
				Eigen::Matrix3d normalizingTransformSource,
					normalizingTransformDestination;

				if (!normalizePoints(kData_,
					kSample_,
					sampleSize(),
					normalizedPoints,
					normalizingTransformSource,
					normalizingTransformDestination))
					return false;

				if (!minimalSolver->estimateModel(normalizedPoints,
					nullptr,
					sampleSize(),
					*models_))
					return false;

				const Eigen::Matrix3d kNormalizingTransformDestinationInverse =
					normalizingTransformDestination.inverse();
				for (auto &model : *models_)
					model.setData(kNormalizingTransformDestinationInverse * model.getData() * normalizingTransformSource);
				return true;
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

				// Thread-local scratch (used in the local-optimization inner loops);
				// fully overwritten by normalizePoints below.
				static thread_local DataMatrix normalizedPoints;
				normalizedPoints.resize(kSampleNumber_, kData_.cols()); // The normalized point coordinates
				Eigen::Matrix3d normalizingTransformSource, // The normalizing transformations in the source image
					normalizingTransformDestination; // The normalizing transformations in the destination image

				// Normalize the point coordinates to achieve numerical stability when
				// applying the least-squares model fitting.
				if (!normalizePoints(kData_, // The data points
					kSample_, // The points to which the model will be fit
					kSampleNumber_, // The number of points
					normalizedPoints, // The normalized point coordinates
					normalizingTransformSource, // The normalizing transformation in the first image
					normalizingTransformDestination)) // The normalizing transformation in the second image
					return false;
				
				const size_t kInitialModelCount = models_->size();

				// The four point fundamental matrix fitting algorithm
				if (!nonMinimalSolver->estimateModel(normalizedPoints,
					nullptr,
					kSampleNumber_,
					*models_,
					kWeights_))
					return false;

				// Denormalizing the newly estimated homography matrices
				const Eigen::Matrix3d kNormalizingTransformDestinationInverse = normalizingTransformDestination.inverse();
				for (size_t i = kInitialModelCount; i < models_->size(); ++i)
					(*models_)[i].setData(kNormalizingTransformDestinationInverse * (*models_)[i].getData() * normalizingTransformSource);

				// Nonlinear refinement: polish the (denormalized) DLT homography by
				// minimizing the robust reprojection error in pixel space. The DLT
				// only minimizes algebraic error; this is the homography analog of
				// the bundle adjustment run for F/E.
				if (useBundleRefinement)
				{
					static thread_local std::vector<Eigen::Vector2d> x1, x2;
					static thread_local std::vector<double> weights;
					x1.resize(kSampleNumber_);
					x2.resize(kSampleNumber_);
					weights.clear();
					const bool kUseWeights = kWeights_ != nullptr;
					if (kUseWeights)
						weights.resize(kSampleNumber_);
					for (size_t i = 0; i < kSampleNumber_; ++i)
					{
						const size_t idx = kSample_ == nullptr ? i : kSample_[i];
						x1[i] = Eigen::Vector2d(kData_(idx, 0), kData_(idx, 1));
						x2[i] = Eigen::Vector2d(kData_(idx, 2), kData_(idx, 3));
						if (kUseWeights)
							weights[i] = kWeights_[i];
					}

					// Robust Cauchy loss at half the inlier threshold, matching the
					// F/E bundle-adjustment configuration.
					poselib::BundleOptions bundleOptions;
					bundleOptions.loss_type = poselib::BundleOptions::LossType::CAUCHY;
					bundleOptions.loss_scale = 0.5 * bundleLossScale;
					bundleOptions.max_iterations = bundleMaxIterations;

					for (auto &model : *models_)
					{
						Eigen::Matrix3d homography = model.getData().block<3, 3>(0, 0);
						poselib::refine_homography(x1, x2, &homography, bundleOptions, weights);
						if (homography.allFinite())
							model.getMutableData() = homography;
					}
				}
				return true;
			}

			FORCE_INLINE double squaredResidual(const double* point_,
				const models::Model& model_) const override
			{
				return squaredResidual(point_, model_.getData());
			}

			FORCE_INLINE double squaredResidual(const double* point_,
				const ModelMatrix& descriptor_) const
			{
				// Use const references to avoid copying
				const double &x1 = point_[0],
							&y1 = point_[1],
							&x2 = point_[2],
							&y2 = point_[3];

				// Cache repeated calculations
				const double &descriptor00 = descriptor_(0, 0),
							&descriptor01 = descriptor_(0, 1),
							&descriptor02 = descriptor_(0, 2),
							&descriptor10 = descriptor_(1, 0),
							&descriptor11 = descriptor_(1, 1),
							&descriptor12 = descriptor_(1, 2),
							&descriptor20 = descriptor_(2, 0),
							&descriptor21 = descriptor_(2, 1),
							&descriptor22 = descriptor_(2, 2);

				// Compute intermediate terms
				const double t3 = descriptor20 * x1 + descriptor21 * y1 + descriptor22;
				if (std::abs(t3) < 1e-12)
					return 1e18;
				const double invT3 = 1.0 / t3;
				const double t1 = (descriptor00 * x1 + descriptor01 * y1 + descriptor02) * invT3;
				const double t2 = (descriptor10 * x1 + descriptor11 * y1 + descriptor12) * invT3;

				// Compute differences
				const double d1 = x2 - t1;
				const double d2 = y2 - t2;

				// Return squared residual
				return d1 * d1 + d2 * d2;
				
				/*const double 
                    &x1 = point_(0),
					&y1 = point_(1),
					&x2 = point_(2),
					&y2 = point_(3);

				const double t1 = descriptor_(0, 0) * x1 + descriptor_(0, 1) * y1 + descriptor_(0, 2);
				const double t2 = descriptor_(1, 0) * x1 + descriptor_(1, 1) * y1 + descriptor_(1, 2);
				const double t3 = descriptor_(2, 0) * x1 + descriptor_(2, 1) * y1 + descriptor_(2, 2);

				const double d1 = x2 - (t1 / t3);
				const double d2 = y2 - (t2 / t3);

				return d1 * d1 + d2 * d2;*/
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

			FORCE_INLINE bool normalizePoints(
				const DataMatrix& kData_, // The data points
				const size_t *kSample_, // The points to which the model will be fit
				const size_t &kSampleNumber_,// The number of points
				DataMatrix &kNormalizedPoints_, // The normalized point coordinates
				Eigen::Matrix3d &kNormalizingTransformSource_, // The normalizing transformation in the first image
				Eigen::Matrix3d &kNormalizingTransformDestination_) const // The normalizing transformation in the second image
			{
				const int &cols = kData_.cols();

				double massPointSrc[2], // Mass point in the first image
					massPointDst[2]; // Mass point in the second image

				// Initializing the mass point coordinates
				massPointSrc[0] =
					massPointSrc[1] =
					massPointDst[0] =
					massPointDst[1] =
					0.0;

				// Calculating the mass points in both images
				for (size_t i = 0; i < kSampleNumber_; ++i)
				{
					// Get index of the current point
					const size_t idx = (kSample_ == nullptr) ? i : kSample_[i];

					// Add the coordinates to that of the mass points
					massPointSrc[0] += kData_(idx, 0);
					massPointSrc[1] += kData_(idx, 1);
					massPointDst[0] += kData_(idx, 2);
					massPointDst[1] += kData_(idx, 3);
				}

				// Get the average
				massPointSrc[0] /= kSampleNumber_;
				massPointSrc[1] /= kSampleNumber_;
				massPointDst[0] /= kSampleNumber_;
				massPointDst[1] /= kSampleNumber_;

				// Get the mean distance from the mass points
				double average_distance_src = 0.0,
					average_distance_dst = 0.0;
				for (size_t i = 0; i < kSampleNumber_; ++i)
				{
					// Get index of the current point
					const size_t idx = (kSample_ == nullptr) ? i : kSample_[i];

					const double &x1 = kData_(idx, 0);
					const double &y1 = kData_(idx, 1);
					const double &x2 = kData_(idx, 2);
					const double &y2 = kData_(idx, 3);

					const double dx1 = massPointSrc[0] - x1;
					const double dy1 = massPointSrc[1] - y1;
					const double dx2 = massPointDst[0] - x2;
					const double dy2 = massPointDst[1] - y2;

					average_distance_src += sqrt(dx1 * dx1 + dy1 * dy1);
					average_distance_dst += sqrt(dx2 * dx2 + dy2 * dy2);
				}

				average_distance_src /= kSampleNumber_;
				average_distance_dst /= kSampleNumber_;
				if (average_distance_src < 1e-9 || average_distance_dst < 1e-9)
					return false;

				// Calculate the sqrt(2) / MeanDistance ratios
				const double ratioSrc = M_SQRT2 / average_distance_src;
				const double ratioDst = M_SQRT2 / average_distance_dst;

				// Compute the normalized coordinates
				for (size_t i = 0; i < kSampleNumber_; ++i)
				{
					// Get index of the current point
					const size_t idx = (kSample_ == nullptr) ? i : kSample_[i];

					const double &x1 = kData_(idx, 0);
					const double &y1 = kData_(idx, 1);
					const double &x2 = kData_(idx, 2);
					const double &y2 = kData_(idx, 3);

                    kNormalizedPoints_(i, 0) = (x1 - massPointSrc[0]) * ratioSrc;
                    kNormalizedPoints_(i, 1) = (y1 - massPointSrc[1]) * ratioSrc;
                    kNormalizedPoints_(i, 2) = (x2 - massPointDst[0]) * ratioDst;
                    kNormalizedPoints_(i, 3) = (y2 - massPointDst[1]) * ratioDst;

					for (size_t c = 4; c < static_cast<size_t>(cols); ++c)
						kNormalizedPoints_(i, c) = kData_(idx, c);
				}

				// Creating the normalizing transformations
				kNormalizingTransformSource_ << ratioSrc, 0, -ratioSrc * massPointSrc[0],
					0, ratioSrc, -ratioSrc * massPointSrc[1],
					0, 0, 1;

				kNormalizingTransformDestination_ << ratioDst, 0, -ratioDst * massPointDst[0],
					0, ratioDst, -ratioDst * massPointDst[1],
					0, 0, 1;
				return true;
			}

			// Calculates the cross-product of two vectors
			FORCE_INLINE void crossProduct(
				Eigen::Vector3d &result_,
				const double *vector1_,
				const double *vector2_,
				const unsigned int st_) const
			{
				const double &v1_0 = vector1_[0];
				const double &v1_st = vector1_[st_];
				const double &v2_0 = vector2_[0];
				const double &v2_st = vector2_[st_];

				result_[0] = v1_st - v2_st;
				result_[1] = v2_0 - v1_0;
				result_[2] = v1_0 * v2_st - v1_st * v2_0;
			}

			// Enable a quick check to see if the model is valid. This can be a geometric
			// check or some other verification of the model structure.
			FORCE_INLINE bool isValidModel(models::Model& model_,
				const DataMatrix& kData_,
				const size_t* kMinimalSample_,
				const double kThreshold_,
				bool& modelUpdated_) const override
			{ 
				const auto &H = model_.getData();
				if (!H.allFinite())
					return false;

				const double fNorm = H.norm();
				if (fNorm < 1e-12)
					return false;

				const double kDeterminant = std::abs(H.determinant()) / (fNorm * fNorm * fNorm);
				if (kDeterminant < 1e-13)
					return false;

				return true;
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			FORCE_INLINE bool isValidSample(
				const DataMatrix& kData_, // All data points
				const size_t *kSample_) const override // The indices of the selected points
			{
				if (sampleSize() < 4)
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
					return false;

				return true;
			}
		};
	}
}