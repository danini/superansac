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

#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class FundamentalMatrixEightPointSolver : public AbstractSolver
			{
			public:
				FundamentalMatrixEightPointSolver()
				{
				}

				~FundamentalMatrixEightPointSolver()
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

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				FORCE_INLINE bool estimateModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_ = nullptr) const override; // The weight for each point
			protected:
				FORCE_INLINE bool normalizePoints(
					const DataMatrix& kData_, // The data points
					const size_t *kSample_, // The points to which the model will be fit
					const size_t &kSampleNumber_,// The number of points
					DataMatrix &kNormalizedPoints_, // The normalized point coordinates
					Eigen::Matrix3d &kNormalizingTransformSource_, // The normalizing transformation in the first image
					Eigen::Matrix3d &kNormalizingTransformDestination_) const; // The normalizing transformation in the second image
			};

			FORCE_INLINE bool FundamentalMatrixEightPointSolver::normalizePoints(
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
					const size_t idx =
						kSample_ == nullptr ? i : kSample_[i];

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
					const size_t idx =
						kSample_ == nullptr ? i : kSample_[i];

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

				// Calculate the sqrt(2) / MeanDistance ratios
				const double ratioSrc = M_SQRT2 / average_distance_src;
				const double ratioDst = M_SQRT2 / average_distance_dst;

				// Compute the normalized coordinates
				for (size_t i = 0; i < kSampleNumber_; ++i)
				{
					// Get index of the current point
					const size_t idx =
						kSample_ == nullptr ? i : kSample_[i];

					const double &x1 = kData_(idx, 0);
					const double &y1 = kData_(idx, 1);
					const double &x2 = kData_(idx, 2);
					const double &y2 = kData_(idx, 3);

                    kNormalizedPoints_(i, 0) = (x1 - massPointSrc[0]) * ratioSrc;
                    kNormalizedPoints_(i, 1) = (y1 - massPointSrc[1]) * ratioSrc;
                    kNormalizedPoints_(i, 2) = (x2 - massPointDst[0]) * ratioDst;
                    kNormalizedPoints_(i, 3) = (y2 - massPointDst[1]) * ratioDst;

					for (size_t i = 4; i < cols; ++i)
						kNormalizedPoints_(idx, i) = kData_(idx, i);
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

			FORCE_INLINE bool FundamentalMatrixEightPointSolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{				
				DataMatrix normalizedPoints(kSampleNumber_, kData_.cols()); // The normalized point coordinates
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
				
				constexpr size_t kEquationNumber = 1;
				Eigen::MatrixXd coefficients;
				coefficients.resize(kSampleNumber_ * kEquationNumber, 9);

				size_t rowIdx = 0;
				double weight = 1.0;
                
				for (size_t i = 0; i < kSampleNumber_; ++i)
				{
					const double
						&x0 = normalizedPoints(i, 0),
						&y0 = normalizedPoints(i, 1),
						&x1 = normalizedPoints(i, 2),
						&y1 = normalizedPoints(i, 3);

					// If not weighted least-squares is applied
					if (kWeights_ == nullptr)
					{
						coefficients(i, 0) = x1 * x0;
						coefficients(i, 1) = x1 * y0;
						coefficients(i, 2) = x1;
						coefficients(i, 3) = y1 * x0;
						coefficients(i, 4) = y1 * y0;
						coefficients(i, 5) = y1;
						coefficients(i, 6) = x0;
						coefficients(i, 7) = y0;
						coefficients(i, 8) = 1;
					}
					else
					{
						// If weighted least-squares is applied
						weight = kWeights_[i];

						// Precalculate these values to avoid calculating them multiple times
						const double
							kWeightTimesX0 = weight * x0,
							kWeightTimesY0 = weight * y0,
							kWeightTimesX1 = weight * x1,
							kWeightTimesY1 = weight * y1;

						coefficients(i, 0) = kWeightTimesX1 * x0;
						coefficients(i, 1) = kWeightTimesX1 * y0;
						coefficients(i, 2) = kWeightTimesX1;
						coefficients(i, 3) = kWeightTimesY1 * x0;
						coefficients(i, 4) = kWeightTimesY1 * y0;
						coefficients(i, 5) = kWeightTimesY1;
						coefficients(i, 6) = kWeightTimesX0;
						coefficients(i, 7) = kWeightTimesY0;
						coefficients(i, 8) = weight;
					}
				}

				// A*(f11 f12 ... f33)' = 0 is singular (8 equations for 9 variables), so
				// the solution is linear subspace of dimensionality 1.
				// => use the last two singular std::vectors as a basis of the space
				// (according to SVD properties)
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					// Theoretically, it would be faster to apply SVD only to matrix coefficients, but
					// multiplication is faster than SVD in the Eigen library. Therefore, it is faster
					// to apply SVD to a smaller matrix.
					coefficients.transpose() * coefficients,
					Eigen::ComputeFullV);
				const Eigen::Matrix<double, 9, 1>& kNullSpace =
					svd.matrixV().rightCols<1>();

				if (kNullSpace.hasNaN())
					return false;

				Eigen::Matrix3d normalizedModel;
				normalizedModel << kNullSpace(0), kNullSpace(1), kNullSpace(2),
					kNullSpace(3), kNullSpace(4), kNullSpace(5),
					kNullSpace(6), kNullSpace(7), kNullSpace(8);

				models::Model model;
				auto &modelData = model.getMutableData();
				
				// Transform the estimated fundamental matrix back to the not normalized space
				modelData = normalizingTransformDestination.transpose() * normalizedModel * normalizingTransformSource;

				models_.emplace_back(model);
				return true;
			}
		}
	}
}