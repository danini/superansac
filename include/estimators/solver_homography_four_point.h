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
			class HomographyFourPointSolver : public AbstractSolver
			{
			public:
				HomographyFourPointSolver()
				{
				}

				~HomographyFourPointSolver()
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
					return 4;
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
				FORCE_INLINE bool estimateNonMinimalModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_) const; // The weight for each point

				FORCE_INLINE bool estimateMinimalModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_) const; // The weight for each point
			};

			FORCE_INLINE bool HomographyFourPointSolver::estimateMinimalModel(
                const DataMatrix& kData_, // The set of data points
                const size_t *kSample_, // The sample used for the estimation
                const size_t kSampleNumber_, // The size of the sample
                std::vector<models::Model> &models_, // The estimated model parameters
                const double *kWeights_) const // The weight for each point
			{
				constexpr size_t kEquationNumber = 2;
				Eigen::Matrix<double, 8, 9> coefficients;
    			coefficients.setZero(); // Initialize the matrix with zeros

				size_t rowIdx = 0;

				// Remove branch from inner loop by handling weighted/unweighted separately
				if (kWeights_ == nullptr)
				{
					// Unweighted case - weight = 1.0, avoid unnecessary multiplications
					for (size_t i = 0; i < kSampleNumber_; ++i)
					{
						const size_t idx = kSample_ == nullptr ? i : kSample_[i];

						const double
							&x1 = kData_(idx, 0),
							&y1 = kData_(idx, 1),
							&x2 = kData_(idx, 2),
							&y2 = kData_(idx, 3);

						coefficients(rowIdx, 0) = -x1;
						coefficients(rowIdx, 1) = -y1;
						coefficients(rowIdx, 2) = -1.0;
						coefficients(rowIdx, 3) = 0.0;
						coefficients(rowIdx, 4) = 0.0;
						coefficients(rowIdx, 5) = 0.0;
						coefficients(rowIdx, 6) = x2 * x1;
						coefficients(rowIdx, 7) = x2 * y1;
						coefficients(rowIdx, 8) = -x2;
						++rowIdx;

						coefficients(rowIdx, 0) = 0.0;
						coefficients(rowIdx, 1) = 0.0;
						coefficients(rowIdx, 2) = 0.0;
						coefficients(rowIdx, 3) = -x1;
						coefficients(rowIdx, 4) = -y1;
						coefficients(rowIdx, 5) = -1.0;
						coefficients(rowIdx, 6) = y2 * x1;
						coefficients(rowIdx, 7) = y2 * y1;
						coefficients(rowIdx, 8) = -y2;
						++rowIdx;
					}
				}
				else
				{
					// Weighted case
					for (size_t i = 0; i < kSampleNumber_; ++i)
					{
						const size_t idx = kSample_ == nullptr ? i : kSample_[i];

						const double
							&x1 = kData_(idx, 0),
							&y1 = kData_(idx, 1),
							&x2 = kData_(idx, 2),
							&y2 = kData_(idx, 3);

						const double weight = kWeights_[idx];
						const double
							wx1 = weight * x1,
							wy1 = weight * y1,
							wx2 = weight * x2,
							wy2 = weight * y2;

						coefficients(rowIdx, 0) = -wx1;
						coefficients(rowIdx, 1) = -wy1;
						coefficients(rowIdx, 2) = -weight;
						coefficients(rowIdx, 3) = 0.0;
						coefficients(rowIdx, 4) = 0.0;
						coefficients(rowIdx, 5) = 0.0;
						coefficients(rowIdx, 6) = wx2 * x1;
						coefficients(rowIdx, 7) = wx2 * y1;
						coefficients(rowIdx, 8) = -wx2;
						++rowIdx;

						coefficients(rowIdx, 0) = 0.0;
						coefficients(rowIdx, 1) = 0.0;
						coefficients(rowIdx, 2) = 0.0;
						coefficients(rowIdx, 3) = -wx1;
						coefficients(rowIdx, 4) = -wy1;
						coefficients(rowIdx, 5) = -weight;
						coefficients(rowIdx, 6) = wy2 * x1;
						coefficients(rowIdx, 7) = wy2 * y1;
						coefficients(rowIdx, 8) = -wy2;
						++rowIdx;
					}
				}

				Eigen::Matrix<double, 8, 1> h;

				// Applying Gaussian Elimination to recover the null-space.
				// Average time over 100000 problem instances  
				// LLT solver (i.e., the fastest one in the Eigen library) = 4.2 microseconds
				// Gaussian Elimination = 3.6 microseconds
				superansac::utils::gaussElimination<8>(
					coefficients,
					h);

				if (h.hasNaN())
					return false;

				models::Model model;
				auto &modelData = model.getMutableData();
				modelData.resize(3, 3);
				modelData << h(0), h(1), h(2),
					h(3), h(4), h(5),
					h(6), h(7), 1.0;
				models_.emplace_back(model);
				return true;
			}

			FORCE_INLINE bool HomographyFourPointSolver::estimateNonMinimalModel(
                const DataMatrix& kData_, // The set of data points
                const size_t *kSample_, // The sample used for the estimation
                const size_t kSampleNumber_, // The size of the sample
                std::vector<models::Model> &models_, // The estimated model parameters
                const double *kWeights_) const // The weight for each point
			{
				constexpr size_t kEquationNumber = 2;
				const size_t columns = kData_.cols();
				const size_t kRowNumber = kEquationNumber * kSampleNumber_;
				DataMatrix coefficients(kRowNumber, 8);
				DataMatrix inhomogeneous(kRowNumber, 1);
    			coefficients.setZero(); // Initialize the matrix with zeros

				size_t rowIdx = 0;
				double weight = 1.0;

				for (size_t i = 0; i < kSampleNumber_; ++i)
				{
					const size_t idx =
						kSample_ == nullptr ? i : kSample_[i];

					const double
						& x1 = kData_(idx, 0),
						& y1 = kData_(idx, 1),
						& x2 = kData_(idx, 2),
						& y2 = kData_(idx, 3);

					if (kWeights_ != nullptr)
						weight = kWeights_[idx];

					const double
						minusWeightTimesX1 = -weight * x1,
						minusWeightTimesY1 = -weight * y1,
						weightTimesX2 = weight * x2,
						weightTimesY2 = weight * y2;

					coefficients(rowIdx, 0) = minusWeightTimesX1;
					coefficients(rowIdx, 1) = minusWeightTimesY1;
					coefficients(rowIdx, 2) = -weight;
					coefficients(rowIdx, 6) = weightTimesX2 * x1;
					coefficients(rowIdx, 7) = weightTimesX2 * y1;
					inhomogeneous(rowIdx) = -weightTimesX2;
					++rowIdx;

					coefficients(rowIdx, 3) = minusWeightTimesX1;
					coefficients(rowIdx, 4) = minusWeightTimesY1;
					coefficients(rowIdx, 5) = -weight;
					coefficients(rowIdx, 6) = weightTimesY2 * x1;
					coefficients(rowIdx, 7) = weightTimesY2 * y1;
					inhomogeneous(rowIdx) = -weightTimesY2;
					++rowIdx;
				}

				// Applying SVD to solve the problem
				Eigen::Matrix<double, 8, 1> 
					h = coefficients.colPivHouseholderQr().solve(inhomogeneous);

				models::Model model;
				auto &data = model.getMutableData();
				data.resize(3, 3);
				data << h(0), h(1), h(2),
					h(3), h(4), h(5),
					h(6), h(7), 1.0;
				models_.emplace_back(model);
				return true;
			}

			FORCE_INLINE bool HomographyFourPointSolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{				
				// If we have a minimal sample, it is usually enough to solve the problem with not necessarily
				// the most accurate solver. Therefore, we use normal equations for this
				if (kSampleNumber_ == sampleSize())
					return estimateMinimalModel(kData_,
						kSample_,
						kSampleNumber_,
						models_,
						kWeights_);
				return estimateNonMinimalModel(kData_,
					kSample_,
					kSampleNumber_,
					models_,
					kWeights_);
			}
		}
	}
}