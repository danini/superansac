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

						const double weight = kWeights_[i];
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
				// Direct block-sparse accumulation of the 9x9 normal matrix A^T A:
				// Every 3x3 block is a weighted sum of P = p1 * p1^T (where p1 = [x1, y1, 1]^T),
				// avoiding the 2N x 9 zero-padded coefficient matrix and full GEMM.
				Eigen::Matrix3d S1 = Eigen::Matrix3d::Zero();
				Eigen::Matrix3d Sx = Eigen::Matrix3d::Zero();
				Eigen::Matrix3d Sy = Eigen::Matrix3d::Zero();
				Eigen::Matrix3d Sr2 = Eigen::Matrix3d::Zero();

				for (size_t i = 0; i < kSampleNumber_; ++i)
				{
					const size_t idx = kSample_ == nullptr ? i : kSample_[i];
					const double x1 = kData_(idx, 0);
					const double y1 = kData_(idx, 1);
					const double x2 = kData_(idx, 2);
					const double y2 = kData_(idx, 3);
					const double w = (kWeights_ != nullptr) ? kWeights_[i] : 1.0;
					const double w2 = w * w;
					if (w2 <= 0.0)
						continue;

					Eigen::Matrix3d P;
					P(0, 0) = x1 * x1; P(0, 1) = x1 * y1; P(0, 2) = x1;
					P(1, 0) = P(0, 1); P(1, 1) = y1 * y1; P(1, 2) = y1;
					P(2, 0) = x1;      P(2, 1) = y1;      P(2, 2) = 1.0;

					S1.noalias()  += w2 * P;
					Sx.noalias()  -= (w2 * x2) * P;
					Sy.noalias()  -= (w2 * y2) * P;
					Sr2.noalias() += (w2 * (x2 * x2 + y2 * y2)) * P;
				}

				Eigen::Matrix<double, 9, 9> kNormalMatrix = Eigen::Matrix<double, 9, 9>::Zero();
				kNormalMatrix.block<3, 3>(0, 0) = S1;
				kNormalMatrix.block<3, 3>(3, 3) = S1;
				kNormalMatrix.block<3, 3>(6, 6) = Sr2;
				kNormalMatrix.block<3, 3>(0, 6) = Sx;
				kNormalMatrix.block<3, 3>(6, 0) = Sx;
				kNormalMatrix.block<3, 3>(3, 6) = Sy;
				kNormalMatrix.block<3, 3>(6, 3) = Sy;

				// Homogeneous Total Least Squares via symmetric eigendecomposition of the 9x9 PSD
				// normal matrix (equivalent to right singular vector of smallest singular value, ~6x faster).
				Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eig(kNormalMatrix);
				if (eig.info() != Eigen::Success)
					return false;
				const Eigen::Matrix<double, 9, 1> h = eig.eigenvectors().col(0);

				if (!h.allFinite())
					return false;

				models::Model model;
				auto &data = model.getMutableData();
				data.resize(3, 3);
				data << h(0), h(1), h(2),
					h(3), h(4), h(5),
					h(6), h(7), h(8);
				if (std::abs(data(2, 2)) > 1e-12)
					data /= data(2, 2);
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