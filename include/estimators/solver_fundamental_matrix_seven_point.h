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
			class FundamentalMatrixSevenPointSolver : public AbstractSolver
			{
			public:
				FundamentalMatrixSevenPointSolver()
				{
				}

				~FundamentalMatrixSevenPointSolver()
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
					return 3;
				}
				
				// The minimum number of points required for the estimation
				size_t sampleSize() const override
				{
					return 7;
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
					
				FORCE_INLINE int solveCubicReal(double c2, double c1, double c0, double roots[3]) const;
			};

			FORCE_INLINE int FundamentalMatrixSevenPointSolver::solveCubicReal(double c2, double c1, double c0, double roots[3]) const
			{
				double a = c1 - c2 * c2 / 3.0;
				double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
				double c = b * b / 4.0 + a * a * a / 27.0;
				int n_roots;
				if (c > 0) {
					c = std::sqrt(c);
					b *= -0.5;
					roots[0] = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
					n_roots = 1;
				} else {
					c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
					double d = 2.0 * std::sqrt(-a / 3.0);
					roots[0] = d * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
					roots[1] = d * std::cos(std::acos(c) / 3.0 - 2.09439510239319526263557236234192) - c2 / 3.0; // 2*pi/3
					roots[2] = d * std::cos(std::acos(c) / 3.0 - 4.18879020478639052527114472468384) - c2 / 3.0; // 4*pi/3
					n_roots = 3;
				}

				// single newton iteration
				for (int i = 0; i < n_roots; ++i) {
					double x = roots[i];
					double x2 = x * x;
					double x3 = x * x2;
					double dx = -(x3 + c2 * x2 + c1 * x + c0) / (3 * x2 + 2 * c2 * x + c1);
					roots[i] += dx;
				}
				return n_roots;
			}

			FORCE_INLINE bool FundamentalMatrixSevenPointSolver::estimateMinimalModel(
                const DataMatrix& kData_, // The set of data points
                const size_t *kSample_, // The sample used for the estimation
                const size_t kSampleNumber_, // The size of the sample
                std::vector<models::Model> &models_, // The estimated model parameters
                const double *kWeights_) const // The weight for each point
			{
				constexpr size_t kEquationNumber = 1;
				Eigen::Matrix<double, 7, 9> coefficients;

				size_t rowIdx = 0;
				double weight = 1.0;
                
				for (size_t i = 0; i < kSampleNumber_; ++i)
				{
					const size_t idx =
						kSample_ == nullptr ? i : kSample_[i];

					const double
						&x0 = kData_(idx, 0),
						&y0 = kData_(idx, 1),
						&x1 = kData_(idx, 2),
						&y1 = kData_(idx, 3);

					// If not weighted least-squares is applied
					if (kWeights_ == nullptr)
					{
						coefficients(rowIdx, 0) = x1 * x0;
						coefficients(rowIdx, 1) = x1 * y0;
						coefficients(rowIdx, 2) = x1;
						coefficients(rowIdx, 3) = y1 * x0;
						coefficients(rowIdx, 4) = y1 * y0;
						coefficients(rowIdx, 5) = y1;
						coefficients(rowIdx, 6) = x0;
						coefficients(rowIdx, 7) = y0;
						coefficients(rowIdx, 8) = 1;
					}
					else
					{
						// If weighted least-squares is applied
						weight = kWeights_[idx];

						// Precalculate these values to avoid calculating them multiple times
						const double
							kWeightTimesX0 = weight * x0,
							kWeightTimesY0 = weight * y0,
							kWeightTimesX1 = weight * x1,
							kWeightTimesY1 = weight * y1;

						coefficients(rowIdx, 0) = kWeightTimesX1 * x0;
						coefficients(rowIdx, 1) = kWeightTimesX1 * y0;
						coefficients(rowIdx, 2) = kWeightTimesX1;
						coefficients(rowIdx, 3) = kWeightTimesY1 * x0;
						coefficients(rowIdx, 4) = kWeightTimesY1 * y0;
						coefficients(rowIdx, 5) = kWeightTimesY1;
						coefficients(rowIdx, 6) = kWeightTimesX0;
						coefficients(rowIdx, 7) = kWeightTimesY0;
						coefficients(rowIdx, 8) = weight;
					}
					++rowIdx;
				}

				// The null-space of the matrix is the solution of the problem
				Eigen::Matrix<double, 9, 1> f1, f2;

				// For the minimal problem, the matrix is small and, thus, fullPivLu decomposition is significantly
				// faster than both JacobiSVD and BDCSVD methods.
				// https://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html
				const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients);
				if (lu.dimensionOfKernel() != 2) 
					return false;

				const Eigen::Matrix<double, 9, 2> N = 
					lu.kernel();

				// coefficients for det(F(x)) = 0
				const double c3 = N(0, 0) * N(4, 0) * N(8, 0) - N(0, 0) * N(5, 0) * N(7, 0) - N(1, 0) * N(3, 0) * N(8, 0) +
								N(1, 0) * N(5, 0) * N(6, 0) + N(2, 0) * N(3, 0) * N(7, 0) - N(2, 0) * N(4, 0) * N(6, 0);
				const double c2 = N(0, 0) * N(4, 0) * N(8, 1) + N(0, 0) * N(4, 1) * N(8, 0) - N(0, 0) * N(5, 0) * N(7, 1) -
								N(0, 0) * N(5, 1) * N(7, 0) + N(0, 1) * N(4, 0) * N(8, 0) - N(0, 1) * N(5, 0) * N(7, 0) -
								N(1, 0) * N(3, 0) * N(8, 1) - N(1, 0) * N(3, 1) * N(8, 0) + N(1, 0) * N(5, 0) * N(6, 1) +
								N(1, 0) * N(5, 1) * N(6, 0) - N(1, 1) * N(3, 0) * N(8, 0) + N(1, 1) * N(5, 0) * N(6, 0) +
								N(2, 0) * N(3, 0) * N(7, 1) + N(2, 0) * N(3, 1) * N(7, 0) - N(2, 0) * N(4, 0) * N(6, 1) -
								N(2, 0) * N(4, 1) * N(6, 0) + N(2, 1) * N(3, 0) * N(7, 0) - N(2, 1) * N(4, 0) * N(6, 0);
				const double c1 = N(0, 0) * N(4, 1) * N(8, 1) - N(0, 0) * N(5, 1) * N(7, 1) + N(0, 1) * N(4, 0) * N(8, 1) +
								N(0, 1) * N(4, 1) * N(8, 0) - N(0, 1) * N(5, 0) * N(7, 1) - N(0, 1) * N(5, 1) * N(7, 0) -
								N(1, 0) * N(3, 1) * N(8, 1) + N(1, 0) * N(5, 1) * N(6, 1) - N(1, 1) * N(3, 0) * N(8, 1) -
								N(1, 1) * N(3, 1) * N(8, 0) + N(1, 1) * N(5, 0) * N(6, 1) + N(1, 1) * N(5, 1) * N(6, 0) +
								N(2, 0) * N(3, 1) * N(7, 1) - N(2, 0) * N(4, 1) * N(6, 1) + N(2, 1) * N(3, 0) * N(7, 1) +
								N(2, 1) * N(3, 1) * N(7, 0) - N(2, 1) * N(4, 0) * N(6, 1) - N(2, 1) * N(4, 1) * N(6, 0);
				const double c0 = N(0, 1) * N(4, 1) * N(8, 1) - N(0, 1) * N(5, 1) * N(7, 1) - N(1, 1) * N(3, 1) * N(8, 1) +
								N(1, 1) * N(5, 1) * N(6, 1) + N(2, 1) * N(3, 1) * N(7, 1) - N(2, 1) * N(4, 1) * N(6, 1);

				// Solve the cubic
				double inv_c3 = 1.0 / c3;
				double roots[3];
				int n_roots = solveCubicReal(c2 * inv_c3, c1 * inv_c3, c0 * inv_c3, roots);

				for (int i = 0; i < n_roots; ++i) 
				{
					Eigen::Matrix<double, 9, 1> f = N.col(0) * roots[i] + N.col(1);
					f.normalize();

					models::Model model;
					auto &modelData = model.getMutableData();
					modelData.resize(3, 3);
					modelData << f[0], f[1], f[2],
						f[3], f[4], f[5],
						f[6], f[7], f[8];
					models_.emplace_back(model);
				}

				return true;
			}

			FORCE_INLINE bool FundamentalMatrixSevenPointSolver::estimateNonMinimalModel(
                const DataMatrix& kData_, // The set of data points
                const size_t *kSample_, // The sample used for the estimation
                const size_t kSampleNumber_, // The size of the sample
                std::vector<models::Model> &models_, // The estimated model parameters
                const double *kWeights_) const // The weight for each point
			{
				constexpr size_t kEquationNumber = 1;
				Eigen::Matrix<double, Eigen::Dynamic, 9> coefficients(kSampleNumber_, 9);

				size_t rowIdx = 0;
				double weight = 1.0;
                
				for (size_t i = 0; i < kSampleNumber_; ++i)
				{
					const size_t idx =
						kSample_ == nullptr ? i : kSample_[i];

					const double
						&x0 = kData_(idx, 0),
						&y0 = kData_(idx, 1),
						&x1 = kData_(idx, 2),
						&y1 = kData_(idx, 3);

					// If not weighted least-squares is applied
					if (kWeights_ == nullptr)
					{
						coefficients(rowIdx, 0) = x1 * x0;
						coefficients(rowIdx, 1) = x1 * y0;
						coefficients(rowIdx, 2) = x1;
						coefficients(rowIdx, 3) = y1 * x0;
						coefficients(rowIdx, 4) = y1 * y0;
						coefficients(rowIdx, 5) = y1;
						coefficients(rowIdx, 6) = x0;
						coefficients(rowIdx, 7) = y0;
						coefficients(rowIdx, 8) = 1;
					}
					else
					{
						// If weighted least-squares is applied
						weight = kWeights_[idx];

						// Precalculate these values to avoid calculating them multiple times
						const double
							kWeightTimesX0 = weight * x0,
							kWeightTimesY0 = weight * y0,
							kWeightTimesX1 = weight * x1,
							kWeightTimesY1 = weight * y1;

						coefficients(rowIdx, 0) = kWeightTimesX1 * x0;
						coefficients(rowIdx, 1) = kWeightTimesX1 * y0;
						coefficients(rowIdx, 2) = kWeightTimesX1;
						coefficients(rowIdx, 3) = kWeightTimesY1 * x0;
						coefficients(rowIdx, 4) = kWeightTimesY1 * y0;
						coefficients(rowIdx, 5) = kWeightTimesY1;
						coefficients(rowIdx, 6) = kWeightTimesX0;
						coefficients(rowIdx, 7) = kWeightTimesY0;
						coefficients(rowIdx, 8) = weight;
					}
					++rowIdx;
				}

				// The null-space of the matrix is the solution of the problem
				Eigen::Matrix<double, 9, 1> f1, f2;

				// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
				// the solution is linear subspace of dimensionality 2.
				// => use the last two singular std::vectors as a basis of the space
				// (according to SVD properties)
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					// Theoretically, it would be faster to apply SVD only to matrix coefficients, but
					// multiplication is faster than SVD in the Eigen library. Therefore, it is faster
					// to apply SVD to a smaller matrix.
					coefficients.transpose() * coefficients,
					Eigen::ComputeFullV);
				f1 = svd.matrixV().block<9, 1>(0, 7);
				f2 = svd.matrixV().block<9, 1>(0, 8);

				// f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary f. matrix.
				// as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
				// so f ~ lambda*f1 + (1 - lambda)*f2.
				// use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
				// it will be a cubic equation.
				// find c - polynomial coefficients.
				f1 -= f2;

				const double 
					&f2_0 = f2[0],
					&f2_1 = f2[1],
					&f2_2 = f2[2],
					&f2_3 = f2[3],
					&f2_4 = f2[4],
					&f2_5 = f2[5],
					&f2_6 = f2[6],
					&f2_7 = f2[7],
					&f2_8 = f2[8],
					&f1_0 = f1[0],
					&f1_1 = f1[1],
					&f1_2 = f1[2],
					&f1_3 = f1[3],
					&f1_4 = f1[4],
					&f1_5 = f1[5],
					&f1_6 = f1[6],
					&f1_7 = f1[7],
					&f1_8 = f1[8];

				double t0, t1, t2;
				t0 = f2_4 * f2_8 - f2_5 * f2_7;
				t1 = f2_3 * f2_8 - f2_5 * f2_6;
				t2 = f2_3 * f2_7 - f2_4 * f2_6;

				double c[4];
				c[0] = f2_0 * t0 - f2_1 * t1 + f2_2 * t2;
				
				c[1] = f1_0 * t0 - f1_1 * t1 + f1_2 * t2 -
					f1_3 * (f2_1 * f2_8 - f2_2 * f2_7) +
					f1_4 * (f2_0 * f2_8 - f2_2 * f2_6) -
					f1_5 * (f2_0 * f2_7 - f2_1 * f2_6) +
					f1_6 * (f2_1 * f2_5 - f2_2 * f2_4) -
					f1_7 * (f2_0 * f2_5 - f2_2 * f2_3) +
					f1_8 * (f2_0 * f2_4 - f2_1 * f2_3);

				t0 = f1_4 * f1_8 - f1_5 * f1_7;
				t1 = f1_3 * f1_8 - f1_5 * f1_6;
				t2 = f1_3 * f1_7 - f1_4 * f1_6;

				c[2] = f2_0 * t0 - f2_1 * t1 + f2_2 * t2 -
					f2_3 * (f1_1 * f1_8 - f1_2 * f1_7) +
					f2_4 * (f1_0 * f1_8 - f1_2 * f1_6) -
					f2_5 * (f1_0 * f1_7 - f1_1 * f1_6) +
					f2_6 * (f1_1 * f1_5 - f1_2 * f1_4) -
					f2_7 * (f1_0 * f1_5 - f1_2 * f1_3) +
					f2_8 * (f1_0 * f1_4 - f1_1 * f1_3);

				c[3] = f1_0 * t0 - f1_1 * t1 + f1_2 * t2;
				
				// Check if the sum of the polynomical coefficients is close to zero. 
				// In this case "psolve.realRoots(real_roots)" gets into an infinite loop.
				if (fabs(c[0]+c[1]+c[2]+c[3]) < 1e-9 || 
					fabs(c[0]) < std::numeric_limits<double>::epsilon() ||
					fabs(c[1]) < std::numeric_limits<double>::epsilon() ||
					fabs(c[2]) < std::numeric_limits<double>::epsilon() ||
					fabs(c[3]) < std::numeric_limits<double>::epsilon()) 
					return false;

				// solve the cubic equation; there can be 1 to 3 roots ...
				Eigen::Matrix<double, 4, 1> polynomial;
				for (auto i = 0; i < 4; ++i)
					polynomial(i) = c[i];
				Eigen::PolynomialSolver<double, 3> psolve(polynomial);

				std::vector<double> real_roots;
				psolve.realRoots(real_roots);

				size_t n = real_roots.size();
				if (n < 1 || n > 3)
					return false;

				Eigen::Matrix<double, 9, 1> f;
				for (const double &root : real_roots)
				{
					// for each root form the fundamental matrix
					double lambda = root, 
						mu = 1.;
					double s = f1[8] * root + f2[8];

					// normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
					if (fabs(s) > std::numeric_limits<double>::epsilon())
					{
						mu = 1.0 / s;
						lambda *= mu;

						f = f1 * lambda + f2 * mu;

						models::Model model;
						auto &modelData = model.getMutableData();
						modelData.resize(3, 3);
						modelData << f[0], f[1], f[2],
							f[3], f[4], f[5],
							f[6], f[7], 1.0;
						models_.emplace_back(model);
					}
				}

				return true;
			}

			FORCE_INLINE bool FundamentalMatrixSevenPointSolver::estimateModel(
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