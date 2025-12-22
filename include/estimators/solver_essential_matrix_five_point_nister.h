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

#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>

#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../utils/sturm.h"
#include "numerical_optimizer/essential.h"

namespace superansac
{
namespace estimator
{
namespace solver
{
	// Estimator for the essential matrix using the five-point Nistér formulation.
	class EssentialMatrixFivePointNisterSolver : public AbstractSolver
	{
	public:
		EssentialMatrixFivePointNisterSolver() {}
		~EssentialMatrixFivePointNisterSolver() {}

		bool   returnMultipleModels() const override { return maximumSolutions() > 1; }
		size_t maximumSolutions()     const override { return 10; }
		size_t sampleSize()           const override { return 5; }

		FORCE_INLINE bool estimateModel(
			const DataMatrix& kData_,
			const size_t *kSample_,
			const size_t kSampleNumber_,
			std::vector<models::Model> &models_,
			const double *kWeights_ = nullptr) const override;

	protected:
		FORCE_INLINE bool estimateMinimalModel(
			const DataMatrix& kData_,
			const size_t *kSample_,
			const size_t kSampleNumber_,
			std::vector<models::Model> &models_,
			const double *kWeights_) const;

		// Column-major coeffs (matches pointer math inside)
		FORCE_INLINE void computeTraceConstraints(
			const Eigen::Matrix<double, 4, 9> &N,
			Eigen::Matrix<double, 10, 20> &coeffs) const;

		FORCE_INLINE void o1 (const double a[4],  const double b[4],  double c[10])  const;
		FORCE_INLINE void o1p(const double a[4],  const double b[4],  double c[10])  const;
		FORCE_INLINE void o1m(const double a[4],  const double b[4],  double c[10])  const;
		FORCE_INLINE void o2 (const double a[10], const double b[4],  double c[20])  const;
		FORCE_INLINE void o2p(const double a[10], const double b[4],  double c[20])  const;
	};

	// a, b are first order polys [x,y,z,1]
	// c is degree 2 poly with order
	// [ x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]
	FORCE_INLINE void EssentialMatrixFivePointNisterSolver::o1(const double a[4], const double b[4], double c[10]) const
	{
		c[0] = a[0] * b[0];
		c[1] = a[0] * b[1] + a[1] * b[0];
		c[2] = a[0] * b[2] + a[2] * b[0];
		c[3] = a[0] * b[3] + a[3] * b[0];
		c[4] = a[1] * b[1];
		c[5] = a[1] * b[2] + a[2] * b[1];
		c[6] = a[1] * b[3] + a[3] * b[1];
		c[7] = a[2] * b[2];
		c[8] = a[2] * b[3] + a[3] * b[2];
		c[9] = a[3] * b[3];
	}

	FORCE_INLINE void EssentialMatrixFivePointNisterSolver::o1p(const double a[4], const double b[4], double c[10]) const
	{
		c[0] += a[0] * b[0];
		c[1] += a[0] * b[1] + a[1] * b[0];
		c[2] += a[0] * b[2] + a[2] * b[0];
		c[3] += a[0] * b[3] + a[3] * b[0];
		c[4] += a[1] * b[1];
		c[5] += a[1] * b[2] + a[2] * b[1];
		c[6] += a[1] * b[3] + a[3] * b[1];
		c[7] += a[2] * b[2];
		c[8] += a[2] * b[3] + a[3] * b[2];
		c[9] += a[3] * b[3];
	}

	FORCE_INLINE void EssentialMatrixFivePointNisterSolver::o1m(const double a[4], const double b[4], double c[10]) const
	{
		c[0] -= a[0] * b[0];
		c[1] -= a[0] * b[1] + a[1] * b[0];
		c[2] -= a[0] * b[2] + a[2] * b[0];
		c[3] -= a[0] * b[3] + a[3] * b[0];
		c[4] -= a[1] * b[1];
		c[5] -= a[1] * b[2] + a[2] * b[1];
		c[6] -= a[1] * b[3] + a[3] * b[1];
		c[7] -= a[2] * b[2];
		c[8] -= a[2] * b[3] + a[3] * b[2];
		c[9] -= a[3] * b[3];
	}

	// a is second degree poly with order
	// [ x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]
	// b is first degree with order [x y z 1]
	// c is third degree with order (same as Nistér's paper)
	// [ x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2, x*y*z, x*y, x*z^2, x*z, x,
	//   y*z^2, y*z, y, z^3, z^2, z, 1]
	FORCE_INLINE void EssentialMatrixFivePointNisterSolver::o2(const double a[10], const double b[4], double c[20]) const
	{
		c[0] = a[0] * b[0];
		c[1] = a[4] * b[1];
		c[2] = a[0] * b[1] + a[1] * b[0];
		c[3] = a[1] * b[1] + a[4] * b[0];
		c[4] = a[0] * b[2] + a[2] * b[0];
		c[5] = a[0] * b[3] + a[3] * b[0];
		c[6] = a[4] * b[2] + a[5] * b[1];
		c[7] = a[4] * b[3] + a[6] * b[1];
		c[8] = a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
		c[9] = a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
		c[10] = a[2] * b[2] + a[7] * b[0];
		c[11] = a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
		c[12] = a[3] * b[3] + a[9] * b[0];
		c[13] = a[5] * b[2] + a[7] * b[1];
		c[14] = a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
		c[15] = a[6] * b[3] + a[9] * b[1];
		c[16] = a[7] * b[2];
		c[17] = a[7] * b[3] + a[8] * b[2];
		c[18] = a[8] * b[3] + a[9] * b[2];
		c[19] = a[9] * b[3];
	}

	FORCE_INLINE void EssentialMatrixFivePointNisterSolver::o2p(const double a[10], const double b[4], double c[20]) const
	{
		c[0] += a[0] * b[0];
		c[1] += a[4] * b[1];
		c[2] += a[0] * b[1] + a[1] * b[0];
		c[3] += a[1] * b[1] + a[4] * b[0];
		c[4] += a[0] * b[2] + a[2] * b[0];
		c[5] += a[0] * b[3] + a[3] * b[0];
		c[6] += a[4] * b[2] + a[5] * b[1];
		c[7] += a[4] * b[3] + a[6] * b[1];
		c[8] += a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
		c[9] += a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
		c[10] += a[2] * b[2] + a[7] * b[0];
		c[11] += a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
		c[12] += a[3] * b[3] + a[9] * b[0];
		c[13] += a[5] * b[2] + a[7] * b[1];
		c[14] += a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
		c[15] += a[6] * b[3] + a[9] * b[1];
		c[16] += a[7] * b[2];
		c[17] += a[7] * b[3] + a[8] * b[2];
		c[18] += a[8] * b[3] + a[9] * b[2];
		c[19] += a[9] * b[3];
	}

	FORCE_INLINE void EssentialMatrixFivePointNisterSolver::computeTraceConstraints(
		const Eigen::Matrix<double, 4, 9> &N,
		Eigen::Matrix<double, 10, 20> &coeffs) const
	{
		double const *N_ptr = N.data();
		#define EE(i, j) N_ptr + 4 * (3 * j + i)

		double d[60];

		// Determinant constraint
		Eigen::Matrix<double, 1, 20> row;
		row.setZero();
		double *c_data = row.data();

		o1(EE(0, 1), EE(1, 2), d);
		o1m(EE(0, 2), EE(1, 1), d);
		o2(d, EE(2, 0), c_data);

		o1(EE(0, 2), EE(1, 0), d);
		o1m(EE(0, 0), EE(1, 2), d);
		o2p(d, EE(2, 1), c_data);

		o1(EE(0, 0), EE(1, 1), d);
		o1m(EE(0, 1), EE(1, 0), d);
		o2p(d, EE(2, 2), c_data);

		coeffs.row(9) = row;

		double *EET[3][3] = {{d, d + 10, d + 20}, {d + 10, d + 40, d + 30}, {d + 20, d + 30, d + 50}};

		// Compute EE^T (equation 20)
		for (int i = 0; i < 3; ++i) {
			for (int j = i; j < 3; ++j) {
				o1(EE(i, 0), EE(j, 0), EET[i][j]);
				o1p(EE(i, 1), EE(j, 1), EET[i][j]);
				o1p(EE(i, 2), EE(j, 2), EET[i][j]);
			}
		}

		// Subtract trace (equation 22)
		for (int i = 0; i < 10; ++i) {
			double t = 0.5 * (EET[0][0][i] + EET[1][1][i] + EET[2][2][i]);
			EET[0][0][i] -= t;
			EET[1][1][i] -= t;
			EET[2][2][i] -= t;
		}

		int cnt = 0;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				row.setZero();
				c_data = row.data();
				o2 (EET[i][0], EE(0, j), c_data);
				o2p(EET[i][1], EE(1, j), c_data);
				o2p(EET[i][2], EE(2, j), c_data);
				coeffs.row(cnt++) = row;
			}
		}

		#undef EE
	}

	FORCE_INLINE bool EssentialMatrixFivePointNisterSolver::estimateMinimalModel(
		const DataMatrix& kData_,
		const size_t *kSample_,
		const size_t kSampleNumber_,
		std::vector<models::Model> &models_,
		const double *kWeights_) const
	{
		// Row-major for contiguous writes during assembly
		using RowMat5x9 = Eigen::Matrix<double, 5, 9, Eigen::RowMajor>;
		using RowMatX9  = Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>;

		RowMat5x9 coefficients_min; coefficients_min.setZero();
		RowMatX9  coefficients_dyn;
		if (kSampleNumber_ != 5) {
			coefficients_dyn.resize(static_cast<int>(kSampleNumber_), 9);
			coefficients_dyn.setZero();
		}

		// Separate weighted/unweighted loops to avoid inner-branch
		auto fill_row = [](auto& M, int row, double x0, double y0, double x1, double y1) {
			M.row(row) << x1 * x0, x1 * y0, x1,
			               y1 * x0, y1 * y0, y1,
			               x0,      y0,      1.0;
		};
		auto fill_row_w = [](auto& M, int row, double x0, double y0, double x1, double y1, double w) {
			const double wx0 = w * x0, wy0 = w * y0, wx1 = w * x1, wy1 = w * y1;
			M.row(row) << wx1 * x0, wx1 * y0, wx1,
			               wy1 * x0, wy1 * y0, wy1,
			               wx0,       wy0,      w;
		};

		int rowIdx = 0;
		if (kWeights_ == nullptr) {
			for (size_t i = 0; i < kSampleNumber_; ++i) {
				const size_t idx = (kSample_ ? kSample_[i] : i);
				const double x0 = kData_(idx, 0), y0 = kData_(idx, 1),
				             x1 = kData_(idx, 2), y1 = kData_(idx, 3);
				if (kSampleNumber_ == 5) fill_row(coefficients_min, rowIdx, x0, y0, x1, y1);
				else                      fill_row(coefficients_dyn, rowIdx, x0, y0, x1, y1);
				++rowIdx;
			}
		} else {
			for (size_t i = 0; i < kSampleNumber_; ++i) {
				const size_t idx = (kSample_ ? kSample_[i] : i);
				const double x0 = kData_(idx, 0), y0 = kData_(idx, 1),
				             x1 = kData_(idx, 2), y1 = kData_(idx, 3);
				const double w  = kWeights_[idx];
				if (kSampleNumber_ == 5) fill_row_w(coefficients_min, rowIdx, x0, y0, x1, y1, w);
				else                      fill_row_w(coefficients_dyn, rowIdx, x0, y0, x1, y1, w);
				++rowIdx;
			}
		}

		// Null-space extraction.
		Eigen::Matrix<double, 4, 9> nullSpace;
		if (kSampleNumber_ == 5) {
			const Eigen::FullPivLU<RowMat5x9> lu(coefficients_min);
			if (lu.dimensionOfKernel() != 4) return false;
			nullSpace = lu.kernel().transpose();
		} else {
			const Eigen::JacobiSVD<RowMatX9> svd(coefficients_dyn, Eigen::ComputeFullV);
			nullSpace = svd.matrixV().rightCols<4>().transpose();
		}

		// Trace constraints + determinant elimination.
		// NOTE: column-major to match computeTraceConstraints signature and internal pointer math.
		Eigen::Matrix<double, 10, 20> coeffs; // column-major
		coeffs.setZero();
		computeTraceConstraints(nullSpace, coeffs);

		// Solve the linear 10×10 without explicit inverse (more stable than partialPivLu for poor scaling).
		const Eigen::Matrix<double, 10, 10> L = coeffs.block<10, 10>(0, 0);
		const Eigen::Matrix<double, 10, 10> R = coeffs.block<10, 10>(0, 10);
		coeffs.block<10, 10>(0, 10).noalias() = L.colPivHouseholderQr().solve(R);

		// Perform eliminations using the 6 bottom rows to build A
		Eigen::Matrix<double, 3, 13> A;
		A.setZero();
		for (int i = 0; i < 3; ++i) {
			A.block<1, 3>(i, 1)  = coeffs.block<1, 3>(4 + 2 * i, 10);
			A.block<1, 3>(i, 0) -= coeffs.block<1, 3>(5 + 2 * i, 10);

			A.block<1, 3>(i, 5)  = coeffs.block<1, 3>(4 + 2 * i, 13);
			A.block<1, 3>(i, 4) -= coeffs.block<1, 3>(5 + 2 * i, 13);

			A.block<1, 4>(i, 9)  = coeffs.block<1, 4>(4 + 2 * i, 16);
			A.block<1, 4>(i, 8) -= coeffs.block<1, 4>(5 + 2 * i, 16);
		}

		// Degree-10 polynomial coefficients (equation 14).
		const double
			&a_0_0 = A(0, 0),  &a_0_1  = A(0, 1),  &a_0_2  = A(0, 2),  &a_0_3  = A(0, 3),
			&a_0_4 = A(0, 4),  &a_0_5  = A(0, 5),  &a_0_6  = A(0, 6),  &a_0_7  = A(0, 7),
			&a_0_8 = A(0, 8),  &a_0_9  = A(0, 9),  &a_0_10 = A(0,10),  &a_0_11 = A(0,11),
			&a_0_12 = A(0,12),
			&a_1_0 = A(1, 0),  &a_1_1  = A(1, 1),  &a_1_2  = A(1, 2),  &a_1_3  = A(1, 3),
			&a_1_4 = A(1, 4),  &a_1_5  = A(1, 5),  &a_1_6  = A(1, 6),  &a_1_7  = A(1, 7),
			&a_1_8 = A(1, 8),  &a_1_9  = A(1, 9),  &a_1_10 = A(1,10),  &a_1_11 = A(1,11),
			&a_1_12 = A(1,12),
			&a_2_0 = A(2, 0),  &a_2_1  = A(2, 1),  &a_2_2  = A(2, 2),  &a_2_3  = A(2, 3),
			&a_2_4 = A(2, 4),  &a_2_5  = A(2, 5),  &a_2_6  = A(2, 6),  &a_2_7  = A(2, 7),
			&a_2_8 = A(2, 8),  &a_2_9  = A(2, 9),  &a_2_10 = A(2,10),  &a_2_11 = A(2,11),
			&a_2_12 = A(2,12);

		double c[11];
		c[0] = a_0_12 * a_1_3 * a_2_7 - a_0_12 * a_1_7 * a_2_3 - a_0_3 * a_2_7 * a_1_12 +
		       a_0_7 * a_2_3 * a_1_12 + a_0_3 * a_1_7 * a_2_12 - a_0_7 * a_1_3 * a_2_12;
		c[1] = a_0_11 * a_1_3 * a_2_7 - a_0_11 * a_1_7 * a_2_3 + a_0_12 * a_1_2 * a_2_7 +
		       a_0_12 * a_1_3 * a_2_6 - a_0_12 * a_1_6 * a_2_3 - a_0_12 * a_1_7 * a_2_2 -
		       a_0_2 * a_2_7 * a_1_12 - a_0_3 * a_2_6 * a_1_12 - a_0_3 * a_2_7 * a_1_11 +
		       a_0_6 * a_2_3 * a_1_12 + a_0_7 * a_2_2 * a_1_12 + a_0_7 * a_2_3 * a_1_11 +
		       a_0_2 * a_1_7 * a_2_12 + a_0_3 * a_1_6 * a_2_12 + a_0_3 * a_1_7 * a_2_11 -
		       a_0_6 * a_1_3 * a_2_12 - a_0_7 * a_1_2 * a_2_12 - a_0_7 * a_1_3 * a_2_11;
		c[2] = a_0_10 * a_1_3 * a_2_7 - a_0_10 * a_1_7 * a_2_3 + a_0_11 * a_1_2 * a_2_7 +
		       a_0_11 * a_1_3 * a_2_6 - a_0_11 * a_1_6 * a_2_3 - a_0_11 * a_1_7 * a_2_2 +
		       a_1_1 * a_0_12 * a_2_7 + a_0_12 * a_1_2 * a_2_6 + a_0_12 * a_1_3 * a_2_5 -
		       a_0_12 * a_1_5 * a_2_3 - a_0_12 * a_1_6 * a_2_2 - a_0_12 * a_1_7 * a_2_1 -
		       a_0_1 * a_2_7 * a_1_12 - a_0_2 * a_2_6 * a_1_12 - a_0_2 * a_2_7 * a_1_11 -
		       a_0_3 * a_2_5 * a_1_12 - a_0_3 * a_2_6 * a_1_11 - a_0_3 * a_2_7 * a_1_10 +
		       a_0_5 * a_2_3 * a_1_12 + a_0_6 * a_2_2 * a_1_12 + a_0_6 * a_2_3 * a_1_11 +
		       a_0_7 * a_2_1 * a_1_12 + a_0_7 * a_2_2 * a_1_11 + a_0_7 * a_2_3 * a_1_10 +
		       a_0_1 * a_1_7 * a_2_12 + a_0_2 * a_1_6 * a_2_12 + a_0_2 * a_1_7 * a_2_11 +
		       a_0_3 * a_1_5 * a_2_12 + a_0_3 * a_1_6 * a_2_11 + a_0_3 * a_1_7 * a_2_10 -
		       a_0_5 * a_1_3 * a_2_12 - a_0_6 * a_1_2 * a_2_12 - a_0_6 * a_1_3 * a_2_11 -
		       a_0_7 * a_1_1 * a_2_12 - a_0_7 * a_1_2 * a_2_11 - a_0_7 * a_1_3 * a_2_10;
		c[3] = a_0_3 * a_1_7 * a_2_9 - a_0_3 * a_1_9 * a_2_7 - a_0_7 * a_1_3 * a_2_9 +
		       a_0_7 * a_1_9 * a_2_3 + a_0_9 * a_1_3 * a_2_7 - a_0_9 * a_1_7 * a_2_3 +
		       a_0_10 * a_1_2 * a_2_7 + a_0_10 * a_1_3 * a_2_6 - a_0_10 * a_1_6 * a_2_3 -
		       a_0_10 * a_1_7 * a_2_2 + a_1_0 * a_0_12 * a_2_7 + a_0_11 * a_1_1 * a_2_7 +
		       a_0_11 * a_1_2 * a_2_6 + a_0_11 * a_1_3 * a_2_5 - a_0_11 * a_1_5 * a_2_3 -
		       a_0_11 * a_1_6 * a_2_2 - a_0_11 * a_1_7 * a_2_1 + a_1_1 * a_0_12 * a_2_6 +
		       a_0_12 * a_1_2 * a_2_5 + a_0_12 * a_1_3 * a_2_4 - a_0_12 * a_1_4 * a_2_3 -
		       a_0_12 * a_1_5 * a_2_2 - a_0_12 * a_1_6 * a_2_1 - a_0_12 * a_1_7 * a_2_0 -
		       a_0_0 * a_2_7 * a_1_12 - a_0_1 * a_2_6 * a_1_12 - a_0_1 * a_2_7 * a_1_11 -
		       a_0_2 * a_2_5 * a_1_12 - a_0_2 * a_2_6 * a_1_11 - a_0_2 * a_2_7 * a_1_10 -
		       a_0_3 * a_2_4 * a_1_12 - a_0_3 * a_2_5 * a_1_11 - a_0_3 * a_2_6 * a_1_10 +
		       a_0_4 * a_2_3 * a_1_12 + a_0_5 * a_2_2 * a_1_12 + a_0_5 * a_2_3 * a_1_11 +
		       a_0_6 * a_2_1 * a_1_12 + a_0_6 * a_2_2 * a_1_11 + a_0_6 * a_2_3 * a_1_10 +
		       a_0_7 * a_2_0 * a_1_12 + a_0_7 * a_2_1 * a_1_11 + a_0_7 * a_2_2 * a_1_10 +
		       a_0_0 * a_1_7 * a_2_12 + a_0_1 * a_1_6 * a_2_12 + a_0_1 * a_1_7 * a_2_11 +
		       a_0_2 * a_1_5 * a_2_12 + a_0_2 * a_1_6 * a_2_11 + a_0_2 * a_1_7 * a_2_10 +
		       a_0_3 * a_1_4 * a_2_12 + a_0_3 * a_1_5 * a_2_11 + a_0_3 * a_1_6 * a_2_10 -
		       a_0_4 * a_1_3 * a_2_12 - a_0_5 * a_1_2 * a_2_12 - a_0_5 * a_1_3 * a_2_11 -
		       a_0_6 * a_1_1 * a_2_12 - a_0_6 * a_1_2 * a_2_11 - a_0_6 * a_1_3 * a_2_10 -
		       a_0_7 * a_1_0 * a_2_12 - a_0_7 * a_1_1 * a_2_11 - a_0_7 * a_1_2 * a_2_10;
		c[4] = a_0_2 * a_1_7 * a_2_9 - a_0_2 * a_1_9 * a_2_7 + a_0_3 * a_1_6 * a_2_9 +
		       a_0_3 * a_1_7 * a_2_8 - a_0_3 * a_1_8 * a_2_7 - a_0_3 * a_1_9 * a_2_6 -
		       a_0_6 * a_1_3 * a_2_9 + a_0_6 * a_1_9 * a_2_3 - a_0_7 * a_1_2 * a_2_9 -
		       a_0_7 * a_1_3 * a_2_8 + a_0_7 * a_1_8 * a_2_3 + a_0_7 * a_1_9 * a_2_2 +
		       a_0_8 * a_1_3 * a_2_7 - a_0_8 * a_1_7 * a_2_3 + a_0_9 * a_1_2 * a_2_7 +
		       a_0_9 * a_1_3 * a_2_6 - a_0_9 * a_1_6 * a_2_3 - a_0_9 * a_1_7 * a_2_2 +
		       a_0_10 * a_1_1 * a_2_7 + a_0_10 * a_1_2 * a_2_6 + a_0_10 * a_1_3 * a_2_5 -
		       a_0_10 * a_1_5 * a_2_3 - a_0_10 * a_1_6 * a_2_2 - a_0_10 * a_1_7 * a_2_1 +
		       a_1_0 * a_0_11 * a_2_7 + a_1_0 * a_0_12 * a_2_6 + a_0_11 * a_1_1 * a_2_6 +
		       a_0_11 * a_1_2 * a_2_5 + a_0_11 * a_1_3 * a_2_4 - a_0_11 * a_1_4 * a_2_3 -
		       a_0_11 * a_1_5 * a_2_2 - a_0_11 * a_1_6 * a_2_1 - a_0_11 * a_1_7 * a_2_0 +
		       a_1_1 * a_0_12 * a_2_5 + a_0_12 * a_1_2 * a_2_4 - a_0_12 * a_1_4 * a_2_2 -
		       a_0_12 * a_1_5 * a_2_1 - a_0_12 * a_1_6 * a_2_0 - a_0_0 * a_2_6 * a_1_12 -
		       a_0_0 * a_2_7 * a_1_11 - a_0_1 * a_2_5 * a_1_12 - a_0_1 * a_2_6 * a_1_11 -
		       a_0_1 * a_2_7 * a_1_10 - a_0_2 * a_2_4 * a_1_12 - a_0_2 * a_2_5 * a_1_11 -
		       a_0_2 * a_2_6 * a_1_10 - a_0_3 * a_2_4 * a_1_11 - a_0_3 * a_2_5 * a_1_10 +
		       a_0_4 * a_2_2 * a_1_12 + a_0_4 * a_2_3 * a_1_11 + a_0_5 * a_2_1 * a_1_12 +
		       a_0_5 * a_2_2 * a_1_11 + a_0_5 * a_2_3 * a_1_10 + a_0_6 * a_2_0 * a_1_12 +
		       a_0_6 * a_2_1 * a_1_11 + a_0_6 * a_2_2 * a_1_10 + a_0_7 * a_2_0 * a_1_11 +
		       a_0_7 * a_2_1 * a_1_10 + a_0_0 * a_1_6 * a_2_12 + a_0_0 * a_1_7 * a_2_11 +
		       a_0_1 * a_1_5 * a_2_12 + a_0_1 * a_1_6 * a_2_11 + a_0_1 * a_1_7 * a_2_10 +
		       a_0_2 * a_1_4 * a_2_12 + a_0_2 * a_1_5 * a_2_11 + a_0_2 * a_1_6 * a_2_10 +
		       a_0_3 * a_1_4 * a_2_11 + a_0_3 * a_1_5 * a_2_10 - a_0_4 * a_1_2 * a_2_12 -
		       a_0_4 * a_1_3 * a_2_11 - a_0_5 * a_1_1 * a_2_12 - a_0_5 * a_1_2 * a_2_11 -
		       a_0_5 * a_1_3 * a_2_10 - a_0_6 * a_1_0 * a_2_12 - a_0_6 * a_1_1 * a_2_11 -
		       a_0_6 * a_1_2 * a_2_10 - a_0_7 * a_1_0 * a_2_11 - a_0_7 * a_1_1 * a_2_10;
		c[5] = a_0_1 * a_1_7 * a_2_9 - a_0_1 * a_1_9 * a_2_7 + a_0_2 * a_1_6 * a_2_9 +
		       a_0_2 * a_1_7 * a_2_8 - a_0_2 * a_1_8 * a_2_7 - a_0_2 * a_1_9 * a_2_6 +
		       a_0_3 * a_1_5 * a_2_9 + a_0_3 * a_1_6 * a_2_8 - a_0_3 * a_1_8 * a_2_6 -
		       a_0_3 * a_1_9 * a_2_5 - a_0_5 * a_1_3 * a_2_9 + a_0_5 * a_1_9 * a_2_3 -
		       a_0_6 * a_1_2 * a_2_9 - a_0_6 * a_1_3 * a_2_8 + a_0_6 * a_1_8 * a_2_3 +
		       a_0_6 * a_1_9 * a_2_2 - a_0_7 * a_1_1 * a_2_9 - a_0_7 * a_1_2 * a_2_8 +
		       a_0_7 * a_1_8 * a_2_2 + a_0_7 * a_1_9 * a_2_1 + a_0_8 * a_1_2 * a_2_7 +
		       a_0_8 * a_1_3 * a_2_6 - a_0_8 * a_1_6 * a_2_3 - a_0_8 * a_1_7 * a_2_2 +
		       a_0_9 * a_1_1 * a_2_7 + a_0_9 * a_1_2 * a_2_6 + a_0_9 * a_1_3 * a_2_5 -
		       a_0_9 * a_1_5 * a_2_3 - a_0_9 * a_1_6 * a_2_2 - a_0_9 * a_1_7 * a_2_1 +
		       a_0_10 * a_1_0 * a_2_7 + a_0_10 * a_1_1 * a_2_6 + a_0_10 * a_1_2 * a_2_5 +
		       a_0_10 * a_1_3 * a_2_4 - a_0_10 * a_1_4 * a_2_3 - a_0_10 * a_1_5 * a_2_2 -
		       a_0_10 * a_1_6 * a_2_1 - a_0_10 * a_1_7 * a_2_0 + a_1_0 * a_0_11 * a_2_6 +
		       a_1_0 * a_0_12 * a_2_5 + a_0_11 * a_1_1 * a_2_5 + a_0_11 * a_1_2 * a_2_4 -
		       a_0_11 * a_1_4 * a_2_2 - a_0_11 * a_1_5 * a_2_1 - a_0_11 * a_1_6 * a_2_0 +
		       a_1_1 * a_0_12 * a_2_4 - a_0_12 * a_1_4 * a_2_1 - a_0_12 * a_1_5 * a_2_0 -
		       a_0_0 * a_2_5 * a_1_12 - a_0_0 * a_2_6 * a_1_11 - a_0_0 * a_2_7 * a_1_10 -
		       a_0_1 * a_2_4 * a_1_12 - a_0_1 * a_2_5 * a_1_11 - a_0_1 * a_2_6 * a_1_10 -
		       a_0_2 * a_2_4 * a_1_11 - a_0_2 * a_2_5 * a_1_10 - a_0_3 * a_2_4 * a_1_10 +
		       a_0_4 * a_2_1 * a_1_12 + a_0_4 * a_2_2 * a_1_11 + a_0_4 * a_2_3 * a_1_10 +
		       a_0_5 * a_2_0 * a_1_12 + a_0_5 * a_2_1 * a_1_11 + a_0_5 * a_2_2 * a_1_10 +
		       a_0_6 * a_2_0 * a_1_11 + a_0_6 * a_2_1 * a_1_10 + a_0_7 * a_2_0 * a_1_10 +
		       a_0_0 * a_1_5 * a_2_12 + a_0_0 * a_1_6 * a_2_11 + a_0_0 * a_1_7 * a_2_10 +
		       a_0_1 * a_1_4 * a_2_12 + a_0_1 * a_1_5 * a_2_11 + a_0_1 * a_1_6 * a_2_10 +
		       a_0_2 * a_1_4 * a_2_11 + a_0_2 * a_1_5 * a_2_10 + a_0_3 * a_1_4 * a_2_10 -
		       a_0_4 * a_1_1 * a_2_12 - a_0_4 * a_1_2 * a_2_11 - a_0_4 * a_1_3 * a_2_10 -
		       a_0_5 * a_1_0 * a_2_12 - a_0_5 * a_1_1 * a_2_11 - a_0_5 * a_1_2 * a_2_10 -
		       a_0_6 * a_1_0 * a_2_11 - a_0_6 * a_1_1 * a_2_10 - a_0_7 * a_1_0 * a_2_10;
		c[6] = a_0_0 * a_1_7 * a_2_9 - a_0_0 * a_1_9 * a_2_7 + a_0_1 * a_1_6 * a_2_9 +
		       a_0_1 * a_1_7 * a_2_8 - a_0_1 * a_1_8 * a_2_7 - a_0_1 * a_1_9 * a_2_6 +
		       a_0_2 * a_1_5 * a_2_9 + a_0_2 * a_1_6 * a_2_8 - a_0_2 * a_1_8 * a_2_6 -
		       a_0_2 * a_1_9 * a_2_5 + a_0_3 * a_1_4 * a_2_9 + a_0_3 * a_1_5 * a_2_8 -
		       a_0_3 * a_1_8 * a_2_5 - a_0_3 * a_1_9 * a_2_4 - a_0_4 * a_1_3 * a_2_9 +
		       a_0_4 * a_1_9 * a_2_3 - a_0_5 * a_1_2 * a_2_9 - a_0_5 * a_1_3 * a_2_8 +
		       a_0_5 * a_1_8 * a_2_3 + a_0_5 * a_1_9 * a_2_2 - a_0_6 * a_1_1 * a_2_9 -
		       a_0_6 * a_1_2 * a_2_8 + a_0_6 * a_1_8 * a_2_2 + a_0_6 * a_1_9 * a_2_1 -
		       a_0_7 * a_1_0 * a_2_9 - a_0_7 * a_1_1 * a_2_8 + a_0_7 * a_1_8 * a_2_1 +
		       a_0_7 * a_1_9 * a_2_0 + a_0_8 * a_1_1 * a_2_7 + a_0_8 * a_1_2 * a_2_6 +
		       a_0_8 * a_1_3 * a_2_5 - a_0_8 * a_1_5 * a_2_3 - a_0_8 * a_1_6 * a_2_2 -
		       a_0_8 * a_1_7 * a_2_1 + a_0_9 * a_1_0 * a_2_7 + a_0_9 * a_1_1 * a_2_6 +
		       a_0_9 * a_1_2 * a_2_5 + a_0_9 * a_1_3 * a_2_4 - a_0_9 * a_1_4 * a_2_3 -
		       a_0_9 * a_1_5 * a_2_2 - a_0_9 * a_1_6 * a_2_1 - a_0_9 * a_1_7 * a_2_0 +
		       a_0_10 * a_1_0 * a_2_6 + a_0_10 * a_1_1 * a_2_5 + a_0_10 * a_1_2 * a_2_4 -
		       a_0_10 * a_1_4 * a_2_2 - a_0_10 * a_1_5 * a_2_1 - a_0_10 * a_1_6 * a_2_0 +
		       a_1_0 * a_0_11 * a_2_5 + a_1_0 * a_0_12 * a_2_4 + a_0_11 * a_1_1 * a_2_4 -
		       a_0_11 * a_1_4 * a_2_1 - a_0_11 * a_1_5 * a_2_0 - a_0_12 * a_1_4 * a_2_0 -
		       a_0_0 * a_2_4 * a_1_12 - a_0_0 * a_2_5 * a_1_11 - a_0_0 * a_2_6 * a_1_10 -
		       a_0_1 * a_2_4 * a_1_11 - a_0_1 * a_2_5 * a_1_10 - a_0_2 * a_2_4 * a_1_10 +
		       a_0_4 * a_2_0 * a_1_12 + a_0_4 * a_2_1 * a_1_11 + a_0_4 * a_2_2 * a_1_10 +
		       a_0_5 * a_2_0 * a_1_11 + a_0_5 * a_2_1 * a_1_10 + a_0_6 * a_2_0 * a_1_10 +
		       a_0_0 * a_1_4 * a_2_12 + a_0_0 * a_1_5 * a_2_11 + a_0_0 * a_1_6 * a_2_10 +
		       a_0_1 * a_1_4 * a_2_11 + a_0_1 * a_1_5 * a_2_10 + a_0_2 * a_1_4 * a_2_10 -
		       a_0_4 * a_1_0 * a_2_12 - a_0_4 * a_1_1 * a_2_11 - a_0_4 * a_1_2 * a_2_10 -
		       a_0_5 * a_1_0 * a_2_11 - a_0_5 * a_1_1 * a_2_10 - a_0_6 * a_1_0 * a_2_10;
		c[7] = a_0_0 * a_1_6 * a_2_9 + a_0_0 * a_1_7 * a_2_8 - a_0_0 * a_1_8 * a_2_7 -
		       a_0_0 * a_1_9 * a_2_6 + a_0_1 * a_1_5 * a_2_9 + a_0_1 * a_1_6 * a_2_8 -
		       a_0_1 * a_1_8 * a_2_6 - a_0_1 * a_1_9 * a_2_5 + a_0_2 * a_1_4 * a_2_9 +
		       a_0_2 * a_1_5 * a_2_8 - a_0_2 * a_1_8 * a_2_5 - a_0_2 * a_1_9 * a_2_4 +
		       a_0_3 * a_1_4 * a_2_8 - a_0_3 * a_1_8 * a_2_4 - a_0_4 * a_1_2 * a_2_9 -
		       a_0_4 * a_1_3 * a_2_8 + a_0_4 * a_1_8 * a_2_3 + a_0_4 * a_1_9 * a_2_2 -
		       a_0_5 * a_1_1 * a_2_9 - a_0_5 * a_1_2 * a_2_8 + a_0_5 * a_1_8 * a_2_2 -
		       a_0_5 * a_1_9 * a_2_1 - a_0_6 * a_1_0 * a_2_9 - a_0_6 * a_1_1 * a_2_8 +
		       a_0_6 * a_1_8 * a_2_1 + a_0_6 * a_1_9 * a_2_0 - a_0_7 * a_1_0 * a_2_8 +
		       a_0_7 * a_1_8 * a_2_0 + a_0_8 * a_1_0 * a_2_7 + a_0_8 * a_1_1 * a_2_6 +
		       a_0_8 * a_1_2 * a_2_5 + a_0_8 * a_1_3 * a_2_4 - a_0_8 * a_1_4 * a_2_3 -
		       a_0_8 * a_1_5 * a_2_2 - a_0_8 * a_1_6 * a_2_1 - a_0_8 * a_1_7 * a_2_0 +
		       a_0_9 * a_1_0 * a_2_6 + a_0_9 * a_1_1 * a_2_5 + a_0_9 * a_1_2 * a_2_4 -
		       a_0_9 * a_1_4 * a_2_2 - a_0_9 * a_1_5 * a_2_1 - a_0_9 * a_1_6 * a_2_0 +
		       a_0_10 * a_1_0 * a_2_5 + a_0_10 * a_1_1 * a_2_4 - a_0_10 * a_1_4 * a_2_1 -
		       a_0_10 * a_1_5 * a_2_0 + a_1_0 * a_0_11 * a_2_4 - a_0_11 * a_1_4 * a_2_0 -
		       a_0_0 * a_2_4 * a_1_11 - a_0_0 * a_2_5 * a_1_10 - a_0_1 * a_2_4 * a_1_10 +
		       a_0_4 * a_2_0 * a_1_11 + a_0_4 * a_2_1 * a_1_10 + a_0_5 * a_2_0 * a_1_10 +
		       a_0_0 * a_1_4 * a_2_11 + a_0_0 * a_1_5 * a_2_10 + a_0_1 * a_1_4 * a_2_10 -
		       a_0_4 * a_1_0 * a_2_11 - a_0_4 * a_1_1 * a_2_10 - a_0_5 * a_1_0 * a_2_10;
		c[8] = a_0_0 * a_1_5 * a_2_9 + a_0_0 * a_1_6 * a_2_8 - a_0_0 * a_1_8 * a_2_6 -
		       a_0_0 * a_1_9 * a_2_5 + a_0_1 * a_1_4 * a_2_9 + a_0_1 * a_1_5 * a_2_8 -
		       a_0_1 * a_1_8 * a_2_5 - a_0_1 * a_1_9 * a_2_4 + a_0_2 * a_1_4 * a_2_8 -
		       a_0_2 * a_1_8 * a_2_4 - a_0_4 * a_1_1 * a_2_9 - a_0_4 * a_1_2 * a_2_8 +
		       a_0_4 * a_1_8 * a_2_2 + a_0_4 * a_1_9 * a_2_1 - a_0_5 * a_1_0 * a_2_9 -
		       a_0_5 * a_1_1 * a_2_8 + a_0_5 * a_1_8 * a_2_1 + a_0_5 * a_1_9 * a_2_0 -
		       a_0_6 * a_1_0 * a_2_8 + a_0_6 * a_1_8 * a_2_0 + a_0_8 * a_1_0 * a_2_6 +
		       a_0_8 * a_1_1 * a_2_5 + a_0_8 * a_1_2 * a_2_4 - a_0_8 * a_1_4 * a_2_2 -
		       a_0_8 * a_1_5 * a_2_1 - a_0_8 * a_1_6 * a_2_0 + a_0_9 * a_1_0 * a_2_5 +
		       a_0_9 * a_1_1 * a_2_4 - a_0_9 * a_1_4 * a_2_1 - a_0_9 * a_1_5 * a_2_0 +
		       a_0_10 * a_1_0 * a_2_4 - a_0_10 * a_1_4 * a_2_0 - a_0_0 * a_2_4 * a_1_10 +
		       a_0_4 * a_2_0 * a_1_10 + a_0_0 * a_1_4 * a_2_10 - a_0_4 * a_1_0 * a_2_10;
		c[9]  = a_0_0 * a_1_4 * a_2_9 + a_0_0 * a_1_5 * a_2_8 - a_0_0 * a_1_8 * a_2_5 -
		        a_0_0 * a_1_9 * a_2_4 + a_0_1 * a_1_4 * a_2_8 - a_0_1 * a_1_8 * a_2_4 -
		        a_0_4 * a_1_0 * a_2_9 - a_0_4 * a_1_1 * a_2_8 + a_0_4 * a_1_8 * a_2_1 -
		        a_0_4 * a_1_9 * a_2_0 - a_0_5 * a_1_0 * a_2_8 + a_0_5 * a_1_8 * a_2_0 +
		        a_0_8 * a_1_0 * a_2_5 + a_0_8 * a_1_1 * a_2_4 - a_0_8 * a_1_4 * a_2_1 -
		        a_0_8 * a_1_5 * a_2_0 + a_0_9 * a_1_0 * a_2_4 - a_0_9 * a_1_4 * a_2_0;
		c[10] = a_0_0 * a_1_4 * a_2_8 - a_0_0 * a_1_8 * a_2_4 - a_0_4 * a_1_0 * a_2_8 +
		        a_0_4 * a_1_8 * a_2_0 + a_0_8 * a_1_0 * a_2_4 - a_0_8 * a_1_4 * a_2_0;

		// Scale polynomial to improve numeric conditioning for Sturm bracketing
		double scale = 0.0;
		for (int i = 0; i <= 10; ++i) scale = std::max(scale, std::abs(c[i]));
		if (scale > 0) for (int i = 0; i <= 10; ++i) c[i] /= scale;

		// Sturm bracketing of roots.
		double roots[10];
		const int n_sols = sturm::bisect_sturm<10>(c, roots);

		// Back substitution to recover essential matrices.
		// Pre-allocate all matrices outside loop to avoid repeated allocations
		Eigen::Matrix<double, 3, 2> B;
		Eigen::Vector3d b;
		Eigen::Vector2d xz;
		Eigen::Matrix3d E;
		Eigen::Map<Eigen::Matrix<double, 1, 9>> e(E.data());

		models_.clear();
		models_.reserve(static_cast<size_t>(n_sols));

		// Pre-extract A columns to avoid repeated block operations
		const auto A_col0 = A.col(0);
		const auto A_col1 = A.col(1);
		const auto A_col2 = A.col(2);
		const auto A_col3 = A.col(3);
		const auto A_col4 = A.col(4);
		const auto A_col5 = A.col(5);
		const auto A_col6 = A.col(6);
		const auto A_col7 = A.col(7);
		const auto A_col8 = A.col(8);
		const auto A_col9 = A.col(9);
		const auto A_col10 = A.col(10);
		const auto A_col11 = A.col(11);
		const auto A_col12 = A.col(12);

		for (int i = 0; i < n_sols; ++i)
		{
			const double z  = roots[i];
			const double z2 = z * z;
			const double z3 = z2 * z;
			const double z4 = z2 * z2;

			// Build B and b using pre-extracted columns (avoids block() overhead)
			B.col(0).noalias() = A_col0 * z3 + A_col1 * z2 + A_col2 * z + A_col3;
			B.col(1).noalias() = A_col4 * z3 + A_col5 * z2 + A_col6 * z + A_col7;
			b.noalias()        = A_col8 * z4 + A_col9 * z3 + A_col10 * z2 + A_col11 * z + A_col12;

			// Direct 2x2 solve instead of QR for speed (2x2 is trivial)
			// For 2x2 system: B2 * xz = b2, solve using Cramer's rule
			const double b00 = B(0,0), b01 = B(0,1);
			const double b10 = B(1,0), b11 = B(1,1);
			const double det = b00 * b11 - b01 * b10;

			if (std::abs(det) > 1e-12) {
				const double inv_det = 1.0 / det;
				xz(0) = (b11 * b(0) - b01 * b(1)) * inv_det;
				xz(1) = (b00 * b(1) - b10 * b(0)) * inv_det;

				// Check residual on 3rd row; use full 3x2 solve if needed
				const double residual = std::abs(B(2,0) * xz(0) + B(2,1) * xz(1) - b(2));
				if (residual > 1e-9) {
					// Fallback to QR for 3x2 overdetermined system
					xz = B.colPivHouseholderQr().solve(b);
				}
			} else {
				// Singular 2x2, use full QR
				xz = B.colPivHouseholderQr().solve(b);
			}

			const double x = -xz(0), y = -xz(1);
			e = nullSpace.row(0) * x + nullSpace.row(1) * y + nullSpace.row(2) * z + nullSpace.row(3);

			// Normalize coefficients (rows of N are orthonormal).
			const double norm_sq = x * x + y * y + z * z + 1.0;
			const double s = std::sqrt(norm_sq);
			if (s > 1e-12) e /= s;

			models::Model model;
			model.getMutableData() = E;
			models_.emplace_back(std::move(model));
		}

		return true;
	}

	FORCE_INLINE bool EssentialMatrixFivePointNisterSolver::estimateModel(
		const DataMatrix& kData_,
		const size_t *kSample_,
		const size_t kSampleNumber_,
		std::vector<models::Model> &models_,
		const double *kWeights_) const
	{
		return estimateMinimalModel(kData_, kSample_, kSampleNumber_, models_, kWeights_);
	}
}
}
}
