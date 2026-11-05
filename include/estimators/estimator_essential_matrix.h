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
#include "numerical_optimizer/essential.h"

namespace superansac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		class EssentialMatrixEstimator : public Estimator
		{
		protected:
			bool enforce3DCheirality = true;
			double invFx1 = 1.0, invFy1 = 1.0, invFx2 = 1.0, invFy2 = 1.0;
			double invFx1Sq = 1.0, invFy1Sq = 1.0, invFx2Sq = 1.0, invFy2Sq = 1.0;
			double cheiralityResidualCutoffSq = 0.01;
			double cheiralityMargin = 0.0;

		public:
			EssentialMatrixEstimator() {}
			~EssentialMatrixEstimator() {}

			void setEnforce3DCheirality(const bool kEnabled_)
			{
				enforce3DCheirality = kEnabled_;
			}

			void setInverseFocalLengths(
				const double kInvFx1_,
				const double kInvFy1_,
				const double kInvFx2_,
				const double kInvFy2_)
			{
				invFx1 = kInvFx1_;
				invFy1 = kInvFy1_;
				invFx2 = kInvFx2_;
				invFy2 = kInvFy2_;
				invFx1Sq = kInvFx1_ * kInvFx1_;
				invFy1Sq = kInvFy1_ * kInvFy1_;
				invFx2Sq = kInvFx2_ * kInvFx2_;
				invFy2Sq = kInvFy2_ * kInvFy2_;
			}

			void setCheiralityResidualCutoffSq(const double kCutoffSq_)
			{
				cheiralityResidualCutoffSq = kCutoffSq_;
			}

			void setCheiralityMargin(const double kMargin_)
			{
				cheiralityMargin = kMargin_;
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

				if (!enforce3DCheirality || kSample_ == nullptr)
				{
					minimalSolver->estimateModel(kData_, kSample_, kSampleSize, *models_);
					return models_->size() > 0;
				}

				static thread_local std::vector<models::Model> rawModels;
				static thread_local std::vector<Eigen::Vector3d> b1, b2;
				static const std::vector<Eigen::Vector3d> kEmptyBearings;
				rawModels.clear();
				minimalSolver->estimateModel(kData_, kSample_, kSampleSize, rawModels);
				if (rawModels.empty())
					return false;

				const size_t kCols = static_cast<size_t>(kData_.cols());
				const double * __restrict__ kBase = kData_.data();
				if (cheiralityMargin <= 0.0)
				{
					b1.resize(kSampleSize);
					b2.resize(kSampleSize);
					for (size_t i = 0; i < kSampleSize; ++i)
					{
						const double * __restrict__ row = kBase + kSample_[i] * kCols;
						b1[i] = Eigen::Vector3d(row[0], row[1], 1.0).normalized();
						b2[i] = Eigen::Vector3d(row[2], row[3], 1.0).normalized();
					}
				}

				static thread_local poselib::CameraPoseVector poses;
				for (const auto &raw : rawModels)
				{
					const Eigen::Matrix3d E = raw.getData().block<3, 3>(0, 0);
					if (!E.allFinite())
						continue;
					poses.clear();
					if (cheiralityMargin <= 0.0)
					{
						poselib::motion_from_essential(E, b1, b2, &poses);
						for (const auto &pose : poses)
						{
							models::Model m;
							m.getMutableData() = E;
							m.setPose(pose.Rt());
							models_->push_back(m);
						}
					}
					else
					{
						poselib::motion_from_essential(E, kEmptyBearings, kEmptyBearings, &poses);
						Eigen::Matrix<double, 3, 4, Eigen::RowMajor> rt;
						for (size_t pIdx = 0; pIdx < poses.size(); ++pIdx)
						{
							const auto &pose = poses[pIdx];
							if ((pIdx & 1u) == 1u && poses[pIdx].q == poses[pIdx - 1].q)
							{
								// motion_from_essential emits pairs (R1, +t), (R1, -t), (R2, +t), (R2, -t);
								// reuse R across paired translations to halve quat_to_rotmat calls.
								rt.col(3) = pose.t;
							}
							else
							{
								rt.block<3, 3>(0, 0) = pose.R();
								rt.col(3) = pose.t;
							}
							bool allCheiral = true;
							for (size_t i = 0; i < kSampleSize; ++i)
							{
								if (!isPointCheiral(kBase + kSample_[i] * kCols, rt))
								{
									allCheiral = false;
									break;
								}
							}
							if (allCheiral)
							{
								models::Model m;
								m.getMutableData() = E;
								m.setPose(rt);
								models_->push_back(m);
							}
						}
					}
				}

				// Return a single unposed placeholder when all minimal roots fail 3D cheirality so
				// SupeRansac::run advances 1 iteration per minimal sample and rejects in isValidModel.
				if (models_->empty())
					models_->push_back(rawModels[0]);
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

				if (!nonMinimalSolver->estimateModel(kData_,
					kSample_,
					kSampleNumber_,
					*models_,
					kWeights_))
					return false;
				return true;
			}

			// Exact division-free and sqrt-free 3D positive-depth check (equivalent to
			// poselib::check_cheirality with min_depth = 0 on unit bearing vectors).
			FORCE_INLINE bool isPointCheiral(
				const double* kPoint_,
				const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>& kRt_) const
			{
				const double x1 = kPoint_[0], y1 = kPoint_[1];
				const double x2 = kPoint_[2], y2 = kPoint_[3];
				const double r1_0 = kRt_(0, 0) * x1 + kRt_(0, 1) * y1 + kRt_(0, 2);
				const double r1_1 = kRt_(1, 0) * x1 + kRt_(1, 1) * y1 + kRt_(1, 2);
				const double r1_2 = kRt_(2, 0) * x1 + kRt_(2, 1) * y1 + kRt_(2, 2);
				const double tx = kRt_(0, 3), ty = kRt_(1, 3), tz = kRt_(2, 3);
				const double c  = -(r1_0 * x2 + r1_1 * y2 + r1_2);
				const double d1 = -(r1_0 * tx + r1_1 * ty + r1_2 * tz);
				const double d2 =   x2 * tx + y2 * ty + tz;
				const double n1 = x1 * x1 + y1 * y1 + 1.0;
				const double n2 = x2 * x2 + y2 * y2 + 1.0;
				const double num1 = d1 * n2 - c * d2;
				const double num2 = d2 * n1 - c * d1;
				if (num1 > 0.0 && num2 > 0.0)
					return true;
				if (cheiralityMargin > 0.0 && c < 0.0)
				{
					const double tol = -cheiralityMargin * n1 * n2;
					return (num1 > tol) && (num2 > tol);
				}
				return false;
			}

			FORCE_INLINE double squaredResidual(const double* point_,
				const models::Model& model_) const override
			{
				const double r2 = squaredSampsonDistance(point_, model_.getData());
				if (enforce3DCheirality && model_.hasPose() && r2 < cheiralityResidualCutoffSq && !isPointCheiral(point_, model_.getPose()))
					return 1e18;
				return r2;
			}

			FORCE_INLINE double squaredResidual(
				const double* kPoint_,
				const ModelMatrix& kDescriptor_) const
			{
				const double kSquaredResidual = squaredSampsonDistance(kPoint_, kDescriptor_);
				return kSquaredResidual;
			}

			// The symmetric epipolar distance between a point correspondence and an essential matrix
			FORCE_INLINE double squaredSymmetricEpipolarDistance(
				const double* kPoint_,
				const ModelMatrix& kDescriptor_) const
			{
				// Use const references to avoid copying
				const double &x1 = kPoint_[0],
							&y1 = kPoint_[1],
							&x2 = kPoint_[2],
							&y2 = kPoint_[3];

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
				const double a = invFx1Sq * rxc * rxc + invFy1Sq * ryc * ryc;
				const double b = invFx2Sq * rx * rx + invFy2Sq * ry * ry;

				return r * r * (a + b) / (a * b);
			}

			// The sampson distance between a point_ correspondence and an essential matrix
			FORCE_INLINE double squaredSampsonDistance(
				const double* kPoint_,
				const ModelMatrix& kDescriptor_) const
			{
				const double E0_0 = kDescriptor_(0, 0), E0_1 = kDescriptor_(0, 1), E0_2 = kDescriptor_(0, 2);
				const double E1_0 = kDescriptor_(1, 0), E1_1 = kDescriptor_(1, 1), E1_2 = kDescriptor_(1, 2);
				const double E2_0 = kDescriptor_(2, 0), E2_1 = kDescriptor_(2, 1), E2_2 = kDescriptor_(2, 2);

				const double x1_0 = kPoint_[0], x1_1 = kPoint_[1];
				const double x2_0 = kPoint_[2], x2_1 = kPoint_[3];

				const double Ex1_0 = E0_0 * x1_0 + E0_1 * x1_1 + E0_2;
				const double Ex1_1 = E1_0 * x1_0 + E1_1 * x1_1 + E1_2;
				const double Ex1_2 = E2_0 * x1_0 + E2_1 * x1_1 + E2_2;

				const double Ex2_0 = E0_0 * x2_0 + E1_0 * x2_1 + E2_0;
				const double Ex2_1 = E0_1 * x2_0 + E1_1 * x2_1 + E2_1;

				const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;
				const double Cx = invFx2Sq * Ex1_0 * Ex1_0 + invFy2Sq * Ex1_1 * Ex1_1;
				const double Cy = invFx1Sq * Ex2_0 * Ex2_0 + invFy1Sq * Ex2_1 * Ex2_1;

				// Use reciprocal multiplication (faster than division with -ffast-math)
				const double denom_sum = Cx + Cy;
				const double r2 = (C * C) / denom_sum;

				return r2;
			}

			FORCE_INLINE double residual(const double* point_,
				const models::Model& model_) const override
			{
				return sqrt(squaredResidual(point_, model_));
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
				double* __restrict__ kOut_) const override
			{
				const ModelMatrix& kDescriptor = kModel_.getData();
				const double e00 = kDescriptor(0, 0), e01 = kDescriptor(0, 1), e02 = kDescriptor(0, 2);
				const double e10 = kDescriptor(1, 0), e11 = kDescriptor(1, 1), e12 = kDescriptor(1, 2);
				const double e20 = kDescriptor(2, 0), e21 = kDescriptor(2, 1), e22 = kDescriptor(2, 2);
				const double ifx1 = invFx1Sq, ify1 = invFy1Sq, ifx2 = invFx2Sq, ify2 = invFy2Sq;
				const size_t kCols = static_cast<size_t>(kData_.cols());
				const double* __restrict__ kStartPtr = kData_.data() + kStartRow_ * kCols;
				const double* __restrict__ row = kStartPtr;
				if (kCols == 4)
				{
					for (size_t i = 0; i < kCount_; ++i, row += 4)
					{
						const double x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
						const double ex1_0 = e00 * x1 + e01 * y1 + e02;
						const double ex1_1 = e10 * x1 + e11 * y1 + e12;
						const double ex1_2 = e20 * x1 + e21 * y1 + e22;
						const double ex2_0 = e00 * x2 + e10 * y2 + e20;
						const double ex2_1 = e01 * x2 + e11 * y2 + e21;
						const double c = x2 * ex1_0 + y2 * ex1_1 + ex1_2;
						const double cx = ifx2 * ex1_0 * ex1_0 + ify2 * ex1_1 * ex1_1;
						const double cy = ifx1 * ex2_0 * ex2_0 + ify1 * ex2_1 * ex2_1;
						kOut_[i] = (c * c) / (cx + cy);
					}
				}
				else
				{
					for (size_t i = 0; i < kCount_; ++i, row += kCols)
					{
						const double x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
						const double ex1_0 = e00 * x1 + e01 * y1 + e02;
						const double ex1_1 = e10 * x1 + e11 * y1 + e12;
						const double ex1_2 = e20 * x1 + e21 * y1 + e22;
						const double ex2_0 = e00 * x2 + e10 * y2 + e20;
						const double ex2_1 = e01 * x2 + e11 * y2 + e21;
						const double c = x2 * ex1_0 + y2 * ex1_1 + ex1_2;
						const double cx = ifx2 * ex1_0 * ex1_0 + ify2 * ex1_1 * ex1_1;
						const double cy = ifx1 * ex2_0 * ex2_0 + ify1 * ex2_1 * ex2_1;
						kOut_[i] = (c * c) / (cx + cy);
					}
				}

				if (enforce3DCheirality && kModel_.hasPose())
				{
					const auto& kRt = kModel_.getPose();
					const double cutoffSq = cheiralityResidualCutoffSq;
					for (size_t i = 0; i < kCount_; ++i)
					{
						if (kOut_[i] < cutoffSq && !isPointCheiral(kStartPtr + i * kCols, kRt))
							kOut_[i] = 1e18;
					}
				}
			}

			void squaredResidualsIndexed(
				const DataMatrix& kData_,
				const models::Model& kModel_,
				const size_t* __restrict__ kIndices_,
				const size_t kCount_,
				double* __restrict__ kOut_) const override
			{
				const ModelMatrix& kDescriptor = kModel_.getData();
				const double e00 = kDescriptor(0, 0), e01 = kDescriptor(0, 1), e02 = kDescriptor(0, 2);
				const double e10 = kDescriptor(1, 0), e11 = kDescriptor(1, 1), e12 = kDescriptor(1, 2);
				const double e20 = kDescriptor(2, 0), e21 = kDescriptor(2, 1), e22 = kDescriptor(2, 2);
				const double ifx1 = invFx1Sq, ify1 = invFy1Sq, ifx2 = invFx2Sq, ify2 = invFy2Sq;
				const size_t kCols = static_cast<size_t>(kData_.cols());
				const double* __restrict__ kBase = kData_.data();
				for (size_t i = 0; i < kCount_; ++i)
				{
					const double* __restrict__ row = kBase + kIndices_[i] * kCols;
					const double x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
					const double ex1_0 = e00 * x1 + e01 * y1 + e02;
					const double ex1_1 = e10 * x1 + e11 * y1 + e12;
					const double ex1_2 = e20 * x1 + e21 * y1 + e22;
					const double ex2_0 = e00 * x2 + e10 * y2 + e20;
					const double ex2_1 = e01 * x2 + e11 * y2 + e21;
					const double c = x2 * ex1_0 + y2 * ex1_1 + ex1_2;
					const double cx = ifx2 * ex1_0 * ex1_0 + ify2 * ex1_1 * ex1_1;
					const double cy = ifx1 * ex2_0 * ex2_0 + ify1 * ex2_1 * ex2_1;
					kOut_[i] = (c * c) / (cx + cy);
				}
				if (enforce3DCheirality && kModel_.hasPose())
				{
					const auto& kRt = kModel_.getPose();
					const double cutoffSq = cheiralityResidualCutoffSq;
					for (size_t i = 0; i < kCount_; ++i)
					{
						if (kOut_[i] < cutoffSq && !isPointCheiral(kBase + kIndices_[i] * kCols, kRt))
							kOut_[i] = 1e18;
					}
				}
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
				if (enforce3DCheirality && kMinimalSample_ != nullptr)
					return model_.hasPose();
				if (kMinimalSample_ != nullptr &&
					!isOrientationValid(model_.getData().block<3, 3>(0, 0), kData_, kMinimalSample_, sampleSize()))
					return false;
				return true;
			}

			FORCE_INLINE void getEpipole(
				Eigen::Vector3d &epipole_,
				const Eigen::Matrix3d &kEssentialMatrix_) const
			{
				constexpr double epsilon = 1.9984e-15;
				epipole_ = kEssentialMatrix_.row(0).cross(kEssentialMatrix_.row(2));
				for (auto i = 0; i < 3; i++)
					if ((epipole_(i) > epsilon) || (epipole_(i) < -epsilon))
						return;
				epipole_ = kEssentialMatrix_.row(1).cross(kEssentialMatrix_.row(2));
			}

			FORCE_INLINE double getOrientationSignum(
				const Eigen::Matrix3d &kEssentialMatrix_,
				const Eigen::Vector3d &kEpipole_,
				const double* kPoint_) const
			{
				double signum1 = kEssentialMatrix_(0, 0) * kPoint_[2] + kEssentialMatrix_(1, 0) * kPoint_[3] + kEssentialMatrix_(2, 0),
					signum2 = kEpipole_(1) - kEpipole_(2) * kPoint_[1];
				return signum1 * signum2;
			}

			FORCE_INLINE bool isOrientationValid(
				const Eigen::Matrix3d &kEssentialMatrix_,
				const DataMatrix &kData_,
				const size_t *kSample_,
				size_t kSampleSize_) const
			{
				Eigen::Vector3d epipole;
				getEpipole(epipole, kEssentialMatrix_);
				double signum1, signum2;
				signum2 = getOrientationSignum(kEssentialMatrix_, epipole, kData_.row(kSample_[0]).data());
				for (size_t i = 1; i < kSampleSize_; i++)
				{
					signum1 = getOrientationSignum(kEssentialMatrix_, epipole, kData_.row(kSample_[i]).data());
					if (signum2 * signum1 < 0)
						return false;
				}
				return true;
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			FORCE_INLINE bool isValidSample(
				const DataMatrix& kData_, // All data points
				const size_t *kSample_) const override // The indices of the selected points
			{
				return true;
			}
		};
	}
}