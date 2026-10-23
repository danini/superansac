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
//     * Neither the name of ETH Zurich nor the
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
// Author: Daniel Barath (majti89@gmail.com)
#pragma once

#include <vector>
#include <Eigen/Eigen>

#include "../utils/types.h"
#include "../utils/uniform_random_generator.h"
#include "../models/model.h"
#include "../scoring/abstract_scoring.h"
#include "abstract_estimator.h"
#include "solver_homography_four_point.h"
#include "solver_fundamental_matrix_plane_and_parallax.h"

namespace superansac
{
	namespace degensac
	{
		// Squared reprojection error of a correspondence w.r.t. a homography
		// (same arithmetic as HomographyEstimator::squaredResidual).
		FORCE_INLINE double squaredHomographyReprojectionError(
			const double *kPoint_,
			const Eigen::Matrix3d &kHomography_)
		{
			const double &x1 = kPoint_[0],
				&y1 = kPoint_[1],
				&x2 = kPoint_[2],
				&y2 = kPoint_[3];

			const double t3 = kHomography_(2, 0) * x1 + kHomography_(2, 1) * y1 + kHomography_(2, 2);
			const double t1 = (kHomography_(0, 0) * x1 + kHomography_(0, 1) * y1 + kHomography_(0, 2)) / t3;
			const double t2 = (kHomography_(1, 0) * x1 + kHomography_(1, 1) * y1 + kHomography_(1, 2)) / t3;

			const double d1 = x2 - t1;
			const double d2 = y2 - t2;
			return d1 * d1 + d2 * d2;
		}

		// DEGENSAC (Chum, Werner, Matas: "Two-View Geometry Estimation Unaffected
		// by a Dominant Plane", CVPR 2005), ported from graph-cut-ransac's
		// FundamentalMatrixEstimator::applyDegensac.
		//
		// Called on a minimal-sample candidate model that is about to become the
		// so-far-best. (1) Tests whether >= 5 of the 7 sample points are
		// consistent with a homography implied by F and a point triplet
		// (H-degenerate sample). (2) If so, fits the homography to its inliers
		// and re-estimates F with the plane-and-parallax solver (2 off-plane
		// points) in a small inner RANSAC, ranking the candidates with the
		// pipeline's own scoring.
		//
		// Returns true if the (possibly replaced) candidate may be accepted:
		//  * the sample is not H-degenerate -> candidate kept as-is;
		//  * the sample is H-degenerate and the plane-and-parallax re-estimation
		//    found a candidate beating kRunningBest_ -> candidate REPLACED
		//    (model/score/inliers are overwritten);
		//  * otherwise -> returns false: the H-degenerate candidate must be
		//    REJECTED (this mirrors the reference, where a degenerate model that
		//    cannot be fixed never becomes the best model -- keeping it would
		//    lock the search onto the dominant plane).
		inline bool applyDegensac(
			const DataMatrix &kData_, // All data points (normalized coordinates)
			const estimator::Estimator *kEstimator_, // The fundamental matrix estimator
			scoring::AbstractScoring *kScoring_, // The scoring object of the pipeline
			const size_t *kMinimalSample_, // The 7-point minimal sample of the candidate
			const double kSquaredHomographyThreshold_, // Squared H-consistency threshold (normalized units)
			const size_t kInnerIterations_, // Number of inner plane-and-parallax draws
			const scoring::Score &kRunningBest_, // The current best score of the outer loop
			utils::UniformRandomGenerator<size_t> &rng_, // Run-scoped RNG (deterministic)
			models::Model &candidateModel_, // In/out: the candidate model
			scoring::Score &candidateScore_, // In/out: the candidate score
			std::vector<size_t> &candidateInliers_) // In/out: the inliers of the candidate
		{
			constexpr size_t kSampleSize = 7;
			// The triplets of sample points used to hypothesize the homography
			constexpr size_t kTriplets[] = {
				0, 1, 2,
				3, 4, 5,
				0, 1, 6,
				3, 4, 6,
				2, 5, 6 };
			constexpr size_t kTripletNumber = 5;

			const size_t kColumns = kData_.cols();
			const double *kDataPtr = kData_.data();

			// The fundamental matrix of the current best model
			const Eigen::Matrix3d kFundamentalMatrix =
				candidateModel_.getData().block<3, 3>(0, 0);

			// The epipole in the second image (left null vector of F)
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(
				kFundamentalMatrix,
				Eigen::ComputeFullU | Eigen::ComputeFullV);

			// H = A - e (M^-1 b)^T is scale-invariant in the epipole, so the
			// normalization by U(2,2) is only done when it is safe.
			Eigen::Vector3d epipole = svd.matrixU().col(2);
			if (std::abs(epipole(2)) > 1e-12)
				epipole /= epipole(2);

			// The cross-product matrix of the epipole
			Eigen::Matrix3d epipolarCross;
			epipolarCross << 0, -epipole(2), epipole(1),
				epipole(2), 0, -epipole(0),
				-epipole(1), epipole(0), 0;

			const Eigen::Matrix3d A = epipolarCross * kFundamentalMatrix;

			// Test the triplets for H-degeneracy
			bool hDegenerateSample = false;
			Eigen::Matrix3d bestHomography;
			size_t bestTripletInliers = 0;
			double bestTripletErrorSum = std::numeric_limits<double>::max();
			for (size_t tripletIdx = 0; tripletIdx < kTripletNumber; ++tripletIdx)
			{
				const size_t kTripletOffset = tripletIdx * 3;
				const size_t kPoint1Idx = kMinimalSample_[kTriplets[kTripletOffset]],
					kPoint2Idx = kMinimalSample_[kTriplets[kTripletOffset + 1]],
					kPoint3Idx = kMinimalSample_[kTriplets[kTripletOffset + 2]];

				const double *kPoint1Ptr = kDataPtr + kPoint1Idx * kColumns;
				const double *kPoint2Ptr = kDataPtr + kPoint2Idx * kColumns;
				const double *kPoint3Ptr = kDataPtr + kPoint3Idx * kColumns;

				const Eigen::Vector3d point11(kPoint1Ptr[0], kPoint1Ptr[1], 1),
					point12(kPoint2Ptr[0], kPoint2Ptr[1], 1),
					point13(kPoint3Ptr[0], kPoint3Ptr[1], 1),
					point21(kPoint1Ptr[2], kPoint1Ptr[3], 1),
					point22(kPoint2Ptr[2], kPoint2Ptr[3], 1),
					point23(kPoint3Ptr[2], kPoint3Ptr[3], 1);

				// The cross-products of the destination points and the epipole
				const Eigen::Vector3d point1CrossEpipole = point21.cross(epipole),
					point2CrossEpipole = point22.cross(epipole),
					point3CrossEpipole = point23.cross(epipole);

				const double kNorm1 = point1CrossEpipole.squaredNorm(),
					kNorm2 = point2CrossEpipole.squaredNorm(),
					kNorm3 = point3CrossEpipole.squaredNorm();
				if (kNorm1 < std::numeric_limits<double>::epsilon() ||
					kNorm2 < std::numeric_limits<double>::epsilon() ||
					kNorm3 < std::numeric_limits<double>::epsilon())
					continue;

				Eigen::Vector3d b;
				b << point21.cross(A * point11).transpose() * point1CrossEpipole / kNorm1,
					point22.cross(A * point12).transpose() * point2CrossEpipole / kNorm2,
					point23.cross(A * point13).transpose() * point3CrossEpipole / kNorm3;

				Eigen::Matrix3d M;
				M << point11(0), point11(1), point11(2),
					point12(0), point12(1), point12(2),
					point13(0), point13(1), point13(2);

				const Eigen::Matrix3d homography =
					A - epipole * (M.inverse() * b).transpose();
				if (!homography.allFinite())
					continue;

				// Count the sample points consistent with the implied homography
				size_t inlierNumber = 3;
				double errorSum = 0.0;
				for (size_t i = 0; i < kSampleSize; ++i)
				{
					const size_t kIdx = kMinimalSample_[i];
					if (kIdx == kPoint1Idx ||
						kIdx == kPoint2Idx ||
						kIdx == kPoint3Idx)
						continue;

					const double sqErr = squaredHomographyReprojectionError(
						kDataPtr + kIdx * kColumns, homography);
					if (sqErr < kSquaredHomographyThreshold_)
					{
						++inlierNumber;
						errorSum += sqErr;
					}
				}

				// If at least 5 of the 7 points are consistent with the homography,
				// the sample is H-degenerate.
				if (inlierNumber >= 5 &&
					(inlierNumber > bestTripletInliers ||
					 (inlierNumber == bestTripletInliers && errorSum < bestTripletErrorSum)))
				{
					bestHomography = homography;
					bestTripletInliers = inlierNumber;
					bestTripletErrorSum = errorSum;
					hDegenerateSample = true;
				}
			}

			if (!hDegenerateSample)
				return true; // Not degenerate: the candidate may be accepted as-is

			// Collect the homography's inliers across all points (or candidateInliers_)
			const size_t kPointNumber = kData_.rows();
			std::vector<size_t> homographyInliers;
			homographyInliers.reserve(candidateInliers_.size());
			for (const size_t &kInlierIdx : candidateInliers_)
				if (squaredHomographyReprojectionError(
						kDataPtr + kInlierIdx * kColumns, bestHomography) < kSquaredHomographyThreshold_)
					homographyInliers.emplace_back(kInlierIdx);

			// The homography must be estimable from its inliers
			static const estimator::solver::HomographyFourPointSolver kHomographySolver;
			if (homographyInliers.size() < kHomographySolver.sampleSize())
				return true;

			// Fit the homography to all of its inliers
			std::vector<models::Model> homographies;
			if (!kHomographySolver.estimateModel(kData_,
					homographyInliers.data(),
					homographyInliers.size(),
					homographies) ||
				homographies.size() != 1 ||
				!homographies[0].getData().allFinite())
				return true;

			const Eigen::Matrix3d kRefinedHomography = homographies[0].getData().block<3, 3>(0, 0);

			// Re-estimate the fundamental matrix by the plane-and-parallax
			// algorithm in a small inner RANSAC. Crucially, the true off-plane
			// correspondences are NOT in candidateInliers_ (since candidateModel_
			// has a wrong epipole induced by the degenerate sample), so we sample
			// from all points in kData_ that lie off the dominant plane H.
			estimator::solver::FundamentalMatrixPlaneParallaxSolver planeParallaxSolver;
			planeParallaxSolver.setHomography(&homographies[0].getData());

			if (kPointNumber < 2)
				return true;

			std::vector<size_t> offPlanePool;
			offPlanePool.reserve(kPointNumber);
			for (size_t idx = 0; idx < kPointNumber; ++idx)
			{
				if (squaredHomographyReprojectionError(
						kDataPtr + idx * kColumns, kRefinedHomography) >= kSquaredHomographyThreshold_)
				{
					offPlanePool.emplace_back(idx);
				}
			}

			if (offPlanePool.size() < 2)
				return true;

			bool updated = false;
			size_t sample[2];
			std::vector<models::Model> candidateModels;
			std::vector<size_t> candidateInliers,
				bestCandidateInliers;
			candidateInliers.reserve(kPointNumber);
			bestCandidateInliers.reserve(kPointNumber);
			models::Model bestCandidateModel;
			scoring::Score runningBest = candidateScore_;

			const size_t kTotalIters = std::max<size_t>(kInnerIterations_, 100);
			for (size_t iteration = 0; iteration < kTotalIters; ++iteration)
			{
				// Hybrid PROSAC / uniform sampling over offPlanePool
				size_t activePool = offPlanePool.size();
				if (iteration * 2 < kTotalIters)
				{
					const double frac = static_cast<double>(iteration + 1) / static_cast<double>(kTotalIters / 2);
					activePool = std::min(offPlanePool.size(),
						std::max<size_t>(12, static_cast<size_t>(frac * std::min<size_t>(offPlanePool.size(), 120))));
				}
				rng_.generateUniqueRandomSet(sample, 2, activePool - 1);
				sample[0] = offPlanePool[sample[0]];
				sample[1] = offPlanePool[sample[1]];

				candidateModels.clear();
				if (!planeParallaxSolver.estimateModel(kData_, sample, 2, candidateModels))
					continue;

				for (const auto &candidate : candidateModels)
				{
					candidateInliers.clear();
					const scoring::Score kCandidateScore = kScoring_->score(
						kData_, candidate, kEstimator_, candidateInliers, true, runningBest);

					if (runningBest < kCandidateScore)
					{
						runningBest = kCandidateScore;
						bestCandidateModel = candidate;
						bestCandidateInliers.swap(candidateInliers);
						updated = true;
					}
				}
			}

			if (!updated)
				return true; // Keep the candidate model if no better plane-and-parallax model was found

			// Replace the candidate with the re-estimated model
			candidateModel_ = bestCandidateModel;
			candidateScore_ = runningBest;
			candidateInliers_.swap(bestCandidateInliers);
			return true;
		}
	}
}
