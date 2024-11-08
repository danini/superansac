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

#include "../utils/macros.h"
#include "../models/model.h"
#include "../estimators/abstract_estimator.h"
#include "../utils/types.h"
#include "abstract_scoring.h"
#include "score.h"
#include <Eigen/Core>

namespace superansac {
namespace scoring {

class ACRANSACScoring : public AbstractScoring
{
    protected:
        const size_t kStepNumber;
        
    public:
        // Constructor 
        ACRANSACScoring() : ACRANSACScoring(10) {}

        ACRANSACScoring(const size_t kStepNumber_) : 
            kStepNumber(kStepNumber_)
        {

        }

        // Destructor
        ~ACRANSACScoring() {}

        // Set the threshold
        FORCE_INLINE void setThreshold(const double kThreshold_)
        {
            threshold = kThreshold_;
            squaredThreshold = threshold * threshold;
        }

        FORCE_INLINE double logAlpha0(const double &kWidth_, const double &kHeight_, const double &kScalingFactor_) const
        {
            // Ratio of area: unit circle over image area
            return log10(M_PI / (kWidth_ * kHeight_) / (kScalingFactor_ * kScalingFactor_));
        }

        /// logarithm (base 10) of binomial coefficient
        FORCE_INLINE double logcombi(
            uint32_t k,
            uint32_t n,
            const std::vector<double> & vec_log10) const // lookuptable in [0,n+1]
        {
            if (k>=n) return 0.f;
            if (n-k<k) k=n-k;
            double r(0.f);
            for (uint32_t i = 1; i <= k; ++i)
                r += vec_log10[n-i+1] - vec_log10[i];
            return r;
        }

        /// tabulate logcombi(.,n)
        FORCE_INLINE void makeLogCombiN(
            uint32_t n,
            std::vector<double> & l,
            std::vector<double> & vec_log10) const // lookuptable [0,n+1]
        {
            l.resize(n+1);
            for (uint32_t k = 0; k <= n; ++k)
                l[k] = logcombi(k, n, vec_log10);
        }

        /// tabulate logcombi(k,.)
        FORCE_INLINE void makeLogCombiK(
            uint32_t k,
            uint32_t nmax,
            std::vector<double> & l,
            std::vector<double> & vec_log10) const // lookuptable [0,n+1]
        {
            l.resize(nmax+1);
            for (uint32_t n = 0; n <= nmax; ++n)
                l[n] = logcombi(k, n, vec_log10);
        }

        FORCE_INLINE void makeLogCombi(
            uint32_t k,
            uint32_t n,
            std::vector<double> & vec_logc_k,
            std::vector<double> & vec_logc_n) const
        {
            // compute a lookuptable of log10 value for the range [0,n+1]
            std::vector<double> vec_log10(n + 1);
            for (uint32_t i = 0; i <= n; ++i)
                vec_log10[i] = log10(static_cast<double>(i));

            makeLogCombiN(n, vec_logc_n, vec_log10);
            makeLogCombiK(k, n, vec_logc_k, vec_log10);
        }

        // Sample function
        FORCE_INLINE Score score(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<size_t> &inliers_, // Inlier indices
            const bool kStoreInliers_ = true,
            const Score& kBestScore_ = Score(),
            std::vector<const std::vector<size_t>*> *kPotentialInlierSets_ = nullptr) const // The potential inlier sets from the inlier selector
        {   
            // Create a static empty Score
            static const Score kEmptyScore;
            // The number of points
            const int kPointNumber = kData_.rows();
            // Minimal sample size and other parameters from the estimator
            const int kMinimalSampleSize = kEstimator_->sampleSize();
            const double kMultError = kEstimator_->multError();
            const double kLogAlpha0 = kEstimator_->logAlpha0(imageWidthDst, imageHeightDst);
            // The squared residual
            double squaredResidual;
            // Score and inlier number
            int inlierNumber = 0;
            // The score of the previous best model
            const double kBestScoreValue = kBestScore_.getValue();
            // The residuals of the points
            std::vector<std::pair<double,size_t>> residuals;
            residuals.reserve(kPointNumber);

            // A-Contrario Epsilon 0 value
            static double mLoge0;

            // Combinatorial log
            static std::vector<double> mLogcN, mLogcK;
            
            // If the log combi is not computed, compute it
            if (mLogcN.size() != kPointNumber + 1)
            {
                // Clear the vectors
                mLogcK.clear();
                mLogcN.clear();

                // Precompute log combi
                mLoge0 = log10(static_cast<double>(kEstimator_->maximumMinimalSolutions()) * (kPointNumber - kMinimalSampleSize));
                makeLogCombi(kMinimalSampleSize, kPointNumber, mLogcK, mLogcN);
            }

            // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
            for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
            {
                // Calculate the point-to-model residual
                squaredResidual =
                    kEstimator_->squaredResidual(kData_.row(pointIdx),
                        kModel_);

                // If the residual is smaller than the threshold, store it as an inlier and
                // increase the score.
                if (squaredResidual < squaredThreshold)
                    residuals.emplace_back(std::make_pair(std::sqrt(squaredResidual),pointIdx));
            }

            // Sort the residuals        
            std::sort(residuals.begin(), residuals.end(), [](const std::pair<double,size_t> &a, const std::pair<double,size_t> &b) { return a.first < b.first; });

            // Get the maximum residual
            const double kMaxResidual = residuals.back().first;
            const double kResidualStep = kMaxResidual / kStepNumber;
            const size_t kSampleSize = kEstimator_->sampleSize();
            double currentThreshold = 0;
            size_t currentMaxIdx = kSampleSize;
            inliers_.reserve(kPointNumber);
            
            if (kResidualStep < std::numeric_limits<float>::epsilon())
                return kEmptyScore;

            // NFA calculation
            using nfaThresholdT = std::tuple<double,double,size_t>; // NFA and residual threshold
            nfaThresholdT currentBestNFA(std::numeric_limits<double>::infinity(), 0.0, 4);

            // Iterate through the thresholds
            for (currentThreshold = kResidualStep; currentThreshold <= kMaxResidual; currentThreshold += kResidualStep)
            {
                // Count the inliers of the current threshold
                while (currentMaxIdx < residuals.size() && residuals[currentMaxIdx].first <= currentThreshold)
                    ++currentMaxIdx;

                // If the number of inliers is smaller than the sample size, continue
                if (residuals[currentMaxIdx].first <= std::numeric_limits<float>::epsilon())
                    continue;

                const double kLogAlpha = kLogAlpha0
                    + kMultError * log10(currentThreshold
                    + std::numeric_limits<double>::epsilon());

                const nfaThresholdT currentNFA( mLoge0
                    + kLogAlpha * static_cast<double>(currentMaxIdx - kMinimalSampleSize)
                    + mLogcN[currentMaxIdx]
                    + mLogcK[currentMaxIdx], 
                    currentThreshold,
                    currentMaxIdx);

                // Keep the best NFA iff it is meaningful ( NFA < 0 ) and better than the existing one
                if (std::get<0>(currentNFA) < std::get<0>(currentBestNFA) && std::get<0>(currentNFA) < 0)
                {
                    currentBestNFA = currentNFA;
                
                    if (kStoreInliers_)
                    {
                        // Add the inliers that hasn't been added yet 
                        for (size_t residualIdx = inliers_.size(); residualIdx < currentMaxIdx; ++residualIdx)
                            inliers_.push_back(residuals[residualIdx].second);
                    }
                }
            }

            return scoring::Score(std::get<2>(currentBestNFA), -std::get<0>(currentBestNFA));
        }

        // Get weights for the points
        FORCE_INLINE void getWeights(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<double> &weights_, // The weights of the points
            const std::vector<size_t> *kIndices_ = nullptr) const // The indices of the points 
        {
            if (kIndices_ == nullptr)
            {
                weights_.resize(kData_.rows());
                for (size_t i = 0; i < kData_.rows(); ++i)
                    weights_[i] = 1.0;
            }
            else
            {
                weights_.resize(kIndices_->size());
                for (size_t i = 0; i < kIndices_->size(); ++i)
                    weights_[i] = 1.0;
            }
        }
};

}
}