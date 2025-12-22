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
#include "../utils/types.h"
#include "../estimators/abstract_estimator.h"
#include "abstract_scoring.h"
#include "score.h"
#include <Eigen/Core>

namespace superansac {
namespace scoring {

class MINPRANScoring : public AbstractScoring
{
    protected:
        const size_t kStepNumber;
        std::vector<std::vector<unsigned long long>> binomCoeffs;


    public:
        // Constructor 
        MINPRANScoring() : kStepNumber(25) {}

        // Destructor
        ~MINPRANScoring() {}

        // Set the threshold
        FORCE_INLINE void setThreshold(const double kThreshold_)
        {
            threshold = kThreshold_;
            squaredThreshold = threshold * threshold;
        }
        
        FORCE_INLINE void updateSPRTParameters(const Score& currentBest, 
            int iterationIndex, 
            size_t totalPoints)
        {
            
        }

        // Function to calculate logarithm of factorial using Stirling's approximation
        double logFactorial(int n) const 
        {
            if (n <= 1) return 0.0;
            double x = n;
            return x * log(x) - x + 0.5 * log(2 * M_PI * x);
        }
        
        // Function to calculate the binomial coefficient N choose k using logarithms to prevent overflow
        double binomialCoefficient(int N, int k) const {
            return exp(logFactorial(N) - logFactorial(k) - logFactorial(N - k));
        }

        // Function to perform numerical integration using the trapezoidal rule
        double integrate(std::function<double(double)> f, double a, double b, int n = 1000) const {
            double h = (b - a) / n;
            double integral = (f(a) + f(b)) / 2.0;
            for (int i = 1; i < n; ++i) {
                integral += f(a + i * h);
            }
            integral *= h;
            return integral;
        }

        // Function to calculate the integrand t^(k-1) * (1 - t)^(N - k)
        std::function<double(double)> getIntegrand(int k, int N) const 
        {
            return [k, N](double t) {
                return pow(t, k - 1) * pow(1 - t, N - k);
            };
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
            // The squared residual
            double squaredResidual;
            // Score and inlier number
            int inlierNumber = 0;
            // The score of the previous best model
            const double kBestScoreValue = kBestScore_.getValue();
            // 
            std::vector<std::pair<double,size_t>> residuals;
            residuals.reserve(kPointNumber);

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
                    residuals.emplace_back(std::make_pair(squaredResidual,pointIdx));
            }

            // Sort the residuals
            std::sort(residuals.begin(), residuals.end(), [](const std::pair<double,size_t> &a, const std::pair<double,size_t> &b) { return a.first < b.first; });

            // Get the maximum residual
            const double kMaxResidual = residuals.back().first;
            const double kResidualStep = kMaxResidual / kStepNumber;
            const size_t kSampleSize = kEstimator_->sampleSize();
            double currentThreshold = 0;
            size_t currentMaxIdx = 0;
            double randomness,
                minRandomness = std::numeric_limits<double>::max(),
                bestThreshold = 0,
                bestInlierNumber = 0;

            // Iterate through the thresholds
            for (currentThreshold = kResidualStep; currentThreshold <= kMaxResidual; currentThreshold += kResidualStep)
            {
                // Count the inliers of the current threshold
                while (currentMaxIdx + 1 < residuals.size() && residuals[currentMaxIdx + 1].first < currentThreshold)
                    ++currentMaxIdx;

                // If the number of inliers is smaller than the sample size, continue
                if (currentMaxIdx < kSampleSize + 1)
                    continue;

                // Calculate the binomial coefficient part
                double binomCoeff = binomialCoefficient(kPointNumber, currentMaxIdx - 1);
                
                // Get the integrand function
                auto integrand = getIntegrand(currentMaxIdx, kPointNumber);
                
                // Calculate the integral part of the expression using adaptive quadrature
                double integralPart = integrate(integrand, 0, currentThreshold / threshold);
                
                // Calculate the final result
                double randomness = binomCoeff * integralPart;

                // Check if the randomness is NaN or inf
                if (std::isnan(randomness) || std::isinf(randomness))
                    continue;

                // Calculate the final result
                if (randomness < minRandomness)
                {
                    minRandomness = randomness;
                    bestThreshold = currentThreshold;
                    bestInlierNumber = currentMaxIdx;
                }
            }

            // Store the inliers
            if (kStoreInliers_)
            {
                inliers_.reserve(bestInlierNumber);
                for (size_t i = 0; i <= bestInlierNumber; ++i)
                    inliers_.push_back(residuals[i].second);
            }

            if (bestInlierNumber < kSampleSize)
                return kEmptyScore;
            
            return Score(bestInlierNumber, 1.0 / abs(minRandomness));
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