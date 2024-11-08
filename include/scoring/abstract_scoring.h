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
#include "score.h"
#include <vector>
#include <Eigen/Core>

namespace superansac {
namespace scoring {

class AbstractScoring
{
    public:
        // Constructor 
        AbstractScoring() : threshold(1.0), squaredThreshold(1.0) {}

        // Destructor
        virtual ~AbstractScoring() {}

        // Set the threshold
        FORCE_INLINE virtual void setThreshold(const double kThreshold_) = 0;

        // Get the threshold
        FORCE_INLINE const double &getThreshold() const { return threshold; };

        // Set the image size
        FORCE_INLINE void setImageSize(
            const double kWidthSrc_, 
            const double kHeightSrc_, 
            const double kWidthDst_,
            const double kHeightDst_)
        {
            imageHeightSrc = kHeightSrc_;
            imageWidthSrc = kWidthSrc_;
            imageHeightDst = kHeightDst_;
            imageWidthDst = kWidthDst_;
        }

        // Sample function
        FORCE_INLINE virtual Score score(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<size_t> &inliers_, // Inlier indices
            const bool kStoreInliers_ = true, // Store inliers or not
            const Score& kBestScore_ = Score(),
            std::vector<const std::vector<size_t>*> *kPotentialInlierSets_ = nullptr) const = 0; // The potential inlier sets from the inlier selector

        // Get weights for the points
        FORCE_INLINE virtual void getWeights(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<double> &weights_, // The weights of the points
            const std::vector<size_t> *kIndices_ = nullptr) const = 0; // The indices of the points
        
        FORCE_INLINE void getInliers(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<std::pair<double, size_t>> &inliers_, // The inliers of the model
            const double kThreshold_, // The threshold for inlier selection
            const bool kReturnSquaredResidual = true) const // Return the squared residuals or not
        {
            // The number of points
            const int kPointNumber = kData_.rows();
            // The squared residual
            double squaredResidual;

            // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
            inliers_.clear();
            inliers_.reserve(kPointNumber);
            for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
            {
                // Calculate the point-to-model residual
                squaredResidual =
                    kEstimator_->squaredResidual(kData_.row(pointIdx),
                        kModel_);

                // If the residual is smaller than the threshold, store it as an inlier and
                // increase the score.
                if (squaredResidual < squaredThreshold)
                    if (kReturnSquaredResidual)
                        inliers_.emplace_back(std::make_pair(squaredResidual, pointIdx));
                    else
                        inliers_.emplace_back(std::make_pair(std::sqrt(squaredResidual), pointIdx));
            }
        }

    protected:
        double threshold,
            squaredThreshold,
            imageWidthSrc,
            imageHeightSrc,
            imageWidthDst,
            imageHeightDst;
};

}
}