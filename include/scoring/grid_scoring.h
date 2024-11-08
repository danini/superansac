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
#include "../neighborhood/grid_neighborhood_graph.h"
#include "../utils/types.h"
#include "abstract_scoring.h"
#include "score.h"
#include <Eigen/Core>

namespace superansac {
namespace scoring {

template<size_t _DimensionNumber>
class GridScoring : public AbstractScoring
{
    protected:
        neighborhood::GridNeighborhoodGraph<_DimensionNumber> *neighborhood;

    public:
        // Constructor 
        GridScoring() : neighborhood(nullptr) 
        {

        }

        // Destructor
        ~GridScoring() {}

        // Set the neighborhood structure
        FORCE_INLINE void setNeighborhood(
            neighborhood::GridNeighborhoodGraph<_DimensionNumber> *neighborhood_)
        {
            if (!neighborhood_->isInitialized())
                throw std::runtime_error("The neighborhood graph is not initialized.");
            neighborhood = neighborhood_;
        }

        // Set the threshold
        FORCE_INLINE void setThreshold(const double kThreshold_)
        {
            threshold = 1.5 * kThreshold_;
            squaredThreshold = threshold * threshold;
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
            // Retrieve the cells in the neighborhood graph
			const std::unordered_map<size_t, std::pair<std::vector<size_t>, std::vector<size_t>>>& cells = neighborhood->getCells();
            // The squared residual
            double squaredResidual;
            // The number of points in the cell
            size_t pointNumber;
            // The number of inliers in the cell
            double lowestSquaredResidual;
            // The score
            double score = 0.0;
            // The index of the cell
            size_t cellIdx = 0;

			for (const auto &[key, value] : cells)
			{
				// Point container
				const auto &kPoints = std::get<0>(value);
                lowestSquaredResidual = std::numeric_limits<double>::max();

                // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
                for (const auto &kPointIdx : kPoints)
                {
                    // Calculate the point-to-model residual
                    squaredResidual =
                        kEstimator_->squaredResidual(kData_.row(kPointIdx),
                            kModel_);

                    // If the residual is smaller than the threshold, store it as an inlier and
                    // increase the score.
                    if (squaredResidual < squaredThreshold)
                    {
                        // Store the inlier if needed
                        if (kStoreInliers_)
                            inliers_.push_back(kPointIdx);
                        // Check whether the current point has better residual than the best inlier in the cell
                        if (squaredResidual < lowestSquaredResidual)
                            lowestSquaredResidual = squaredResidual;
                    }
                }

                // Increase the current score by the score of the best-fitting inlier in the cell
                if (lowestSquaredResidual < squaredThreshold)
                    score += 1.0 - lowestSquaredResidual / squaredThreshold;

                // If the best potential score is smaller than the best score, return
                if (score + cells.size() - cellIdx < kBestScore_.getValue())
                    return kEmptyScore;
                ++cellIdx;
			}

            return Score(inliers_.size(), score);
        }

        // Get weights for the points
        FORCE_INLINE void getWeights(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<double> &weights_, // The weights of the points
            const std::vector<size_t> *kIndices_ = nullptr) const  // The indices of the points
        {
            if (kIndices_ == nullptr)
            {
                // The number of points
                const int kPointNumber = kData_.rows();
                // The squared residual
                double squaredResidual;
                // Allocate memory for the weights
                weights_.resize(kPointNumber);

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
                        weights_[pointIdx] = 1.0 - squaredResidual / squaredThreshold;
                    else
                        weights_[pointIdx] = 0.0;
                }
            }
            else
            {
                // The number of points
                const int kPointNumber = kIndices_->size();
                // The squared residual
                double squaredResidual;
                // Allocate memory for the weights
                weights_.resize(kPointNumber);

                // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
                for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
                {
                    // Calculate the point-to-model residual
                    squaredResidual =
                        kEstimator_->squaredResidual(kData_.row((*kIndices_)[pointIdx]),
                            kModel_);

                    // If the residual is smaller than the threshold, store it as an inlier and
                    // increase the score.
                    if (squaredResidual < squaredThreshold)
                        weights_[pointIdx] = 1.0 - squaredResidual / squaredThreshold;
                    else
                        weights_[pointIdx] = 0.0;
                }
            }
        }
};

}
}