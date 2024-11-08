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

#include "abstract_sampler.h"
#include "../utils/uniform_random_generator.h"
#include "../neighborhood/abstract_neighborhood.h"
#include "../utils/types.h"

#include <vector>
#include <queue>

namespace superansac {
namespace samplers {

class AdaptiveReorderingSampler : public AbstractSampler
{
    protected:
        std::vector<std::tuple<double, size_t, size_t, double, double>> probabilities;
        double estimatorVariance;
        double randomness,
            randomness2,
            randomnessRandMax;
			
        std::priority_queue<std::pair<double, size_t>, 
            std::vector<std::pair<double, size_t>> > processingQueue;
    public:
        // Constructor 
        AdaptiveReorderingSampler() {}
        // Destructor
        ~AdaptiveReorderingSampler() {}

        // Return the name of the sampler
        constexpr static const char *name()
        {
            return "Adaptive Reordering Sampler";
        }

        // Initializes any non-trivial variables and sets up sampler if
        // necessary. Must be called before sample is called.
        FORCE_INLINE void initialize(
            const DataMatrix &kData_)
        {
            initialize(kData_.rows());
        }

        // Initialize function
        FORCE_INLINE void initialize(
            const DataMatrix * const kData_,
            const std::vector<double> &kInlierProbabilities_,
            const double kEstimatorVariance_ = 0.9765, //0.12,
            const double kRandomness_ = 0.01) 
        {
            // Check if the number of correspondences and the number of provided probabilities match
            if (kInlierProbabilities_.size() != kData_->rows())
                throw std::runtime_error("The number of correspondences and the number of provided probabilities do not match.");

            // Set the variables
            randomness = kRandomness_;
            randomness2 = randomness / 2.0;
            randomnessRandMax = randomness / static_cast<double>(RAND_MAX);

            // Saving the probabilities
            double a, b, probability;
            probabilities.reserve(kInlierProbabilities_.size());
            for (size_t pointIdx = 0; pointIdx < kInlierProbabilities_.size(); ++pointIdx)
            {
                probability = kInlierProbabilities_[pointIdx];
                if (probability == 1.0)
                    probability -= 1e-6; 

                a = probability * probability * (1.0 - probability) / kEstimatorVariance_ - probability;
                b = a * (1.0 - probability) / probability;
                
                probabilities.emplace_back(std::make_tuple(probability, pointIdx, 0, a, b));
                processingQueue.emplace(std::make_pair(probability, pointIdx));
            }
        }

        // Initialize function
        FORCE_INLINE void initialize(
            const size_t kPointNumber_) // Data matrix
        {
            if (probabilities.empty())
                throw std::runtime_error("The AdaptiveReorderingSampler should be initialized with the data matrix and the probabilities.");
        }

        FORCE_INLINE void update(
            const size_t* const kSample_,
            const size_t& kSampleSize_,
            const size_t& kIterationNumber_,
            const double& kInlierRatio_)
        {
            for (size_t i = 0; i < kSampleSize_; ++i)
			{
				const size_t& kSampleIdx = kSample_[i];
				size_t& appearanceNumber = std::get<2>(probabilities[kSampleIdx]);
				++appearanceNumber;

				const double &a = std::get<3>(probabilities[kSampleIdx]); 
				const double &b = std::get<4>(probabilities[kSampleIdx]); 

				double& updatedInlierRatio = std::get<0>(probabilities[kSampleIdx]);

				updatedInlierRatio = 
					abs(a / (a + b + appearanceNumber)) + 
					randomnessRandMax * static_cast<double>(rand()) - randomness2;
					
				updatedInlierRatio = 
					std::max(0.0, std::min(0.999, updatedInlierRatio));

				processingQueue.emplace(std::make_pair(updatedInlierRatio, kSampleIdx));
			}
        }

        void reset(const size_t &kDataSize_)
        {
        }


        // Sample function
        FORCE_INLINE bool sample(
            const DataMatrix &kData_, // Data matrix
            const int kNumSamples_, // Number of samples
            size_t *kSamples_) // Sample indices
        {
            return sample(kData_.rows(), kNumSamples_, kSamples_);
        }

        // Sample function
        FORCE_INLINE bool sample(
            const size_t kPointNumber_,
            const int kNumSamples_, 
            size_t *samples_)
        {
            for (size_t i = 0; i < kNumSamples_; ++i)
			{
				const auto& item = processingQueue.top();
				samples_[i] = item.second;
				processingQueue.pop();
			}
			return true;
        }
};

}
}