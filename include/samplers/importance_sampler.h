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
#include <iostream>

namespace superansac {
namespace samplers {

class ImportanceSampler : public AbstractSampler
{
    protected:
        std::unique_ptr<utils::UniformRandomGenerator<size_t>> randomGenerator;
        std::discrete_distribution<int> multinomialDistribution;

    public:
        // Constructor 
        ImportanceSampler() {}
        // Destructor
        ~ImportanceSampler() {}

        // Return the name of the sampler
        constexpr static const char *name()
        {
            return "ImportanceSampler";
        }

        // Initializes any non-trivial variables and sets up sampler if
        // necessary. Must be called before sample is called.
        FORCE_INLINE void initialize(
            const DataMatrix &kData_)
        {
            initialize(kData_.rows());
        }

        FORCE_INLINE void setProbabilities(const std::vector<double> &kProbabilities_)
        {
            // Initialize the distribution from the point probabilities
            multinomialDistribution = std::discrete_distribution<int>(
                std::begin(kProbabilities_), 
                std::end(kProbabilities_));
        }

        // Initialize function
        FORCE_INLINE void initialize(
            const size_t kPointNumber_) // Data matrix
        {
            // Initialize the random generator
            randomGenerator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
            // Reset the random generator
            randomGenerator->resetGenerator(0, kPointNumber_ - 1);
        }

        FORCE_INLINE void update(
            const size_t* const subset_,
            const size_t& sampleSize_,
            const size_t& iteration_number_,
            const double& inlier_ratio_)
        {
            
        }

        void reset(const size_t &kDataSize_)
        {
            // Reset the random generator
            randomGenerator->resetGenerator(0, kDataSize_ - 1);
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
            for (size_t sample_idx = 0; sample_idx < kNumSamples_; ++sample_idx)
                samples_[sample_idx] = multinomialDistribution(randomGenerator->getGenerator());
            return true;
        }
};

}
}