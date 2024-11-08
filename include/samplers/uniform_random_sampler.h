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
#include "../utils/types.h"

#include <vector>
#include <iostream>

namespace superansac {
namespace samplers {

class UniformRandomSampler : public AbstractSampler
{
    protected:
        std::unique_ptr<utils::UniformRandomGenerator<size_t>> randomGenerator;

    public:
        // Constructor 
        UniformRandomSampler() {}
        // Destructor
        ~UniformRandomSampler() {}

        // Return the name of the sampler
        constexpr static const char *name()
        {
            return "UniformRandomSampler";
        }

        // Initializes any non-trivial variables and sets up sampler if
        // necessary. Must be called before sample is called.
        FORCE_INLINE void initialize(
            const DataMatrix &kData_)
        {
            randomGenerator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
            randomGenerator->resetGenerator(0,
                static_cast<size_t>(kData_.rows()));
        }

        // Initializes any non-trivial variables and sets up sampler if
        // necessary. Must be called before sample is called.
        FORCE_INLINE void initialize(
            const size_t kPointNumber_)
        {
            randomGenerator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
            randomGenerator->resetGenerator(0,
                static_cast<size_t>(kPointNumber_));
        }
        			
        FORCE_INLINE void update(
            const size_t* const subset_,
            const size_t& sample_size_,
            const size_t& iteration_number_,
            const double& inlier_ratio_)
        {
            
        }

        void reset(const size_t &kDataSize_)
        {
            randomGenerator->resetGenerator(0, kDataSize_);
        }

        // Sample function
        FORCE_INLINE bool sample(
            const DataMatrix &kData_,
            const int kNumSamples_, 
            size_t *samples_)
        {                
			return sample(kData_.rows(), kNumSamples_, samples_);
        }

        // Sample function
        FORCE_INLINE bool sample(
            const size_t kPointNumber_,
            const int kNumSamples_, 
            size_t *samples_)
        {
            // If there are not enough points in the pool, interrupt the procedure.
			if (kNumSamples_ > kPointNumber_)
				return false;

			// Generate a unique random set of indices.
			randomGenerator->generateUniqueRandomSet(samples_,
				kNumSamples_);
                
			return true;
        }
};

}
}