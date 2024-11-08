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

class NAPSACSampler : public AbstractSampler
{
    protected:
        std::unique_ptr<utils::UniformRandomGenerator<size_t>> randomGenerator;
        neighborhood::AbstractNeighborhoodGraph *neighborhood;
        const size_t kMaximumAttemps;
        size_t attempts;

    public:
        // Constructor 
        NAPSACSampler() : kMaximumAttemps(100) {}
        // Destructor
        ~NAPSACSampler() {}

        // Return the name of the sampler
        constexpr static const char *name()
        {
            return "NAPSACSampler";
        }

        void setNeighborhood(neighborhood::AbstractNeighborhoodGraph *kNeighborhood_)
        {
            neighborhood = kNeighborhood_;
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
			attempts = 0;
			while (attempts++ < kMaximumAttemps)
			{
				// Select a point randomly
				randomGenerator->generateUniqueRandomSet(samples_, // The sample to be selected
					1); // Only a single point is selected to be the center
                const auto &grid = neighborhood->getCells();

				// The indices of the points which are in the same cell as the
				// initially selected one.
				const std::vector<size_t> &neighbors =
					neighborhood->getNeighbors(samples_[0]);

				// Try again with another first point since the current one does not have enough neighbors
				if (neighbors.size() < kNumSamples_)
					continue;

				// If the selected point has just enough neighbors use them all.
				if (neighbors.size() == kNumSamples_)
				{
					for (size_t i = 0; i < kNumSamples_; ++i)
						samples_[i] = neighbors[i];
					break;	
				}

				// If the selected point has more neighbors than required select randomly.
				randomGenerator->generateUniqueRandomSet(samples_ + 1, // The sample to be selected
					kNumSamples_ - 1, // Only a single point is selected to be the center
					neighbors.size() - 1, // The index upper bound
					samples_[0]); // Index to skip
                    
				// Replace the indices reffering to neighbor to the ones that refer to points
				for (size_t i = 1; i < kNumSamples_; ++i)
					samples_[i] = neighbors[samples_[i]];
				break;
			}
			// Return true only if the iteration was interrupted due to finding a good sample
			return attempts < kMaximumAttemps;
        }
};

}
}