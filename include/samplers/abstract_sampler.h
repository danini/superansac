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
#include <vector>
#include <Eigen/Core>
#include "../utils/types.h"

namespace superansac {
namespace samplers {

class AbstractSampler
{
    public:
        // Constructor 
        AbstractSampler() {}

        // Destructor
        virtual ~AbstractSampler() {}

        // Sample function
        FORCE_INLINE virtual bool sample(
            const DataMatrix &kData_, // Data matrix
            const int kNumSamples_, // Number of samples
            size_t *kSamples_) = 0; // Sample indices

        // Initialize function
        FORCE_INLINE virtual void initialize(
            const DataMatrix &kData_) = 0; // Data matrix
            
        // Sample function
        FORCE_INLINE virtual bool sample(
            const size_t kPointNumber_, // Data matrix
            const int kNumSamples_, // Number of samples
            size_t *kSamples_) = 0; // Sample indices

        // Initialize function
        FORCE_INLINE virtual void initialize(
            const size_t kPointNumber_) = 0; // Data matrix
            
        // Update function
        FORCE_INLINE virtual void update(
            const size_t* const kSample_,
            const size_t& kSampleSize_,
            const size_t& kIterationNumber_,
            const double& kInlierRatio_) = 0;
};

}
}