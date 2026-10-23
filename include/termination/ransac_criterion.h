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

#include "abstract_criterion.h"
#include "../utils/macros.h"
#include "../scoring/score.h"
#include "../utils/types.h"
#include <Eigen/Core>

namespace superansac {
namespace termination {

class RANSACCriterion : public AbstractCriterion
{
    protected:
        // Required confidence
        double confidence,
            logConfidence;

    public:
        // Constructor 
        RANSACCriterion() {}

        // Destructor
        ~RANSACCriterion() {}

        // Set the required confidence
        FORCE_INLINE void setConfidence(const double kConfidence_)
        {
            confidence = kConfidence_;
            logConfidence = log(1.0 - confidence);
        }

        // Sample function
        FORCE_INLINE void check(
            const DataMatrix &kData_, // Data matrix
            const scoring::Score &kScore_, // Score object
            const size_t &kSampleSize_, // Sample size
            size_t &iterationNumber_, /// Number of iterations
            bool &immediateTermination_) // Immediate termination flag
        {
            const double inlierRatio = static_cast<double>(kScore_.getInlierNumber()) /
                static_cast<double>(std::max<Eigen::Index>(1, kData_.rows()));
            if (inlierRatio <= 0.0)
            {
                iterationNumber_ = std::numeric_limits<size_t>::max();
                return;
            }
            if (inlierRatio >= 1.0 - 1e-9)
            {
                iterationNumber_ = 1;
                return;
            }
            const double q = std::pow(inlierRatio, static_cast<double>(kSampleSize_));
            if (q <= 1e-12)
            {
                iterationNumber_ = std::numeric_limits<size_t>::max();
                return;
            }
            const double log2 = std::log(1.0 - q);
            if (log2 >= -std::numeric_limits<double>::epsilon())
            {
                iterationNumber_ = std::numeric_limits<size_t>::max();
                return;
            }

            const double iter = logConfidence / log2;
            if (!std::isfinite(iter) || iter < 0.0 ||
                iter >= static_cast<double>(std::numeric_limits<size_t>::max() - 1))
            {
                iterationNumber_ = std::numeric_limits<size_t>::max();
                return;
            }
            iterationNumber_ = static_cast<size_t>(iter) + 1;
        }
       
};

}
}