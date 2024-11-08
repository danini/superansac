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

#include "utils/macros.h"

#include "abstract_scoring.h"
#include "ransac_scoring.h"
#include "msac_scoring.h"
#include "magsac_scoring.h"
#include "minpran_scoring.h"
#include "acransac_scoring.h"
#include "grid_scoring.h"
#include "gau_scoring.h"
#include "ml_scoring.h"

namespace superansac {
namespace scoring {

    // Enum defining available sampler types
    enum class ScoringType {
        RANSAC,
        MSAC,
        MAGSAC,
        MINPRAN,
        ACRANSAC,
        GAU,
        ML,
        GRID
    };

    // Factory function to create samplers
    template <size_t _DimensionNumber>
    FORCE_INLINE std::unique_ptr<AbstractScoring> createScoring(const ScoringType kType_) 
    {
        switch (kType_) 
        {
            case ScoringType::RANSAC:
                return std::make_unique<RANSACScoring>();
            case ScoringType::MSAC:
                return std::make_unique<MSACScoring>();
            case ScoringType::MINPRAN:
                return std::make_unique<MINPRANScoring>();
            case ScoringType::ACRANSAC:
                return std::make_unique<ACRANSACScoring>();
            case ScoringType::GAU:
                return std::make_unique<GAUScoring>();
            case ScoringType::ML:
                return std::make_unique<MLScoring>();
            case ScoringType::GRID:
                return std::make_unique<GridScoring<_DimensionNumber>>();
            case ScoringType::MAGSAC:
                return std::make_unique<MAGSACScoring>();
            default:
                throw std::invalid_argument("Unknown Sampler Type");
        }
    }

}
}