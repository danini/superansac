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

#include "scoring/types.h"
#include "samplers/types.h"
#include "neighborhood/types.h"
#include "local_optimization/types.h"
#include "termination/types.h"
#include "inlier_selectors/types.h"

namespace superansac {
    struct ARSamplerSettings
    {
        double estimatorVariance = 0.9765;
        double randomness = 0.01;
    };

    struct LocalOptimizationSettings
    {
        size_t maxIterations = 20,  // Reduced from 50 (diminishing returns after 20)
            graphCutNumber = 20;
        double sampleSizeMultiplier = 3.0;  // Reduced from 7.0 (3x sufficient for robustness)
        double spatialCoherenceWeight = 0.1;
    };

    struct NeighborhoodSettings
    {
        double neighborhoodSize = 20.0,
            neighborhoodGridDensity = 4.0;
        size_t nearestNeighborNumber = 6;
    };

    // DEGENSAC (Chum et al. 2005) dominant-plane degeneracy handling for
    // fundamental-matrix estimation. When enabled, every so-far-best
    // minimal-sample model is tested for H-degeneracy and re-estimated by the
    // plane-and-parallax algorithm if degenerate.
    struct DegensacSettings
    {
        bool enabled = true; // Enabled by default for F estimation
        double homographyThreshold = 2.0; // H-consistency threshold (pixels; scaled with the data normalization)
        size_t innerRansacIterations = 50; // Plane-and-parallax draws per degenerate event
    };

    struct RANSACSettings
    {
        size_t topKForLocalOptimization = 3, // Number of best models used for local optimization (reduced from 5)
            minIterations = 1000, // Minimum number of iterations
            maxIterations = 5000, // Maximum number of iterations
            finalRefinementRounds = 0; // Extra score-gated re-collect-and-refit rounds in the final LSQ optimization
        double finalRefinementScheduleStart = 3.0, // Widest sigma multiplier in the final-refinement schedule
            finalRefinementScheduleEnd = 0.5; // Narrowest sigma multiplier in the final-refinement schedule
        double inlierThreshold = 1.5; // Inlier threshold
        double confidence = 0.99; // Confidence
        bool localOptimizationInsideTheLoop = false, // Whether to locally optimize models when you find a new best or later
            useSprt = true, // Whether to use SPRT test to speed up
            fundamentalOrientationCheck = true, // Reject minimal F models violating the oriented epipolar constraint
            fundamentalSymmetricValidation = true, // Validate new-best F models with the symmetric epipolar distance
            finalRefinementSchedule = false, // Use a shrinking-threshold schedule in the final refinement rounds
            finalPolishTopK = false, // Run the final optimizer on every top-K LO candidate before selecting the best
            finalRefinementWeighted = false; // Feed per-point confidences into the final-refinement bundle adjustment
        size_t essentialCheiralityPointNumber = 1; // Cheirality-vote points for E pose disambiguation (raise to ~25 for dense/RoMA matches; splg gains nothing as cv2.recoverPose redoes cheirality)
        bool homographyBundleRefinement = false; // Nonlinear (LM) reprojection-error refinement of the non-minimal homography fit (H only)
        size_t homographyBundleMaxIterations = 5; // LM iterations: early stopping regularizes (full convergence over-fits the plane) and stays within the runtime budget
        
        scoring::ScoringType scoring = 
            scoring::ScoringType::MAGSAC; // Scoring type

        samplers::SamplerType sampler = 
            samplers::SamplerType::PROSAC; // Sampler type
            
        neighborhood::NeighborhoodType neighborhood = 
            neighborhood::NeighborhoodType::Grid; // Neighborhood type
            
        local_optimization::LocalOptimizationType localOptimization = 
            local_optimization::LocalOptimizationType::NestedRANSAC; // Local optimization type

        local_optimization::LocalOptimizationType finalOptimization = 
            local_optimization::LocalOptimizationType::IRLS; // Final optimization type

        termination::TerminationType terminationCriterion = 
            termination::TerminationType::RANSAC; // Termination criterion type

        inlier_selector::InlierSelectorType inlierSelector = 
            inlier_selector::InlierSelectorType::None; // Inlier selector type

        ARSamplerSettings arSamplerSettings;
        LocalOptimizationSettings localOptimizationSettings,
            finalOptimizationSettings;
        NeighborhoodSettings neighborhoodSettings;
        DegensacSettings degensacSettings;
    };

}