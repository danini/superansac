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

#include <Eigen/Core>

#include "estimators/abstract_estimator.h"
#include "samplers/abstract_sampler.h"
#include "scoring/abstract_scoring.h"
#include "local_optimization/abstract_local_optimizer.h"
#include "termination/abstract_criterion.h"
#include "settings.h"
#include "utils/types.h"

namespace superansac {

class SupeRansac
{
    public:
        SupeRansac();
        virtual ~SupeRansac();

        void run(const DataMatrix &kData_);

        // Create the space partitioning inlier selector object
        void setInlierSelector(inlier_selector::AbstractInlierSelector *inlierSelector_);

        // Return a constant pointer to the space partitioning inlier selector object
        const inlier_selector::AbstractInlierSelector *getInlierSelector() const;

        // Return a mutable pointer to the space partitioning inlier selector object
        inlier_selector::AbstractInlierSelector *getMutableInlierSelector();

        // Set the scoring object
        void setTerminationCriterion(termination::AbstractCriterion *terminationCriterion_);

        // Return a constant pointer to the scoring object
        const termination::AbstractCriterion *getTerminationCriterion() const;

        // Return a mutable pointer to the scoring object
        termination::AbstractCriterion *getMutableTerminationCriterion();

        // Set the scoring object
        void setScoring(scoring::AbstractScoring *sampler_);

        // Return a constant pointer to the scoring object
        const scoring::AbstractScoring *getScoring() const;

        // Return a mutable pointer to the scoring object
        scoring::AbstractScoring *getMutableScoring();

        // Set the local optimization object
        void setFinalOptimizer(local_optimization::LocalOptimizer *finalOptimizer_);

        // Return a constant pointer to the local optimization object
        const local_optimization::LocalOptimizer *getFinalOptimizer() const;

        // Return a mutable pointer to the local optimization object
        local_optimization::LocalOptimizer *getMutableFinalOptimizer();

        // Set the local optimization object
        void setLocalOptimizer(local_optimization::LocalOptimizer *localOptimizer_);

        // Return a constant pointer to the local optimization object
        const local_optimization::LocalOptimizer *getLocalOptimizer() const;

        // Return a mutable pointer to the local optimization object
        local_optimization::LocalOptimizer *getMutableLocalOptimizer();

        // Set the sampler
        void setSampler(samplers::AbstractSampler *sampler_);

        // Return a constant pointer to the sampler
        const samplers::AbstractSampler *getSampler() const;

        // Return a mutable pointer to the sampler
        samplers::AbstractSampler *getMutableSampler();

        // Set the settings
        void setSettings(const RANSACSettings &settings_);

        // Return the settings
        const RANSACSettings &getSettings() const;

        // Return a mutable reference to the settings
        RANSACSettings &getMutableSettings();

        // Set the estimator
        void setEstimator(estimator::Estimator *kEstimator_);

        // Return a constant pointer to the estimator
        const estimator::Estimator *getEstimator() const;

        // Return a mutable pointer to the estimator
        estimator::Estimator *getMutableEstimator();

        // Get the best model
        const models::Model &getBestModel() const;

        // Get the inliers of the best model
        const std::vector<size_t> &getInliers() const;

        // Get the score of the best model
        const scoring::Score &getBestScore() const;

        // Get the number of iterations
        size_t getIterationNumber() const;

    protected:
        // The sampler for selecting the minimal samples
        samplers::AbstractSampler *sampler;

        // The estimator object
        estimator::Estimator *estimator;

        // The scoring object
        scoring::AbstractScoring *scoring;

        // The local optimization object
        local_optimization::LocalOptimizer *localOptimizer,
            *finalOptimizer;

        // The termination criterion object
        termination::AbstractCriterion *terminationCriterion;

        // The space partitioning inlier selector object
        inlier_selector::AbstractInlierSelector *inlierSelector;

        // The settings for RANSAC
        RANSACSettings settings;

        // Variables used during the RANSAC process
        std::vector<size_t> inliers, // The inliers of the best model
            tmpInliers; // The inliers of the current model
        size_t iterationNumber, // The number of iterations
            minIterations, // The minimum number of iterations
            maxIterations; // The maximum number of iterations
        size_t *currentSample; // The current minimal sample
        std::vector<models::Model> currentModels; // The current models
        scoring::Score currentScore, // The score of the current model
            bestScore; // The score of the best model
        models::Model bestModel, // The best model
            locallyOptimizedModel; // The locally optimized model
};

}
