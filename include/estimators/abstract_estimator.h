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
//     * Neither the name of Czech Technical University nor the
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
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <vector>
#include <Eigen/Core>
#include <memory>
#include "abstract_solver.h"
#include "../utils/macros.h"
#include "../utils/types.h"
#include "../models/model.h"

#include <iostream>

namespace superansac
{
	namespace estimator
	{
		// Templated class for estimating a model for RANSAC. This class is purely a
		// virtual class and should be implemented for the specific task that RANSAC is
		// being used for. Two methods must be implemented: estimateModel and residual. All
		// other methods are optional, but will likely enhance the quality of the RANSAC
		// output.
		class Estimator
		{
		protected:
			// The minimal solver used for estimating the model parameters
			std::unique_ptr<solver::AbstractSolver> minimalSolver;

			// The non-minimal solver used for estimating the model parameters
			std::unique_ptr<solver::AbstractSolver> nonMinimalSolver;

		public:
			Estimator() {}
			virtual ~Estimator() {}

			// Return the minimal solver
			const solver::AbstractSolver *getMinimalSolver() const
			{
				return minimalSolver.get();
			}

			// Return a mutable minimal solver
			solver::AbstractSolver *getMutableMinimalSolver()
			{
				return minimalSolver.get();
			}

			// Return the minimal solver
			const solver::AbstractSolver *getNonMinimalSolver() const
			{
				return nonMinimalSolver.get();
			}

			// Return a mutable minimal solver
			solver::AbstractSolver *getMutableNonMinimalSolver()
			{
				return nonMinimalSolver.get();
			}

			// Set the minimal solver
			void setMinimalSolver(solver::AbstractSolver *minimalSolver_)
			{
				minimalSolver.reset(minimalSolver_);
			}

			// Set the non-minimal solver
			void setNonMinimalSolver(solver::AbstractSolver *nonMinimalSolver_)
			{
				nonMinimalSolver.reset(nonMinimalSolver_);
			}

			// The size of a non-minimal sample required for the estimation
			size_t nonMinimalSampleSize() const
			{
				return nonMinimalSolver->sampleSize();
			}

			// The size of a minimal sample required for the estimation
			size_t sampleSize() const
			{
				return minimalSolver->sampleSize();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			virtual bool isWeightingApplicable() const = 0;

			// The size of a minimal sample_ required for the estimation
			size_t maximumMinimalSolutions() const 
			{
				return minimalSolver->maximumSolutions();
			}

			// Mult error for the AC-RANSAC scoring
            virtual double multError() const = 0;
			// Log Alpha 0 for the AC-RANSAC scoring
            virtual double logAlpha0(size_t w, size_t h, double scalingFactor = 0.5) const = 0;
			// Degrees of freedom for the MAGSAC++ scoring
			virtual size_t getDegreesOfFreedom() const = 0;

			// Given a set of data points, estimate the model. Users should implement this
			// function appropriately for the task being solved. Returns true for
			// successful model estimation (and outputs model), false for failed
			// estimation. Typically, this is a minimal set, but it is not required to be.
			FORCE_INLINE virtual bool estimateModel(
				const DataMatrix& kData_, // All data points
				const size_t *kSample_, // The indices of the selected points
				std::vector<models::Model>* model_) const = 0; // The estimated model parameters

			// Estimate a model from a non-minimal sampling of the data. E.g. for a line,
			// use SVD on a set of points instead of constructing a line from two points.
			// By default, this simply implements the minimal case.
			// In case of weighted least-squares, the weights can be fed into the
			// function.
			FORCE_INLINE virtual bool estimateModelNonminimal(
				const DataMatrix& kData_, // All data points
				const size_t *kSample_, // The indices of the selected points
				const size_t &kSampleNumber_, // The number of selected points
				std::vector<models::Model>* model_, // The estimated model parameters
				const double *kWeights_ = nullptr) const = 0; // The weights of the points

			// Given a model and a data point, calculate the error. Users should implement
			// this function appropriately for the task being solved.
			FORCE_INLINE virtual double residual(const DataMatrix& kData_, const models::Model& kModel_) const = 0;
			FORCE_INLINE virtual double squaredResidual(const DataMatrix& kData_, const models::Model& kModel_) const = 0;
			
			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			FORCE_INLINE virtual bool isValidSample(
				const DataMatrix& kData_, // All data points
				const size_t *kSample_) const // The indices of the selected points
			{
				return true;
			}

			// Enable a quick check to see if the model is valid. This can be a geometric
			// check or some other verification of the model structure.
			FORCE_INLINE virtual bool isValidModel(const models::Model& kModel_) const { return true; }

			// Enable a quick check to see if the model is valid. This can be a geometric
			// check or some other verification of the model structure.
			FORCE_INLINE virtual bool isValidModel(
				models::Model& kModel_, // The model parameters
				const DataMatrix& kData_, // All data points
				const size_t *kMinimalSample_, // The indices of the selected points
				const double kThreshold_, // The inlier-outlier threshold
				bool &modelUpdated_) const // The indicator whether the model has been updated
			{
				return true;
			}
		};
	}
}  // namespace gcransac