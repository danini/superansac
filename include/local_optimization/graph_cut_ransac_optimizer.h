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
#include "abstract_local_optimizer.h"
#include "../samplers/uniform_random_sampler.h"
#include "../neighborhood/abstract_neighborhood.h"
#include "../utils/types.h"
#include "GCoptimization.h"

namespace superansac
{
	namespace local_optimization
	{
		// Templated class for estimating a model for RANSAC. This class is purely a
		// virtual class and should be implemented for the specific task that RANSAC is
		// being used for. Two methods must be implemented: estimateModel and residual. All
		// other methods are optional, but will likely enhance the quality of the RANSAC
		// output.
		class GraphCutRANSACOptimizer : public LocalOptimizer
		{
		protected:
        	neighborhood::AbstractNeighborhoodGraph *neighborhoodGraph;
			size_t maxIterations,
				graphCutNumber,
				sampleSizeMultiplier;
			double spatialCoherenceWeight;

		public:
			GraphCutRANSACOptimizer() : maxIterations(50), 
				graphCutNumber(20),
				sampleSizeMultiplier(7)
			{

			}

			~GraphCutRANSACOptimizer() {}

			void setNeighborhood(neighborhood::AbstractNeighborhoodGraph *kNeighborhood_)
			{
				neighborhoodGraph = kNeighborhood_;
			}

			void setMaxIterations(const size_t kMaxIterations_)
			{
				maxIterations = kMaxIterations_;
			}

			void setGraphCutNumber(const size_t kGraphCutNumber_)
			{
				graphCutNumber = kGraphCutNumber_;
			}

			void setSampleSizeMultiplier(const size_t kSampleSizeMultiplier_)
			{
				sampleSizeMultiplier = kSampleSizeMultiplier_;
			}

			void setSpatialCoherenceWeight(const double kSpatialCoherenceWeight_)
			{
				spatialCoherenceWeight = kSpatialCoherenceWeight_;
			}

			// The function for estimating the model parameters from the data points.
			void run(const DataMatrix &kData_, // The data points
				const std::vector<size_t> &kInliers_, // The inliers of the previously estimated model
				const models::Model &kModel_, // The previously estimated model 
				const scoring::Score &kScore_, // The of the previously estimated model
				const estimator::Estimator *kEstimator_, // The estimator used for the model estimation
				const scoring::AbstractScoring *kScoring_, // The scoring object used for the model estimation
				models::Model &estimatedModel_, // The estimated model
				scoring::Score &estimatedScore_, // The score of the estimated model
				std::vector<size_t> &estimatedInliers_) const // The inliers of the estimated model
			{
				//std::cout <<  1 << std::endl;
				if (neighborhoodGraph == nullptr)
					throw std::runtime_error("The neighborhood graph is not set.");

				// The invalid score
				static const scoring::Score kInvalidScore = scoring::Score();

				// The sampler used for selecting minimal samples
				samplers::UniformRandomSampler sampler;

				// Initialize the estimated model and score
				estimatedModel_ = kModel_;

				// The size of the non-minimal samples
				const size_t kNonMinimalSampleSize = sampleSizeMultiplier * kEstimator_->sampleSize();
				size_t currentSampleSize;

				// The currently estimated models
				std::vector<models::Model> currentlyEstimatedModels;
				scoring::Score currentScore = kInvalidScore;
				std::vector<size_t> currentInliers,
					tmpInliers;
				currentInliers.reserve(kData_.rows());
				tmpInliers.reserve(kData_.rows());

				// Allocate memory for the current sample
				size_t *currentSample = new size_t[kNonMinimalSampleSize];

				// A flag indicating whether the model has been updated in the LO step
				bool updated;

				// Neighborhood properties
				const size_t &kNeighborNumber = neighborhoodGraph->getNeighborNumber();

				// The inlier-outlier threshold
				const double &kThreshold = kScoring_->getThreshold();
				//std::cout << 2 << std::endl;

				// The inner RANSAC loop
				for (size_t iteration = 0; iteration < graphCutNumber; ++iteration)
				{
					//std::cout << 3 << std::endl;
					// In the beginning, the best model is not updated
					updated = false;

					// Apply the graph-cut-based inlier/outlier labeling.
					// The inlier set will contain the points closer than the threshold and
					// their neighbors depending on the weight of the spatial coherence term.
					labeling(
						kData_, // The input points
						kNeighborNumber, // The number of neighbors, i.e. the edge number of the graph 
						estimatedModel_, // The best model parameters
						kEstimator_, // The model estimator
						spatialCoherenceWeight, // The weight of the spatial coherence term
						kThreshold, // The inlier-outlier threshold
						currentInliers); // The selected inliers
					//std::cout << 4 << std::endl;

					// Calculate the current sample size
					currentSampleSize = currentInliers.size() - 1;
					if (currentSampleSize >= kNonMinimalSampleSize)
						currentSampleSize = kNonMinimalSampleSize;

					// Break if the sample size is too small
					if (currentSampleSize < kEstimator_->sampleSize())
						break;
					//std::cout << 5 << std::endl;
						
					// Re-initialize the sampler with the current inliers
					sampler.initialize(currentInliers.size() - 1);
					//std::cout << 6 << std::endl;

					// Doing inner RANSACs using the current pool of potential inliers
					for (size_t innerIterations = 0; innerIterations < maxIterations; ++innerIterations)
					{
						//std::cout << 61 << std::endl;
						// Remove the previous models
						currentlyEstimatedModels.clear();
						// Add the previous model to the list of models so that if iterative optimization is applied, it can use it as a starting point
						currentlyEstimatedModels.emplace_back(estimatedModel_);

						// If there are enough inliers to estimate the model, use all of them
						if (currentSampleSize == currentInliers.size())
						{
							//std::cout << 62 << std::endl;
							// Estimate the model
							if (!kEstimator_->estimateModelNonminimal(
								kData_,  // The data points
								&currentInliers[0], // Selected minimal sample
								currentSampleSize, // The size of the minimal sample
								&currentlyEstimatedModels, // The estimated models
								nullptr)) // The indices of the inliers
								continue;
							//std::cout << 63 << std::endl;
						} else
						{
							//std::cout << 64 << std::endl;
							// Sample minimal set
							if (!sampler.sample(currentInliers.size(), // Data matrix
								currentSampleSize, // Selected minimal sample 
								currentSample)) // Sample indices
								continue;

							// Estimate the model
							if (currentSampleSize > kEstimator_->sampleSize())
								if (!kEstimator_->estimateModelNonminimal(
									kData_,  // The data points
									currentSample, // Selected minimal sample
									currentSampleSize, // The size of the minimal sample
									&currentlyEstimatedModels, // The estimated models
									nullptr)) // The indices of the inliers
									continue;
							else
								if (!kEstimator_->estimateModel(
									kData_,  // The data points
									currentSample, // Selected minimal sample
									&currentlyEstimatedModels)) // The estimated models
									continue;
							//std::cout << 65 << std::endl;
						}

						//std::cout << 66 << std::endl;
						// Calculate the scoring of the estimated model
						for (const auto &model : currentlyEstimatedModels)
						{
							// Calculate the score of the estimated model
							tmpInliers.clear();
							currentScore = kScoring_->score(kData_, model, kEstimator_, tmpInliers);

							// Check if the current model is better than the previous one
							if (currentScore > estimatedScore_)
							{
								// Update the estimated model
								estimatedModel_ = model;
								estimatedScore_ = currentScore;
								estimatedInliers_.swap(tmpInliers);
								updated = true;
							}
						}
						//std::cout << 67 << std::endl;
					}
					//std::cout << 7 << std::endl;

					// If the model is not updated, interrupt the procedure
					if (!updated)
						break;
				}
				//std::cout << 8 << std::endl;


				// Clean up
				delete[] currentSample;
			}

			// Returns a labeling w.r.t. the current model and point set
			void labeling(
				const DataMatrix &kData_, // The input data points
				size_t kNeighborNumber_, // The neighbor number in the graph
				const models::Model &kModel_, // The current model
				const estimator::Estimator *kEstimator_, // The estimator used for the model estimation
				const double kLambda_, // The weight for the spatial coherence term
				const double kThreshold_, // The kThreshold_ for the inlier-outlier decision
				std::vector<size_t> &inliers_) const // The resulting inlier set
			{
				//std::cout << 31 << std::endl;
				// The number of points in the data set
				const int &pointNumber = kData_.rows();

				// Initializing the problem graph for the graph-cut algorithm.
				Energy<double, double, double> *problemGraph =
					new Energy<double, double, double>(pointNumber, // The number of vertices
						kNeighborNumber_, // The number of edges
						NULL);

				// Add a vertex for each point
				for (auto i = 0; i < pointNumber; ++i)
					problemGraph->add_node();

				// The distance and energy for each point
				std::vector<double> distancePerThreshold;
				distancePerThreshold.reserve(pointNumber);
				double tmpSquaredDistance,
					tmpEnergy;
				const double squaredTruncatedThreshold = kThreshold_ * kThreshold_;
				const double oneMinusLambda = 1.0 - kLambda_;

				//std::cout << 32 << std::endl;
				// Estimate the vertex capacities
				for (size_t i = 0; i < pointNumber; ++i)
				{
					// Calculating the point-to-model squared residual
					tmpSquaredDistance = kEstimator_->squaredResidual(kData_.row(i),
						kModel_);
					// Storing the residual divided by the squared threshold 
					distancePerThreshold.emplace_back(
						std::min(std::max(tmpSquaredDistance / squaredTruncatedThreshold, 0.0), 1.0));
					// Calculating the implied unary energy
					tmpEnergy = 1.0 - distancePerThreshold.back();

					// Adding the unary energy to the graph
					if (tmpSquaredDistance <= squaredTruncatedThreshold)
						problemGraph->add_term1(i, oneMinusLambda * tmpEnergy, 0);
					else
						problemGraph->add_term1(i, 0, oneMinusLambda * (1 - tmpEnergy));
				}

				//std::cout << 33 << std::endl;
				std::vector<std::vector<int>> usedEdges(pointNumber, std::vector<int>(pointNumber, 0));

				if (kLambda_ > 0)
				{
					double energy1, energy2, energySum;
					double e00, e11 = 0; // Unused: e01 = 1.0, e10 = 1.0,

					// Iterate through all points and set their edges
					for (auto pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
					{
						energy1 = distancePerThreshold[pointIdx]; // Truncated quadratic cost

						// Iterate through  all neighbors
						const auto &neighbors = neighborhoodGraph->getNeighbors(pointIdx);
						for (const size_t &actualNeighborIdx : neighbors)
						{
							if (actualNeighborIdx == pointIdx)
								continue;

							if (actualNeighborIdx == pointIdx || actualNeighborIdx < 0)
								continue;

							if (usedEdges[actualNeighborIdx][pointIdx] == 1 ||
								usedEdges[pointIdx][actualNeighborIdx] == 1)
								continue;

							usedEdges[actualNeighborIdx][pointIdx] = 1;
							usedEdges[pointIdx][actualNeighborIdx] = 1;

							energy2 = distancePerThreshold[actualNeighborIdx]; // Truncated quadratic cost
							energySum = energy1 + energy2;

							e00 = 0.5 * energySum;

							constexpr double e01_plus_e10 = 2.0; // e01 + e10 = 2
							if (e00 + e11 > e01_plus_e10)
								throw std::runtime_error("Non-submodular expansion term detected; smooth costs must be a metric for expansion.\n");

							problemGraph->add_term2(pointIdx, // The current point's index
								actualNeighborIdx, // The current neighbor's index
								e00 * kLambda_,
								kLambda_, // = e01 * lambda
								kLambda_, // = e10 * lambda
								e11 * kLambda_);
						}
					}
				}
				//std::cout << 34 << std::endl;

				// Run the standard st-graph-cut algorithm
				problemGraph->minimize();
				//std::cout << 35<< std::endl;

				// Select the inliers, i.e., the points labeled as SINK.
				inliers_.reserve(pointNumber); 
				for (auto pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
					if (problemGraph->what_segment(pointIdx) == Graph<double, double, double>::SINK)
						inliers_.emplace_back(pointIdx);
				//std::cout << 36 << std::endl;

				// Clean the memory
				delete problemGraph;
				//std::cout << 37 << std::endl;
			} 
		};
	}
}  // namespace gcransac