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

#include "abstract_neighborhood.h"
#include <vector>
#include <unordered_map>
#include <Eigen/Eigen>
#include "nanoflann.hpp"
#include "../utils/types.h"

namespace superansac 
{
	namespace neighborhood
	{
		// Template class definition with _DimensionNumber as the template parameter
		template <size_t _DimensionNumber, size_t _Type = 0> // _Type == 0: knn, _Type == 1: radius
		class FlannNeighborhoodGraph : public AbstractNeighborhoodGraph
		{
		protected:
			// Define a structure to hold the point cloud data
			struct DataPointCloud
			{
				// Pointer to an Eigen matrix containing the points
				const DataMatrix *points;

				// Function to return the number of points in the point cloud
				inline size_t kdtree_get_point_count() const { return points->rows(); }

				// Function to get a specific dimension of a point in the cloud
				inline double kdtree_get_pt(const size_t idx, int dim) const
				{
					return (*points)(idx, dim);
				}

				// Optional function to get the bounding box of the point cloud (not used here)
				template <class BBOX>
				bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
			};

			template <class _T, class _DataSource, typename _DistanceType = _T,
				typename IndexType = uint32_t>
			struct CustomMetricAdaptor
			{
				using ElementType  = _T;
				using DistanceType = _DistanceType;

				const _DataSource& kDataSource;

				CustomMetricAdaptor(const _DataSource& kDataSource_)
					: kDataSource(kDataSource_)
				{
				}

				inline DistanceType evalMetric(
					const _T* kA_, const IndexType kBIdx, size_t kSize_) const
				{
					DistanceType result = DistanceType();
					for (size_t i = 0; i < kSize_; ++i)
					{
						const DistanceType diff =
							kA_[i] - kDataSource.kdtree_get_pt(kBIdx, i);
						result += diff * diff;
					}
					return result;
				}

				template <typename U, typename V>
				inline DistanceType accum_dist(const U kA_, const V kB_, const size_t) const
				{
					return std::pow((kA_ - kB_), 2);
				}
			};

			// The data structure for the KD-tree
			DataPointCloud cloud;

			// The KD-tree index
			using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
				CustomMetricAdaptor<double, DataPointCloud>, 
				DataPointCloud,
				_DimensionNumber /* dim */
				>;

			// Unique pointer to the KD-tree index
			std::unique_ptr<my_kd_tree_t> index;

			// Mutable cache for storing neighbors
			mutable std::unordered_map<size_t, std::vector<size_t>> neighborsCache;

			// Vector to store cell sizes
			std::vector<double> cellSizes;

			// Map to store cells with their respective points
			std::unordered_map<size_t, std::pair<std::vector<size_t>, std::vector<size_t>>> cells;

			// Number of nearest neighbors to search for
			size_t nearestNeighborNumber;

			// Distance in the radius search
			double squaredRadius;

		public:
			// Default constructor initializing the base class and nearestNeighborNumber
			FlannNeighborhoodGraph() : AbstractNeighborhoodGraph(), nearestNeighborNumber(6) {}

			// Constructor that initializes with a given container
			FlannNeighborhoodGraph(const DataMatrix *const kContainer_) :
				AbstractNeighborhoodGraph(kContainer_)
			{
				initialize(kContainer_);
			}

			~FlannNeighborhoodGraph()
			{
				// Destructor
			}

			// Setter for nearestNeighborNumber
			void setNearestNeighborNumber(const size_t nearestNeighborNumber_) 
			{
				nearestNeighborNumber = nearestNeighborNumber_;
			}

			void setRadius(const double radius_)
			{
				squaredRadius = radius_ * radius_;
			}

			// Function to initialize the KD-tree with the given container
			bool initialize(const DataMatrix *const kContainer_) override
			{
				// Return false if the container is null
				if (!kContainer_)
					throw std::runtime_error("The container is null.");

				// Set the point cloud data
				cloud.points = kContainer_;

				// Reset and build the KD-tree index
				index.reset(new my_kd_tree_t(_DimensionNumber,  /* dim */
					cloud, 
					{10})); /* max leaf */

				// Set the initialized flag to true
				initialized = true;
				return true;
			}

			// Function to get the neighbors of a given point
			const std::vector<size_t> &getNeighbors(const size_t pointIdx_) const override
			{
				// Check if neighbors are already cached and return directly
				auto it = neighborsCache.find(pointIdx_);
				if (it != neighborsCache.end())
				{
					return it->second;
				}

				// If the index is not initialized, throw an error
				const auto &point = cloud.points->row(pointIdx_);

				// Cache the neighbors in the mutable map using emplace for efficiency
				auto &neighbors = neighborsCache[pointIdx_];

				// If _Type is 0, do a knn search
				if constexpr (_Type == 0)
				{
					// Find the k nearest neighbors
					neighbors.resize(nearestNeighborNumber);
					std::vector<double> outDistSqr(nearestNeighborNumber);
					nanoflann::KNNResultSet<double> resultSet(nearestNeighborNumber);
					resultSet.init(&neighbors[0], &outDistSqr[0]);
					index->findNeighbors(resultSet, point.data());
				} else if (_Type == 1) // If _Type is 1, do a radius search
				{
        			std::vector<nanoflann::ResultItem<size_t, double>> indicesDists;
        			nanoflann::RadiusResultSet<double, size_t> resultSet(squaredRadius, indicesDists);
					index->findNeighbors(resultSet, point.data());
					// Reserve space to avoid multiple allocations
					neighbors.reserve(indicesDists.size());
					for (size_t i = 0; i < indicesDists.size(); ++i)
						neighbors.emplace_back(indicesDists[i].first);
				} else
					throw std::runtime_error("Invalid type.");

				// Return the neighbors
				return neighbors;
			}

			// Function to get the cell sizes (not implemented, returns member variable)
			const std::vector<double> &getCellSizes() const override
			{
				return cellSizes;
			}

			// Function to get the cells (not implemented, returns member variable)
			const std::unordered_map<size_t, std::pair<std::vector<size_t>, std::vector<size_t>>> &getCells() const override
			{
				return cells;
			}

			// Function to get the number of divisions (size of cellSizes vector)
			size_t getDivisionNumber() const override
			{
				return cellSizes.size();
			}

			// Function to get the number of filled cells (size of cells map)
			size_t filledCellNumber() const override
			{
				return cells.size();
			}
		};

}
}