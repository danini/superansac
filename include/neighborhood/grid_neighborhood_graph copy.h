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
#include "../utils/types.h"

namespace superansac 
{
	namespace neighborhood
	{
		// The cell structure used in the HashMap
		template <size_t _DimensionNumber>
		class GridCell {
		public:
			// The cell index along a particular axis
			const std::vector<size_t> idxAlongAxes;
			// The index of the cell used in the hashing function
			size_t index;

			GridCell(
				const std::vector<size_t>& idxAlongAxes_,
				const std::vector<size_t>& cell_number_along_axes) :
				idxAlongAxes(idxAlongAxes_)
			{
				size_t offset = 1;
				index = 0;
				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx) {
					index += offset * idxAlongAxes_[dimensionIdx];
					offset *= cell_number_along_axes[dimensionIdx];
				}
			}

			GridCell(
				const std::vector<size_t>& idxAlongAxes_,
				size_t cellNumberAlongAllAxes_) : // The number of cells along all axis in both images
				idxAlongAxes(idxAlongAxes_)
			{
				size_t offset = 1;
				index = 0;
				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx) {
					index += offset * idxAlongAxes_[dimensionIdx];
					offset *= cellNumberAlongAllAxes_;
				}
			}

			// Two cells are equal if their indices along all axes are equal.
			bool operator==(const GridCell& o) const {
				return idxAlongAxes == o.idxAlongAxes;
			}

			// The cells are ordered in ascending order along axes
			bool operator<(const GridCell& o) const {
				return idxAlongAxes < o.idxAlongAxes;
			}
		};
	}
}

namespace std {
	template <size_t _DimensionNumber>
	struct hash<superansac::neighborhood::GridCell<_DimensionNumber>>
	{
		std::size_t operator()(const superansac::neighborhood::GridCell<_DimensionNumber>& coord) const noexcept
		{
			// The cell's index value is used in the hashing function
			return coord.index;
		}
	};
}

namespace superansac 
{
	namespace neighborhood
	{
		// A sparse grid implementation. 
		// Sparsity means that only those cells are instantiated that contain elements. 
		template <size_t _DimensionNumber>
		class GridNeighborhoodGraph : public AbstractNeighborhoodGraph
		{
		protected:
			// The size of a cell along a particular dimension
			std::vector<double> cellSizes; 

			// Bounding box sizes
			std::vector<double> boundingBox;

			// Number of cells along the image axes.
			size_t cellNumberAlongAllAxes;

			// The grid is stored in the HashMap where the key is defined
			// via the cell coordinates.
			std::unordered_map<size_t, std::pair<std::vector<size_t>, std::vector<size_t>>> grid;

			// The pointer to the cell (i.e., key in the grid) for each point.
			// It is faster to store them than to recreate the cell structure
			// whenever is needed.
			std::vector<size_t> cellsOfPoints;

		public:
			// The default constructor of the neighborhood graph 
			GridNeighborhoodGraph() : 
                AbstractNeighborhoodGraph() 
            {
            }

			GridNeighborhoodGraph(const DataMatrix * const kContainer_) :
				AbstractNeighborhoodGraph(kContainer_)
			{
				// Checking if the dimension number of the graph and the points are the same. 
				if (kContainer_->cols() != _DimensionNumber)
                    throw std::invalid_argument("The data dimension does not match with the expected dimension.");
			}

			// The destructor of the neighborhood graph
			~GridNeighborhoodGraph() override = default;

            void setCellSizes(
				const std::vector<double> &cellSizes_, // The sizes of the cells along each axis
				const size_t &cellNumberAlongAllAxes_) // The number of cells along the axes
            {
                cellSizes = cellSizes_;
                cellNumberAlongAllAxes = cellNumberAlongAllAxes_;
            }

			// A function initializing the neighborhood graph given the data points
			bool initialize(const DataMatrix * const kContainer_);

			// A function returning the neighboring points of a particular one
			inline const std::vector<size_t> &getNeighbors(const size_t kPointIdx_) const;

			// A function returning the identifier of the cell in which the input point falls
			std::size_t getCellIdentifier(const size_t kPointIdx_) const;

			// A function returning all cells in the graph
			const std::unordered_map<size_t, std::pair<std::vector<size_t>, std::vector<size_t>>>& getCells() const;

			// A function returning the bounding box
			const std::vector<double> &getBoundingBox() const { return boundingBox; }

			// A function returning the cell sizes
			const std::vector<double> &getCellSizes() const { return cellSizes; }

			// The number of divisions/cells along an axis
			size_t getDivisionNumber() const { return cellNumberAlongAllAxes; }

			// A function returning the number of cells filled
			size_t filledCellNumber() const
			{
				return grid.size();
			}
		};

		// A function returning all cells in the graph
		template <size_t _DimensionNumber>
		const std::unordered_map<size_t, std::pair<std::vector<size_t>, std::vector<size_t>>>& GridNeighborhoodGraph<_DimensionNumber>::getCells() const
		{
			return grid;
		}

		// A function returning the identifier of the cell in which the input point falls
		template <size_t _DimensionNumber>
		std::size_t GridNeighborhoodGraph<_DimensionNumber>::getCellIdentifier(const size_t kPointIdx_) const
		{
			// Get the pointer of the cell in which the point is.
			return cellsOfPoints[kPointIdx_];
		}

		// A function initializing the neighborhood graph given the data points
		template <size_t _DimensionNumber>
		bool GridNeighborhoodGraph<_DimensionNumber>::initialize(const DataMatrix * const kContainer_)
		{
			// Initialize the bounding box
			boundingBox.resize(_DimensionNumber);
			for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx)
				boundingBox[dimensionIdx] = cellNumberAlongAllAxes * cellSizes[dimensionIdx] + std::numeric_limits<double>::epsilon();

			// The number of points
			const size_t pointNumber = kContainer_->rows();
			cellsOfPoints.resize(pointNumber);

			// Iterate through the points and put each into the grid.
			std::vector<size_t> indices(_DimensionNumber);
			for (size_t row = 0; row < pointNumber; ++row) {
				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx) {
					// Get the current value of the point along the current axis
					const double value = (*kContainer_)(row, dimensionIdx);

					if (value < 0)
						throw std::runtime_error("The grid neighborhood structure does not support negative values.");

					// The index of the cell along the current axis
					indices[dimensionIdx] =
						static_cast<size_t>(std::floor(value / cellSizes[dimensionIdx]));
				}

				// Constructing the cell structure which is used in the HashMap.
				const GridCell<_DimensionNumber> cell(indices, cellNumberAlongAllAxes); // The cell indices along the axes

				// Get the pointer of the current cell in the hashmap
				auto& cellValue = grid[cell.index];

				// If the cell did not exist.
				if (cellValue.second.empty())
					cellValue.second = cell.idxAlongAxes;

				// Add the current point's index to the grid.
				cellValue.first.push_back(row);
			}

			// Iterate through all cells and store the corresponding
			// cell pointer for each point.
			neighborNumber = 0;
			for (const auto& [key, value] : grid) {
				// Point container
				const auto& points = value.first;

				// Increase the edge number in the neighborhood graph.
				// All possible pairs of points in each cell are neighbors,
				// therefore, the neighbor number is "n choose 2" for the
				// current cell.
				const size_t n = points.size();
				neighborNumber += n * (n - 1) / 2;

				// Iterate through all points in the cell.
				for (const size_t kPointIdx : points)
					cellsOfPoints[kPointIdx] = key; // Store the cell pointer for each contained point.
			}

			// The graph is initialized
			initialized = !grid.empty();

			return initialized;
		}

		template <size_t _DimensionNumber>
		inline const std::vector<size_t> &GridNeighborhoodGraph<_DimensionNumber>::getNeighbors(const size_t kPointIdx_) const
		{
			// Get the pointer of the cell in which the point is.
			const size_t cell_idx = cellsOfPoints[kPointIdx_];
			// Return the vector containing all the points in the cell.
			return grid.at(cell_idx).first;
		}
	}
}