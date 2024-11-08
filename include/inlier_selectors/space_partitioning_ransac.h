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
#include <Eigen/Eigen>
#include <unordered_map>
#include "types.h"
#include "../neighborhood/abstract_neighborhood.h"
#include "../utils/macros.h"
#include "../models/model.h"
#include "../scoring/score.h"
#include "../utils/types.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace superansac 
{
	namespace inlier_selector
	{
		class SpacePartitioningRANSAC : public AbstractInlierSelector
		{
        protected:
            std::vector<bool> gridCornerMask;
            std::vector<bool> gridAngleMask;
	        std::vector<std::tuple<double, double, double, double>> gridCornerCoordinates;
            std::vector<std::tuple<int, int, double, double>> gridCornerCoordinatesH;
            std::vector<std::tuple<int, int, int, double, double, double>> gridCornerCoordinates3D;
	        std::vector<std::tuple<double, double, double, double, double, double>> gridCornerAngles;
            std::vector<double> additionalParameters;
            double scaleSrc, offsetXSrc, offsetYSrc, scaleDst, offsetXDst, offsetYDst;

			neighborhood::AbstractNeighborhoodGraph* neighborhoodGraph;

            FORCE_INLINE void runHomography(
				const DataMatrix &kData_,
				const models::Model &kModel_,
                const double& kInlierOutlierThreshold_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_);
            
            FORCE_INLINE void runFundamentalMatrix(
				const DataMatrix &kData_,
				const models::Model &kModel_,
                const double& kInlierOutlierThreshold_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_);

		public:
            cv::Mat img1, img2;

			SpacePartitioningRANSAC() : scaleSrc(1.0), offsetXSrc(0.0), offsetYSrc(0.0), scaleDst(1.0), offsetXDst(0.0), offsetYDst(0.0)
			{
			}

            ~SpacePartitioningRANSAC() {}

            void setNormalizers(
                const double &kScaleSrc_,
                const double &kOffsetXSrc_,
                const double &kOffsetYSrc_,
                const double &kScaleDst_,
                const double &kOffsetXDst_,
                const double &kOffsetYDst_)
            {
                scaleSrc = kScaleSrc_;
                offsetXSrc = kOffsetXSrc_;
                offsetYSrc = kOffsetYSrc_;
                scaleDst = kScaleDst_;
                offsetXDst = kOffsetXDst_;
                offsetYDst = kOffsetYDst_;
            }

			void initialize(
				neighborhood::AbstractNeighborhoodGraph* neighborhoodGraph_,
				const models::Types kModelType_)
			{
				neighborhoodGraph = neighborhoodGraph_;
				modelType = kModelType_;

                // Save additional info needed for the selection
                const auto& kSizes = neighborhoodGraph_->getCellSizes();
                // Number of dimensions
                const size_t& kDimensions = kSizes.size();

                // The number cells filled in the grid
                const size_t& kCellNumber = neighborhoodGraph_->filledCellNumber();
                const size_t& kDivisionNumber = neighborhoodGraph_->getDivisionNumber();
                const size_t kMaximumCellNumber = std::pow(kDivisionNumber, kDimensions);

                // Initialize the structures speeding up the selection by caching data
                gridCornerMask.resize(kMaximumCellNumber, false);

                switch (kModelType_)
                {
                case models::Types::Homography:
                    gridCornerCoordinates.resize(kMaximumCellNumber);
                    break;
                case models::Types::FundamentalMatrix:
		            gridCornerAngles.resize(kMaximumCellNumber);
                    gridCornerCoordinates.resize(kMaximumCellNumber);
                    gridCornerMask.resize(kMaximumCellNumber, false);
                    gridAngleMask.resize(kMaximumCellNumber, false);
                    break;
                }
                
                /*if (kDimensions == 6)
                    gridCornerCoordinates3D.resize(kMaximumCellNumber);
                else
                    gridCornerCoordinatesH.resize(kMaximumCellNumber);*/

                additionalParameters.resize(kDimensions + 1);
                for (size_t dimension = 0; dimension < kDimensions; ++dimension)
                    additionalParameters[dimension] = kSizes[dimension]; // The cell size along the current dimension
                additionalParameters[kDimensions] = kDivisionNumber; // The number of cells along an axis 
			}

            // The function that runs the model-based inlier selector
            void run(
                const DataMatrix &kData_, // The data points
                const models::Model &kModel_, // The model estimated
                const scoring::AbstractScoring *kScoring_, // The scoring object used for the model estimation
                std::vector<const std::vector<size_t>*>& selectedCells_, // The indices of the points selected
                size_t& pointNumber_) // The number of points selected
            {
                // Initializing the selected point number to zero
                pointNumber_ = 0;

                // Get the current inlier-outlier threshold
                const double& kInlierOutlierThreshold_ = 3.0 / 2.0 * kScoring_->getThreshold();

                if (modelType == models::Types::Homography)
                    runHomography(
                        kData_,
                        kModel_,
                        kInlierOutlierThreshold_,
                        selectedCells_,
                        pointNumber_);
                else if (modelType == models::Types::FundamentalMatrix)
                    runFundamentalMatrix(
                        kData_,
                        kModel_,
                        kInlierOutlierThreshold_,
                        selectedCells_,
                        pointNumber_);
                else
                    throw std::runtime_error("The estimator type is not supported by the space partitioning RANSAC.");

                /*else if constexpr (std::is_same<_Estimator, gcransac::utils::DefaultRigidTransformationEstimator>())
                    runRigidTransformation(
                        kCorrespondences_,
                        kModel_,
                        kNeighborhood_,
                        selectedCells_,
                        pointNumber_,
                        kInlierOutlierThreshold_);
                else  // Other then homography estimation is not implemented yet. 
                {
                    // Return all points
                    const auto& cellMap = neighborhoodGraph->getCells();
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [cell, value] : cellMap)
                    {
                        const auto& points = value.first;
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                    }
                }*/
            }

            /*void runRigidTransformation(
				const DataMatrix &kData_,
				const models::Model &kModel_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_,
                const double& kInlierOutlierThreshold_);*/
				
		};
        
        FORCE_INLINE void SpacePartitioningRANSAC::runHomography(
            const DataMatrix &kData_,
            const models::Model &kModel_,
            const double& kInlierOutlierThreshold_,
            std::vector<const std::vector<size_t>*>& selectedCells_,
            size_t& pointNumber_)
        {
            
        }

        FORCE_INLINE void SpacePartitioningRANSAC::runFundamentalMatrix(
            const DataMatrix &kData_,
            const models::Model &kModel_,
            const double& kInlierOutlierThreshold_,
            std::vector<const std::vector<size_t>*>& selectedCells_,
            size_t& pointNumber_)
        {
            // The actual descriptor of the model
            const auto &kModelData = kModel_.getData();

            const double& kSourceCellWidth = additionalParameters[0], // The width of the source image
                & kSourceCellHeight = additionalParameters[1], // The height of the source images
                & kDestinationCellWidth = additionalParameters[2], // The width of the destination image
                & kDestinationCellHeight = additionalParameters[3], // The height of the destination image
                & kPartitionNumber = additionalParameters[4]; // The number of cells in the neighborhood structure along an axis

            // The sizes of the cells along each axis
            const double kCellSize1 = kSourceCellWidth * scaleSrc, 
                kCellSize2 = kSourceCellHeight * scaleSrc,
                kCellSize3 = kDestinationCellWidth * scaleDst,
                kCellSize4 = kDestinationCellHeight * scaleDst;

            // Calculate the normalized image corners
            double normDstX0 = offsetXDst,
                normDstX1 = kCellSize3 * kPartitionNumber + offsetXDst,
                normDstY0 = offsetYDst,
                normDstY1 = kCellSize4 * kPartitionNumber + offsetYDst;

            //std::cout << normDstX0 << " " << normDstX1 << " " << normDstY0 << " " << normDstY1 << std::endl;

            // Iterate through all cells and project their corners to the second image
            const static std::vector<int> steps = { 0, 0,
                0, 1,
                1, 0,
                1, 1 };

            // Filling the vectors with zeros
            std::fill(std::begin(gridCornerMask), std::end(gridCornerMask), 0);
            std::fill(std::begin(gridAngleMask), std::end(gridAngleMask), 0);

            // Iterating through all cells in the neighborhood graph
            for (const auto& [kCell, kValue] : neighborhoodGraph->getCells())
            {
                const auto& kPoints = kValue.first;

                // Checking if there are enough points in the cell to make the cell selection worth it.
                if (kPoints.size() < 20)
                {
                    // If not, simply test all points from the cell and continue
                    selectedCells_.emplace_back(&kPoints);
                    pointNumber_ += kPoints.size();
                    continue;
                }

                // The coordinates of the cell corners.
                const auto& kCornerIndices = kValue.second;
                // The index of the cell in the source image.
                size_t cellIdx = kCornerIndices[0] * kPartitionNumber + kCornerIndices[1];
                
                // The parameters of the epipolar lines corresponding to the minimum and maximum angles
                double &A1 = std::get<0>(gridCornerAngles[cellIdx]), 
                    &B1 = std::get<1>(gridCornerAngles[cellIdx]),
                    &C1 = std::get<2>(gridCornerAngles[cellIdx]),
                    &A2 = std::get<3>(gridCornerAngles[cellIdx]), 
                    &B2 = std::get<4>(gridCornerAngles[cellIdx]), 
                    &C2 = std::get<5>(gridCornerAngles[cellIdx]);

                // Iterate through the corners of the current cell
                // TODO(danini): Handle the case when the epipole falls inside the image
                if (!gridAngleMask[cellIdx])
                {
                    bool insideImage = false;
                    double minAngle = std::numeric_limits<double>::max(),
                        maxAngle = std::numeric_limits<double>::lowest();
                    for (size_t stepIdx = 0; stepIdx < 8; stepIdx += 2)
                    {
                        // Stepping in the current direction from the currently selected corner
                        size_t colIdx = kCornerIndices[0] + steps[stepIdx],
                            rowIdx = kCornerIndices[1] + steps[stepIdx + 1];
                            
                        if (rowIdx >= kPartitionNumber ||
                            colIdx >= kPartitionNumber)
                            continue;

                        // Calculating the selected cell's index
                        const size_t kIdx2d = rowIdx * kPartitionNumber + colIdx;

                        // Get the index of the corner's projection in the destination image
                        auto& lineTuple = gridCornerCoordinates[kIdx2d];
                        auto& angle = std::get<0>(lineTuple);
                        auto& a2 = std::get<1>(lineTuple);
                        auto& b2 = std::get<2>(lineTuple);
                        auto& c2 = std::get<3>(lineTuple);

                        // If the corner hasn't yet been projected to the destination image
                        if (!gridCornerMask[kIdx2d])
                        {
                            // Get the coordinates of the corner
                            double kX1 = colIdx * kCellSize1 + offsetXSrc,
                                kY1 = rowIdx * kCellSize2 + offsetYSrc;

                            // Move the corner by the threshold
                            if (stepIdx == 0)
                            {
                                kX1 -= kInlierOutlierThreshold_;
                                kY1 -= kInlierOutlierThreshold_;
                            }
                            else if (stepIdx == 2)
                            {
                                kX1 -= kInlierOutlierThreshold_;
                                kY1 += kInlierOutlierThreshold_;
                            }
                            else if (stepIdx == 4)
                            {
                                kX1 += kInlierOutlierThreshold_;
                                kY1 -= kInlierOutlierThreshold_;
                            }
                            else
                            {
                                kX1 += kInlierOutlierThreshold_;
                                kY1 += kInlierOutlierThreshold_;
                            }

                            // Project them by the estimated homography matrix
                            a2 = kX1 * kModelData(0, 0) + kY1 * kModelData(0, 1) + kModelData(0, 2);
                            b2 = kX1 * kModelData(1, 0) + kY1 * kModelData(1, 1) + kModelData(1, 2);
                            c2 = kX1 * kModelData(2, 0) + kY1 * kModelData(2, 1) + kModelData(2, 2);

                            // Angle of direction vector v
                            // n = [a2, b2]
                            // v = [-b2, a2]
                            angle = std::atan2(a2, b2);

                            // Note that the corner has been already projected
                            gridCornerMask[kIdx2d] = true;

                            // Check if the line intersects the image borders
                            // a * x + b * y + c
                            if (!insideImage)
                            {
                                double yLeft = -(a2 * normDstX0 + c2) / b2;
                                double yRight = -(a2 * normDstX1 + c2) / b2;

                                if (yLeft >= normDstY0 && yLeft <= normDstY1 ||
                                    yRight >= normDstY0 && yRight <= normDstY1)
                                    insideImage = true;
                                else
                                {
                                    double xTop = -(b2 * normDstY0 + c2) / a2;
                                    double xBottom = -(b2 * normDstY1 + c2) / a2;

                                    if (xTop >= normDstX0 && xTop <= normDstX1 ||
                                        xBottom >= normDstX0 && xBottom <= normDstX1)
                                        insideImage = true;
                                }
                            }
                        }

                        // Save the epipolar line's parameters if the angle is the new mininum
                        if (angle < minAngle)
                        {
                            minAngle = angle;
                            A1 = a2;
                            B1 = b2;
                            C1 = c2;
                        }

                        // Save the epipolar line's parameters if the angle is the new maximum
                        if (angle > maxAngle)
                        {
                            maxAngle = angle;
                            A2 = a2;
                            B2 = b2;
                            C2 = c2;
                        }
                    }

                    // Note that the cell has already been processed
                    gridAngleMask[cellIdx] = true;

                    if (!insideImage)
                        continue;
                }

                // Iterate through the corners of the cell and check if any of the corners fall 
                // between the two selected epipolar lines.
                double distance1, distance2;
                bool overlaps = false;
                for (size_t stepIdx = 0; stepIdx < 8; stepIdx += 2)
                {
                    // Get the coordinates of the corner
                    double kX2 = (kCornerIndices[2] + steps[stepIdx]) * kCellSize3 + offsetXDst,
                        kY2 = (kCornerIndices[3] + steps[stepIdx + 1]) * kCellSize4 + offsetYDst;

                    distance1 = A1 * kX2 + B1 * kY2 + C1;
                    distance2 = A2 * kX2 + B2 * kY2 + C2;

                    // If the distance sign is different, the point falls between the lines.
                    if (distance1 * distance2 <= 0)
                    {
                        overlaps = true;
                        break;
                    }
                }

                if (!overlaps)
                {
                    // Get the coordinates of the corner
                    double kX20 = (kCornerIndices[2]) * kCellSize3 + offsetXDst,
                        kY20 = (kCornerIndices[3]) * kCellSize4 + offsetYDst,
                        kX21 = (kCornerIndices[2] + 1) * kCellSize3 + offsetXDst,
                        kY21 = (kCornerIndices[3] + 1) * kCellSize4 + offsetYDst;

                    distance1 = A1 * kX20 + B1 * kY20 + C1;
                    distance2 = A1 * kX21 + B1 * kY20 + C1;

                    if (distance1 * distance2)
                        overlaps = true;
                    else
                    {
                        distance2 = A1 * kX20 + B1 * kY21 + C1;
                        if (distance1 * distance2)
                            overlaps = true;
                        else
                        {
                            distance1 = A2 * kX20 + B2 * kY20 + C2;
                            distance2 = A2 * kX21 + B2 * kY20 + C2;

                            if (distance1 * distance2)
                                overlaps = true;
                            else
                            {
                                distance2 = A2 * kX20 + B2 * kY21 + C2;
                                if (distance1 * distance2)
                                    overlaps = true;
                            }
                        }
                    }
                }

                if (overlaps)
                {
                    // Store the points in the cell to be tested
                    selectedCells_.emplace_back(&kPoints);
                    pointNumber_ += kPoints.size();
                }
            }
        }

        /*template <typename _Estimator,
            typename _NeighborhoodStructure>
            OLGA_INLINE void SpacePartitioningRANSAC<_Estimator, _NeighborhoodStructure>::runRigidTransformation(
                const cv::Mat& kCorrespondences_,
                const gcransac::Model& kModel_,
                const _NeighborhoodStructure& kNeighborhood_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_,
                const double& kInlierOutlierThreshold_)
        {
            const double& kCellSize1 = additionalParameters[0],
                & kCellSize2 = additionalParameters[1],
                & kCellSize3 = additionalParameters[2],
                & kCellSize4 = additionalParameters[3],
                & kCellSize5 = additionalParameters[4],
                & kCellSize6 = additionalParameters[5],
                & kPartitionNumber = additionalParameters[6];

            const Eigen::Matrix4d& descriptor = kModelData;

            // Iterate through all cells and project their corners to the second image
            const static std::vector<int> steps = {
                0, 0, 0,
                0, 1, 0,
                1, 0, 0,
                1, 1, 0,
                0, 0, 1,
                0, 1, 1,
                1, 0, 1,
                1, 1, 1 };

            std::fill(std::begin(gridCornerMask), std::end(gridCornerMask), 0);

            // Iterating through all cells in the neighborhood graph
            for (const auto& [cell, value] : neighborhoodGraph->getCells())
            {
                // The points in the cell
                const auto& points = value.first;

                // Checking if there are enough points in the cell to make the cell selection worth it
                if (points.size() < 8)
                {
                    // If not, simply test all points from the cell and continue
                    selectedCells_.emplace_back(&points);
                    pointNumber_ += points.size();
                    continue;
                }

                const auto& kCornerIndices = value.second;
                bool overlaps = false;

                // Iterate through the corners of the current cell
                for (size_t stepIdx = 0; stepIdx < 24; stepIdx += 3)
                {
                    // The index of the currently projected corner
                    const size_t kCornerXIndex = kCornerIndices[0] + steps[stepIdx];
                    if (kCornerXIndex >= kPartitionNumber)
                        continue;

                    const size_t kCornerYIndex = kCornerIndices[1] + steps[stepIdx + 1];
                    if (kCornerYIndex >= kPartitionNumber)
                        continue;

                    const size_t kCornerZIndex = kCornerIndices[2] + steps[stepIdx + 2];
                    if (kCornerZIndex >= kPartitionNumber)
                        continue;

                    // Get the index of the corner's projection in the destination image
                    const size_t kIdx3d = kCornerXIndex * kPartitionNumber * kPartitionNumber + kCornerYIndex * kPartitionNumber + kCornerYIndex;

                    // This is already or will be the horizontal and vertical indices in the destination image
                    auto& indexPair = gridCornerCoordinates3D[kIdx3d];

                    // If the corner hasn't yet been projected to the destination image
                    if (!gridCornerMask[kIdx3d])
                    {
                        // Get the coordinates of the corner
                        const double kX1 = kCornerXIndex * kCellSize1,
                            kY1 = kCornerYIndex * kCellSize2,
                            kZ1 = kCornerZIndex * kCellSize3;

                        const double x2p = descriptor(0, 0) * kX1 + descriptor(1, 0) * kY1 + descriptor(2, 0) * kZ1 + descriptor(3, 0),
                            y2p = descriptor(0, 1) * kX1 + descriptor(1, 1) * kY1 + descriptor(2, 1) * kZ1 + descriptor(3, 1),
                            z2p = descriptor(0, 2) * kX1 + descriptor(1, 2) * kY1 + descriptor(2, 2) * kZ1 + descriptor(3, 2);

                        // Store the projected corner's cell indices
                        std::get<0>(indexPair) = x2p / kCellSize4;
                        std::get<1>(indexPair) = y2p / kCellSize5;
                        std::get<2>(indexPair) = z2p / kCellSize6;
                        std::get<3>(indexPair) = x2p;
                        std::get<4>(indexPair) = y2p;
                        std::get<5>(indexPair) = z2p;

                        // Note that the corner has been already projected
                        gridCornerMask[kIdx3d] = true;
                    }

                    // Check if the projected corner is equal to the correspondence's destination point's grid cell.
                    // This works due to the coordinate truncation.
                    if (std::get<0>(indexPair) == kCornerIndices[3] &&
                        std::get<1>(indexPair) == kCornerIndices[4] &&
                        std::get<2>(indexPair) == kCornerIndices[5])
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                        break;
                    }
                }

                // Check if there is an overlap
                if (!overlaps)
                {
                    // The X index of the bottom-right corner
                    const size_t kCornerXIndex111 = kCornerIndices[0] + 1;
                    // The Y index of the bottom-right corner
                    const size_t kCornerYIndex111 = kCornerIndices[1] + 1;
                    // The Z index of the bottom-right corner
                    const size_t kCornerZIndex111 = kCornerIndices[2] + 1;

                    // Calculating the index of the top-left corner
                    const size_t kIdx3d000 = kCornerIndices[0] * kPartitionNumber * kPartitionNumber + kCornerIndices[1] * kPartitionNumber + kCornerIndices[2];
                    // Calculating the index of the bottom-right corner
                    const size_t kIdx3d111 = kCornerXIndex111 * kPartitionNumber * kPartitionNumber + kCornerYIndex111 * kPartitionNumber + kCornerZIndex111;

                    // Coordinates of the top-left and bottom-right corners in the destination image
                    auto& indexPair000 = gridCornerCoordinates3D[kIdx3d000];

                    std::tuple<int, int, int, double, double, double> indexPair111;
                    if (kCornerYIndex111 >= kPartitionNumber ||
                        kCornerXIndex111 >= kPartitionNumber ||
                        kCornerZIndex111 >= kPartitionNumber)
                    {
                        // Get the coordinates of the corner
                        const double kX111 = kCornerXIndex111 * kCellSize1,
                            kY111 = kCornerYIndex111 * kCellSize2,
                            kZ111 = kCornerZIndex111 * kCellSize3;

                        // Project them by the estimated homography matrix
                        const double x2p = descriptor(0, 0) * kX111 + descriptor(1, 0) * kY111 + descriptor(2, 0) * kZ111 + descriptor(3, 0),
                            y2p = descriptor(0, 1) * kX111 + descriptor(1, 1) * kY111 + descriptor(2, 1) * kZ111 + descriptor(3, 1),
                            z2p = descriptor(0, 2) * kX111 + descriptor(1, 2) * kY111 + descriptor(2, 2) * kZ111 + descriptor(3, 2);

                        indexPair111 = std::tuple<int, int, int, double, double, double>(
                            x2p / kCellSize4, y2p / kCellSize5, z2p / kCellSize6,
                            x2p, y2p, z2p);
                    }
                    else
                        indexPair111 = gridCornerCoordinates3D[kIdx3d111];

                    const double &l1x = std::get<3>(indexPair000) - kInlierOutlierThreshold_;
                    const double &l1y = std::get<4>(indexPair000) - kInlierOutlierThreshold_;
                    const double &l1z = std::get<5>(indexPair000) - kInlierOutlierThreshold_;
                    const double &r1x = std::get<3>(indexPair111) + kInlierOutlierThreshold_;
                    const double &r1y = std::get<4>(indexPair111) + kInlierOutlierThreshold_;
                    const double &r1z = std::get<5>(indexPair111) + kInlierOutlierThreshold_;

                    const double l2x = kCellSize4 * kCornerIndices[3];
                    const double l2y = kCellSize5 * kCornerIndices[4];
                    const double l2z = kCellSize6 * kCornerIndices[5];
                    const double r2x = l2x + kCellSize4;
                    const double r2y = l2y + kCellSize5;
                    const double r2z = l2z + kCellSize6;

                    // If one rectangle is on left side of other
                    if (l1x <= r2x && l2x <= r1x ||
                        // If one rectangle is above other
                        r1y <= l2y && r2y <= l1y ||
                        // If one rectangle is above other
                        r1z <= l2z && r2z <= l1z)
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                    }
                }
            }
        }

		FORCE_INLINE void SpacePartitioningRANSAC::runHomography(
			const DataMatrix &kData_,
			const models::Model &kModel_,
			const double& kInlierOutlierThreshold_,
			std::vector<const std::vector<size_t>*>& selectedCells_,
			size_t& pointNumber_)
        {*/
            /*
                Selecting cells based on mutual visibility
            */
            /*constexpr double kDeterminantEpsilon = 1e-3;
            const Eigen::Matrix3d& descriptor = kModelData;
            const double kDeterminant = descriptor.determinant();
            if (abs(kDeterminant) < kDeterminantEpsilon)
                return;

            const double& kCellSize1 = additionalParameters[0],
                & kCellSize2 = additionalParameters[1],
                & kCellSize3 = additionalParameters[2],
                & kCellSize4 = additionalParameters[3],
                & kPartitionNumber = additionalParameters[4];

            // Iterate through all cells and project their corners to the second image
            const static std::vector<int> steps = { 0, 0,
                0, 1,
                1, 0,
                1, 1 };

            std::fill(std::begin(gridCornerMask), std::end(gridCornerMask), 0);

            // Iterating through all cells in the neighborhood graph
            for (const auto& [cell, value] : neighborhoodGraph->getCells())
            {
                // The points in the cell
                const auto& points = value.first;

                // Checking if there are enough points in the cell to make the cell selection worth it
                if (points.size() < 4)
                {
                    // If not, simply test all points from the cell and continue
                    selectedCells_.emplace_back(&points);
                    pointNumber_ += points.size();
                    continue;
                }

                const auto& kCornerIndices = value.second;
                bool overlaps = false;

                // Iterate through the corners of the current cell
                for (size_t stepIdx = 0; stepIdx < 8; stepIdx += 2)
                {
                    // The index of the currently projected corner
                    const size_t kCornerHorizontalIndex = kCornerIndices[0] + steps[stepIdx];
                    if (kCornerHorizontalIndex >= kPartitionNumber)
                        continue;

                    const size_t kCornerVerticalIndex = kCornerIndices[1] + steps[stepIdx + 1];
                    if (kCornerVerticalIndex >= kPartitionNumber)
                        continue;

                    // Get the index of the corner's projection in the destination image
                    const size_t kIdx2d = kCornerHorizontalIndex * kPartitionNumber + kCornerVerticalIndex;

                    // This is already or will be the horizontal and vertical indices in the destination image 
                    auto& indexPair = gridCornerCoordinatesH[kIdx2d];

                    // If the corner hasn't yet been projected to the destination image
                    if (!gridCornerMask[kIdx2d])
                    {
                        // Get the coordinates of the corner
                        const double kX1 = kCornerHorizontalIndex * kCellSize1,
                            kY1 = kCornerVerticalIndex * kCellSize2;

                        // Project them by the estimated homography matrix
                        double x2p = kX1 * descriptor(0, 0) + kY1 * descriptor(0, 1) + descriptor(0, 2),
                            y2p = kX1 * descriptor(1, 0) + kY1 * descriptor(1, 1) + descriptor(1, 2),
                            h2p = kX1 * descriptor(2, 0) + kY1 * descriptor(2, 1) + descriptor(2, 2);

                        x2p /= h2p;
                        y2p /= h2p;

                        // Store the projected corner's cell indices
                        std::get<0>(indexPair) = x2p / kCellSize3;
                        std::get<1>(indexPair) = y2p / kCellSize4;
                        std::get<2>(indexPair) = x2p;
                        std::get<3>(indexPair) = y2p;

                        // Note that the corner has been already projected
                        gridCornerMask[kIdx2d] = true;
                    }

                    // Check if the projected corner is equal to the correspondence's destination point's grid cell.
                    // This works due to the coordinate truncation.
                    if (std::get<0>(indexPair) == kCornerIndices[2] &&
                        std::get<1>(indexPair) == kCornerIndices[3])
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                        break;
                    }
                }

                // Check if there is an overlap
                if (!overlaps)
                {
                    // The horizontal index of the bottom-right corner
                    const size_t kCornerHorizontalIndex11 = kCornerIndices[0] + steps[6];
                    // The vertical index of the bottom-right corner
                    const size_t kCornerVerticalIndex11 = kCornerIndices[1] + steps[7];

                    // Calculating the index of the top-left corner
                    const size_t kIdx2d00 = kCornerIndices[0] * kPartitionNumber + kCornerIndices[1];
                    // Calculating the index of the bottom-right corner
                    const size_t kIdx2d11 = kCornerHorizontalIndex11 * kPartitionNumber + kCornerVerticalIndex11;

                    // Coordinates of the top-left and bottom-right corners in the destination image
                    auto& indexPair00 = gridCornerCoordinatesH[kIdx2d00];

                    std::tuple<int, int, double, double> indexPair11;
                    if (kCornerVerticalIndex11 >= kPartitionNumber ||
                        kCornerHorizontalIndex11 >= kPartitionNumber)
                    {
                        // Get the coordinates of the corner
                        const double kX11 = kCornerHorizontalIndex11 * kCellSize1,
                            kY11 = kCornerVerticalIndex11 * kCellSize2;

                        // Project them by the estimated homography matrix
                        double x2p = kX11 * descriptor(0, 0) + kY11 * descriptor(0, 1) + descriptor(0, 2),
                            y2p = kX11 * descriptor(1, 0) + kY11 * descriptor(1, 1) + descriptor(1, 2),
                            h2p = kX11 * descriptor(2, 0) + kY11 * descriptor(2, 1) + descriptor(2, 2);

                        x2p /= h2p;
                        y2p /= h2p;

                        indexPair11 = std::tuple<int, int, double, double>(x2p / kCellSize3, y2p / kCellSize4, x2p, y2p);
                    }
                    else
                        indexPair11 = gridCornerCoordinatesH[kIdx2d11];

                    const double& l1x = std::get<2>(indexPair00) - kInlierOutlierThreshold_;
                    const double& l1y = std::get<3>(indexPair00) - kInlierOutlierThreshold_;
                    const double& r1x = std::get<2>(indexPair11) + kInlierOutlierThreshold_;
                    const double& r1y = std::get<3>(indexPair11) + kInlierOutlierThreshold_;

                    const double l2x = kCellSize3 * kCornerIndices[2];
                    const double l2y = kCellSize4 * kCornerIndices[3];
                    const double r2x = l2x + kCellSize3;
                    const double r2y = l2y + kCellSize4;

                    // If one rectangle is on left side of other
                    if (l1x <= r2x && l2x <= r1x ||
                        // If one rectangle is above other
                        r1y <= l2y && r2y <= l1y)
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                    }
                }
            }*/
        //}
	}
}