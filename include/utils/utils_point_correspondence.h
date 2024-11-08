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
#include <algorithm>
#include <cmath>
#include "types.h"

void normalizePointsByIntrinsics(
    const DataMatrix &kCorrespondences_,
    const Eigen::Matrix3d &kIntrinsicsSource_,
    const Eigen::Matrix3d &kIntrinsicsDestination_,
    DataMatrix &normalizedCorrespondences_,
    const size_t kSampleNumber = 0,
    const size_t *kSample_ = nullptr)
{
    // Get the number of correspondences
    const int &rows =  
        kSample_ == nullptr ? kCorrespondences_.rows() : kSampleNumber;
    // Get the number of columns
    const int &cols = kCorrespondences_.cols();

    // Resize the normalized correspondences matrix
    normalizedCorrespondences_.resize(rows, cols);

    // Inverse of the intrinsic matrix
    const Eigen::Matrix3d kIntrinsicsSourceInv = kIntrinsicsSource_.inverse();
    const Eigen::Matrix3d kIntrinsicsDestinationInv = kIntrinsicsDestination_.inverse();

    // Create a point to store the current point
    Eigen::Vector3d point(0, 0, 1);

    // Normalize each point
    for (size_t pointIdx = 0; pointIdx < rows; ++pointIdx)
    {
        // Get index of the current point
        const size_t &idx = kSample_ == nullptr ? pointIdx : kSample_[pointIdx];

        // Get the (x, y) point from source point
        point(0) = kCorrespondences_(idx, 0);
        point(1) = kCorrespondences_(idx, 1);

        // Apply the inverse intrinsic matrix to get the normalized point
        Eigen::Vector3d normalizedPoint = kIntrinsicsSourceInv * point;

        // Store the normalized coordinates in the output matrix
        normalizedCorrespondences_(idx, 0) = normalizedPoint(0) / normalizedPoint(2);
        normalizedCorrespondences_(idx, 1) = normalizedPoint(1) / normalizedPoint(2);
        
        // Get the (x, y) point from the destination point
        point(0) = kCorrespondences_(idx, 2);
        point(1) = kCorrespondences_(idx, 3);

        // Apply the inverse intrinsic matrix to get the normalized point
        normalizedPoint = kIntrinsicsDestinationInv * point;

        // Store the normalized coordinates in the output matrix
        normalizedCorrespondences_(idx, 2) = normalizedPoint(0) / normalizedPoint(2);
        normalizedCorrespondences_(idx, 3) = normalizedPoint(1) / normalizedPoint(2);

        // Copy the rest of the columns
        for (size_t i = 4; i < cols; ++i)
            normalizedCorrespondences_(idx, i) = kCorrespondences_(idx, i);
    }
}

void normalizePointsByIntrinsics(
    const DataMatrix &kPoints_,
    const Eigen::Matrix3d &kIntrinsics_,
    DataMatrix &normalizedCorrespondences_,
    const size_t kSampleNumber = 0,
    const size_t *kSample_ = nullptr)
{
    // Get the number of correspondences
    const int &rows =  
        kSample_ == nullptr ? kPoints_.rows() : kSampleNumber;
    // Get the number of columns
    const int &cols = kPoints_.cols();

    // Resize the normalized correspondences matrix
    normalizedCorrespondences_.resize(rows, cols);

    // Inverse of the intrinsic matrix
    const Eigen::Matrix3d kIntrinsicsInv = kIntrinsics_.inverse();

    // Create a point to store the current point
    Eigen::Vector3d point(0, 0, 1);

    // Normalize each point
    for (size_t pointIdx = 0; pointIdx < rows; ++pointIdx)
    {
        // Get the (x, y) point from correspondences
        point(0) = kPoints_(pointIdx, 0);
        point(1) = kPoints_(pointIdx, 1);

        // Apply the inverse intrinsic matrix to get the normalized point
        const Eigen::Vector3d kNormalizedPoint = kIntrinsicsInv * point;

        // Store the normalized coordinates in the output matrix
        normalizedCorrespondences_(pointIdx, 0) = kNormalizedPoint(0) / kNormalizedPoint(2);
        normalizedCorrespondences_(pointIdx, 1) = kNormalizedPoint(1) / kNormalizedPoint(2);

        // Copy the rest of the columns
        for (size_t i = 2; i < cols; ++i)
            normalizedCorrespondences_(pointIdx, i) = kPoints_(pointIdx, i);
    }
}

void normalize2D2DPointCorrespondences(
    const DataMatrix &kCorrespondences_,
    DataMatrix &normalizedCorrespondences_,
    Eigen::Matrix3d &normalizingTransformSource_,
    Eigen::Matrix3d &normalizingTransformDestination_,
    const size_t kSampleNumber = 0,
    const size_t *kSample_ = nullptr,
    const bool kSharedScale = true)
{
    // Get the number of correspondences
    const int &rows =  
        kSample_ == nullptr ? kCorrespondences_.rows() : kSampleNumber;
    // Get the number of columns
    const int &cols = kCorrespondences_.cols();

    if (cols < 4)
        throw std::runtime_error("The number of columns in the input correspondences should be at least 4.");
    if (rows < 3)
        throw std::runtime_error("The number of rows in the input correspondences should be at least 3.");

    // Resize the normalized correspondences matrix
    normalizedCorrespondences_.resize(rows, cols);

    double massPointSrc[2], // Mass point in the first image
        massPointDst[2]; // Mass point in the second image

    // Initializing the mass point coordinates
    massPointSrc[0] =
        massPointSrc[1] =
        massPointDst[0] =
        massPointDst[1] =
        0.0;

    // Calculating the mass points in both images
    for (size_t i = 0; i < rows; ++i)
    {
        // Get index of the current point
        const size_t &idx = kSample_ == nullptr ? i : kSample_[i];

        // Add the coordinates to that of the mass points
        massPointSrc[0] += kCorrespondences_(idx, 0);
        massPointSrc[1] += kCorrespondences_(idx, 1);
        massPointDst[0] += kCorrespondences_(idx, 2);
        massPointDst[1] += kCorrespondences_(idx, 3);
    }

    // Get the average
    massPointSrc[0] /= rows;
    massPointSrc[1] /= rows;
    massPointDst[0] /= rows;
    massPointDst[1] /= rows;

    // Get the mean distance from the mass points
    double averageDistanceSrc = 0.0,
        averageDistanceDst = 0.0;
    for (size_t i = 0; i < rows; ++i)
    {
        // Get index of the current point
        const size_t &idx = kSample_ == nullptr ? i : kSample_[i];

        const double &x1 = kCorrespondences_(idx, 0);
        const double &y1 = kCorrespondences_(idx, 1);
        const double &x2 = kCorrespondences_(idx, 2);
        const double &y2 = kCorrespondences_(idx, 3);

        const double dx1 = massPointSrc[0] - x1;
        const double dy1 = massPointSrc[1] - y1;
        const double dx2 = massPointDst[0] - x2;
        const double dy2 = massPointDst[1] - y2;

        if (kSharedScale)
        {
            averageDistanceSrc += sqrt(dx1 * dx1 + dy1 * dy1);
            averageDistanceSrc += sqrt(dx2 * dx2 + dy2 * dy2);
        }
        else
        {
            averageDistanceSrc += sqrt(dx1 * dx1 + dy1 * dy1);
            averageDistanceDst += sqrt(dx2 * dx2 + dy2 * dy2);
        }
    }

    averageDistanceSrc /= rows;
    if (!kSharedScale)
        averageDistanceDst /= rows;

    // Calculate the sqrt(2) / MeanDistance ratios
    double ratioSrc = M_SQRT2 / averageDistanceSrc;
    double ratioDst;
    if (kSharedScale)
        ratioDst = ratioSrc;
    else
        ratioDst = M_SQRT2 / averageDistanceDst;

    // Compute the normalized coordinates
    for (size_t i = 0; i < rows; ++i)
    {
        // Get index of the current point
        const size_t &idx = kSample_ == nullptr ? i : kSample_[i];

        const double &x1 = kCorrespondences_(idx, 0);
        const double &y1 = kCorrespondences_(idx, 1);
        const double &x2 = kCorrespondences_(idx, 2);
        const double &y2 = kCorrespondences_(idx, 3);

        normalizedCorrespondences_(i, 0) = (x1 - massPointSrc[0]) * ratioSrc;
        normalizedCorrespondences_(i, 1) = (y1 - massPointSrc[1]) * ratioSrc;
        normalizedCorrespondences_(i, 2) = (x2 - massPointDst[0]) * ratioDst;
        normalizedCorrespondences_(i, 3) = (y2 - massPointDst[1]) * ratioDst;

        for (size_t i = 4; i < cols; ++i)
            normalizedCorrespondences_(idx, i) = kCorrespondences_(idx, i);
    }

    // Creating the normalizing transformations
    normalizingTransformSource_ << ratioSrc, 0, -ratioSrc * massPointSrc[0],
        0, ratioSrc, -ratioSrc * massPointSrc[1],
        0, 0, 1;

    normalizingTransformDestination_ << ratioDst, 0, -ratioDst * massPointDst[0],
        0, ratioDst, -ratioDst * massPointDst[1],
        0, 0, 1;
}


void normalize3D3DPointCorrespondences(
    const DataMatrix &kCorrespondences_,
    DataMatrix &normalizedCorrespondences_,
    Eigen::Matrix4d &normalizingTransformSource_,
    Eigen::Matrix4d &normalizingTransformDestination_,
    const size_t kSampleNumber = 0,
    const size_t *kSample_ = nullptr,
    const bool kSharedScale = true)
{
    // Get the number of correspondences
    const int &rows =  
        kSample_ == nullptr ? kCorrespondences_.rows() : kSampleNumber;
    // Get the number of columns
    const int &cols = kCorrespondences_.cols();

    if (cols < 6)
        throw std::runtime_error("The number of columns in the input correspondences should be at least 4.");
    if (rows < 3)
        throw std::runtime_error("The number of rows in the input correspondences should be at least 3.");

    // Resize the normalized correspondences matrix
    normalizedCorrespondences_.resize(rows, cols);

    double massPointSrc[3], // Mass point in the first image
        massPointDst[3]; // Mass point in the second image

    // Initializing the mass point coordinates
    massPointSrc[0] =
        massPointSrc[1] =
        massPointSrc[2] =
        massPointDst[0] =
        massPointDst[1] =
        massPointDst[2] =
        0.0;

    // Calculating the mass points in both images
    for (size_t i = 0; i < rows; ++i)
    {
        // Get index of the current point
        const size_t &idx = kSample_ == nullptr ? i : kSample_[i];

        // Add the coordinates to that of the mass points
        massPointSrc[0] += kCorrespondences_(idx, 0);
        massPointSrc[1] += kCorrespondences_(idx, 1);
        massPointSrc[2] += kCorrespondences_(idx, 2);
        massPointDst[0] += kCorrespondences_(idx, 3);
        massPointDst[1] += kCorrespondences_(idx, 4);
        massPointDst[2] += kCorrespondences_(idx, 5);
    }

    // Get the average
    massPointSrc[0] /= rows;
    massPointSrc[1] /= rows;
    massPointSrc[2] /= rows;
    massPointDst[0] /= rows;
    massPointDst[1] /= rows;
    massPointDst[2] /= rows;

    // Get the mean distance from the mass points
    double averageDistanceSrc = 0.0,
        averageDistanceDst = 0.0;
    for (size_t i = 0; i < rows; ++i)
    {
        // Get index of the current point
        const size_t &idx = kSample_ == nullptr ? i : kSample_[i];

        const double &x1 = kCorrespondences_(idx, 0);
        const double &y1 = kCorrespondences_(idx, 1);
        const double &z1 = kCorrespondences_(idx, 2);
        const double &x2 = kCorrespondences_(idx, 3);
        const double &y2 = kCorrespondences_(idx, 4);
        const double &z2 = kCorrespondences_(idx, 5);

        const double dx1 = massPointSrc[0] - x1;
        const double dy1 = massPointSrc[1] - y1;
        const double dz1 = massPointSrc[2] - z1;
        const double dx2 = massPointDst[0] - x2;
        const double dy2 = massPointDst[1] - y2;
        const double dz2 = massPointDst[2] - z2;

        if (kSharedScale)
        {
            averageDistanceSrc += sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
            averageDistanceSrc += sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
        }
        else
        {
            averageDistanceSrc += sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
            averageDistanceDst += sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
        }
    }

    averageDistanceSrc /= rows;
    if (!kSharedScale)
        averageDistanceDst /= rows;

    // Calculate the sqrt(2) / MeanDistance ratios
    double ratioSrc = M_SQRT2 / averageDistanceSrc;
    double ratioDst;
    if (kSharedScale)
        ratioDst = ratioSrc;
    else
        ratioDst = M_SQRT2 / averageDistanceDst;

    // Compute the normalized coordinates
    for (size_t i = 0; i < rows; ++i)
    {
        // Get index of the current point
        const size_t &idx = kSample_ == nullptr ? i : kSample_[i];

        const double &x1 = kCorrespondences_(idx, 0);
        const double &y1 = kCorrespondences_(idx, 1);
        const double &z1 = kCorrespondences_(idx, 2);
        const double &x2 = kCorrespondences_(idx, 3);
        const double &y2 = kCorrespondences_(idx, 4);
        const double &z2 = kCorrespondences_(idx, 5);

        normalizedCorrespondences_(i, 0) = (x1 - massPointSrc[0]) * ratioSrc;
        normalizedCorrespondences_(i, 1) = (y1 - massPointSrc[1]) * ratioSrc;
        normalizedCorrespondences_(i, 2) = (z1 - massPointSrc[2]) * ratioSrc;
        normalizedCorrespondences_(i, 3) = (x2 - massPointDst[0]) * ratioDst;
        normalizedCorrespondences_(i, 4) = (y2 - massPointDst[1]) * ratioDst;
        normalizedCorrespondences_(i, 5) = (z2 - massPointDst[2]) * ratioDst;

        for (size_t i = 6; i < cols; ++i)
            normalizedCorrespondences_(idx, i) = kCorrespondences_(idx, i);
    }

    // Creating the normalizing transformations
    normalizingTransformSource_ << ratioSrc, 0, 0, -ratioSrc * massPointSrc[0],
        0, ratioSrc, 0, -ratioSrc * massPointSrc[1],
        0, 0, ratioSrc, -ratioSrc * massPointSrc[2],
        0, 0, 0, 1;

    normalizingTransformDestination_ << ratioDst, 0, 0, -ratioDst * massPointDst[0],
        0, ratioDst, 0, -ratioDst * massPointDst[1],
        0, 0, ratioDst, -ratioDst * massPointDst[2],
        0, 0, 0, 1;
}