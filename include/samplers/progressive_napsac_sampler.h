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

#include "abstract_sampler.h"
#include "../utils/uniform_random_generator.h"
#include "../neighborhood/abstract_neighborhood.h"
#include "../utils/types.h"
#include "prosac_sampler.h"

#include <vector>
#include <iostream>

namespace superansac {
namespace samplers {

template <size_t _DimensionNumber>
class ProgressiveNAPSACSampler : public AbstractSampler 
{
protected:
    std::unique_ptr<utils::UniformRandomGenerator<size_t>> randomGenerator; // The random number generator
    double samplerLength; // The length of fully blending to global sampling 
    size_t layerNumber; // The number of overlapping neighborhood grids
    std::vector<double> sizes; // The sizes along each axis
	mutable const DataMatrix *kData;

    std::vector<std::unique_ptr<neighborhood::GridNeighborhoodGraph<_DimensionNumber>>> gridLayers; // The overlapping neighborhood grids
    std::vector<size_t> layerData; // The sizes of the grids

    std::vector<size_t> currentLayerPerPoint, // It stores the index of the layer which is used for each point
        hitsPerPoint, // It stores how many times each point has been selected
        subsetSizePerPoint, // It stores what is the size of subset (~the size of the neighborhood ball) for each point
        growthFunctionProgressiveNapsac; // The P-NAPSAC growth function.

    PROSACSampler onePointProsacSampler, // The PROSAC sampler used for selecting the initial points, i.e., the center of the hypersphere.
        prosacSampler; // The PROSAC sampler used when the sampling has been fully blended to be global sampling.

    size_t kthSampleNumber, // The kth sample of prosac sampling.
        maxProgressiveNapsacIterations, // The maximum number of local sampling iterations before applying global sampling
        sampleSize, // The size of a minimal sample to fit a model.
        pointNumber; // The number of data points.

    void initializeGridLayer(size_t kLayerIdx_, const DataMatrix& kData_) 
	{
        if (!gridLayers[kLayerIdx_]) 
		{
            std::vector<double> cellSizes(_DimensionNumber);
            const size_t cellNumberInGrid = layerData[kLayerIdx_];

            for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx)
                cellSizes[dimensionIdx] = sizes[dimensionIdx] / cellNumberInGrid;

            gridLayers[kLayerIdx_] = std::make_unique<neighborhood::GridNeighborhoodGraph<_DimensionNumber>>();
            gridLayers[kLayerIdx_]->setCellSizes(cellSizes, cellNumberInGrid);
            gridLayers[kLayerIdx_]->initialize(&kData_);
        }
    }

public:
    // Constructor 
    ProgressiveNAPSACSampler() 
        : samplerLength(20),
          kthSampleNumber(0),
          maxProgressiveNapsacIterations(0),
          sampleSize(0),
          pointNumber(0) {}

    // Destructor
    ~ProgressiveNAPSACSampler() override = default;

    // Return the name of the sampler
    constexpr static const char* name() 
	{
        return "ProgressiveNAPSACSampler";
    }

    FORCE_INLINE void setLayerData(
        const std::vector<size_t>& kLayerData_,
        const std::vector<double>& kSizes_) 
	{
        layerData = kLayerData_;
        layerNumber = layerData.size();
        sizes = kSizes_;
        gridLayers.resize(layerNumber); // Resize the vector for lazy initialization
    }

    FORCE_INLINE void setSamplerLength(double kSamplerLength_) 
	{
        samplerLength = kSamplerLength_;
    }

    FORCE_INLINE void setSampleSize(size_t kSampleSize_) 
	{
        sampleSize = kSampleSize_;
    }

    // Initializes any non-trivial variables and sets up sampler if
    // necessary. Must be called before sample is called.
    FORCE_INLINE void initialize(const DataMatrix &kData_) 
	{
        // Initialize the variables
        pointNumber = kData_.rows();
		kData = &kData_;
        currentLayerPerPoint.assign(pointNumber, 0);
        hitsPerPoint.assign(pointNumber, 0);
        subsetSizePerPoint.assign(pointNumber, sampleSize);
        onePointProsacSampler.setSampleSize(1);
        onePointProsacSampler.initialize(pointNumber);
        prosacSampler.setSampleSize(sampleSize);
        prosacSampler.initialize(pointNumber);
        kthSampleNumber = 0;

        // Initialize the random generator
        randomGenerator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
        randomGenerator->resetGenerator(0, pointNumber);

        maxProgressiveNapsacIterations = static_cast<size_t>(samplerLength * pointNumber);

        // Inititalize the P-NAPSAC growth function
        growthFunctionProgressiveNapsac.resize(pointNumber);
        size_t local_sampleSize = sampleSize - 1;
        double T_n = maxProgressiveNapsacIterations;
        for (size_t i = 0; i < local_sampleSize; ++i) {
            T_n *= static_cast<double>(local_sampleSize - i) / (pointNumber - i);
        }

        unsigned int T_n_prime = 1;
        for (size_t i = 0; i < pointNumber; ++i) {
            if (i + 1 <= local_sampleSize) {
                growthFunctionProgressiveNapsac[i] = T_n_prime;
                continue;
            }
            double Tn_plus1 = static_cast<double>(i + 1) * T_n /
                (i + 1 - local_sampleSize);
            growthFunctionProgressiveNapsac[i] = T_n_prime + static_cast<size_t>(ceil(Tn_plus1 - T_n));
            T_n = Tn_plus1;
            T_n_prime = growthFunctionProgressiveNapsac[i];
        }
    }

    // Initialize function
    FORCE_INLINE void initialize(size_t kPointNumber_) override 
	{
        throw std::runtime_error("The initialize function should be used with the data matrix.");
    }

    FORCE_INLINE void update(
        const size_t* const samples_,
        const size_t& sampleSize_,
        const size_t& iteration_number_,
        const double& inlier_ratio_) override 
	{
        // This method remains unimplemented
    }

    void reset(size_t kDataSize_) 
	{
        randomGenerator->resetGenerator(0, pointNumber);
        kthSampleNumber = 0;
        hitsPerPoint.assign(pointNumber, 0);
        currentLayerPerPoint.assign(pointNumber, 0);
        subsetSizePerPoint.assign(pointNumber, sampleSize);
        onePointProsacSampler.reset(pointNumber);
        prosacSampler.reset(pointNumber);
    }

    // Sample function
    FORCE_INLINE bool sample(
        const DataMatrix& kData_, // Data matrix
        const int kNumSamples_, // Number of samples
        size_t* kSamples_) override 
	{
        return sample(kData_.rows(), kNumSamples_, kSamples_);
    }

    // Sample function
    FORCE_INLINE bool sample(
        size_t kPointNumber_,
        const int kNumSamples_,
        size_t* samples_) 
	{
        ++kthSampleNumber;

        if (kNumSamples_ != sampleSize)
			throw std::runtime_error("An error occurred when sampling. Progressive NAPSAC is not yet implemented to change the sample size after being initialized.");

        // If there are not enough points in the pool, interrupt the procedure.
        if (sampleSize > kPointNumber_)
            return false;

        // Do completely global sampling (PROSAC is used now), instead of Progressive NAPSAC,
        // if the maximum iterations has been done without finding the sought model.
        if (kthSampleNumber > maxProgressiveNapsacIterations) {
            prosacSampler.setSampleNumber(kthSampleNumber);
            return prosacSampler.sample(pointNumber, sampleSize, samples_);
        }

        // Select the first point used as the center of the
        // hypersphere in the local sampling.
        const bool success = onePointProsacSampler.sample(pointNumber, // The pool from which the indices are chosen
            1, // Only a single point is selected to be the center
            samples_); // The sample to be selected
        if (!success) // Return false, if the selection of the initial point was not successful.
            return false;

        // The index of the selected center
        const size_t initial_point = samples_[0];

        // Increase the number of hits of the selected point
        size_t& hits = ++hitsPerPoint[initial_point];

        // Get the subset size (i.e., the size of the neighborhood sphere) of the 
        // selected initial point.
        size_t& sampleSizeProgressiveNapsac = subsetSizePerPoint[initial_point];
        while (hits > growthFunctionProgressiveNapsac[sampleSizeProgressiveNapsac - 1] &&
            sampleSizeProgressiveNapsac < pointNumber)
            sampleSizeProgressiveNapsac = std::min(sampleSizeProgressiveNapsac + 1, pointNumber);

        // Get the neighborhood from the grids
        size_t& currentLayer = currentLayerPerPoint[initial_point];
        bool isLastLayer = false;
        do // Try to find the grid which contains enough points
        {
            // In the case when the grid with a single cell is used,
            // apply PROSAC.
            if (currentLayer >= layerNumber) {
                isLastLayer = true;
                break;
            }

            // Lazily initialize the grid layer if it has not been initialized yet
            initializeGridLayer(currentLayer, *kData);

            // Get the points in the cell where the selected initial
            // points is located.
            const std::vector<size_t>& neighbors =
                gridLayers[currentLayer]->getNeighbors(initial_point);

            // If there are not enough points in the cell, start using a 
            // less fine grid.
            if (neighbors.size() < sampleSizeProgressiveNapsac) {
                ++currentLayer; // Jump to the next layer with bigger cells.
                continue;
            }

            // If the procedure got to this point, there is no reason to choose a different layer of grids
            // since the current one has enough points. 
            break;
        } while (true);

        // If not the last layer has been chosen, sample from the neighbors of the initially selected point.
        if (!isLastLayer) 
		{
            // The indices of the points which are in the same cell as the
            // initially selected one.
            const std::vector<size_t>& neighbors =
                gridLayers[currentLayer]->getNeighbors(initial_point);

            // Put the selected point to the end of the sample array to avoid
            // being overwritten when sampling the remaining points.
            samples_[sampleSize - 1] = initial_point;

            // The next point should be the farthest one from the initial point. Note that the points in the grid cell are
            // not ordered w.r.t. to their distances from the initial point. However, they are ordered as in PROSAC.
            samples_[sampleSize - 2] = neighbors[sampleSizeProgressiveNapsac - 1];

            // Select n - 2 points randomly  
            randomGenerator->generateUniqueRandomSet(samples_,
                sampleSize - 2,
                sampleSizeProgressiveNapsac - 2,
                initial_point);

            for (size_t i = 0; i < sampleSize - 2; ++i) {
                samples_[i] = neighbors[samples_[i]];  // Replace the neighbor index by the index of the point
                ++hitsPerPoint[samples_[i]]; // Increase the hit number of each selected point
            }
            ++hitsPerPoint[samples_[sampleSize - 2]]; // Increase the hit number of each selected point
        }
        // If the last layer (i.e., the layer with a single cell) has been chosen, do global sampling
        // by PROSAC sampler. 
        else 
		{
            // If local sampling
            prosacSampler.setSampleNumber(kthSampleNumber);
            const bool success = prosacSampler.sample(pointNumber, sampleSize, samples_);
            samples_[sampleSize - 1] = initial_point;
            return success;
        }

        return true;
    }
};

} // namespace samplers
} // namespace superansac
