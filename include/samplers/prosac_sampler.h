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
#include "../utils/types.h"

#include <vector>
#include <iostream>

namespace superansac {
namespace samplers {

class PROSACSampler : public AbstractSampler
{
protected:
    std::unique_ptr<utils::UniformRandomGenerator<size_t>> randomGenerator;
    std::vector<size_t> growthFunction; // The growth function of PROSAC.
    size_t sampleSize, // The size of the current sample.
           pointNumber, // The number of points in the data.
           ransacConvergenceIterations, // Number of iterations of PROSAC before it just acts like RANSAC.
           kthSampleNumber, // The kth sample of PROSAC sampling.
           largestSampleSize, // The largest sample size that has been used.
           subsetSize; // The size of the current subset of points.

    FORCE_INLINE void incrementIterationNumber()
    {
        ++kthSampleNumber;

        if (kthSampleNumber > ransacConvergenceIterations)
        {
            randomGenerator->resetGenerator(0, pointNumber - 1);
        }
        else if (kthSampleNumber > growthFunction[subsetSize - 1])
        {
            ++subsetSize;
            subsetSize = std::min(subsetSize, pointNumber);
            largestSampleSize = std::max(largestSampleSize, subsetSize);

            randomGenerator->resetGenerator(0, subsetSize - 2);
        }
    }

public:
    // Constructors
    PROSACSampler() : PROSACSampler(100000) {}
    PROSACSampler(const size_t& kRansacConvergenceIterations_)
        : ransacConvergenceIterations(kRansacConvergenceIterations_), kthSampleNumber(1)
    {
    }

    // Destructor
    ~PROSACSampler() {}

    // Return the name of the sampler
    constexpr static const char* name()
    {
        return "PROSACSampler";
    }

    void setSampleSize(const size_t kSampleSize_)
    {
        sampleSize = kSampleSize_;
    }

    // Set the sample such that you are sampling the kth PROSAC sample (Eq. 6).
    void setSampleNumber(int k)
    {
        kthSampleNumber = k;

        if (kthSampleNumber > ransacConvergenceIterations)
        {
            randomGenerator->resetGenerator(0, pointNumber - 1);
        }
        else
        {
            while (kthSampleNumber > growthFunction[subsetSize - 1] && subsetSize != pointNumber)
            {
                ++subsetSize;
                subsetSize = std::min(subsetSize, pointNumber);
                largestSampleSize = std::max(largestSampleSize, subsetSize);

                randomGenerator->resetGenerator(0, subsetSize - 2);
            }
        }
    }

    FORCE_INLINE void initialize(const DataMatrix& kData_)
    {
        initialize(kData_.rows());
    }

    FORCE_INLINE void initialize(const size_t kPointNumber_)
    {
        pointNumber = kPointNumber_;
        growthFunction.resize(pointNumber, 0);

        double T_n = ransacConvergenceIterations;
        for (size_t i = 0; i < sampleSize; i++)
        {
            T_n *= static_cast<double>(sampleSize - i) / (pointNumber - i);
        }

        size_t T_n_prime = 1;
        for (size_t i = 0; i < pointNumber; ++i)
        {
            if (i + 1 <= sampleSize)
            {
                growthFunction[i] = T_n_prime;
                continue;
            }
            double Tn_plus1 = static_cast<double>(i + 1) * T_n / (i + 1 - sampleSize);
            growthFunction[i] = T_n_prime + static_cast<size_t>(ceil(Tn_plus1 - T_n));
            T_n = Tn_plus1;
            T_n_prime = growthFunction[i];
        }

        largestSampleSize = sampleSize;
        subsetSize = sampleSize;

        randomGenerator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
        randomGenerator->resetGenerator(0, subsetSize - 1);
    }

    FORCE_INLINE void update(const size_t* const subset_, const size_t& sampleSize_, const size_t& iteration_number_, const double& inlier_ratio_)
    {
        // Implementation of update function if needed
    }

    void reset(const size_t& kDataSize_)
    {
        kthSampleNumber = 1;
        largestSampleSize = sampleSize;
        subsetSize = sampleSize;
        randomGenerator->resetGenerator(0, subsetSize - 1);
    }

    FORCE_INLINE bool sample(const DataMatrix& kData_, const int kNumSamples_, size_t* samples_)
    {
        return sample(kData_.rows(), kNumSamples_, samples_);
    }

    FORCE_INLINE bool sample(const size_t kPointNumber_, const int kNumSamples_, size_t* samples_)
    {
        if (kNumSamples_ != sampleSize)
            throw std::invalid_argument("An error occurred when sampling. PROSAC is not implemented to change the sample size after being initialized.");

        if (kthSampleNumber > ransacConvergenceIterations)
        {
            randomGenerator->generateUniqueRandomSet(samples_, sampleSize);
            return true;
        }

        randomGenerator->generateUniqueRandomSet(samples_, sampleSize - 1);
        samples_[sampleSize - 1] = subsetSize - 1;

        incrementIterationNumber();
        return true;
    }
};

}
}