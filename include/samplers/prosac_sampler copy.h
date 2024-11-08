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
            ransacConvergenceIterations, // Number of iterations of PROSAC before it just acts like ransac.
			kthSampleNumber, // The kth sample of prosac sampling.
            largestSampleSize, // The largest sample size that has been used.
            subsetSize; // The size of the current subset of points.

        FORCE_INLINE void incrementIterationNumber()
        {
            // Increase the iteration number
            ++kthSampleNumber;

            // If the method should act exactly like RANSAC, set the random generator to
            // generate values from all possible indices.
            if (kthSampleNumber > ransacConvergenceIterations)
                randomGenerator->resetGenerator(0,
                    pointNumber - 1);
            else // Increment the size of the sampling pool if required			
                if (kthSampleNumber > growthFunction[subsetSize - 1]) 
                {
                    ++subsetSize; // n = n + 1
                    if (subsetSize > pointNumber)
                        subsetSize = pointNumber;
                    if (largestSampleSize < subsetSize)
                        largestSampleSize = subsetSize;

                    // Reset the random generator to generate values from the current subset of points,
                    // except the last one since it will always be used. 
                    randomGenerator->resetGenerator(0,
                        subsetSize - 2);
                }
        }
        
    public:
        // Constructor 
        PROSACSampler() : PROSACSampler(100000) {}
        PROSACSampler(const size_t &kRansacConvergenceIterations_) : 
            ransacConvergenceIterations(kRansacConvergenceIterations_),
            kthSampleNumber(1) 
        {

        }
        // Destructor
        ~PROSACSampler() {}

        // Return the name of the sampler
        constexpr static const char *name()
        {
            return "PROSACSampler";
        }

        void setSampleSize(const size_t kSampleSize_)
        {
            sampleSize = kSampleSize_;
        }

        // Set the sample such that you are sampling the kth prosac sample (Eq. 6).
        void setSampleNumber(int k)
        {
            kthSampleNumber = k;

            // If the method should act exactly like RANSAC, set the random generator to
            // generate values from all possible indices.
            if (kthSampleNumber > ransacConvergenceIterations)
                randomGenerator->resetGenerator(0,
                    pointNumber - 1);
            else // Increment the size of the sampling pool while required			
                while (kthSampleNumber > growthFunction[subsetSize - 1] && 
                    subsetSize != pointNumber)
                {
                    ++subsetSize; // n = n + 1
                    if (subsetSize > pointNumber)
                        subsetSize = pointNumber;
                    if (largestSampleSize < subsetSize)
                        largestSampleSize = subsetSize;

                    // Reset the random generator to generate values from the current subset of points,
                    // except the last one since it will always be used. 
                    randomGenerator->resetGenerator(0,
                        subsetSize - 2);
                }
        }

        // Initializes any non-trivial variables and sets up sampler if
        // necessary. Must be called before sample is called.
        FORCE_INLINE void initialize(
            const DataMatrix &kData_)
        {
            initialize(kData_.rows());
        }

        // Initializes any non-trivial variables and sets up sampler if
        // necessary. Must be called before sample is called.
        FORCE_INLINE void initialize(
            const size_t kPointNumber_)
        {
            pointNumber = kPointNumber_;
            // Set T_n according to the PROSAC paper's recommendation.
            growthFunction.resize(pointNumber, 0);

            // Tq.he data points in U_N are sorted in descending order w.r.t. the quality function 
            // Let {Mi}i = 1...T_N denote the sequence of samples Mi c U_N that are uniformly drawn by Ransac.

            // Let T_n be an average number of samples from {Mi}i=1...T_N that contain data points from U_n only.
            // compute initial value for T_n
            //                                  n - i
            // T_n = T_N * Product i = 0...m-1 -------, n >= sample size, N = points size
            //                                  N - i
            double T_n = ransacConvergenceIterations;
            for (size_t i = 0; i < sampleSize; i++)
                T_n *= static_cast<double>(sampleSize - i) / (pointNumber - i);

            size_t T_n_prime = 1;
            // compute values using recurrent relation
            //             n + 1
            // T(n+1) = --------- T(n), m is sample size.
            //           n + 1 - m

            // growth function is defined as
            // g(t) = min {n, T'_(n) >= t}
            // T'_(n+1) = T'_(n) + (T_(n+1) - T_(n))
            for (size_t i = 0; i < pointNumber; ++i) {
                if (i + 1 <= sampleSize) {
                    growthFunction[i] = T_n_prime;
                    continue;
                }
                double Tn_plus1 = static_cast<double>(i + 1) * T_n / (i + 1 - sampleSize);
                growthFunction[i] = T_n_prime + (unsigned int)ceil(Tn_plus1 - T_n);
                T_n = Tn_plus1;
                T_n_prime = growthFunction[i];
            }

            largestSampleSize = sampleSize; // largest set sampled in PROSAC
            subsetSize = sampleSize; // The size of the current sampling pool		

            // Initialize the random generator
            randomGenerator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
            randomGenerator->resetGenerator(0,
                subsetSize - 1);
        }
        			
        FORCE_INLINE void update(
            const size_t* const subset_,
            const size_t& sampleSize_,
            const size_t& iteration_number_,
            const double& inlier_ratio_)
        {
            
        }

        void reset(const size_t &kDataSize_)
        {
            kthSampleNumber = 1;
            largestSampleSize = sampleSize; // largest set sampled in PROSAC
            subsetSize = sampleSize; // The size of the current sampling pool		
            randomGenerator->resetGenerator(0,
                subsetSize - 1);
        }

        // Sample function
        FORCE_INLINE bool sample(
            const DataMatrix &kData_,
            const int kNumSamples_, 
            size_t *samples_)
        {                
			return sample(kData_.rows(), kNumSamples_, samples_);
        }

        // Sample function
        FORCE_INLINE bool sample(
            const size_t kPointNumber_,
            const int kNumSamples_, 
            size_t *samples_)
        {
            if (kNumSamples_ != sampleSize)
                throw std::invalid_argument("An error occured when sampling. PROSAC is not implemented to change the sample size after being initialized..");

            // If the method should act exactly like RANSAC, sample from all points.
            // From this point, function 'incrementIterationNumber()' is not called
            // since it is not important to increase the iteration number.
            if (kthSampleNumber > ransacConvergenceIterations) 
            {
                randomGenerator->generateUniqueRandomSet(
                    samples_, // The set of points' indices to be selected
                    sampleSize); // The number of points to be selected
                return true;
            }

            // Generate PROSAC sample in range [0, subset_size-2]
            randomGenerator->generateUniqueRandomSet(
                samples_, // The set of points' indices to be selected
                sampleSize - 1); // The number of points to be selected
            samples_[sampleSize - 1] = subsetSize - 1; // The last index is that of the point at the end of the current subset used.

            incrementIterationNumber(); // Increase the iteration number
            return true;
        }
};

}
}