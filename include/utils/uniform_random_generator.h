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

#include <random>
#include <algorithm>
#include "macros.h"

namespace superansac
{
	namespace utils
	{
        template <typename _Type>
        class UniformRandomGenerator
        {
        protected:
            std::mt19937 generator;
            std::uniform_int_distribution<_Type> generate;

        public:
            UniformRandomGenerator() 
            {
                std::random_device rand_dev;
                generator = std::mt19937(rand_dev());
            }

            ~UniformRandomGenerator() 
            {

            }

            std::mt19937 &getGenerator()
            {
                return generator;
            }

            FORCE_INLINE int getRandomNumber() 
            {
                return generate(generator);
            }

            FORCE_INLINE void resetGenerator(
                const _Type &kMinRange_,
                const _Type &kMaxRange_) 
            {
                generate = std::uniform_int_distribution<_Type>(kMinRange_, kMaxRange_);
            }

            FORCE_INLINE void generateUniqueRandomSet(
                _Type * sample_,
                const _Type &kSampleSize_)
            {
                for (_Type i = 0; i < kSampleSize_; i++)
                {
                    sample_[i] = generate(generator);
                    for (int j = i - 1; j >= 0; j--) 
                    {
                        if (sample_[i] == sample_[j]) 
                        {
                            i--;
                            break;
                        }
                    }
                }
            }

            FORCE_INLINE void generateUniqueRandomSet(
                _Type * sample_,
                const _Type &kSampleSize_,
                const _Type &kMax_) 
            {
                resetGenerator(0, kMax_);
                for (_Type i = 0; i < kSampleSize_; i++) 
                {
                    sample_[i] = generate(generator);
                    for (int j = i - 1; j >= 0; j--) 
                    {
                        if (sample_[i] == sample_[j]) 
                        {
                            i--;
                            break;
                        }
                    }
                }
            }

            FORCE_INLINE void generateUniqueRandomSet(_Type * sample_,
                const _Type kSampleSize_,
                const _Type kMax_,
                const _Type kToSkip_) {
                resetGenerator(0, kMax_);
                for (_Type i = 0; i < kSampleSize_; i++) {
                    sample_[i] = generate(generator);
                    if (sample_[i] == kToSkip_) {
                        i--;
                        continue;
                    }

                    for (int j = i - 1; j >= 0; j--) {
                        if (sample_[i] == sample_[j]) {
                            i--;
                            break;
                        }
                    }
                }
            }
        };

    }
}