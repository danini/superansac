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
#include "../utils/types.h"
#include "../models/model.h"

namespace superansac 
{
	namespace camera
	{
		class AbstractCamera
		{
		protected:
            std::vector<double> parameters;

		public:
			AbstractCamera(const std::vector<double> &kParams_) : 
                parameters(kParams_)
			{
			}

            virtual ~AbstractCamera() {}

			// Project a 2D point from pixel to image coordinates
            virtual void fromPixelToImageCoordinates(
				const DataMatrix &kData_,
				DataMatrix &kNormalizedData_) const = 0; 
			
			// Project a 2D point from pixel to image coordinates
            virtual void fromImageToPixelCoordinates(
				const std::vector<Eigen::Vector2d> &kNormalizedData_,
				std::vector<Eigen::Vector2d> &kData_) const = 0;

			// Project a 2D point from pixel to image coordinates
            virtual void fromImageToPixelCoordinates(
				const DataMatrix &kNormalizedData_,
				DataMatrix &kData_) const = 0; 

            virtual bool validateParameters(const std::vector<double> &kParams_) const = 0; 
            virtual double normalizeThreshold(double kThreshold_) const = 0; 
            virtual double unnormalizeThreshold(double kThreshold_) const = 0; 
			virtual void rescale(double kScale_) = 0;
			virtual double focalLength() const = 0; 
			virtual size_t getModelId() const = 0;

            const std::vector<double> &getParameters() { return parameters; }
			std::vector<double> &getMutableParameters() { return parameters; }

		};
	}
}

