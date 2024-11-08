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
		class SimplePinholeCamera : public AbstractCamera
		{
		public:
			SimplePinholeCamera(const std::vector<double> &kParams_) : 
				AbstractCamera(kParams_)
			{
				if (!validateParameters(kParams_))
					throw std::invalid_argument("The number of parameters for the simple radial camera should be 4.");
			}

            ~SimplePinholeCamera() override = default;

			double focalLength() const override
			{
				return parameters[0];
			}

			void rescale(double kScale_) override
			{
				parameters = {parameters[0] * kScale_, // focal length
					parameters[1] * kScale_, // principal point x
					parameters[2] * kScale_}; // principal point y
			}

			// Project a 2D point from pixel to image coordinates
            void fromImageToPixelCoordinates(
				const std::vector<Eigen::Vector2d> &kNormalizedData_,
				std::vector<Eigen::Vector2d> &kData_) const override
			{
				if (kData_.size() != kNormalizedData_.size())
					kData_.resize(kNormalizedData_.size());
				
				// Get the camera parameters
				const double &focalLength = parameters[0];
				const double &cx = parameters[1];
				const double &cy = parameters[2];

				// Normalize the points
				for (size_t pointIdx = 0; pointIdx < kNormalizedData_.size(); ++pointIdx)
				{
					// Get the point
					double x = kNormalizedData_[pointIdx](0);
					double y = kNormalizedData_[pointIdx](1);
					
					// Apply radial distortion and scale the point
					x = x * focalLength + cx;
					y = y * focalLength + cy;

					// Store the normalized point
					kData_[pointIdx](0) = x;
					kData_[pointIdx](1) = y;

					// Copy the rest of the columns
					for (size_t i = 2; i < kData_[pointIdx].cols(); ++i)
						kData_[pointIdx](i) = kNormalizedData_[pointIdx](i);
				}
			}

			// Project a 2D point from pixel to image coordinates
            void fromImageToPixelCoordinates(
				const DataMatrix &kNormalizedData_,
				DataMatrix &kData_) const override
			{
				if (kData_.rows() != kNormalizedData_.rows())
					kData_.resize(kNormalizedData_.rows(), kNormalizedData_.cols());
				
				// Get the camera parameters
				const double &focalLength = parameters[0];
				const double &cx = parameters[1];
				const double &cy = parameters[2];

				// Normalize the points
				for (size_t pointIdx = 0; pointIdx < kNormalizedData_.rows(); ++pointIdx)
				{
					// Get the point
					double x = kNormalizedData_(pointIdx, 0);
					double y = kNormalizedData_(pointIdx, 1);
					
					// Apply radial distortion and scale the point
					x = x * focalLength + cx;
					y = y * focalLength + cy;

					// Store the normalized point
					kData_(pointIdx, 0) = x;
					kData_(pointIdx, 1) = y;

					// Copy the rest of the columns
					for (size_t i = 2; i < kData_.cols(); ++i)
						kData_(pointIdx, i) = kNormalizedData_(pointIdx, i);
				}
			}
			
			// Project a 2D point from pixel to image coordinates
            void fromPixelToImageCoordinates(
				const DataMatrix &kData_,
				DataMatrix &kNormalizedData_) const override
			{
				if (kData_.rows() != kNormalizedData_.rows())
					kNormalizedData_.resize(kData_.rows(), kData_.cols());
				
				// Get the camera parameters
				const double &focalLength = parameters[0];
				const double &cx = parameters[1];
				const double &cy = parameters[2];

				// Normalize the points
				for (size_t pointIdx = 0; pointIdx < kData_.rows(); ++pointIdx)
				{
					// Get the point
					double x = kData_(pointIdx, 0);
					double y = kData_(pointIdx, 1);

					// Normalize the point
					x = (x - cx) / focalLength;
					y = (y - cy) / focalLength;

					// Store the normalized point
					kNormalizedData_(pointIdx, 0) = x;
					kNormalizedData_(pointIdx, 1) = y;

					// Copy the rest of the columns
					for (size_t i = 2; i < kData_.cols(); ++i)
						kNormalizedData_(pointIdx, i) = kData_(pointIdx, i);
				}
			}
			
            double normalizeThreshold(double kThreshold_) const override
			{
				return kThreshold_ / parameters[0];
			} 
			
            double unnormalizeThreshold(double kThreshold_) const override
			{
				return kThreshold_ * parameters[0];
			} 

            bool validateParameters(const std::vector<double> &kParams_) const override
			{
				return kParams_.size() == 3;
			}

			size_t getModelId() const override
			{
				return 0;
			}
		};
	}
}
