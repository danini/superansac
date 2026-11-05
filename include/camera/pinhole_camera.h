// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// Author: Daniel Barath (majti89@gmail.com)
#pragma once

#include <stdexcept>
#include <vector>
#include <Eigen/Eigen>
#include "../utils/types.h"
#include "../models/model.h"
#include "abstract_camera.h"

namespace superansac
{
	namespace camera
	{
		class PinholeCamera : public AbstractCamera
		{
		public:
			PinholeCamera(const std::vector<double> &kParams_) :
				AbstractCamera(kParams_)
			{
				if (!validateParameters(kParams_))
					throw std::invalid_argument("The number of parameters for the pinhole camera should be 4 (fx, fy, cx, cy).");
			}

			~PinholeCamera() override = default;

			double focalLength() const override
			{
				return 0.5 * (parameters[0] + parameters[1]);
			}

			void rescale(double kScale_) override
			{
				parameters = {
					parameters[0] * kScale_,
					parameters[1] * kScale_,
					parameters[2] * kScale_,
					parameters[3] * kScale_
				};
			}

			void fromImageToPixelCoordinates(
				const std::vector<Eigen::Vector2d> &kNormalizedData_,
				std::vector<Eigen::Vector2d> &kData_) const override
			{
				if (kData_.size() != kNormalizedData_.size())
					kData_.resize(kNormalizedData_.size());

				const double &fx = parameters[0];
				const double &fy = parameters[1];
				const double &cx = parameters[2];
				const double &cy = parameters[3];

				for (size_t pointIdx = 0; pointIdx < kNormalizedData_.size(); ++pointIdx)
				{
					kData_[pointIdx](0) = kNormalizedData_[pointIdx](0) * fx + cx;
					kData_[pointIdx](1) = kNormalizedData_[pointIdx](1) * fy + cy;
				}
			}

			void fromImageToPixelCoordinates(
				const DataMatrix &kNormalizedData_,
				DataMatrix &kData_) const override
			{
				if (kData_.rows() != kNormalizedData_.rows() || kData_.cols() != kNormalizedData_.cols())
					kData_.resize(kNormalizedData_.rows(), kNormalizedData_.cols());

				const double &fx = parameters[0];
				const double &fy = parameters[1];
				const double &cx = parameters[2];
				const double &cy = parameters[3];

				for (Eigen::Index pointIdx = 0; pointIdx < kNormalizedData_.rows(); ++pointIdx)
				{
					kData_(pointIdx, 0) = kNormalizedData_(pointIdx, 0) * fx + cx;
					kData_(pointIdx, 1) = kNormalizedData_(pointIdx, 1) * fy + cy;
					for (Eigen::Index i = 2; i < kData_.cols(); ++i)
						kData_(pointIdx, i) = kNormalizedData_(pointIdx, i);
				}
			}

			void fromPixelToImageCoordinates(
				const DataMatrix &kData_,
				DataMatrix &kNormalizedData_) const override
			{
				if (kData_.rows() != kNormalizedData_.rows() || kData_.cols() != kNormalizedData_.cols())
					kNormalizedData_.resize(kData_.rows(), kData_.cols());

				const double invFx = 1.0 / parameters[0];
				const double invFy = 1.0 / parameters[1];
				const double &cx = parameters[2];
				const double &cy = parameters[3];

				for (Eigen::Index pointIdx = 0; pointIdx < kData_.rows(); ++pointIdx)
				{
					kNormalizedData_(pointIdx, 0) = (kData_(pointIdx, 0) - cx) * invFx;
					kNormalizedData_(pointIdx, 1) = (kData_(pointIdx, 1) - cy) * invFy;
					for (Eigen::Index i = 2; i < kData_.cols(); ++i)
						kNormalizedData_(pointIdx, i) = kData_(pointIdx, i);
				}
			}

			double normalizeThreshold(double kThreshold_) const override
			{
				return kThreshold_ / focalLength();
			}

			double unnormalizeThreshold(double kThreshold_) const override
			{
				return kThreshold_ * focalLength();
			}

			bool validateParameters(const std::vector<double> &kParams_) const override
			{
				return kParams_.size() == 4;
			}

			size_t getModelId() const override
			{
				return 1;
			}
		};
	}
}

