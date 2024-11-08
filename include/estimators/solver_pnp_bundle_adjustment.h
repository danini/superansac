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

#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../camera/types.h"
#include "solver_epnp.h"

#include "numerical_optimizer/bundle.h"
#include "numerical_optimizer/camera_pose.h"
#include "numerical_optimizer/essential.h"

namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class PnPBundleAdjustmentSolver : public AbstractSolver
			{
			protected:
				poselib::BundleOptions options;
				superansac::camera::AbstractCamera *camera;
				const EPnPSolver epnpSolver;

			public:
				PnPBundleAdjustmentSolver() : camera(nullptr)
				{
				}

				~PnPBundleAdjustmentSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				bool returnMultipleModels() const override
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				size_t maximumSolutions() const override
				{
					return 1;
				}
				
				// The minimum number of points required for the estimation
				size_t sampleSize() const override
				{
					return 3;
				}

				poselib::BundleOptions &getMutableOptions()
				{
					return options;
				}

				void setCamera(superansac::camera::AbstractCamera *kCamera_)
				{
					camera = kCamera_;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				FORCE_INLINE bool estimateModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_ = nullptr) const override; // The weight for each point
			};
			
			FORCE_INLINE bool PnPBundleAdjustmentSolver::estimateModel(
				const DataMatrix& kData_, // The set of data points
				const size_t *kSample_, // The sample used for the estimation
				const size_t kSampleNumber_, // The size of the sample
				std::vector<models::Model> &models_, // The estimated model parameters
				const double *kWeights_) const // The weight for each point
			{			
				// Check whether the camera has been set
				if (camera == nullptr)
					throw std::runtime_error("The PnPBundleAdjustmentSolver requires a camera to be set.");

				// Check if we have enough points for the bundle adjustment
				if (kSampleNumber_ < sampleSize())
					return false;

				// The point correspondences
				std::vector<Eigen::Vector2d> points2d(kSampleNumber_),
					unnormalizedPoint2d; 
				std::vector<Eigen::Vector3d> points3d(kSampleNumber_); 

				// Filling the point correspondences if the sample is not provided
				if (kSample_ == nullptr)
				{
					// Filling the point correspondences
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						points2d[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 0), kData_(pointIdx, 1));
						points3d[pointIdx] = Eigen::Vector3d(kData_(pointIdx, 2), kData_(pointIdx, 3), kData_(pointIdx, 4));
					}
				} else // Filling the point correspondences if the sample is provided
				{
					for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++)
					{
						const size_t& idx = kSample_[pointIdx];
						points2d[pointIdx] = Eigen::Vector2d(kData_(idx, 0), kData_(idx, 1));
						points3d[pointIdx] = Eigen::Vector3d(kData_(idx, 2), kData_(idx, 3), kData_(idx, 4));
					}
				}
				
				if (models_.size() == 0)
				{
					/*P3PLambdaTwistSolver p3pSolver;
					samplers::UniformRandomSampler sampler;
					sampler.initialize(kSampleNumber_);
					size_t localSample[3];

					for (size_t iter = 0; iter < 10; ++iter)
					{
						sampler.sample(kSampleNumber_, 3, localSample);
						localSample[0] = kSample_[localSample[0]];
						localSample[1] = kSample_[localSample[1]];
						localSample[2] = kSample_[localSample[2]];
						p3pSolver.estimateModel(kData_, localSample, 3, models_);
					}*/

					epnpSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);

					if (models_.size() == 0)
						return false;
				}

				// The options for the bundle adjustment
				poselib::BundleOptions tmpOptions = options;
				// If the sample is provided, we use a more robust loss function. This typically runs in the end of the robust estimation
				if (kSample_ != nullptr) 
				{
					tmpOptions.loss_scale = 1.0 * options.loss_scale;
					//tmpOptions.loss_scale = 0.5 * options.loss_scale;
					tmpOptions.max_iterations = 100;
					tmpOptions.loss_type = poselib::BundleOptions::LossType::CAUCHY;
				
					/*// Unnormalize the points by the focal length to numerical stability
					camera->fromImageToPixelCoordinates(points2d, unnormalizedPoint2d);

					// Unnormalize the threshold
					tmpOptions.loss_scale = camera->unnormalizeThreshold(tmpOptions.loss_scale);*/
				}
				
				// The pose with the lowest cost
				double bestCost = std::numeric_limits<double>::max();
				poselib::CameraPose bestPose;

				// Iterating through the potential models.
				for (auto& model : models_)
				{
					// Get the pose parameters
					Eigen::Matrix3d R = model.getData().block<3, 3>(0, 0);
					Eigen::Vector3d t = model.getData().block<3, 1>(0, 3);

					// Decompose the essential matrix to camera poses
					poselib::CameraPose pose(R, t);
					
					// Perform the bundle adjustment
					poselib::BundleStats stats;

					//if (kSample_ != nullptr) 
					poselib::bundle_adjust(points2d, points3d, &pose, tmpOptions);
					/*else
					{
						poselib::Camera dummyCamera;
						dummyCamera.model_id = camera->getModelId();
						dummyCamera.params = camera->getParameters();
						poselib::bundle_adjust(unnormalizedPoint2d, points3d, dummyCamera, &pose, tmpOptions);
						
					}*/

					if (stats.cost < bestCost)
					{
						bestCost = stats.cost;
						bestPose = pose;
					}
				}
				
				// Composing the essential matrix from the pose
				if (bestCost < std::numeric_limits<double>::max())
				{
					// Adding the essential matrix as the estimated models.
					models_.resize(1);
					auto &modelData = models_[0].getMutableData();
					modelData = bestPose.Rt();
				}

				return models_.size();
			}
		}
	}
}