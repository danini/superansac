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

#include <Eigen/Dense>
#include <vector>
#include "../utils/types.h"

namespace superansac {
namespace models {

enum Types
{
    Homography,
    FundamentalMatrix,
    EssentialMatrix,
    RigidTransformation,
    AbsolutePose
};

// This is the model class. It is used to store the data and the model parameters.
class Model {
public:
    Model () {}
    ~Model() {}

    // Set the data of the model. Accepting the fixed-capacity type means
    // expressions are evaluated into a stack temporary (no aliasing issues,
    // no heap allocation).
    void setData(const ModelMatrix &kData_) {
        data = kData_;
        hasPoseFlag = false;
    }

    // Return the data of the model
    const ModelMatrix &getData() const {
        return data;
    }

    // Return the data of the model
    ModelMatrix &getMutableData() {
        hasPoseFlag = false;
        return data;
    }

    bool hasPose() const {
        return hasPoseFlag;
    }

    const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> &getPose() const {
        return poseRt;
    }

    void setPose(const Eigen::Ref<const Eigen::Matrix<double, 3, 4>> &kPoseRt_) {
        poseRt = kPoseRt_;
        hasPoseFlag = true;
    }

    void clearPose() {
        hasPoseFlag = false;
    }

protected:
    ModelMatrix data;
    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> poseRt =
        Eigen::Matrix<double, 3, 4, Eigen::RowMajor>::Zero();
    bool hasPoseFlag = false;
};

}
}