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

#include "abstract_type.h"

class PointCorrespondence2D2D : public AbstractData
{
    protected:
        // 2D point in the first image
        double x1, y1;
        // 2D point in the second image
        double x2, y2;

    public:
        // Constructor 
        PointCorrespondence2D2D() : x1(0.0), y1(0.0), x2(0.0), y2(0.0)
        {
            
        }

        // Constructor
        PointCorrespondence2D2D(
            const double x1_, 
            const double y1_, 
            const double x2_, 
            const double y2_) : x1(x1_), y1(y1_), x2(x2_), y2(y2_)
        {
            
        }

        // Destructor
        ~PointCorrespondence2D2D() {}

        // Getter for the 2D point in the first image
        FORCE_INLINE double getX1() const
        {
            return x1;
        }

        // Getter for the 2D point in the first image
        FORCE_INLINE double getY1() const
        {
            return y1;
        }

        // Getter for the 2D point in the second image
        FORCE_INLINE double getX2() const
        {
            return x2;
        }

        // Getter for the 2D point in the second image
        FORCE_INLINE double getY2() const
        {
            return y2;
        }

        // Getter for mutable 2D point in the first image
        FORCE_INLINE double& getX1()
        {
            return x1;
        }

        // Getter for mutable 2D point in the first image
        FORCE_INLINE double& getY1()
        {
            return y1;
        }

        // Getter for mutable 2D point in the second image
        FORCE_INLINE double& getX2()
        {
            return x2;
        }

        // Getter for mutable 2D point in the second image
        FORCE_INLINE double& getY2()
        {
            return y2;
        }
}; 