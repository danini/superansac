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

#include <Eigen/Eigen>

namespace superansac
{
	namespace utils
	{
		// Pivoting In-Place Gauss Elimination to solve problem A * x = b,
		// where A is the known coefficient matrix, b is the inhomogeneous part and x is the unknown vector.
		// Form: matrix_ = [A, b].
		template<size_t _Size>
		void gaussElimination(
			Eigen::Matrix<double, _Size, _Size + 1>& matrix_, // The matrix to which the elimination is applied
			Eigen::Matrix<double, _Size, 1>& result_) // The resulting null-space
		{
			/*int i, j, k;
			double temp;

			//Pivotisation
			for (i = 0; i < _Size; i++)                    
				for (k = i + 1; k < _Size; k++)
					if (abs(matrix_(i, i)) < abs(matrix_(k, i)))
						for (j = 0; j <= _Size; j++)
						{
							temp = matrix_(i, j);
							matrix_(i, j) = matrix_(k, j);
							matrix_(k, j) = temp;
						}

			//loop to perform the gauss elimination
			for (i = 0; i < _Size - 1; i++)            
				for (k = i + 1; k < _Size; k++)
				{
					double temp = matrix_(k, i) / matrix_(i, i);
					for (j = 0; j <= _Size; j++)
						// make the elements below the pivot elements equal to zero or elimnate the variables
						matrix_(k, j) = matrix_(k, j) - temp * matrix_(i, j);    
				}

			//back-substitution
			for (i = _Size - 1; i >= 0; i--)                
			{                       
				// result_ is an array whose values correspond to the values of x,y,z..
				result_(i) = matrix_(i, _Size);                
				//make the variable to be calculated equal to the rhs of the last equation
				for (j = i + 1; j < _Size; j++)
					if (j != i)            
						//then subtract all the lhs values except the coefficient of the variable whose value is being calculated
						result_(i) = result_(i) - matrix_(i, j) * result_(j);
				//now finally divide the rhs by the coefficient of the variable to be calculated
				result_(i) = result_(i) / matrix_(i, i);            
			}*/

			// Pivotisation
			for (size_t i = 0; i < _Size; ++i)
			{
				// Find the row with the largest pivot element
				size_t maxRow = i;
				for (size_t k = i + 1; k < _Size; ++k)
				{
					if (std::abs(matrix_(k, i)) > std::abs(matrix_(maxRow, i)))
					{
						maxRow = k;
					}
				}

				// Swap the current row with the row with the largest pivot
				if (maxRow != i)
				{
					matrix_.row(i).swap(matrix_.row(maxRow));
				}

				// Elimination process
				for (size_t k = i + 1; k < _Size; ++k)
				{
					double factor = matrix_(k, i) / matrix_(i, i);
					matrix_.row(k).segment(i, _Size + 1 - i) -= factor * matrix_.row(i).segment(i, _Size + 1 - i);
				}
			}

			// Back-substitution
			for (int i = _Size - 1; i >= 0; --i)
			{
				result_(i) = matrix_(i, _Size);
				for (int j = i + 1; j < _Size; ++j)
				{
					result_(i) -= matrix_(i, j) * result_(j);
				}
				result_(i) /= matrix_(i, i);
			}
		}
    }
}