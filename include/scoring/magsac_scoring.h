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

#include "../utils/macros.h"
#include "../models/model.h"
#include "../estimators/abstract_estimator.h"
#include "../utils/types.h"
#include "abstract_scoring.h"
#include "score.h"
#include <Eigen/Core>
#include "magsac_look_up_table.h"
#include <boost/math/special_functions/gamma.hpp>
#include <fstream>

namespace superansac {
namespace scoring {

class MAGSACScoring : public AbstractScoring
{
    protected:
        static constexpr bool kUseLookUpTable = true;
        static constexpr bool kGenerateLookUpTable = false;

        size_t degreesOfFreedom;
        size_t dofIndex_;  // Cached DOF index for lookup table
        double k;
        double Cn;
        double squaredSigmaMax;
        double squaredSigmaMaxPerTwo;
        double squaredSigmaMaxPerFour;
        double twoTimesSquaredSigmaMax;
        double invTwoTimesSquaredSigmaMax;  // Precomputed inverse for optimization
        double zeroResidualLoss;
        double nPlus1Per2;
        double nMinus1Per2;
        double twoNPlus1;
        double lossOutlier;
        double premultiplier;
        double value0;
        double squaredTruncatedThreshold;
        double weightPremultiplier;
        
        FORCE_INLINE void updateSPRTParameters(const Score& currentBest, 
            int iterationIndex, 
            size_t totalPoints)
        {
            
        }
        
        double upperIncompleteGamma(double a, double x) const
        {
            // boost::math::tgamma and boost::math::tgamma_upper
            // T(a, x) = tgamma(a) - tgamma_lower(a, x)
            return boost::math::tgamma(a) - boost::math::tgamma_lower(a, x);
        }

        FORCE_INLINE std::pair<double, double> getGammaValues(double residual_) const
        {
            size_t index = static_cast<size_t>(residual_ * lookupTableSize);
            index = (index < lookupTableSize) ? index : (lookupTableSize - 1);
            // Use cached dofIndex_ to reduce repeated computation
            return { lowerIncompleteGammaLookupTable[dofIndex_][index], upperIncompleteGammaLookupTable[dofIndex_][index] };
        }

        FORCE_INLINE double getUpperGammaValue(double residual_) const
        {
            size_t index = static_cast<size_t>(residual_ * lookupTableSize);
            index = (index < lookupTableSize) ? index : (lookupTableSize - 1);
            return upperIncompleteGammaLookupTable[dofIndex_][index];
        }

        FORCE_INLINE double getLowerGammaValue(double residual_) const
        {
            size_t index = static_cast<size_t>(residual_ * lookupTableSize);
            index = (index < lookupTableSize) ? index : (lookupTableSize - 1);
            return lowerIncompleteGammaLookupTable[dofIndex_][index];
        }

    public:
        // Constructor 
        MAGSACScoring()
        {
            
        }

        // Destructor
        ~MAGSACScoring() {}

        static constexpr double getOutlierLoss(const size_t &kDegreesOfFreedom_)
        {
            switch (kDegreesOfFreedom_)
            {
                case 2:
                    return 0.215658; // 0.220642416155;
                case 3:
                    return 0.306123;
                case 4:
                    return 0.488088;
                case 5:
                    return 0.921592;
                case 6:
                    return 2.03833;
                default: 
                    throw std::runtime_error("The degrees of freedom is not supported.");
            }
        }

        static constexpr double getK(const size_t &kDegreesOfFreedom_)
        {
            switch (kDegreesOfFreedom_)
            {
                case 2:
                    return 3.034798181;
                case 3:
                    return 3.367491648;
                case 4:
                    return 3.644173432;
                case 5:
                    return 3.88458492;
                case 6:
                    return 4.1;
                default: 
                    throw std::runtime_error("The degrees of freedom is not supported.");
            }
        }

        static constexpr double getSubstractTerm(const size_t &kDegreesOfFreedom_)
        {
            switch (kDegreesOfFreedom_)
            {
                case 2:
                    return 0.00426624; 
                case 3:
                    return 0.00344787;
                case 4:
                    return 0.00360571;
                case 5:
                    return 0.00451815;
                case 6:
                    return 0.00648;
                default: 
                    throw std::runtime_error("The degrees of freedom is not supported.");
            }
        }

        // Initialize the gamma lookup table
        void initialize(const estimator::Estimator *kEstimator_)
        {
            initialize(kEstimator_->getDegreesOfFreedom());
        }
        
        // Initialize the gamma lookup table
        void initialize(const size_t kDegreesOfFreedom_)
        {
            if (threshold == 0.0)
                throw std::runtime_error("The threshold is not set for the MAGSAC scoring object.");
            degreesOfFreedom = kDegreesOfFreedom_; // Degrees of freedom
            dofIndex_ = degreesOfFreedom - 2;  // Cache DOF index for lookup table optimization
            k = getK(degreesOfFreedom); //kEstimator_->getK(); // The 0.99 quantile of the distribution
            //std::cout << degreesOfFreedom << std::endl;
            Cn = 1.0 / std::pow(2, degreesOfFreedom / 2.0) * boost::math::tgamma(degreesOfFreedom / 2.0); // Normalization constant
            squaredSigmaMax = threshold * threshold; // The squared threshold
            squaredSigmaMaxPerTwo = squaredSigmaMax / 2.0; // The squared threshold divided by two
            squaredSigmaMaxPerFour = squaredSigmaMaxPerTwo / 2.0; // The squared threshold divided by four
            twoTimesSquaredSigmaMax = 2.0 * squaredSigmaMax; // Two times the squared threshold
            invTwoTimesSquaredSigmaMax = 1.0 / twoTimesSquaredSigmaMax; // Precomputed inverse for optimization
            nPlus1Per2 = (degreesOfFreedom + 1) / 2.0; // (n + 1) / 2
            nMinus1Per2 = (degreesOfFreedom - 1) / 2.0; // (n - 1) / 2
            twoNPlus1 = std::pow(2.0, nPlus1Per2); // 2 ^ ((n + 1) / 2)
            premultiplier = 1.0 / threshold * Cn * twoNPlus1; // The premultiplier
            //value0 = upperIncompleteGamma(nMinus1Per2, k * k / 2.0); // The value of the upper incomplete gamma function at k * k / 2
            //std::cout << "value0: " << value0 << std::endl;
            value0 = getSubstractTerm(degreesOfFreedom); // The value of the upper incomplete gamma function at k * k / 2
            //std::cout << "value0: " << value0 << std::endl;
            //std::cout << "value0: " << value0 << std::endl;
            squaredTruncatedThreshold = k * k * squaredSigmaMax; // The squared truncated threshold
            weightPremultiplier = 1.0 / threshold * Cn * nMinus1Per2; // The weight premultiplier
            //lossOutlier = threshold * Cn * nMinus1Per2 * boost::math::tgamma_lower(nPlus1Per2, k * k / 2.0); // The loss of an outlier
            //std::cout << "Loss outlier: " << Cn * nMinus1Per2 * boost::math::tgamma_lower(nPlus1Per2, k * k / 2.0) << std::endl;   
            //std::cout << std::setprecision(12) << Cn * nMinus1Per2 * boost::math::tgamma_lower(nPlus1Per2, k * k / 2.0) << std::endl;
            //lossOutlier = lossOutlier / premultiplier; // Normalize the loss of an outlier by the premultiplier which we will not multiply the loss with
            lossOutlier = threshold * getOutlierLoss(degreesOfFreedom); // The loss of an outlier
            //std::cout << "Loss outlier: " << getOutlierLoss(degreesOfFreedom) << std::endl;   
            
            const auto &zeroGammaValues = getGammaValues(0.0); // Get the gamma values
            zeroResidualLoss = squaredSigmaMaxPerTwo * zeroGammaValues.first + squaredSigmaMaxPerFour * (zeroGammaValues.second - value0); 

            // Initialize the lookup table
            if constexpr (kGenerateLookUpTable)
            {
                std::vector<size_t> degreesOfFreedomToGenerate = { 2, 3, 4, 5, 6 };
                size_t lookupTableSize = 10000; // The size of the lookup table

                // Save these values to a file
                std::ofstream file("include/scoring/magsac_look_up_table.h"); // Open the file
                file << "namespace superansac {\nnamespace scoring {\n"; // Write the namespace
                file << "static constexpr size_t lookupTableSize = " << lookupTableSize << ";\n"; // Write the size of the lookup table
                
                file << "static constexpr double upperIncompleteGammaLookupTable[" << degreesOfFreedomToGenerate.size() << "][lookupTableSize] = {"; // Write the upper incomplete gamma lookup table
                
                for (size_t dof : degreesOfFreedomToGenerate)
                {
                    std::vector<double> upperIncompleteGammaLookupTable; // The lookup table for the upper incomplete gamma function
                    upperIncompleteGammaLookupTable.resize(lookupTableSize); // Resize the lookup table

                    // Fill the lookup table
                    for (size_t i = 0; i < lookupTableSize; ++i)
                    {
                        double value = static_cast<double>(i) / lookupTableSize; // The value for which the incomplete gamma function is calculated
                        upperIncompleteGammaLookupTable[i] = upperIncompleteGamma((dof - 1.0) / 2.0, value); // Calculate the upper incomplete gamma function
                    }

                    file << "{"; // Write the upper incomplete gamma lookup table
                    // Write the values of the upper incomplete gamma lookup table
                    for (size_t i = 0; i < lookupTableSize; ++i)
                    {
                        // Change precision
                        file << std::setprecision(8) << upperIncompleteGammaLookupTable[i];
                        if (i < lookupTableSize - 1)
                            file << ", ";
                    }
                    file << "}, \n";
                }
                file << "};\n";

                // Write the values of the lower incomplete gamma lookup table
                file << "static constexpr double lowerIncompleteGammaLookupTable[" << degreesOfFreedomToGenerate.size() << "][lookupTableSize] = {";
                for (size_t dof : degreesOfFreedomToGenerate)
                {
                    std::vector<double> lowerIncompleteGammaLookupTable; // The lookup table for the lower incomplete gamma function
                    lowerIncompleteGammaLookupTable.resize(lookupTableSize); // Resize the lookup table

                    // Fill the lookup table
                    for (size_t i = 0; i < lookupTableSize; ++i)
                    {
                        double value = static_cast<double>(i) / lookupTableSize; // The value for which the incomplete gamma function is calculated
                        lowerIncompleteGammaLookupTable[i] = boost::math::tgamma_lower((dof + 1.0) / 2.0, value); // Calculate the lower incomplete gamma function
                    }

                    file << "{"; // Write the lower incomplete gamma lookup table
                    // Write the values of the lower incomplete gamma lookup table
                    for (size_t i = 0; i < lookupTableSize; ++i)
                    {
                        file << std::setprecision(8) << lowerIncompleteGammaLookupTable[i];
                        if (i < lookupTableSize - 1)
                            file << ", ";
                    }
                    file << "}, \n";
                }
                file << "}; }}\n"; // Close the namespace
                file.close(); // Close the file

                std::cout << "Lookup table saved." << std::endl;

                // Exit program after generating lookup table
                std::exit(0);
            }
        }

        // Set the threshold
        FORCE_INLINE void setThreshold(const double kThreshold_)
        {
            threshold = kThreshold_; // Set the threshold
            squaredThreshold = threshold * threshold; // Set the squared threshold
        }

        // Loss function
        FORCE_INLINE double getLoss(const double &kSquaredResidual_) const
        {
            double loss = 0;
            // If the residual is smaller than the threshold, store it as an inlier and
            // increase the score.
            if (kSquaredResidual_ < squaredThreshold)
            {
                // Increase the score (use precomputed inverse for optimization)
                double residualPerTwoTimesSquaredSigmaMax = kSquaredResidual_ * invTwoTimesSquaredSigmaMax;
                // Calculate the loss by using a look-up table or by calculating the incomplete gamma function
                if constexpr (kUseLookUpTable)
                {
                    std::pair<double, double> gammaValues = getGammaValues(residualPerTwoTimesSquaredSigmaMax); // Get the gamma values
                    loss = squaredSigmaMaxPerTwo * gammaValues.first + squaredSigmaMaxPerFour * (gammaValues.second - value0); 
                } else // Calculate the loss directly by using the incomplete gamma function
                    loss = (squaredSigmaMaxPerTwo * boost::math::tgamma_lower(nPlus1Per2, residualPerTwoTimesSquaredSigmaMax) +
                        squaredSigmaMaxPerFour * (upperIncompleteGamma(nMinus1Per2, residualPerTwoTimesSquaredSigmaMax) - value0)); 

                // Commenting "premultiplier" as it does not affect the final result. It is just a constant.
                loss = premultiplier * loss; // Increase the loss value
            } else
                loss = lossOutlier;
            return loss;
        }

        // Loss function
        FORCE_INLINE double getWeight(const double &kSquaredResidual_) const
        {
            // If the residual is smaller than the threshold, store it as an inlier and
            // increase the score.
            if (kSquaredResidual_ < squaredThreshold)
            {
                double residualPerTwoTimesSquaredSigmaMax = kSquaredResidual_ * invTwoTimesSquaredSigmaMax;
                double upperIncompleteGammaValue = getUpperGammaValue(residualPerTwoTimesSquaredSigmaMax);
                return weightPremultiplier * (upperIncompleteGammaValue - value0);
            } 

            return 0.0;
        }

        // Sample function
        FORCE_INLINE Score score(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<size_t> &inliers_, // Inlier indices
            const bool kStoreInliers_ = true,
            const Score& kBestScore_ = Score(),
            std::vector<const std::vector<size_t>*> *kPotentialInlierSets_ = nullptr) const // The potential inlier sets from the inlier selector
        {   
            // Create a static empty Score
            static const Score kEmptyScore;
            // The number of points
            const int kPointNumber = kData_.rows();
            // The squared residual
            double squaredResidual;
            // Score and inlier number
            int inlierNumber = 0;
            double scoreValue = 0.0;
            // The score of the previous best model
            const double kBestInlierNumber = kBestScore_.getInlierNumber();
            double residualPerTwoTimesSquaredSigmaMax,
                loss;
            std::pair<double, double> gammaValues;

            if (kPotentialInlierSets_ != nullptr)
            {
                const double kBestScoreValue = kBestScore_.getValue();
                const double kBestPossibleGain = premultiplier * zeroResidualLoss;
                const double kInvTwoTimesSquaredSigmaMax = 1.0 / twoTimesSquaredSigmaMax;

                // Process potential inlier sets
                size_t testedPoints = 0;
                for (const auto &potentialInlierSet : *kPotentialInlierSets_)
                {
                    // Increase the number of tested points
                    testedPoints += potentialInlierSet->size();

                    for (const auto &pointIdx : *potentialInlierSet)
                    {
                        // Calculate the point-to-model residual
                        squaredResidual =
                            kEstimator_->squaredResidual(kData_.row(pointIdx),
                                kModel_);

                        // If the residual is smaller than the threshold, store it as an inlier and
                        // increase the score.
                        if (squaredResidual < squaredThreshold)
                        {
                            if (kStoreInliers_) // Store the point as an inlier if needed.
                                inliers_.emplace_back(pointIdx);

                            // Increase the inlier number
                            ++inlierNumber;

                            residualPerTwoTimesSquaredSigmaMax = squaredResidual * kInvTwoTimesSquaredSigmaMax;
                            // Calculate the loss by using a look-up table or by calculating the incomplete gamma function
                            if constexpr (kUseLookUpTable)
                            {
                                gammaValues = getGammaValues(residualPerTwoTimesSquaredSigmaMax); // Get the gamma values
                                loss = squaredSigmaMaxPerTwo * gammaValues.first + squaredSigmaMaxPerFour * (gammaValues.second - value0);
                            } else // Calculate the loss directly by using the incomplete gamma function
                                loss = (squaredSigmaMaxPerTwo * boost::math::tgamma_lower(nPlus1Per2, residualPerTwoTimesSquaredSigmaMax) +
                                    squaredSigmaMaxPerFour * (upperIncompleteGamma(nMinus1Per2, residualPerTwoTimesSquaredSigmaMax) - value0));

                            // Commenting "premultiplier" as it does not affect the final result. It is just a constant.
                            scoreValue += premultiplier * loss; // Increase the loss value
                        } else
                            scoreValue += lossOutlier;

                        if (kBestPossibleGain * (kPointNumber - pointIdx) + scoreValue < kBestScoreValue)
                            return kEmptyScore;
                    }
                }

                // Increase the score by the loss of the untested outliers
                scoreValue += (kData_.rows() - testedPoints) * lossOutlier;
            } else
            {
                // Pre-compute values for early exit optimization
                const double kBestPossibleGain = premultiplier * zeroResidualLoss;
                const double kBestScoreValue = kBestScore_.getValue();

                // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
                #ifdef __GNUC__
                #pragma GCC ivdep  // Tell compiler iterations are independent for vectorization
                #endif
                for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
                {
                    // Calculate the point-to-model residual
                    squaredResidual =
                        kEstimator_->squaredResidual(kData_.row(pointIdx),
                            kModel_);

                    // If the residual is smaller than the threshold, store it as an inlier and
                    // increase the score.
                    if (squaredResidual < squaredThreshold)
                    {
                        if (kStoreInliers_) // Store the point as an inlier if needed.
                            inliers_.emplace_back(pointIdx);

                        // Increase the inlier number
                        ++inlierNumber;

                        // Increase the score (use precomputed inverse for optimization)
                        residualPerTwoTimesSquaredSigmaMax = squaredResidual * invTwoTimesSquaredSigmaMax;
                        // Calculate the loss by using a look-up table or by calculating the incomplete gamma function
                        if constexpr (kUseLookUpTable)
                        {
                            gammaValues = getGammaValues(residualPerTwoTimesSquaredSigmaMax); // Get the gamma values
                            loss = squaredSigmaMaxPerTwo * gammaValues.first + squaredSigmaMaxPerFour * (gammaValues.second - value0);
                        } else // Calculate the loss directly by using the incomplete gamma function
                            loss = (squaredSigmaMaxPerTwo * boost::math::tgamma_lower(nPlus1Per2, residualPerTwoTimesSquaredSigmaMax) +
                                squaredSigmaMaxPerFour * (upperIncompleteGamma(nMinus1Per2, residualPerTwoTimesSquaredSigmaMax) - value0));

                        // Commenting "premultiplier" as it does not affect the final result. It is just a constant.
                        scoreValue += premultiplier * loss; // Increase the loss value
                    } else
                        scoreValue += lossOutlier;

                    // Early exit AFTER processing: if remaining perfect inliers can't beat best score
                    if (kBestPossibleGain * (kPointNumber - pointIdx - 1) + scoreValue < kBestScoreValue)
                        return kEmptyScore;
                }
            }
            
            return Score(inlierNumber, scoreValue);
        }

        // Get weights for the points
        FORCE_INLINE void getWeights(
            const DataMatrix &kData_, // Data matrix
            const models::Model &kModel_, // The model to be scored
            const estimator::Estimator *kEstimator_, // Estimator
            std::vector<double> &weights_, // The weights of the points
            const std::vector<size_t> *kIndices_ = nullptr) const  // The indices of the points
        {
            double residualPerTwoTimesSquaredSigmaMax,
                upperIncompleteGamma;

            if (kIndices_ == nullptr)
            {
                // The number of points
                const int kPointNumber = kData_.rows();
                // The squared residual
                double squaredResidual;
                // Allocate memory for the weights
                weights_.resize(kPointNumber);

                // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
                for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
                {
                    // Calculate the point-to-model residual
                    squaredResidual =
                        kEstimator_->squaredResidual(kData_.row(pointIdx),
                            kModel_);
                            
                    // If the residual is smaller than the threshold, store it as an inlier and
                    // increase the score.
                    if (squaredResidual < squaredThreshold)
                    {
                        residualPerTwoTimesSquaredSigmaMax = squaredResidual * invTwoTimesSquaredSigmaMax;
                        upperIncompleteGamma = getUpperGammaValue(residualPerTwoTimesSquaredSigmaMax);

                        weights_[pointIdx] = weightPremultiplier * (upperIncompleteGamma - value0);
                        // Commenting "weightPremultiplier" as it does not affect the final result. It is just a constant.
                        //weights_[pointIdx] = upperIncompleteGamma - value0;
                    } else
                        weights_[pointIdx] = 0.0;
                }
            }
            else
            {
                // The number of points
                const int kPointNumber = kIndices_->size();
                // The squared residual
                double squaredResidual;
                // Allocate memory for the weights
                weights_.resize(kPointNumber);

                // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
                for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
                {
                    // Calculate the point-to-model residual
                    squaredResidual =
                        kEstimator_->squaredResidual(kData_.row((*kIndices_)[pointIdx]),
                            kModel_);
                            
                    // If the residual is smaller than the threshold, store it as an inlier and
                    // increase the score.
                    if (squaredResidual < squaredThreshold)
                    {
                        residualPerTwoTimesSquaredSigmaMax = squaredResidual * invTwoTimesSquaredSigmaMax;
                        upperIncompleteGamma = getUpperGammaValue(residualPerTwoTimesSquaredSigmaMax);
                        weights_[pointIdx] = weightPremultiplier * (upperIncompleteGamma - value0);
                        // Commenting "weightPremultiplier" as it does not affect the final result. It is just a constant.
                        //weights_[pointIdx] = upperIncompleteGamma - value0;
                    } else
                        weights_[pointIdx] = 0.0;
                }
            }
        }
};

}
}