// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// MAGSAC + SPRT scoring. API-compatible with MAGSACScoring but performs
// SPRT-based preemptive verification while keeping MAGSAC loss.
//
// Please contact the author if you have any questions.
// Author of original MAGSAC: Daniel Barath (majti89@gmail.com)
// SPRT adaptation: <your name/email>
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
#include <chrono>
#include <random>
#include <numeric>
#include <cmath>

namespace superansac {
namespace scoring {

class MAGSACSPRTScoring : public AbstractScoring
{
protected:
    // ====== MAGSAC core (unchanged from MAGSACScoring) ======
    static constexpr bool kUseLookUpTable = true;

    size_t degreesOfFreedom = 0;
    double k = 0.0;
    double Cn = 0.0;
    double squaredSigmaMax = 0.0;
    double squaredSigmaMaxPerTwo = 0.0;
    double squaredSigmaMaxPerFour = 0.0;
    double twoTimesSquaredSigmaMax = 0.0;
    double zeroResidualLoss = 0.0;
    double nPlus1Per2 = 0.0;
    double nMinus1Per2 = 0.0;
    double twoNPlus1 = 0.0;
    double lossOutlier = 0.0;
    double premultiplier = 0.0;
    double value0 = 0.0;
    double squaredTruncatedThreshold = 0.0;
    double weightPremultiplier = 0.0;

    FORCE_INLINE std::pair<double,double> getGammaValues(double residual_) const {
        size_t idx = static_cast<size_t>(residual_ * lookupTableSize);
        if (idx >= lookupTableSize) idx = lookupTableSize - 1;
        return { lowerIncompleteGammaLookupTable[degreesOfFreedom - 2][idx],
                 upperIncompleteGammaLookupTable[degreesOfFreedom - 2][idx] };
    }
    FORCE_INLINE double getUpperGammaValue(double residual_) const {
        size_t idx = static_cast<size_t>(residual_ * lookupTableSize);
        if (idx >= lookupTableSize) idx = lookupTableSize - 1;
        return upperIncompleteGammaLookupTable[degreesOfFreedom - 2][idx];
    }

    static constexpr double getOutlierLoss(const size_t &dof)
    {
        switch (dof)
        {
            case 2: return 0.215658;
            case 3: return 0.306123;
            case 4: return 0.488088;
            case 5: return 0.921592;
            case 6: return 2.03833;
            default: throw std::runtime_error("Unsupported degrees of freedom.");
        }
    }
    static constexpr double getK(const size_t &dof)
    {
        switch (dof)
        {
            case 2: return 3.034798181;
            case 3: return 3.367491648;
            case 4: return 3.644173432;
            case 5: return 3.88458492;
            case 6: return 4.1;
            default: throw std::runtime_error("Unsupported degrees of freedom.");
        }
    }
    static constexpr double getSubtractTerm(const size_t &dof)
    {
        switch (dof)
        {
            case 2: return 0.00426624;
            case 3: return 0.00344787;
            case 4: return 0.00360571;
            case 5: return 0.00451815;
            case 6: return 0.00648;
            default: throw std::runtime_error("Unsupported degrees of freedom.");
        }
    }

    // ====== SPRT state ======
    struct SPRTHistory {
        double epsilon;  // P(inlier | good)
        double delta;    // P(inlier | bad)
        double A;        // LR threshold
    };

    // Tunables
    static constexpr bool   kUseRuntimeA   = false;  // set true to use K-based threshold
    static constexpr double kDefaultAlpha  = 0.05;
    static constexpr double kDefaultBeta   = 0.05;
    static constexpr double kMinEpsilon    = 1e-3;
    static constexpr double kMinDelta      = 1e-4;
    static constexpr double kMaxDeltaFrac  = 0.5;
    static constexpr double kUpdateTolFrac = 0.05;

    // Runtime model (used only if kUseRuntimeA)
    double tM_ms = 0.05;
    double mS    = 1.0;

    SPRTHistory sprt_{0.05, 0.005, 0.0};
    size_t lastUpdateIteration_ = 0;

    // Point order buffer
    mutable std::vector<size_t> perm_;
    mutable size_t permHead_ = 0;

    // Rejection statistics for delta update
    size_t rejectedCount_ = 0;
    double rejectedInlierFracSum_ = 0.0;

    // Permutation management
    void ensurePermutation(size_t N) const
    {
        if (perm_.size() != N) {
            perm_.resize(N);
            std::iota(perm_.begin(), perm_.end(), 0);
            std::mt19937 rng(0xC0FFEE);
            std::shuffle(perm_.begin(), perm_.end(), rng);
            permHead_ = 0;
        }
        if (permHead_ >= perm_.size()) permHead_ = 0;
    }

    static inline double clampProb(double x, double lo, double hi) {
        return std::max(lo, std::min(hi, x));
    }

    static inline double waldA(double alpha = kDefaultAlpha, double beta = kDefaultBeta) {
        return (1.0 - beta) / alpha;
    }

    static inline double informationC(double eps, double del) {
        return (1.0 - del) * std::log((1.0 - del) / (1.0 - eps)) + del * std::log(del / eps);
    }
    double estimateThresholdA_runtime(double eps, double del) const
    {
        const double C = informationC(eps, del);
        if (C <= 0.0) return waldA();
        const double K = (tM_ms * C) / std::max(1.0, mS) + 1.0;
        double A_prev = K, A = K;
        for (int i = 0; i < 10; ++i) {
            A = K + std::log(std::max(1e-12, A_prev));
            if (std::abs(A - A_prev) < 1.5e-8) break;
            A_prev = A;
        }
        return std::max(A, 1.0);
    }

    template <typename EstimatorT>
    void microBenchmarkResiduals(const DataMatrix &X, const EstimatorT *E, size_t trials = 256)
    {
        if (!kUseRuntimeA) return;
        const size_t n = static_cast<size_t>(X.rows());
        if (n == 0) return;
        trials = std::min(trials, n);
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile double sink = 0.0;
        for (size_t i = 0; i < trials; ++i) {
            sink += E->squaredResidual(X.row(static_cast<int>(i)), models::Model());
        }
        (void)sink;
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        tM_ms = std::max(0.01, ms / static_cast<double>(trials) * 4.0);
        mS    = 1.0;
    }

    void refreshA()
    {
        if (kUseRuntimeA) sprt_.A = estimateThresholdA_runtime(sprt_.epsilon, sprt_.delta);
        else              sprt_.A = waldA();
    }

    // MAGSAC loss and weight
    FORCE_INLINE double magsacLoss(const double &kSquaredResidual_) const
    {
        if (kSquaredResidual_ < squaredThreshold)
        {
            const double r = kSquaredResidual_ / twoTimesSquaredSigmaMax;
            double loss = 0.0;
            if constexpr (kUseLookUpTable) {
                const auto g = getGammaValues(r);
                loss = squaredSigmaMaxPerTwo * g.first + squaredSigmaMaxPerFour * (g.second - value0);
            }
            return premultiplier * loss;
        }
        return lossOutlier;
    }
    FORCE_INLINE double magsacWeight(const double &kSquaredResidual_) const
    {
        if (kSquaredResidual_ >= squaredThreshold) return 0.0;
        const double r = kSquaredResidual_ / twoTimesSquaredSigmaMax;
        return weightPremultiplier * (getUpperGammaValue(r) - value0);
    }

public:
    MAGSACSPRTScoring()
    {
        refreshA();
    }
    ~MAGSACSPRTScoring() {}

    // ====== Initialization (no 'override' here) ======
    void initialize(const estimator::Estimator *kEstimator_)
    {
        if (threshold == 0.0)
            throw std::runtime_error("Threshold must be set before initialize().");
        initialize(kEstimator_->getDegreesOfFreedom());
    }

    // Match MAGSACScoring’s overload
    void initialize(const size_t kDegreesOfFreedom_)
    {
        degreesOfFreedom = kDegreesOfFreedom_;
        k  = getK(degreesOfFreedom);
        Cn = 1.0 / std::pow(2.0, degreesOfFreedom / 2.0) * boost::math::tgamma(degreesOfFreedom / 2.0);
        squaredSigmaMax         = threshold * threshold;
        squaredSigmaMaxPerTwo   = squaredSigmaMax / 2.0;
        squaredSigmaMaxPerFour  = squaredSigmaMaxPerTwo / 2.0;
        twoTimesSquaredSigmaMax = 2.0 * squaredSigmaMax;
        nPlus1Per2  = (degreesOfFreedom + 1) / 2.0;
        nMinus1Per2 = (degreesOfFreedom - 1) / 2.0;
        twoNPlus1   = std::pow(2.0, nPlus1Per2);
        premultiplier       = 1.0 / threshold * Cn * twoNPlus1;
        value0              = getSubtractTerm(degreesOfFreedom);
        squaredTruncatedThreshold = k * k * squaredSigmaMax;
        weightPremultiplier = 1.0 / threshold * Cn * nMinus1Per2;
        lossOutlier         = threshold * getOutlierLoss(degreesOfFreedom);

        const auto zeroGammaValues = getGammaValues(0.0);
        zeroResidualLoss = squaredSigmaMaxPerTwo * zeroGammaValues.first
                         + squaredSigmaMaxPerFour * (zeroGammaValues.second - value0);
    }

    // ====== AbstractScoring API ======
    FORCE_INLINE void setThreshold(const double kThreshold_) override
    {
        threshold = kThreshold_;
        squaredThreshold = threshold * threshold;
    }

    FORCE_INLINE Score score(
        const DataMatrix &kData_,
        const models::Model &kModel_,
        const estimator::Estimator *kEstimator_,
        std::vector<size_t> &inliers_,
        const bool kStoreInliers_ = true,
        const Score& kBestScore_ = Score(),
        std::vector<const std::vector<size_t>*> *kPotentialInlierSets_ = nullptr
    ) const override
    {
        static const Score kEmptyScore;

        const int N = kData_.rows();
        if (N == 0) return Score();

        ensurePermutation(static_cast<size_t>(N));
        if constexpr (kUseRuntimeA) {
            if (tM_ms <= 0.0) const_cast<MAGSACSPRTScoring*>(this)->microBenchmarkResiduals(kData_, kEstimator_, 128);
        }

        const double eps = clampProb(sprt_.epsilon, kMinEpsilon, 1.0 - 1e-6);
        const double del = clampProb(sprt_.delta,   kMinDelta,   kMaxDeltaFrac);
        const double A   = std::max(1.0, sprt_.A);

        double lambdaLR = 1.0;
        int inlierCount = 0;
        double scoreVal = 0.0;

        auto accumulate_point = [&](int iPos, size_t trueIdx) -> bool {
            const double sqr =
                kEstimator_->squaredResidual(kData_.row(static_cast<int>(trueIdx)), kModel_);

            if (sqr < squaredThreshold) {
                lambdaLR *= (del / eps);
                if (kStoreInliers_) inliers_.push_back(trueIdx);
                ++inlierCount;
                scoreVal += magsacLoss(sqr);
            } else {
                lambdaLR *= ((1.0 - del) / (1.0 - eps));
                scoreVal += lossOutlier;
            }

            if (lambdaLR > A) {
                // record partial stats to refine delta
                const double obsFrac = static_cast<double>(inlierCount) / static_cast<double>(iPos + 1);
                const_cast<MAGSACSPRTScoring*>(this)->rejectedInlierFracSum_ += obsFrac;
                const_cast<MAGSACSPRTScoring*>(this)->rejectedCount_ += 1;
                return false;
            }

            const int remaining = N - iPos - 1;
            if (premultiplier * zeroResidualLoss * remaining + scoreVal < kBestScore_.getValue())
                return false;

            return true;
        };

        if (kPotentialInlierSets_ == nullptr) {
            for (int i = 0; i < N; ++i) {
                const size_t idx = perm_[permHead_];
                permHead_ = (permHead_ + 1) % perm_.size();
                if (!accumulate_point(i, idx))
                    return kEmptyScore;
            }
        } else {
            int tested = 0;
            for (const auto *setPtr : *kPotentialInlierSets_) {
                for (size_t trueIdx : *setPtr) {
                    if (!accumulate_point(tested, trueIdx))
                        return kEmptyScore;
                    ++tested;
                }
            }
            const int remaining = N - tested;
            scoreVal += remaining * lossOutlier;
        }

        return Score(inlierCount, scoreVal);
    }

    FORCE_INLINE void getWeights(
        const DataMatrix &kData_,
        const models::Model &kModel_,
        const estimator::Estimator *kEstimator_,
        std::vector<double> &weights_,
        const std::vector<size_t> *kIndices_ = nullptr
    ) const override
    {
        if (kIndices_ == nullptr) {
            const int N = kData_.rows();
            weights_.resize(N);
            for (int i = 0; i < N; ++i) {
                const double sqr = kEstimator_->squaredResidual(kData_.row(i), kModel_);
                weights_[i] = magsacWeight(sqr);
            }
        } else {
            const int N = static_cast<int>(kIndices_->size());
            weights_.resize(N);
            for (int i = 0; i < N; ++i) {
                const double sqr = kEstimator_->squaredResidual(kData_.row(static_cast<int>((*kIndices_)[i])), kModel_);
                weights_[i] = magsacWeight(sqr);
            }
        }
    }

    // ====== Optional external updater for SPRT parameters ======
    void updateSPRTParameters(const Score& currentBest, int iterationIndex, size_t totalPoints)
    {
        if (currentBest.getInlierNumber() > 0 && currentBest.getValue() > 0.0) {
            const double newEps = clampProb(
                static_cast<double>(currentBest.getInlierNumber()) / static_cast<double>(std::max<size_t>(1, totalPoints)),
                kMinEpsilon, 0.999);

            if (std::abs(newEps - sprt_.epsilon) / std::max(sprt_.epsilon, 1e-6) > kUpdateTolFrac) {
                sprt_.epsilon = newEps;
                if (rejectedCount_ > 0) {
                    const double avgBadInlier = rejectedInlierFracSum_ / static_cast<double>(rejectedCount_);
                    sprt_.delta = clampProb(avgBadInlier, kMinDelta, kMaxDeltaFrac);
                } else {
                    sprt_.delta = std::min(sprt_.delta, sprt_.epsilon / 10.0);
                }
                refreshA();
                if (iterationIndex >= 0)
                    lastUpdateIteration_ = iterationIndex;
                else
                    ++lastUpdateIteration_;
                rejectedCount_ = 0;
                rejectedInlierFracSum_ = 0.0;
            }
        } else {
            if (rejectedCount_ >= 5) {
                const double avgBadInlier = rejectedInlierFracSum_ / static_cast<double>(rejectedCount_);
                const double newDelta = clampProb(avgBadInlier, kMinDelta, kMaxDeltaFrac);
                if (std::abs(newDelta - sprt_.delta) / std::max(sprt_.delta, 1e-6) > kUpdateTolFrac) {
                    sprt_.delta = newDelta;
                    refreshA();
                }
                rejectedCount_ = 0;
                rejectedInlierFracSum_ = 0.0;
            }
        }
    }
};

} // namespace scoring
} // namespace superansac
