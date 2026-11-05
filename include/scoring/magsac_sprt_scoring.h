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
    // ====== MAGSAC core (with optimizations) ======
    static constexpr bool kUseLookUpTable = true;

    size_t degreesOfFreedom = 0;
    size_t dofIndex_ = 0;  // Cached DOF index for lookup table
    double k = 0.0;
    double Cn = 0.0;
    double squaredSigmaMax = 0.0;
    double squaredSigmaMaxPerTwo = 0.0;
    double squaredSigmaMaxPerFour = 0.0;
    double twoTimesSquaredSigmaMax = 0.0;
    double invTwoTimesSquaredSigmaMax = 0.0;  // Precomputed inverse for optimization
    double zeroResidualLoss = 0.0;
    double nPlus1Per2 = 0.0;
    double nMinus1Per2 = 0.0;
    double twoNPlus1 = 0.0;
    double lossOutlier = 0.0;
    double minPointGain_ = 0.0;
    double premultiplier = 0.0;
    double value0 = 0.0;
    double squaredTruncatedThreshold = 0.0;
    double weightPremultiplier = 0.0;

    // Cached pointer to the interleaved (lower, upper) row of the active DOF;
    // one cache line per lookup. Values are identical to the constexpr tables.
    const double* gammaTable_ = nullptr;

    FORCE_INLINE std::pair<double,double> getGammaValues(double residual_) const {
        int idx = static_cast<int>(residual_ * static_cast<double>(lookupTableSize));
        if (idx >= static_cast<int>(lookupTableSize)) idx = static_cast<int>(lookupTableSize) - 1;
        if (idx < 0) idx = 0;
        const double* kEntry = gammaTable_ + 2 * idx;
        return { kEntry[0], kEntry[1] };
    }
    FORCE_INLINE double getUpperGammaValue(double residual_) const {
        int idx = static_cast<int>(residual_ * static_cast<double>(lookupTableSize));
        if (idx >= static_cast<int>(lookupTableSize)) idx = static_cast<int>(lookupTableSize) - 1;
        if (idx < 0) idx = 0;
        return gammaTable_[2 * idx + 1];
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

    // Tunables (balanced for performance and accuracy)
    static constexpr bool   kUseRuntimeA   = false;  // set true to use K-based threshold
    static constexpr double kDefaultAlpha  = 0.02;  // Balanced false positive rate (was 0.05, tried 0.01)
    static constexpr double kDefaultBeta   = 0.02;  // Balanced false negative rate (was 0.05, tried 0.01)
    static constexpr double kMinEpsilon    = 1e-3;
    static constexpr double kMinDelta      = 1e-4;
    static constexpr double kMaxDeltaFrac  = 0.5;
    static constexpr double kUpdateTolFrac = 0.05;

    // Runtime model (used only if kUseRuntimeA)
    double tM_ms = 0.05;
    double mS    = 1.0;

    SPRTHistory sprt_{0.05, 0.005, 0.0};
    size_t lastUpdateIteration_ = 0;

    // Rejection statistics for delta update
    size_t rejectedCount_ = 0;
    double rejectedInlierFracSum_ = 0.0;

    // Reset SPRT state (called from initialize to ensure clean state)
    void resetSPRT() {
        sprt_ = {0.05, 0.005, 0.0};
        refreshA();
        rejectedCount_ = 0;
        rejectedInlierFracSum_ = 0.0;
        lastUpdateIteration_ = 0;
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
            sink += E->squaredResidual(X.row(static_cast<int>(i)).data(), models::Model());
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
            const double r = kSquaredResidual_ * invTwoTimesSquaredSigmaMax;
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
        const double r = kSquaredResidual_ * invTwoTimesSquaredSigmaMax;
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
        dofIndex_ = degreesOfFreedom - 2;  // Cache DOF index for lookup table optimization
        gammaTable_ = interleavedGammaLookupTable(dofIndex_); // Interleaved (lower, upper) row
        k  = getK(degreesOfFreedom);
        nPlus1Per2  = (degreesOfFreedom + 1) / 2.0;
        nMinus1Per2 = (degreesOfFreedom - 1) / 2.0;
        if (degreesOfFreedom >= 2 && degreesOfFreedom <= 6)
        {
            struct DofConsts { double cn[5]; double twoNPlus1[5]; };
            static const DofConsts kDofConsts = []() {
                DofConsts c{};
                for (size_t d = 2; d <= 6; ++d)
                {
                    c.cn[d - 2] = 1.0 / (std::pow(2.0, d / 2.0) * boost::math::tgamma(d / 2.0));
                    c.twoNPlus1[d - 2] = std::pow(2.0, (d + 1) / 2.0);
                }
                return c;
            }();
            Cn = kDofConsts.cn[dofIndex_];
            twoNPlus1 = kDofConsts.twoNPlus1[dofIndex_];
        }
        else
        {
            Cn = 1.0 / (std::pow(2.0, degreesOfFreedom / 2.0) * boost::math::tgamma(degreesOfFreedom / 2.0));
            twoNPlus1 = std::pow(2.0, nPlus1Per2);
        }
        squaredSigmaMax         = threshold * threshold;
        squaredSigmaMaxPerTwo   = squaredSigmaMax / 2.0;
        squaredSigmaMaxPerFour  = squaredSigmaMaxPerTwo / 2.0;
        twoTimesSquaredSigmaMax = 2.0 * squaredSigmaMax;
        invTwoTimesSquaredSigmaMax = 1.0 / twoTimesSquaredSigmaMax;  // Precomputed inverse
        premultiplier       = 1.0 / threshold * Cn * twoNPlus1;
        value0              = getSubtractTerm(degreesOfFreedom);
        squaredTruncatedThreshold = k * k * squaredSigmaMax;
        weightPremultiplier = 1.0 / threshold * Cn * nMinus1Per2;
        lossOutlier         = threshold * getOutlierLoss(degreesOfFreedom);

        const auto zeroGammaValues = getGammaValues(0.0);
        zeroResidualLoss = squaredSigmaMaxPerTwo * zeroGammaValues.first
                         + squaredSigmaMaxPerFour * (zeroGammaValues.second - value0);
        const double* lastGamma = gammaTable_ + 2 * (lookupTableSize - 1);
        const double minInlierLoss = premultiplier * (squaredSigmaMaxPerTwo * lastGamma[0]
                                   + squaredSigmaMaxPerFour * (lastGamma[1] - value0));
        minPointGain_ = std::min(lossOutlier, minInlierLoss);

        // Reset SPRT state to ensure clean state between runs
        resetSPRT();
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

        // Note: Removed permutation - sequential order is sufficient for SPRT
        if constexpr (kUseRuntimeA) {
            if (tM_ms <= 0.0) const_cast<MAGSACSPRTScoring*>(this)->microBenchmarkResiduals(kData_, kEstimator_, 128);
        }

        const int kSampleSize = static_cast<int>(kEstimator_->sampleSize());
        const double eps = clampProb(sprt_.epsilon, 0.05, 1.0 - 1e-6);
        const double del = clampProb(sprt_.delta,   kMinDelta,   std::min(kMaxDeltaFrac, 0.5 * eps));
        const double A   = std::max(1.0, sprt_.A);
        const double kInlierLR = del / eps;
        const double kOutlierLR = (1.0 - del) / (1.0 - eps);
        const double kMaxGainPerPoint = premultiplier * zeroResidualLoss;
        const double kBestScoreValue = kBestScore_.getValue();

        double lambdaLR = 1.0;
        int inlierCount = 0;
        double scoreVal = 0.0;

        auto accumulate_point = [&](int iPos, size_t trueIdx, const double sqr) -> bool {
            if (sqr < squaredThreshold) {
                lambdaLR *= kInlierLR;
                if (kStoreInliers_) inliers_.push_back(trueIdx);
                ++inlierCount;
                scoreVal += magsacLoss(sqr);
            } else {
                lambdaLR *= kOutlierLR;
                scoreVal += lossOutlier;
            }

            if (lambdaLR > A) {
                // record partial stats to refine delta (discounting up to kSampleSize minimal-sample points)
                if (iPos + 1 > kSampleSize) {
                    const int nonSampleInliers = std::max(0, inlierCount - kSampleSize);
                    const double obsFrac = static_cast<double>(nonSampleInliers) / static_cast<double>(iPos + 1 - kSampleSize);
                    const_cast<MAGSACSPRTScoring*>(this)->rejectedInlierFracSum_ += obsFrac;
                    const_cast<MAGSACSPRTScoring*>(this)->rejectedCount_ += 1;
                }
                return false;
            }

            const int remaining = N - iPos - 1;
            if (kMaxGainPerPoint * remaining + scoreVal < kBestScoreValue)
                return false;

            return true;
        };

        if (kPotentialInlierSets_ == nullptr) {
            constexpr int kMaxBlockSize = 256;
            double sqrBuffer[kMaxBlockSize];
            const bool kHasBest = (kBestScoreValue > -1e300);
            const bool kCollectOnFly = kStoreInliers_ && !kHasBest;
            size_t* __restrict__ outInliers = nullptr;
            if (kStoreInliers_) {
                if (kCollectOnFly) {
                    inliers_.resize(static_cast<size_t>(N));
                    outInliers = inliers_.data();
                } else {
                    inliers_.clear();
                }
            }

            const double sqTh = squaredThreshold;
            const double inv2SigSq = invTwoTimesSquaredSigmaMax;
            const double s2 = squaredSigmaMaxPerTwo;
            const double s4 = squaredSigmaMaxPerFour;
            const double val0 = value0;
            const double premul = premultiplier;
            const double outLoss = lossOutlier;
            const double* __restrict__ gTab = gammaTable_;
            constexpr double kTableSizeD = static_cast<double>(lookupTableSize);
            constexpr int kTableMaxIdx = static_cast<int>(lookupTableSize) - 1;

            bool sprtDead = false;
            int blockSize = 8;
            for (int base = 0; base < N; ) {
                const int kCount = std::min(blockSize, N - base);
                kEstimator_->squaredResiduals(kData_, kModel_, base, kCount, sqrBuffer);

                if (!sprtDead && lambdaLR < 1e-20) {
                    const double logOutlierLR = (kOutlierLR > 1.0) ? std::log(kOutlierLR) : 0.0;
                    if (lambdaLR == 0.0 ||
                        std::log(lambdaLR) + static_cast<double>(N - base) * logOutlierLR < std::log(A) - 1e-6) {
                        sprtDead = true;
                    }
                }

                const bool canPruneByBest = kHasBest &&
                    ((scoreVal + kCount * minPointGain_) + kMaxGainPerPoint * (N - base - kCount) <= kBestScoreValue + 1e-9);

                if (sprtDead && !canPruneByBest && !kCollectOnFly) {
                    for (int j = 0; j < kCount; ++j) {
                        const double sqr = sqrBuffer[j];
                        if (sqr < sqTh) {
                            ++inlierCount;
                            int idx = static_cast<int>((sqr * inv2SigSq) * kTableSizeD);
                            if (idx > kTableMaxIdx) idx = kTableMaxIdx;
                            if (idx < 0) idx = 0;
                            const double* g = gTab + 2 * idx;
                            scoreVal += premul * (s2 * g[0] + s4 * (g[1] - val0));
                        } else {
                            scoreVal += outLoss;
                        }
                    }
                } else {
                    for (int j = 0; j < kCount; ++j) {
                        const int iPos = base + j;
                        const double sqr = sqrBuffer[j];
                        if (sqr < sqTh) {
                            if (!sprtDead) lambdaLR *= kInlierLR;
                            if (kCollectOnFly) outInliers[inlierCount] = static_cast<size_t>(iPos);
                            ++inlierCount;
                            int idx = static_cast<int>((sqr * inv2SigSq) * kTableSizeD);
                            if (idx > kTableMaxIdx) idx = kTableMaxIdx;
                            if (idx < 0) idx = 0;
                            const double* g = gTab + 2 * idx;
                            scoreVal += premul * (s2 * g[0] + s4 * (g[1] - val0));
                        } else {
                            if (!sprtDead) lambdaLR *= kOutlierLR;
                            scoreVal += outLoss;
                        }

                        if (!sprtDead && lambdaLR > A) {
                            if (iPos + 1 > kSampleSize) {
                                const int nonSampleInliers = std::max(0, inlierCount - kSampleSize);
                                const double obsFrac = static_cast<double>(nonSampleInliers) / static_cast<double>(iPos + 1 - kSampleSize);
                                const_cast<MAGSACSPRTScoring*>(this)->rejectedInlierFracSum_ += obsFrac;
                                const_cast<MAGSACSPRTScoring*>(this)->rejectedCount_ += 1;
                            }
                            if (kCollectOnFly) inliers_.clear();
                            return kEmptyScore;
                        }

                        if (canPruneByBest) {
                            const int remaining = N - iPos - 1;
                            if (kMaxGainPerPoint * remaining + scoreVal < kBestScoreValue) {
                                if (kCollectOnFly) inliers_.clear();
                                return kEmptyScore;
                            }
                        }
                    }
                }
                base += kCount;
                blockSize = std::min(blockSize << 1, kMaxBlockSize);
            }

            if (kStoreInliers_) {
                if (kCollectOnFly) {
                    inliers_.resize(static_cast<size_t>(inlierCount));
                } else if (scoreVal > kBestScoreValue) {
                    inliers_.resize(static_cast<size_t>(inlierCount));
                    size_t* __restrict__ winOut = inliers_.data();
                    int written = 0;
                    int bSz = 8;
                    for (int base = 0; base < N; ) {
                        const int kCount = std::min(bSz, N - base);
                        kEstimator_->squaredResiduals(kData_, kModel_, base, kCount, sqrBuffer);
                        for (int j = 0; j < kCount; ++j) {
                            if (sqrBuffer[j] < sqTh)
                                winOut[written++] = static_cast<size_t>(base + j);
                        }
                        base += kCount;
                        bSz = std::min(bSz << 1, kMaxBlockSize);
                    }
                    inliers_.resize(static_cast<size_t>(written));
                }
            }
        } else {
            int tested = 0;
            for (const auto *setPtr : *kPotentialInlierSets_) {
                for (size_t trueIdx : *setPtr) {
                    if (!accumulate_point(tested, trueIdx,
                            kEstimator_->squaredResidual(kData_.row(static_cast<int>(trueIdx)).data(), kModel_)))
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
            // One batched call for all residuals, then transform in place.
            kEstimator_->squaredResiduals(kData_, kModel_, 0, N, weights_.data());
            for (int i = 0; i < N; ++i)
                weights_[i] = magsacWeight(weights_[i]);
        } else {
            const int N = static_cast<int>(kIndices_->size());
            weights_.resize(N);
            kEstimator_->squaredResidualsIndexed(kData_, kModel_, kIndices_->data(), N, weights_.data());
            for (int i = 0; i < N; ++i)
                weights_[i] = magsacWeight(weights_[i]);
        }
    }

    // ====== Optional external updater for SPRT parameters ======
    void updateSPRTParameters(const Score& currentBest, int iterationIndex, size_t totalPoints)
    {
        if (currentBest.getInlierNumber() > 0 && currentBest.getValue() > 0.0) {
            const double newEps = clampProb(
                static_cast<double>(currentBest.getInlierNumber()) / static_cast<double>(std::max<size_t>(1, totalPoints)),
                0.05, 0.999);

            if (std::abs(newEps - sprt_.epsilon) / std::max(sprt_.epsilon, 1e-6) > kUpdateTolFrac) {
                sprt_.epsilon = newEps;
                if (rejectedCount_ > 0) {
                    const double avgBadInlier = rejectedInlierFracSum_ / static_cast<double>(rejectedCount_);
                    sprt_.delta = clampProb(avgBadInlier, kMinDelta, std::min(kMaxDeltaFrac, 0.5 * sprt_.epsilon));
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
                const double newDelta = clampProb(avgBadInlier, kMinDelta, std::min(kMaxDeltaFrac, 0.5 * sprt_.epsilon));
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
