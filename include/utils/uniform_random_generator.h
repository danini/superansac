#pragma once
#include <cstdint>
#include <type_traits>
#include <limits>
#include <random>
#include <algorithm>
#include <vector>
#include <unordered_set> 

#include "macros.h"

namespace superansac { namespace utils {

// -------- splitmix64 (seed expander) --------
struct SplitMix64 {
    uint64_t x;
    explicit SplitMix64(uint64_t seed) : x(seed) {}
    inline uint64_t next() {
        uint64_t z = (x += 0x9E3779B97f4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

// -------- xoshiro256** (public-domain) --------
struct Xoshiro256ss {
    uint64_t s[4];
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    explicit Xoshiro256ss(uint64_t seed = 1) {
        SplitMix64 sm(seed);
        for (int i = 0; i < 4; ++i) s[i] = sm.next();
    }
    inline uint64_t operator()() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return result;
    }
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return ~uint64_t{0}; }
};

// -------- Lemire mapping to [0, n) (closed -> adjust) --------
inline uint64_t uniform_u64_closed(Xoshiro256ss& rng, uint64_t lo, uint64_t hi) {
    const uint64_t n = hi - lo + 1;
    __uint128_t m = (__uint128_t)rng() * (__uint128_t)n;
    uint64_t l = (uint64_t)m;
    if (l < n) {
        const uint64_t t = -n % n;
        while (l < t) { m = (__uint128_t)rng() * (__uint128_t)n; l = (uint64_t)m; }
    }
    return (uint64_t)(m >> 64) + lo;
}

template <typename T>
inline T uniform_closed(Xoshiro256ss& rng, T lo, T hi) {
    static_assert(std::is_integral<T>::value, "integral type required");
    return static_cast<T>(uniform_u64_closed(rng, (uint64_t)lo, (uint64_t)hi));
}

// -------- Adapter to satisfy UniformRandomBitGenerator for <random> dists --------
struct XoshiroAdapter {
    using result_type = uint64_t;
    Xoshiro256ss* p{};
    explicit XoshiroAdapter(Xoshiro256ss& r) : p(&r) {}
    inline result_type operator()() { return (*p)(); }
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }
};

// -------- Drop-in class with original API --------
template <typename _Type>
class UniformRandomGenerator {
public:
    using value_type = _Type;

    UniformRandomGenerator() = default;

    // Legacy API: returns a generator usable by std::discrete_distribution
    inline XoshiroAdapter& getGenerator() { return adapter_; }

    // Legacy API: set range (stored only; no heavy distribution object)
    FORCE_INLINE void resetGenerator(const _Type& minv, const _Type& maxv) {
        min_ = minv; max_ = maxv;
    }

    // Legacy API: draw one
    FORCE_INLINE _Type getRandomNumber() {
        return uniform_closed<_Type>(engine_, min_, max_);
    }

    // Unique draws without replacement (sparse -> Floyd, dense -> partial Fisher–Yates)
    FORCE_INLINE void generateUniqueRandomSet(_Type* sample, const _Type& k) {
        generateUniqueRandomSet(sample, k, max_);
    }

    FORCE_INLINE void generateUniqueRandomSet(_Type* sample,
                                              const _Type& k,
                                              const _Type& maxv) {
        resetGenerator(0, maxv);
        choose_without_replacement(sample, k, (uint64_t)maxv + 1);
    }

    FORCE_INLINE void generateUniqueRandomSet(_Type* sample,
                                              const _Type k,
                                              const _Type maxv,
                                              const _Type toSkip) {
        resetGenerator(0, maxv);
        // draw k+1 then drop toSkip if present
        std::vector<_Type> tmp((size_t)k + 1);
        choose_without_replacement(tmp.data(), (uint64_t)k + 1, (uint64_t)maxv + 1);
        size_t w = 0;
        for (auto v : tmp) if (v != toSkip && w < (size_t)k) sample[w++] = v;
        while (w < (size_t)k) {
            _Type x;
            do { x = uniform_closed<_Type>(engine_, 0, maxv); }
            while (x == toSkip || std::find(sample, sample + w, x) != sample + w);
            sample[w++] = x;
        }
    }

private:
    // Choose k values from [0, N) without replacement into sample (unordered)
    FORCE_INLINE void choose_without_replacement(_Type* sample, uint64_t k, uint64_t N) {
        if (k * 8ull <= N) { // Floyd for sparse case
            std::unordered_set<uint64_t> S;
            // Reserve more space to avoid rehashing: k * 3 ensures load factor stays below 0.66
            S.reserve((size_t)k * 3);
            for (uint64_t j = N - k; j < N; ++j) {
                uint64_t t = uniform_u64_closed(engine_, 0, j);
                if (!S.insert(t).second) S.insert(j);
            }
            size_t i = 0; for (auto v : S) sample[i++] = (_Type)v;
        } else { // partial Fisher–Yates for dense case
            std::vector<uint64_t> a(N);
            for (uint64_t i = 0; i < N; ++i) a[i] = i;
            for (uint64_t i = 0; i < k; ++i) {
                uint64_t j = i + uniform_u64_closed(engine_, 0, N - 1 - i);
                std::swap(a[i], a[j]);
            }
            for (uint64_t i = 0; i < k; ++i) sample[i] = (_Type)a[i];
        }
    }

    // State
    Xoshiro256ss engine_{0x1234567890ABCDEFULL};
    XoshiroAdapter adapter_{engine_};
    _Type min_ = 0;
    _Type max_ = std::numeric_limits<_Type>::max() - 1;
};

}} // namespace
