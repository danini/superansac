#pragma once
#include <cstdint>
#include <type_traits>
#include <limits>
#include <random>
#include <algorithm>
#include <vector>
#include <unordered_set> 
#include <unordered_map>

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
        choose_without_replacement(sample, k, (uint64_t)maxv + 1);
    }

    FORCE_INLINE void generateUniqueRandomSet(_Type* sample,
                                              const _Type k,
                                              const _Type maxv,
                                              const _Type toSkip) {
        // draw k+1 then drop toSkip if present
        static thread_local std::vector<_Type> tmp;
        tmp.resize((size_t)k + 1);
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
    // Zero-allocation flat hash table matching std::unordered_set<uint64_t>
    // (after reserve(k * 3)) bucket structure and iteration order bit-for-bit,
    // avoiding per-iteration node new/delete and unordered_map lookups.
    struct FlatHashSetSlot {
        size_t bucketCount = 0;
        uint32_t gen = 1;
        std::vector<uint32_t> bucketGen;
        std::vector<int> bucketPrev;
        std::vector<int> next;
        std::vector<uint64_t> val;
        std::vector<size_t> bkt;
        int size = 0;

        void init(uint64_t k) {
            std::unordered_set<uint64_t> tmp;
            tmp.reserve((size_t)k * 3);
            bucketCount = tmp.bucket_count();
            bucketGen.assign(bucketCount, 0);
            bucketPrev.resize(bucketCount);
            next.resize((size_t)k + 2);
            val.resize((size_t)k + 2);
            bkt.resize((size_t)k + 2);
            gen = 1;
            next[0] = -1;
            size = 0;
        }

        FORCE_INLINE void clear() {
            ++gen;
            if (gen == 0) {
                std::fill(bucketGen.begin(), bucketGen.end(), 0);
                gen = 1;
            }
            next[0] = -1;
            size = 0;
        }

        FORCE_INLINE bool insert(uint64_t v) {
            const size_t b = std::hash<uint64_t>{}(v) % bucketCount;
            if (bucketGen[b] == gen) {
                const int prev = bucketPrev[b];
                for (int p = next[prev]; p != -1 && bkt[p] == b; p = next[p]) {
                    if (val[p] == v) return false;
                }
                const int u = ++size;
                val[u] = v;
                bkt[u] = b;
                next[u] = next[prev];
                next[prev] = u;
                return true;
            } else {
                const int u = ++size;
                val[u] = v;
                bkt[u] = b;
                const int oldHead = next[0];
                if (oldHead != -1) {
                    bucketPrev[bkt[oldHead]] = u;
                }
                next[u] = oldHead;
                next[0] = u;
                bucketGen[b] = gen;
                bucketPrev[b] = 0;
                return true;
            }
        }
    };

    // Choose k values from [0, N) without replacement into sample (unordered)
    FORCE_INLINE void choose_without_replacement(_Type* sample, uint64_t k, uint64_t N) {
        if (k == 0) return;
        if (k * 8ull <= N) { // Floyd for sparse case
            static constexpr size_t kDirectMax = 128;
            static thread_local FlatHashSetSlot directSlots[kDirectMax];
            FlatHashSetSlot* slot = nullptr;
            if (k < kDirectMax) {
                slot = &directSlots[k];
                if (slot->bucketCount == 0)
                    slot->init(k);
                else
                    slot->clear();
            } else {
                static thread_local std::unordered_map<uint64_t, FlatHashSetSlot> extraSlots;
                auto [it, isNew] = extraSlots.try_emplace(k);
                slot = &it->second;
                if (isNew)
                    slot->init(k);
                else
                    slot->clear();
            }
            for (uint64_t j = N - k; j < N; ++j) {
                uint64_t t = uniform_u64_closed(engine_, 0, j);
                if (!slot->insert(t)) slot->insert(j);
            }
            size_t i = 0;
            for (int p = slot->next[0]; p != -1; p = slot->next[p])
                sample[i++] = (_Type)slot->val[p];
        } else { // partial Fisher–Yates for dense case
            // Persistent identity array: only swap k positions and restore in reverse,
            // avoiding O(N) re-initialization and vector resizing every call.
            static thread_local std::vector<uint64_t> a;
            if (a.size() < N) {
                const size_t oldSize = a.size();
                a.resize(N);
                for (size_t i = oldSize; i < N; ++i) a[i] = i;
            }
            uint64_t stackSwaps[128];
            static thread_local std::vector<uint64_t> heapSwaps;
            uint64_t* swaps = stackSwaps;
            if (k > 128) {
                heapSwaps.resize(k);
                swaps = heapSwaps.data();
            }
            for (uint64_t i = 0; i < k; ++i) {
                uint64_t j = i + uniform_u64_closed(engine_, 0, N - 1 - i);
                swaps[i] = j;
                std::swap(a[i], a[j]);
                sample[i] = (_Type)a[i];
            }
            for (uint64_t idx = k; idx-- > 0; ) {
                std::swap(a[idx], a[swaps[idx]]);
            }
        }
    }

    // State
    Xoshiro256ss engine_{0x1234567890ABCDEFULL};
    XoshiroAdapter adapter_{engine_};
    _Type min_ = 0;
    _Type max_ = std::numeric_limits<_Type>::max() - 1;
};

}} // namespace
