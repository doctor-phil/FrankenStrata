// franken_ipc.cpp — POSIX shm lifecycle, cross-process Linux futexes, and wait loops.

#include "franken/wire/franken_ipc.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <ctime>
#include <thread>

#include <fcntl.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace franken {
namespace ipc {

static_assert(sizeof(std::atomic<uint32_t>) == 4, "atomic<uint32_t> must be 32-bit for futex");
static_assert(sizeof(std::atomic<uint64_t>) == 8, "atomic<uint64_t> must be 64-bit");
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "franken_ipc low-word futex assumes little-endian layout");

SegmentPlan compute_segment_plan(const wire::Geometry& g) {
    SegmentPlan p{};
    p.layout = wire::compute_layout(g);

    uint64_t cur = p.layout.var_offset;
    p.blob_bytes_offset = cur;
    cur = wire::align_up(cur + uint64_t(std::max<int32_t>(0, g.n_layers)) * sizeof(uint64_t), 64);

    p.primary_pairs_offset = cur;
    const uint64_t total_experts =
        uint64_t(std::max<int32_t>(0, g.n_layers)) * uint64_t(std::max<int32_t>(0, g.n_expert));
    p.primary_pairs_bytes = total_experts * 2ull * sizeof(int32_t);
    cur = wire::align_up(cur + p.primary_pairs_bytes, 64);

    p.agent_bitmap_offset = cur;
    p.agent_bitmap_bytes = bitmap_words(g.n_layers, g.n_expert) * sizeof(uint64_t);
    cur = wire::align_up(cur + p.agent_bitmap_bytes, 64);

    p.agent_error_offset = cur;
    p.agent_error_bytes = kErrorTextCapacity;
    cur = wire::align_up(cur + p.agent_error_bytes, 64);

    p.total_bytes = cur;
    p.layout.total_bytes = cur;
    return p;
}

uint64_t now_ns() {
    struct timespec ts{};
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

bool futex_wait_word(std::atomic<uint32_t>* word, uint32_t expected, uint64_t timeout_ns) {
    struct timespec ts{};
    struct timespec* tsp = nullptr;
    if (timeout_ns > 0) {
        ts.tv_sec = time_t(timeout_ns / 1000000000ull);
        ts.tv_nsec = long(timeout_ns % 1000000000ull);
        tsp = &ts;
    }
    // Non-private FUTEX_WAIT (NOT FUTEX_WAIT_PRIVATE) so host primary and
    // containerized SYCL agent synchronize across shared /dev/shm.
    const long rc = ::syscall(SYS_futex, reinterpret_cast<uint32_t*>(word),
                              FUTEX_WAIT, expected, tsp, nullptr, 0);
    if (rc == 0) return true;
    if (errno == EAGAIN || errno == EINTR) return true;
    return false;
}

void futex_wake_one(std::atomic<uint32_t>* word) {
    (void)::syscall(SYS_futex, reinterpret_cast<uint32_t*>(word),
                    FUTEX_WAKE, 1, nullptr, nullptr, 0);
}

void futex_wake_all(std::atomic<uint32_t>* word) {
    (void)::syscall(SYS_futex, reinterpret_cast<uint32_t*>(word),
                    FUTEX_WAKE, INT32_MAX, nullptr, nullptr, 0);
}

bool futex_wait_u64_low(std::atomic<uint64_t>* counter, uint64_t expected, uint64_t timeout_ns) {
    // Recheck the full 64-bit value before entering the kernel so a 32-bit wrap
    // cannot miss a change.
    if (counter->load(std::memory_order_acquire) != expected) return true;
    auto* low = reinterpret_cast<std::atomic<uint32_t>*>(counter);
    return futex_wait_word(low, static_cast<uint32_t>(expected & 0xffffffffu), timeout_ns);
}

void futex_wake_u64_low(std::atomic<uint64_t>* counter) {
    auto* low = reinterpret_cast<std::atomic<uint32_t>*>(counter);
    futex_wake_all(low);
}

bool normalize_shm_name(const std::string& raw, std::string& out, std::string& err) {
    if (raw.empty()) {
        err = "shared-memory name must not be empty";
        return false;
    }
    std::string s = raw[0] == '/' ? raw : ("/" + raw);
    if (s.size() < 2 || s.find('/', 1) != std::string::npos) {
        err = "invalid POSIX shm name (must not contain nested '/'): " + raw;
        return false;
    }
    out = std::move(s);
    return true;
}

ShmSegment::~ShmSegment() { close(); }

ShmSegment::ShmSegment(ShmSegment&& other) noexcept
    : name_(std::move(other.name_)),
      fd_(other.fd_),
      base_(other.base_),
      size_(other.size_),
      owner_(other.owner_),
      plan_(other.plan_) {
    other.fd_ = -1;
    other.base_ = nullptr;
    other.size_ = 0;
    other.owner_ = false;
}

ShmSegment& ShmSegment::operator=(ShmSegment&& other) noexcept {
    if (this != &other) {
        close();
        name_ = std::move(other.name_);
        fd_ = other.fd_;
        base_ = other.base_;
        size_ = other.size_;
        owner_ = other.owner_;
        plan_ = other.plan_;
        other.fd_ = -1;
        other.base_ = nullptr;
        other.size_ = 0;
        other.owner_ = false;
    }
    return *this;
}

void ShmSegment::close() {
    if (base_ != nullptr) {
        ::munmap(base_, size_);
        base_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    if (owner_ && !name_.empty()) {
        ::shm_unlink(name_.c_str());
    }
    name_.clear();
    size_ = 0;
    owner_ = false;
    plan_ = SegmentPlan{};
}

bool ShmSegment::create(const std::string& name, const wire::Geometry& geom_offer,
                        const std::vector<uint64_t>& blob_bytes, std::string& err) {
    close();
    std::string norm;
    if (!normalize_shm_name(name, norm, err)) return false;

    if (geom_offer.n_layers <= 0 || geom_offer.n_expert <= 0 ||
        geom_offer.k <= 0 || geom_offer.n_embd <= 0 ||
        geom_offer.max_tokens == 0 || geom_offer.max_entries == 0) {
        err = "ShmSegment::create: invalid geometry dimensions";
        return false;
    }
    if (blob_bytes.size() != size_t(geom_offer.n_layers)) {
        err = "ShmSegment::create: blob_bytes length != n_layers";
        return false;
    }

    wire::Geometry g = geom_offer;
    g.magic = wire::kMagic;
    g.protocol_version = wire::kProtocolVersion;
    g.header_bytes = sizeof(wire::Geometry);
    const SegmentPlan plan = compute_segment_plan(g);
    g.blob_bytes_offset = plan.blob_bytes_offset;
    g.segment_bytes = plan.total_bytes;

    // Clean any stale leftover segment with the same name before creating.
    ::shm_unlink(norm.c_str());

    const int fd = ::shm_open(norm.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    if (fd < 0) {
        err = "shm_open(" + norm + ") failed: " + std::strerror(errno);
        return false;
    }
    // Ensure host primary and containerized agent (potentially different UIDs) can both map R/W.
    (void)::fchmod(fd, 0666);

    if (::ftruncate(fd, off_t(plan.total_bytes)) != 0) {
        err = "ftruncate(" + norm + ") failed: " + std::strerror(errno);
        ::close(fd);
        ::shm_unlink(norm.c_str());
        return false;
    }

    void* addr = ::mmap(nullptr, size_t(plan.total_bytes), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        err = "mmap(" + norm + ") failed: " + std::strerror(errno);
        ::close(fd);
        ::shm_unlink(norm.c_str());
        return false;
    }

    std::memset(addr, 0, size_t(plan.total_bytes));

    name_ = std::move(norm);
    fd_ = fd;
    base_ = static_cast<uint8_t*>(addr);
    size_ = plan.total_bytes;
    owner_ = true;
    plan_ = plan;

    wire::ControlBlock* cb = control();
    cb->geometry = g;
    cb->agent_state.value.store(uint32_t(wire::AgentState::Absent), std::memory_order_relaxed);
    cb->heartbeat.value.store(0, std::memory_order_relaxed);
    cb->residency_gen.value.store(0, std::memory_order_relaxed);
    cb->doorbell_seq.value.store(0, std::memory_order_relaxed);

    // Populate per-layer blob sizes in the variable area.
    std::memcpy(blob_bytes_table(), blob_bytes.data(), blob_bytes.size() * sizeof(uint64_t));

    // Pre-initialize descriptor slot arena offsets relative to each slot's header.
    const uint32_t ids_off = uint32_t(sizeof(wire::DescriptorHeader));
    const uint32_t kind_off = ids_off + uint32_t(g.max_entries * sizeof(int32_t));
    const uint32_t x_off = kind_off + uint32_t(g.max_entries * sizeof(int32_t));
    for (uint32_t s = 0; s < wire::kRingSlots; ++s) {
        wire::DescriptorHeader* dh = descriptor(s);
        dh->state.store(uint32_t(wire::SlotState::Empty), std::memory_order_relaxed);
        dh->payload_kind = uint32_t(wire::PayloadKind::ExpertRows);
        dh->seq = 0;
        dh->ids_offset = ids_off;
        dh->kind_offset = kind_off;
        dh->x_offset = x_off;
    }
    std::atomic_thread_fence(std::memory_order_release);
    return true;
}

bool ShmSegment::attach(const std::string& name, std::string& err) {
    close();
    std::string norm;
    if (!normalize_shm_name(name, norm, err)) return false;

    const int fd = ::shm_open(norm.c_str(), O_RDWR, 0);
    if (fd < 0) {
        err = "shm_open(" + norm + ") failed: " + std::strerror(errno);
        return false;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        err = "fstat(" + norm + ") failed: " + std::strerror(errno);
        ::close(fd);
        return false;
    }
    if (uint64_t(st.st_size) < sizeof(wire::ControlBlock)) {
        err = "shm segment " + norm + " is too small for ControlBlock";
        ::close(fd);
        return false;
    }
    void* addr = ::mmap(nullptr, size_t(st.st_size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        err = "mmap(" + norm + ") failed: " + std::strerror(errno);
        ::close(fd);
        return false;
    }

    const auto* cb = static_cast<const wire::ControlBlock*>(addr);
    std::atomic_thread_fence(std::memory_order_acquire);
    const wire::Geometry& g = cb->geometry;
    if (g.magic != wire::kMagic) {
        err = "shm segment " + norm + " has wrong magic (not a FrankenStrata segment)";
        ::munmap(addr, size_t(st.st_size));
        ::close(fd);
        return false;
    }
    if (g.protocol_version != wire::kProtocolVersion) {
        err = "shm segment " + norm + " protocol version mismatch: got " +
              std::to_string(g.protocol_version) + ", expected " +
              std::to_string(wire::kProtocolVersion);
        ::munmap(addr, size_t(st.st_size));
        ::close(fd);
        return false;
    }
    if (g.header_bytes != sizeof(wire::Geometry)) {
        err = "shm segment " + norm + " Geometry size mismatch";
        ::munmap(addr, size_t(st.st_size));
        ::close(fd);
        return false;
    }
    const SegmentPlan plan = compute_segment_plan(g);
    if (g.segment_bytes < plan.total_bytes || uint64_t(st.st_size) < g.segment_bytes) {
        err = "shm segment " + norm + " truncated vs advertised segment_bytes";
        ::munmap(addr, size_t(st.st_size));
        ::close(fd);
        return false;
    }

    name_ = std::move(norm);
    fd_ = fd;
    base_ = static_cast<uint8_t*>(addr);
    size_ = uint64_t(st.st_size);
    owner_ = false;
    plan_ = plan;
    return true;
}

void ShmSegment::write_error_text(const std::string& msg, uint64_t& out_offset, uint32_t& out_len) {
    const size_t n = std::min<size_t>(msg.size(), size_t(plan_.agent_error_bytes));
    if (n > 0) {
        std::memcpy(base_ + plan_.agent_error_offset, msg.data(), n);
    }
    out_offset = plan_.agent_error_offset;
    out_len = uint32_t(n);
}

std::string ShmSegment::read_error_text(uint64_t offset, uint32_t len) const {
    if (len == 0 || !var_range_ok(offset, len)) return {};
    return std::string(reinterpret_cast<const char*>(base_ + offset), size_t(len));
}

SlotWaitResult wait_slot_or_salvage(wire::DescriptorHeader* slot,
                                    uint64_t deadline_ns,
                                    uint32_t spin_us,
                                    const std::function<bool()>& is_dead) {
    const uint64_t start_ns = now_ns();
    const uint64_t spin_until_ns =
        std::min<uint64_t>(deadline_ns, start_ns + uint64_t(spin_us) * uint64_t(1000));

    auto try_salvage = [&](uint32_t cur) -> SlotWaitResult {
        while (cur == uint32_t(wire::SlotState::Published) ||
               cur == uint32_t(wire::SlotState::Claimed)) {
            if (slot->state.compare_exchange_strong(
                    cur, uint32_t(wire::SlotState::Salvaged),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                futex_wake_all(&slot->state);
                return SlotWaitResult::Salvaged;
            }
        }
        return cur == uint32_t(wire::SlotState::Done) ? SlotWaitResult::Done
                                                      : SlotWaitResult::Salvaged;
    };

    // Bounded spin phase
    while (true) {
        const uint32_t st = slot->state.load(std::memory_order_acquire);
        if (st == uint32_t(wire::SlotState::Done)) return SlotWaitResult::Done;
        if (st == uint32_t(wire::SlotState::Salvaged)) return SlotWaitResult::Salvaged;
        if (is_dead && is_dead()) return try_salvage(st);
        const uint64_t t = now_ns();
        if (t >= deadline_ns) return try_salvage(st);
        if (t >= spin_until_ns) break;
        cpu_relax();
    }

    // Park phase on the slot's futex state word
    while (true) {
        const uint32_t st = slot->state.load(std::memory_order_acquire);
        if (st == uint32_t(wire::SlotState::Done)) return SlotWaitResult::Done;
        if (st == uint32_t(wire::SlotState::Salvaged)) return SlotWaitResult::Salvaged;
        if (is_dead && is_dead()) return try_salvage(st);
        const uint64_t t = now_ns();
        if (t >= deadline_ns) return try_salvage(st);
        // Bound each futex slice to <= 5 ms so heartbeat staleness is noticed promptly
        // even if the deadline is long.
        const uint64_t rem = deadline_ns - t;
        const uint64_t slice = std::min<uint64_t>(rem, 5000000ull);
        (void)futex_wait_word(&slot->state, st, slice);
    }
}

void wait_agent_wakeup(ShmSegment& seg,
                       uint64_t last_doorbell_seq,
                       uint32_t spin_us,
                       uint32_t park_slice_us,
                       const std::function<bool()>& should_stop) {
    wire::ControlBlock* cb = seg.control();
    auto* cring = seg.ctrl_ring();

    auto has_work = [&]() -> bool {
        if (cb->doorbell_seq.value.load(std::memory_order_acquire) != last_doorbell_seq) return true;
        if (!ring_empty(*cring)) return true;
        if (should_stop && should_stop()) return true;
        return false;
    };

    if (has_work()) return;

    const uint64_t spin_until_ns = now_ns() + uint64_t(spin_us) * 1000ull;
    while (now_ns() < spin_until_ns) {
        if (has_work()) return;
        cpu_relax();
    }

    while (!has_work()) {
        (void)futex_wait_u64_low(&cb->doorbell_seq.value, last_doorbell_seq,
                                 uint64_t(park_slice_us) * 1000ull);
    }
}

bool try_claim_slot(wire::DescriptorHeader* slot) {
    uint32_t expected = uint32_t(wire::SlotState::Published);
    return slot->state.compare_exchange_strong(
        expected, uint32_t(wire::SlotState::Claimed),
        std::memory_order_acq_rel, std::memory_order_acquire);
}

bool commit_slot_if_claimed(wire::DescriptorHeader* slot,
                            const std::function<void()>& write_rows) {
    if (slot->state.load(std::memory_order_acquire) != uint32_t(wire::SlotState::Claimed)) {
        return false;
    }
    if (write_rows) write_rows();
    uint32_t expected = uint32_t(wire::SlotState::Claimed);
    if (slot->state.compare_exchange_strong(
            expected, uint32_t(wire::SlotState::Done),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        futex_wake_all(&slot->state);
        return true;
    }
    return false;
}

uint64_t fnv1a_init() { return 14695981039346656037ull; }

void fnv1a_update(uint64_t& h, const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= uint64_t(p[i]);
        h *= 1099511628211ull;
    }
}

void fnv1a_u64(uint64_t& h, uint64_t v) { fnv1a_update(h, &v, sizeof(v)); }

bool pack_identity_matches(const wire::PackIdentity& a, const wire::PackIdentity& b) {
    return a.format == b.format &&
           a.experts_bin_size == b.experts_bin_size &&
           a.gguf_shard_list_hash == b.gguf_shard_list_hash;
}

namespace {

// Exact round-to-nearest-even f32 -> fp16 bits (mirrors include/strata/kernels/f16_bits.hpp).
inline uint16_t f16_bits_rne(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    const int exp = int(rawexp) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (rawexp == 0xFFu) return uint16_t(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 31) return uint16_t(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        man |= 0x800000u;
        const uint32_t sh = uint32_t(14 - exp);
        uint32_t h = (man >> sh) & 0x3FFu;
        const uint32_t rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) ++h;
        return uint16_t(sign | h);
    }
    uint16_t h = uint16_t(sign | (uint32_t(exp) << 10) | (man >> 13));
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return h;
}

inline float clamp_half_finite(float v) {
    constexpr float kHalfMax = 65504.0f;
    return std::fabs(v) > kHalfMax ? std::copysign(kHalfMax, v) : v;
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,f16c")))
void quantize_q8_1_rows_avx2(const float* x, int32_t n_blocks, uint8_t* dst) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 v_min = _mm256_set1_ps(-127.0f);
    const __m256 v_max = _mm256_set1_ps(127.0f);

    for (int32_t b = 0; b < n_blocks; ++b) {
        const float* src = x + size_t(b) * 32;
        const __m256 v0 = _mm256_loadu_ps(src);
        const __m256 v1 = _mm256_loadu_ps(src + 8);
        const __m256 v2 = _mm256_loadu_ps(src + 16);
        const __m256 v3 = _mm256_loadu_ps(src + 24);

        __m256 am = _mm256_max_ps(
            _mm256_max_ps(_mm256_andnot_ps(sign_mask, v0), _mm256_andnot_ps(sign_mask, v1)),
            _mm256_max_ps(_mm256_andnot_ps(sign_mask, v2), _mm256_andnot_ps(sign_mask, v3)));
        __m128 am_lo = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        am_lo = _mm_max_ps(am_lo, _mm_movehl_ps(am_lo, am_lo));
        am_lo = _mm_max_ss(am_lo, _mm_shuffle_ps(am_lo, am_lo, 1));
        const float amax = _mm_cvtss_f32(am_lo);

        __m256 sm = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3));
        __m128 sm_lo = _mm_add_ps(_mm256_castps256_ps128(sm), _mm256_extractf128_ps(sm, 1));
        sm_lo = _mm_add_ps(sm_lo, _mm_movehl_ps(sm_lo, sm_lo));
        sm_lo = _mm_add_ss(sm_lo, _mm_shuffle_ps(sm_lo, sm_lo, 1));
        const float sum = _mm_cvtss_f32(sm_lo);

        const float d = clamp_half_finite(amax / 127.0f);
        const float sf = clamp_half_finite(sum);
        const __m128 ds_f32 = _mm_set_ps(0.0f, 0.0f, sf, d);
        const __m128i ds_f16 = _mm_cvtps_ph(ds_f32, _MM_FROUND_TO_NEAREST_INT);
        const uint32_t ds_bits = uint32_t(_mm_cvtsi128_si32(ds_f16));

        uint8_t* out = dst + size_t(b) * wire::kQ8_1BlockBytes;
        std::memcpy(out, &ds_bits, 4);

        if (amax == 0.0f) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 4), _mm256_setzero_si256());
            continue;
        }
        const __m256 inv_d = _mm256_set1_ps(1.0f / d);
        const __m256i q0 = _mm256_cvtps_epi32(_mm256_min_ps(v_max, _mm256_max_ps(v_min, _mm256_mul_ps(v0, inv_d))));
        const __m256i q1 = _mm256_cvtps_epi32(_mm256_min_ps(v_max, _mm256_max_ps(v_min, _mm256_mul_ps(v1, inv_d))));
        const __m256i q2 = _mm256_cvtps_epi32(_mm256_min_ps(v_max, _mm256_max_ps(v_min, _mm256_mul_ps(v2, inv_d))));
        const __m256i q3 = _mm256_cvtps_epi32(_mm256_min_ps(v_max, _mm256_max_ps(v_min, _mm256_mul_ps(v3, inv_d))));

        // Pack 4 x 8 int32 -> 2 x 16 int16 -> 32 int8 preserving exact lane order.
        const __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xD8);
        const __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q2, q3), 0xD8);
        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packs_epi16(p01, p23), 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 4), packed);
    }
}
#endif

}  // namespace

void quantize_q8_1_rows_host(const float* x, int32_t n_rows, int32_t n_cols, uint8_t* dst) {
    if (n_rows <= 0 || n_cols <= 0 || (n_cols & 31) != 0) return;
    const int32_t n_blocks = (n_rows * n_cols) / 32;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool has_avx2_f16c = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
    }();
    if (has_avx2_f16c) {
        quantize_q8_1_rows_avx2(x, n_blocks, dst);
        return;
    }
#endif
    for (int32_t b = 0; b < n_blocks; ++b) {
        const float* src = x + size_t(b) * 32;
        float amax = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float xi = src[i];
            const float ax = std::fabs(xi);
            if (ax > amax) amax = ax;
            sum += xi;
        }
        const float d = clamp_half_finite(amax / 127.0f);
        const uint16_t d16 = f16_bits_rne(d);
        const uint16_t s16 = f16_bits_rne(clamp_half_finite(sum));
        uint8_t* out = dst + size_t(b) * wire::kQ8_1BlockBytes;
        std::memcpy(out, &d16, 2);
        std::memcpy(out + 2, &s16, 2);
        auto* qs = reinterpret_cast<int8_t*>(out + 4);
        if (amax == 0.0f) {
            std::memset(qs, 0, 32);
        } else {
            for (int i = 0; i < 32; ++i) {
                const float q = std::roundf(src[i] / d);
                qs[i] = int8_t(q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q));
            }
        }
    }
}

}  // namespace ipc
}  // namespace franken
