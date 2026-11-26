// franken_ipc.cpp — POSIX shm lifecycle, cross-process Linux futexes, and wait loops.

#include "franken/wire/franken_ipc.hpp"

#include <algorithm>
#include <cerrno>
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

}  // namespace ipc
}  // namespace franken
