// franken_ipc_test.cpp — GPU-less unit tests for franken/wire/franken_ipc.{hpp,cpp}.
//
// Covers:
//   1. Layout & variable-area non-overlap math
//   2. Cross-process shm create/attach + handshake (Absent -> Loading -> HELLO -> OPEN -> OPEN_ACK -> Ready -> BYE)
//   3. SPSC ring wraparound (including > 3 * kControlRingSlots pushes/pops and uint32 wrap)
//   4. SlotState::Salvaged deadline race guard (late agent writes nothing; on-time agent wins Done)
//   5. Heartbeat staleness detection (synthetic timestamps + real SIGKILL child process)
//   6. Multi-round cross-process doorbell/futex ring wraparound (> 32 rounds over kRingSlots = 8)

#include "franken/wire/franken_ipc.hpp"

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define FRANKEN_CHECK(cond)                                                            \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            std::fflush(stderr);                                                       \
            std::abort();                                                              \
        }                                                                              \
    } while (0)
#ifdef assert
#undef assert
#endif
#define assert(cond) FRANKEN_CHECK(cond)

namespace fw = franken::wire;
namespace fi = franken::ipc;

namespace {

fw::Geometry sample_geometry() {
    fw::Geometry g{};
    g.n_layers = 48;
    g.n_expert = 256;
    g.k = 10;
    g.n_embd = 2560;
    g.max_tokens = fw::kDefaultMaxTokens;
    g.max_entries = fw::kDefaultMaxEntries;
    g.pack.format = uint32_t(fw::PackFormat::GgufRoles);
    g.pack.experts_bin_size = 0;
    g.pack.gguf_shard_list_hash = 0x123456789abcdef0ull;
    return g;
}

std::string unique_shm_name(const char* tag) {
    return "/franken_test_" + std::string(tag) + "_" + std::to_string(::getpid());
}

void test_layout_math() {
    const fw::Geometry g = sample_geometry();
    const fi::SegmentPlan p = fi::compute_segment_plan(g);
    assert(p.layout.control_offset == 0);
    assert(p.layout.desc_offset >= sizeof(fw::ControlBlock));
    assert(p.layout.desc_offset % 64 == 0);
    assert(p.layout.desc_slot_bytes % 64 == 0);
    assert(p.layout.row_slot_bytes == uint64_t(g.max_entries) * uint64_t(g.n_embd) * sizeof(float));
    assert(p.layout.row_offset >= p.layout.desc_offset + p.layout.desc_slot_bytes * fw::kRingSlots);
    assert(p.layout.ctrl_offset >= p.layout.row_offset + p.layout.row_slot_bytes * fw::kRingSlots);
    assert(p.layout.ack_offset >= p.layout.ctrl_offset + sizeof(fw::MsgRing<fw::CtrlMsg, fw::kControlRingSlots>));
    assert(p.layout.var_offset >= p.layout.ack_offset + sizeof(fw::MsgRing<fw::AckMsg, fw::kAckRingSlots>));
    assert(p.blob_bytes_offset == p.layout.var_offset);
    assert(p.primary_pairs_offset >= p.blob_bytes_offset + uint64_t(g.n_layers) * sizeof(uint64_t));
    assert(p.agent_bitmap_offset >= p.primary_pairs_offset + p.primary_pairs_bytes);
    assert(p.agent_error_offset >= p.agent_bitmap_offset + p.agent_bitmap_bytes);
    assert(p.total_bytes >= p.agent_error_offset + p.agent_error_bytes);
    std::printf("[PASS] test_layout_math (segment %.2f MiB)\n", double(p.total_bytes) / (1024.0 * 1024.0));
}

void test_spsc_ring_wraparound() {
    fw::MsgRing<fw::CtrlMsg, fw::kControlRingSlots> ring{};
    ring.head.value.store(0, std::memory_order_relaxed);
    ring.tail.value.store(0, std::memory_order_relaxed);

    for (uint32_t i = 0; i < fw::kControlRingSlots; ++i) {
        fw::CtrlMsg m{};
        m.type = uint32_t(fw::CtrlType::Promote);
        m.seq = i + 1;
        m.promote = {int32_t(i % 48), int32_t(i), i + 1};
        assert(fi::ring_push(ring, m));
    }
    fw::CtrlMsg overflow{};
    assert(!fi::ring_push(ring, overflow));
    assert(fi::ring_size(ring) == fw::kControlRingSlots);

    uint64_t next_push = fw::kControlRingSlots + 1;
    uint64_t next_pop = 1;
    for (int round = 0; round < 10; ++round) {
        for (uint32_t i = 0; i < fw::kControlRingSlots / 2; ++i) {
            fw::CtrlMsg out{};
            assert(fi::ring_pop(ring, out));
            assert(out.seq == next_pop++);
        }
        for (uint32_t i = 0; i < fw::kControlRingSlots / 2; ++i) {
            fw::CtrlMsg m{};
            m.type = uint32_t(fw::CtrlType::Demote);
            m.seq = next_push++;
            assert(fi::ring_push(ring, m));
        }
    }
    while (!fi::ring_empty(ring)) {
        fw::CtrlMsg out{};
        assert(fi::ring_pop(ring, out));
        assert(out.seq == next_pop++);
    }
    assert(next_pop == next_push);

    // Explicitly exercise uint32_t counter wraparound.
    ring.head.value.store(0xfffffffeU, std::memory_order_relaxed);
    ring.tail.value.store(0xfffffffeU, std::memory_order_relaxed);
    for (uint64_t k = 1; k <= 16; ++k) {
        fw::CtrlMsg m{};
        m.seq = k;
        assert(fi::ring_push(ring, m));
    }
    for (uint64_t k = 1; k <= 16; ++k) {
        fw::CtrlMsg out{};
        assert(fi::ring_pop(ring, out));
        assert(out.seq == k);
    }
    assert(fi::ring_empty(ring));
    std::printf("[PASS] test_spsc_ring_wraparound\n");
}

void test_salvage_race_guard() {
    const std::string name = unique_shm_name("salvage");
    const fw::Geometry g = sample_geometry();
    std::vector<uint64_t> blobs(size_t(g.n_layers), 1382400);
    fi::ShmSegment seg;
    std::string err;
    assert(seg.create(name, g, blobs, err));

    // Case A: Deadline expires while agent is stalled -> primary writes Salvaged,
    // late agent sees state != Claimed and writes NOTHING to the row buffer.
    fw::DescriptorHeader* slot = seg.descriptor(1);
    float* rows = seg.row_slot(1);
    rows[0] = 42.0f;
    slot->seq = 1;
    slot->state.store(uint32_t(fw::SlotState::Published), std::memory_order_release);
    assert(fi::try_claim_slot(slot));

    const uint64_t short_deadline = fi::now_ns() + 2000000ull;  // 2 ms
    const fi::SlotWaitResult r1 = fi::wait_slot_or_salvage(slot, short_deadline, 50);
    assert(r1 == fi::SlotWaitResult::Salvaged);
    assert(slot->state.load(std::memory_order_acquire) == uint32_t(fw::SlotState::Salvaged));

    bool wrote = false;
    const bool committed = fi::commit_slot_if_claimed(slot, [&] {
        wrote = true;
        rows[0] = -999.0f;
    });
    assert(!committed);
    assert(!wrote);
    assert(rows[0] == 42.0f);

    // Case B: Agent completes before deadline -> Done wins cleanly.
    fw::DescriptorHeader* slot2 = seg.descriptor(2);
    float* rows2 = seg.row_slot(2);
    rows2[0] = 0.0f;
    slot2->seq = 2;
    slot2->state.store(uint32_t(fw::SlotState::Published), std::memory_order_release);

    std::thread agent([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        assert(fi::try_claim_slot(slot2));
        const bool ok = fi::commit_slot_if_claimed(slot2, [&] { rows2[0] = 123.5f; });
        assert(ok);
    });
    const fi::SlotWaitResult r2 = fi::wait_slot_or_salvage(slot2, fi::now_ns() + 100000000ull, 50);
    agent.join();
    assert(r2 == fi::SlotWaitResult::Done);
    assert(rows2[0] == 123.5f);

    seg.close();
    std::printf("[PASS] test_salvage_race_guard\n");
}

void test_heartbeat_staleness_and_sigkill() {
    fi::HeartbeatMonitor hm;
    hm.reset(10, 1000000000ull);
    assert(!hm.check_stale(10, 1200000000ull));  // 200 ms < 250 ms
    assert(hm.check_stale(10, 1260000000ull));   // 260 ms >= 250 ms
    assert(!hm.check_stale(11, 1260000001ull));  // beat advanced -> fresh

    const std::string name = unique_shm_name("hbkill");
    const fw::Geometry g = sample_geometry();
    std::vector<uint64_t> blobs(size_t(g.n_layers), 1382400);
    fi::ShmSegment seg;
    std::string err;
    assert(seg.create(name, g, blobs, err));

    const pid_t pid = ::fork();
    assert(pid >= 0);
    if (pid == 0) {
        fi::ShmSegment child;
        std::string cerr;
        if (!child.attach(name, cerr)) _exit(2);
        auto* cb = child.control();
        cb->agent_state.value.store(uint32_t(fw::AgentState::Ready), std::memory_order_release);
        for (;;) {
            cb->heartbeat.value.fetch_add(1, std::memory_order_release);
            ::usleep(10000);
        }
        _exit(0);
    }

    auto* cb = seg.control();
    const uint64_t t_wait0 = fi::now_ns();
    while (cb->heartbeat.value.load(std::memory_order_acquire) < 3) {
        assert(fi::now_ns() - t_wait0 < 2000000000ull);
        ::usleep(5000);
    }
    fi::HeartbeatMonitor live_hm;
    live_hm.reset(cb->heartbeat.value.load(std::memory_order_acquire));
    ::usleep(40000);
    assert(!live_hm.check_stale(cb->heartbeat.value.load(std::memory_order_acquire)));

    assert(::kill(pid, SIGKILL) == 0);
    int status = 0;
    assert(::waitpid(pid, &status, 0) == pid);
    const uint64_t tkill = fi::now_ns();
    bool went_stale = false;
    while (fi::now_ns() - tkill < 800000000ull) {
        if (live_hm.check_stale(cb->heartbeat.value.load(std::memory_order_acquire))) {
            went_stale = true;
            break;
        }
        ::usleep(10000);
    }
    assert(went_stale);
    seg.close();
    std::printf("[PASS] test_heartbeat_staleness_and_sigkill\n");
}

void test_cross_process_handshake_and_ring_rounds() {
    const std::string name = unique_shm_name("xproc");
    const fw::Geometry g = sample_geometry();
    std::vector<uint64_t> blobs(size_t(g.n_layers), 1382400);
    fi::ShmSegment seg;
    std::string err;
    assert(seg.create(name, g, blobs, err));

    constexpr int kRounds = 36;  // > 4x kRingSlots (8)

    const pid_t pid = ::fork();
    assert(pid >= 0);
    if (pid == 0) {
        fi::ShmSegment agent;
        std::string aerr;
        if (!agent.attach(name, aerr)) _exit(10);
        auto* cb = agent.control();
        if (!fi::pack_identity_matches(cb->geometry.pack, g.pack)) _exit(11);

        cb->caps = {fw::kDefaultMaxTokens, fw::kDefaultMaxEntries, 128, uint32_t(fw::AgentVendor::Cpu)};
        cb->agent_state.value.store(uint32_t(fw::AgentState::Loading), std::memory_order_release);
        fw::AckMsg hello{};
        hello.type = uint32_t(fw::AckType::Hello);
        hello.seq = 0;
        hello.hello.caps = cb->caps;
        hello.hello.pack_size = g.pack.gguf_shard_list_hash;
        if (!fi::ring_push(*agent.ack_ring(), hello)) _exit(12);

        uint64_t last_seq = 0;
        bool running = true;
        while (running) {
            fi::wait_agent_wakeup(agent, last_seq, fw::kSpinAgentUs, 2000);

            fw::CtrlMsg cm{};
            while (fi::ring_pop(*agent.ctrl_ring(), cm)) {
                if (cm.type == uint32_t(fw::CtrlType::Open)) {
                    auto* bm = reinterpret_cast<uint64_t*>(agent.var_ptr(agent.plan().agent_bitmap_offset));
                    std::memset(bm, 0, size_t(agent.plan().agent_bitmap_bytes));
                    const auto* pairs = reinterpret_cast<const int32_t*>(agent.var_ptr(cm.open.fill_offset));
                    for (uint32_t i = 0; i < cm.open.n_fill; ++i) {
                        const uint64_t idx = uint64_t(pairs[2 * i]) * uint64_t(g.n_expert) + uint64_t(pairs[2 * i + 1]);
                        bm[idx / 64] |= (1ull << (idx % 64));
                    }
                    fw::AckMsg ack{};
                    ack.type = uint32_t(fw::AckType::OpenAck);
                    ack.seq = cm.seq;
                    ack.open_ack = {cm.open.gen, cm.open.n_fill, agent.plan().agent_bitmap_offset};
                    if (!fi::ring_push(*agent.ack_ring(), ack)) _exit(13);
                    cb->agent_state.value.store(uint32_t(fw::AgentState::Ready), std::memory_order_release);
                } else if (cm.type == uint32_t(fw::CtrlType::Bye)) {
                    cb->agent_state.value.store(uint32_t(fw::AgentState::Dead), std::memory_order_release);
                    running = false;
                }
            }

            const uint64_t cur_seq = cb->doorbell_seq.value.load(std::memory_order_acquire);
            while (last_seq < cur_seq) {
                ++last_seq;
                fw::DescriptorHeader* dh = agent.descriptor(last_seq);
                if (!fi::try_claim_slot(dh)) continue;
                const int32_t* ids = fw::slot_ids(dh);
                const int32_t* kind = fw::slot_kind(dh);
                const float* x = fw::slot_x(dh);
                float* rows = agent.row_slot(last_seq);
                const int32_t n = dh->n_tok * dh->k;
                (void)fi::commit_slot_if_claimed(dh, [&] {
                    for (int32_t e = 0; e < n; ++e) {
                        if (kind[e] != 3) continue;
                        const int32_t tok = e / dh->k;
                        const float scale = float(ids[e] + 1);
                        for (int32_t h = 0; h < g.n_embd; ++h) {
                            rows[size_t(e) * size_t(g.n_embd) + size_t(h)] =
                                x[size_t(tok) * size_t(g.n_embd) + size_t(h)] * scale;
                        }
                    }
                });
            }
        }
        _exit(0);
    }

    fw::AckMsg ack{};
    const uint64_t t0 = fi::now_ns();
    while (!fi::ring_pop(*seg.ack_ring(), ack)) {
        assert(fi::now_ns() - t0 < 2000000000ull);
        ::usleep(1000);
    }
    assert(ack.type == uint32_t(fw::AckType::Hello));
    assert(ack.hello.caps.slot_count == 128);

    auto* pairs = reinterpret_cast<int32_t*>(seg.var_ptr(seg.plan().primary_pairs_offset));
    for (int i = 0; i < 4; ++i) {
        pairs[2 * i] = 0;
        pairs[2 * i + 1] = i;
    }
    fw::CtrlMsg open_msg{};
    open_msg.type = uint32_t(fw::CtrlType::Open);
    open_msg.seq = 1;
    open_msg.open = {1, 4, seg.plan().primary_pairs_offset};
    assert(fi::ring_push(*seg.ctrl_ring(), open_msg));
    fi::futex_wake_u64_low(&seg.control()->doorbell_seq.value);

    while (!fi::ring_pop(*seg.ack_ring(), ack)) {
        assert(fi::now_ns() - t0 < 2000000000ull);
        ::usleep(1000);
    }
    assert(ack.type == uint32_t(fw::AckType::OpenAck));
    assert(ack.open_ack.n_resident == 4);
    assert(seg.control()->agent_state.value.load(std::memory_order_acquire) == uint32_t(fw::AgentState::Ready));

    for (int r = 1; r <= kRounds; ++r) {
        fw::DescriptorHeader* dh = seg.descriptor(uint64_t(r));
        dh->seq = uint64_t(r);
        dh->layer = r % g.n_layers;
        dh->n_tok = 2;
        dh->k = 4;
        dh->n_agent_entries = 2;
        dh->deadline_hint_ns = 100000000ull;

        int32_t* ids = fw::slot_ids(dh);
        int32_t* kind = fw::slot_kind(dh);
        float* x = fw::slot_x(dh);
        for (int e = 0; e < 8; ++e) {
            ids[e] = e;
            kind[e] = (e == 1 || e == 5) ? 3 : -1;
        }
        for (int h = 0; h < 2 * g.n_embd; ++h) x[h] = float(r) * 0.25f;

        dh->state.store(uint32_t(fw::SlotState::Published), std::memory_order_release);
        seg.control()->doorbell_seq.value.store(uint64_t(r), std::memory_order_release);
        fi::futex_wake_u64_low(&seg.control()->doorbell_seq.value);

        const fi::SlotWaitResult res = fi::wait_slot_or_salvage(dh, fi::now_ns() + 200000000ull);
        assert(res == fi::SlotWaitResult::Done);

        const float* rows = seg.row_slot(uint64_t(r));
        assert(std::fabs(rows[1 * g.n_embd] - float(r) * 0.25f * 2.0f) < 1e-6f);
        assert(std::fabs(rows[5 * g.n_embd] - float(r) * 0.25f * 6.0f) < 1e-6f);
    }

    fw::CtrlMsg bye{};
    bye.type = uint32_t(fw::CtrlType::Bye);
    bye.seq = 2;
    assert(fi::ring_push(*seg.ctrl_ring(), bye));
    fi::futex_wake_u64_low(&seg.control()->doorbell_seq.value);

    int status = 0;
    assert(::waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    seg.close();
    std::printf("[PASS] test_cross_process_handshake_and_ring_rounds (%d rounds)\n", kRounds);
}

}  // namespace

int main() {
    test_layout_math();
    test_spsc_ring_wraparound();
    test_salvage_race_guard();
    test_heartbeat_staleness_and_sigkill();
    test_cross_process_handshake_and_ring_rounds();
    std::printf("ALL franken_ipc_test CHECKS PASSED\n");
    return 0;
}
