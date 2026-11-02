// agent_spike.cpp — FrankenStrata Phase 0 agent spike (DESIGN §7, Phase 0).
//
// THE SPIKE SHAPE (§7): "a minimal agent skeleton (no protocol) that opens the
// validation pack with the SYCL ports, fills the SYCL ExpertCache, and
// reproduces CPU-computed rows for recorded (activation, ids) inputs. This is
// the extraction-risk burn-down (§11)."
//
// What this file is:
//   - The extraction skeleton for the SYCL agent: pack source, cache fill,
//     grouped compute, and the §8.2 row-comparison statistics, structured
//     around the upstream SYCL ports listed in DESIGN §6.2.
//   - The recorded-input replay harness that Phase 0 uses for the divergence
//     control (§8.2 step 1) and that grows into Phase 1's validation harness.
//   - Deliberately NO protocol: no segment, no doorbell, no kind 3. The full
//     agent loop (§6.1) arrives in Phase 1 around franken/wire/franken_wire.hpp.
//
// What it is not: a working agent. Every body that needs the SYCL runtime or
// an upstream engine class is a TODO stub that fails loudly (returns false /
// throws), so the skeleton can never silently produce numbers. The host-side
// pieces — recorded-window loading, the fill-list builder, the epsilon
// statistics — are real code.
//
// Exit criterion this spike serves (§7 Phase 0): "spike rows within the SYCL
// parity tests' own tolerances against the CPU path" — i.e. its epsilon
// distribution must sit inside what sycl/src/kernels/s2_expert_grouped_parity
// .cpp already accepts for the same kernels, measured against CPU rows
// (kind -1), the engine's reference implementation of an expert (§8.2).
//
// Build note: this file compiles standalone (host C++17, no SYCL) so its
// structure and statistics can be exercised in CI / on GPU-less machines —
// the §14 conformance-floor idea applied to the spike. The real build links
// it into the SYCL tree and fills the TODO bodies with the ports below.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spike {

// ---------------------------------------------------------------------------
// Recorded inputs (§7 Phase 0, §8.2 step 1): windows captured from real
// engine sessions — per MoE layer, the fp32 activations the session
// published and the routed ids it chose. The spike replays them offline.
//
// File format (little-endian, written by the capture side — Phase 0 work in
// the primary tree, format owned here):
//   char[4]  magic "FRKW"
//   u32      format version (1)
//   u32      n_layers (L)          u32 n_expert (E)
//   u32      k                     u32 n_embd (H)
//   then per window:
//     u32 layer  u32 n_tok
//     i32 ids[n_tok * k]
//     f32 x[n_tok * H]
// ---------------------------------------------------------------------------

struct Window {
    int32_t layer = 0;
    int32_t n_tok = 0;
    std::vector<int32_t> ids;  // [n_tok * k]
    std::vector<float> x;      // [n_tok * H]
};

struct RecordedSet {
    int32_t n_layers = 0;
    int32_t n_expert = 0;
    int32_t k = 0;
    int32_t n_embd = 0;
    std::vector<Window> windows;
};

// Real code: strict loader; any truncation or magic mismatch is an error,
// never a partial read (a half-loaded window would corrupt the statistics).
inline RecordedSet load_recorded(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open recorded inputs: " + path);
    auto read = [&](void* dst, size_t bytes) {
        in.read(reinterpret_cast<char*>(dst), std::streamsize(bytes));
        if (!in) throw std::runtime_error("truncated recorded inputs: " + path);
    };
    char magic[4];
    read(magic, 4);
    if (std::memcmp(magic, "FRKW", 4) != 0)
        throw std::runtime_error("bad magic in recorded inputs: " + path);
    uint32_t version = 0;
    read(&version, 4);
    if (version != 1) throw std::runtime_error("unsupported recorded format version");
    RecordedSet rs;
    read(&rs.n_layers, 4);
    read(&rs.n_expert, 4);
    read(&rs.k, 4);
    read(&rs.n_embd, 4);
    while (in.peek() != std::char_traits<char>::eof()) {
        Window w;
        uint32_t layer = 0, n_tok = 0;
        read(&layer, 4);
        read(&n_tok, 4);
        w.layer = int32_t(layer);
        w.n_tok = int32_t(n_tok);
        w.ids.resize(size_t(w.n_tok) * size_t(rs.k));
        w.x.resize(size_t(w.n_tok) * size_t(rs.n_embd));
        read(w.ids.data(), w.ids.size() * sizeof(int32_t));
        read(w.x.data(), w.x.size() * sizeof(float));
        rs.windows.push_back(std::move(w));
    }
    return rs;
}

// ---------------------------------------------------------------------------
// Pack source — the extraction seam over the SYCL FileExpertSource ports
// (DESIGN §6.2: sycl/src/core/expert_source.cpp and
// sycl/src/core/gguf_expert_source.cpp, FileExpertSource ports over
// experts.bin and native GGUF role slices).
// ---------------------------------------------------------------------------

class PackSource {
public:
    // Opens the pack directory and verifies it against the expected geometry
    // (the spike's stand-in for the §4.3 pack-identity check).
    bool open(const std::string& pack_dir, int32_t n_layers, int32_t n_expert,
              std::string& err) {
        // TODO(sycl-port): construct the ported FileExpertSource over
        // pack_dir (experts.bin, or GGUF shards via gguf_expert_source.cpp),
        // read per-layer blob_bytes from the pack, and verify
        // (n_layers, n_expert) match. Blob addressing is upstream's:
        // experts.bin blob at layer_offset(layer) + expert * blob_bytes(layer)
        // (expert_layout.hpp:35-40), blob index layer * n_expert + expert.
        err = "PackSource::open: SYCL FileExpertSource port not linked (Phase 0 TODO)";
        (void)pack_dir; (void)n_layers; (void)n_expert;
        return false;
    }

    // Bytes of one expert blob, host-readable (the CPU path's view).
    const uint8_t* blob(int32_t layer, int32_t expert) {
        // TODO(sycl-port): delegate to FileExpertSource::blob(l, e).
        (void)layer; (void)expert;
        return nullptr;
    }

    uint64_t blob_bytes(int32_t /*layer*/) const {
        // TODO(sycl-port): per-layer blob size from the opened pack.
        return 0;
    }
};

// ---------------------------------------------------------------------------
// Agent cache — the extraction seam over the SYCL ExpertCache port
// (DESIGN §6.2: sycl/src/core/expert_cache.cpp; note its open_sized applies
// no per-slot guard, expert_cache.hpp:130 — the spike therefore validates
// slot counts itself before filling).
// ---------------------------------------------------------------------------

class AgentCache {
public:
    bool open_sized(int64_t n_slots, int32_t n_layers, int32_t n_expert,
                    uint64_t blob_bytes, std::string& err) {
        // TODO(sycl-port): allocate the ported ExpertCache on the B70
        // (device plumbing: sycl/src/core/device.dp.cpp, copy_queue.cpp,
        // pinned.dp.cpp) sized by the card's free VRAM — the agent-side
        // analog of device_free_bytes() (§4.3 step 2).
        err = "AgentCache::open_sized: SYCL ExpertCache port not linked (Phase 0 TODO)";
        (void)n_slots; (void)n_layers; (void)n_expert; (void)blob_bytes;
        return false;
    }

    // Loads one blob from the pack into a free slot (page-cache read; no
    // bytes cross any segment — §4.3 step 4).
    bool load(PackSource& /*src*/, int32_t /*layer*/, int32_t /*expert*/,
              std::string& /*err*/) {
        // TODO(sycl-port): ExpertCache fill via source copy_blob into the
        // slot, on the cache's copy queue.
        return false;
    }

    bool has(int32_t /*layer*/, int32_t /*expert*/) const {
        // TODO(sycl-port): consult the residency table.
        return false;
    }
};

// ---------------------------------------------------------------------------
// Fill list (the spike's static fill; DESIGN §4.3 step 3 computes the same
// list for OPEN, and §13 Q5 fixes the policy): ranked (layer, expert) pairs,
// hottest first, from the expert profile; absent a profile, degrade to
// layer-early. Manual pins override rank (§9.2) — Phase 1 config surface,
// not the spike.
// ---------------------------------------------------------------------------

using RankedFill = std::vector<std::pair<int32_t, int32_t>>;

inline RankedFill layer_early_fill(int32_t n_layers, int32_t n_expert,
                                   int64_t slot_budget) {
    RankedFill out;
    for (int32_t l = 0; l < n_layers && int64_t(out.size()) < slot_budget; ++l)
        for (int32_t e = 0; e < n_expert && int64_t(out.size()) < slot_budget; ++e)
            out.emplace_back(l, e);
    return out;
}

inline RankedFill load_profile_fill(const std::string& /*profile_bin*/,
                                    int64_t /*slot_budget*/) {
    // TODO(phase0): read the ranked pairs from an --expert-profile capture
    // (profile.bin written by the stock engine's R4.2e profiler,
    // src/program/generate.cpp) and truncate to the slot budget, exactly as
    // PeerExperts::open consumes the same profile (§4.3 step 3).
    return {};
}

// ---------------------------------------------------------------------------
// Row computation. One entry = one (token, routed slot) pair; rows are fp32
// of width H and land at out[entry] in router order (§4.4 step 3).
// ---------------------------------------------------------------------------

// Agent side: grouped expert compute over the agent cache's resident blobs.
// Grouped/deduplicated by expert exactly like the primary's first_of pass
// (§4.4 step 3).
inline bool compute_rows_agent(AgentCache& /*cache*/, const Window& /*w*/,
                               int32_t /*k*/, int32_t /*H*/,
                               float* /*out*/, std::string& err) {
    // TODO(sycl-kernels): dispatch the ported grouped kernels —
    //   s2 packs:     sycl/src/kernels/cuda/s2_expert_grouped.dp.cpp
    //   native packs: sycl/src/kernels/cuda/native_moe.dp.cpp
    // (both parity-tested upstream against the CUDA originals, §6.2), then
    // synchronize the queue before returning rows (the spike's analog of
    // §4.4 step 4's "synchronize before DONE, never stream-and-pray").
    err = "compute_rows_agent: SYCL grouped kernels not linked (Phase 0 TODO)";
    return false;
}

// Reference side: the CPU-row path (kind -1) — the engine's own reference
// implementation of an expert (§8.1, §8.2: "All comparisons are against the
// CPU-row path"). On the reference rig this is native_expert_grouped (the
// ggml-cpu kernels over host blobs).
inline bool compute_rows_cpu(PackSource& /*src*/, const Window& /*w*/,
                             int32_t /*k*/, int32_t /*H*/,
                             float* /*out*/, std::string& err) {
    // TODO(cpu-path): call the engine's CPU grouped-expert path over
    // PackSource blobs (src/core/verify.cpp's native_expert_grouped, or the
    // SYCL tree's CPU equivalent) for every entry of the window.
    err = "compute_rows_cpu: CPU reference path not linked (Phase 0 TODO)";
    return false;
}

// ---------------------------------------------------------------------------
// Row comparison statistics (DESIGN §8.2 step 1): for every sampled entry,
// relative L2 ||a-b|| / ||b||, max-abs error normalized by row RMS, and
// cosine similarity; reported as median / p99 / max per model. Real code.
// ---------------------------------------------------------------------------

struct RowStats {
    double rel_l2 = 0;
    double max_abs_over_rms = 0;
    double cosine = 1;
};

inline RowStats compare_rows(const float* a, const float* b, int32_t h) {
    double num = 0, den = 0, dot = 0, na = 0, nb = 0, max_abs = 0, sq = 0;
    for (int32_t i = 0; i < h; ++i) {
        const double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(b[i]) * double(b[i]);
        dot += double(a[i]) * double(b[i]);
        na += double(a[i]) * double(a[i]);
        nb += double(b[i]) * double(b[i]);
        max_abs = std::max(max_abs, std::fabs(d));
        sq += double(b[i]) * double(b[i]);
    }
    RowStats s;
    s.rel_l2 = den > 0 ? std::sqrt(num / den) : 0;
    const double rms = h > 0 ? std::sqrt(sq / h) : 0;
    s.max_abs_over_rms = rms > 0 ? max_abs / rms : 0;
    s.cosine = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 1;
    return s;
}

struct Summary {
    double median = 0, p99 = 0, max = 0;
};

inline Summary summarize(std::vector<double> v) {
    if (v.empty()) return {};
    std::sort(v.begin(), v.end());
    auto q = [&](double p) {
        const size_t i = std::min(v.size() - 1, size_t(p * double(v.size())));
        return v[i];
    };
    return {q(0.50), q(0.99), v.back()};
}

inline void print_summary(const char* name, const Summary& s) {
    std::printf("  %-18s median %.3e  p99 %.3e  max %.3e\n", name, s.median, s.p99, s.max);
}

}  // namespace spike

// ---------------------------------------------------------------------------
// main: open pack -> build fill -> fill cache -> replay recorded windows ->
// compare agent rows vs CPU rows -> print the §8.2 statistics.
// Mirrors §7's spike description step for step; stops at the first unlinked
// TODO with a nonzero exit, so a green run always means real numbers.
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string pack_dir, profile_bin, recorded_path;
    int64_t slot_budget = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--pack") pack_dir = argv[i + 1];
        else if (a == "--profile") profile_bin = argv[i + 1];
        else if (a == "--recorded") recorded_path = argv[i + 1];
        else if (a == "--slots") slot_budget = std::stoll(argv[i + 1]);
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (pack_dir.empty() || recorded_path.empty() || slot_budget <= 0) {
        std::fprintf(stderr,
                     "usage: agent_spike --pack <dir> --recorded <file> --slots <n> "
                     "[--profile <profile.bin>]\n");
        return 2;
    }

    using namespace spike;
    try {
        const RecordedSet rs = load_recorded(recorded_path);
        std::printf("recorded: %zu windows, L=%d E=%d k=%d H=%d\n", rs.windows.size(),
                    rs.n_layers, rs.n_expert, rs.k, rs.n_embd);

        std::string err;
        PackSource src;
        if (!src.open(pack_dir, rs.n_layers, rs.n_expert, err)) {
            std::fprintf(stderr, "spike stop: %s\n", err.c_str());
            return 1;
        }

        const RankedFill fill = profile_bin.empty()
                                    ? layer_early_fill(rs.n_layers, rs.n_expert, slot_budget)
                                    : load_profile_fill(profile_bin, slot_budget);
        AgentCache cache;
        if (!cache.open_sized(int64_t(fill.size()), rs.n_layers, rs.n_expert,
                              src.blob_bytes(0), err)) {
            std::fprintf(stderr, "spike stop: %s\n", err.c_str());
            return 1;
        }
        for (const auto& [l, e] : fill)
            if (!cache.load(src, l, e, err)) {
                std::fprintf(stderr, "spike stop: %s\n", err.c_str());
                return 1;
            }

        // Replay: sampling per §8.2 step 1 — every layer of the first 20
        // windows, then a 1% random sample thereafter (deterministic stride
        // here; the harness grows the full sampler in Phase 1).
        std::vector<double> rel_l2, max_abs, cosine;
        std::vector<float> rows_agent, rows_cpu;
        for (size_t wi = 0; wi < rs.windows.size(); ++wi) {
            const Window& w = rs.windows[wi];
            const size_t entries = size_t(w.n_tok) * size_t(rs.k);
            rows_agent.assign(entries * size_t(rs.n_embd), 0.f);
            rows_cpu.assign(entries * size_t(rs.n_embd), 0.f);
            if (!compute_rows_agent(cache, w, rs.k, rs.n_embd, rows_agent.data(), err) ||
                !compute_rows_cpu(src, w, rs.k, rs.n_embd, rows_cpu.data(), err)) {
                std::fprintf(stderr, "spike stop: %s\n", err.c_str());
                return 1;
            }
            const bool sample = wi < 20 || (wi % 100 == 0);
            if (!sample) continue;
            for (size_t en = 0; en < entries; ++en) {
                const RowStats s = compare_rows(&rows_agent[en * size_t(rs.n_embd)],
                                                &rows_cpu[en * size_t(rs.n_embd)], rs.n_embd);
                rel_l2.push_back(s.rel_l2);
                max_abs.push_back(s.max_abs_over_rms);
                cosine.push_back(s.cosine);
            }
        }

        std::printf("row divergence vs CPU reference (%zu entries):\n", rel_l2.size());
        print_summary("rel_l2", summarize(rel_l2));
        print_summary("max_abs/rms", summarize(max_abs));
        print_summary("cosine", summarize(cosine));
        // Gate (§7 Phase 0 exit): this distribution must sit within the SYCL
        // parity tests' own tolerances (s2_expert_grouped_parity /
        // native_grouped_parity). Wiring the numeric comparison is Phase 0
        // work once the kernels are linked; until then the run above cannot
        // have produced numbers at all (the TODO stubs stop it first).
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "spike error: %s\n", ex.what());
        return 1;
    }
}
