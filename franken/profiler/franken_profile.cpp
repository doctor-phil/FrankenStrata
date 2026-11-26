// franken_profile.cpp — Optional hardware profiler for FrankenStrata (Phase 2).
//
// Measures:
//   1. Host CPU/RAM metrics & CPU expert microbenchmark latency for n = 1, 2, 4, 8
//      using strata_kernels_cpu (real pack blobs if --pack is provided, else synthetic Q2_0).
//   2. Each attached foreign-vendor agent (--agent <shm_name>[,slots=N], repeatable)
//      over POSIX /dev/shm using the real franken_wire.hpp protocol (HELLO + small
//      benchmark OPEN fill + timed rounds for n = 1, 2, 4, 8, 16 experts) to measure
//      IPC doorbell overhead `a_ipc_ms`, compute slope `b_per_expert_ms`, and VRAM slot budget.
//   3. Optionally includes Primary GPU metrics (--primary-vendor/--primary-name/
//      --primary-vram-gib/--primary-ms) so `recommend_tier_plan` produces the full rig plan.
//
// Writes `franken-hw-profile.json` and prints a summary table + recommended CLI flags.

#include "franken/primary/agent_tier.hpp"
#include "franken/profiler/hw_profile.hpp"
#include "franken/wire/pack_identity.hpp"

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace fw = franken::wire;
namespace fi = franken::ipc;
namespace fp = franken::primary;
namespace pr = franken::profiler;
namespace cpu = strata::kernels::cpu;

std::string utc_timestamp_now() {
    const std::time_t t = std::time(nullptr);
    struct tm tm_utc{};
    ::gmtime_r(&t, &tm_utc);
    char buf[64];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return buf;
}

void read_host_meminfo(uint64_t& total_b, uint64_t& avail_b) {
    total_b = 0;
    avail_b = 0;
    std::ifstream in("/proc/meminfo");
    std::string line;
    while (std::getline(in, line)) {
        unsigned long long kb = 0;
        if (std::sscanf(line.c_str(), "MemTotal: %llu kB", &kb) == 1) total_b = uint64_t(kb) * 1024ull;
        else if (std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kb) == 1) avail_b = uint64_t(kb) * 1024ull;
    }
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

bool pread_exact(int fd, void* dst, size_t n, uint64_t off, std::string& err) {
    auto* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd, out + done, n - done, off_t(off + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            err = std::string("pread failed: ") + std::strerror(errno);
            return false;
        }
        if (r == 0) { err = "unexpected EOF in pread"; return false; }
        done += size_t(r);
    }
    return true;
}

bool load_layer0_blobs(const std::string& pack_dir, const std::string& native_gguf,
                       int32_t count, std::vector<std::vector<uint8_t>>& out_blobs,
                       std::string& err) {
    const cpu::ExpertLayout& lay = cpu::expert_layout();
    const size_t bb = size_t(lay.blob_bytes(0));
    out_blobs.assign(size_t(count), std::vector<uint8_t>(bb));

    const std::filesystem::path exp_bin = std::filesystem::path(pack_dir) / "experts.bin";
    std::error_code ec;
    if (std::filesystem::exists(exp_bin, ec) && !ec) {
        const int fd = ::open(exp_bin.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { err = "cannot open " + exp_bin.string(); return false; }
        for (int32_t e = 0; e < count; ++e) {
            if (!pread_exact(fd, out_blobs[size_t(e)].data(), bb, lay.blob_offset(0, e), err)) {
                ::close(fd);
                return false;
            }
        }
        ::close(fd);
        return true;
    }

    if (!lay.native || native_gguf.empty()) {
        err = "load_layer0_blobs: missing experts.bin and no --native shard";
        return false;
    }
    const std::filesystem::path shard1(native_gguf);
    const std::filesystem::path dir = shard1.parent_path();
    const cpu::NativeFmt& fm = lay.fmt[0];
    const uint64_t per[3] = {fm.up_off, fm.up_off, uint64_t(bb) - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    int fds[3] = {-1, -1, -1};
    for (int r = 0; r < 3; ++r) {
        std::filesystem::path p = shard1;
        if (size_t(r) < lay.gguf_file.size() && !lay.gguf_file[size_t(r)].empty()) {
            p = dir.empty() ? std::filesystem::path(lay.gguf_file[size_t(r)]) : (dir / lay.gguf_file[size_t(r)]);
        }
        fds[r] = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
        if (fds[r] < 0) {
            for (int k = 0; k < r; ++k) if (fds[k] >= 0) ::close(fds[k]);
            err = "cannot open GGUF shard " + p.string();
            return false;
        }
    }
    bool ok = true;
    for (int32_t e = 0; e < count && ok; ++e) {
        for (int r = 0; r < 3 && ok; ++r) {
            const uint64_t off = lay.gguf_off[size_t(r)] + per[r] * uint64_t(e);
            ok = pread_exact(fds[r], out_blobs[size_t(e)].data() + at[r], size_t(per[r]), off, err);
        }
    }
    for (int r = 0; r < 3; ++r) if (fds[r] >= 0) ::close(fds[r]);
    return ok;
}

void benchmark_cpu(const std::vector<std::vector<uint8_t>>& blobs, bool native,
                   int32_t H, int iters, int threads, pr::CpuProfile& out) {
    const cpu::CpuTopology topo = cpu::detect_cpu_topology(/*skip_first=*/true);
    out.cpu_model = cpu::cpu_name();
    out.logical_cores = topo.logical > 0 ? topo.logical : int32_t(std::thread::hardware_concurrency());
    out.physical_cores = int32_t(topo.worker_cores.size()) + (topo.host_core >= 0 ? 1 : 0);
    read_host_meminfo(out.ram_total_bytes, out.ram_available_bytes);

    cpu::ExpertPool pool(threads, /*pin=*/false, /*host_works=*/true);
    const cpu::ExpertLayout& lay = cpu::expert_layout();

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> x(static_cast<size_t>(H), 0.0f);
    for (float& v : x) v = nd(rng);

    cpu::ActQ act{};
    std::vector<uint8_t> nact(cpu::kNativeActBytes, 0);
    if (native && cpu::q2_native_kernels(lay.fmt[0].gu_type)) cpu::act_quant_any(x.data(), H, act);
    else if (native) cpu::native_quant_act(lay.fmt[0], x.data(), nact.data());
    else cpu::act_quant_q8_1(x.data(), H, act);

    std::vector<float> out_buf(size_t(16) * size_t(H), 0.0f);
    const int batch_sizes[4] = {1, 2, 4, 8};
    out.expert_latency.clear();

    for (int n : batch_sizes) {
        std::vector<cpu::ExpertJobMulti> jobs(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            jobs[size_t(i)].blob = blobs[size_t(i % int(blobs.size()))].data();
            jobs[size_t(i)].nt = 1;
            jobs[size_t(i)].act[0] = &act;
            jobs[size_t(i)].nact[0] = native ? nact.data() : nullptr;
            jobs[size_t(i)].out[0] = out_buf.data() + size_t(i) * size_t(H);
            jobs[size_t(i)].fmt = nullptr;
        }
        // Warmup
        for (int w = 0; w < 3; ++w) {
            if (native) pool.run_split_multi_native(lay.fmt[0], jobs.data(), n);
            else pool.run_split_multi(jobs.data(), n);
        }
        std::vector<double> samples;
        samples.reserve(size_t(iters));
        for (int it = 0; it < iters; ++it) {
            const uint64_t t0 = fi::now_ns();
            if (native) pool.run_split_multi_native(lay.fmt[0], jobs.data(), n);
            else pool.run_split_multi(jobs.data(), n);
            const uint64_t t1 = fi::now_ns();
            samples.push_back(double(t1 - t0) / 1e6);
        }
        out.expert_latency.push_back({n, median_of(samples)});
    }
    pr::fit_linear_latency(out.expert_latency, out.ms_base, out.ms_per_expert);
}

bool benchmark_agent(const std::string& raw_spec,
                     const fw::Geometry& geom,
                     const std::vector<uint64_t>& blob_bytes,
                     uint64_t avg_blob_bytes,
                     int iters,
                     uint32_t hello_timeout_ms,
                     pr::GpuProfile& out,
                     std::string& err) {
    fp::AgentSpec spec{};
    if (!fp::parse_agent_spec(raw_spec, spec, err)) return false;

    // For profiling we only need 16 experts from layer 0 resident on the agent
    // (fast fill!), while recording the agent's full advertised slot_count!
    constexpr uint32_t kBenchExperts = 16;
    std::vector<std::pair<int32_t, int32_t>> bench_ranked;
    for (uint32_t e = 0; e < kBenchExperts && int32_t(e) < geom.n_expert; ++e) {
        bench_ranked.emplace_back(0, int32_t(e));
    }

    fp::AgentSpec probe_spec = spec;
    probe_spec.max_slots = std::min<uint32_t>(kBenchExperts, uint32_t(geom.n_expert));
    std::vector<uint8_t> claimed(size_t(geom.n_layers) * size_t(geom.n_expert), 0);

    fp::AgentTier tier;
    if (!tier.open(probe_spec, geom, blob_bytes, bench_ranked, claimed,
                   /*fixed_deadline_ms=*/200.0, hello_timeout_ms, 180000, err)) {
        return false;
    }

    out = pr::GpuProfile{};
    out.id = tier.shm_name();
    out.vendor = tier.vendor_str();
    out.device_name = std::string(tier.vendor_str()) + " agent (" + tier.shm_name() + ")";
    const uint32_t adv_slots = spec.max_slots > 0
                                   ? std::min(spec.max_slots, tier.advertised_slots())
                                   : tier.advertised_slots();
    out.recommended_slots = adv_slots;
    out.vram_usable_bytes = uint64_t(adv_slots) * avg_blob_bytes;
    out.vram_total_bytes = out.vram_usable_bytes;

    std::mt19937 rng(19);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> x(size_t(2) * size_t(geom.n_embd));
    for (float& v : x) v = nd(rng);
    std::vector<float> out_rows(size_t(fw::kDefaultMaxEntries) * size_t(geom.n_embd), 0.0f);

    const int batch_sizes[5] = {1, 2, 4, 8, 16};
    out.expert_latency.clear();
    std::vector<int32_t> salvaged;

    for (int n : batch_sizes) {
        // Fit n distinct experts into (n_tok, k) within agent caps (k = geom.k, e.g. 10 -> n=16 uses n_tok=2).
        const int64_t k_use = geom.k;
        const int64_t n_tok = (n + k_use - 1) / k_use;
        if (!tier.can_dispatch(n_tok, k_use)) continue;

        const int64_t total_ent = n_tok * k_use;
        std::vector<int32_t> ids(size_t(total_ent), 0);
        std::vector<int32_t> kind(size_t(total_ent), -1);
        for (int i = 0; i < n; ++i) {
            ids[size_t(i)] = i % int(kBenchExperts);
            kind[size_t(i)] = 3;
        }

        // Warmup
        for (int w = 0; w < 5; ++w) {
            salvaged.clear();
            if (tier.launch(0, x.data(), ids.data(), n_tok, k_use, kind.data())) {
                (void)tier.finish(out_rows.data(), salvaged);
            }
        }
        std::vector<double> samples;
        samples.reserve(size_t(iters));
        for (int it = 0; it < iters; ++it) {
            salvaged.clear();
            const uint64_t t0 = fi::now_ns();
            if (!tier.launch(0, x.data(), ids.data(), n_tok, k_use, kind.data()) ||
                !tier.finish(out_rows.data(), salvaged)) {
                err = "agent benchmark round salvaged/failed at n=" + std::to_string(n);
                tier.close();
                return false;
            }
            const uint64_t t1 = fi::now_ns();
            samples.push_back(double(t1 - t0) / 1e6);
        }
        out.expert_latency.push_back({n, median_of(samples)});
    }

    pr::fit_linear_latency(out.expert_latency, out.a_ipc_ms, out.b_per_expert_ms);
    tier.close();
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string pack_dir, native_gguf, out_path = "franken-hw-profile.json";
    std::vector<std::string> agent_specs;
    int32_t n_layers = 48, n_expert = 256, k = 10, n_embd = 2560;
    int iters = 25;
    int cpu_threads = 0;
    uint32_t hello_timeout_ms = 60000;

    // Optional primary GPU entry for generating a complete multi-GPU TierPlan.
    std::string primary_vendor, primary_name;
    double primary_vram_gib = 0.0;
    double primary_a_ms = 0.0, primary_b_ms = 0.0;
    uint32_t primary_slots = 0;

    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const std::string v = argv[i + 1];
        if (a == "--pack") pack_dir = v;
        else if (a == "--native") native_gguf = v;
        else if (a == "--agent") agent_specs.push_back(v);
        else if (a == "--out") out_path = v;
        else if (a == "--layers") n_layers = std::stoi(v);
        else if (a == "--experts") n_expert = std::stoi(v);
        else if (a == "--k") k = std::stoi(v);
        else if (a == "--embd") n_embd = std::stoi(v);
        else if (a == "--iters") iters = std::max(1, std::stoi(v));
        else if (a == "--threads") cpu_threads = std::stoi(v);
        else if (a == "--wait-ms") hello_timeout_ms = uint32_t(std::stoul(v));
        else if (a == "--primary-vendor") primary_vendor = v;
        else if (a == "--primary-name") primary_name = v;
        else if (a == "--primary-vram-gib") primary_vram_gib = std::stod(v);
        else if (a == "--primary-slots") primary_slots = uint32_t(std::stoul(v));
        else if (a == "--primary-ms") {
            const size_t c = v.find(',');
            if (c != std::string::npos) {
                primary_a_ms = std::stod(v.substr(0, c));
                primary_b_ms = std::stod(v.substr(c + 1));
            } else {
                primary_b_ms = std::stod(v);
            }
        } else {
            std::fprintf(stderr, "franken-profile: unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    pr::HwProfile prof{};
    prof.schema_version = 1;
    prof.timestamp_utc = utc_timestamp_now();
    prof.pack_dir = pack_dir;
    prof.n_layers = n_layers;
    prof.n_expert = n_expert;
    prof.k = k;
    prof.n_embd = n_embd;

    std::string err;
    bool native = false;
    std::vector<uint64_t> blob_bytes(static_cast<size_t>(n_layers), static_cast<uint64_t>(cpu::BLOB));
    fw::PackIdentity pack_id{};
    std::vector<std::vector<uint8_t>> cpu_blobs;

    if (!pack_dir.empty()) {
        if (!cpu::expert_layout_load(pack_dir, n_layers, n_expert, err)) {
            std::fprintf(stderr, "franken-profile: %s\n", err.c_str());
            return 1;
        }
        const cpu::ExpertLayout& lay = cpu::expert_layout();
        native = lay.native;
        for (int32_t l = 0; l < n_layers; ++l) blob_bytes[size_t(l)] = lay.blob_bytes(l);
        if (!fi::compute_pack_identity(pack_dir, native_gguf, lay, pack_id, err)) {
            std::fprintf(stderr, "franken-profile: %s\n", err.c_str());
            return 1;
        }
        if (!load_layer0_blobs(pack_dir, native_gguf, 8, cpu_blobs, err)) {
            std::fprintf(stderr, "franken-profile: %s\n", err.c_str());
            return 1;
        }
    } else {
        if (!agent_specs.empty()) {
            std::fprintf(stderr, "franken-profile: --agent requires --pack <dir> so agent and profiler share PackIdentity\n");
            return 2;
        }
        std::mt19937 rng(123);
        cpu_blobs.assign(8, std::vector<uint8_t>(cpu::BLOB, 0));
        for (auto& b : cpu_blobs) {
            for (size_t i = 0; i < b.size(); i += 17) b[i] = uint8_t(rng() & 0xffu);
        }
    }

    uint64_t sum_blob = 0;
    for (uint64_t b : blob_bytes) sum_blob += b;
    prof.avg_blob_bytes = sum_blob / uint64_t(std::max<int32_t>(1, n_layers));

    std::printf("franken-profile: benchmarking host CPU (%s, %d iters/batch)...\n",
                native ? "native IQ" : "canonical s2", iters);
    benchmark_cpu(cpu_blobs, native, n_embd, iters, cpu_threads, prof.cpu);

    if (!primary_vendor.empty()) {
        pr::GpuProfile pg{};
        pg.id = "primary:0";
        pg.vendor = primary_vendor;
        pg.device_name = primary_name.empty() ? primary_vendor : primary_name;
        pg.vram_total_bytes = uint64_t(primary_vram_gib * 1073741824.0);
        pg.vram_usable_bytes = primary_slots > 0 ? uint64_t(primary_slots) * prof.avg_blob_bytes : pg.vram_total_bytes;
        pg.recommended_slots = primary_slots;
        pg.a_ipc_ms = primary_a_ms;
        pg.b_per_expert_ms = primary_b_ms;
        for (int n : {1, 2, 4, 8, 16}) {
            pg.expert_latency.push_back({n, primary_a_ms + primary_b_ms * double(n)});
        }
        prof.gpus.push_back(std::move(pg));
    }

    fw::Geometry geom{};
    geom.n_layers = n_layers;
    geom.n_expert = n_expert;
    geom.k = k;
    geom.n_embd = n_embd;
    geom.max_tokens = fw::kDefaultMaxTokens;
    geom.max_entries = fw::kDefaultMaxEntries;
    geom.pack = pack_id;

    for (const std::string& spec : agent_specs) {
        std::printf("franken-profile: probing agent '%s' over /dev/shm...\n", spec.c_str());
        pr::GpuProfile gp{};
        if (!benchmark_agent(spec, geom, blob_bytes, prof.avg_blob_bytes, iters,
                             hello_timeout_ms, gp, err)) {
            std::fprintf(stderr, "franken-profile: agent probe failed: %s\n", err.c_str());
            return 1;
        }
        prof.gpus.push_back(std::move(gp));
    }

    if (!pr::save_hw_profile(out_path, prof, err)) {
        std::fprintf(stderr, "franken-profile: %s\n", err.c_str());
        return 1;
    }

    // Print summary table & recommended tier plan
    std::printf("\n=== FrankenStrata Hardware Profile (%s) ===\n", out_path.c_str());
    std::printf("CPU : %s (%d phys / %d logical cores, %.1f / %.1f GiB RAM avail)\n",
                prof.cpu.cpu_model.c_str(), prof.cpu.physical_cores, prof.cpu.logical_cores,
                double(prof.cpu.ram_available_bytes) / 1073741824.0,
                double(prof.cpu.ram_total_bytes) / 1073741824.0);
    std::printf("      fit T_cpu(n) = %.3f + %.3f * n ms  [", prof.cpu.ms_base, prof.cpu.ms_per_expert);
    for (size_t i = 0; i < prof.cpu.expert_latency.size(); ++i) {
        std::printf("%sn=%d:%.2fms", i ? " " : "", prof.cpu.expert_latency[i].n_experts,
                    prof.cpu.expert_latency[i].latency_ms);
    }
    std::printf("]\n");

    for (const auto& g : prof.gpus) {
        std::printf("GPU : %-12s (%s) usable=%.2f GiB slots=%u fit T(n)=%.3f + %.3f*n ms\n",
                    g.id.c_str(), g.vendor.c_str(),
                    double(g.vram_usable_bytes) / 1073741824.0, g.recommended_slots,
                    g.a_ipc_ms, g.b_per_expert_ms);
    }

    const pr::TierPlan plan = pr::recommend_tier_plan(prof);
    if (plan.primary_gpu_index >= 0) {
        std::printf("\nRecommended tier plan:\n");
        std::printf("  Primary : %s (%s, %u slots)\n",
                    plan.primary_id.c_str(), plan.primary_vendor.c_str(), plan.primary_slot_budget);
        for (size_t i = 0; i < plan.agents.size(); ++i) {
            const auto& a = plan.agents[i];
            std::printf("  Tier %zu  : %s (%s, %u slots, min_entries=%d, %.2f exp/ms)\n",
                        i + 1, a.id.c_str(), a.vendor.c_str(), a.slot_budget,
                        a.min_entries_threshold, a.effective_experts_per_ms);
        }
        if (!plan.recommended_cli_flags.empty()) {
            std::printf("  Flags   : %s --franken-profile %s\n",
                        plan.recommended_cli_flags.c_str(), out_path.c_str());
        }
    }
    return 0;
}
