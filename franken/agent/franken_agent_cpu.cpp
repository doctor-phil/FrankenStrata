// franken_agent_cpu.cpp — Vendor-free reference CPU agent (DESIGN §14 conformance floor).
//
// Compiles and runs with ZERO GPU dependencies (only strata_kernels_cpu + POSIX shm/futex).
// Holds its assigned expert blobs in RAM and computes kind == 3 entries using the exact
// same CPU kernels (ExpertPool::run_split_multi / run_split_multi_native) as the primary.

#include "franken/agent/agent_loop.hpp"
#include "franken/wire/pack_identity.hpp"

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace fw = franken::wire;
namespace fi = franken::ipc;
namespace fa = franken::agent;
namespace cpu = strata::kernels::cpu;

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
        if (r == 0) {
            err = "unexpected EOF in pread";
            return false;
        }
        done += size_t(r);
    }
    return true;
}

class CpuAgentBackend final : public fa::AgentBackend {
public:
    CpuAgentBackend(std::string pack_dir, std::string native_gguf, uint32_t max_slots, int threads)
        : pack_dir_(std::move(pack_dir)),
          native_gguf_(std::move(native_gguf)),
          max_slots_(max_slots),
          threads_(threads) {}

    ~CpuAgentBackend() override { shutdown(); }

    fw::AgentVendor vendor() const override { return fw::AgentVendor::Cpu; }

    bool init(const fw::Geometry& g,
              const uint64_t* blob_bytes,
              fw::Caps& out_caps,
              fw::PackIdentity& out_pack,
              std::string& err) override {
        g_ = g;
        if (!cpu::expert_layout_load(pack_dir_, g.n_layers, g.n_expert, err)) {
            return false;
        }
        const cpu::ExpertLayout& lay = cpu::expert_layout();
        for (int32_t l = 0; l < g.n_layers; ++l) {
            if (lay.blob_bytes(l) != blob_bytes[l]) {
                err = "CpuAgentBackend: blob_bytes mismatch at layer " + std::to_string(l);
                return false;
            }
        }
        if (!fi::compute_pack_identity(pack_dir_, native_gguf_, lay, out_pack, err)) {
            return false;
        }

        // Open backing file(s): experts.bin if present, else GGUF shards.
        const std::filesystem::path exp_path = std::filesystem::path(pack_dir_) / "experts.bin";
        std::error_code ec;
        if (std::filesystem::exists(exp_path, ec) && !ec) {
            experts_bin_fd_ = ::open(exp_path.c_str(), O_RDONLY | O_CLOEXEC);
            if (experts_bin_fd_ < 0) {
                err = "cannot open " + exp_path.string() + ": " + std::strerror(errno);
                return false;
            }
        } else {
            if (!lay.native || native_gguf_.empty()) {
                err = "CpuAgentBackend: missing experts.bin and no --native GGUF shard given";
                return false;
            }
            const std::filesystem::path shard1(native_gguf_);
            const std::filesystem::path shard_dir = shard1.parent_path();
            role_fd_.assign(size_t(3 * g.n_layers), -1);
            for (int32_t l = 0; l < g.n_layers; ++l) {
                for (int r = 0; r < 3; ++r) {
                    const size_t idx = size_t(3 * l + r);
                    std::filesystem::path p = shard1;
                    if (idx < lay.gguf_file.size() && !lay.gguf_file[idx].empty()) {
                        p = shard_dir.empty() ? std::filesystem::path(lay.gguf_file[idx])
                                              : (shard_dir / lay.gguf_file[idx]);
                    }
                    const std::string ps = p.string();
                    auto it = shard_fds_.find(ps);
                    if (it == shard_fds_.end()) {
                        const int fd = ::open(ps.c_str(), O_RDONLY | O_CLOEXEC);
                        if (fd < 0) {
                            err = "cannot open GGUF shard " + ps + ": " + std::strerror(errno);
                            return false;
                        }
                        shard_fds_[ps] = fd;
                        role_fd_[idx] = fd;
                    } else {
                        role_fd_[idx] = it->second;
                    }
                }
            }
        }

        pool_ = std::make_unique<cpu::ExpertPool>(threads_, /*pin=*/false, /*host_works=*/true);

        const uint32_t total_experts = uint32_t(g.n_layers * g.n_expert);
        const uint32_t offer_slots = max_slots_ > 0 ? std::min(max_slots_, total_experts) : total_experts;
        out_caps.max_tokens_per_window = fw::kDefaultMaxTokens;
        out_caps.max_entries_per_window = fw::kDefaultMaxEntries;
        out_caps.slot_count = offer_slots;
        out_caps.vendor = uint32_t(fw::AgentVendor::Cpu);
        return true;
    }

    bool open_fill(const std::vector<std::pair<int32_t, int32_t>>& pairs,
                   std::vector<std::pair<int32_t, int32_t>>& out_loaded,
                   std::string& err) override {
        const size_t total = size_t(g_.n_layers) * size_t(g_.n_expert);
        slot_of_.assign(total, -1);
        blobs_.clear();
        free_slots_.clear();
        out_loaded.clear();

        const size_t limit = max_slots_ > 0 ? std::min<size_t>(pairs.size(), size_t(max_slots_))
                                            : pairs.size();
        blobs_.reserve(limit);
        out_loaded.reserve(limit);

        for (size_t i = 0; i < limit; ++i) {
            const auto [l, e] = pairs[i];
            if (l < 0 || l >= g_.n_layers || e < 0 || e >= g_.n_expert) continue;
            const size_t idx = size_t(l) * size_t(g_.n_expert) + size_t(e);
            if (slot_of_[idx] >= 0) continue;

            std::vector<uint8_t> buf;
            if (!read_blob(l, e, buf, err)) return false;
            const int32_t slot = int32_t(blobs_.size());
            blobs_.push_back(std::move(buf));
            slot_of_[idx] = slot;
            out_loaded.emplace_back(l, e);
        }
        return true;
    }

    bool promote(int32_t layer, int32_t expert, std::string& err) override {
        if (layer < 0 || layer >= g_.n_layers || expert < 0 || expert >= g_.n_expert) {
            err = "CpuAgentBackend::promote: (layer, expert) out of range";
            return false;
        }
        const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
        if (slot_of_[idx] >= 0) return true;  // already resident

        std::vector<uint8_t> buf;
        if (!read_blob(layer, expert, buf, err)) return false;

        int32_t slot = -1;
        if (!free_slots_.empty()) {
            slot = free_slots_.back();
            free_slots_.pop_back();
            blobs_[size_t(slot)] = std::move(buf);
        } else if (max_slots_ == 0 || blobs_.size() < size_t(max_slots_)) {
            slot = int32_t(blobs_.size());
            blobs_.push_back(std::move(buf));
        } else {
            err = "CpuAgentBackend::promote: no free slots available";
            return false;
        }
        slot_of_[idx] = slot;
        return true;
    }

    bool demote(int32_t layer, int32_t expert, std::string& err) override {
        if (layer < 0 || layer >= g_.n_layers || expert < 0 || expert >= g_.n_expert) {
            err = "CpuAgentBackend::demote: (layer, expert) out of range";
            return false;
        }
        const size_t idx = size_t(layer) * size_t(g_.n_expert) + size_t(expert);
        const int32_t slot = slot_of_[idx];
        if (slot < 0) return true;  // idempotent: already not resident
        slot_of_[idx] = -1;
        free_slots_.push_back(slot);
        return true;
    }

    bool compute_slot(int32_t layer,
                      int32_t n_tok,
                      int32_t k,
                      const int32_t* ids,
                      const int32_t* kind,
                      const float* x,
                      float* staging_out,
                      std::string& err) override {
        const cpu::ExpertLayout& lay = cpu::expert_layout();
        const bool native = lay.native;
        const int64_t n = int64_t(n_tok) * int64_t(k);

        if (int32_t(act_multi_.size()) < n_tok) act_multi_.resize(size_t(n_tok));
        if (native && nact_multi_.size() < size_t(n_tok) * cpu::kNativeActBytes) {
            nact_multi_.resize(size_t(n_tok) * cpu::kNativeActBytes);
        }

        for (int32_t t = 0; t < n_tok; ++t) {
            if (native && cpu::q2_native_kernels(lay.fmt[size_t(layer)].gu_type)) {
                cpu::act_quant_any(x + size_t(t) * size_t(g_.n_embd), g_.n_embd, act_multi_[size_t(t)]);
            } else if (native) {
                cpu::native_quant_act(lay.fmt[size_t(layer)], x + size_t(t) * size_t(g_.n_embd),
                                      nact_multi_.data() + size_t(t) * cpu::kNativeActBytes);
            } else {
                cpu::act_quant_q8_1(x + size_t(t) * size_t(g_.n_embd), g_.n_embd, act_multi_[size_t(t)]);
            }
        }

        jobs_.clear();
        distinct_.clear();
        for (int64_t i = 0; i < n; ++i) {
            if (kind[i] != 3) continue;
            const int32_t e = ids[i];
            if (e < 0 || e >= g_.n_expert) {
                err = "CpuAgentBackend: expert id out of range";
                return false;
            }
            bool seen = false;
            for (int32_t prev : distinct_) {
                if (prev == e) { seen = true; break; }
            }
            if (seen) continue;
            distinct_.push_back(e);

            const int32_t slot = slot_of_[size_t(layer) * size_t(g_.n_expert) + size_t(e)];
            if (slot < 0) {
                err = "CpuAgentBackend: requested (layer=" + std::to_string(layer) +
                      ", expert=" + std::to_string(e) + ") is not resident";
                return false;
            }
            const uint8_t* blob_ptr = blobs_[size_t(slot)].data();
            cpu::ExpertJobMulti cur{};
            cur.blob = blob_ptr;
            cur.nt = 0;
            for (int64_t j = i; j < n; ++j) {
                if (kind[j] != 3 || ids[j] != e) continue;
                if (cur.nt == cpu::MAXT) {
                    jobs_.push_back(cur);
                    cur = cpu::ExpertJobMulti{};
                    cur.blob = blob_ptr;
                    cur.nt = 0;
                }
                const int32_t tok = int32_t(j / k);
                cur.act[cur.nt] = &act_multi_[size_t(tok)];
                cur.nact[cur.nt] = native ? nact_multi_.data() + size_t(tok) * cpu::kNativeActBytes : nullptr;
                cur.out[cur.nt] = staging_out + size_t(j) * size_t(g_.n_embd);
                ++cur.nt;
            }
            if (cur.nt > 0) jobs_.push_back(cur);
        }

        for (size_t off = 0; off < jobs_.size(); off += size_t(cpu::ExpertPool::kMaxSplitMulti)) {
            const int chunk = int(std::min<size_t>(size_t(cpu::ExpertPool::kMaxSplitMulti), jobs_.size() - off));
            if (native) {
                pool_->run_split_multi_native(lay.fmt[size_t(layer)], jobs_.data() + off, chunk);
            } else {
                pool_->run_split_multi(jobs_.data() + off, chunk);
            }
        }
        return true;
    }

    void shutdown() override {
        pool_.reset();
        blobs_.clear();
        slot_of_.clear();
        free_slots_.clear();
        if (experts_bin_fd_ >= 0) {
            ::close(experts_bin_fd_);
            experts_bin_fd_ = -1;
        }
        for (auto& kv : shard_fds_) {
            if (kv.second >= 0) ::close(kv.second);
        }
        shard_fds_.clear();
        role_fd_.clear();
    }

private:
    bool read_blob(int32_t l, int32_t e, std::vector<uint8_t>& buf, std::string& err) const {
        const cpu::ExpertLayout& lay = cpu::expert_layout();
        const size_t bb = size_t(lay.blob_bytes(l));
        buf.resize(bb);
        if (experts_bin_fd_ >= 0) {
            return pread_exact(experts_bin_fd_, buf.data(), bb, lay.blob_offset(l, e), err);
        }
        const cpu::NativeFmt& fm = lay.fmt[size_t(l)];
        const uint64_t per[3] = {fm.up_off, fm.up_off, uint64_t(bb) - fm.down_off};
        const uint64_t at[3] = {0, fm.up_off, fm.down_off};
        for (int r = 0; r < 3; ++r) {
            const size_t ri = size_t(3 * l + r);
            const uint64_t off = lay.gguf_off[ri] + per[r] * uint64_t(e);
            if (!pread_exact(role_fd_[ri], buf.data() + at[r], size_t(per[r]), off, err)) {
                return false;
            }
        }
        return true;
    }

    std::string pack_dir_;
    std::string native_gguf_;
    uint32_t max_slots_ = 0;
    int threads_ = 0;
    fw::Geometry g_{};
    int experts_bin_fd_ = -1;
    std::unordered_map<std::string, int> shard_fds_;
    std::vector<int> role_fd_;
    std::unique_ptr<cpu::ExpertPool> pool_;
    std::vector<int32_t> slot_of_;
    std::vector<int32_t> free_slots_;
    std::vector<std::vector<uint8_t>> blobs_;
    std::vector<cpu::ActQ> act_multi_;
    std::vector<uint8_t> nact_multi_;
    std::vector<cpu::ExpertJobMulti> jobs_;
    std::vector<int32_t> distinct_;
};

}  // namespace

int main(int argc, char** argv) {
    std::string shm_name, pack_dir, native_gguf;
    uint32_t max_slots = 0;
    int threads = 0;
    uint32_t wait_ms = 30000;

    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--shm") shm_name = argv[i + 1];
        else if (a == "--pack") pack_dir = argv[i + 1];
        else if (a == "--native") native_gguf = argv[i + 1];
        else if (a == "--slots") max_slots = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--threads") threads = std::stoi(argv[i + 1]);
        else if (a == "--wait-ms") wait_ms = uint32_t(std::stoul(argv[i + 1]));
        else {
            std::fprintf(stderr, "franken-agent-cpu: unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (shm_name.empty() || pack_dir.empty()) {
        std::fprintf(stderr,
                     "usage: franken-agent-cpu --shm <name> --pack <dir> "
                     "[--native <shard1.gguf>] [--slots <N>] [--threads <T>] [--wait-ms <ms>]\n");
        return 2;
    }

    fa::install_agent_signal_handlers();
    CpuAgentBackend backend(pack_dir, native_gguf, max_slots, threads);
    fa::AgentLoopConfig cfg{};
    cfg.shm_name = shm_name;
    cfg.attach_timeout_ms = wait_ms;
    fa::AgentLoopStats stats{};
    std::string err;
    if (!fa::run_agent_loop(cfg, backend, stats, err)) {
        std::fprintf(stderr, "franken-agent-cpu: %s\n", err.c_str());
        return 1;
    }
    return 0;
}
