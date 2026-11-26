// pack_identity.hpp — Deterministic PackIdentity computation shared by primary & agents.
//
// Produces identical wire::PackIdentity values across host and container mounts
// by hashing pack format, per-layer blob sizes, per-layer quant types, per-layer
// GGUF role offsets, and sorted unique shard file sizes (not raw host paths).

#pragma once

#include "franken/wire/franken_ipc.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace franken {
namespace ipc {

inline bool compute_pack_identity(const std::string& pack_dir,
                                  const std::string& native_gguf,
                                  const strata::kernels::cpu::ExpertLayout& lay,
                                  wire::PackIdentity& out,
                                  std::string& err) {
    out = wire::PackIdentity{};
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path experts_bin = fs::path(pack_dir) / "experts.bin";
    if (fs::exists(experts_bin, ec) && !ec) {
        const uint64_t sz = fs::file_size(experts_bin, ec);
        if (ec) {
            err = "compute_pack_identity: cannot stat " + experts_bin.string();
            return false;
        }
        out.format = uint32_t(wire::PackFormat::ExpertsBin);
        out.experts_bin_size = sz;
        out.gguf_shard_list_hash = 0;
        return true;
    }

    if (!lay.native) {
        err = "compute_pack_identity: canonical pack missing experts.bin at " + experts_bin.string();
        return false;
    }
    if (native_gguf.empty()) {
        err = "compute_pack_identity: native pack without experts.bin requires --native <shard1.gguf>";
        return false;
    }

    const fs::path shard1_path(native_gguf);
    const fs::path shard_dir = shard1_path.parent_path();

    uint64_t h = fnv1a_init();
    fnv1a_u64(h, uint64_t(lay.n_layers));
    fnv1a_u64(h, uint64_t(lay.n_expert));
    fnv1a_u64(h, uint64_t(lay.version));

    std::set<std::string> shard_names;
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        fnv1a_u64(h, lay.blob_bytes(l));
        if (size_t(l) < lay.fmt.size()) {
            fnv1a_u64(h, uint64_t(uint32_t(lay.fmt[size_t(l)].gu_type)));
            fnv1a_u64(h, uint64_t(uint32_t(lay.fmt[size_t(l)].d_type)));
            fnv1a_u64(h, uint64_t(lay.fmt[size_t(l)].had_block));
        }
        for (int r = 0; r < 3; ++r) {
            const size_t idx = size_t(3 * l + r);
            if (idx < lay.gguf_off.size()) fnv1a_u64(h, lay.gguf_off[idx]);
            std::string fname;
            if (idx < lay.gguf_file.size() && !lay.gguf_file[idx].empty()) {
                fname = lay.gguf_file[idx];
            } else {
                fname = shard1_path.filename().string();
            }
            shard_names.insert(fname);
        }
    }

    for (const std::string& fname : shard_names) {
        const fs::path full = shard_dir.empty() ? fs::path(fname) : (shard_dir / fname);
        const uint64_t fsz = fs::file_size(full, ec);
        if (ec) {
            err = "compute_pack_identity: cannot stat GGUF shard " + full.string();
            return false;
        }
        fnv1a_update(h, fname.data(), fname.size());
        fnv1a_u64(h, fsz);
    }

    out.format = uint32_t(wire::PackFormat::GgufRoles);
    out.experts_bin_size = 0;
    out.gguf_shard_list_hash = h;
    return true;
}

}  // namespace ipc
}  // namespace franken
