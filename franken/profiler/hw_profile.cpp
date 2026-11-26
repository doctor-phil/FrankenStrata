// hw_profile.cpp — FrankenStrata Phase 2 hardware profile JSON I/O and tier planner.

#include "franken/profiler/hw_profile.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>

namespace franken {
namespace profiler {

void fit_linear_latency(const std::vector<BatchLatencyPoint>& pts,
                        double& out_a_ms, double& out_b_ms) {
    out_a_ms = 0.0;
    out_b_ms = 0.0;
    if (pts.empty()) return;
    if (pts.size() == 1) {
        const double n = std::max(1.0, double(pts[0].n_experts));
        out_a_ms = 0.0;
        out_b_ms = std::max(0.0, pts[0].latency_ms / n);
        return;
    }

    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    const double m = double(pts.size());
    for (const auto& p : pts) {
        const double x = double(p.n_experts);
        const double y = p.latency_ms;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double det = m * sxx - sx * sx;
    if (std::fabs(det) < 1e-12) {
        out_a_ms = std::max(0.0, sy / m);
        out_b_ms = 0.0;
        return;
    }
    double b = (m * sxy - sx * sy) / det;
    double a = (sy - b * sx) / m;
    if (b < 0.0) {
        b = 0.0;
        a = std::max(0.0, sy / m);
    }
    if (a < 0.0) {
        // Re-fit through origin: y = b * x
        a = 0.0;
        b = sxx > 1e-12 ? std::max(0.0, sxy / sxx) : 0.0;
    }
    out_a_ms = a;
    out_b_ms = b;
}

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(c));
                    out += buf;
                } else {
                    out.push_back(char(c));
                }
        }
    }
    out.push_back('"');
    return out;
}

std::string fmt_double(double v) {
    if (!std::isfinite(v)) return "0.0";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6f", v);
    return buf;
}

void write_batch_array(std::ostringstream& os, const std::vector<BatchLatencyPoint>& pts,
                       const char* indent) {
    os << "[";
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i > 0) os << ", ";
        os << "{\"n_experts\": " << pts[i].n_experts
           << ", \"latency_ms\": " << fmt_double(pts[i].latency_ms) << "}";
    }
    (void)indent;
    os << "]";
}

// ---------------------------------------------------------------------------
// Minimal self-contained JSON DOM & recursive-descent parser
// ---------------------------------------------------------------------------

struct JsonVal {
    enum class Kind { Null, Bool, Number, String, Array, Object } kind = Kind::Null;
    bool b = false;
    double num = 0.0;
    uint64_t u64 = 0;
    bool has_u64 = false;
    std::string str;
    std::vector<JsonVal> arr;
    std::map<std::string, JsonVal> obj;

    const JsonVal* get(const std::string& key) const {
        if (kind != Kind::Object) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
    std::string as_string(const std::string& def = {}) const {
        return kind == Kind::String ? str : def;
    }
    double as_double(double def = 0.0) const {
        return kind == Kind::Number ? num : def;
    }
    int32_t as_i32(int32_t def = 0) const {
        return kind == Kind::Number ? int32_t(std::llround(num)) : def;
    }
    uint32_t as_u32(uint32_t def = 0) const {
        if (kind != Kind::Number) return def;
        return has_u64 ? uint32_t(u64) : uint32_t(std::max(0.0, std::floor(num + 0.5)));
    }
    uint64_t as_u64(uint64_t def = 0) const {
        if (kind != Kind::Number) return def;
        return has_u64 ? u64 : uint64_t(std::max(0.0, std::floor(num + 0.5)));
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : s_(src) {}

    bool parse(JsonVal& out, std::string& err) {
        skip_ws();
        if (!parse_value(out, err)) return false;
        skip_ws();
        if (pos_ != s_.size()) {
            err = "unexpected trailing characters in JSON at offset " + std::to_string(pos_);
            return false;
        }
        return true;
    }

private:
    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }

    bool parse_value(JsonVal& out, std::string& err) {
        skip_ws();
        if (pos_ >= s_.size()) { err = "unexpected EOF in JSON"; return false; }
        const char c = s_[pos_];
        if (c == '{') return parse_object(out, err);
        if (c == '[') return parse_array(out, err);
        if (c == '"') {
            out = JsonVal{};
            out.kind = JsonVal::Kind::String;
            return parse_string(out.str, err);
        }
        if (c == 't' && s_.compare(pos_, 4, "true") == 0) {
            pos_ += 4; out = JsonVal{}; out.kind = JsonVal::Kind::Bool; out.b = true; return true;
        }
        if (c == 'f' && s_.compare(pos_, 5, "false") == 0) {
            pos_ += 5; out = JsonVal{}; out.kind = JsonVal::Kind::Bool; out.b = false; return true;
        }
        if (c == 'n' && s_.compare(pos_, 4, "null") == 0) {
            pos_ += 4; out = JsonVal{}; out.kind = JsonVal::Kind::Null; return true;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parse_number(out, err);
        }
        err = std::string("unexpected char '") + c + "' at offset " + std::to_string(pos_);
        return false;
    }

    bool parse_string(std::string& out, std::string& err) {
        if (pos_ >= s_.size() || s_[pos_] != '"') { err = "expected '\"'"; return false; }
        ++pos_;
        out.clear();
        while (pos_ < s_.size()) {
            const char c = s_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= s_.size()) { err = "unterminated escape"; return false; }
                const char e = s_[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > s_.size()) { err = "truncated \\u escape"; return false; }
                        unsigned cp = 0;
                        for (int k = 0; k < 4; ++k) {
                            const char h = s_[pos_++];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= unsigned(h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= unsigned(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= unsigned(h - 'A' + 10);
                            else { err = "invalid hex in \\u escape"; return false; }
                        }
                        if (cp < 0x80) out.push_back(char(cp));
                        else out.push_back('?');
                        break;
                    }
                    default: err = "invalid escape in JSON string"; return false;
                }
            } else {
                out.push_back(c);
            }
        }
        err = "unterminated JSON string";
        return false;
    }

    bool parse_number(JsonVal& out, std::string& err) {
        const size_t start = pos_;
        bool is_int = true;
        if (s_[pos_] == '-') { ++pos_; is_int = false; }
        if (pos_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
            err = "invalid JSON number";
            return false;
        }
        while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        if (pos_ < s_.size() && s_[pos_] == '.') {
            is_int = false;
            ++pos_;
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            is_int = false;
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        }
        const std::string tok = s_.substr(start, pos_ - start);
        out = JsonVal{};
        out.kind = JsonVal::Kind::Number;
        char* end = nullptr;
        out.num = std::strtod(tok.c_str(), &end);
        if (end != tok.c_str() + tok.size()) {
            err = "malformed JSON number: " + tok;
            return false;
        }
        if (is_int) {
            char* uend = nullptr;
            out.u64 = std::strtoull(tok.c_str(), &uend, 10);
            out.has_u64 = (uend == tok.c_str() + tok.size());
        }
        return true;
    }

    bool parse_array(JsonVal& out, std::string& err) {
        ++pos_;  // skip '['
        out = JsonVal{};
        out.kind = JsonVal::Kind::Array;
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return true; }
        while (true) {
            JsonVal elem;
            if (!parse_value(elem, err)) return false;
            out.arr.push_back(std::move(elem));
            skip_ws();
            if (pos_ >= s_.size()) { err = "unterminated JSON array"; return false; }
            if (s_[pos_] == ']') { ++pos_; return true; }
            if (s_[pos_] != ',') { err = "expected ',' or ']' in JSON array"; return false; }
            ++pos_;
        }
    }

    bool parse_object(JsonVal& out, std::string& err) {
        ++pos_;  // skip '{'
        out = JsonVal{};
        out.kind = JsonVal::Kind::Object;
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return true; }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(key, err)) return false;
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ':') {
                err = "expected ':' after JSON object key";
                return false;
            }
            ++pos_;
            JsonVal val;
            if (!parse_value(val, err)) return false;
            out.obj[std::move(key)] = std::move(val);
            skip_ws();
            if (pos_ >= s_.size()) { err = "unterminated JSON object"; return false; }
            if (s_[pos_] == '}') { ++pos_; return true; }
            if (s_[pos_] != ',') { err = "expected ',' or '}' in JSON object"; return false; }
            ++pos_;
        }
    }

    const std::string& s_;
    size_t pos_ = 0;
};

void parse_batch_array(const JsonVal* arr, std::vector<BatchLatencyPoint>& out) {
    out.clear();
    if (!arr || arr->kind != JsonVal::Kind::Array) return;
    for (const auto& item : arr->arr) {
        if (item.kind != JsonVal::Kind::Object) continue;
        BatchLatencyPoint pt{};
        if (const auto* n = item.get("n_experts")) pt.n_experts = n->as_i32();
        if (const auto* l = item.get("latency_ms")) pt.latency_ms = l->as_double();
        if (pt.n_experts > 0) out.push_back(pt);
    }
}

std::string to_upper(std::string s) {
    for (char& c : s) c = char(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

double eval_gpu_latency(const GpuProfile& g, int32_t n) {
    if (g.a_ipc_ms > 0.0 || g.b_per_expert_ms > 0.0) {
        return g.a_ipc_ms + g.b_per_expert_ms * double(n);
    }
    double a = 0.0, b = 0.0;
    fit_linear_latency(g.expert_latency, a, b);
    return a + b * double(n);
}

double eval_cpu_latency(const CpuProfile& c, int32_t n) {
    if (c.ms_base > 0.0 || c.ms_per_expert > 0.0) {
        return c.ms_base + c.ms_per_expert * double(n);
    }
    double a = 0.0, b = 0.0;
    fit_linear_latency(c.expert_latency, a, b);
    return a + b * double(n);
}

}  // namespace

std::string serialize_hw_profile_json(const HwProfile& prof) {
    std::ostringstream os;
    os << "{\n";
    os << "  \"schema_version\": " << prof.schema_version << ",\n";
    os << "  \"timestamp_utc\": " << json_escape(prof.timestamp_utc) << ",\n";
    os << "  \"pack_dir\": " << json_escape(prof.pack_dir) << ",\n";
    os << "  \"n_layers\": " << prof.n_layers << ",\n";
    os << "  \"n_expert\": " << prof.n_expert << ",\n";
    os << "  \"k\": " << prof.k << ",\n";
    os << "  \"n_embd\": " << prof.n_embd << ",\n";
    os << "  \"avg_blob_bytes\": " << prof.avg_blob_bytes << ",\n";
    os << "  \"cpu\": {\n";
    os << "    \"cpu_model\": " << json_escape(prof.cpu.cpu_model) << ",\n";
    os << "    \"logical_cores\": " << prof.cpu.logical_cores << ",\n";
    os << "    \"physical_cores\": " << prof.cpu.physical_cores << ",\n";
    os << "    \"ram_total_bytes\": " << prof.cpu.ram_total_bytes << ",\n";
    os << "    \"ram_available_bytes\": " << prof.cpu.ram_available_bytes << ",\n";
    os << "    \"ms_base\": " << fmt_double(prof.cpu.ms_base) << ",\n";
    os << "    \"ms_per_expert\": " << fmt_double(prof.cpu.ms_per_expert) << ",\n";
    os << "    \"expert_latency\": ";
    write_batch_array(os, prof.cpu.expert_latency, "    ");
    os << "\n  },\n";
    os << "  \"gpus\": [\n";
    for (size_t i = 0; i < prof.gpus.size(); ++i) {
        const auto& g = prof.gpus[i];
        os << "    {\n";
        os << "      \"id\": " << json_escape(g.id) << ",\n";
        os << "      \"vendor\": " << json_escape(g.vendor) << ",\n";
        os << "      \"device_name\": " << json_escape(g.device_name) << ",\n";
        os << "      \"vram_total_bytes\": " << g.vram_total_bytes << ",\n";
        os << "      \"vram_usable_bytes\": " << g.vram_usable_bytes << ",\n";
        os << "      \"recommended_slots\": " << g.recommended_slots << ",\n";
        os << "      \"pcie_h2d_gbps\": " << fmt_double(g.pcie_h2d_gbps) << ",\n";
        os << "      \"pcie_d2h_gbps\": " << fmt_double(g.pcie_d2h_gbps) << ",\n";
        os << "      \"a_ipc_ms\": " << fmt_double(g.a_ipc_ms) << ",\n";
        os << "      \"b_per_expert_ms\": " << fmt_double(g.b_per_expert_ms) << ",\n";
        os << "      \"expert_latency\": ";
        write_batch_array(os, g.expert_latency, "      ");
        os << "\n    }" << (i + 1 < prof.gpus.size() ? "," : "") << "\n";
    }
    os << "  ]\n";
    os << "}\n";
    return os.str();
}

bool parse_hw_profile_json(const std::string& json, HwProfile& out, std::string& err) {
    out = HwProfile{};
    JsonVal root;
    JsonParser parser(json);
    if (!parser.parse(root, err)) return false;
    if (root.kind != JsonVal::Kind::Object) {
        err = "hw_profile root must be a JSON object";
        return false;
    }

    if (const auto* v = root.get("schema_version")) out.schema_version = v->as_i32(1);
    if (out.schema_version != 1) {
        err = "unsupported franken-hw-profile schema_version: " + std::to_string(out.schema_version);
        return false;
    }
    if (const auto* v = root.get("timestamp_utc")) out.timestamp_utc = v->as_string();
    if (const auto* v = root.get("pack_dir")) out.pack_dir = v->as_string();
    if (const auto* v = root.get("n_layers")) out.n_layers = v->as_i32();
    if (const auto* v = root.get("n_expert")) out.n_expert = v->as_i32();
    if (const auto* v = root.get("k")) out.k = v->as_i32();
    if (const auto* v = root.get("n_embd")) out.n_embd = v->as_i32();
    if (const auto* v = root.get("avg_blob_bytes")) out.avg_blob_bytes = v->as_u64();

    if (const auto* cpu = root.get("cpu"); cpu && cpu->kind == JsonVal::Kind::Object) {
        if (const auto* v = cpu->get("cpu_model")) out.cpu.cpu_model = v->as_string();
        if (const auto* v = cpu->get("logical_cores")) out.cpu.logical_cores = v->as_i32();
        if (const auto* v = cpu->get("physical_cores")) out.cpu.physical_cores = v->as_i32();
        if (const auto* v = cpu->get("ram_total_bytes")) out.cpu.ram_total_bytes = v->as_u64();
        if (const auto* v = cpu->get("ram_available_bytes")) out.cpu.ram_available_bytes = v->as_u64();
        if (const auto* v = cpu->get("ms_base")) out.cpu.ms_base = v->as_double();
        if (const auto* v = cpu->get("ms_per_expert")) out.cpu.ms_per_expert = v->as_double();
        parse_batch_array(cpu->get("expert_latency"), out.cpu.expert_latency);
        if (out.cpu.ms_base == 0.0 && out.cpu.ms_per_expert == 0.0 && !out.cpu.expert_latency.empty()) {
            fit_linear_latency(out.cpu.expert_latency, out.cpu.ms_base, out.cpu.ms_per_expert);
        }
    }

    if (const auto* gpus = root.get("gpus"); gpus && gpus->kind == JsonVal::Kind::Array) {
        for (const auto& item : gpus->arr) {
            if (item.kind != JsonVal::Kind::Object) continue;
            GpuProfile g{};
            if (const auto* v = item.get("id")) g.id = v->as_string();
            if (const auto* v = item.get("vendor")) g.vendor = v->as_string();
            if (const auto* v = item.get("device_name")) g.device_name = v->as_string();
            if (const auto* v = item.get("vram_total_bytes")) g.vram_total_bytes = v->as_u64();
            if (const auto* v = item.get("vram_usable_bytes")) g.vram_usable_bytes = v->as_u64();
            if (const auto* v = item.get("recommended_slots")) g.recommended_slots = v->as_u32();
            if (const auto* v = item.get("pcie_h2d_gbps")) g.pcie_h2d_gbps = v->as_double();
            if (const auto* v = item.get("pcie_d2h_gbps")) g.pcie_d2h_gbps = v->as_double();
            if (const auto* v = item.get("a_ipc_ms")) g.a_ipc_ms = v->as_double();
            if (const auto* v = item.get("b_per_expert_ms")) g.b_per_expert_ms = v->as_double();
            parse_batch_array(item.get("expert_latency"), g.expert_latency);
            if (g.a_ipc_ms == 0.0 && g.b_per_expert_ms == 0.0 && !g.expert_latency.empty()) {
                fit_linear_latency(g.expert_latency, g.a_ipc_ms, g.b_per_expert_ms);
            }
            out.gpus.push_back(std::move(g));
        }
    }
    return true;
}

bool save_hw_profile(const std::string& path, const HwProfile& prof, std::string& err) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        err = "cannot open hardware profile for writing: " + path;
        return false;
    }
    const std::string text = serialize_hw_profile_json(prof);
    out.write(text.data(), std::streamsize(text.size()));
    if (!out) {
        err = "failed writing hardware profile: " + path;
        return false;
    }
    return true;
}

bool load_hw_profile(const std::string& path, HwProfile& out, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot open hardware profile: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_hw_profile_json(ss.str(), out, err);
}

TierPlan recommend_tier_plan(const HwProfile& prof,
                             uint32_t total_model_experts,
                             const std::string& preferred_primary_id) {
    TierPlan plan{};
    if (prof.gpus.empty()) return plan;

    const uint32_t total_exp = total_model_experts > 0
                                   ? total_model_experts
                                   : uint32_t(std::max<int64_t>(0, int64_t(prof.n_layers) * int64_t(prof.n_expert)));
    const uint64_t blob_b = prof.avg_blob_bytes > 0 ? prof.avg_blob_bytes : 1382400ull;

    auto gpu_slots = [&](const GpuProfile& g) -> uint32_t {
        if (g.recommended_slots > 0) return g.recommended_slots;
        return uint32_t(g.vram_usable_bytes / blob_b);
    };

    // Rank candidates by lowest decode latency at a typical 4-expert batch,
    // preferring true GPUs over CPU-emulated agents for Primary selection.
    std::vector<size_t> order(prof.gpus.size());
    std::iota(order.begin(), order.end(), 0);

    auto is_real_gpu = [&](size_t idx) { return to_upper(prof.gpus[idx].vendor) != "CPU"; };
    auto score_t4 = [&](size_t idx) { return std::max(1e-6, eval_gpu_latency(prof.gpus[idx], 4)); };

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (is_real_gpu(a) != is_real_gpu(b)) return is_real_gpu(a);
        const double ta = score_t4(a), tb = score_t4(b);
        if (std::fabs(ta - tb) > 1e-6) return ta < tb;
        if (prof.gpus[a].vram_usable_bytes != prof.gpus[b].vram_usable_bytes) {
            return prof.gpus[a].vram_usable_bytes > prof.gpus[b].vram_usable_bytes;
        }
        return prof.gpus[a].id < prof.gpus[b].id;
    });

    size_t primary_pos = 0;
    if (!preferred_primary_id.empty()) {
        for (size_t i = 0; i < order.size(); ++i) {
            if (prof.gpus[order[i]].id == preferred_primary_id) {
                primary_pos = i;
                break;
            }
        }
    }

    const size_t p_idx = order[primary_pos];
    const GpuProfile& pg = prof.gpus[p_idx];
    plan.primary_gpu_index = int32_t(p_idx);
    plan.primary_id = pg.id;
    plan.primary_vendor = pg.vendor;
    plan.primary_device_name = pg.device_name;
    plan.primary_slot_budget = total_exp > 0 ? std::min(total_exp, gpu_slots(pg)) : gpu_slots(pg);

    uint32_t remaining = total_exp > plan.primary_slot_budget ? (total_exp - plan.primary_slot_budget) : 0;

    std::ostringstream cli;
    for (size_t i = 0; i < order.size(); ++i) {
        if (i == primary_pos) continue;
        const GpuProfile& ag = prof.gpus[order[i]];
        double a_ms = ag.a_ipc_ms, b_ms = ag.b_per_expert_ms;
        if (a_ms == 0.0 && b_ms == 0.0 && !ag.expert_latency.empty()) {
            fit_linear_latency(ag.expert_latency, a_ms, b_ms);
        }

        AgentTierPlan ap{};
        ap.id = ag.id;
        ap.vendor = ag.vendor;
        ap.device_name = ag.device_name;
        const uint32_t cap = gpu_slots(ag);
        ap.slot_budget = total_exp > 0 ? std::min(cap, remaining) : cap;
        if (remaining >= ap.slot_budget) remaining -= ap.slot_budget;
        else remaining = 0;

        ap.a_ipc_ms = a_ms;
        ap.b_per_expert_ms = b_ms;
        const double t4 = std::max(1e-6, a_ms + b_ms * 4.0);
        ap.effective_experts_per_ms = 4.0 / t4;

        // Compute minimum batch size where agent beats CPU.
        int32_t thresh = 1;
        const bool has_cpu_curve = prof.cpu.ms_base > 0.0 || prof.cpu.ms_per_expert > 0.0 ||
                                   !prof.cpu.expert_latency.empty();
        if (has_cpu_curve) {
            thresh = 16;
            for (int32_t n = 1; n <= 16; ++n) {
                if (eval_gpu_latency(ag, n) < eval_cpu_latency(prof.cpu, n)) {
                    thresh = n;
                    break;
                }
            }
        }
        ap.min_entries_threshold = thresh;

        if (!ap.id.empty()) {
            if (cli.tellp() > 0) cli << " ";
            cli << "--franken-agent " << ap.id;
            if (ap.slot_budget > 0) cli << ",slots=" << ap.slot_budget;
        }
        plan.agents.push_back(std::move(ap));
    }

    if (!plan.agents.empty()) {
        cli << " --franken-cost-model on";
    }
    plan.recommended_cli_flags = cli.str();
    return plan;
}

const GpuProfile* find_gpu_profile(const HwProfile& prof,
                                   const std::string& shm_name,
                                   const std::string& vendor_str) {
    auto norm = [](const std::string& s) -> std::string {
        return (!s.empty() && s[0] == '/') ? s.substr(1) : s;
    };
    const std::string want_id = norm(shm_name);
    for (const auto& g : prof.gpus) {
        if (!want_id.empty() && norm(g.id) == want_id) return &g;
    }
    const std::string want_v = to_upper(vendor_str);
    for (const auto& g : prof.gpus) {
        if (!want_v.empty() && to_upper(g.vendor) == want_v) return &g;
    }
    return nullptr;
}

}  // namespace profiler
}  // namespace franken
