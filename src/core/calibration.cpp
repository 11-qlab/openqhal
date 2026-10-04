#include "qhal/calibration.hpp"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cctype>

namespace qhal {

const QubitCalibration& Calibration::get(int32_t q) const {
    auto it = qubits.find(q);
    if (it == qubits.end())
        throw std::runtime_error("Calibration: unknown qubit " + std::to_string(q));
    return it->second;
}

// ---------------------------------------------------------------------
// Minimal JSON parser. Handles: objects, arrays, numbers, strings,
// booleans. Does NOT handle escapes, nested arrays, or null.
// Sufficient for the calibration schema. Swap for nlohmann/json in
// production.
// ---------------------------------------------------------------------
namespace {

struct JsonValue {
    enum Type { Null, Bool, Number, String, Object, Array } type = Null;
    bool            b = false;
    double          n = 0.0;
    std::string     s;
    std::vector<std::pair<std::string, JsonValue>> obj;
    std::vector<JsonValue> arr;

    const JsonValue* find(const std::string& key) const {
        for (const auto& [k, v] : obj) if (k == key) return &v;
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : s_(src), i_(0) {}

    JsonValue parse() {
        skip_ws();
        return parse_value();
    }

private:
    const std::string& s_;
    size_t i_;

    void skip_ws() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }

    char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }
    char get()  { return i_ < s_.size() ? s_[i_++] : '\0'; }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string_value();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        throw std::runtime_error("JSON: unexpected character at " + std::to_string(i_));
    }

    JsonValue parse_object() {
        JsonValue v; v.type = JsonValue::Object;
        get();   // consume '{'
        skip_ws();
        if (peek() == '}') { get(); return v; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (get() != ':') throw std::runtime_error("JSON: expected ':'");
            JsonValue val = parse_value();
            v.obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            char c = get();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("JSON: expected ',' or '}'");
        }
        return v;
    }

    JsonValue parse_array() {
        JsonValue v; v.type = JsonValue::Array;
        get();   // consume '['
        skip_ws();
        if (peek() == ']') { get(); return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            char c = get();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("JSON: expected ',' or ']'");
        }
        return v;
    }

    std::string parse_string() {
        if (get() != '"') throw std::runtime_error("JSON: expected '\"'");
        std::string out;
        while (i_ < s_.size()) {
            char c = get();
            if (c == '"') return out;
            out.push_back(c);
        }
        throw std::runtime_error("JSON: unterminated string");
    }

    JsonValue parse_string_value() {
        JsonValue v; v.type = JsonValue::String; v.s = parse_string();
        return v;
    }

    JsonValue parse_bool() {
        JsonValue v; v.type = JsonValue::Bool;
        if (s_.compare(i_, 4, "true") == 0)  { v.b = true;  i_ += 4; }
        else if (s_.compare(i_, 5, "false") == 0) { v.b = false; i_ += 5; }
        else throw std::runtime_error("JSON: bad bool");
        return v;
    }

    JsonValue parse_number() {
        JsonValue v; v.type = JsonValue::Number;
        size_t start = i_;
        if (peek() == '-') ++i_;
        while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) ||
                                  s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' ||
                                  s_[i_] == '+' || s_[i_] == '-')) ++i_;
        v.n = std::stod(s_.substr(start, i_ - start));
        return v;
    }
};

double num_or(const JsonValue* v, double def) {
    return (v && v->type == JsonValue::Number) ? v->n : def;
}
bool bool_or(const JsonValue* v, bool def) {
    return (v && v->type == JsonValue::Bool) ? v->b : def;
}

} // namespace

// ---------------------------------------------------------------------
// Public loaders
// ---------------------------------------------------------------------

Calibration load_calibration_string(const std::string& json) {
    Calibration cal;
    if (json.empty()) {
        // Return default 4-qubit calibration
        cal.device_name = "default";
        for (int q = 0; q < 4; ++q) {
            QubitCalibration qc;
            qc.qubit_id = q;
            cal.qubits[q] = qc;
        }
        return cal;
    }

    JsonParser parser(json);
    JsonValue root = parser.parse();

    if (root.type != JsonValue::Object)
        throw std::runtime_error("Calibration: JSON root must be object");

    if (auto* dn = root.find("device_name"); dn && dn->type == JsonValue::String)
        cal.device_name = dn->s;

    if (auto* cd = root.find("cnot_duration_ns"))
        cal.cnot_duration_ns = num_or(cd, cal.cnot_duration_ns);
    if (auto* cz = root.find("cz_duration_ns"))
        cal.cz_duration_ns = num_or(cz, cal.cz_duration_ns);
    if (auto* md = root.find("measure_duration_ns"))
        cal.measure_duration_ns = num_or(md, cal.measure_duration_ns);

    if (auto* qs = root.find("qubits"); qs && qs->type == JsonValue::Object) {
        for (const auto& [key, val] : qs->obj) {
            if (val.type != JsonValue::Object) continue;
            QubitCalibration qc;
            qc.qubit_id            = std::stoi(key);
            qc.rabi_hz             = num_or(val.find("rabi_hz"),             qc.rabi_hz);
            qc.anharm_hz           = num_or(val.find("anharm_hz"),           qc.anharm_hz);
            qc.drive_freq_hz       = num_or(val.find("drive_freq_hz"),       qc.drive_freq_hz);
            qc.t1_ns               = num_or(val.find("t1_ns"),               qc.t1_ns);
            qc.t2_ns               = num_or(val.find("t2_ns"),               qc.t2_ns);
            qc.drag_beta           = num_or(val.find("drag_beta"),           qc.drag_beta);
            qc.sigma_frac          = num_or(val.find("sigma_frac"),          qc.sigma_frac);
            qc.readout_duration_ns = num_or(val.find("readout_duration_ns"), qc.readout_duration_ns);
            qc.virtual_z           = bool_or(val.find("virtual_z"),          qc.virtual_z);
            cal.qubits[qc.qubit_id] = qc;
        }
    }

    // If no qubits were defined, fill in defaults
    if (cal.qubits.empty()) {
        for (int q = 0; q < 4; ++q) {
            QubitCalibration qc;
            qc.qubit_id = q;
            cal.qubits[q] = qc;
        }
    }

    return cal;
}

Calibration load_calibration_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Calibration: cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return load_calibration_string(ss.str());
}

void save_calibration_json(const Calibration& cal, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Calibration: cannot write " + path);

    f << "{\n";
    f << "  \"version\": 1,\n";
    f << "  \"device_name\": \"" << cal.device_name << "\",\n";
    f << "  \"cnot_duration_ns\": " << cal.cnot_duration_ns << ",\n";
    f << "  \"cz_duration_ns\": "   << cal.cz_duration_ns   << ",\n";
    f << "  \"measure_duration_ns\": " << cal.measure_duration_ns << ",\n";
    f << "  \"qubits\": {\n";

    size_t idx = 0;
    for (const auto& [qid, qc] : cal.qubits) {
        f << "    \"" << qid << "\": {\n";
        f << "      \"rabi_hz\": "             << qc.rabi_hz             << ",\n";
        f << "      \"anharm_hz\": "           << qc.anharm_hz           << ",\n";
        f << "      \"drive_freq_hz\": "       << qc.drive_freq_hz       << ",\n";
        f << "      \"t1_ns\": "               << qc.t1_ns               << ",\n";
        f << "      \"t2_ns\": "               << qc.t2_ns               << ",\n";
        f << "      \"drag_beta\": "           << qc.drag_beta           << ",\n";
        f << "      \"sigma_frac\": "          << qc.sigma_frac          << ",\n";
        f << "      \"readout_duration_ns\": " << qc.readout_duration_ns << ",\n";
        f << "      \"virtual_z\": "           << (qc.virtual_z ? "true" : "false") << "\n";
        f << "    }" << (++idx < cal.qubits.size() ? "," : "") << "\n";
    }
    f << "  }\n";
    f << "}\n";
}

} // namespace qhal
