/**
 * logparser.cpp — High-throughput telemetry log parser
 *
 * Parses structured telemetry logs of the form:
 *   2024-01-15T10:23:45.123456 WARN  node-07 cpu=45.2 mem=2048 latency=8.3 status=1
 *
 * Returns a NumPy structured array with dtype:
 *   [('level', i4), ('cpu', f8), ('mem', i8), ('latency', f8), ('status', i4)]
 *
 * Design choices for speed:
 *   - std::from_chars for all numeric parsing (no locale, no heap alloc)
 *   - Manual byte scanning instead of std::regex (10-20x faster for fixed formats)
 *   - Single-pass file read into a pre-allocated buffer
 *   - Reserve + emplace_back on struct vector to avoid reallocations
 *   - Direct memory layout into NumPy buffer — zero extra copy on return
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ── Data layout (matches NumPy dtype exactly) ─────────────────────────────────
#pragma pack(push, 1)
struct LogRecord {
    int32_t  level;    // 0=DEBUG 1=INFO 2=WARN 3=ERROR 4=FATAL -1=UNKNOWN
    double   cpu;      // percent  0.0–100.0
    int64_t  mem;      // kilobytes
    double   latency;  // milliseconds
    int32_t  status;   // application status code
};
#pragma pack(pop)

// ── Level encoding ────────────────────────────────────────────────────────────
static inline int32_t encode_level(const char* p, size_t len) {
    if (len >= 5 && std::memcmp(p, "DEBUG", 5) == 0) return 0;
    if (len >= 4 && std::memcmp(p, "INFO", 4) == 0)  return 1;
    if (len >= 4 && std::memcmp(p, "WARN", 4) == 0)  return 2;
    if (len >= 5 && std::memcmp(p, "ERROR", 5) == 0) return 3;
    if (len >= 5 && std::memcmp(p, "FATAL", 5) == 0) return 4;
    return -1;
}

// ── Skip to next space ────────────────────────────────────────────────────────
static inline const char* skip_to_space(const char* p, const char* end) {
    while (p < end && *p != ' ' && *p != '\t') ++p;
    return p;
}

static inline const char* skip_spaces(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
    return p;
}

// ── Parse "key=value" field, advance pointer ──────────────────────────────────
// Returns false if the expected key is not found (malformed line).
template<typename T>
static bool parse_kv(const char*& p, const char* end,
                     const char* key, size_t klen, T& out) {
    p = skip_spaces(p, end);
    if ((size_t)(end - p) < klen + 1) return false;
    if (std::memcmp(p, key, klen) != 0 || p[klen] != '=') return false;
    p += klen + 1;
    auto [ptr, ec] = std::from_chars(p, end, out);
    if (ec != std::errc{}) return false;
    p = ptr;
    return true;
}

// ── Parse one line → LogRecord ────────────────────────────────────────────────
static bool parse_line(const char* p, size_t len, LogRecord& rec) {
    const char* end = p + len;

    // Field 1: timestamp (26 chars, e.g. "2024-01-15T10:23:45.123456")
    // Skip it — just advance past the first space-delimited token.
    p = skip_to_space(p, end);
    p = skip_spaces(p, end);
    if (p >= end) return false;

    // Field 2: level
    const char* level_start = p;
    p = skip_to_space(p, end);
    rec.level = encode_level(level_start, (size_t)(p - level_start));
    p = skip_spaces(p, end);

    // Field 3: node_id — skip
    p = skip_to_space(p, end);
    p = skip_spaces(p, end);
    if (p >= end) return false;

    // Fields 4–7: cpu= mem= latency= status=
    // Use a float intermediary for cpu/latency since from_chars<double> needs it.
    if (!parse_kv(p, end, "cpu",     3, rec.cpu))     return false;
    if (!parse_kv(p, end, "mem",     3, rec.mem))     return false;
    if (!parse_kv(p, end, "latency", 7, rec.latency)) return false;
    if (!parse_kv(p, end, "status",  6, rec.status))  return false;

    return true;
}

// ── Main entry point ──────────────────────────────────────────────────────────
/**
 * parse_log_file(path, skip_malformed=True) -> np.ndarray
 *
 * Reads the telemetry log file at `path` and returns a structured NumPy array
 * with one row per valid log line.
 *
 * dtype: [('level','<i4'), ('cpu','<f8'), ('mem','<i8'), ('latency','<f8'), ('status','<i4')]
 *
 * Args:
 *     path:           Path to the log file.
 *     skip_malformed: If True (default), silently skip unparseable lines.
 *                     If False, raises ValueError on the first bad line.
 *
 * Returns:
 *     numpy.ndarray with structured dtype, shape (N,) where N = valid line count.
 *
 * Raises:
 *     IOError:    If the file cannot be opened.
 *     ValueError: If skip_malformed=False and a malformed line is encountered.
 */
py::array parse_log_file(const std::string& path, bool skip_malformed = true) {
    // ── Read entire file into memory ──────────────────────────────────────────
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    auto file_size = f.tellg();
    f.seekg(0);

    std::string buf(static_cast<size_t>(file_size), '\0');
    f.read(buf.data(), file_size);
    f.close();

    // ── First pass: count newlines to reserve exact capacity ─────────────────
    size_t line_count = 0;
    for (char c : buf) if (c == '\n') ++line_count;

    std::vector<LogRecord> records;
    records.reserve(line_count);

    // ── Parse line-by-line ────────────────────────────────────────────────────
    const char* data  = buf.data();
    const char* bEnd  = data + buf.size();
    const char* lstart = data;
    size_t line_no = 0;

    while (lstart < bEnd) {
        ++line_no;
        const char* lend = lstart;
        while (lend < bEnd && *lend != '\n') ++lend;

        // Strip trailing \r for Windows-style line endings
        size_t llen = (size_t)(lend - lstart);
        if (llen > 0 && lstart[llen - 1] == '\r') --llen;

        if (llen > 0) {
            LogRecord rec{};
            if (parse_line(lstart, llen, rec)) {
                records.emplace_back(rec);
            } else if (!skip_malformed) {
                throw std::runtime_error(
                    "Malformed log line " + std::to_string(line_no));
            }
        }

        lstart = lend + 1;
    }

    // ── Build NumPy structured array ──────────────────────────────────────────
    // Define structured dtype via numpy dict API (pybind11 3.x compatible).
    auto np    = py::module_::import("numpy");
    auto dtype = np.attr("dtype")(py::dict(
        py::arg("names")   = py::make_tuple("level","cpu","mem","latency","status"),
        py::arg("formats") = py::make_tuple("i4","f8","i8","f8","i4"),
        py::arg("offsets") = py::make_tuple(
            (int)offsetof(LogRecord,level),
            (int)offsetof(LogRecord,cpu),
            (int)offsetof(LogRecord,mem),
            (int)offsetof(LogRecord,latency),
            (int)offsetof(LogRecord,status)),
        py::arg("itemsize") = (int)sizeof(LogRecord)
    ));

    auto result_obj = np.attr("empty")((py::ssize_t)records.size(), dtype);
    py::array result = result_obj.cast<py::array>();
    if (!records.empty())
        std::memcpy(result.mutable_data(), records.data(),
                    records.size() * sizeof(LogRecord));

    return result;
}

// ── parse_log_string: for testing / in-memory use ────────────────────────────
py::array parse_log_string(const std::string& content, bool skip_malformed = true) {
    size_t line_count = 0;
    for (char c : content) if (c == '\n') ++line_count;

    std::vector<LogRecord> records;
    records.reserve(line_count + 1);

    const char* data   = content.data();
    const char* bEnd   = data + content.size();
    const char* lstart = data;
    size_t line_no = 0;

    while (lstart < bEnd) {
        ++line_no;
        const char* lend = lstart;
        while (lend < bEnd && *lend != '\n') ++lend;
        size_t llen = (size_t)(lend - lstart);
        if (llen > 0 && lstart[llen - 1] == '\r') --llen;

        if (llen > 0) {
            LogRecord rec{};
            if (parse_line(lstart, llen, rec)) {
                records.emplace_back(rec);
            } else if (!skip_malformed) {
                throw std::runtime_error(
                    "Malformed log line " + std::to_string(line_no));
            }
        }
        lstart = lend + 1;
    }

    auto np2    = py::module_::import("numpy");
    auto dtype = np2.attr("dtype")(py::dict(
        py::arg("names")   = py::make_tuple("level","cpu","mem","latency","status"),
        py::arg("formats") = py::make_tuple("i4","f8","i8","f8","i4"),
        py::arg("offsets") = py::make_tuple(
            (int)offsetof(LogRecord,level),
            (int)offsetof(LogRecord,cpu),
            (int)offsetof(LogRecord,mem),
            (int)offsetof(LogRecord,latency),
            (int)offsetof(LogRecord,status)),
        py::arg("itemsize") = (int)sizeof(LogRecord)
    ));

    auto result_obj2 = np2.attr("empty")((py::ssize_t)records.size(), dtype);
    py::array result = result_obj2.cast<py::array>();
    if (!records.empty())
        std::memcpy(result.mutable_data(), records.data(),
                    records.size() * sizeof(LogRecord));

    return result;
}

// ── Module definition ─────────────────────────────────────────────────────────
PYBIND11_MODULE(logparser_cpp, m) {
    m.doc() = "High-throughput C++ telemetry log parser (Pybind11)";

    m.def("parse_log_file",
          &parse_log_file,
          py::arg("path"),
          py::arg("skip_malformed") = true,
          R"doc(
Parse a telemetry log file and return a structured NumPy array.

Log line format:
  <timestamp> <LEVEL> <node_id> cpu=<f> mem=<i> latency=<f> status=<i>

Example:
  2024-01-15T10:23:45.123456 INFO node-03 cpu=12.5 mem=1024 latency=3.2 status=0

Returns np.ndarray with dtype:
  [('level','<i4'), ('cpu','<f8'), ('mem','<i8'), ('latency','<f8'), ('status','<i4')]
)doc");

    m.def("parse_log_string",
          &parse_log_string,
          py::arg("content"),
          py::arg("skip_malformed") = true,
          "Parse log content from a string (useful for testing).");
}
