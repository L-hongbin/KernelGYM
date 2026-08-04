// Production LD_PRELOAD shim: avoid the CUDA 12.6u2-13.0 CUPTI timestamp bug.
//
// NVIDIA-confirmed defect: with cuptiActivityRegisterTimestampCallback
// registered (Kineto's TSC fast path), CUPTI can emit kernel activity records
// with start=0, which Kineto drops as out-of-window, so torch.profiler
// captures zero CUDA kernels. Fixed in CUPTI 13.1.
//
// On affected CUPTI versions this shim suppresses the callback registration
// and flips Kineto's exported process-global `libkineto::use_cupti_tsc()`
// flag to false so Kineto interprets CUPTI's native nanosecond timestamps
// correctly (suppressing registration alone would make Kineto misread native
// nanoseconds as TSC ticks). On unaffected versions, or whenever anything
// about the environment is not exactly as expected, it passes the call
// through to the real CUPTI function so behavior is stock.
//
// The current state is queryable via kernelgym_cupti_tsc_shim_state() so the
// Python side can verify the shim actually engaged before trusting
// single-forward profiling (see kernelgym/utils/cupti_tsc_shim.py and
// docs/design-doc/PROFILER_EMPTY_CAPTURE.md).
//
// Deliberately header-free: only two C symbols are touched, declared locally,
// and there are no static constructors, so preloading this into unrelated
// child processes (nvcc, ninja, redis) is inert.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <initializer_list>

namespace {

// Keep in sync with _CUPTI_TSC_BUG_MIN_CUDA / _CUPTI_TSC_BUG_FIXED_CUDA in
// kernelgym/toolkit/kernelbench/timing.py. CUPTI API version 24 is CUDA 12.6
// (GA/U1/U2 are indistinguishable, so the gate is conservative); 130100 is
// CUDA 13.1, which ships the vendor fix.
constexpr uint32_t kAffectedMinCuptiVersion = 24;
constexpr uint32_t kFixedCuptiVersion = 130100;

constexpr int kCuptiSuccess = 0;

// States exposed through kernelgym_cupti_tsc_shim_state().
enum ShimState : int {
  kStateNotCalled = 0,        // shim loaded, Kineto has not registered yet
  kStateEngagedNative = 1,    // registration suppressed, Kineto on native timestamps
  kStatePassthroughFixed = 2, // CUPTI is unaffected, real callback registered
  kStatePassthroughError = 3, // could not engage safely, stock behavior kept
  kStateFailed = 4,           // could not engage nor reach the real CUPTI symbol
};

std::atomic<int> g_state{kStateNotCalled};
std::atomic<uint32_t> g_cupti_version{0};

void report_once(const char* message) {
  static std::atomic<bool> reported{false};
  if (!reported.exchange(true)) {
    std::fprintf(stderr, "[kernelgym-cupti-tsc-shim] %s\n", message);
    std::fflush(stderr);
  }
}

using TimestampCallback = uint64_t (*)();
using RegisterFn = int (*)(TimestampCallback);
using GetVersionFn = int (*)(uint32_t*);
using UseCuptiTscFn = bool& (*)();

// Look up a symbol from the already-loaded CUPTI. libcupti may live outside
// the global symbol scope (dlopened RTLD_LOCAL by torch), where RTLD_NEXT and
// RTLD_DEFAULT cannot see it, so also try handles to its known sonames with
// RTLD_NOLOAD (which finds the library regardless of scope and never loads a
// new copy).
void* cupti_symbol(const char* name) {
  void* symbol = dlsym(RTLD_NEXT, name);
  if (symbol == nullptr) {
    symbol = dlsym(RTLD_DEFAULT, name);
  }
  if (symbol == nullptr) {
    for (const char* soname : {"libcupti.so.12", "libcupti.so.13", "libcupti.so"}) {
      void* handle = dlopen(soname, RTLD_NOW | RTLD_NOLOAD);
      if (handle != nullptr) {
        symbol = dlsym(handle, name);
        if (symbol != nullptr) {
          break;
        }
      }
    }
  }
  return symbol;
}

RegisterFn real_register_fn() {
  return reinterpret_cast<RegisterFn>(
      cupti_symbol("cuptiActivityRegisterTimestampCallback"));
}

// True when the loaded CUPTI is in the affected range. Unknown or unqueryable
// versions count as affected (fail-safe: prefer native timestamps over
// re-triggering the bad callback on an unidentified CUPTI).
bool cupti_version_affected() {
  auto* get_version = reinterpret_cast<GetVersionFn>(cupti_symbol("cuptiGetVersion"));
  uint32_t version = 0;
  if (get_version == nullptr || get_version(&version) != kCuptiSuccess) {
    report_once("cuptiGetVersion unavailable; treating CUPTI as affected");
    return true;
  }
  g_cupti_version.store(version);
  return version >= kAffectedMinCuptiVersion && version < kFixedCuptiVersion;
}

UseCuptiTscFn kineto_use_cupti_tsc_fn() {
  // libtorch_cpu.so is already loaded by the time Kineto registers the
  // callback; RTLD_NOLOAD only obtains a handle to it. The direct RTLD_DEFAULT
  // lookup is a fallback for builds where the library scope differs.
  void* torch_cpu = dlopen("libtorch_cpu.so", RTLD_NOW | RTLD_NOLOAD);
  void* symbol = torch_cpu == nullptr
      ? nullptr
      : dlsym(torch_cpu, "_ZN9libkineto13use_cupti_tscEv");
  if (symbol == nullptr) {
    symbol = dlsym(RTLD_DEFAULT, "_ZN9libkineto13use_cupti_tscEv");
  }
  return reinterpret_cast<UseCuptiTscFn>(symbol);
}

}  // namespace

extern "C" int kernelgym_cupti_tsc_shim_state() {
  return g_state.load();
}

extern "C" uint32_t kernelgym_cupti_tsc_shim_cupti_version() {
  return g_cupti_version.load();
}

extern "C" int cuptiActivityRegisterTimestampCallback(TimestampCallback callback) {
  const bool affected = cupti_version_affected();

  if (!affected) {
    RegisterFn real_register = real_register_fn();
    if (real_register == nullptr) {
      report_once("real cuptiActivityRegisterTimestampCallback not found");
      g_state.store(kStateFailed);
      return kCuptiSuccess;  // pretend success; Kineto stays on its TSC path
    }
    g_state.store(kStatePassthroughFixed);
    report_once("CUPTI unaffected; passing TSC callback registration through");
    return real_register(callback);
  }

  UseCuptiTscFn use_cupti_tsc = kineto_use_cupti_tsc_fn();
  if (use_cupti_tsc == nullptr) {
    // Cannot flip Kineto's interpretation flag. Suppressing registration
    // anyway would make Kineto misinterpret native timestamps, which is worse
    // than the original bug, so keep stock behavior.
    RegisterFn real_register = real_register_fn();
    if (real_register == nullptr) {
      report_once("Kineto TSC flag and real CUPTI symbol both unavailable");
      g_state.store(kStateFailed);
      return kCuptiSuccess;
    }
    g_state.store(kStatePassthroughError);
    report_once("Kineto use_cupti_tsc flag not found; keeping stock TSC callback");
    return real_register(callback);
  }

  use_cupti_tsc() = false;
  g_state.store(kStateEngagedNative);
  report_once("affected CUPTI detected; TSC callback suppressed, Kineto on native timestamps");
  (void)callback;
  return kCuptiSuccess;
}
