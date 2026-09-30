// Elementwise add/mul/div/fma loops for binary32 and binary64, compiled
// natively, with the PRISM SR pass, and with Verificarlo MCA instrumentation.
// Usage: bench <reps>. Prints the median ns/element over TRIALS trials.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr size_t N = 1 << 16;
constexpr int TRIALS = 7;
static volatile double sink = 0;

#define KERNEL(name, T, expr)                                                  \
  __attribute__((noinline)) void name(const T *__restrict__ a,                 \
                                      const T *__restrict__ b,                 \
                                      const T *__restrict__ c,                 \
                                      T *__restrict__ o) {                     \
    for (size_t i = 0; i < N; ++i) o[i] = expr;                                \
  }

KERNEL(add_f32, float, a[i] + b[i])
KERNEL(mul_f32, float, a[i] * b[i])
KERNEL(div_f32, float, a[i] / b[i])
KERNEL(fma_f32, float, __builtin_fmaf(a[i], b[i], c[i]))
KERNEL(add_f64, double, a[i] + b[i])
KERNEL(mul_f64, double, a[i] * b[i])
KERNEL(div_f64, double, a[i] / b[i])
KERNEL(fma_f64, double, __builtin_fma(a[i], b[i], c[i]))

template <typename T>
using Kernel = void (*)(const T *, const T *, const T *, T *);

template <typename T>
void run(const char *op, const char *type, Kernel<T> k, int reps) {
  std::vector<T> a(N), b(N), c(N), o(N);
  for (size_t i = 0; i < N; ++i) {
    a[i] = T(1) + T(i % 97) / T(97);
    b[i] = T(1) + T(i % 89) / T(89);
    c[i] = T(1) + T(i % 83) / T(83);
  }
  k(a.data(), b.data(), c.data(), o.data());
  double t[TRIALS];
  for (int tr = 0; tr < TRIALS; ++tr) {
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
      k(a.data(), b.data(), c.data(), o.data());
      asm volatile("" ::: "memory");
    }
    auto stop = std::chrono::steady_clock::now();
    t[tr] = std::chrono::duration<double, std::nano>(stop - start).count() /
            (double(reps) * N);
    sink += o[N / 2];
  }
  std::sort(t, t + TRIALS);
  std::printf("%-4s %-9s %10.4f\n", op, type, t[TRIALS / 2]);
}

int main(int argc, char **argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 100;
  std::printf("# N=%zu reps=%d trials=%d; median ns/element\n", N, reps, TRIALS);
  run<float>("add", "binary32", add_f32, reps);
  run<float>("mul", "binary32", mul_f32, reps);
  run<float>("div", "binary32", div_f32, reps);
  run<float>("fma", "binary32", fma_f32, reps);
  run<double>("add", "binary64", add_f64, reps);
  run<double>("mul", "binary64", mul_f64, reps);
  run<double>("div", "binary64", div_f64, reps);
  run<double>("fma", "binary64", fma_f64, reps);
}
