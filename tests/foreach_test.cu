#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "./catch.hpp"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <ckt/include/utility.hpp>
#include <ckt/include/cuda_config.hpp>
#include <ckt/include/for_each.hpp>

using namespace ckt;

// a lambda-like CUDA kernel
template <class T>
class atomic_run_func: public nvstd::function<void(T)> {
public:
  __device__ void operator()(int gid, thrust::tuple<T*, T> &tuple) const {
    atomicAdd(thrust::get<0>(tuple), thrust::get<1>(tuple));
  }
};

TEST_CASE( "ForEachStride", "[sum]" ) {
  
  thrust::device_vector<int> d_sum(1);
  
  int n1 = 3;
  int add_per_thread = 2;
  ForEachKernel<StridePolicy, ckt::konst::default_cuda_blocksize, false> kernel(n1, "AtomicAdd");

  kernel.run<atomic_run_func<int>, int *, int>(thrust::raw_pointer_cast(&d_sum[0]), add_per_thread);
  int sum_now = d_sum[0];
  check_cuda_error("atomic", __FILE__, __LINE__);
  REQUIRE(sum_now == n1*add_per_thread);

  kernel.set_N(257);
  kernel.run<atomic_run_func<int>, int *, int >(thrust::raw_pointer_cast(&d_sum[0]), 12);
  REQUIRE(d_sum[0] == (sum_now + 257*12));

  check_cuda_error("atomic", __FILE__, __LINE__);
}


