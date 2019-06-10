#include "./catch.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <ckt/include/cuda_config.hpp>
#include <ckt/include/utility.hpp>
#include <ckt/include/for_each.hpp>

using namespace ckt;

template <class T>
class atomic_run_nv_block: public nvstd::function<void(T)> {
public:
  __device__ void operator()(int gid, thrust::tuple<T*, T> &tuple) const {
    atomicAdd(thrust::get<0>(tuple) + gid, thrust::get<1>(tuple));
  }
};


TEST_CASE( "ForEachBlock", "[sum]" ) {
  int  n1 = 5120129;//1200000;
  thrust::host_vector<int> sum(n1);

  for (int i = 0; i != n1; i++)
    sum[i] = 1;

  thrust::device_vector<int> d_sum = sum;
  
  int add_per_thread = 1;
  ForEachKernel<BlockPolicy, konst::cuda_warpsize, false> kernel(n1, "AtomicBlockUt");

  kernel.run<atomic_run_nv_block<int>, int *, int >(thrust::raw_pointer_cast(&d_sum[0]), add_per_thread);
  check_cuda_error("AtomicBlockUt", __FILE__, __LINE__);

  thrust::host_vector<int> h_sum = d_sum;
  for (int i = 0; i != n1; i++ )
  {
    REQUIRE(h_sum[i] == 1+add_per_thread);
  }

}


