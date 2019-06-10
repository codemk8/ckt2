#include "catch.hpp"
#include <ckt/include/utility.hpp>
#include <ckt/include/cuda_config.hpp>
#include <ckt/include/for_each_shm.hpp>
#include <ckt/include/for_each.hpp>


using namespace ckt;

template <class T>
class reduce_run_nv: public nvstd::function<void(T)> {
public:
  __device__ reduce_run_nv() {
    m_reduce = {};
  }
  __device__ void operator()(int gid, thrust::tuple<const T*, T*> &tuple)  {
    m_reduce += (thrust::get<0>(tuple))[gid];
    //    atomicAdd(thrust::get<0>(tuple), thrust::get<1>(tuple));
  }

  __device__ void post_proc(int gid, thrust::tuple<const T*, T*> &tuple)  {
    //    printf("here in post proc\n");
    // if (blockIdx.x == 0 & threadIdx.x < 2)
    //   printf("my reduce %d tid %d.\n", m_reduce, threadIdx.x);
    m_reduce = ckt::cuda::blockReduceSum(m_reduce);
    if (threadIdx.x == 0)
      (thrust::get<1>(tuple))[blockIdx.x] = m_reduce;

    // if (blockIdx.x == 0 & threadIdx.x < 2)
    //   printf("after my reduce %d tid %d.\n", m_reduce, threadIdx.x);

  } 
private:
  T m_reduce;
};



TEST_CASE( "ForEachShmReduce", "[sum]" ) {
   int n = 2000;
   JVector<int> sum(n);
   thrust::fill(sum.gbegin(), sum.gend(), 1);
   //  ForEachKernel<StridePolicy, 256, false> fe(300);
   //  AtomicAdd kernel(300);
   //  AtomicAdd/*<decltype(atomic_run)>*/ kernel(/*atomic_run,*/ n1);
   ForEachShmKernel<BlockPolicy, konst::default_cuda_blocksize, false> kernel(n, "Reduction");
   kernel.set_block_size(1024);
   kernel.set_max_block(1024);
   JVector<int> inter_sum(1024);
   constexpr int shared_bsize = sizeof(int)*1024/32;
   
   kernel.run<reduce_run_nv<int>, int, shared_bsize, const int *, int *>(sum.getROGpuPtr(), inter_sum.getGpuPtr());
   //   inter_sum.print("intersum");
   kernel.set_N(kernel.get_num_blocks());
   kernel.run<reduce_run_nv<int>, int, shared_bsize, const int *, int *>(inter_sum.getROGpuPtr(), inter_sum.getGpuPtr());
   cudaDeviceSynchronize();
   //   inter_sum.print("intersum after");
   check_cuda_error("inter_sum", __FILE__, __LINE__);
   
   REQUIRE(inter_sum.getElementAt(0) == n);
   // int sum_now = sum[0];
   // REQUIRE(sum_now == n1*add_per_thread);
   
   // kernel.set_N(257);
   // kernel.run<atomic_run_nv<int>, int *, int >(sum.getGpuPtr(), 12);
   // REQUIRE(sum[0] == (sum_now + 257*12));

   check_cuda_error("inter_sum", __FILE__, __LINE__);
}

TEST_CASE( "ForEachShmReduceDouble", "[sum]" ) {
   int n = 2000;
   JVector<double> sum(n);
   thrust::fill(sum.gbegin(), sum.gend(), 1);
   //  ForEachKernel<StridePolicy, 256, false> fe(300);
   //  AtomicAdd kernel(300);
   //  AtomicAdd/*<decltype(atomic_run)>*/ kernel(/*atomic_run,*/ n1);
   ForEachShmKernel<BlockPolicy, CKT_cuda_warpsize, false> kernel(n, "Reduction");
   kernel.set_block_size(1024);
   kernel.set_max_block(1024);
   JVector<double> inter_sum(1024);
   constexpr int shared_bsize = sizeof(double)*1024/32;
   
   kernel.run<reduce_run_nv<double>, double, shared_bsize, const double *, double *>(sum.getROGpuPtr(), inter_sum.getGpuPtr());
   //   inter_sum.print("intersum");
   kernel.set_N(kernel.get_num_blocks());
   kernel.run<reduce_run_nv<double>, double, shared_bsize, const double *, double *>(inter_sum.getROGpuPtr(), inter_sum.getGpuPtr());
   cudaDeviceSynchronize();
   //   inter_sum.print("intersum after");
   check_cuda_error("inter_sum", __FILE__, __LINE__);
   
   REQUIRE(inter_sum.getElementAt(0) == n);
   // int sum_now = sum[0];
   // REQUIRE(sum_now == n1*add_per_thread);
   
   // kernel.set_N(257);
   // kernel.run<atomic_run_nv<int>, int *, int >(sum.getGpuPtr(), 12);
   // REQUIRE(sum[0] == (sum_now + 257*12));

   check_cuda_error("inter_sum", __FILE__, __LINE__);
}



