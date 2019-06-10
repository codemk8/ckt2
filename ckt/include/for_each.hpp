#pragma once

 #include <thrust/tuple.h>
#include <nvfunctional>
#include <typeinfo>
#include "kernel.hpp"
#include "policy.hpp"

namespace ckt {
  /*!
   * The for_each kernel template for simple kernels that do not use shared memory
   */
  template <template <bool> class Policy, int groupsize, int need_sync, class Fn, class... Args>
  static __global__ void for_each_kernel(int N, Args... args)
  {
    int max_id = blockDim.x * gridDim.x;
    int id = threadIdx.x + blockDim.x*blockIdx.x; 
    ForEach<Policy, groupsize, need_sync> fe(N, id, max_id);
    int stride = fe.get_stride();

    thrust::tuple<Args...> tuple(args...);
    int batches = fe.num_batches();

    Fn _operator; 

    while (fe.not_last_batch()) {
      _operator(fe.get_id(), tuple);
      fe.next_batch();
    }

    while (fe.not_done()) {
      if (fe.is_active(N))
      {
        _operator(fe.get_id(), tuple);
      }
      fe.next_batch();
    }
  }

  /*!
   * A template for each kernel class that does not use shared memorya
   */ 
  template <template<bool> class Policy, int group_size, bool need_sync>
  class ForEachKernel: public CudaKernel {
  public: 
    ForEachKernel(int32_t _N, const std::string &tag): m_N(_N), CudaKernel(tag) {
    }

    template <class Method, class... Args>
    void run(Args... args) {
      if (m_N == 0) return;
      int blocks = ckt::cuda::get_blocks(m_N, m_block_size);
      int BS = m_block_size; //ckt::cuda::konst::cuda_blocksize;
      if (m_auto_tuning) {
        //must be equal to or above cuda 6.5 
        int min_gridsize, blocksize;
        cudaOccupancyMaxPotentialBlockSize(&min_gridsize, &blocksize, for_each_kernel<Policy, group_size, need_sync, Method, Args...>,
                                           0, m_N);
        // sometimes CUDA API returns 0
        if (blocksize == 0) 
          blocksize = m_block_size;
        BS = blocksize;
        blocks = ckt::cuda::get_blocks(m_N, blocksize);
        blocks = std::min(blocks, 8 * min_gridsize);
      }
      blocks = std::min(blocks, m_max_blocks);
      for_each_kernel<Policy, group_size, need_sync, Method, Args...><<<blocks, BS>>>(m_N, args...);
    }
      
    void set_N(int32_t _N) {
      m_N = _N;
    }
  private:
    int m_N{0};
  };
}
