#pragma once
#include <cuda_runtime_api.h>

namespace ckt 
{
  namespace konst 
  {
    const int32_t default_cuda_blocksize = 512;
    const int32_t cuda_max_blocks = 1024;
    const int32_t cuda_warpsize = 32;
    const int32_t cuda_warpsize_shift = 5;
    const int32_t cuda_warpsize_mask = 0x1F;
  };
  namespace cuda 
  {
    __inline__ void get_cuda_property(cudaDeviceProp &property) {
      int gpu; cudaGetDevice(&gpu);
      cudaGetDeviceProperties(&property, gpu);
    }

    static int32_t cap_block_num(const int32_t blocks)
    {
      return blocks >  ckt::konst::cuda_max_blocks? ckt::konst::cuda_max_blocks : blocks;
    }    

    static int32_t get_blocks(const int32_t N, const int32_t bs) 
    {
      auto block_num = (N + bs - 1)/bs;
      return cap_block_num(block_num);
    }
  }
  
}

// #define CAP_BLOCK_SIZE(block) (block > ckt::konst::cuda_max_blocks ? ckt::konst::cuda_max_blocks:block)
//#define GET_BLOCKS(N, bs) CAP_BLOCK_SIZE( (N + bs -1 )/bs)




 


