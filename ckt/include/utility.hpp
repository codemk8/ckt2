#pragma once

#include <iostream>
#include <cstdio>
#include <cuda_runtime_api.h>

namespace ckt {

  inline void ckt_assert_fail (const char *expr, const char *file, int line, const char *func)
  {
    fprintf(stderr, "Assertion %s failed:  File %s , line %d, function %s\n",
                   expr, file, line, func);
  }

#ifdef WIN32
#define __Ckt_FUNC_ __FUNCTION__
#else
#define __Ckt_FUNC_ __PRETTY_FUNCTION__
#endif

  inline static void check_cuda_error_always(const char *kernelname, const char *file, int line_no, cudaStream_t stream = 0)
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error @ kernel " << kernelname << " @ file " << file << " line " << line_no << \
          " for reason: "  << cudaGetErrorString(err) << std::endl;
    }
    if (stream)
      err = cudaStreamSynchronize(stream);
    else
      err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error synchronizing device @ kernel " << kernelname << " @ file " << file << " line " << line_no << \
          " for reason: "  << cudaGetErrorString(err) << std::endl;
    }
  }

  inline static void check_cuda_error(const char *kernelname, const char *file, int line_no, cudaStream_t stream = 0)
  {
#ifdef DEBUG
    check_cuda_error_always(kernelname, file, line_no, stream);
#endif
  }

  //double ckt_get_wtime()

  // void print_stacktrace(FILE *out = stderr, unsigned int max_frames = 63);
}

#define cassert(expression)       \
  assert(expression); \
  ((expression) ? (void) 0  \
   : ckt::ckt_assert_fail(#expression, __FILE__, __LINE__, __Ckt_FUNC_))


