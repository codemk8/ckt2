#pragma once
#include <cxxabi.h>
#include <sstream>
#include <string>
#include "cuda_config.hpp"


namespace ckt {
/* Base class for all cuda kernel classes
 * Do some loggings
 */
class CudaKernel {
  public: 
  explicit CudaKernel(const std::string &tag) {
      std::stringstream tag_stream ;
#ifdef __GNUG__
      int status;
      char * demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
      tag_stream << demangled;
      free(demangled);
#else
      tag_stream << typeid(this).name();
#endif
      tag_stream << ":" << tag;
      set_tag(tag_stream.str());
  }

  const std::string &get_tag() const { return m_tag; }
  void set_tag(const std::string &str) { m_tag = str; }

  void set_block_size(int block_size) { 
    m_block_size = block_size;
    set_auto_tuning(false);
  }

  void set_max_block(int max_block) { 
    m_max_blocks = max_block;
  }

  void set_auto_tuning(bool auto_tune) {
    m_auto_tuning = auto_tune;
  }

  int get_num_blocks() const {
    return m_blocks;
  }
protected:
  std::string m_tag;
  int m_block_size = ckt::konst::default_cuda_blocksize;
  int m_max_blocks = ckt::konst::cuda_max_blocks;
  bool m_auto_tuning = false;
  cudaStream_t m_stream = nullptr;
  int m_blocks = 0;
};
}
