#ifndef CKT_POLICY_HPP
#define CKT_POLICY_HPP


namespace ckt {

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 300
#error  // only support kepler above card
#endif
#endif

  template <int groupsize /* no use */, bool need_sync>
  class StridePolicy { // : public ForEachPolicy<groupsize> {
public:
    __device__ void init(const int & _N, const int &my_id, const int &max_id, const bool &_need_sync, 
                         /* outputs */
                         int &n_batches, int &id) {
      // strided id is equivalent to the global id
      id = my_id;
      if (!_need_sync) {
        n_batches = (_N/max_id);
        n_batches += my_id < (_N - n_batches * max_id)? 1:0 ;
      } else {
        n_batches = (_N+max_id-1)/max_id;
      }
    }
    __device__ __forceinline__ int stride() const {
      return blockDim.x * gridDim.x;
    }
    __device__  __forceinline__ void next_batch(int &batches, int &my_id) {
     --batches;
     my_id += stride();
   }
    __device__ __forceinline__ bool is_active(const int &my_id, const int &N)  {
      return my_id < N;
    }
};

  template <int groupsize/* = ckt::cuda::CKT_cuda_blocksize*/, bool need_sync>
  class BlockPolicy { // :public ForEachPolicy{
public:
    __device__ void init(const int & _N, const int &my_id, const int &max_id, const bool &_need_sync, int &n_batches, int &id) {
      n_batches = num_batches(_N, my_id, max_id, _need_sync, id);
    }

    __device__ __forceinline__ int stride() const {
      return groupsize;
    }

    virtual __device__ int num_batches(int _N, int my_id, int max_id, bool _need_sync, int &id) {
    assert((max_id % groupsize) == 0);
    int num_groups = max_id /groupsize;
    int group_id = my_id/groupsize;
    int lane_id = my_id %groupsize;
    int total_batches = (_N+groupsize-1)/groupsize;
    int batch_per_group = (total_batches + num_groups - 1)/num_groups;
    int batch_start = batch_per_group * group_id;
    batch_start = batch_start > total_batches? total_batches : batch_start;
    int batch_end = batch_start + batch_per_group;
    batch_end = batch_end > total_batches? total_batches : batch_end;
    int num_batch = batch_end - batch_start;

    if (batch_end == total_batches) {
      m_last_index = _N;
    } else {
      m_last_index = batch_end * groupsize;
    }
    // update id for block 
    id = batch_start * groupsize + lane_id;
    return num_batch;
  }

  __device__  __forceinline__ void next_batch(int &batches, int &my_id) {
    --batches;
    my_id += stride();
  }

  __device__  __forceinline__ bool is_active(const int &id, const int &N) {
    return id < m_last_index;
  }

  private: 
    int m_last_index;

};

  template <template<int, bool> class Policy, int group_size, bool need_sync>
class ForEach {
public:
  __device__ ForEach(int32_t _N, int32_t my_id, int32_t max_id):
    m_id(my_id){
    
    //    Policy<group_size> policy;
    policy.init(_N, my_id, max_id, need_sync, m_batches, m_id);

    //    int lane_id = my_id % group_size;
  }

  __device__ __forceinline__ int num_batches() {
    //    Policy<group_size
    return m_batches;
  }
  
  __device__ __forceinline__ bool not_done() const {
    return m_batches > 0;
  }

  __device__ __forceinline__ bool not_last_batch() const {
    return m_batches > 1;
  }

  __device__ __forceinline__ bool is_active(int N) {
    return policy.is_active(m_id, N);
  }

  __device__ __forceinline__ int get_id() const {
    return m_id;
  }

    __device__ __forceinline__ int get_stride() const {
      return policy.stride();
    }
  __device__ __forceinline__  void next_batch() {
    policy.next_batch(m_batches, m_id);
    //    Policy<group_size> policy;
    //   --m_batches;
    // m_id += m_stride;
    // m_is_active = (m_id < N);

  //    policy.next_batch(m_batches, m_id, m_stride, m_is_active, N);
   }
  
private:
  Policy<group_size, need_sync> policy;
  int m_id = 0;
  //  int m_group_size = 0;
  int m_batches = 0;
  //  bool m_is_active = false;
};


}

#endif