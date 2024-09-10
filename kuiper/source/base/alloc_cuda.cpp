#include "base/alloc_cuda.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace base {

void* CUDADeviceAllocator::Allocate(std::size_t byte_size) const {
  int id = -1;
  auto state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);
  // 申请字节数大于1MB
  if (byte_size > ONE_MB) {
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    int big_buffers_size = big_buffers.size();
    for (int i = 0; i < big_buffers_size; ++i) {
      // 遍历big_buffer当中空闲的，满足要求的内存块
      if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
          big_buffers[i].byte_size - byte_size < ONE_MB) {
        if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = i;
        }
      }
    }
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }
    // 如果没有找到空闲，再去调用cudaMalloc
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (state != cudaSuccess) {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! but there's no enough memory left "
               "on device",
               byte_size >> 20);
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }
  // 处理小块显存的情况
  auto& cuda_buffers = cuda_buffers_map_[id];
  int cuda_buffers_size = cuda_buffers.size();
  for (int i = 0; i < cuda_buffers_size; ++i) {
    if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
      return cuda_buffers[i].data;
    }
  }
  // 调用cudaMalloc来分配现存
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byte_size);
  if (state != cudaSuccess) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! but there's no enough memory left "
             "on device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, byte_size, true);
  return ptr;
}

void* CUDADeviceAllocator::Release(void* ptr) const {
  cudaError_t state;
  for (auto& it : cuda_buffers_map_) {
    if (no_busy_cnt_[it.first] > ONE_GB) {
      auto& cuda_buffers = it.second;
      std::vector<CUDAMemoryBuffer> temp;
      int cuda_buffers_size = cuda_buffers.size();
      for (int i = 0; i < cuda_buffers_size; ++i) {
        if (!cuda_buffers[i].busy) {
          state = cudaSetDevice(it.first);
          state = cudaFree(cuda_buffers[i].data);
          CHECK(state == cudaSuccess)
              << "Error: CUDA error when release memory on device" << it.first;
        } else {
          temp.push_back(cuda_buffers[i]);
        }
      }
      cuda_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
    }
  }
  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    int cuda_buffers_size = cuda_buffers.size();
    for (int i = 0; i < cuda_buffers_size; ++i) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        return nullptr;
      }
    }
    auto& big_buffers = big_buffers_map_[it.first];
    int big_buffers_size = big_buffers.size();
    for (int i = 0; i < big_buffers_size; ++i) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
      }
    }
  }
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
  return nullptr;
}

}  // namespace base