#include "base/alloc.h"
#include "base/base.h"
#include "tensor/tensor.hpp"

namespace base {

void DeviceAllocator::Memcpy(const void* src, void* dest, std::size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  CHECK_NE(src, nullptr);
  CHECK_NE(dest, nullptr);
  if (!byte_size) {
    return;
  }
  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }
  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      cudaMemcpy(dest, src, byte_size, cudaMemcpyHostToDevice);
    } else {
      auto e = cudaGetLastError();
      cudaMemcpyAsync(dest, src, byte_size, cudaMemcpyHostToDevice, stream_);
    }
  }
}

}  // namespace base