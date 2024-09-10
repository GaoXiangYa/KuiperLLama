#pragma once

#include <map>
#include <cstddef>
#include <memory>
#include <vector>
#include "base/alloc.h"

namespace base {

#define ONE_MB 1024 * 1024
#define ONE_GB 1024 * 1024 * 1024
struct CUDAMemoryBuffer {
  void* data;
  std::size_t byte_size;
  bool busy;
  CUDAMemoryBuffer() = default;
  CUDAMemoryBuffer(void* data, std::size_t byte_size, bool busy) : data(data), byte_size(byte_size), busy(busy){}
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
  explicit CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA){}

  virtual ~CUDADeviceAllocator() {}

  virtual void* Allocate(std::size_t size) const override;

  virtual void* Release(void* ptr) const override;

  virtual void Memcpy(const void* src, void* dest, std::size_t byte_size, MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr, bool need_sync = false) const override;
private:
  mutable std::map<int, std::size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<CUDAMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CUDAMemoryBuffer>> cuda_buffers_map_; 
};

class CUDADeviceAllocatorFactory {
public:
  static auto GetInstance() -> std::shared_ptr<CUDADeviceAllocator> {
    if (instance == nullptr) {
      return std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }
private:
  inline static std::shared_ptr<CUDADeviceAllocator> instance = nullptr;
};

}   // namespace base