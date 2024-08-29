#pragma once

#include <cstddef>
#include <memory>
#include "base/alloc.h"
#include "base/base.h"
namespace base {

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU){}

  virtual ~CPUDeviceAllocator() {}

  virtual void* Allocate(std::size_t size) const override;

  virtual void* Release(void* ptr) const override;

  virtual void Memcpy(const void* src, void* dest, std::size_t byte_size, MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr, bool need_sync = false) const override;

};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

}  // namespace base