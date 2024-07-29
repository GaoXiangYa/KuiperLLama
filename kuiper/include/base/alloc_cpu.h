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

  virtual void* Memcpy(void* src, void* dest, std::size_t size) const override;

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