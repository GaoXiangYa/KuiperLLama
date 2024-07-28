#pragma once

#include <cstddef>
#include "base/base.h"

namespace base {

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

  virtual auto getDeviceType() -> const DeviceType { return device_type_; }

  virtual void* Allocate(std::size_t size) const = 0;

  virtual void* Memcpy(void* src, void* dest, std::size_t size) const = 0;

  virtual void* Release(void* ptr) const = 0;

 private:
  DeviceType device_type_{DeviceType::kDeviceUnknown};
};

}  // namespace base