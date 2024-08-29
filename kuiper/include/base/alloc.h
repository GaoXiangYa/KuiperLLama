#pragma once

#include <cstddef>
#include "base/base.h"

namespace base {

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

  virtual ~DeviceAllocator() {}

  virtual auto getDeviceType() -> DeviceType { return device_type_; }

  virtual void* Allocate(std::size_t size) const = 0;

  virtual void* Release(void* ptr) const = 0;

  virtual void Memcpy(const void* src, void* dest, std::size_t byte_size, MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr, bool need_sync = false) const;

  virtual void MemsetZero(void* ptr, std::size_t byte_size, void* stream, bool need_sync = false);
 private:
  DeviceType device_type_{DeviceType::kDeviceUnknown};
};

}  // namespace base