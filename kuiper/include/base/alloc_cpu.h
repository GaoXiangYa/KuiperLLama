#pragma once

#include <cstddef>
#include "base/alloc.h"
namespace base {

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  virtual void* Allocate(std::size_t size) const override;

  virtual void* Release(void* ptr) const override;

  virtual void* Memcpy(void* src, void* dest, std::size_t size) const override;

};

}  // namespace base