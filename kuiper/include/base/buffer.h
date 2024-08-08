#pragma once

#include <cstddef>
#include <memory>
#include "base/alloc.h"
#include "base/base.h"
namespace base {

class Buffer : NoCopyable, std::enable_shared_from_this<Buffer> {
 public:
  explicit Buffer(std::size_t size, const std::shared_ptr<DeviceAllocator>& allocator, void* ptr = nullptr,
                  bool use_external = false)
      : use_external_(use_external), size_(size), ptr_(ptr), allocator_(allocator) {
    if (ptr_ == nullptr && allocator_ != nullptr) {
      device_type_ = allocator_->getDeviceType();
      ptr_ = allocator_->Allocate(size_);
      use_external_ = false;
    }
  }

  ~Buffer() {
    if(!use_external_) {
      allocator_->Release(ptr_);
    }
  }

  auto Allocate(std::size_t size) -> bool; 

  auto IsExternal() -> bool { return use_external_; }

  auto GetPtr() -> void* { return ptr_; }

  auto GetBufferSize() -> std::size_t { return size_; }

 private:
  DeviceType device_type_{DeviceType::kDeviceUnknown};
  bool use_external_{false};
  std::size_t size_;
  void* ptr_;
  std::shared_ptr<DeviceAllocator> allocator_;
};

}  // namespace base