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
      : size_(size), allocator_(allocator), ptr_(ptr), use_external_(use_external) {
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

  auto IsExternal() -> bool const { return use_external_; }

  auto GetPtr() -> void* const { return ptr_; }

 private:
  DeviceType device_type_{DeviceType::kDeviceUnknown};
  bool use_external_{false};
  std::size_t size_;
  void* ptr_;
  std::shared_ptr<DeviceAllocator> allocator_;
};

}  // namespace base