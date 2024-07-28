#include "base/buffer.h"
#include <cstddef>

namespace base {

auto Buffer::Allocate(std::size_t size) -> bool {
  if (allocator_ != nullptr) {
    ptr_ = allocator_->Allocate(size);
  }
  return ptr_ != nullptr;
}

} // namespace base