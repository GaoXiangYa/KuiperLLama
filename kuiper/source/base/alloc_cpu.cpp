#include "base/alloc_cpu.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace base {

void* CPUDeviceAllocator::Allocate(std::size_t size) const {
  if (size == 0) {
    return nullptr;
  }
  void* ptr = std::malloc(size);
  return ptr;
}

void* CPUDeviceAllocator::Release(void* ptr) const {
  if (ptr == nullptr) {
    std::cerr << "Release nullptr\n";
  }
  std::free(ptr);
}

void* CPUDeviceAllocator::Memcpy(void* src, void* dest, std::size_t size) const {
  if (src == nullptr) {
    std::cerr << "Memcpy src is nullptr\n";
  }
  std::memcpy(dest, src, size);
}

} // namespace base