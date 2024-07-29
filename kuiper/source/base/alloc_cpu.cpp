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
  if (ptr) {
    std::free(ptr);
  }
  return nullptr;
}

void* CPUDeviceAllocator::Memcpy(void* src, void* dest, std::size_t size) const {
  if (src == nullptr) {
    std::cerr << "Memcpy src is nullptr\n";
  }
  std::memcpy(dest, src, size);
  return nullptr;
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;

} // namespace base