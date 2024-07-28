#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include "base/alloc_cpu.h"
#include "base/buffer.h"

TEST(test_buffer, allocate) {
  using namespace base;
  auto alloc = std::make_shared<CPUDeviceAllocator>();
  Buffer buffer(32, alloc);
  ASSERT_NE(buffer.GetPtr(), nullptr);
}

TEST(test_buffer, use_external) {
  using namespace base;
  auto alloc = std::make_shared<CPUDeviceAllocator>();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.IsExternal(), true);
  delete[] ptr;
}

