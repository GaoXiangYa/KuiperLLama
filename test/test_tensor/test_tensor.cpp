#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include "base/alloc_cpu.h"
#include "base/base.h"
#include "tensor/tensor.hpp"

TEST(test_tensor, init1) {
  using namespace base;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  std::int32_t size = 32 * 151;
  tensor::Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cpu);
  ASSERT_EQ(t1.Isempty(), false);
}

TEST(test_tensor, init2) {
  using namespace base;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  int32_t size = 32 * 151;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, false, alloc_cpu);
  ASSERT_EQ(t1.Isempty(), true);
}

TEST(test_tensor, init3) {
  using namespace base;
  auto* ptr = new float[32];
  ptr[0] = 31;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, nullptr, ptr);
  ASSERT_EQ(t1.Isempty(), false);
  ASSERT_EQ(t1.Ptr<float>(), ptr);
  ASSERT_EQ(*t1.Ptr<float>(), 31);
  delete [] ptr;
}

TEST(test_tensor, dims_stride) {
  using namespace base;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1(base::DataType::kDataTypeFp32, {32, 32, 3}, true, alloc_cpu);
  ASSERT_EQ(t1.Isempty(), false);
  ASSERT_EQ(t1.GetDim(0), 32);
  ASSERT_EQ(t1.GetDim(1), 32);
  ASSERT_EQ(t1.GetDim(2), 3);

  const auto& strides = t1.Strides();
  ASSERT_EQ(strides.at(0), 32 * 3);
  ASSERT_EQ(strides.at(1), 3);
  ASSERT_EQ(strides.at(2), 1);
}
