#include "tensor/tensor.hpp"
#include <glog/logging.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>
#include "base/alloc.h"
#include "base/base.h"
#include "base/buffer.h"

namespace tensor {

Tensor::Tensor(base::DataType data_type, std::initializer_list<std::int32_t> init_dim,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type), dims_(init_dim){
  size_ = std::accumulate(init_dim.begin(), init_dim.end(), 1, [&](int a, int b) { return a * b; });
  // tensor 使用自己的alloctor 来给自己分配内存
  if (need_alloc && alloc) {
    // Todo : allocate tensor
    // buffer_->Allocate(std::size_t size)
    Allocate(alloc, need_alloc);
  } else {
    // 使用已有的资源来给tensor分配内存
    is_empty_ = true;
    if (ptr != nullptr) {
      CHECK(need_alloc == false)
          << "The need_alloc is true when ptr parameter is not a null pointer.";
      InitBuffer(alloc);
    }
  }
}

auto Tensor::GetDim(std::int32_t idx) const -> std::int32_t {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

auto Tensor::Strides() const -> std::vector<std::size_t> {
  std::vector<std::size_t> strides;
  if (!dims_.empty()) {
    for (auto ite = dims_.begin(); ite != dims_.end() - 1; ++ ite) {
      std::size_t stride = std::accumulate(ite + 1, dims_.end(), 1, std::multiplies<>());
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}

void Tensor::InitBuffer(std::shared_ptr<base::DeviceAllocator> alloc, void* ptr) {
  if (alloc == nullptr) {
    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(GetByteSize(), nullptr, ptr, true);
    this->buffer_ = buffer;
  } else {
    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(GetByteSize(), alloc, ptr, false);
  }
}

auto Tensor::GetDataTypeSize(base::DataType data_type) -> std::size_t {
  switch (data_type) {
    case base::DataType::kDataTensor:
      break;
    case base::DataType::kDataTypeFp32:
      return 4;
    case base::DataType::kDataTypeInt8:
      return 1;
    case base::DataType::kDataTypeInt32:
      return 4;
    case base::DataType::kDataUnknown:
      break;
  }
  return 0;
}

auto Tensor::Allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_alloc) -> bool {
  if (allocator == nullptr) {
    return false;
  }
  auto byte_size = GetByteSize();
  if (!byte_size) {
    return false;
  }
  if (buffer_ && byte_size <= buffer_->GetBufferSize() && !need_alloc) {
    return true;
  }
  buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
  if (buffer_->GetPtr() == nullptr) {
    return false;
  }
  return true;
}

}  // namespace tensor