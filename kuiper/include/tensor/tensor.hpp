#pragma once

#include <glog/logging.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <vector>
#include "base/alloc.h"
#include "base/base.h"
#include "base/buffer.h"

namespace tensor {

class Tensor {
 public:
  explicit Tensor(base::DataType data_type, std::initializer_list<std::int32_t> init_dim,
                  bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);
  explicit Tensor(base::DataType data_type, std::size_t size, bool need_alloc,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr);
  explicit Tensor(base::DataType data_type, std::size_t size, bool need_alloc,
                  std::shared_ptr<base::DeviceAllocator> alloc, void* ptr);
  auto Size() -> std::size_t { return size_; }
  auto Isempty() -> bool;
  auto GetDataType() -> base::DataType { return data_type_; }
  auto GetDeviceType() -> base::DeviceType {
    if (buffer_ == nullptr) {
      return base::DeviceType::kDeviceUnknown;
    }
    return buffer_->GetDeviceType();
  }

  template <typename T>
  T* Ptr();

  template <typename T>
  T* Ptr(std::int64_t index);

  template <typename T>
  const T& index(std::int64_t offset);

  auto GetDim(std::int32_t idx) const -> std::int32_t;

  auto Strides() const -> std::vector<std::size_t>;

 private:
  auto GetDataTypeSize(base::DataType data_type) -> std::size_t;
  auto GetByteSize() -> std::size_t { return GetDataTypeSize(data_type_) * size_; }
  auto Allocate(std::shared_ptr<base::DeviceAllocator> alloc, bool need_alloc = false) -> bool;
  void InitBuffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type, void* ptr = nullptr, bool need_alloc = false);

 private:
  base::DataType data_type_{base::DataType::kDataUnknown};
  std::size_t size_{0};
  std::vector<std::int32_t> dims_;
  std::shared_ptr<base::Buffer> buffer_;
};

template <typename T>
T* Tensor::Ptr() {
  return buffer_ == nullptr ? nullptr : reinterpret_cast<T*>(buffer_->GetPtr());
}

template <typename T>
T* Tensor::Ptr(std::int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->GetPtr() != nullptr);
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->GetPtr())) + index;
}

template <typename T>
const T& Tensor::index(std::int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->Size());
  const T& val = *(reinterpret_cast<T*>(buffer_->GetPtr()) + offset);
  return val;
}

}  // namespace tensor