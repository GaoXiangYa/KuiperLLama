#pragma once

#include <glog/types.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.hpp"
namespace op {

enum class LayerType { kLayerUnknown = 0 };

// 所有算子的抽象
class BaseLayer {
 public:
  explicit BaseLayer(std::string layer_name, base::DataType data_type, base::DeviceType device_type,
                     LayerType layer_type)
      : layer_name_(layer_name),
        data_type_(data_type),
        device__type_(device_type),
        layer_type_(layer_type) {}

  virtual ~BaseLayer() {}

  virtual void SetInput(int32_t idx, tensor::Tensor& input) = 0;

  virtual void SetOutput(int32_t idx, tensor::Tensor& output) = 0;

  virtual auto GetInput(int32_t idx) -> tensor::Tensor = 0;

  virtual auto GetOutput(int32_t idx) -> tensor::Tensor = 0;

  virtual auto GetInputSize() -> size_t = 0;

  virtual auto GetOutputSize() -> size_t = 0;

  auto GetDataType() -> base::DataType { return data_type_; }

  auto GetDeviceType() -> base::DeviceType { return device__type_; }

  auto GetLayerType() -> LayerType { return layer_type_; }

  void SetDataType(base::DataType dType) { data_type_ = dType; }

  auto GetLayerName() -> std::string { return layer_name_; }

 protected:
  std::string layer_name_;                                  // 层名
  base::DataType data_type_{base::DataType::kDataUnknown};  // 层的类型
  base::DeviceType device__type_{base::DeviceType::kDeviceUnknown};
  LayerType layer_type_{LayerType::kLayerUnknown};
};

// 不带参数的算子设计，只需要处理输入和输出
class Layer : public BaseLayer {
 public:
  explicit Layer(std::string layer_name, base::DeviceType device_type, LayerType layer_type);

  virtual void SetInput(int32_t idx, tensor::Tensor& input) override;

  virtual void SetOutput(int32_t idx, tensor::Tensor& output) override;

  virtual auto GetInput(int32_t idx) -> tensor::Tensor override;

  virtual auto GetOutput(int32_t idx) -> tensor::Tensor override;

  virtual auto GetInputSize() -> size_t override;

  virtual auto GetOutputSize() -> size_t override;

  void RestInputSize(size_t size);

  void RestOutputSize(size_t size);

 protected:
  std::vector<tensor::Tensor> inputs_;
  std::vector<tensor::Tensor> outputs_;
};

// 带参数的算子类设计，需要处理权重比如matmul算子
class LayerFp32Param : public Layer {
 public:
  explicit LayerFp32Param(std::string layer_name, base::DeviceType device_type,
                          LayerType layer_type);

  auto WeightSize() -> size_t { return weights_.size(); }

  void SetWeight(int32_t idx, const tensor::Tensor& tensor);

  void SetWeight(int32_t idx, const std::vector<int32_t>& dims, const float* weight_ptr,
                 base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

  void ResetWeightSize(size_t size);

  auto GetWeight(int32_t idx) -> tensor::Tensor;

 private:
  std::vector<tensor::Tensor> weights_;
};

}  // namespace op