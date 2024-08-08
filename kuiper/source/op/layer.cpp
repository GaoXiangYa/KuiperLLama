#include "op/layer.h"
#include <glog/logging.h>
#include "base/base.h"
#include "tensor/tensor.hpp"

namespace op {

Layer::Layer(std::string layer_name, base::DeviceType device_type, LayerType layer_type)
    : BaseLayer(layer_name, base::DataType::kDataFp32, device_type, layer_type){}

void Layer::SetOutput(int32_t idx, tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  if (!output.empty()) {
    CHECK(output.GetDeviceType() == device__type_);
  }
  this->outputs_.at(idx) = output;
}

void Layer::SetInput(int32_t idx, tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  if (!input.empty()) {
    CHECK(input.GetDeviceType() == device__type_);
  }
  this->inputs_.at(idx) = input;
}

auto Layer::GetInput(int32_t idx) -> tensor::Tensor {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return this->inputs_.at(idx);
}

auto Layer::GetOutput(int32_t idx) -> tensor::Tensor {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return this->outputs_.at(idx);
}

}  // namespace op