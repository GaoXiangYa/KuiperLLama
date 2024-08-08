#pragma once

namespace base {

enum class DeviceType { kDeviceCPU = 0, kDeviceCUDA, kDeviceOPENCL, kDeviceUnknown };

enum class DataType { kDataTensor = 0, kDataTypeFp32, kDataTypeInt8, kDataTypeInt32, kDataUnknown};

class NoCopyable {
public:
  NoCopyable() = default;
  NoCopyable(const NoCopyable &rhs) = delete;
  NoCopyable& operator=(const NoCopyable &rhs) = delete;
};
}  // namespace base