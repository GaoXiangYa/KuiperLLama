#pragma once

namespace base {

enum class DeviceType { kDeviceCPU = 0, kDeviceCUDA, kDeviceOPENCL, kDeviceUnknown };

class NoCopyable {
public:
  NoCopyable() = default;
  NoCopyable(const NoCopyable &rhs) = delete;
  NoCopyable& operator=(const NoCopyable &rhs) = delete;
};
}  // namespace base