// The probe deliberately crosses the boundary with both identity and binary32
// coordinates. No Rust simulator struct is reinterpreted as device storage.
#pragma once
#include <cstddef>
#include <cstdint>

struct Probe {
    std::uint64_t identity;
    float position[3];
    std::uint32_t marker;
};
static_assert(sizeof(Probe) == 24 && alignof(Probe) == 8);
static_assert(offsetof(Probe, position) == 8 && offsetof(Probe, marker) == 20);

struct Status { int code; int stage; };
struct DeviceInfo {
    char name[128];
    std::uint64_t memory_bytes;
    int major;
    int minor;
    int driver_version;
};
static_assert(sizeof(Status) == 8 && sizeof(DeviceInfo) == 152);
