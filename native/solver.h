// Explicit transfer layouts shared by the driver bridge and device solver.
#pragma once
#include <cstdint>
#include <cstddef>

struct Vector { float x, y, z; };
struct DevicePlane { Vector normal; float offset; };
struct Cell { std::int64_t x, y, z; std::uint32_t index, padding; };
// Private Jacobi results occupy candidate scratch until distance validation has
// completed globally. Projection and velocity effects later reuse that storage.
struct PendingJacobi { float density, constraint, multiplier; std::uint32_t error; };
// Each velocity allocation includes an aligned containment result and padding.
// Both buffers keep the trailer when accepted and working vectors are exchanged.
constexpr std::size_t velocity_observation_bytes=16;
// Device records remain the ordering authority. The extra word points at a
// private mapped host mirror, accessed by the device only when an error wins.
constexpr std::size_t diagnostic_record_count=4;
constexpr std::size_t diagnostic_records_bytes=diagnostic_record_count*sizeof(std::uint64_t);
constexpr std::size_t diagnostic_allocation_bytes=diagnostic_records_bytes+sizeof(std::uint64_t);
struct SolverSettings {
    std::uint32_t count, padded, planes, validate_pairs;
    float timestep, radius, rest_density, relaxation;
    float strength, reference_weight, exponent, correction_scale;
    float mass;
    Vector acceleration;
    double width;
    std::uint32_t stride, span;
    std::uint64_t entries;
    float xsph, vorticity;
};
struct SolverBuffers {
    Vector *old_position, *old_velocity, *position, *velocity;
    float *density, *constraint, *multiplier;
    Vector *correction, *candidate;
    DevicePlane *planes;
    Cell *cells;
    std::uint32_t *neighbours, *errors;
    std::uint64_t *offsets;
    unsigned long long *diagnostic;
    // Zero for a single pass; otherwise each pass owns one ordered result slot.
    std::uint32_t first_operation;
};
static_assert(sizeof(Vector) == 12 && sizeof(DevicePlane) == 16 && sizeof(Cell) == 32);
static_assert(sizeof(PendingJacobi) == 16);
static_assert(sizeof(SolverSettings) == 96 && offsetof(SolverSettings, width) == 64);
static_assert(sizeof(SolverBuffers) == 128 && offsetof(SolverBuffers,diagnostic) == 112 &&
              offsetof(SolverBuffers,first_operation) == 120);
