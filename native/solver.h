// Explicit transfer layouts shared by the driver bridge and device solver.
#pragma once
#include <cstdint>
#include <cstddef>

struct Vector { float x, y, z; };
struct DevicePlane { Vector normal; float offset; };
struct Cell { std::int64_t x, y, z; std::uint32_t index, padding; };
struct SolverSettings {
    std::uint32_t count, padded, words, planes;
    float timestep, radius, rest_density, relaxation;
    float strength, reference_weight, exponent, correction_scale;
    float mass;
    Vector acceleration;
    double width;
    std::uint32_t stride, span, validate_pairs, reserved;
};
struct SolverBuffers {
    Vector *old_position, *old_velocity, *position, *velocity;
    float *density, *constraint, *multiplier;
    Vector *correction, *candidate;
    DevicePlane *planes;
    Cell *cells;
    std::uint32_t *membership, *errors;
};
static_assert(sizeof(Vector) == 12 && sizeof(DevicePlane) == 16 && sizeof(Cell) == 32);
static_assert(sizeof(SolverSettings) == 88 && offsetof(SolverSettings, width) == 64);
static_assert(sizeof(SolverBuffers) == 104);
