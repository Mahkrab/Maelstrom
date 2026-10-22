// Explicit transfer layouts shared by the driver bridge and device solver.
#pragma once
#include <cstdint>
#include <cstddef>

struct Vector { float x, y, z; };
struct DevicePlane { Vector normal; float offset; };
struct Cell { std::int64_t x, y, z; std::uint32_t index, padding; };
struct SolverSettings {
    std::uint32_t count, padded, planes, validate_pairs;
    float timestep, radius, rest_density, relaxation;
    float strength, reference_weight, exponent, correction_scale;
    float mass;
    Vector acceleration;
    double width;
    std::uint32_t stride, span;
    std::uint64_t entries;
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
static_assert(sizeof(SolverSettings) == 88 && offsetof(SolverSettings, width) == 64);
static_assert(sizeof(SolverBuffers) == 128 && offsetof(SolverBuffers,diagnostic) == 112 &&
              offsetof(SolverBuffers,first_operation) == 120);
