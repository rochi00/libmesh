#ifndef KOKKOS_NUMERICS_ORACLE_TEST_UTILS_H
#define KOKKOS_NUMERICS_ORACLE_TEST_UTILS_H

// Shared plumbing for the Kokkos numerics oracle tests: uploading test
// data to the device and comparing device-computed results against
// host-computed references.  The comparison helpers mirror device data
// back to the host and return a count of components outside tolerance
// (rather than asserting), so the runners can aggregate failures across
// suites and report them all in one run.

#include "libmesh/libmesh.h"
#include "libmesh/int_range.h"
#include "libmesh/point.h"

// Avoid conflicting complex operators between CUDA and PETSc
#define PETSC_SKIP_CXX_COMPLEX_FIX 1
#include "libmesh/ignore_warnings.h"
#include <Kokkos_Core.hpp>
#include "libmesh/restore_warnings.h"
#undef __CUDACC_VER__

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace libMeshTest
{
namespace KokkosOracle
{

using libMesh::Real;

// Copies a host vector of plain objects into a new device view, for
// test inputs that don't go through the gpu/ storage-policy views
// (e.g. the per-case leading dims in the tensor suite).
template <typename T>
inline ::Kokkos::View<T *>
upload_objects(const std::vector<T> & values, const char * label)
{
  ::Kokkos::View<T *> d(std::string(label), values.size());
  auto h = ::Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < values.size(); ++i)
    h(i) = values[i];
  ::Kokkos::deep_copy(d, h);
  return d;
}

// Whether a and b have the same bits, so that a kernel reproducing a
// host NaN matches it too
inline bool
same_bits(const libMesh::Point & a, const libMesh::Point & b)
{
  for (const auto d : libMesh::make_range(LIBMESH_DIM))
    {
      const Real x = a(d), y = b(d);
      if (std::memcmp(&x, &y, sizeof(Real)))
        return false;
    }
  return true;
}

// Counts the values a kernel computed and those that differ from the
// host's, printing the first few that do.  A value matches if it has
// the host's bits, so that a kernel reproducing a host NaN matches it,
// or is within tol of the host's, relative to 1 + |host|: a device may
// contract the arithmetic differently from the host compiler.
struct ValueCheck
{
  // Prefixes each printed mismatch, as the oracle's name does its
  // summary
  const char * oracle;

  // The relative tolerance
  Real tol;

  // The values checked so far, and those of them that mismatched
  int n_values = 0, n_fail = 0;

  // describe() says which value of which case mismatched; it is called
  // only to print a mismatch, and only for the first 10, so a broken
  // kernel doesn't flood the output.
  template <typename Describe>
  void operator() (const Real host, const Real kernel, const Describe & describe);
};

template <typename Describe>
inline void
ValueCheck::operator() (const Real host, const Real kernel, const Describe & describe)
{
  ++n_values;
  if (!std::memcmp(&host, &kernel, sizeof(Real)) ||
      std::abs(kernel - host) <= tol * (1 + std::abs(host)))
    return;
  if (n_fail < 10)
    std::printf("[%s] mismatch: %s: host %.17g kernel %.17g\n",
                oracle, describe().c_str(), double(host), double(kernel));
  ++n_fail;
}

// The compare_device_* helpers below deep-copy the device results to a
// host mirror and count individual components that differ from the
// reference by more than tol; boolean results are encoded as 1.0/0.0
// by the kernels so a single tolerance comparison covers
// everything.

inline int
compare_device_scalars(const ::Kokkos::View<double *> & d_values,
                       const std::vector<Real> & ref_values,
                       const double tol)
{
  auto h_values = ::Kokkos::create_mirror_view(d_values);
  ::Kokkos::deep_copy(h_values, d_values);

  int fail = 0;
  for (std::size_t i = 0; i < ref_values.size(); ++i)
    if (std::fabs(h_values(i) - ref_values[i]) > tol)
      ++fail;

  return fail;
}

template <typename ViewType, typename VectorType>
inline int
compare_device_vectors(const ViewType & d_values,
                       const std::vector<VectorType> & ref_values,
                       const double tol)
{
  auto h_values = ::Kokkos::create_mirror_view(d_values);
  ::Kokkos::deep_copy(h_values, d_values);

  int fail = 0;
  for (std::size_t i = 0; i < ref_values.size(); ++i)
    for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
      if (std::fabs(h_values(i, d) - ref_values[i](d)) > tol)
        ++fail;

  return fail;
}

template <typename ViewType, typename TensorType>
inline int
compare_device_tensors(const ViewType & d_values,
                       const std::vector<TensorType> & ref_values,
                       const double tol)
{
  auto h_values = ::Kokkos::create_mirror_view(d_values);
  ::Kokkos::deep_copy(h_values, d_values);

  int fail = 0;
  for (std::size_t i = 0; i < ref_values.size(); ++i)
    for (unsigned int row = 0; row < LIBMESH_DIM; ++row)
      for (unsigned int col = 0; col < LIBMESH_DIM; ++col)
        if (std::fabs(h_values(i, row, col) - ref_values[i](row, col)) > tol)
          ++fail;

  return fail;
}

} // namespace KokkosOracle
} // namespace libMeshTest

#endif
