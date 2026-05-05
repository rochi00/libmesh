// GPU kernel oracle tests for libMesh::TypeVector / VectorValue numerics.
//
// Standalone executable (no CppUnit). Compares device-side vector algebra
// against host libMesh TypeVector oracles built from direct hand-written
// fixtures.
//
// Returns 0 on success, non-zero on failure.

#include "libmesh/libmesh_config.h"

#include "libmesh/libmesh.h"
#include "libmesh/type_vector.h"
#include "libmesh/vector_value.h"

// Avoid conflicting complex operators between CUDA and PETSc
#define PETSC_SKIP_CXX_COMPLEX_FIX 1
#include <Kokkos_Core.hpp>
#undef __CUDACC_VER__

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using libMesh::Real;

static constexpr double tol = 2.0e-13;
static constexpr double unit_tol = 1.0e-14;
static constexpr unsigned int n_results =
  3 * LIBMESH_DIM + 2 + ((LIBMESH_DIM > 2) ? (2 * LIBMESH_DIM) : 0) + 1;

namespace
{

inline int
compare_device_values(const Kokkos::View<double *> & d_values,
                      const std::vector<Real> & ref_values,
                      const double local_tol = tol)
{
  auto h_values = Kokkos::create_mirror_view(d_values);
  Kokkos::deep_copy(h_values, d_values);

  int fail = 0;
  for (std::size_t i = 0; i < ref_values.size(); ++i)
    if (std::fabs(h_values(i) - ref_values[i]) > local_tol)
      ++fail;

  return fail;
}

inline Kokkos::View<double *>
upload_real(const std::vector<Real> & values, const char * label)
{
  Kokkos::View<double *> d(std::string(label), values.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < values.size(); ++i)
    h(i) = values[i];
  Kokkos::deep_copy(d, h);
  return d;
}

inline libMesh::TypeVector<Real>
make_host_vector(const Real x, const Real y = 0, const Real z = 0)
{
  libMesh::TypeVector<Real> v;
  v.zero();
  v(0) = x;
#if LIBMESH_DIM > 1
  v(1) = y;
#endif
#if LIBMESH_DIM > 2
  v(2) = z;
#endif
  return v;
}

struct vector_kernel_case
{
  const char * name;
  Real ax, ay, az;
  Real bx, by, bz;
  Real cx, cy, cz;
};

static const vector_kernel_case cases[] = {
#if LIBMESH_DIM >= 1
  { "line_case_a", 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.5, 0.0, 0.0 },
  { "line_case_b", -1.25, 0.0, 0.0, 4.5, 0.0, 0.0, -2.0, 0.0, 0.0 },
#endif
#if LIBMESH_DIM >= 2
  { "plane_case_a", 2.0, 3.0, 0.0, 5.0, -6.0, 0.0, 1.25, -0.5, 0.0 },
  { "plane_case_b", -1.0, 4.0, 0.0, 0.5, 2.5, 0.0, -3.0, 1.5, 0.0 },
#endif
#if LIBMESH_DIM >= 3
  { "space_case_a", 2.0, 3.0, 4.0, 5.0, -6.0, 7.0, 1.25, -0.5, 2.0 },
  { "space_case_b", -1.0, 4.0, 0.75, 0.5, 2.5, -3.5, -3.0, 1.5, 2.25 },
#endif
};

inline void
fill_manual_oracle(std::vector<Real> & ref_values,
                   const libMesh::TypeVector<Real> & a,
                   const libMesh::TypeVector<Real> & b,
                   const libMesh::TypeVector<Real> & c)
{
  libMesh::TypeVector<Real> mix = a + b;
  mix -= c;

  libMesh::TypeVector<Real> scaled = 1.25 * a;
  scaled += (-0.5) * b;
  scaled += (0.25) * c;

  unsigned int offset = 0;
  for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
    ref_values[offset++] = mix(d);

  for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
    ref_values[offset++] = scaled(d);

  for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
    ref_values[offset++] = scaled(d);

  ref_values[offset++] = a * b;
  ref_values[offset++] = a.contract(b);

#if LIBMESH_DIM > 2
  const auto cross = a.cross(b);
  auto unit_cross = cross;
  if (cross.norm() > unit_tol)
    unit_cross = cross.unit();

  for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
    ref_values[offset++] = cross(d);

  for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
    ref_values[offset++] = unit_cross(d);
#endif

  ref_values[offset++] = mix.norm();
}

template <typename Vec>
static int
test_vector_ops_case(const vector_kernel_case & info)
{
  const auto a = make_host_vector(info.ax, info.ay, info.az);
  const auto b = make_host_vector(info.bx, info.by, info.bz);
  const auto c = make_host_vector(info.cx, info.cy, info.cz);

  std::vector<Real> ref_values(n_results);
  fill_manual_oracle(ref_values, a, b, c);

  std::vector<Real> a_h(LIBMESH_DIM), b_h(LIBMESH_DIM), c_h(LIBMESH_DIM);
  for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
  {
    a_h[i] = a(i);
    b_h[i] = b(i);
    c_h[i] = c(i);
  }

  auto d_a = upload_real(a_h, "vector_ops_a");
  auto d_b = upload_real(b_h, "vector_ops_b");
  auto d_c = upload_real(c_h, "vector_ops_c");
  Kokkos::View<double *> d_values("vector_ops_values", n_results);

  Kokkos::parallel_for(
    1,
    KOKKOS_LAMBDA(int) {
      Vec a_d;
      Vec b_d;
      Vec c_d;
      a_d.zero();
      b_d.zero();
      c_d.zero();

      for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
      {
        a_d(i) = d_a(i);
        b_d(i) = d_b(i);
        c_d(i) = d_c(i);
      }

      Vec mix = a_d + b_d;
      mix -= c_d;

      Vec scaled = 1.25 * a_d;
      scaled += (-0.5) * b_d;
      scaled += (0.25) * c_d;

      Vec accum;
      accum.zero();
      accum.add_scaled(a_d, 1.25);
      accum.add_scaled(b_d, -0.5);
      accum.subtract_scaled(c_d, -0.25);

      const Real dot = a_d * b_d;
      const Real contract = a_d.contract(b_d);

      unsigned int offset = 0;
      for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
        d_values(offset++) = mix(d);

      for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
        d_values(offset++) = scaled(d);

      for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
        d_values(offset++) = accum(d);

      d_values(offset++) = dot;
      d_values(offset++) = contract;

#if LIBMESH_DIM > 2
      const Vec cross = a_d.cross(b_d);
      Vec unit_cross = cross;
      if (cross.norm() > unit_tol)
        unit_cross = cross.unit();

      for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
        d_values(offset++) = cross(d);

      for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
        d_values(offset++) = unit_cross(d);
#endif

      d_values(offset++) = mix.norm();
    });
  Kokkos::fence();

  return compare_device_values(d_values, ref_values);
}

} // anonymous namespace

int
main(int argc, char ** argv)
{
  Kokkos::initialize(argc, argv);
  libMesh::LibMeshInit init(argc, argv);

  int total_fail = 0;

  for (const auto & info : cases)
  {
    const int f = test_vector_ops_case<libMesh::TypeVector<Real>>(info);
    std::printf("[typevector_kernel_oracle] [%s] %s  (%d failures)\n",
                info.name, f ? "FAIL" : "PASS", f);
    total_fail += f;
  }

  for (const auto & info : cases)
  {
    const int f = test_vector_ops_case<libMesh::VectorValue<Real>>(info);
    std::printf("[vectorvalue_kernel_oracle] [%s] %s  (%d failures)\n",
                info.name, f ? "FAIL" : "PASS", f);
    total_fail += f;
  }

  Kokkos::finalize();

  if (total_fail == 0)
    std::printf("ALL TESTS PASSED\n");
  else
    std::printf("%d TEST(S) FAILED\n", total_fail);

  return total_fail ? 1 : 0;
}
