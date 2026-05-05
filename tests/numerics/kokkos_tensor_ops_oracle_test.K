// GPU kernel oracle tests for libMesh tensor algebra used by Kokkos kernels.
//
// Standalone executable (no CppUnit). Compares device-side tensor/vector
// operations against host libMesh TypeTensor / TypeVector oracles built from
// direct hand-written fixtures.
//
// Returns 0 on success, non-zero on failure.

#include "libmesh/libmesh_config.h"

#include "libmesh/libmesh.h"
#include "libmesh/tensor_value.h"
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
static constexpr unsigned int tensor_entries = LIBMESH_DIM * LIBMESH_DIM;
static constexpr unsigned int vector_entries = LIBMESH_DIM;
static constexpr unsigned int dim_ops_results_per_case = 1 + 4 * tensor_entries;
static constexpr unsigned int tensor_ops_results = 4 * tensor_entries + 2 * vector_entries + 2;

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

inline Kokkos::View<unsigned int *>
upload_uint(const std::vector<unsigned int> & values, const char * label)
{
  Kokkos::View<unsigned int *> d(std::string(label), values.size());
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

inline libMesh::TypeTensor<Real>
make_host_tensor(const Real xx,
                 const Real xy = 0,
                 const Real xz = 0,
                 const Real yx = 0,
                 const Real yy = 0,
                 const Real yz = 0,
                 const Real zx = 0,
                 const Real zy = 0,
                 const Real zz = 0)
{
  libMesh::TypeTensor<Real> T;
  T.zero();
  T(0, 0) = xx;
#if LIBMESH_DIM > 1
  T(0, 1) = xy;
  T(1, 0) = yx;
  T(1, 1) = yy;
#endif
#if LIBMESH_DIM > 2
  T(0, 2) = xz;
  T(1, 2) = yz;
  T(2, 0) = zx;
  T(2, 1) = zy;
  T(2, 2) = zz;
#endif
  return T;
}

struct tensor_dim_case
{
  libMesh::TypeTensor<Real> J;
  unsigned int dim;
  const char * name;
};

static const tensor_dim_case dim_cases[] = {
  { make_host_tensor(1.7, -0.2, 0.5,
                     0.3,  1.1, -0.4,
                     -0.6, 0.8, 0.9),
    1,
    "leading_1d" },
#if LIBMESH_DIM > 1
  { make_host_tensor(2.5, -0.75, 0.4,
                     1.2,  1.8, -0.6,
                     -0.3, 0.9, 1.4),
    2,
    "leading_2d" },
#endif
#if LIBMESH_DIM > 2
  { make_host_tensor(9.08973348886179e-01, 3.36455579239923e-01, 5.16389236893863e-01,
                     9.44156071777472e-01, 1.35610910092516e-01, 1.49881119060538e-02,
                     1.15988384086146e-01, 6.79845197685518e-03, 3.77028969454745e-01),
    3,
    "leading_3d" }
#endif
};

inline libMesh::TypeTensor<Real>
host_identity(const unsigned int dim)
{
  libMesh::TypeTensor<Real> I;
  I.zero();
  for (unsigned int i = 0; i < dim; ++i)
    I(i, i) = Real(1);
  return I;
}

inline Real
host_leading_determinant(const libMesh::TypeTensor<Real> & J, const unsigned int dim)
{
  if (dim == 0)
    return Real(1);
  if (dim == 1)
    return J(0, 0);
  if (dim == 2)
    return J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0);
#if LIBMESH_DIM > 2
  return J.det();
#else
  return Real(0);
#endif
}

inline libMesh::TypeTensor<Real>
host_leading_inverse(const libMesh::TypeTensor<Real> & J, const unsigned int dim)
{
  libMesh::TypeTensor<Real> inv;
  inv.zero();

  if (dim == 1)
  {
    inv(0, 0) = Real(1) / J(0, 0);
    return inv;
  }

  if (dim == 2)
  {
    const Real det = host_leading_determinant(J, dim);
    inv(0, 0) =  J(1, 1) / det;
    inv(0, 1) = -J(0, 1) / det;
    inv(1, 0) = -J(1, 0) / det;
    inv(1, 1) =  J(0, 0) / det;
    return inv;
  }

#if LIBMESH_DIM > 2
  return libMesh::TypeTensor<Real>(J.inverse());
#else
  return inv;
#endif
}

using device_vector = libMesh::TypeVector<Real>;
using device_tensor = libMesh::TypeTensor<Real>;

LIBMESH_DEVICE_INLINE
device_vector
zero_device_vector()
{
  device_vector v;
  v.zero();
  return v;
}

LIBMESH_DEVICE_INLINE
device_tensor
zero_device_tensor()
{
  device_tensor T;
  T.zero();
  return T;
}

LIBMESH_DEVICE_INLINE
device_tensor
device_leading_identity(const unsigned int dim)
{
  device_tensor I = zero_device_tensor();
  for (unsigned int i = 0; i < dim; ++i)
    I(i, i) = Real(1);
  return I;
}

LIBMESH_DEVICE_INLINE
Real
device_leading_determinant(const device_tensor & J, const unsigned int dim)
{
  if (dim == 0)
    return Real(1);
  if (dim == 1)
    return J(0, 0);
  if (dim == 2)
    return J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0);
#if LIBMESH_DIM > 2
  return J.det();
#else
  return Real(0);
#endif
}

LIBMESH_DEVICE_INLINE
device_tensor
device_leading_inverse(const device_tensor & J, const unsigned int dim)
{
  device_tensor inv = zero_device_tensor();

  if (dim == 1)
  {
    inv(0, 0) = Real(1) / J(0, 0);
    return inv;
  }

  if (dim == 2)
  {
    const Real det = device_leading_determinant(J, dim);
    inv(0, 0) =  J(1, 1) / det;
    inv(0, 1) = -J(0, 1) / det;
    inv(1, 0) = -J(1, 0) / det;
    inv(1, 1) =  J(0, 0) / det;
    return inv;
  }

#if LIBMESH_DIM > 2
  return J.inverse();
#else
  return inv;
#endif
}

inline void
fill_tensor(std::vector<Real> & ref_values,
            const unsigned int base,
            const libMesh::TypeTensor<Real> & T)
{
  for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
    for (unsigned int j = 0; j < LIBMESH_DIM; ++j)
      ref_values[base + LIBMESH_DIM * i + j] = T(i, j);
}

inline void
fill_vector(std::vector<Real> & ref_values,
            const unsigned int base,
            const libMesh::TypeVector<Real> & v)
{
  for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
    ref_values[base + i] = v(i);
}

static int
test_dim_ops()
{
  const unsigned int ncases = sizeof(dim_cases) / sizeof(dim_cases[0]);

  std::vector<Real> ref_values(dim_ops_results_per_case * ncases);
  std::vector<Real> J_entries(tensor_entries * ncases);
  std::vector<unsigned int> dims(ncases);

  for (unsigned int c = 0; c < ncases; ++c)
  {
    const auto & info = dim_cases[c];
    dims[c] = info.dim;

    for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
      for (unsigned int j = 0; j < LIBMESH_DIM; ++j)
        J_entries[tensor_entries * c + LIBMESH_DIM * i + j] = info.J(i, j);

    const Real det = host_leading_determinant(info.J, info.dim);
    const auto inv = host_leading_inverse(info.J, info.dim);
    const auto I = host_identity(info.dim);
    const auto prod_left = info.J * inv;
    const auto prod_right = inv * info.J;

    const unsigned int base = dim_ops_results_per_case * c;
    ref_values[base] = det;
    unsigned int offset = base + 1;
    fill_tensor(ref_values, offset, inv);
    offset += tensor_entries;
    fill_tensor(ref_values, offset, I);
    offset += tensor_entries;
    fill_tensor(ref_values, offset, prod_left);
    offset += tensor_entries;
    fill_tensor(ref_values, offset, prod_right);
  }

  auto d_J = upload_real(J_entries, "tensor_dim_ops_J");
  auto d_dims = upload_uint(dims, "tensor_dim_ops_dim");
  Kokkos::View<double *> d_values("tensor_dim_ops_values", dim_ops_results_per_case * ncases);

  Kokkos::parallel_for(
    static_cast<int>(ncases),
    KOKKOS_LAMBDA(int c) {
      device_tensor J = zero_device_tensor();
      for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
        for (unsigned int j = 0; j < LIBMESH_DIM; ++j)
          J(i, j) = d_J(tensor_entries * c + LIBMESH_DIM * i + j);

      const unsigned int dim = d_dims(c);
      const Real det = device_leading_determinant(J, dim);
      const auto inv = device_leading_inverse(J, dim);
      const auto I = device_leading_identity(dim);
      const auto prod_left = J * inv;
      const auto prod_right = inv * J;

      const unsigned int base = dim_ops_results_per_case * static_cast<unsigned int>(c);
      d_values(base) = det;
      unsigned int offset = base + 1;

      for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
        for (unsigned int j = 0; j < LIBMESH_DIM; ++j)
        {
          const unsigned int ij = LIBMESH_DIM * i + j;
          d_values(offset + ij) = inv(i, j);
          d_values(offset + tensor_entries + ij) = I(i, j);
          d_values(offset + 2 * tensor_entries + ij) = prod_left(i, j);
          d_values(offset + 3 * tensor_entries + ij) = prod_right(i, j);
        }
    });
  Kokkos::fence();

  return compare_device_values(d_values, ref_values);
}

static int
test_tensor_ops()
{
  const auto A = make_host_tensor(1.1, -0.4, 0.7,
                                  0.3,  1.9, -1.2,
                                  -0.8, 0.5, 2.2);
  const auto a = make_host_vector(2.0, 3.0, 4.0);
  const auto b = make_host_vector(5.0, -6.0, 7.0);
  const auto c = make_host_vector(1.25, -0.5, 2.0);

  const auto outer = libMesh::outer_product(a, b);
  const auto transpose = A.transpose();
  const auto mix = 1.5 * A - 0.25 * outer;
  const auto right = A * c;
  const auto left = c * A;
  const Real contract = A.contract(outer);
  const Real norm = A.norm();

  std::vector<Real> ref_values(tensor_ops_results);
  fill_tensor(ref_values, 0, outer);
  fill_tensor(ref_values, tensor_entries, transpose);
  fill_tensor(ref_values, 2 * tensor_entries, mix);
  unsigned int offset = 3 * tensor_entries;
  for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
  {
    fill_vector(ref_values, offset, A.row(i));
    offset += vector_entries;
  }
  fill_vector(ref_values, offset, right);
  offset += vector_entries;
  fill_vector(ref_values, offset, left);
  offset += vector_entries;
  ref_values[offset++] = contract;
  ref_values[offset++] = norm;

  std::vector<Real> tensor_entries_h(tensor_entries);
  std::vector<Real> a_h(vector_entries), b_h(vector_entries), c_h(vector_entries);
  for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
  {
    a_h[i] = a(i);
    b_h[i] = b(i);
    c_h[i] = c(i);
    for (unsigned int j = 0; j < LIBMESH_DIM; ++j)
      tensor_entries_h[LIBMESH_DIM * i + j] = A(i, j);
  }

  auto d_A = upload_real(tensor_entries_h, "tensor_ops_A");
  auto d_a = upload_real(a_h, "tensor_ops_a");
  auto d_b = upload_real(b_h, "tensor_ops_b");
  auto d_c = upload_real(c_h, "tensor_ops_c");
  Kokkos::View<double *> d_values("tensor_ops_values", tensor_ops_results);

  Kokkos::parallel_for(
    1,
    KOKKOS_LAMBDA(int) {
      device_tensor A_d = zero_device_tensor();
      device_vector a_d = zero_device_vector();
      device_vector b_d = zero_device_vector();
      device_vector c_d = zero_device_vector();

      for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
      {
        a_d(i) = d_a(i);
        b_d(i) = d_b(i);
        c_d(i) = d_c(i);
        for (unsigned int j = 0; j < LIBMESH_DIM; ++j)
          A_d(i, j) = d_A(LIBMESH_DIM * i + j);
      }

      const auto outer_d = libMesh::outer_product(a_d, b_d);
      const auto transpose_d = A_d.transpose();
      const auto mix_d = 1.5 * A_d - 0.25 * outer_d;
      const auto right_d = A_d * c_d;
      const auto left_d = c_d * A_d;
      const Real contract_d = A_d.contract(outer_d);
      const Real norm_d = A_d.norm();

      unsigned int row_offset = 3 * tensor_entries;
      for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
      {
        const auto row = A_d.row(i);
        for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
          d_values(row_offset + i * LIBMESH_DIM + d) = row(d);
      }

      unsigned int vector_offset = row_offset + tensor_entries;
      for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
      {
        d_values(vector_offset + i) = right_d(i);
        d_values(vector_offset + vector_entries + i) = left_d(i);
        for (unsigned int j = 0; j < LIBMESH_DIM; ++j)
        {
          const unsigned int ij = LIBMESH_DIM * i + j;
          d_values(ij) = outer_d(i, j);
          d_values(tensor_entries + ij) = transpose_d(i, j);
          d_values(2 * tensor_entries + ij) = mix_d(i, j);
        }
      }

      d_values(vector_offset + 2 * vector_entries) = contract_d;
      d_values(vector_offset + 2 * vector_entries + 1) = norm_d;
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

  const int dim_fail = test_dim_ops();
  std::printf("[tensor_dim_kernel_oracle] %s  (%d failures)\n",
              dim_fail ? "FAIL" : "PASS",
              dim_fail);
  total_fail += dim_fail;

  const int tensor_fail = test_tensor_ops();
  std::printf("[tensor_ops_kernel_oracle] %s  (%d failures)\n",
              tensor_fail ? "FAIL" : "PASS",
              tensor_fail);
  total_fail += tensor_fail;

  Kokkos::finalize();

  if (total_fail == 0)
    std::printf("ALL TESTS PASSED\n");
  else
    std::printf("%d TEST(S) FAILED\n", total_fail);

  return total_fail ? 1 : 0;
}
