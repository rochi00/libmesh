// GPU kernel tests for libMesh::Kokkos HDG local assembly parity.
//
// Standalone executable fragment included by driver_kokkos.K through
// kokkos_fe_oracle_runners.h. Uses host FEBase values as the oracle for
// one-element HDG local block assembly, matrix condensation, and residual
// condensation.

#include "libmesh/libmesh_config.h"

#include "gpu/kokkos_fe_base.h"
#include "gpu/kokkos_fe_evaluator.h"
#include "gpu/kokkos_fe_face_map.h"
#include "gpu/kokkos_fe_map.h"
#include "gpu/kokkos_fe_types.h"
#include "gpu/kokkos_hdg_assembly.h"

#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_base_kokkos.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/quadrature_gauss.h"
#include "timpi/communicator.h"

#define PETSC_SKIP_CXX_COMPLEX_FIX 1
#include <Kokkos_Core.hpp>
#undef __CUDACC_VER__

#include "kokkos_fe_oracle_test_utils.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace libMeshTest
{
namespace KokkosHDGAssemblyOracle
{

using kokkos_test_utils::compare_device_values;
using kokkos_test_utils::upload_point_coordinates;
using kokkos_test_utils::upload_real;
using kokkos_test_utils::vector_component;
using libMesh::Kokkos::Real;
using libMesh::Kokkos::RealVector;

namespace
{

struct hdg_assembly_case
{
  libMesh::ElemType parent_type;
  libMesh::Order order;
  const char * name;
};

static const hdg_assembly_case hdg_assembly_cases[] = {
    {libMesh::TRI6, libMesh::FIRST, "HDG_ASSEMBLY/TRI6/FIRST"},
    {libMesh::TRI7, libMesh::FIRST, "HDG_ASSEMBLY/TRI7/FIRST"},
    {libMesh::QUAD8, libMesh::FIRST, "HDG_ASSEMBLY/QUAD8/FIRST"},
    {libMesh::QUAD9, libMesh::FIRST, "HDG_ASSEMBLY/QUAD9/FIRST"}};

} // namespace

static int
compare_device_rows(const Kokkos::View<double **> & d_values,
                    const std::vector<Real> & ref_values,
                    const double tol)
{
  auto h_values = Kokkos::create_mirror_view(d_values);
  Kokkos::deep_copy(h_values, d_values);

  int fail = 0;
  for (std::size_t row = 0; row != d_values.extent(0); ++row)
    for (std::size_t i = 0; i != ref_values.size(); ++i)
      if (std::fabs(h_values(row, i) - ref_values[i]) > tol)
        ++fail;

  return fail;
}

static int
count_nonzero_statuses(const Kokkos::View<int *> & d_status)
{
  auto h_status = Kokkos::create_mirror_view(d_status);
  Kokkos::deep_copy(h_status, d_status);

  int fail = 0;
  for (std::size_t i = 0; i != d_status.extent(0); ++i)
    if (h_status(i))
      ++fail;

  return fail;
}

static int
condense_hdg_linear_reference(const libMesh::Kokkos::detail::HDGLinearBlockLayout layout,
                              const std::vector<Real> & blocks,
                              std::vector<Real> & condensed)
{
  const unsigned int n_internal =
      libMesh::Kokkos::detail::hdg_linear_condensed_internal_n_dofs(layout);
  const unsigned int n_external =
      libMesh::Kokkos::detail::hdg_linear_condensed_external_n_dofs(layout);
  const unsigned int n_cols = n_internal + n_external;
  const unsigned int q0 = 0;
  const unsigned int u0 = q0 + layout.vector_n_dofs;
  const unsigned int p0 = u0 + layout.scalar_n_dofs;
  const unsigned int lm0 = p0 + layout.scalar_n_dofs;

  condensed.assign(n_external * n_external, Real(0));

  std::vector<Real> a(n_cols * n_cols, Real(0));
  std::vector<Real> aug(n_internal * n_cols, Real(0));
  std::vector<Real> x(n_internal * n_external, Real(0));

  const auto scatter = [&a, &blocks, n_cols](unsigned int offset,
                                             unsigned int rows,
                                             unsigned int cols,
                                             unsigned int row0,
                                             unsigned int col0)
  {
    for (unsigned int i = 0; i != rows; ++i)
      for (unsigned int j = 0; j != cols; ++j)
        a[(row0 + i) * n_cols + col0 + j] = blocks[offset + i * cols + j];
  };

  scatter(layout.off_Jqq, layout.vector_n_dofs, layout.vector_n_dofs, q0, q0);
  scatter(layout.off_Jqu, layout.vector_n_dofs, layout.scalar_n_dofs, q0, u0);
  scatter(layout.off_Juq, layout.scalar_n_dofs, layout.vector_n_dofs, u0, q0);
  scatter(layout.off_Jup, layout.scalar_n_dofs, layout.scalar_n_dofs, u0, p0);
  scatter(layout.off_Juu, layout.scalar_n_dofs, layout.scalar_n_dofs, u0, u0);
  scatter(layout.off_Jpu, layout.scalar_n_dofs, layout.scalar_n_dofs, p0, u0);
  scatter(layout.off_Jplm, layout.scalar_n_dofs, layout.lm_n_dofs, p0, lm0);
  scatter(layout.off_Jqlm, layout.vector_n_dofs, layout.lm_n_dofs, q0, lm0);
  scatter(layout.off_Julm, layout.scalar_n_dofs, layout.lm_n_dofs, u0, lm0);
  scatter(layout.off_Jlmq, layout.lm_n_dofs, layout.vector_n_dofs, lm0, q0);
  scatter(layout.off_Jlmp, layout.lm_n_dofs, layout.scalar_n_dofs, lm0, p0);
  scatter(layout.off_Jlms, layout.lm_n_dofs, layout.scalar_n_dofs, lm0, u0);
  scatter(layout.off_Jlmlm, layout.lm_n_dofs, layout.lm_n_dofs, lm0, lm0);

  for (unsigned int i = 0; i != n_internal; ++i)
  {
    for (unsigned int j = 0; j != n_internal; ++j)
      aug[i * n_cols + j] = a[i * n_cols + j];

    for (unsigned int j = 0; j != n_external; ++j)
      aug[i * n_cols + n_internal + j] = a[i * n_cols + n_internal + j];
  }

  for (unsigned int k = 0; k != n_internal; ++k)
  {
    unsigned int pivot_row = k;
    Real pivot_abs = std::fabs(aug[k * n_cols + k]);

    for (unsigned int r = k + 1; r != n_internal; ++r)
    {
      const Real candidate = std::fabs(aug[r * n_cols + k]);
      if (candidate > pivot_abs)
      {
        pivot_abs = candidate;
        pivot_row = r;
      }
    }

    if (pivot_abs <= Real(1.e-20))
      return 1;

    if (pivot_row != k)
      for (unsigned int c = 0; c != n_cols; ++c)
      {
        const Real tmp = aug[k * n_cols + c];
        aug[k * n_cols + c] = aug[pivot_row * n_cols + c];
        aug[pivot_row * n_cols + c] = tmp;
      }

    const Real pivot = aug[k * n_cols + k];
    for (unsigned int r = k + 1; r != n_internal; ++r)
    {
      const Real factor = aug[r * n_cols + k] / pivot;
      aug[r * n_cols + k] = Real(0);

      for (unsigned int c = k + 1; c != n_cols; ++c)
        aug[r * n_cols + c] -= factor * aug[k * n_cols + c];
    }
  }

  for (unsigned int rhs = 0; rhs != n_external; ++rhs)
    for (unsigned int rr = n_internal; rr != 0; --rr)
    {
      const unsigned int r = rr - 1;
      Real value = aug[r * n_cols + n_internal + rhs];

      for (unsigned int c = r + 1; c != n_internal; ++c)
        value -= aug[r * n_cols + c] * x[c * n_external + rhs];

      x[r * n_external + rhs] = value / aug[r * n_cols + r];
    }

  for (unsigned int i = 0; i != n_external; ++i)
    for (unsigned int j = 0; j != n_external; ++j)
    {
      Real value = a[(n_internal + i) * n_cols + n_internal + j];

      for (unsigned int k = 0; k != n_internal; ++k)
        value -= a[(n_internal + i) * n_cols + k] * x[k * n_external + j];

      condensed[i * n_external + j] = value;
    }

  return 0;
}

static int
condense_hdg_linear_residual_reference(const libMesh::Kokkos::detail::HDGLinearBlockLayout layout,
                                       const std::vector<Real> & blocks,
                                       const std::vector<Real> & residual,
                                       std::vector<Real> & condensed)
{
  const unsigned int n_internal =
      libMesh::Kokkos::detail::hdg_linear_condensed_internal_n_dofs(layout);
  const unsigned int n_external =
      libMesh::Kokkos::detail::hdg_linear_condensed_external_n_dofs(layout);
  const unsigned int n_total = n_internal + n_external;
  const unsigned int q0 = 0;
  const unsigned int u0 = q0 + layout.vector_n_dofs;
  const unsigned int p0 = u0 + layout.scalar_n_dofs;
  const unsigned int lm0 = p0 + layout.scalar_n_dofs;

  condensed.assign(n_external, Real(0));

  std::vector<Real> a(n_total * n_total, Real(0));
  std::vector<Real> aug(n_internal * (n_internal + 1), Real(0));
  std::vector<Real> y(n_internal, Real(0));

  const auto scatter = [&a, &blocks, n_total](unsigned int offset,
                                              unsigned int rows,
                                              unsigned int cols,
                                              unsigned int row0,
                                              unsigned int col0)
  {
    for (unsigned int i = 0; i != rows; ++i)
      for (unsigned int j = 0; j != cols; ++j)
        a[(row0 + i) * n_total + col0 + j] = blocks[offset + i * cols + j];
  };

  scatter(layout.off_Jqq, layout.vector_n_dofs, layout.vector_n_dofs, q0, q0);
  scatter(layout.off_Jqu, layout.vector_n_dofs, layout.scalar_n_dofs, q0, u0);
  scatter(layout.off_Juq, layout.scalar_n_dofs, layout.vector_n_dofs, u0, q0);
  scatter(layout.off_Jup, layout.scalar_n_dofs, layout.scalar_n_dofs, u0, p0);
  scatter(layout.off_Juu, layout.scalar_n_dofs, layout.scalar_n_dofs, u0, u0);
  scatter(layout.off_Jpu, layout.scalar_n_dofs, layout.scalar_n_dofs, p0, u0);
  scatter(layout.off_Jqlm, layout.vector_n_dofs, layout.lm_n_dofs, q0, lm0);
  scatter(layout.off_Jlmq, layout.lm_n_dofs, layout.vector_n_dofs, lm0, q0);
  scatter(layout.off_Jlms, layout.lm_n_dofs, layout.scalar_n_dofs, lm0, u0);

  for (unsigned int i = 0; i != n_internal; ++i)
  {
    for (unsigned int j = 0; j != n_internal; ++j)
      aug[i * (n_internal + 1) + j] = a[i * n_total + j];

    aug[i * (n_internal + 1) + n_internal] = residual[i];
  }

  for (unsigned int k = 0; k != n_internal; ++k)
  {
    unsigned int pivot_row = k;
    Real pivot_abs = std::fabs(aug[k * (n_internal + 1) + k]);

    for (unsigned int r = k + 1; r != n_internal; ++r)
    {
      const Real candidate = std::fabs(aug[r * (n_internal + 1) + k]);
      if (candidate > pivot_abs)
      {
        pivot_abs = candidate;
        pivot_row = r;
      }
    }

    if (pivot_abs <= Real(1.e-20))
      return 1;

    if (pivot_row != k)
      for (unsigned int c = 0; c != n_internal + 1; ++c)
      {
        const Real tmp = aug[k * (n_internal + 1) + c];
        aug[k * (n_internal + 1) + c] = aug[pivot_row * (n_internal + 1) + c];
        aug[pivot_row * (n_internal + 1) + c] = tmp;
      }

    const Real pivot = aug[k * (n_internal + 1) + k];
    for (unsigned int r = k + 1; r != n_internal; ++r)
    {
      const Real factor = aug[r * (n_internal + 1) + k] / pivot;
      aug[r * (n_internal + 1) + k] = Real(0);

      for (unsigned int c = k + 1; c != n_internal + 1; ++c)
        aug[r * (n_internal + 1) + c] -= factor * aug[k * (n_internal + 1) + c];
    }
  }

  for (unsigned int rr = n_internal; rr != 0; --rr)
  {
    const unsigned int r = rr - 1;
    Real value = aug[r * (n_internal + 1) + n_internal];

    for (unsigned int c = r + 1; c != n_internal; ++c)
      value -= aug[r * (n_internal + 1) + c] * y[c];

    y[r] = value / aug[r * (n_internal + 1) + r];
  }

  for (unsigned int i = 0; i != n_external; ++i)
  {
    Real value = residual[n_internal + i];

    for (unsigned int k = 0; k != n_internal; ++k)
      value -= a[(n_internal + i) * n_total + k] * y[k];

    condensed[i] = value;
  }

  return 0;
}

template <libMesh::ElemType ExactTopo>
static int
test_hdg_linear_block_case(const hdg_assembly_case & info)
{
  libMesh::Parallel::Communicator comm(libMesh::GLOBAL_COMM_WORLD);
  libMesh::Mesh mesh(comm, 2);
  libMesh::MeshTools::Generation::build_square(mesh, 1, 1, 0., 1., 0., 1., info.parent_type);
  mesh.prepare_for_use();

  const auto & cache = mesh.get_kokkos_geometry_cache();
  const auto edge_orientations = cache.element_edge_positive_orientations;
  const auto elem_id = cache.host_element_ids.front();
  const libMesh::Elem & elem = *mesh.query_elem_ptr(elem_id);
  const unsigned int elem_index = mesh.get_kokkos_elem_index(elem);

  const libMesh::FEType vector_fe_type(info.order, libMesh::L2_LAGRANGE_VEC);
  const libMesh::FEType scalar_fe_type(info.order, libMesh::L2_LAGRANGE);
  const libMesh::FEType lm_fe_type(info.order, libMesh::SIDE_HIERARCHIC);
  const libMesh::Kokkos::FEShapeKey vector_key{
      libMesh::L2_LAGRANGE_VEC, info.parent_type, info.order};
  const libMesh::Kokkos::FEShapeKey scalar_key{libMesh::L2_LAGRANGE, info.parent_type, info.order};
  const libMesh::Kokkos::FEShapeKey lm_key{libMesh::SIDE_HIERARCHIC, info.parent_type, info.order};

  auto vector_fe = libMesh::FEVectorBase::build(2, vector_fe_type);
  auto scalar_fe = libMesh::FEBase::build(2, scalar_fe_type);
  auto vector_fe_face = libMesh::FEVectorBase::build(2, vector_fe_type);
  auto scalar_fe_face = libMesh::FEBase::build(2, scalar_fe_type);
  auto lm_fe_face = libMesh::FEBase::build(2, lm_fe_type);

  libMesh::QGauss qrule(2, libMesh::FOURTH);
  qrule.allow_rules_with_negative_weights = true;
  vector_fe->attach_quadrature_rule(&qrule);
  scalar_fe->attach_quadrature_rule(&qrule);

  const auto & JxW = vector_fe->get_JxW();
  const auto & vector_phi = vector_fe->get_phi();
  const auto & div_vector_phi = vector_fe->get_div_phi();
  const auto & scalar_phi = scalar_fe->get_phi();
  const auto & grad_scalar_phi = scalar_fe->get_dphi();
  vector_fe->reinit(&elem);
  scalar_fe->reinit(&elem);

  const unsigned int vector_n_dofs = libMesh::FEInterface::n_dofs(vector_fe_type, 0, &elem);
  const unsigned int scalar_n_dofs = libMesh::FEInterface::n_dofs(scalar_fe_type, 0, &elem);
  const unsigned int lm_n_dofs = libMesh::FEInterface::n_dofs(lm_fe_type, 0, &elem);

  const auto layout = libMesh::Kokkos::detail::make_hdg_linear_block_layout(
      vector_n_dofs, scalar_n_dofs, lm_n_dofs);
  const unsigned int off_Jqq = layout.off_Jqq;
  const unsigned int off_Jqu = layout.off_Jqu;
  const unsigned int off_Juq = layout.off_Juq;
  const unsigned int off_Jup = layout.off_Jup;
  const unsigned int off_Jpu = layout.off_Jpu;
  const unsigned int off_Jplm = layout.off_Jplm;
  const unsigned int off_Jqlm = layout.off_Jqlm;
  const unsigned int off_Jlmq = layout.off_Jlmq;
  const unsigned int off_Jlmp = layout.off_Jlmp;
  const unsigned int off_Jlms = layout.off_Jlms;
  const unsigned int off_Jlmlm = layout.off_Jlmlm;
  const unsigned int off_Juu = layout.off_Juu;
  const unsigned int off_Julm = layout.off_Julm;
  const unsigned int n_entries = layout.n_entries;
  const unsigned int q0 = 0;
  const unsigned int u0 = q0 + vector_n_dofs;
  const unsigned int p0 = u0 + scalar_n_dofs;
  const unsigned int lm0 = p0 + scalar_n_dofs;
  const unsigned int residual_n_dofs = libMesh::Kokkos::detail::hdg_linear_n_dofs(layout);
  const Real nu = Real(1);
  const Real tau = Real(1);

  std::vector<Real> q_dof_values(vector_n_dofs), u_dof_values(scalar_n_dofs),
      p_dof_values(scalar_n_dofs), lm_dof_values(lm_n_dofs);
  for (unsigned int i = 0; i != vector_n_dofs; ++i)
    q_dof_values[i] =
        Real(0.125) * static_cast<Real>(i + 1) + Real(0.03125) * static_cast<Real>((i * i + 3) % 7);
  for (unsigned int i = 0; i != scalar_n_dofs; ++i)
  {
    u_dof_values[i] = Real(-0.075) * static_cast<Real>(i + 2);
    p_dof_values[i] = Real(0.05) * static_cast<Real>((i + 1) * (i + 3));
  }
  for (unsigned int i = 0; i != lm_n_dofs; ++i)
    lm_dof_values[i] = Real(0.0625) * static_cast<Real>(i + 1);

  const auto scalar_qp_value =
      [](const auto & phi, const std::vector<Real> & dof_values, const unsigned int qp)
  {
    Real value = Real(0);
    for (unsigned int i = 0; i != dof_values.size(); ++i)
      value += phi[i][qp] * dof_values[i];
    return value;
  };

  const auto vector_qp_value =
      [](const auto & phi, const std::vector<Real> & dof_values, const unsigned int qp)
  {
    libMesh::VectorValue<Real> value;
    for (unsigned int i = 0; i != dof_values.size(); ++i)
      value += phi[i][qp] * dof_values[i];
    return value;
  };

  std::vector<Real> ref(n_entries, Real(0));
  std::vector<Real> ref_residual(residual_n_dofs, Real(0));
  const auto add_ref =
      [&ref](unsigned int offset, unsigned int cols, unsigned int i, unsigned int j, Real value)
  { ref[offset + i * cols + j] += value; };

  const unsigned int volume_nqp = qrule.n_points();
  std::vector<Real> xi_h(volume_nqp), eta_h(volume_nqp), zeta_h(volume_nqp), jxw_h(volume_nqp);
  for (unsigned int q = 0; q != volume_nqp; ++q)
  {
    xi_h[q] = qrule.qp(q)(0);
    eta_h[q] = qrule.qp(q)(1);
    zeta_h[q] = Real(0);
    jxw_h[q] = JxW[q];

    const auto q_value = vector_qp_value(vector_phi, q_dof_values, q);
    const Real u_value = scalar_qp_value(scalar_phi, u_dof_values, q);
    const Real p_value = scalar_qp_value(scalar_phi, p_dof_values, q);

    for (unsigned int i = 0; i != vector_n_dofs; ++i)
    {
      ref_residual[q0 + i] +=
          JxW[q] * ((vector_phi[i][q] * q_value) + div_vector_phi[i][q] * u_value);

      for (unsigned int j = 0; j != vector_n_dofs; ++j)
        add_ref(off_Jqq, vector_n_dofs, i, j, JxW[q] * (vector_phi[i][q] * vector_phi[j][q]));

      for (unsigned int j = 0; j != scalar_n_dofs; ++j)
        add_ref(off_Jqu, scalar_n_dofs, i, j, JxW[q] * div_vector_phi[i][q] * scalar_phi[j][q]);
    }

    for (unsigned int i = 0; i != scalar_n_dofs; ++i)
    {
      auto stress = q_value * nu;
      stress(0) -= p_value;
      ref_residual[u0 + i] += JxW[q] * (grad_scalar_phi[i][q] * stress);
      ref_residual[p0 + i] -= JxW[q] * grad_scalar_phi[i][q](0) * u_value;

      for (unsigned int j = 0; j != vector_n_dofs; ++j)
        add_ref(
            off_Juq, vector_n_dofs, i, j, JxW[q] * nu * (grad_scalar_phi[i][q] * vector_phi[j][q]));

      for (unsigned int j = 0; j != scalar_n_dofs; ++j)
      {
        add_ref(
            off_Jup, scalar_n_dofs, i, j, -JxW[q] * grad_scalar_phi[i][q](0) * scalar_phi[j][q]);
        add_ref(
            off_Jpu, scalar_n_dofs, i, j, -JxW[q] * grad_scalar_phi[i][q](0) * scalar_phi[j][q]);
      }
    }
  }

  Kokkos::View<double *> d_blocks(std::string("hdg_linear_blocks"), n_entries);
  Kokkos::deep_copy(d_blocks, 0.);
  Kokkos::View<double *> d_residual(std::string("hdg_linear_residual"), residual_n_dofs);
  Kokkos::deep_copy(d_residual, 0.);
  auto d_q_dofs = upload_real(q_dof_values, "hdg_linear_q_dofs");
  auto d_u_dofs = upload_real(u_dof_values, "hdg_linear_u_dofs");
  auto d_p_dofs = upload_real(p_dof_values, "hdg_linear_p_dofs");
  auto d_lm_dofs = upload_real(lm_dof_values, "hdg_linear_lm_dofs");
  auto d_coords = upload_point_coordinates(elem, "hdg_linear_coords");
  auto d_xi = upload_real(xi_h, "hdg_linear_xi");
  auto d_eta = upload_real(eta_h, "hdg_linear_eta");
  auto d_zeta = upload_real(zeta_h, "hdg_linear_zeta");
  auto d_jxw = upload_real(jxw_h, "hdg_linear_jxw");

  const unsigned int n_geom_nodes = elem.n_nodes();
  libMesh::Kokkos::detail::assemble_hdg_linear_volume_blocks<ExactTopo>(vector_key,
                                                                        scalar_key,
                                                                        layout,
                                                                        d_coords,
                                                                        n_geom_nodes,
                                                                        d_xi,
                                                                        d_eta,
                                                                        d_zeta,
                                                                        d_jxw,
                                                                        volume_nqp,
                                                                        nu,
                                                                        d_blocks,
                                                                        "hdg_linear_volume_blocks");
  Kokkos::fence();

  libMesh::Kokkos::detail::assemble_hdg_linear_volume_residual<ExactTopo>(
      vector_key,
      scalar_key,
      layout,
      d_coords,
      n_geom_nodes,
      d_xi,
      d_eta,
      d_zeta,
      d_jxw,
      volume_nqp,
      nu,
      d_q_dofs,
      d_u_dofs,
      d_p_dofs,
      d_residual,
      "hdg_linear_volume_residual");
  Kokkos::fence();

  const unsigned int n_sides = elem.n_sides();
  auto first_side = elem.build_side_ptr(0);
  libMesh::QGauss qface_probe(1, libMesh::FOURTH);
  qface_probe.allow_rules_with_negative_weights = true;
  qface_probe.init(first_side->type());
  const unsigned int side_nqp = qface_probe.n_points();
  std::vector<Real> batch_parent_xi_h(n_sides * side_nqp), batch_parent_eta_h(n_sides * side_nqp),
      batch_parent_zeta_h(n_sides * side_nqp), batch_side_jxw_h(n_sides * side_nqp),
      batch_normal_x_h(n_sides * side_nqp), batch_normal_y_h(n_sides * side_nqp),
      batch_normal_z_h(n_sides * side_nqp);

  for (unsigned int side_id = 0; side_id != n_sides; ++side_id)
  {
    auto side = elem.build_side_ptr(side_id);
    libMesh::QGauss qface(1, libMesh::FOURTH);
    qface.allow_rules_with_negative_weights = true;
    qface.init(side->type());

    vector_fe_face->attach_quadrature_rule(&qface);
    scalar_fe_face->attach_quadrature_rule(&qface);
    lm_fe_face->attach_quadrature_rule(&qface);

    const auto & vector_phi_face = vector_fe_face->get_phi();
    const auto & scalar_phi_face = scalar_fe_face->get_phi();
    const auto & lm_phi_face = lm_fe_face->get_phi();
    const auto & JxW_face = vector_fe_face->get_JxW();
    const auto & normals = vector_fe_face->get_normals();
    vector_fe_face->reinit(&elem, side_id);
    scalar_fe_face->reinit(&elem, side_id);
    lm_fe_face->reinit(&elem, side_id);

    const unsigned int nqp = qface.n_points();
    if (nqp != side_nqp)
    {
      std::printf("  HDG batch fixture side quadrature mismatch: %s side=%u\n", info.name, side_id);
      return 1;
    }
    std::vector<Real> parent_xi_h(nqp), parent_eta_h(nqp), parent_zeta_h(nqp), side_jxw_h(nqp),
        normal_x_h(nqp), normal_y_h(nqp), normal_z_h(nqp);

    for (unsigned int q = 0; q != nqp; ++q)
    {
      const RealVector face_qpt = libMesh::Kokkos::make_vector(qface.qp(q)(0));
      const RealVector parent_qpt = libMesh::Kokkos::map_face_qp_to_parent(
          *side, libMesh::LAGRANGE_MAP, side->type(), face_qpt);
      parent_xi_h[q] = vector_component(parent_qpt, 0);
      parent_eta_h[q] = vector_component(parent_qpt, 1);
      parent_zeta_h[q] = Real(0);
      side_jxw_h[q] = JxW_face[q];
      normal_x_h[q] = normals[q](0);
      normal_y_h[q] = normals[q](1);
      normal_z_h[q] = normals[q](2);
      const unsigned int side_q = side_id * side_nqp + q;
      batch_parent_xi_h[side_q] = parent_xi_h[q];
      batch_parent_eta_h[side_q] = parent_eta_h[q];
      batch_parent_zeta_h[side_q] = parent_zeta_h[q];
      batch_side_jxw_h[side_q] = side_jxw_h[q];
      batch_normal_x_h[side_q] = normal_x_h[q];
      batch_normal_y_h[side_q] = normal_y_h[q];
      batch_normal_z_h[side_q] = normal_z_h[q];
      const Real normal_norm2 = normals[q] * normals[q];
      const auto q_value = vector_qp_value(vector_phi_face, q_dof_values, q);
      const Real u_value = scalar_qp_value(scalar_phi_face, u_dof_values, q);
      const Real p_value = scalar_qp_value(scalar_phi_face, p_dof_values, q);
      const Real lm_value = scalar_qp_value(lm_phi_face, lm_dof_values, q);
      const Real q_normal = q_value * normals[q];

      for (unsigned int i = 0; i != vector_n_dofs; ++i)
        ref_residual[q0 + i] -= JxW_face[q] * (vector_phi_face[i][q] * normals[q]) * lm_value;

      for (unsigned int i = 0; i != scalar_n_dofs; ++i)
      {
        ref_residual[u0 + i] +=
            JxW_face[q] * scalar_phi_face[i][q] *
            (-nu * q_normal + p_value * normals[q](0) + tau * (u_value - lm_value) * normal_norm2);
        ref_residual[p0 + i] += JxW_face[q] * scalar_phi_face[i][q] * lm_value * normals[q](0);

        for (unsigned int j = 0; j != vector_n_dofs; ++j)
          add_ref(off_Juq,
                  vector_n_dofs,
                  i,
                  j,
                  -JxW_face[q] * nu * scalar_phi_face[i][q] * (vector_phi_face[j][q] * normals[q]));

        for (unsigned int j = 0; j != scalar_n_dofs; ++j)
          add_ref(off_Jup,
                  scalar_n_dofs,
                  i,
                  j,
                  JxW_face[q] * scalar_phi_face[i][q] * scalar_phi_face[j][q] * normals[q](0));

        for (unsigned int j = 0; j != lm_n_dofs; ++j)
        {
          add_ref(off_Jplm,
                  lm_n_dofs,
                  i,
                  j,
                  JxW_face[q] * scalar_phi_face[i][q] * lm_phi_face[j][q] * normals[q](0));
          add_ref(off_Julm,
                  lm_n_dofs,
                  i,
                  j,
                  -JxW_face[q] * scalar_phi_face[i][q] * tau * lm_phi_face[j][q] * normal_norm2);
        }

        for (unsigned int j = 0; j != scalar_n_dofs; ++j)
          add_ref(off_Juu,
                  scalar_n_dofs,
                  i,
                  j,
                  JxW_face[q] * scalar_phi_face[i][q] * tau * scalar_phi_face[j][q] * normal_norm2);
      }

      for (unsigned int i = 0; i != vector_n_dofs; ++i)
        for (unsigned int j = 0; j != lm_n_dofs; ++j)
          add_ref(off_Jqlm,
                  lm_n_dofs,
                  i,
                  j,
                  -JxW_face[q] * (vector_phi_face[i][q] * normals[q]) * lm_phi_face[j][q]);

      for (unsigned int i = 0; i != lm_n_dofs; ++i)
      {
        ref_residual[lm0 + i] +=
            JxW_face[q] * lm_phi_face[i][q] *
            (-nu * q_normal + p_value * normals[q](0) + tau * (u_value - lm_value) * normal_norm2);

        for (unsigned int j = 0; j != vector_n_dofs; ++j)
          add_ref(off_Jlmq,
                  vector_n_dofs,
                  i,
                  j,
                  -JxW_face[q] * nu * lm_phi_face[i][q] * (vector_phi_face[j][q] * normals[q]));

        for (unsigned int j = 0; j != scalar_n_dofs; ++j)
        {
          add_ref(off_Jlmp,
                  scalar_n_dofs,
                  i,
                  j,
                  JxW_face[q] * lm_phi_face[i][q] * scalar_phi_face[j][q] * normals[q](0));
          add_ref(off_Jlms,
                  scalar_n_dofs,
                  i,
                  j,
                  JxW_face[q] * lm_phi_face[i][q] * tau * scalar_phi_face[j][q] * normal_norm2);
        }

        for (unsigned int j = 0; j != lm_n_dofs; ++j)
          add_ref(off_Jlmlm,
                  lm_n_dofs,
                  i,
                  j,
                  -JxW_face[q] * lm_phi_face[i][q] * tau * lm_phi_face[j][q] * normal_norm2);
      }
    }

    auto d_parent_xi = upload_real(parent_xi_h, "hdg_linear_parent_xi");
    auto d_parent_eta = upload_real(parent_eta_h, "hdg_linear_parent_eta");
    auto d_parent_zeta = upload_real(parent_zeta_h, "hdg_linear_parent_zeta");
    auto d_side_jxw = upload_real(side_jxw_h, "hdg_linear_side_jxw");
    auto d_normal_x = upload_real(normal_x_h, "hdg_linear_normal_x");
    auto d_normal_y = upload_real(normal_y_h, "hdg_linear_normal_y");
    auto d_normal_z = upload_real(normal_z_h, "hdg_linear_normal_z");

    const bool positive_edge_orientation =
        static_cast<bool>(edge_orientations(elem_index, side_id));
    libMesh::Kokkos::detail::add_hdg_linear_side_blocks(vector_key,
                                                        scalar_key,
                                                        lm_key,
                                                        layout,
                                                        side_id,
                                                        positive_edge_orientation,
                                                        d_parent_xi,
                                                        d_parent_eta,
                                                        d_parent_zeta,
                                                        d_side_jxw,
                                                        d_normal_x,
                                                        d_normal_y,
                                                        d_normal_z,
                                                        nqp,
                                                        nu,
                                                        tau,
                                                        d_blocks,
                                                        "hdg_linear_side_blocks");
    Kokkos::fence();

    libMesh::Kokkos::detail::add_hdg_linear_side_residual(vector_key,
                                                          scalar_key,
                                                          lm_key,
                                                          layout,
                                                          side_id,
                                                          positive_edge_orientation,
                                                          d_parent_xi,
                                                          d_parent_eta,
                                                          d_parent_zeta,
                                                          d_side_jxw,
                                                          d_normal_x,
                                                          d_normal_y,
                                                          d_normal_z,
                                                          nqp,
                                                          nu,
                                                          tau,
                                                          d_q_dofs,
                                                          d_u_dofs,
                                                          d_p_dofs,
                                                          d_lm_dofs,
                                                          d_residual,
                                                          "hdg_linear_side_residual");
    Kokkos::fence();
  }

  int fail = compare_device_values(d_blocks, ref, 1.0e-11);
  if (fail)
    std::printf("  HDG linear block mismatch: %s elem=%llu (%d failures)\n",
                info.name,
                libMesh::cast_int<unsigned long long>(elem_id),
                fail);

  const int assembled_residual_fail = compare_device_values(d_residual, ref_residual, 1.0e-11);
  if (assembled_residual_fail)
  {
    std::printf("  HDG linear residual mismatch: %s elem=%llu (%d failures)\n",
                info.name,
                libMesh::cast_int<unsigned long long>(elem_id),
                assembled_residual_fail);
    fail += assembled_residual_fail;
  }

  std::vector<Real> ref_condensed;
  const int ref_condense_status = condense_hdg_linear_reference(layout, ref, ref_condensed);
  const unsigned int condensed_n_dofs =
      libMesh::Kokkos::detail::hdg_linear_condensed_external_n_dofs(layout);

  Kokkos::View<double *> d_condensed(std::string("hdg_linear_condensed"),
                                     condensed_n_dofs * condensed_n_dofs);
  Kokkos::View<int> d_condense_status(std::string("hdg_linear_condense_status"));
  Kokkos::deep_copy(d_condensed, 0.);
  Kokkos::deep_copy(d_condense_status, 0);

  libMesh::Kokkos::detail::condense_hdg_linear_qu_blocks(
      layout, d_blocks, d_condensed, d_condense_status, "hdg_linear_condense");
  Kokkos::fence();

  int condense_status = 0;
  Kokkos::deep_copy(condense_status, d_condense_status);

  if (condense_status || ref_condense_status)
  {
    std::printf("  HDG condensed block failed: %s elem=%llu (device=%d host=%d)\n",
                info.name,
                libMesh::cast_int<unsigned long long>(elem_id),
                condense_status,
                ref_condense_status);
    ++fail;
  }
  else
  {
    const int condensed_fail = compare_device_values(d_condensed, ref_condensed, 1.0e-10);
    if (condensed_fail)
    {
      std::printf("  HDG condensed block mismatch: %s elem=%llu (%d failures)\n",
                  info.name,
                  libMesh::cast_int<unsigned long long>(elem_id),
                  condensed_fail);
      fail += condensed_fail;
    }
  }

  std::vector<Real> ref_condensed_residual;
  const int ref_residual_status =
      condense_hdg_linear_residual_reference(layout, ref, ref_residual, ref_condensed_residual);

  Kokkos::View<double *> d_condensed_residual(std::string("hdg_linear_condensed_residual"),
                                              condensed_n_dofs);
  Kokkos::View<int> d_residual_status(std::string("hdg_linear_residual_status"));
  Kokkos::deep_copy(d_condensed_residual, 0.);
  Kokkos::deep_copy(d_residual_status, 0);

  libMesh::Kokkos::detail::condense_hdg_linear_qu_residual(layout,
                                                           d_blocks,
                                                           d_residual,
                                                           d_condensed_residual,
                                                           d_residual_status,
                                                           "hdg_linear_residual_condense");
  Kokkos::fence();

  int residual_status = 0;
  Kokkos::deep_copy(residual_status, d_residual_status);

  if (residual_status || ref_residual_status)
  {
    std::printf("  HDG condensed residual failed: %s elem=%llu (device=%d host=%d)\n",
                info.name,
                libMesh::cast_int<unsigned long long>(elem_id),
                residual_status,
                ref_residual_status);
    ++fail;
  }
  else
  {
    const int residual_fail =
        compare_device_values(d_condensed_residual, ref_condensed_residual, 1.0e-10);
    if (residual_fail)
    {
      std::printf("  HDG condensed residual mismatch: %s elem=%llu (%d failures)\n",
                  info.name,
                  libMesh::cast_int<unsigned long long>(elem_id),
                  residual_fail);
      fail += residual_fail;
    }
  }

  const unsigned int batch_n = 2;
  Kokkos::View<double ***> d_batch_coords(
      std::string("hdg_batch_coords"), batch_n, n_geom_nodes, LIBMESH_DIM);
  auto h_batch_coords = Kokkos::create_mirror_view(d_batch_coords);
  for (unsigned int b = 0; b != batch_n; ++b)
    for (unsigned int i = 0; i != n_geom_nodes; ++i)
      for (unsigned int c = 0; c != LIBMESH_DIM; ++c)
        h_batch_coords(b, i, c) = elem.point(i)(c);
  Kokkos::deep_copy(d_batch_coords, h_batch_coords);

  Kokkos::View<double **> d_batch_q_dofs(std::string("hdg_batch_q_dofs"), batch_n, vector_n_dofs);
  Kokkos::View<double **> d_batch_u_dofs(std::string("hdg_batch_u_dofs"), batch_n, scalar_n_dofs);
  Kokkos::View<double **> d_batch_p_dofs(std::string("hdg_batch_p_dofs"), batch_n, scalar_n_dofs);
  Kokkos::View<double **> d_batch_lm_dofs(std::string("hdg_batch_lm_dofs"), batch_n, lm_n_dofs);
  const auto fill_batch_dofs =
      [](Kokkos::View<double **> d_values, const std::vector<Real> & values)
  {
    auto h_values = Kokkos::create_mirror_view(d_values);
    for (std::size_t b = 0; b != d_values.extent(0); ++b)
      for (std::size_t i = 0; i != values.size(); ++i)
        h_values(b, i) = values[i];
    Kokkos::deep_copy(d_values, h_values);
  };
  fill_batch_dofs(d_batch_q_dofs, q_dof_values);
  fill_batch_dofs(d_batch_u_dofs, u_dof_values);
  fill_batch_dofs(d_batch_p_dofs, p_dof_values);
  fill_batch_dofs(d_batch_lm_dofs, lm_dof_values);

  Kokkos::View<double ***> d_batch_parent_xi(
      std::string("hdg_batch_parent_xi"), batch_n, n_sides, side_nqp);
  Kokkos::View<double ***> d_batch_parent_eta(
      std::string("hdg_batch_parent_eta"), batch_n, n_sides, side_nqp);
  Kokkos::View<double ***> d_batch_parent_zeta(
      std::string("hdg_batch_parent_zeta"), batch_n, n_sides, side_nqp);
  Kokkos::View<double ***> d_batch_side_jxw(
      std::string("hdg_batch_side_jxw"), batch_n, n_sides, side_nqp);
  Kokkos::View<double ***> d_batch_normal_x(
      std::string("hdg_batch_normal_x"), batch_n, n_sides, side_nqp);
  Kokkos::View<double ***> d_batch_normal_y(
      std::string("hdg_batch_normal_y"), batch_n, n_sides, side_nqp);
  Kokkos::View<double ***> d_batch_normal_z(
      std::string("hdg_batch_normal_z"), batch_n, n_sides, side_nqp);
  const auto fill_batch_sides =
      [n_sides, side_nqp](Kokkos::View<double ***> d_values, const std::vector<Real> & values)
  {
    auto h_values = Kokkos::create_mirror_view(d_values);
    for (std::size_t b = 0; b != d_values.extent(0); ++b)
      for (unsigned int side = 0; side != n_sides; ++side)
        for (unsigned int q = 0; q != side_nqp; ++q)
          h_values(b, side, q) = values[side * side_nqp + q];
    Kokkos::deep_copy(d_values, h_values);
  };
  fill_batch_sides(d_batch_parent_xi, batch_parent_xi_h);
  fill_batch_sides(d_batch_parent_eta, batch_parent_eta_h);
  fill_batch_sides(d_batch_parent_zeta, batch_parent_zeta_h);
  fill_batch_sides(d_batch_side_jxw, batch_side_jxw_h);
  fill_batch_sides(d_batch_normal_x, batch_normal_x_h);
  fill_batch_sides(d_batch_normal_y, batch_normal_y_h);
  fill_batch_sides(d_batch_normal_z, batch_normal_z_h);

  Kokkos::View<unsigned char **> d_batch_orientations(
      std::string("hdg_batch_orientations"), batch_n, n_sides);
  auto h_batch_orientations = Kokkos::create_mirror_view(d_batch_orientations);
  for (unsigned int b = 0; b != batch_n; ++b)
    for (unsigned int side = 0; side != n_sides; ++side)
      h_batch_orientations(b, side) = edge_orientations(elem_index, side);
  Kokkos::deep_copy(d_batch_orientations, h_batch_orientations);

  Kokkos::View<double **> d_batch_blocks(std::string("hdg_batch_blocks"), batch_n, n_entries);
  Kokkos::View<double **> d_batch_residual(
      std::string("hdg_batch_residual"), batch_n, residual_n_dofs);
  Kokkos::View<double **> d_batch_condensed(
      std::string("hdg_batch_condensed"), batch_n, condensed_n_dofs * condensed_n_dofs);
  Kokkos::View<double **> d_batch_condensed_residual(
      std::string("hdg_batch_condensed_residual"), batch_n, condensed_n_dofs);
  Kokkos::View<int *> d_batch_condense_status(std::string("hdg_batch_condense_status"), batch_n);
  Kokkos::View<int *> d_batch_residual_status(std::string("hdg_batch_residual_status"), batch_n);

  libMesh::Kokkos::detail::assemble_hdg_linear_batch<ExactTopo>(vector_key,
                                                                scalar_key,
                                                                lm_key,
                                                                layout,
                                                                d_batch_coords,
                                                                batch_n,
                                                                n_geom_nodes,
                                                                d_xi,
                                                                d_eta,
                                                                d_zeta,
                                                                d_jxw,
                                                                volume_nqp,
                                                                d_batch_parent_xi,
                                                                d_batch_parent_eta,
                                                                d_batch_parent_zeta,
                                                                d_batch_side_jxw,
                                                                d_batch_normal_x,
                                                                d_batch_normal_y,
                                                                d_batch_normal_z,
                                                                d_batch_orientations,
                                                                n_sides,
                                                                side_nqp,
                                                                nu,
                                                                tau,
                                                                d_batch_q_dofs,
                                                                d_batch_u_dofs,
                                                                d_batch_p_dofs,
                                                                d_batch_lm_dofs,
                                                                d_batch_blocks,
                                                                d_batch_residual,
                                                                d_batch_condensed,
                                                                d_batch_condensed_residual,
                                                                d_batch_condense_status,
                                                                d_batch_residual_status,
                                                                "hdg_linear_batch");
  Kokkos::fence();

  const int batch_block_fail = compare_device_rows(d_batch_blocks, ref, 1.0e-11);
  if (batch_block_fail)
  {
    std::printf("  HDG batch block mismatch: %s elem=%llu (%d failures)\n",
                info.name,
                libMesh::cast_int<unsigned long long>(elem_id),
                batch_block_fail);
    fail += batch_block_fail;
  }

  const int batch_residual_fail = compare_device_rows(d_batch_residual, ref_residual, 1.0e-11);
  if (batch_residual_fail)
  {
    std::printf("  HDG batch residual mismatch: %s elem=%llu (%d failures)\n",
                info.name,
                libMesh::cast_int<unsigned long long>(elem_id),
                batch_residual_fail);
    fail += batch_residual_fail;
  }

  const int batch_condense_status_fail = count_nonzero_statuses(d_batch_condense_status);
  const int batch_residual_status_fail = count_nonzero_statuses(d_batch_residual_status);
  if (batch_condense_status_fail || batch_residual_status_fail)
  {
    std::printf("  HDG batch condensation failed: %s elem=%llu (block=%d residual=%d)\n",
                info.name,
                libMesh::cast_int<unsigned long long>(elem_id),
                batch_condense_status_fail,
                batch_residual_status_fail);
    fail += batch_condense_status_fail + batch_residual_status_fail;
  }
  else
  {
    const int batch_condensed_fail = compare_device_rows(d_batch_condensed, ref_condensed, 1.0e-10);
    const int batch_condensed_residual_fail =
        compare_device_rows(d_batch_condensed_residual, ref_condensed_residual, 1.0e-10);
    if (batch_condensed_fail || batch_condensed_residual_fail)
    {
      std::printf("  HDG batch condensed mismatch: %s elem=%llu (block=%d residual=%d failures)\n",
                  info.name,
                  libMesh::cast_int<unsigned long long>(elem_id),
                  batch_condensed_fail,
                  batch_condensed_residual_fail);
      fail += batch_condensed_fail + batch_condensed_residual_fail;
    }
  }

  return fail;
}

static int
test_hdg_linear_block_case(const hdg_assembly_case & info)
{
  switch (info.parent_type)
  {
    case libMesh::TRI6:
      return test_hdg_linear_block_case<libMesh::TRI6>(info);
    case libMesh::TRI7:
      return test_hdg_linear_block_case<libMesh::TRI7>(info);
    case libMesh::QUAD8:
      return test_hdg_linear_block_case<libMesh::QUAD8>(info);
    case libMesh::QUAD9:
      return test_hdg_linear_block_case<libMesh::QUAD9>(info);
    default:
      std::printf("  unexpected unsupported HDG block parent type=%d\n",
                  static_cast<int>(info.parent_type));
      return 1;
  }
}

inline int
run_all_oracles()
{
  int total_fail = 0;

  for (const auto & info : hdg_assembly_cases)
  {
    const int f = test_hdg_linear_block_case(info);
    std::printf(
        "[hdg_linear_block]      [%s] %s  (%d failures)\n", info.name, f ? "FAIL" : "PASS", f);
    total_fail += f;
  }

  return total_fail ? 1 : 0;
}

} // namespace KokkosHDGAssemblyOracle
} // namespace libMeshTest
