// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

#include "libmesh/libmesh_config.h"

#if defined(LIBMESH_HAVE_KOKKOS) && defined(LIBMESH_HAVE_PETSC) &&                                 \
    !defined(LIBMESH_USE_COMPLEX_NUMBERS)
#include <petscvec_kokkos.hpp>
#endif

#include "hdg_problem.h"

#include "libmesh/boundary_info.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_shape_traits.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh_base.h"
#include "libmesh/nonlinear_implicit_system.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/quadrature.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/static_condensation.h"
#include "libmesh/system.h"

#ifdef LIBMESH_HAVE_PETSC
#include "libmesh/petsc_macro.h"
#include "libmesh/petsc_matrix_base.h"
#include "libmesh/petsc_vector.h"
#endif

#if defined(LIBMESH_HAVE_KOKKOS) && defined(LIBMESH_HAVE_PETSC) &&                                 \
    !defined(LIBMESH_USE_COMPLEX_NUMBERS)
#include "gpu/kokkos_fe_face_map.h"
#include "gpu/kokkos_hdg_assembly.h"

#include "libmesh/mesh_base_kokkos.h"

#include <Kokkos_Core.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace libMesh
{

namespace
{

bool
is_hdg_elem_type_supported(const ElemType elem_type)
{
  switch (elem_type)
  {
    case TRI6:
    case TRI7:
      return true;

    default:
      return false;
  }
}

FEShapeKey
variable_shape_key(const System & system, const char * name, const ElemType elem_type)
{
  const FEType fe_type = system.variable_type(system.variable_number(name));
  return {fe_type.family, elem_type, static_cast<Order>(fe_type.order.get_order())};
}

#if defined(LIBMESH_HAVE_KOKKOS) && defined(LIBMESH_HAVE_PETSC) &&                                 \
    !defined(LIBMESH_USE_COMPLEX_NUMBERS)

using DeviceReal1D = ::Kokkos::View<Real *>;
using DeviceReal2D = ::Kokkos::View<Real **>;
using DeviceReal3D = ::Kokkos::View<Real ***>;
using DeviceInt1D = ::Kokkos::View<int *>;
using DeviceInt2D = ::Kokkos::View<int **>;
using DeviceDofId1D = ::Kokkos::View<dof_id_type *>;
using DeviceDofId2D = ::Kokkos::View<dof_id_type **>;
using DeviceUChar2D = ::Kokkos::View<unsigned char **>;
using HDGLinearBlockLayout = Kokkos::detail::HDGLinearBlockLayout;

constexpr unsigned int ex9_kokkos_max_qp = 16;
constexpr unsigned int ex9_kokkos_max_side_qp = 8;
constexpr unsigned int ex9_kokkos_max_scalar_dofs = 8;
constexpr unsigned int ex9_kokkos_max_lm_dofs = 12;

bool
ex9_kokkos_timers_enabled()
{
  static const bool enabled = libMesh::on_command_line("--ex9-kokkos-timers");
  return enabled;
}

unsigned int
ex9_kokkos_team_size()
{
  static const unsigned int team_size =
      libMesh::command_line_value("--ex9-kokkos-team-size", 0u);
  return team_size;
}

class Ex9KokkosTimer
{
public:
  explicit Ex9KokkosTimer(std::string label, const bool fence = true)
    : _label(std::move(label)),
      _fence(fence),
      _enabled(ex9_kokkos_timers_enabled())
  {
    if (_enabled)
    {
      if (_fence)
        ::Kokkos::fence();
      _start = Clock::now();
    }
  }

  ~Ex9KokkosTimer()
  {
    if (!_enabled)
      return;

    if (_fence)
      ::Kokkos::fence();
    const auto stop = Clock::now();
    const std::chrono::duration<double> elapsed = stop - _start;
    std::cout << "[ex9 kokkos timer] " << _label << ": " << elapsed.count() << " s\n";
  }

private:
  using Clock = std::chrono::steady_clock;
  std::string _label;
  bool _fence = true;
  bool _enabled = false;
  Clock::time_point _start;
};

template <typename ViewType>
void
copy_vector_to_view(const std::vector<Real> & values, const ViewType & view)
{
  auto host = ::Kokkos::create_mirror_view(view);
  for (std::size_t i = 0; i != values.size(); ++i)
    host(i) = values[i];
  ::Kokkos::deep_copy(view, host);
}

template <typename ViewType>
void
copy_vector_to_side_view(const std::vector<Real> & values,
                         const unsigned int n_elem,
                         const unsigned int n_sides,
                         const unsigned int n_qp,
                         const ViewType & view)
{
  auto host = ::Kokkos::create_mirror_view(view);
  for (unsigned int e = 0; e != n_elem; ++e)
    for (unsigned int s = 0; s != n_sides; ++s)
      for (unsigned int q = 0; q != n_qp; ++q)
        host(e, s, q) = values[(e * n_sides + s) * n_qp + q];
  ::Kokkos::deep_copy(view, host);
}

struct Ex9FullBlockLayout
{
  unsigned int q_n_dofs = 0;
  unsigned int scalar_n_dofs = 0;
  unsigned int lm_n_dofs = 0;
  unsigned int off_qu = 0;
  unsigned int off_qv = 0;
  unsigned int off_u = 0;
  unsigned int off_v = 0;
  unsigned int off_p = 0;
  unsigned int off_lm_u = 0;
  unsigned int off_lm_v = 0;
  unsigned int external_off_p = 0;
  unsigned int external_off_lm_u = 0;
  unsigned int external_off_lm_v = 0;
  unsigned int internal_n_dofs = 0;
  unsigned int external_n_dofs = 0;
  unsigned int n_dofs = 0;
  unsigned int n_entries = 0;
  unsigned int condensed_n_entries = 0;
};

Ex9FullBlockLayout
make_ex9_full_block_layout(const unsigned int q_n_dofs,
                           const unsigned int scalar_n_dofs,
                           const unsigned int lm_n_dofs)
{
  Ex9FullBlockLayout layout;
  layout.q_n_dofs = q_n_dofs;
  layout.scalar_n_dofs = scalar_n_dofs;
  layout.lm_n_dofs = lm_n_dofs;
  layout.off_qv = layout.off_qu + q_n_dofs;
  layout.off_u = layout.off_qv + q_n_dofs;
  layout.off_v = layout.off_u + scalar_n_dofs;
  layout.off_p = layout.off_v + scalar_n_dofs;
  layout.off_lm_u = layout.off_p + scalar_n_dofs;
  layout.off_lm_v = layout.off_lm_u + lm_n_dofs;
  layout.external_off_lm_u = layout.external_off_p + scalar_n_dofs;
  layout.external_off_lm_v = layout.external_off_lm_u + lm_n_dofs;
  layout.internal_n_dofs = 2 * q_n_dofs + 2 * scalar_n_dofs;
  layout.external_n_dofs = scalar_n_dofs + 2 * lm_n_dofs;
  layout.n_dofs = layout.internal_n_dofs + layout.external_n_dofs;
  layout.n_entries = layout.n_dofs * layout.n_dofs;
  layout.condensed_n_entries = layout.external_n_dofs * layout.external_n_dofs;
  return layout;
}

struct DeviceBackSubstitutionPlan
{
  Ex9FullBlockLayout layout;
  std::size_t n_elem = 0;
  DeviceReal2D full_blocks;
  DeviceReal2D internal_inverse;
  DeviceReal2D internal_external_solve;
  DeviceDofId2D internal_dof_ids;
  DeviceDofId2D external_dof_ids;
  DeviceDofId2D internal_vector_ids;
  DeviceDofId2D external_vector_ids;
  dof_id_type first_local_dof = 0;
  dof_id_type last_local_dof = 0;
  dof_id_type n_local_dofs = 0;
  std::vector<numeric_index_type> ghost_dofs;
  std::size_t forward_base_n_entries = 0;
  DeviceDofId1D forward_base_vector_ids;
  std::vector<PetscInt> forward_coo_rows;
  DeviceReal1D forward_coo_values;
  Vec forward_coo_target = nullptr;
  std::vector<PetscInt> coo_rows;
  DeviceReal1D coo_values;
  Vec coo_target = nullptr;
  std::unique_ptr<PetscVector<Number>> ghosted_rhs;
  std::unique_ptr<PetscVector<Number>> ghosted_sol;
};

struct Ex9ResidualCooPlan
{
  std::vector<PetscInt> rows;
  DeviceReal1D values;
  Vec target = nullptr;
};

struct Ex9CondensedJacobianCooPlan
{
  bool initialized = false;
  std::vector<PetscInt> rows;
  std::vector<PetscInt> cols;
  DeviceReal1D values;
  Mat target = nullptr;
};

template <typename RhsView, typename SolView>
void
fill_back_substitution_coo_values(DeviceBackSubstitutionPlan & plan,
                                  const RhsView & rhs_view,
                                  const SolView & sol_view)
{
  const auto layout = plan.layout;
  const std::size_t n_entries = plan.n_elem * layout.internal_n_dofs;
  DeviceReal1D coo_values = plan.coo_values;
  const auto internal_ids = plan.internal_vector_ids;
  const auto external_ids = plan.external_vector_ids;
  const auto internal_inverse = plan.internal_inverse;
  const auto internal_external_solve = plan.internal_external_solve;
  const unsigned int n_internal = layout.internal_n_dofs;
  const unsigned int n_external = layout.external_n_dofs;

  Ex9KokkosTimer timer("back-substitution values kernel");
  ::Kokkos::parallel_for(
      "vector_fe_ex9_backwards_substitution",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_entries)),
      KOKKOS_LAMBDA(const int entry) {
        const int elem = entry / static_cast<int>(n_internal);
        const int i = entry - elem * static_cast<int>(n_internal);
        Real value = Real(0);
        for (unsigned int j = 0; j != n_internal; ++j)
          value += internal_inverse(elem, i * n_internal + j) * rhs_view(internal_ids(elem, j));
        for (unsigned int j = 0; j != n_external; ++j)
          value -=
              internal_external_solve(elem, i * n_external + j) * sol_view(external_ids(elem, j));
        coo_values(entry) = value;
      });
}

template <typename RhsView>
void
fill_forward_elimination_coo_values(DeviceBackSubstitutionPlan & plan, const RhsView & rhs_view)
{
  const auto layout = plan.layout;
  const std::size_t n_base_entries = plan.forward_base_n_entries;
  const std::size_t n_entries = n_base_entries + plan.n_elem * layout.external_n_dofs;
  DeviceReal1D coo_values = plan.forward_coo_values;
  const auto full_blocks = plan.full_blocks;
  const auto internal_inverse = plan.internal_inverse;
  const auto internal_ids = plan.internal_vector_ids;
  const auto base_ids = plan.forward_base_vector_ids;
  const unsigned int n_internal = layout.internal_n_dofs;
  const unsigned int n_external = layout.external_n_dofs;
  const unsigned int n_dofs = layout.n_dofs;
  const unsigned int external_offset = layout.off_p;
  const std::size_t base_entries = n_base_entries;

  Ex9KokkosTimer timer("forward elimination values kernel");
  ::Kokkos::parallel_for(
      "vector_fe_ex9_forward_elimination",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_entries)),
      KOKKOS_LAMBDA(const int entry) {
        if (static_cast<std::size_t>(entry) < base_entries)
        {
          coo_values(entry) = rhs_view(base_ids(entry));
          return;
        }

        const int correction_entry = entry - static_cast<int>(base_entries);
        const int elem = correction_entry / static_cast<int>(n_external);
        const int i = correction_entry - elem * static_cast<int>(n_external);
        Real value = Real(0);

        for (unsigned int j = 0; j != n_internal; ++j)
        {
          Real solved_rhs = Real(0);
          for (unsigned int k = 0; k != n_internal; ++k)
            solved_rhs +=
                internal_inverse(elem, j * n_internal + k) * rhs_view(internal_ids(elem, k));
          value -= full_blocks(elem, (external_offset + i) * n_dofs + j) * solved_rhs;
        }

        coo_values(entry) = value;
      });
}

struct Ex9FullBatch
{
  Ex9FullBlockLayout layout;
  std::vector<const Elem *> elems;
  std::vector<std::vector<dof_id_type>> qu_indices;
  std::vector<std::vector<dof_id_type>> qv_indices;
  std::vector<std::vector<dof_id_type>> u_indices;
  std::vector<std::vector<dof_id_type>> v_indices;
  std::vector<std::vector<dof_id_type>> p_indices;
  std::vector<std::vector<dof_id_type>> lm_u_indices;
  std::vector<std::vector<dof_id_type>> lm_v_indices;
  std::vector<std::vector<dof_id_type>> global_lm_indices;
  DeviceReal2D device_residual;
  DeviceReal2D device_condensed_blocks;
  DeviceReal2D device_p_dofs;
  DeviceReal2D device_global_lm_dofs;
  std::shared_ptr<DeviceBackSubstitutionPlan> backsub_plan;
  FEShapeKey scalar_key{INVALID_FE, INVALID_ELEM, INVALID_ORDER};
  unsigned int n_qp = 0;
  DeviceReal1D xi, eta, zeta, jxw;
};

enum class AssemblyPurpose : unsigned char
{
  residual,
  jacobian
};

struct DeviceAssemblyData
{
  const HDGProblem * problem = nullptr;
  const NonlinearImplicitSystem * system = nullptr;
  AssemblyPurpose purpose = AssemblyPurpose::residual;
  ElemType elem_type = INVALID_ELEM;
  unsigned int q_var = invalid_uint;
  unsigned int u_var = invalid_uint;
  unsigned int lm_var = invalid_uint;
  unsigned int p_var = invalid_uint;
  unsigned int other_u_var = invalid_uint;
  unsigned int other_lm_var = invalid_uint;
  unsigned int velocity_component = invalid_uint;
  unsigned int n_elem = 0;
  unsigned int n_geom_nodes = 0;
  unsigned int n_sides = 0;
  unsigned int n_qp = 0;
  unsigned int n_side_qp = 0;
  unsigned int residual_n_dofs = 0;
  FEShapeKey vector_key{INVALID_FE, INVALID_ELEM, INVALID_ORDER};
  FEShapeKey scalar_key{INVALID_FE, INVALID_ELEM, INVALID_ORDER};
  FEShapeKey lm_key{INVALID_FE, INVALID_ELEM, INVALID_ORDER};
  HDGLinearBlockLayout layout;
  std::vector<const Elem *> elems;
  std::vector<std::vector<dof_id_type>> q_indices;
  std::vector<std::vector<dof_id_type>> u_indices;
  std::vector<std::vector<dof_id_type>> p_indices;
  std::vector<std::vector<dof_id_type>> lm_indices;
  std::vector<std::vector<dof_id_type>> other_u_indices;
  std::vector<std::vector<dof_id_type>> other_lm_indices;
  DeviceDofId2D q_dof_ids, u_dof_ids, p_dof_ids, lm_dof_ids, other_u_dof_ids, other_lm_dof_ids;
  DeviceReal3D coords;
  DeviceReal1D xi, eta, zeta, jxw;
  DeviceReal2D scalar_forcing, pressure_forcing;
  DeviceReal3D parent_xi, parent_eta, parent_zeta, side_jxw, normal_x, normal_y, normal_z;
  DeviceReal3D dirichlet_component, dirichlet_vdotn;
  DeviceUChar2D orientations;
  DeviceInt2D side_types;
};

struct Ex9DeviceAssemblyData
{
  const HDGProblem * problem = nullptr;
  const NonlinearImplicitSystem * system = nullptr;
  ElemType elem_type = INVALID_ELEM;
  unsigned int n_elem = 0;
  Ex9FullBlockLayout layout;
  std::vector<const Elem *> elems;
  std::vector<std::vector<dof_id_type>> qu_indices;
  std::vector<std::vector<dof_id_type>> qv_indices;
  std::vector<std::vector<dof_id_type>> u_indices;
  std::vector<std::vector<dof_id_type>> v_indices;
  std::vector<std::vector<dof_id_type>> p_indices;
  std::vector<std::vector<dof_id_type>> lm_u_indices;
  std::vector<std::vector<dof_id_type>> lm_v_indices;
  std::vector<std::vector<dof_id_type>> global_lm_indices;
  DeviceDofId2D qu_dof_ids, qv_dof_ids, u_dof_ids, v_dof_ids, p_dof_ids, lm_u_dof_ids, lm_v_dof_ids,
      global_lm_dof_ids;
};

struct Ex9DeviceWork
{
  Ex9DeviceWork(const Ex9DeviceAssemblyData & data)
    : qu_dofs("vector_fe_ex9_full_qu_dofs", data.n_elem, data.layout.q_n_dofs),
      qv_dofs("vector_fe_ex9_full_qv_dofs", data.n_elem, data.layout.q_n_dofs),
      u_dofs("vector_fe_ex9_full_u_dofs", data.n_elem, data.layout.scalar_n_dofs),
      v_dofs("vector_fe_ex9_full_v_dofs", data.n_elem, data.layout.scalar_n_dofs),
      p_dofs("vector_fe_ex9_full_p_dofs", data.n_elem, data.layout.scalar_n_dofs),
      lm_u_dofs("vector_fe_ex9_full_lm_u_dofs", data.n_elem, data.layout.lm_n_dofs),
      lm_v_dofs("vector_fe_ex9_full_lm_v_dofs", data.n_elem, data.layout.lm_n_dofs),
      global_lm_dofs("vector_fe_ex9_full_global_lm_dofs", data.n_elem, 1)
  {
    ::Kokkos::deep_copy(qu_dofs, Real(0));
    ::Kokkos::deep_copy(qv_dofs, Real(0));
    ::Kokkos::deep_copy(u_dofs, Real(0));
    ::Kokkos::deep_copy(v_dofs, Real(0));
    ::Kokkos::deep_copy(p_dofs, Real(0));
    ::Kokkos::deep_copy(lm_u_dofs, Real(0));
    ::Kokkos::deep_copy(lm_v_dofs, Real(0));
    ::Kokkos::deep_copy(global_lm_dofs, Real(0));
  }

  DeviceReal2D qu_dofs, qv_dofs, u_dofs, v_dofs, p_dofs, lm_u_dofs, lm_v_dofs, global_lm_dofs;
};

struct ComponentDofViews
{
  DeviceReal2D q_dofs, u_dofs, p_dofs, lm_dofs, other_u_dofs, other_lm_dofs;
};

struct ComponentBatchWork
{
  ComponentBatchWork(const DeviceAssemblyData & data)
    : blocks("vector_fe_ex9_hdg_component_blocks", data.n_elem, data.layout.n_entries),
      residual("vector_fe_ex9_hdg_component_residual", data.n_elem, data.residual_n_dofs),
      condense_status("vector_fe_ex9_hdg_condense_status", data.n_elem),
      residual_status("vector_fe_ex9_hdg_residual_status", data.n_elem)
  {
  }

  DeviceReal2D blocks, residual;
  DeviceInt1D condense_status, residual_status;
};

struct Ex9FullBatchWork
{
  Ex9FullBatchWork(const Ex9DeviceAssemblyData & data)
    : device_blocks("vector_fe_ex9_full_blocks", data.n_elem, data.layout.n_entries),
      device_residual("vector_fe_ex9_full_residual", data.n_elem, data.layout.n_dofs),
      condensed_blocks(
          "vector_fe_ex9_full_condensed_blocks", data.n_elem, data.layout.condensed_n_entries),
      internal_inverse("vector_fe_ex9_full_internal_inverse",
                       data.n_elem,
                       data.layout.internal_n_dofs * data.layout.internal_n_dofs),
      internal_external_solve("vector_fe_ex9_full_internal_external_solve",
                              data.n_elem,
                              data.layout.internal_n_dofs * data.layout.external_n_dofs),
      condense_status("vector_fe_ex9_full_condense_status", data.n_elem),
      failure_pivot("vector_fe_ex9_full_condense_failure_pivot", data.n_elem),
      failure_pivot_abs("vector_fe_ex9_full_condense_failure_pivot_abs", data.n_elem),
      failure_initial_diag("vector_fe_ex9_full_condense_failure_initial_diag", data.n_elem),
      failure_initial_row_abs("vector_fe_ex9_full_condense_failure_initial_row_abs", data.n_elem)
  {
  }

  DeviceReal2D device_blocks, device_residual, condensed_blocks, internal_inverse,
      internal_external_solve;
  DeviceInt1D condense_status, failure_pivot;
  DeviceReal1D failure_pivot_abs, failure_initial_diag, failure_initial_row_abs;
};

void
copy_dof_indices_to_view(const std::vector<std::vector<dof_id_type>> & indices,
                         const DeviceDofId2D & view)
{
  auto host = ::Kokkos::create_mirror_view(view);
  for (std::size_t e = 0; e != indices.size(); ++e)
    for (std::size_t i = 0; i != indices[e].size(); ++i)
      host(e, i) = indices[e][i];
  ::Kokkos::deep_copy(view, host);
}

void
copy_dof_indices_to_view(const std::vector<dof_id_type> & indices, const DeviceDofId1D & view)
{
  auto host = ::Kokkos::create_mirror_view(view);
  for (std::size_t i = 0; i != indices.size(); ++i)
    host(i) = indices[i];
  ::Kokkos::deep_copy(view, host);
}

bool
collect_supported_elements(const HDGProblem & problem,
                           std::vector<const Elem *> & elems,
                           ElemType & elem_type)
{
  elems.clear();
  elem_type = INVALID_ELEM;

  for (const Elem * elem : problem.mesh->active_local_element_ptr_range())
  {
    if (!is_hdg_elem_type_supported(elem->type()))
      return false;

    if (elem_type == INVALID_ELEM)
      elem_type = elem->type();
    else if (elem_type != elem->type())
      return false;

    elems.push_back(elem);
  }

  return elem_type != INVALID_ELEM;
}

std::unique_ptr<DeviceAssemblyData>
make_device_assembly_data(NonlinearImplicitSystem & system,
                          HDGProblem & problem,
                          const AssemblyPurpose purpose,
                          const std::vector<const Elem *> & elems,
                          const ElemType elem_type,
                          const unsigned int q_var,
                          const unsigned int u_var,
                          const unsigned int lm_var,
                          const unsigned int p_var,
                          const unsigned int other_u_var,
                          const unsigned int other_lm_var,
                          const unsigned int velocity_component)
{
  const unsigned int n_elem = cast_int<unsigned int>(elems.size());
  auto data_ptr = std::make_unique<DeviceAssemblyData>();
  DeviceAssemblyData & data = *data_ptr;
  data.problem = &problem;
  data.system = &system;
  data.purpose = purpose;
  data.elem_type = elem_type;
  data.q_var = q_var;
  data.u_var = u_var;
  data.lm_var = lm_var;
  data.p_var = p_var;
  data.other_u_var = other_u_var;
  data.other_lm_var = other_lm_var;
  data.velocity_component = velocity_component;
  data.n_elem = n_elem;
  DeviceAssemblyData & cache = data;

  const Elem & first_elem = *elems.front();
  cache.n_geom_nodes = first_elem.n_nodes();
  cache.n_sides = first_elem.n_sides();
  cache.elems = elems;

  cache.q_indices.resize(n_elem);
  cache.u_indices.resize(n_elem);
  cache.p_indices.resize(n_elem);
  cache.lm_indices.resize(n_elem);
  cache.other_u_indices.resize(n_elem);
  cache.other_lm_indices.resize(n_elem);

  const DofMap & dof_map = system.get_dof_map();
  for (unsigned int e = 0; e != n_elem; ++e)
  {
    dof_map.dof_indices(elems[e], cache.q_indices[e], q_var);
    dof_map.dof_indices(elems[e], cache.u_indices[e], u_var);
    dof_map.dof_indices(elems[e], cache.p_indices[e], p_var);
    dof_map.dof_indices(elems[e], cache.lm_indices[e], lm_var);
    dof_map.dof_indices(elems[e], cache.other_u_indices[e], other_u_var);
    dof_map.dof_indices(elems[e], cache.other_lm_indices[e], other_lm_var);
  }

  cache.layout = Kokkos::detail::make_hdg_linear_block_layout(
      cast_int<unsigned int>(cache.q_indices.front().size()),
      cast_int<unsigned int>(cache.u_indices.front().size()),
      cast_int<unsigned int>(cache.lm_indices.front().size()));
  cache.residual_n_dofs = Kokkos::detail::hdg_linear_n_dofs(cache.layout);
  cache.vector_key = variable_shape_key(system, system.variable_name(q_var).c_str(), elem_type);
  cache.scalar_key = variable_shape_key(system, system.variable_name(u_var).c_str(), elem_type);
  cache.lm_key = variable_shape_key(system, system.variable_name(lm_var).c_str(), elem_type);

  problem.vector_fe->reinit(&first_elem);
  const auto & first_jxw = problem.vector_fe->get_JxW();
  cache.n_qp = cast_int<unsigned int>(first_jxw.size());
  std::vector<Real> xi(cache.n_qp), eta(cache.n_qp), zeta(cache.n_qp), jxw(cache.n_qp);
  std::vector<Real> scalar_forcing(n_elem * cache.n_qp, Real(0));
  std::vector<Real> pressure_forcing(n_elem * cache.n_qp, Real(0));
  for (unsigned int q = 0; q != cache.n_qp; ++q)
  {
    const Point & qp = problem.qrule->qp(q);
    xi[q] = qp(0);
    eta[q] = qp(1);
    zeta[q] = 0;
    jxw[q] = first_jxw[q];
  }

  for (const Elem * elem : elems)
  {
    problem.vector_fe->reinit(elem);
    const auto & elem_jxw = problem.vector_fe->get_JxW();
    for (unsigned int q = 0; q != cache.n_qp; ++q)
      libmesh_error_msg_if(std::abs(elem_jxw[q] - jxw[q]) > 1.e-12,
                           "vector_fe_ex9 Kokkos HDG assembly currently requires identical "
                           "volume JxW values across a batch. elem="
                               << elem->id() << " qp=" << q << " expected=" << jxw[q]
                               << " actual=" << elem_jxw[q]);
  }

  if (problem.mms)
    for (unsigned int e = 0; e != n_elem; ++e)
    {
      problem.vector_fe->reinit(elems[e]);
      const auto & q_points = problem.vector_fe->get_xyz();
      for (unsigned int q = 0; q != cache.n_qp; ++q)
      {
        scalar_forcing[e * cache.n_qp + q] = velocity_component == 0
                                                 ? problem.u_true_soln.forcing(q_points[q])
                                                 : problem.v_true_soln.forcing(q_points[q]);
        pressure_forcing[e * cache.n_qp + q] = problem.p_true_soln.forcing(q_points[q]);
      }
    }

  std::vector<Real> coords(n_elem * cache.n_geom_nodes * 3);
  for (unsigned int e = 0; e != n_elem; ++e)
    for (unsigned int n = 0; n != cache.n_geom_nodes; ++n)
      for (unsigned int d = 0; d != 3; ++d)
        coords[(e * cache.n_geom_nodes + n) * 3 + d] = elems[e]->point(n)(d);

  problem.vector_fe_face->reinit(&first_elem, static_cast<unsigned int>(0));
  cache.n_side_qp = cast_int<unsigned int>(problem.vector_fe_face->get_JxW().size());
  const std::size_t side_values_size =
      static_cast<std::size_t>(n_elem) * cache.n_sides * cache.n_side_qp;
  std::vector<Real> parent_xi(side_values_size), parent_eta(side_values_size),
      parent_zeta(side_values_size), side_jxw(side_values_size), normal_x(side_values_size),
      normal_y(side_values_size), normal_z(side_values_size),
      dirichlet_component(side_values_size, Real(0)), dirichlet_vdotn(side_values_size, Real(0));
  std::vector<unsigned char> orientations(n_elem * cache.n_sides, 1);
  std::vector<int> side_types(n_elem * cache.n_sides, Kokkos::detail::hdg_linear_side_interior);
  std::vector<boundary_id_type> boundary_ids;
  const BoundaryInfo & boundary_info = problem.mesh->get_boundary_info();
  const auto & geometry_cache = problem.mesh->get_kokkos_geometry_cache();
  const auto h_edge_orientations =
      geometry_cache.element_edge_positive_orientations_dual.view_host();

  for (unsigned int e = 0; e != n_elem; ++e)
    for (unsigned int s = 0; s != cache.n_sides; ++s)
    {
      auto side = elems[e]->build_side_ptr(s);
      problem.vector_fe_face->reinit(elems[e], s);
      const auto & elem_side_jxw = problem.vector_fe_face->get_JxW();
      const auto & normals = problem.vector_fe_face->get_normals();
      if (!elems[e]->neighbor_ptr(s))
      {
        boundary_ids.clear();
        boundary_info.boundary_ids(elems[e], s, boundary_ids);
        libmesh_assert_equal_to(boundary_ids.size(), 1);
        side_types[e * cache.n_sides + s] =
            (!problem.cavity && boundary_ids[0] == problem.right_bnd)
                ? Kokkos::detail::hdg_linear_side_outlet
                : Kokkos::detail::hdg_linear_side_dirichlet;
      }
      orientations[e * cache.n_sides + s] =
          h_edge_orientations(problem.mesh->get_kokkos_elem_index(*elems[e]), s);

      for (unsigned int q = 0; q != cache.n_side_qp; ++q)
      {
        const Point & face_qp_point = problem.qface->qp(q);
        const Kokkos::RealVector face_qp = Kokkos::make_vector(face_qp_point(0), 0, 0);
        const Kokkos::RealVector parent_qp =
            Kokkos::map_face_qp_to_parent(*side, LAGRANGE_MAP, side->type(), face_qp);
        const std::size_t idx = (e * cache.n_sides + s) * cache.n_side_qp + q;
        parent_xi[idx] = parent_qp(0);
        parent_eta[idx] = parent_qp(1);
        parent_zeta[idx] = parent_qp(2);
        side_jxw[idx] = elem_side_jxw[q];
        normal_x[idx] = normals[q](0);
        normal_y[idx] = normals[q](1);
        normal_z[idx] = normals[q](2);

        if (side_types[e * cache.n_sides + s] == Kokkos::detail::hdg_linear_side_dirichlet)
        {
          const Point & qp_point = problem.vector_fe_face->get_xyz()[q];
          Real dirichlet_u = Real(0);
          Real dirichlet_v = Real(0);
          if (problem.mms)
          {
            dirichlet_u = problem.u_true_soln(qp_point);
            dirichlet_v = problem.v_true_soln(qp_point);
          }
          else if (problem.cavity)
            dirichlet_u = boundary_ids[0] == problem.top_bnd ? Real(1) : Real(0);
          else
            dirichlet_u = boundary_ids[0] == problem.left_bnd ? Real(1) : Real(0);
          dirichlet_component[idx] = velocity_component == 0 ? dirichlet_u : dirichlet_v;
          dirichlet_vdotn[idx] = dirichlet_u * normal_x[idx] + dirichlet_v * normal_y[idx];
        }
      }
    }

  cache.q_dof_ids =
      DeviceDofId2D("vector_fe_ex9_hdg_q_dof_ids", n_elem, cache.layout.vector_n_dofs);
  cache.u_dof_ids =
      DeviceDofId2D("vector_fe_ex9_hdg_u_dof_ids", n_elem, cache.layout.scalar_n_dofs);
  cache.p_dof_ids =
      DeviceDofId2D("vector_fe_ex9_hdg_p_dof_ids", n_elem, cache.layout.scalar_n_dofs);
  cache.lm_dof_ids = DeviceDofId2D("vector_fe_ex9_hdg_lm_dof_ids", n_elem, cache.layout.lm_n_dofs);
  cache.other_u_dof_ids =
      DeviceDofId2D("vector_fe_ex9_hdg_other_u_dof_ids", n_elem, cache.layout.scalar_n_dofs);
  cache.other_lm_dof_ids =
      DeviceDofId2D("vector_fe_ex9_hdg_other_lm_dof_ids", n_elem, cache.layout.lm_n_dofs);
  copy_dof_indices_to_view(cache.q_indices, cache.q_dof_ids);
  copy_dof_indices_to_view(cache.u_indices, cache.u_dof_ids);
  copy_dof_indices_to_view(cache.p_indices, cache.p_dof_ids);
  copy_dof_indices_to_view(cache.lm_indices, cache.lm_dof_ids);
  copy_dof_indices_to_view(cache.other_u_indices, cache.other_u_dof_ids);
  copy_dof_indices_to_view(cache.other_lm_indices, cache.other_lm_dof_ids);

  cache.coords = DeviceReal3D("vector_fe_ex9_hdg_coords", n_elem, cache.n_geom_nodes, 3);
  auto h_coords = ::Kokkos::create_mirror_view(cache.coords);
  for (unsigned int e = 0; e != n_elem; ++e)
    for (unsigned int n = 0; n != cache.n_geom_nodes; ++n)
      for (unsigned int d = 0; d != 3; ++d)
        h_coords(e, n, d) = coords[(e * cache.n_geom_nodes + n) * 3 + d];
  ::Kokkos::deep_copy(cache.coords, h_coords);

  cache.xi = DeviceReal1D("vector_fe_ex9_hdg_xi", cache.n_qp);
  cache.eta = DeviceReal1D("vector_fe_ex9_hdg_eta", cache.n_qp);
  cache.zeta = DeviceReal1D("vector_fe_ex9_hdg_zeta", cache.n_qp);
  cache.jxw = DeviceReal1D("vector_fe_ex9_hdg_jxw", cache.n_qp);
  copy_vector_to_view(xi, cache.xi);
  copy_vector_to_view(eta, cache.eta);
  copy_vector_to_view(zeta, cache.zeta);
  copy_vector_to_view(jxw, cache.jxw);

  cache.scalar_forcing = DeviceReal2D("vector_fe_ex9_hdg_scalar_forcing", n_elem, cache.n_qp);
  cache.pressure_forcing = DeviceReal2D("vector_fe_ex9_hdg_pressure_forcing", n_elem, cache.n_qp);
  auto h_scalar_forcing = ::Kokkos::create_mirror_view(cache.scalar_forcing);
  auto h_pressure_forcing = ::Kokkos::create_mirror_view(cache.pressure_forcing);
  for (unsigned int e = 0; e != n_elem; ++e)
    for (unsigned int q = 0; q != cache.n_qp; ++q)
    {
      h_scalar_forcing(e, q) = scalar_forcing[e * cache.n_qp + q];
      h_pressure_forcing(e, q) = pressure_forcing[e * cache.n_qp + q];
    }
  ::Kokkos::deep_copy(cache.scalar_forcing, h_scalar_forcing);
  ::Kokkos::deep_copy(cache.pressure_forcing, h_pressure_forcing);

  cache.parent_xi =
      DeviceReal3D("vector_fe_ex9_hdg_parent_xi", n_elem, cache.n_sides, cache.n_side_qp);
  cache.parent_eta =
      DeviceReal3D("vector_fe_ex9_hdg_parent_eta", n_elem, cache.n_sides, cache.n_side_qp);
  cache.parent_zeta =
      DeviceReal3D("vector_fe_ex9_hdg_parent_zeta", n_elem, cache.n_sides, cache.n_side_qp);
  cache.side_jxw =
      DeviceReal3D("vector_fe_ex9_hdg_side_jxw", n_elem, cache.n_sides, cache.n_side_qp);
  cache.normal_x =
      DeviceReal3D("vector_fe_ex9_hdg_normal_x", n_elem, cache.n_sides, cache.n_side_qp);
  cache.normal_y =
      DeviceReal3D("vector_fe_ex9_hdg_normal_y", n_elem, cache.n_sides, cache.n_side_qp);
  cache.normal_z =
      DeviceReal3D("vector_fe_ex9_hdg_normal_z", n_elem, cache.n_sides, cache.n_side_qp);
  cache.dirichlet_component =
      DeviceReal3D("vector_fe_ex9_hdg_dirichlet_component", n_elem, cache.n_sides, cache.n_side_qp);
  cache.dirichlet_vdotn =
      DeviceReal3D("vector_fe_ex9_hdg_dirichlet_vdotn", n_elem, cache.n_sides, cache.n_side_qp);
  copy_vector_to_side_view(parent_xi, n_elem, cache.n_sides, cache.n_side_qp, cache.parent_xi);
  copy_vector_to_side_view(parent_eta, n_elem, cache.n_sides, cache.n_side_qp, cache.parent_eta);
  copy_vector_to_side_view(parent_zeta, n_elem, cache.n_sides, cache.n_side_qp, cache.parent_zeta);
  copy_vector_to_side_view(side_jxw, n_elem, cache.n_sides, cache.n_side_qp, cache.side_jxw);
  copy_vector_to_side_view(normal_x, n_elem, cache.n_sides, cache.n_side_qp, cache.normal_x);
  copy_vector_to_side_view(normal_y, n_elem, cache.n_sides, cache.n_side_qp, cache.normal_y);
  copy_vector_to_side_view(normal_z, n_elem, cache.n_sides, cache.n_side_qp, cache.normal_z);
  copy_vector_to_side_view(
      dirichlet_component, n_elem, cache.n_sides, cache.n_side_qp, cache.dirichlet_component);
  copy_vector_to_side_view(
      dirichlet_vdotn, n_elem, cache.n_sides, cache.n_side_qp, cache.dirichlet_vdotn);

  cache.orientations = DeviceUChar2D("vector_fe_ex9_hdg_edge_orientations", n_elem, cache.n_sides);
  auto h_orientations = ::Kokkos::create_mirror_view(cache.orientations);
  for (unsigned int e = 0; e != n_elem; ++e)
    for (unsigned int s = 0; s != cache.n_sides; ++s)
      h_orientations(e, s) = orientations[e * cache.n_sides + s];
  ::Kokkos::deep_copy(cache.orientations, h_orientations);

  cache.side_types = DeviceInt2D("vector_fe_ex9_hdg_side_types", n_elem, cache.n_sides);
  auto h_side_types = ::Kokkos::create_mirror_view(cache.side_types);
  for (unsigned int e = 0; e != n_elem; ++e)
    for (unsigned int s = 0; s != cache.n_sides; ++s)
      h_side_types(e, s) = side_types[e * cache.n_sides + s];
  ::Kokkos::deep_copy(cache.side_types, h_side_types);

  return data_ptr;
}

std::unique_ptr<Ex9DeviceAssemblyData>
make_ex9_device_assembly_data(NonlinearImplicitSystem & system,
                              HDGProblem & problem,
                              const std::vector<const Elem *> & elems,
                              const ElemType elem_type,
                              const unsigned int qu_var,
                              const unsigned int qv_var,
                              const unsigned int u_var,
                              const unsigned int v_var,
                              const unsigned int p_var,
                              const unsigned int lm_u_var,
                              const unsigned int lm_v_var,
                              const unsigned int global_lm_var)
{
  const unsigned int n_elem = cast_int<unsigned int>(elems.size());
  auto data_ptr = std::make_unique<Ex9DeviceAssemblyData>();
  Ex9DeviceAssemblyData & data = *data_ptr;
  data.problem = &problem;
  data.system = &system;
  data.elem_type = elem_type;
  data.n_elem = n_elem;
  data.elems = elems;

  data.qu_indices.resize(n_elem);
  data.qv_indices.resize(n_elem);
  data.u_indices.resize(n_elem);
  data.v_indices.resize(n_elem);
  data.p_indices.resize(n_elem);
  data.lm_u_indices.resize(n_elem);
  data.lm_v_indices.resize(n_elem);
  data.global_lm_indices.resize(n_elem);

  const DofMap & dof_map = system.get_dof_map();
  const bool has_global_lm = global_lm_var != invalid_uint;
  for (unsigned int e = 0; e != n_elem; ++e)
  {
    dof_map.dof_indices(elems[e], data.qu_indices[e], qu_var);
    dof_map.dof_indices(elems[e], data.qv_indices[e], qv_var);
    dof_map.dof_indices(elems[e], data.u_indices[e], u_var);
    dof_map.dof_indices(elems[e], data.v_indices[e], v_var);
    dof_map.dof_indices(elems[e], data.p_indices[e], p_var);
    dof_map.dof_indices(elems[e], data.lm_u_indices[e], lm_u_var);
    dof_map.dof_indices(elems[e], data.lm_v_indices[e], lm_v_var);
    if (has_global_lm)
    {
      dof_map.dof_indices(elems[e], data.global_lm_indices[e], global_lm_var);
      libmesh_assert_equal_to(data.global_lm_indices[e].size(), 1);
    }
  }

  libmesh_assert_equal_to(data.qu_indices.front().size(), data.qv_indices.front().size());
  libmesh_assert_equal_to(data.u_indices.front().size(), data.v_indices.front().size());
  libmesh_assert_equal_to(data.u_indices.front().size(), data.p_indices.front().size());
  libmesh_assert_equal_to(data.lm_u_indices.front().size(), data.lm_v_indices.front().size());
  libmesh_error_msg_if(data.qu_indices.front().size() != 2 * data.u_indices.front().size(),
                       "vector_fe_ex9 Kokkos fused q-mass assembly expects 2D vector gradient "
                       "dofs.");
  data.layout =
      make_ex9_full_block_layout(cast_int<unsigned int>(data.qu_indices.front().size()),
                                 cast_int<unsigned int>(data.u_indices.front().size()),
                                 cast_int<unsigned int>(data.lm_u_indices.front().size()));

  data.qu_dof_ids = DeviceDofId2D("vector_fe_ex9_full_qu_dof_ids", n_elem, data.layout.q_n_dofs);
  data.qv_dof_ids = DeviceDofId2D("vector_fe_ex9_full_qv_dof_ids", n_elem, data.layout.q_n_dofs);
  data.u_dof_ids = DeviceDofId2D("vector_fe_ex9_full_u_dof_ids", n_elem, data.layout.scalar_n_dofs);
  data.v_dof_ids = DeviceDofId2D("vector_fe_ex9_full_v_dof_ids", n_elem, data.layout.scalar_n_dofs);
  data.p_dof_ids = DeviceDofId2D("vector_fe_ex9_full_p_dof_ids", n_elem, data.layout.scalar_n_dofs);
  data.lm_u_dof_ids =
      DeviceDofId2D("vector_fe_ex9_full_lm_u_dof_ids", n_elem, data.layout.lm_n_dofs);
  data.lm_v_dof_ids =
      DeviceDofId2D("vector_fe_ex9_full_lm_v_dof_ids", n_elem, data.layout.lm_n_dofs);
  data.global_lm_dof_ids =
      DeviceDofId2D("vector_fe_ex9_full_global_lm_dof_ids", n_elem, has_global_lm ? 1 : 0);
  copy_dof_indices_to_view(data.qu_indices, data.qu_dof_ids);
  copy_dof_indices_to_view(data.qv_indices, data.qv_dof_ids);
  copy_dof_indices_to_view(data.u_indices, data.u_dof_ids);
  copy_dof_indices_to_view(data.v_indices, data.v_dof_ids);
  copy_dof_indices_to_view(data.p_indices, data.p_dof_ids);
  copy_dof_indices_to_view(data.lm_u_indices, data.lm_u_dof_ids);
  copy_dof_indices_to_view(data.lm_v_indices, data.lm_v_dof_ids);
  copy_dof_indices_to_view(data.global_lm_indices, data.global_lm_dof_ids);

  return data_ptr;
}

DeviceReal1D
copy_global_solution_to_device(const NumericVector<Number> & X)
{
  std::vector<Number> host_values;
  X.localize(host_values);
  DeviceReal1D device_values("vector_fe_ex9_hdg_global_x", host_values.size());
  auto host_view = ::Kokkos::create_mirror_view(device_values);
  for (std::size_t i = 0; i != host_values.size(); ++i)
    host_view(i) = host_values[i];
  ::Kokkos::deep_copy(device_values, host_view);
  return device_values;
}

template <typename GlobalXView>
void
gather_ex9_dofs_on_device(const Ex9DeviceAssemblyData & data,
                          const GlobalXView & global_x,
                          const Ex9DeviceWork & work)
{
  const auto qu_ids = data.qu_dof_ids;
  const auto qv_ids = data.qv_dof_ids;
  const auto u_ids = data.u_dof_ids;
  const auto v_ids = data.v_dof_ids;
  const auto p_ids = data.p_dof_ids;
  const auto lm_u_ids = data.lm_u_dof_ids;
  const auto lm_v_ids = data.lm_v_dof_ids;
  const auto global_lm_ids = data.global_lm_dof_ids;
  const auto qu_dofs = work.qu_dofs;
  const auto qv_dofs = work.qv_dofs;
  const auto u_dofs = work.u_dofs;
  const auto v_dofs = work.v_dofs;
  const auto p_dofs = work.p_dofs;
  const auto lm_u_dofs = work.lm_u_dofs;
  const auto lm_v_dofs = work.lm_v_dofs;
  const auto global_lm_dofs = work.global_lm_dofs;
  const unsigned int q_n_dofs = data.layout.q_n_dofs;
  const unsigned int scalar_n_dofs = data.layout.scalar_n_dofs;
  const unsigned int lm_n_dofs = data.layout.lm_n_dofs;
  const bool has_global_lm = data.problem->cavity;

  ::Kokkos::parallel_for(
      "vector_fe_ex9_full_device_gather",
      ::Kokkos::RangePolicy<>(0, static_cast<int>(data.n_elem)),
      KOKKOS_LAMBDA(const int raw_e) {
        const unsigned int e = static_cast<unsigned int>(raw_e);
        for (unsigned int i = 0; i != q_n_dofs; ++i)
        {
          qu_dofs(e, i) = global_x(qu_ids(e, i));
          qv_dofs(e, i) = global_x(qv_ids(e, i));
        }
        for (unsigned int i = 0; i != scalar_n_dofs; ++i)
        {
          u_dofs(e, i) = global_x(u_ids(e, i));
          v_dofs(e, i) = global_x(v_ids(e, i));
          p_dofs(e, i) = global_x(p_ids(e, i));
        }
        for (unsigned int i = 0; i != lm_n_dofs; ++i)
        {
          lm_u_dofs(e, i) = global_x(lm_u_ids(e, i));
          lm_v_dofs(e, i) = global_x(lm_v_ids(e, i));
        }
        if (has_global_lm)
          global_lm_dofs(e, 0) = global_x(global_lm_ids(e, 0));
      });
}

LIBMESH_DEVICE_INLINE unsigned int
ex9_component_full_block_offset(const Ex9FullBlockLayout full,
                                const HDGLinearBlockLayout component,
                                const unsigned int full_q_offset,
                                const unsigned int full_u_offset,
                                const unsigned int full_lm_offset,
                                const unsigned int idx)
{
  unsigned int row = 0;
  unsigned int col = 0;

  if (idx < component.off_Jqu)
  {
    const unsigned int local = idx - component.off_Jqq;
    row = full_q_offset + local / component.vector_n_dofs;
    col = full_q_offset + local % component.vector_n_dofs;
  }
  else if (idx < component.off_Juq)
  {
    const unsigned int local = idx - component.off_Jqu;
    row = full_q_offset + local / component.scalar_n_dofs;
    col = full_u_offset + local % component.scalar_n_dofs;
  }
  else if (idx < component.off_Jup)
  {
    const unsigned int local = idx - component.off_Juq;
    row = full_u_offset + local / component.vector_n_dofs;
    col = full_q_offset + local % component.vector_n_dofs;
  }
  else if (idx < component.off_Jpu)
  {
    const unsigned int local = idx - component.off_Jup;
    row = full_u_offset + local / component.scalar_n_dofs;
    col = full.off_p + local % component.scalar_n_dofs;
  }
  else if (idx < component.off_Jplm)
  {
    const unsigned int local = idx - component.off_Jpu;
    row = full.off_p + local / component.scalar_n_dofs;
    col = full_u_offset + local % component.scalar_n_dofs;
  }
  else if (idx < component.off_Jqlm)
  {
    const unsigned int local = idx - component.off_Jplm;
    row = full.off_p + local / component.lm_n_dofs;
    col = full_lm_offset + local % component.lm_n_dofs;
  }
  else if (idx < component.off_Jlmq)
  {
    const unsigned int local = idx - component.off_Jqlm;
    row = full_q_offset + local / component.lm_n_dofs;
    col = full_lm_offset + local % component.lm_n_dofs;
  }
  else if (idx < component.off_Jlmp)
  {
    const unsigned int local = idx - component.off_Jlmq;
    row = full_lm_offset + local / component.vector_n_dofs;
    col = full_q_offset + local % component.vector_n_dofs;
  }
  else if (idx < component.off_Jlms)
  {
    const unsigned int local = idx - component.off_Jlmp;
    row = full_lm_offset + local / component.scalar_n_dofs;
    col = full.off_p + local % component.scalar_n_dofs;
  }
  else if (idx < component.off_Jlmlm)
  {
    const unsigned int local = idx - component.off_Jlms;
    row = full_lm_offset + local / component.scalar_n_dofs;
    col = full_u_offset + local % component.scalar_n_dofs;
  }
  else if (idx < component.off_Juu)
  {
    const unsigned int local = idx - component.off_Jlmlm;
    row = full_lm_offset + local / component.lm_n_dofs;
    col = full_lm_offset + local % component.lm_n_dofs;
  }
  else if (idx < component.off_Julm)
  {
    const unsigned int local = idx - component.off_Juu;
    row = full_u_offset + local / component.scalar_n_dofs;
    col = full_u_offset + local % component.scalar_n_dofs;
  }
  else
  {
    const unsigned int local = idx - component.off_Julm;
    row = full_u_offset + local / component.lm_n_dofs;
    col = full_lm_offset + local % component.lm_n_dofs;
  }

  return row * full.n_dofs + col;
}

LIBMESH_DEVICE_INLINE unsigned int
ex9_component_full_residual_offset(const Ex9FullBlockLayout full,
                                   const HDGLinearBlockLayout component,
                                   const unsigned int full_q_offset,
                                   const unsigned int full_u_offset,
                                   const unsigned int full_lm_offset,
                                   const unsigned int idx)
{
  const unsigned int cq = 0;
  const unsigned int cu = cq + component.vector_n_dofs;
  const unsigned int cp = cu + component.scalar_n_dofs;
  const unsigned int clm = cp + component.scalar_n_dofs;

  if (idx < cu)
    return full_q_offset + idx - cq;
  if (idx < cp)
    return full_u_offset + idx - cu;
  if (idx < clm)
    return full.off_p + idx - cp;
  return full_lm_offset + idx - clm;
}

void
scatter_component_pair_into_full(const DeviceAssemblyData & u_data,
                                 const DeviceAssemblyData & v_data,
                                 const Ex9FullBlockLayout full_layout,
                                 const DeviceReal2D & u_blocks,
                                 const DeviceReal2D & u_residual,
                                 const DeviceReal2D & v_blocks,
                                 const DeviceReal2D & v_residual,
                                 const DeviceReal2D & full_blocks,
                                 const DeviceReal2D & full_residual,
                                 const bool assemble_residual)
{
  const unsigned int n_elem = u_data.n_elem;
  const auto component_layout = u_data.layout;
  const unsigned int n_entries = component_layout.n_entries;
  const unsigned int residual_n_dofs = u_data.residual_n_dofs;

  ::Kokkos::parallel_for(
      "vector_fe_ex9_scatter_component_blocks_to_full",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_elem * n_entries)),
      KOKKOS_LAMBDA(const int raw_entry) {
        const unsigned int entry = static_cast<unsigned int>(raw_entry);
        const unsigned int e = entry / n_entries;
        const unsigned int idx = entry - e * n_entries;
        const unsigned int u_full_idx =
            ex9_component_full_block_offset(full_layout,
                                            component_layout,
                                            full_layout.off_qu,
                                            full_layout.off_u,
                                            full_layout.off_lm_u,
                                            idx);
        const unsigned int v_full_idx =
            ex9_component_full_block_offset(full_layout,
                                            component_layout,
                                            full_layout.off_qv,
                                            full_layout.off_v,
                                            full_layout.off_lm_v,
                                            idx);
        full_blocks(e, u_full_idx) += u_blocks(e, idx);
        full_blocks(e, v_full_idx) += v_blocks(e, idx);
      });

  if (!assemble_residual)
    return;

  ::Kokkos::parallel_for(
      "vector_fe_ex9_scatter_component_residual_to_full",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_elem * residual_n_dofs)),
      KOKKOS_LAMBDA(const int raw_entry) {
        const unsigned int entry = static_cast<unsigned int>(raw_entry);
        const unsigned int e = entry / residual_n_dofs;
        const unsigned int idx = entry - e * residual_n_dofs;
        const unsigned int u_full_idx =
            ex9_component_full_residual_offset(full_layout,
                                               component_layout,
                                               full_layout.off_qu,
                                               full_layout.off_u,
                                               full_layout.off_lm_u,
                                               idx);
        const unsigned int v_full_idx =
            ex9_component_full_residual_offset(full_layout,
                                               component_layout,
                                               full_layout.off_qv,
                                               full_layout.off_v,
                                               full_layout.off_lm_v,
                                               idx);
        full_residual(e, u_full_idx) += u_residual(e, idx);
        full_residual(e, v_full_idx) += v_residual(e, idx);
      });
}

template <ElemType ExactTopo>
void
assemble_component_pair_volume_blocks_range(const DeviceAssemblyData & data,
                                            const DeviceReal2D & u_blocks,
                                            const DeviceReal2D & v_blocks)
{
  const unsigned int n_elem = data.n_elem;
  const HDGLinearBlockLayout layout = data.layout;
  const unsigned int n_entries = layout.off_Jplm;
  const FEShapeKey vector_key = data.vector_key;
  const FEShapeKey scalar_key = data.scalar_key;
  const DeviceReal3D coords = data.coords;
  const unsigned int n_geom_nodes = data.n_geom_nodes;
  const DeviceReal1D xi = data.xi;
  const DeviceReal1D eta = data.eta;
  const DeviceReal1D zeta = data.zeta;
  const DeviceReal1D jxw = data.jxw;
  const unsigned int n_qp = data.n_qp;
  const Real nu = data.problem->nu;

  ::Kokkos::parallel_for(
      "vector_fe_ex9_split_component_volume_blocks",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_elem * 2 * n_entries)),
      KOKKOS_LAMBDA(const int raw_entry) {
        const unsigned int entry = static_cast<unsigned int>(raw_entry);
        const unsigned int e = entry / (2 * n_entries);
        const unsigned int component_entry = entry - e * 2 * n_entries;
        const unsigned int component = component_entry / n_entries;
        const unsigned int idx = component_entry - component * n_entries;
        Real value = Real(0);

        if (idx < layout.off_Jqu)
        {
          const unsigned int local = idx - layout.off_Jqq;
          const unsigned int i = local / layout.vector_n_dofs;
          const unsigned int j = local % layout.vector_n_dofs;
          for (unsigned int q = 0; q != n_qp; ++q)
          {
            const libMesh::Kokkos::RealVector phi_i =
                libMesh::Kokkos::vector_shape(vector_key, i, xi(q), eta(q), zeta(q));
            const libMesh::Kokkos::RealVector phi_j =
                libMesh::Kokkos::vector_shape(vector_key, j, xi(q), eta(q), zeta(q));
            Real dot = Real(0);
            for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
              dot += Kokkos::detail::component_or_zero(phi_i, d) *
                     Kokkos::detail::component_or_zero(phi_j, d);
            value += jxw(q) * dot;
          }
        }
        else if (idx < layout.off_Juq)
        {
          const unsigned int local = idx - layout.off_Jqu;
          const unsigned int i = local / layout.scalar_n_dofs;
          const unsigned int j = local % layout.scalar_n_dofs;
          for (unsigned int q = 0; q != n_qp; ++q)
          {
            const unsigned int scalar_i = i / 2;
            const unsigned int grad_component = i - scalar_i * 2;
            const libMesh::Kokkos::RealTensor J =
                libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                    Kokkos::detail::HDGElementNodeView<DeviceReal3D>{coords, e},
                    n_geom_nodes,
                    xi(q),
                    eta(q),
                    zeta(q));
            const libMesh::Kokkos::RealTensor Jinv =
                libMesh::Kokkos::inverse<libMesh::Kokkos::RealTensor>(J, 2);
            const libMesh::Kokkos::RealVector grad_phys =
                Jinv * libMesh::Kokkos::grad_shape(scalar_key, scalar_i, xi(q), eta(q), zeta(q));
            value += jxw(q) * Kokkos::detail::component_or_zero(grad_phys, grad_component) *
                     libMesh::Kokkos::shape(scalar_key, j, xi(q), eta(q), zeta(q));
          }
        }
        else if (idx < layout.off_Jup)
        {
          const unsigned int local = idx - layout.off_Juq;
          const unsigned int i = local / layout.vector_n_dofs;
          const unsigned int j = local % layout.vector_n_dofs;
          for (unsigned int q = 0; q != n_qp; ++q)
          {
            const libMesh::Kokkos::RealTensor J =
                libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                    Kokkos::detail::HDGElementNodeView<DeviceReal3D>{coords, e},
                    n_geom_nodes,
                    xi(q),
                    eta(q),
                    zeta(q));
            const libMesh::Kokkos::RealTensor Jinv =
                libMesh::Kokkos::inverse<libMesh::Kokkos::RealTensor>(J, 2);
            const libMesh::Kokkos::RealVector grad_phys =
                Jinv * libMesh::Kokkos::grad_shape(scalar_key, i, xi(q), eta(q), zeta(q));
            const libMesh::Kokkos::RealVector phi_j =
                libMesh::Kokkos::vector_shape(vector_key, j, xi(q), eta(q), zeta(q));
            Real dot = Real(0);
            for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
              dot += Kokkos::detail::component_or_zero(grad_phys, d) *
                     Kokkos::detail::component_or_zero(phi_j, d);
            value += jxw(q) * nu * dot;
          }
        }
        else
        {
          const unsigned int local =
              idx < layout.off_Jpu ? idx - layout.off_Jup : idx - layout.off_Jpu;
          const unsigned int i = local / layout.scalar_n_dofs;
          const unsigned int j = local % layout.scalar_n_dofs;
          for (unsigned int q = 0; q != n_qp; ++q)
          {
            const libMesh::Kokkos::RealTensor J =
                libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                    Kokkos::detail::HDGElementNodeView<DeviceReal3D>{coords, e},
                    n_geom_nodes,
                    xi(q),
                    eta(q),
                    zeta(q));
            const libMesh::Kokkos::RealTensor Jinv =
                libMesh::Kokkos::inverse<libMesh::Kokkos::RealTensor>(J, 2);
            const libMesh::Kokkos::RealVector grad_phys =
                Jinv * libMesh::Kokkos::grad_shape(scalar_key, i, xi(q), eta(q), zeta(q));
            value -= jxw(q) * Kokkos::detail::component_or_zero(grad_phys, component) *
                     libMesh::Kokkos::shape(scalar_key, j, xi(q), eta(q), zeta(q));
          }
        }

        if (component == 0)
          u_blocks(e, idx) += value;
        else
          v_blocks(e, idx) += value;
      });
}

void
assemble_component_pair_side_blocks_range(const DeviceAssemblyData & data,
                                          const DeviceReal2D & u_blocks,
                                          const DeviceReal2D & v_blocks)
{
  const unsigned int n_elem = data.n_elem;
  const HDGLinearBlockLayout layout = data.layout;
  const unsigned int n_entries = layout.n_entries - layout.off_Juq;
  const FEShapeKey vector_key = data.vector_key;
  const FEShapeKey scalar_key = data.scalar_key;
  const FEShapeKey lm_key = data.lm_key;
  const DeviceReal3D parent_xi = data.parent_xi;
  const DeviceReal3D parent_eta = data.parent_eta;
  const DeviceReal3D parent_zeta = data.parent_zeta;
  const DeviceReal3D side_jxw = data.side_jxw;
  const DeviceReal3D normal_x = data.normal_x;
  const DeviceReal3D normal_y = data.normal_y;
  const DeviceReal3D normal_z = data.normal_z;
  const DeviceUChar2D orientations = data.orientations;
  const DeviceInt2D side_types = data.side_types;
  const unsigned int n_sides = data.n_sides;
  const unsigned int n_side_qp = data.n_side_qp;
  const Real nu = data.problem->nu;
  const Real tau = Real(1);

  ::Kokkos::parallel_for(
      "vector_fe_ex9_split_component_side_blocks",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_elem * n_sides * 2 * n_entries)),
      KOKKOS_LAMBDA(const int raw_entry) {
        const unsigned int entry = static_cast<unsigned int>(raw_entry);
        const unsigned int entries_per_elem = n_sides * 2 * n_entries;
        const unsigned int e = entry / entries_per_elem;
        const unsigned int elem_entry = entry - e * entries_per_elem;
        const unsigned int side = elem_entry / (2 * n_entries);
        const unsigned int side_entry = elem_entry - side * 2 * n_entries;
        const unsigned int component = side_entry / n_entries;
        const unsigned int idx = side_entry - component * n_entries + layout.off_Juq;
        const unsigned int velocity_component = component;
        const bool positive_edge_orientation = static_cast<bool>(orientations(e, side));
        const int side_type = side_types(e, side);
        const bool is_outlet = side_type == Kokkos::detail::hdg_linear_side_outlet;
        const bool is_dirichlet = side_type == Kokkos::detail::hdg_linear_side_dirichlet;
        Real value = Real(0);

        if (idx < layout.off_Jup)
        {
          if (!is_outlet)
          {
            const unsigned int local = idx - layout.off_Juq;
            const unsigned int i = local / layout.vector_n_dofs;
            const unsigned int j = local % layout.vector_n_dofs;
            for (unsigned int q = 0; q != n_side_qp; ++q)
            {
              const Real px = parent_xi(e, side, q);
              const Real py = parent_eta(e, side, q);
              const Real pz = parent_zeta(e, side, q);
              const libMesh::Kokkos::RealVector phi_j =
                  libMesh::Kokkos::vector_shape(vector_key, j, px, py, pz);
              Real vector_normal = Kokkos::detail::component_or_zero(phi_j, 0) *
                                   normal_x(e, side, q);
#if LIBMESH_DIM > 1
              vector_normal += Kokkos::detail::component_or_zero(phi_j, 1) *
                               normal_y(e, side, q);
#endif
#if LIBMESH_DIM > 2
              vector_normal += Kokkos::detail::component_or_zero(phi_j, 2) *
                               normal_z(e, side, q);
#endif
              value -= side_jxw(e, side, q) * nu *
                       libMesh::Kokkos::shape(scalar_key, i, px, py, pz) * vector_normal;
            }
          }
        }
        else if (idx < layout.off_Jpu)
        {
          if (!is_outlet)
          {
            const unsigned int local = idx - layout.off_Jup;
            const unsigned int i = local / layout.scalar_n_dofs;
            const unsigned int j = local % layout.scalar_n_dofs;
            for (unsigned int q = 0; q != n_side_qp; ++q)
            {
              const Real px = parent_xi(e, side, q);
              const Real py = parent_eta(e, side, q);
              const Real pz = parent_zeta(e, side, q);
              const Real normal_component =
                  velocity_component == 0 ? normal_x(e, side, q)
#if LIBMESH_DIM > 1
                                          : normal_y(e, side, q)
#else
                                          : Real(0)
#endif
                  ;
              value += side_jxw(e, side, q) *
                       libMesh::Kokkos::shape(scalar_key, i, px, py, pz) *
                       libMesh::Kokkos::shape(scalar_key, j, px, py, pz) * normal_component;
            }
          }
        }
        else if (idx < layout.off_Jplm)
        {
        }
        else if (idx < layout.off_Jqlm)
        {
          if (!is_dirichlet)
          {
            const unsigned int local = idx - layout.off_Jplm;
            const unsigned int i = local / layout.lm_n_dofs;
            const unsigned int j = local % layout.lm_n_dofs;
            for (unsigned int q = 0; q != n_side_qp; ++q)
            {
              const Real px = parent_xi(e, side, q);
              const Real py = parent_eta(e, side, q);
              const Real pz = parent_zeta(e, side, q);
              const Real normal_component =
                  velocity_component == 0 ? normal_x(e, side, q)
#if LIBMESH_DIM > 1
                                          : normal_y(e, side, q)
#else
                                          : Real(0)
#endif
                  ;
              value += side_jxw(e, side, q) *
                       libMesh::Kokkos::shape(scalar_key, i, px, py, pz) *
                       libMesh::Kokkos::side_trace_shape(
                           lm_key, j, side, positive_edge_orientation, px, py, pz) *
                       normal_component;
            }
          }
        }
        else if (idx < layout.off_Jlmq)
        {
          if (!is_dirichlet)
          {
            const unsigned int local = idx - layout.off_Jqlm;
            const unsigned int i = local / layout.lm_n_dofs;
            const unsigned int j = local % layout.lm_n_dofs;
            for (unsigned int q = 0; q != n_side_qp; ++q)
            {
              const Real px = parent_xi(e, side, q);
              const Real py = parent_eta(e, side, q);
              const Real pz = parent_zeta(e, side, q);
              const libMesh::Kokkos::RealVector phi_i =
                  libMesh::Kokkos::vector_shape(vector_key, i, px, py, pz);
              Real vector_normal = Kokkos::detail::component_or_zero(phi_i, 0) *
                                   normal_x(e, side, q);
#if LIBMESH_DIM > 1
              vector_normal += Kokkos::detail::component_or_zero(phi_i, 1) *
                               normal_y(e, side, q);
#endif
#if LIBMESH_DIM > 2
              vector_normal += Kokkos::detail::component_or_zero(phi_i, 2) *
                               normal_z(e, side, q);
#endif
              value -= side_jxw(e, side, q) * vector_normal *
                       libMesh::Kokkos::side_trace_shape(
                           lm_key, j, side, positive_edge_orientation, px, py, pz);
            }
          }
        }
        else if (idx < layout.off_Jlmp)
        {
          if (!is_dirichlet)
          {
            const unsigned int local = idx - layout.off_Jlmq;
            const unsigned int i = local / layout.vector_n_dofs;
            const unsigned int j = local % layout.vector_n_dofs;
            for (unsigned int q = 0; q != n_side_qp; ++q)
            {
              const Real px = parent_xi(e, side, q);
              const Real py = parent_eta(e, side, q);
              const Real pz = parent_zeta(e, side, q);
              const libMesh::Kokkos::RealVector phi_j =
                  libMesh::Kokkos::vector_shape(vector_key, j, px, py, pz);
              Real vector_normal = Kokkos::detail::component_or_zero(phi_j, 0) *
                                   normal_x(e, side, q);
#if LIBMESH_DIM > 1
              vector_normal += Kokkos::detail::component_or_zero(phi_j, 1) *
                               normal_y(e, side, q);
#endif
#if LIBMESH_DIM > 2
              vector_normal += Kokkos::detail::component_or_zero(phi_j, 2) *
                               normal_z(e, side, q);
#endif
              value -= side_jxw(e, side, q) * nu *
                       libMesh::Kokkos::side_trace_shape(
                           lm_key, i, side, positive_edge_orientation, px, py, pz) *
                       vector_normal;
            }
          }
        }
        else if (idx < layout.off_Jlms)
        {
          if (!is_dirichlet)
          {
            const unsigned int local = idx - layout.off_Jlmp;
            const unsigned int i = local / layout.scalar_n_dofs;
            const unsigned int j = local % layout.scalar_n_dofs;
            for (unsigned int q = 0; q != n_side_qp; ++q)
            {
              const Real px = parent_xi(e, side, q);
              const Real py = parent_eta(e, side, q);
              const Real pz = parent_zeta(e, side, q);
              const Real normal_component =
                  velocity_component == 0 ? normal_x(e, side, q)
#if LIBMESH_DIM > 1
                                          : normal_y(e, side, q)
#else
                                          : Real(0)
#endif
                  ;
              value += side_jxw(e, side, q) *
                       libMesh::Kokkos::side_trace_shape(
                           lm_key, i, side, positive_edge_orientation, px, py, pz) *
                       libMesh::Kokkos::shape(scalar_key, j, px, py, pz) * normal_component;
            }
          }
        }
        else if (idx < layout.off_Jlmlm)
        {
          if (!is_dirichlet)
          {
            const unsigned int local = idx - layout.off_Jlms;
            const unsigned int i = local / layout.scalar_n_dofs;
            const unsigned int j = local % layout.scalar_n_dofs;
            for (unsigned int q = 0; q != n_side_qp; ++q)
            {
              const Real px = parent_xi(e, side, q);
              const Real py = parent_eta(e, side, q);
              const Real pz = parent_zeta(e, side, q);
              Real normal_norm2 = normal_x(e, side, q) * normal_x(e, side, q);
#if LIBMESH_DIM > 1
              normal_norm2 += normal_y(e, side, q) * normal_y(e, side, q);
#endif
#if LIBMESH_DIM > 2
              normal_norm2 += normal_z(e, side, q) * normal_z(e, side, q);
#endif
              value += side_jxw(e, side, q) *
                       libMesh::Kokkos::side_trace_shape(
                           lm_key, i, side, positive_edge_orientation, px, py, pz) *
                       tau * libMesh::Kokkos::shape(scalar_key, j, px, py, pz) * normal_norm2;
            }
          }
        }
        else if (idx < layout.off_Juu)
        {
          const unsigned int local = idx - layout.off_Jlmlm;
          const unsigned int i = local / layout.lm_n_dofs;
          const unsigned int j = local % layout.lm_n_dofs;
          for (unsigned int q = 0; q != n_side_qp; ++q)
          {
            const Real px = parent_xi(e, side, q);
            const Real py = parent_eta(e, side, q);
            const Real pz = parent_zeta(e, side, q);
            const Real trace_i =
                libMesh::Kokkos::side_trace_shape(
                    lm_key, i, side, positive_edge_orientation, px, py, pz);
            const Real trace_j =
                libMesh::Kokkos::side_trace_shape(
                    lm_key, j, side, positive_edge_orientation, px, py, pz);
            if (is_dirichlet)
            {
              value -= side_jxw(e, side, q) * trace_i * trace_j;
              continue;
            }
            Real normal_norm2 = normal_x(e, side, q) * normal_x(e, side, q);
#if LIBMESH_DIM > 1
            normal_norm2 += normal_y(e, side, q) * normal_y(e, side, q);
#endif
#if LIBMESH_DIM > 2
            normal_norm2 += normal_z(e, side, q) * normal_z(e, side, q);
#endif
            value -= side_jxw(e, side, q) * trace_i * tau * trace_j * normal_norm2;
          }
        }
        else if (idx < layout.off_Julm)
        {
          if (!is_outlet)
          {
            const unsigned int local = idx - layout.off_Juu;
            const unsigned int i = local / layout.scalar_n_dofs;
            const unsigned int j = local % layout.scalar_n_dofs;
            for (unsigned int q = 0; q != n_side_qp; ++q)
            {
              const Real px = parent_xi(e, side, q);
              const Real py = parent_eta(e, side, q);
              const Real pz = parent_zeta(e, side, q);
              Real normal_norm2 = normal_x(e, side, q) * normal_x(e, side, q);
#if LIBMESH_DIM > 1
              normal_norm2 += normal_y(e, side, q) * normal_y(e, side, q);
#endif
#if LIBMESH_DIM > 2
              normal_norm2 += normal_z(e, side, q) * normal_z(e, side, q);
#endif
              value += side_jxw(e, side, q) *
                       libMesh::Kokkos::shape(scalar_key, i, px, py, pz) * tau *
                       libMesh::Kokkos::shape(scalar_key, j, px, py, pz) * normal_norm2;
            }
          }
        }
        else
        {
          if (!is_outlet && !is_dirichlet)
          {
            const unsigned int local = idx - layout.off_Julm;
            const unsigned int i = local / layout.lm_n_dofs;
            const unsigned int j = local % layout.lm_n_dofs;
            for (unsigned int q = 0; q != n_side_qp; ++q)
            {
              const Real px = parent_xi(e, side, q);
              const Real py = parent_eta(e, side, q);
              const Real pz = parent_zeta(e, side, q);
              Real normal_norm2 = normal_x(e, side, q) * normal_x(e, side, q);
#if LIBMESH_DIM > 1
              normal_norm2 += normal_y(e, side, q) * normal_y(e, side, q);
#endif
#if LIBMESH_DIM > 2
              normal_norm2 += normal_z(e, side, q) * normal_z(e, side, q);
#endif
              value -= side_jxw(e, side, q) *
                       libMesh::Kokkos::shape(scalar_key, i, px, py, pz) * tau *
                       libMesh::Kokkos::side_trace_shape(
                           lm_key, j, side, positive_edge_orientation, px, py, pz) *
                       normal_norm2;
            }
          }
        }

        if (component == 0)
          u_blocks(e, idx) += value;
        else
          v_blocks(e, idx) += value;
      });
}

template <ElemType ExactTopo>
void
assemble_component_pair_volume_residual_range(const DeviceAssemblyData & u_data,
                                              const DeviceAssemblyData & v_data,
                                              const ComponentDofViews & u_dofs,
                                              const ComponentDofViews & v_dofs,
                                              const DeviceReal2D & u_residual,
                                              const DeviceReal2D & v_residual)
{
  const unsigned int n_elem = u_data.n_elem;
  const HDGLinearBlockLayout layout = u_data.layout;
  const unsigned int n_entries = layout.vector_n_dofs + 2 * layout.scalar_n_dofs;
  const unsigned int u0 = layout.vector_n_dofs;
  const unsigned int p0 = u0 + layout.scalar_n_dofs;
  const FEShapeKey vector_key = u_data.vector_key;
  const FEShapeKey scalar_key = u_data.scalar_key;
  const DeviceReal3D coords = u_data.coords;
  const unsigned int n_geom_nodes = u_data.n_geom_nodes;
  const DeviceReal1D xi = u_data.xi;
  const DeviceReal1D eta = u_data.eta;
  const DeviceReal1D zeta = u_data.zeta;
  const DeviceReal1D jxw = u_data.jxw;
  const DeviceReal2D scalar_forcing_0 = u_data.scalar_forcing;
  const DeviceReal2D scalar_forcing_1 = v_data.scalar_forcing;
  const DeviceReal2D pressure_forcing = u_data.pressure_forcing;
  const unsigned int n_qp = u_data.n_qp;
  const Real nu = u_data.problem->nu;
  const bool include_convection = true;
  const DeviceReal2D u_q_dofs = u_dofs.q_dofs;
  const DeviceReal2D u_u_dofs = u_dofs.u_dofs;
  const DeviceReal2D u_p_dofs = u_dofs.p_dofs;
  const DeviceReal2D u_other_u_dofs = u_dofs.other_u_dofs;
  const DeviceReal2D v_q_dofs = v_dofs.q_dofs;
  const DeviceReal2D v_u_dofs = v_dofs.u_dofs;
  const DeviceReal2D v_other_u_dofs = v_dofs.other_u_dofs;

  ::Kokkos::parallel_for(
      "vector_fe_ex9_split_component_volume_residual",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_elem * 2 * n_entries)),
      KOKKOS_LAMBDA(const int raw_entry) {
        const unsigned int entry = static_cast<unsigned int>(raw_entry);
        const unsigned int e = entry / (2 * n_entries);
        const unsigned int component_entry = entry - e * 2 * n_entries;
        const unsigned int component = component_entry / n_entries;
        const unsigned int idx = component_entry - component * n_entries;
        const unsigned int velocity_component = component;
        Real value = Real(0);

        if (idx < u0)
        {
          const unsigned int i = idx;
          for (unsigned int q = 0; q != n_qp; ++q)
          {
            const Real xiq = xi(q);
            const Real etaq = eta(q);
            const Real zetaq = zeta(q);
            const unsigned int scalar_i = i / 2;
            const unsigned int grad_component = i - scalar_i * 2;
            const libMesh::Kokkos::RealTensor J =
                libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                    Kokkos::detail::HDGElementNodeView<DeviceReal3D>{coords, e},
                    n_geom_nodes,
                    xiq,
                    etaq,
                    zetaq);
            const libMesh::Kokkos::RealTensor Jinv =
                libMesh::Kokkos::inverse<libMesh::Kokkos::RealTensor>(J, 2);
            const libMesh::Kokkos::RealVector grad_phys =
                Jinv * libMesh::Kokkos::grad_shape(scalar_key, scalar_i, xiq, etaq, zetaq);
            const libMesh::Kokkos::RealVector phi_i =
                libMesh::Kokkos::vector_shape(vector_key, i, xiq, etaq, zetaq);
            Real phi_dot_q = Real(0);
            for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
            {
              Real q_value = Real(0);
              for (unsigned int j = 0; j != layout.vector_n_dofs; ++j)
              {
                const libMesh::Kokkos::RealVector phi_j =
                    libMesh::Kokkos::vector_shape(vector_key, j, xiq, etaq, zetaq);
                const Real q_dof = component == 0 ? u_q_dofs(e, j) : v_q_dofs(e, j);
                q_value += Kokkos::detail::component_or_zero(phi_j, d) * q_dof;
              }
              phi_dot_q += Kokkos::detail::component_or_zero(phi_i, d) * q_value;
            }

            Real u_value = Real(0);
            for (unsigned int j = 0; j != layout.scalar_n_dofs; ++j)
            {
              const Real phi = libMesh::Kokkos::shape(scalar_key, j, xiq, etaq, zetaq);
              const Real u_dof = component == 0 ? u_u_dofs(e, j) : v_u_dofs(e, j);
              u_value += phi * u_dof;
            }

            value += jxw(q) *
                     (phi_dot_q +
                      Kokkos::detail::component_or_zero(grad_phys, grad_component) * u_value);
          }
        }
        else if (idx < p0)
        {
          const unsigned int i = idx - u0;
          for (unsigned int q = 0; q != n_qp; ++q)
          {
            const Real xiq = xi(q);
            const Real etaq = eta(q);
            const Real zetaq = zeta(q);
            const libMesh::Kokkos::RealTensor J =
                libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                    Kokkos::detail::HDGElementNodeView<DeviceReal3D>{coords, e},
                    n_geom_nodes,
                    xiq,
                    etaq,
                    zetaq);
            const libMesh::Kokkos::RealTensor Jinv =
                libMesh::Kokkos::inverse<libMesh::Kokkos::RealTensor>(J, 2);
            const libMesh::Kokkos::RealVector grad_phys =
                Jinv * libMesh::Kokkos::grad_shape(scalar_key, i, xiq, etaq, zetaq);

            Real grad_dot_stress = Real(0);
            for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
            {
              Real q_value = Real(0);
              for (unsigned int j = 0; j != layout.vector_n_dofs; ++j)
              {
                const libMesh::Kokkos::RealVector phi_j =
                    libMesh::Kokkos::vector_shape(vector_key, j, xiq, etaq, zetaq);
                const Real q_dof = component == 0 ? u_q_dofs(e, j) : v_q_dofs(e, j);
                q_value += Kokkos::detail::component_or_zero(phi_j, d) * q_dof;
              }

              Real p_value = Real(0);
              for (unsigned int j = 0; j != layout.scalar_n_dofs; ++j)
                p_value +=
                    libMesh::Kokkos::shape(scalar_key, j, xiq, etaq, zetaq) * u_p_dofs(e, j);

              Real stress_component = nu * q_value;
              if (d == velocity_component)
                stress_component -= p_value;
              grad_dot_stress +=
                  Kokkos::detail::component_or_zero(grad_phys, d) * stress_component;
            }

            Real active_u = Real(0);
            Real other_u = Real(0);
            for (unsigned int j = 0; j != layout.scalar_n_dofs; ++j)
            {
              const Real phi = libMesh::Kokkos::shape(scalar_key, j, xiq, etaq, zetaq);
              const Real u_dof = component == 0 ? u_u_dofs(e, j) : v_u_dofs(e, j);
              const Real other_u_dof =
                  component == 0 ? u_other_u_dofs(e, j) : v_other_u_dofs(e, j);
              active_u += phi * u_dof;
              other_u += phi * other_u_dof;
            }

            const libMesh::Kokkos::RealVector convective_flux =
                include_convection
                    ? Kokkos::detail::hdg_velocity_from_components(
                          active_u, other_u, velocity_component) *
                          active_u
                    : libMesh::Kokkos::RealVector();
            Real grad_dot_convective_flux = Real(0);
            for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
              grad_dot_convective_flux +=
                  Kokkos::detail::component_or_zero(grad_phys, d) *
                  Kokkos::detail::component_or_zero(convective_flux, d);
            const Real forcing =
                component == 0 ? scalar_forcing_0(e, q) : scalar_forcing_1(e, q);

            value += jxw(q) *
                     (grad_dot_stress - grad_dot_convective_flux -
                      libMesh::Kokkos::shape(scalar_key, i, xiq, etaq, zetaq) * forcing);
          }
        }
        else
        {
          const unsigned int i = idx - p0;
          for (unsigned int q = 0; q != n_qp; ++q)
          {
            const Real xiq = xi(q);
            const Real etaq = eta(q);
            const Real zetaq = zeta(q);
            const libMesh::Kokkos::RealTensor J =
                libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                    Kokkos::detail::HDGElementNodeView<DeviceReal3D>{coords, e},
                    n_geom_nodes,
                    xiq,
                    etaq,
                    zetaq);
            const libMesh::Kokkos::RealTensor Jinv =
                libMesh::Kokkos::inverse<libMesh::Kokkos::RealTensor>(J, 2);
            const libMesh::Kokkos::RealVector grad_phys =
                Jinv * libMesh::Kokkos::grad_shape(scalar_key, i, xiq, etaq, zetaq);
            Real u_value = Real(0);
            for (unsigned int j = 0; j != layout.scalar_n_dofs; ++j)
            {
              const Real phi = libMesh::Kokkos::shape(scalar_key, j, xiq, etaq, zetaq);
              const Real u_dof = component == 0 ? u_u_dofs(e, j) : v_u_dofs(e, j);
              u_value += phi * u_dof;
            }

            value -= jxw(q) *
                     Kokkos::detail::component_or_zero(grad_phys, velocity_component) * u_value;
            if (velocity_component == 0)
              value -= jxw(q) *
                       libMesh::Kokkos::shape(scalar_key, i, xiq, etaq, zetaq) *
                       pressure_forcing(e, q);
          }
        }

        if (component == 0)
          u_residual(e, idx) += value;
        else
          v_residual(e, idx) += value;
      });
}

void
assemble_component_pair_side_residual_range(const DeviceAssemblyData & u_data,
                                            const DeviceAssemblyData & v_data,
                                            const ComponentDofViews & u_dofs,
                                            const ComponentDofViews & v_dofs,
                                            const DeviceReal2D & u_residual,
                                            const DeviceReal2D & v_residual)
{
  const unsigned int n_elem = u_data.n_elem;
  const HDGLinearBlockLayout layout = u_data.layout;
  const unsigned int n_entries = u_data.residual_n_dofs;
  const unsigned int u0 = layout.vector_n_dofs;
  const unsigned int p0 = u0 + layout.scalar_n_dofs;
  const unsigned int lm0 = p0 + layout.scalar_n_dofs;
  const FEShapeKey vector_key = u_data.vector_key;
  const FEShapeKey scalar_key = u_data.scalar_key;
  const FEShapeKey lm_key = u_data.lm_key;
  const DeviceReal3D parent_xi = u_data.parent_xi;
  const DeviceReal3D parent_eta = u_data.parent_eta;
  const DeviceReal3D parent_zeta = u_data.parent_zeta;
  const DeviceReal3D side_jxw = u_data.side_jxw;
  const DeviceReal3D normal_x = u_data.normal_x;
  const DeviceReal3D normal_y = u_data.normal_y;
  const DeviceReal3D normal_z = u_data.normal_z;
  const DeviceReal3D dirichlet_component_0 = u_data.dirichlet_component;
  const DeviceReal3D dirichlet_component_1 = v_data.dirichlet_component;
  const DeviceReal3D dirichlet_vdotn = u_data.dirichlet_vdotn;
  const DeviceUChar2D orientations = u_data.orientations;
  const DeviceInt2D side_types = u_data.side_types;
  const unsigned int n_sides = u_data.n_sides;
  const unsigned int n_side_qp = u_data.n_side_qp;
  const Real nu = u_data.problem->nu;
  const Real tau = Real(1);
  const bool include_convection = true;
  const DeviceReal2D u_q_dofs = u_dofs.q_dofs;
  const DeviceReal2D u_u_dofs = u_dofs.u_dofs;
  const DeviceReal2D u_p_dofs = u_dofs.p_dofs;
  const DeviceReal2D u_lm_dofs = u_dofs.lm_dofs;
  const DeviceReal2D u_other_lm_dofs = u_dofs.other_lm_dofs;
  const DeviceReal2D v_q_dofs = v_dofs.q_dofs;
  const DeviceReal2D v_u_dofs = v_dofs.u_dofs;
  const DeviceReal2D v_lm_dofs = v_dofs.lm_dofs;
  const DeviceReal2D v_other_lm_dofs = v_dofs.other_lm_dofs;

  ::Kokkos::parallel_for(
      "vector_fe_ex9_split_component_side_residual",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_elem * n_sides * 2 * n_entries)),
      KOKKOS_LAMBDA(const int raw_entry) {
        const unsigned int entry = static_cast<unsigned int>(raw_entry);
        const unsigned int entries_per_elem = n_sides * 2 * n_entries;
        const unsigned int e = entry / entries_per_elem;
        const unsigned int elem_entry = entry - e * entries_per_elem;
        const unsigned int side = elem_entry / (2 * n_entries);
        const unsigned int side_entry = elem_entry - side * 2 * n_entries;
        const unsigned int component = side_entry / n_entries;
        const unsigned int idx = side_entry - component * n_entries;
        const unsigned int velocity_component = component;
        const bool positive_edge_orientation = static_cast<bool>(orientations(e, side));
        const int side_type = side_types(e, side);
        const bool is_outlet = side_type == Kokkos::detail::hdg_linear_side_outlet;
        const bool is_dirichlet = side_type == Kokkos::detail::hdg_linear_side_dirichlet;
        Real value = Real(0);

        if (idx < u0)
        {
          const unsigned int i = idx;
          for (unsigned int q = 0; q != n_side_qp; ++q)
          {
            const Real px = parent_xi(e, side, q);
            const Real py = parent_eta(e, side, q);
            const Real pz = parent_zeta(e, side, q);
            const libMesh::Kokkos::RealVector phi_i =
                libMesh::Kokkos::vector_shape(vector_key, i, px, py, pz);
            const Real nx = normal_x(e, side, q);
#if LIBMESH_DIM > 1
            const Real ny = normal_y(e, side, q);
#else
            const Real ny = Real(0);
#endif
#if LIBMESH_DIM > 2
            const Real nz = normal_z(e, side, q);
#else
            const Real nz = Real(0);
#endif
            const Real boundary_value =
                is_dirichlet
                    ? (component == 0 ? dirichlet_component_0(e, side, q)
                                      : dirichlet_component_1(e, side, q))
                    : Real(0);
            Real lm_value = Real(0);
            if (!is_dirichlet)
              for (unsigned int j = 0; j != layout.lm_n_dofs; ++j)
              {
                const Real trace = libMesh::Kokkos::side_trace_shape(
                    lm_key, j, side, positive_edge_orientation, px, py, pz);
                const Real lm_dof = component == 0 ? u_lm_dofs(e, j) : v_lm_dofs(e, j);
                lm_value += trace * lm_dof;
              }
            const Real active_boundary_value = is_dirichlet ? boundary_value : lm_value;
            Real vector_normal = Kokkos::detail::component_or_zero(phi_i, 0) * nx;
#if LIBMESH_DIM > 1
            vector_normal += Kokkos::detail::component_or_zero(phi_i, 1) * ny;
#endif
#if LIBMESH_DIM > 2
            vector_normal += Kokkos::detail::component_or_zero(phi_i, 2) * nz;
#endif
            value -= side_jxw(e, side, q) * vector_normal * active_boundary_value;
          }
        }
        else if (idx < p0)
        {
          const unsigned int i = idx - u0;
          for (unsigned int q = 0; q != n_side_qp; ++q)
          {
            const Real px = parent_xi(e, side, q);
            const Real py = parent_eta(e, side, q);
            const Real pz = parent_zeta(e, side, q);
            const Real nx = normal_x(e, side, q);
#if LIBMESH_DIM > 1
            const Real ny = normal_y(e, side, q);
#else
            const Real ny = Real(0);
#endif
#if LIBMESH_DIM > 2
            const Real nz = normal_z(e, side, q);
#else
            const Real nz = Real(0);
#endif
            Real normal_norm2 = nx * nx;
#if LIBMESH_DIM > 1
            normal_norm2 += ny * ny;
#endif
#if LIBMESH_DIM > 2
            normal_norm2 += nz * nz;
#endif
            Real q_normal = Real(0);
            for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
            {
              Real q_value = Real(0);
              for (unsigned int j = 0; j != layout.vector_n_dofs; ++j)
              {
                const libMesh::Kokkos::RealVector phi_j =
                    libMesh::Kokkos::vector_shape(vector_key, j, px, py, pz);
                const Real q_dof = component == 0 ? u_q_dofs(e, j) : v_q_dofs(e, j);
                q_value += Kokkos::detail::component_or_zero(phi_j, d) * q_dof;
              }
              const Real normal_component =
                  d == 0 ? nx :
#if LIBMESH_DIM > 1
                           d == 1 ? ny :
#endif
                                    nz;
              q_normal += q_value * normal_component;
            }

            Real u_value = Real(0);
            Real p_value = Real(0);
            for (unsigned int j = 0; j != layout.scalar_n_dofs; ++j)
            {
              const Real phi = libMesh::Kokkos::shape(scalar_key, j, px, py, pz);
              const Real u_dof = component == 0 ? u_u_dofs(e, j) : v_u_dofs(e, j);
              u_value += phi * u_dof;
              p_value += phi * u_p_dofs(e, j);
            }

            Real lm_value = Real(0);
            Real other_lm_value = Real(0);
            for (unsigned int j = 0; j != layout.lm_n_dofs; ++j)
            {
              const Real trace = libMesh::Kokkos::side_trace_shape(
                  lm_key, j, side, positive_edge_orientation, px, py, pz);
              const Real lm_dof = component == 0 ? u_lm_dofs(e, j) : v_lm_dofs(e, j);
              const Real other_lm_dof =
                  component == 0 ? u_other_lm_dofs(e, j) : v_other_lm_dofs(e, j);
              lm_value += trace * lm_dof;
              other_lm_value += trace * other_lm_dof;
            }

            const Real boundary_value =
                is_dirichlet
                    ? (component == 0 ? dirichlet_component_0(e, side, q)
                                      : dirichlet_component_1(e, side, q))
                    : lm_value;
            const Real advective_vdotn =
                velocity_component == 0 ? nx * lm_value + ny * other_lm_value
                                        : nx * other_lm_value + ny * lm_value;
            const Real advective_flux =
                !include_convection ? Real(0)
                : is_dirichlet      ? dirichlet_vdotn(e, side, q) * boundary_value
                                    : advective_vdotn * lm_value;
            const Real normal_component = velocity_component == 0 ? nx : ny;
            const Real linear_flux =
                is_outlet ? Real(0)
                          : -nu * q_normal + p_value * normal_component +
                                tau * (u_value - boundary_value) * normal_norm2;
            value += side_jxw(e, side, q) *
                     libMesh::Kokkos::shape(scalar_key, i, px, py, pz) *
                     (linear_flux + advective_flux);
          }
        }
        else if (idx < lm0)
        {
          const unsigned int i = idx - p0;
          for (unsigned int q = 0; q != n_side_qp; ++q)
          {
            const Real px = parent_xi(e, side, q);
            const Real py = parent_eta(e, side, q);
            const Real pz = parent_zeta(e, side, q);
            const Real nx = normal_x(e, side, q);
#if LIBMESH_DIM > 1
            const Real ny = normal_y(e, side, q);
#else
            const Real ny = Real(0);
#endif
            Real lm_value = Real(0);
            if (!is_dirichlet)
              for (unsigned int j = 0; j != layout.lm_n_dofs; ++j)
              {
                const Real trace = libMesh::Kokkos::side_trace_shape(
                    lm_key, j, side, positive_edge_orientation, px, py, pz);
                const Real lm_dof = component == 0 ? u_lm_dofs(e, j) : v_lm_dofs(e, j);
                lm_value += trace * lm_dof;
              }
            const Real boundary_value =
                is_dirichlet
                    ? (component == 0 ? dirichlet_component_0(e, side, q)
                                      : dirichlet_component_1(e, side, q))
                    : lm_value;
            const Real normal_component = velocity_component == 0 ? nx : ny;
            value += side_jxw(e, side, q) *
                     libMesh::Kokkos::shape(scalar_key, i, px, py, pz) * boundary_value *
                     normal_component;
          }
        }
        else
        {
          const unsigned int i = idx - lm0;
          for (unsigned int q = 0; q != n_side_qp; ++q)
          {
            const Real px = parent_xi(e, side, q);
            const Real py = parent_eta(e, side, q);
            const Real pz = parent_zeta(e, side, q);
            const Real phi_i = libMesh::Kokkos::side_trace_shape(
                lm_key, i, side, positive_edge_orientation, px, py, pz);
            Real lm_value = Real(0);
            for (unsigned int j = 0; j != layout.lm_n_dofs; ++j)
            {
              const Real trace = libMesh::Kokkos::side_trace_shape(
                  lm_key, j, side, positive_edge_orientation, px, py, pz);
              const Real lm_dof = component == 0 ? u_lm_dofs(e, j) : v_lm_dofs(e, j);
              lm_value += trace * lm_dof;
            }

            if (is_dirichlet)
            {
              value -= side_jxw(e, side, q) * phi_i * lm_value;
              continue;
            }

            const Real nx = normal_x(e, side, q);
#if LIBMESH_DIM > 1
            const Real ny = normal_y(e, side, q);
#else
            const Real ny = Real(0);
#endif
#if LIBMESH_DIM > 2
            const Real nz = normal_z(e, side, q);
#else
            const Real nz = Real(0);
#endif
            Real normal_norm2 = nx * nx;
#if LIBMESH_DIM > 1
            normal_norm2 += ny * ny;
#endif
#if LIBMESH_DIM > 2
            normal_norm2 += nz * nz;
#endif
            Real q_normal = Real(0);
            for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
            {
              Real q_value = Real(0);
              for (unsigned int j = 0; j != layout.vector_n_dofs; ++j)
              {
                const libMesh::Kokkos::RealVector phi_j =
                    libMesh::Kokkos::vector_shape(vector_key, j, px, py, pz);
                const Real q_dof = component == 0 ? u_q_dofs(e, j) : v_q_dofs(e, j);
                q_value += Kokkos::detail::component_or_zero(phi_j, d) * q_dof;
              }
              const Real normal_component =
                  d == 0 ? nx :
#if LIBMESH_DIM > 1
                           d == 1 ? ny :
#endif
                                    nz;
              q_normal += q_value * normal_component;
            }

            Real u_value = Real(0);
            Real p_value = Real(0);
            for (unsigned int j = 0; j != layout.scalar_n_dofs; ++j)
            {
              const Real phi = libMesh::Kokkos::shape(scalar_key, j, px, py, pz);
              const Real u_dof = component == 0 ? u_u_dofs(e, j) : v_u_dofs(e, j);
              u_value += phi * u_dof;
              p_value += phi * u_p_dofs(e, j);
            }

            Real other_lm_value = Real(0);
            for (unsigned int j = 0; j != layout.lm_n_dofs; ++j)
            {
              const Real trace = libMesh::Kokkos::side_trace_shape(
                  lm_key, j, side, positive_edge_orientation, px, py, pz);
              const Real other_lm_dof =
                  component == 0 ? u_other_lm_dofs(e, j) : v_other_lm_dofs(e, j);
              other_lm_value += trace * other_lm_dof;
            }

            const Real advective_vdotn =
                velocity_component == 0 ? nx * lm_value + ny * other_lm_value
                                        : nx * other_lm_value + ny * lm_value;
            const Real advective_flux =
                (!include_convection || is_outlet) ? Real(0) : advective_vdotn * lm_value;
            const Real normal_component = velocity_component == 0 ? nx : ny;
            value += side_jxw(e, side, q) * phi_i *
                     (-nu * q_normal + p_value * normal_component +
                      tau * (u_value - lm_value) * normal_norm2 + advective_flux);
          }
        }

        if (component == 0)
          u_residual(e, idx) += value;
        else
          v_residual(e, idx) += value;
      });
}

template <ElemType ExactTopo>
void
assemble_full_convective_volume_blocks_range(const DeviceAssemblyData & data,
                                             const ComponentDofViews & u_dofs,
                                             const ComponentDofViews & v_dofs,
                                             const Ex9FullBlockLayout full_layout,
                                             const DeviceReal2D & full_blocks)
{
  const unsigned int n_elem = data.n_elem;
  const HDGLinearBlockLayout layout = data.layout;
  const unsigned int n_entries = layout.scalar_n_dofs * layout.scalar_n_dofs;
  const FEShapeKey scalar_key = data.scalar_key;
  const DeviceReal3D coords = data.coords;
  const unsigned int n_geom_nodes = data.n_geom_nodes;
  const DeviceReal1D xi = data.xi;
  const DeviceReal1D eta = data.eta;
  const DeviceReal1D zeta = data.zeta;
  const DeviceReal1D jxw = data.jxw;
  const unsigned int n_qp = data.n_qp;
  const DeviceReal2D u_values = u_dofs.u_dofs;
  const DeviceReal2D v_values = v_dofs.u_dofs;
  const unsigned int n_full = full_layout.n_dofs;

  ::Kokkos::parallel_for(
      "vector_fe_ex9_split_full_convective_volume_blocks",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_elem * n_entries)),
      KOKKOS_LAMBDA(const int raw_entry) {
        const unsigned int entry = static_cast<unsigned int>(raw_entry);
        const unsigned int e = entry / n_entries;
        const unsigned int idx = entry - e * n_entries;
        const unsigned int i = idx / layout.scalar_n_dofs;
        const unsigned int j = idx - i * layout.scalar_n_dofs;
        Real u_u_col = Real(0);
        Real ux_v_col = Real(0);
        Real vy_u_col = Real(0);
        Real v_v_col = Real(0);

        for (unsigned int q = 0; q != n_qp; ++q)
        {
          const Real xiq = xi(q);
          const Real etaq = eta(q);
          const Real zetaq = zeta(q);
          const libMesh::Kokkos::RealTensor J =
              libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                  Kokkos::detail::HDGElementNodeView<DeviceReal3D>{coords, e},
                  n_geom_nodes,
                  xiq,
                  etaq,
                  zetaq);
          const libMesh::Kokkos::RealTensor Jinv =
              libMesh::Kokkos::inverse<libMesh::Kokkos::RealTensor>(J, 2);
          const libMesh::Kokkos::RealVector grad_phys =
              Jinv * libMesh::Kokkos::grad_shape(scalar_key, i, xiq, etaq, zetaq);
          const Real phi_j = libMesh::Kokkos::shape(scalar_key, j, xiq, etaq, zetaq);
          Real u_value = Real(0);
          Real v_value = Real(0);
          for (unsigned int k = 0; k != layout.scalar_n_dofs; ++k)
          {
            const Real phi_k = libMesh::Kokkos::shape(scalar_key, k, xiq, etaq, zetaq);
            u_value += phi_k * u_values(e, k);
            v_value += phi_k * v_values(e, k);
          }

          const Real grad_x = Kokkos::detail::component_or_zero(grad_phys, 0);
          const Real grad_y = Kokkos::detail::component_or_zero(grad_phys, 1);
          u_u_col -= jxw(q) * phi_j * (Real(2) * grad_x * u_value + grad_y * v_value);
          ux_v_col -= jxw(q) * grad_y * phi_j * u_value;
          vy_u_col -= jxw(q) * grad_x * phi_j * v_value;
          v_v_col -= jxw(q) * phi_j * (grad_x * u_value + Real(2) * grad_y * v_value);
        }

        full_blocks(e, (full_layout.off_u + i) * n_full + full_layout.off_u + j) += u_u_col;
        full_blocks(e, (full_layout.off_u + i) * n_full + full_layout.off_v + j) += ux_v_col;
        full_blocks(e, (full_layout.off_v + i) * n_full + full_layout.off_u + j) += vy_u_col;
        full_blocks(e, (full_layout.off_v + i) * n_full + full_layout.off_v + j) += v_v_col;
      });
}

void
assemble_full_convective_side_blocks_range(const DeviceAssemblyData & data,
                                           const ComponentDofViews & u_dofs,
                                           const ComponentDofViews & v_dofs,
                                           const Ex9FullBlockLayout full_layout,
                                           const DeviceReal2D & full_blocks)
{
  const unsigned int n_elem = data.n_elem;
  const HDGLinearBlockLayout layout = data.layout;
  const unsigned int scalar_lm_entries = layout.scalar_n_dofs * layout.lm_n_dofs;
  const unsigned int lm_lm_entries = layout.lm_n_dofs * layout.lm_n_dofs;
  const unsigned int entries_per_side = scalar_lm_entries + lm_lm_entries;
  const FEShapeKey scalar_key = data.scalar_key;
  const FEShapeKey lm_key = data.lm_key;
  const DeviceReal3D parent_xi = data.parent_xi;
  const DeviceReal3D parent_eta = data.parent_eta;
  const DeviceReal3D parent_zeta = data.parent_zeta;
  const DeviceReal3D side_jxw = data.side_jxw;
  const DeviceReal3D normal_x = data.normal_x;
  const DeviceReal3D normal_y = data.normal_y;
  const DeviceUChar2D orientations = data.orientations;
  const DeviceInt2D side_types = data.side_types;
  const unsigned int n_sides = data.n_sides;
  const unsigned int n_side_qp = data.n_side_qp;
  const DeviceReal2D lm_u_values = u_dofs.lm_dofs;
  const DeviceReal2D lm_v_values = v_dofs.lm_dofs;
  const unsigned int n_full = full_layout.n_dofs;

  ::Kokkos::parallel_for(
      "vector_fe_ex9_split_full_convective_side_blocks",
      ::Kokkos::RangePolicy<>(0, cast_int<int>(n_elem * n_sides * entries_per_side)),
      KOKKOS_LAMBDA(const int raw_entry) {
        const unsigned int entry = static_cast<unsigned int>(raw_entry);
        const unsigned int entries_per_elem = n_sides * entries_per_side;
        const unsigned int e = entry / entries_per_elem;
        const unsigned int elem_entry = entry - e * entries_per_elem;
        const unsigned int side = elem_entry / entries_per_side;
        const unsigned int side_entry = elem_entry - side * entries_per_side;
        const bool positive_edge_orientation = static_cast<bool>(orientations(e, side));
        const int side_type = side_types(e, side);
        const bool is_dirichlet = side_type == Kokkos::detail::hdg_linear_side_dirichlet;
        const bool is_outlet = side_type == Kokkos::detail::hdg_linear_side_outlet;

        if (is_dirichlet)
          return;

        if (side_entry < scalar_lm_entries)
        {
          const unsigned int i = side_entry / layout.lm_n_dofs;
          const unsigned int j = side_entry - i * layout.lm_n_dofs;
          Real u_lmu = Real(0);
          Real u_lmv = Real(0);
          Real v_lmu = Real(0);
          Real v_lmv = Real(0);

          for (unsigned int q = 0; q != n_side_qp; ++q)
          {
            const Real px = parent_xi(e, side, q);
            const Real py = parent_eta(e, side, q);
            const Real pz = parent_zeta(e, side, q);
            const Real phi_i = libMesh::Kokkos::shape(scalar_key, i, px, py, pz);
            const Real trace_j =
                libMesh::Kokkos::side_trace_shape(
                    lm_key, j, side, positive_edge_orientation, px, py, pz);
            Real lm_u_value = Real(0);
            Real lm_v_value = Real(0);
            for (unsigned int k = 0; k != layout.lm_n_dofs; ++k)
            {
              const Real trace_k =
                  libMesh::Kokkos::side_trace_shape(
                      lm_key, k, side, positive_edge_orientation, px, py, pz);
              lm_u_value += trace_k * lm_u_values(e, k);
              lm_v_value += trace_k * lm_v_values(e, k);
            }
            const Real nx = normal_x(e, side, q);
            const Real ny = normal_y(e, side, q);

            u_lmu += side_jxw(e, side, q) * phi_i *
                     (Real(2) * nx * lm_u_value + ny * lm_v_value) * trace_j;
            u_lmv += side_jxw(e, side, q) * phi_i * ny * lm_u_value * trace_j;
            v_lmu += side_jxw(e, side, q) * phi_i * nx * lm_v_value * trace_j;
            v_lmv += side_jxw(e, side, q) * phi_i *
                     (nx * lm_u_value + Real(2) * ny * lm_v_value) * trace_j;
          }

          full_blocks(e, (full_layout.off_u + i) * n_full + full_layout.off_lm_u + j) += u_lmu;
          full_blocks(e, (full_layout.off_u + i) * n_full + full_layout.off_lm_v + j) += u_lmv;
          full_blocks(e, (full_layout.off_v + i) * n_full + full_layout.off_lm_u + j) += v_lmu;
          full_blocks(e, (full_layout.off_v + i) * n_full + full_layout.off_lm_v + j) += v_lmv;
          return;
        }

        if (is_outlet)
          return;

        const unsigned int idx = side_entry - scalar_lm_entries;
        const unsigned int i = idx / layout.lm_n_dofs;
        const unsigned int j = idx - i * layout.lm_n_dofs;
        Real lmu_lmu = Real(0);
        Real lmu_lmv = Real(0);
        Real lmv_lmu = Real(0);
        Real lmv_lmv = Real(0);

        for (unsigned int q = 0; q != n_side_qp; ++q)
        {
          const Real px = parent_xi(e, side, q);
          const Real py = parent_eta(e, side, q);
          const Real pz = parent_zeta(e, side, q);
          const Real trace_i =
              libMesh::Kokkos::side_trace_shape(
                  lm_key, i, side, positive_edge_orientation, px, py, pz);
          const Real trace_j =
              libMesh::Kokkos::side_trace_shape(
                  lm_key, j, side, positive_edge_orientation, px, py, pz);
          Real lm_u_value = Real(0);
          Real lm_v_value = Real(0);
          for (unsigned int k = 0; k != layout.lm_n_dofs; ++k)
          {
            const Real trace_k =
                libMesh::Kokkos::side_trace_shape(
                    lm_key, k, side, positive_edge_orientation, px, py, pz);
            lm_u_value += trace_k * lm_u_values(e, k);
            lm_v_value += trace_k * lm_v_values(e, k);
          }
          const Real nx = normal_x(e, side, q);
          const Real ny = normal_y(e, side, q);

          lmu_lmu += side_jxw(e, side, q) * trace_i *
                     (Real(2) * nx * lm_u_value + ny * lm_v_value) * trace_j;
          lmu_lmv += side_jxw(e, side, q) * trace_i * ny * lm_u_value * trace_j;
          lmv_lmu += side_jxw(e, side, q) * trace_i * nx * lm_v_value * trace_j;
          lmv_lmv += side_jxw(e, side, q) * trace_i *
                     (nx * lm_u_value + Real(2) * ny * lm_v_value) * trace_j;
        }

        full_blocks(e, (full_layout.off_lm_u + i) * n_full + full_layout.off_lm_u + j) +=
            lmu_lmu;
        full_blocks(e, (full_layout.off_lm_u + i) * n_full + full_layout.off_lm_v + j) +=
            lmu_lmv;
        full_blocks(e, (full_layout.off_lm_v + i) * n_full + full_layout.off_lm_u + j) +=
            lmv_lmu;
        full_blocks(e, (full_layout.off_lm_v + i) * n_full + full_layout.off_lm_v + j) +=
            lmv_lmv;
      });
}

template <ElemType ExactTopo>
bool
assemble_component_pair_into_full(const DeviceAssemblyData & u_data,
                                  const DeviceAssemblyData & v_data,
                                  const ComponentDofViews & u_dofs,
                                  const ComponentDofViews & v_dofs,
                                  const Ex9FullBlockLayout full_layout,
                                  const DeviceReal2D & full_blocks,
                                  const DeviceReal2D & full_residual,
                                  ComponentBatchWork & u_work,
                                  ComponentBatchWork & v_work,
                                  const bool assemble_residual)
{
  const unsigned int n_elem = u_data.n_elem;
  libmesh_error_msg_if(!n_elem, "vector_fe_ex9 Kokkos HDG assembly received an empty batch.");
  libmesh_error_msg_if(v_data.n_elem != n_elem || v_data.layout.n_entries != u_data.layout.n_entries,
                       "vector_fe_ex9 direct Kokkos HDG assembly received inconsistent batches.");

  DeviceReal2D u_blocks = u_work.blocks;
  DeviceReal2D v_blocks = v_work.blocks;
  DeviceReal2D u_residual = u_work.residual;
  DeviceReal2D v_residual = v_work.residual;
  DeviceInt1D u_condense_status = u_work.condense_status;
  DeviceInt1D u_residual_status = u_work.residual_status;
  DeviceInt1D v_condense_status = v_work.condense_status;
  DeviceInt1D v_residual_status = v_work.residual_status;

  {
    Ex9KokkosTimer timer("direct full component kernel");
    Kokkos::detail::assemble_hdg_linear_batch_boundary_pair<ExactTopo>(
        u_data.vector_key,
        u_data.scalar_key,
        u_data.lm_key,
        u_data.layout,
        u_data.coords,
        n_elem,
        u_data.n_geom_nodes,
        u_data.xi,
        u_data.eta,
        u_data.zeta,
        u_data.jxw,
        u_data.scalar_forcing,
        v_data.scalar_forcing,
        u_data.pressure_forcing,
        u_data.n_qp,
        u_data.parent_xi,
        u_data.parent_eta,
        u_data.parent_zeta,
        u_data.side_jxw,
        u_data.normal_x,
        u_data.normal_y,
        u_data.normal_z,
        u_data.orientations,
        u_data.side_types,
        u_data.dirichlet_component,
        v_data.dirichlet_component,
        u_data.dirichlet_vdotn,
        u_data.n_sides,
        u_data.n_side_qp,
        u_data.problem->nu,
        Real(1),
        true,
        u_dofs.q_dofs,
        u_dofs.u_dofs,
        u_dofs.lm_dofs,
        u_dofs.other_u_dofs,
        u_dofs.other_lm_dofs,
        v_dofs.q_dofs,
        v_dofs.u_dofs,
        u_dofs.p_dofs,
        v_dofs.lm_dofs,
        v_dofs.other_u_dofs,
        v_dofs.other_lm_dofs,
        u_blocks,
        u_residual,
        v_blocks,
        v_residual,
        full_blocks,
        false,
        full_layout.n_dofs,
        full_layout.off_u,
        full_layout.off_v,
        full_layout.off_lm_u,
        full_layout.off_lm_v,
        u_condense_status,
        u_residual_status,
        v_condense_status,
        v_residual_status,
        "vector_fe_ex9_hdg_direct_full_pair_batch",
        ex9_kokkos_team_size(),
        false,
        false,
        false);
  }

  if (assemble_residual)
  {
    Ex9KokkosTimer timer("split component residual zero");
    ::Kokkos::deep_copy(u_residual, Real(0));
    ::Kokkos::deep_copy(v_residual, Real(0));
  }

  {
    Ex9KokkosTimer timer("split component volume blocks kernel");
    assemble_component_pair_volume_blocks_range<ExactTopo>(u_data, u_blocks, v_blocks);
  }
  {
    Ex9KokkosTimer timer("split component side blocks kernel");
    assemble_component_pair_side_blocks_range(u_data, u_blocks, v_blocks);
  }
  {
    Ex9KokkosTimer timer("split full convective volume blocks kernel");
    assemble_full_convective_volume_blocks_range<ExactTopo>(
        u_data, u_dofs, v_dofs, full_layout, full_blocks);
  }
  {
    Ex9KokkosTimer timer("split full convective side blocks kernel");
    assemble_full_convective_side_blocks_range(u_data, u_dofs, v_dofs, full_layout, full_blocks);
  }
  if (assemble_residual)
  {
    {
      Ex9KokkosTimer timer("split component volume residual kernel");
      assemble_component_pair_volume_residual_range<ExactTopo>(
          u_data, v_data, u_dofs, v_dofs, u_residual, v_residual);
    }
    {
      Ex9KokkosTimer timer("split component side residual kernel");
      assemble_component_pair_side_residual_range(
          u_data, v_data, u_dofs, v_dofs, u_residual, v_residual);
    }
  }

  int n_failures = 0;
  {
    Ex9KokkosTimer timer("direct full component status reduction");
    ::Kokkos::parallel_reduce(
        "vector_fe_ex9_direct_full_component_status_reduce",
        ::Kokkos::RangePolicy<>(0, cast_int<int>(n_elem)),
        KOKKOS_LAMBDA(const int e, int & update) {
          if (u_condense_status(e) || u_residual_status(e) || v_condense_status(e) ||
              v_residual_status(e))
            ++update;
        },
        n_failures);
  }

  if (!n_failures)
  {
    Ex9KokkosTimer timer("scatter component blocks to full");
    scatter_component_pair_into_full(u_data,
                                     v_data,
                                     full_layout,
                                     u_blocks,
                                     u_residual,
                                     v_blocks,
                                     v_residual,
                                     full_blocks,
                                     full_residual,
                                     assemble_residual);
    return true;
  }

  auto h_u_condense_status = ::Kokkos::create_mirror_view(u_condense_status);
  auto h_u_residual_status = ::Kokkos::create_mirror_view(u_residual_status);
  auto h_v_condense_status = ::Kokkos::create_mirror_view(v_condense_status);
  auto h_v_residual_status = ::Kokkos::create_mirror_view(v_residual_status);
  {
    Ex9KokkosTimer timer("direct full component failure diagnostics copy");
    ::Kokkos::deep_copy(h_u_condense_status, u_condense_status);
    ::Kokkos::deep_copy(h_u_residual_status, u_residual_status);
    ::Kokkos::deep_copy(h_v_condense_status, v_condense_status);
    ::Kokkos::deep_copy(h_v_residual_status, v_residual_status);
  }

  for (unsigned int e = 0; e != n_elem; ++e)
  {
    libmesh_error_msg_if(h_u_condense_status(e),
                         "vector_fe_ex9 Kokkos HDG direct full assembly failed for u elem "
                             << u_data.elems[e]->id() << " with status "
                             << h_u_condense_status(e));
    libmesh_error_msg_if(h_u_residual_status(e),
                         "vector_fe_ex9 Kokkos HDG direct full residual assembly failed for u elem "
                             << u_data.elems[e]->id() << " with status "
                             << h_u_residual_status(e));
    libmesh_error_msg_if(h_v_condense_status(e),
                         "vector_fe_ex9 Kokkos HDG direct full assembly failed for v elem "
                             << v_data.elems[e]->id() << " with status "
                             << h_v_condense_status(e));
    libmesh_error_msg_if(h_v_residual_status(e),
                         "vector_fe_ex9 Kokkos HDG direct full residual assembly failed for v elem "
                             << v_data.elems[e]->id() << " with status "
                             << h_v_residual_status(e));
  }

  return true;
}

bool
assemble_component_pair_into_full_dispatch(const DeviceAssemblyData & u_data,
                                           const DeviceAssemblyData & v_data,
                                           const ComponentDofViews & u_dofs,
                                           const ComponentDofViews & v_dofs,
                                           const Ex9FullBlockLayout full_layout,
                                           const DeviceReal2D & full_blocks,
                                           const DeviceReal2D & full_residual,
                                           ComponentBatchWork & u_work,
                                           ComponentBatchWork & v_work,
                                           const bool assemble_residual)
{
  switch (u_data.elem_type)
  {
    case TRI6:
      return assemble_component_pair_into_full<TRI6>(
          u_data,
          v_data,
          u_dofs,
          v_dofs,
          full_layout,
          full_blocks,
          full_residual,
          u_work,
          v_work,
          assemble_residual);
    case TRI7:
      return assemble_component_pair_into_full<TRI7>(
          u_data,
          v_data,
          u_dofs,
          v_dofs,
          full_layout,
          full_blocks,
          full_residual,
          u_work,
          v_work,
          assemble_residual);
    default:
      return false;
  }
}

template <typename BlockStorage, typename CondensedStorage, typename StatusStorage>
void
condense_ex9_full_on_device(const Ex9FullBlockLayout layout,
                            const unsigned int n_elem,
                            const BlockStorage & blocks,
                            const CondensedStorage & condensed_blocks,
                            const CondensedStorage & internal_inverse,
                            const CondensedStorage & internal_external_solve,
                            const StatusStorage & block_status,
                            const DeviceInt1D & failure_pivot,
                            const DeviceReal1D & failure_pivot_abs,
                            const DeviceReal1D & failure_initial_diag,
                            const DeviceReal1D & failure_initial_row_abs)
{
  constexpr unsigned int max_internal = Kokkos::detail::hdg_linear_condense_max_internal;
  constexpr unsigned int max_external = Kokkos::detail::hdg_linear_condense_max_external;
  constexpr unsigned int max_cols = Kokkos::detail::hdg_linear_condense_max_cols;

  ::Kokkos::parallel_for(
      "vector_fe_ex9_full_condense",
      ::Kokkos::RangePolicy<>(0, static_cast<int>(n_elem)),
      KOKKOS_LAMBDA(const int raw_e) {
        const unsigned int e = static_cast<unsigned int>(raw_e);
        const unsigned int n_internal = layout.internal_n_dofs;
        const unsigned int n_external = layout.external_n_dofs;

        for (unsigned int i = 0; i != n_external * n_external; ++i)
          condensed_blocks(e, i) = Real(0);
        for (unsigned int i = 0; i != n_internal * n_internal; ++i)
          internal_inverse(e, i) = Real(0);
        for (unsigned int i = 0; i != n_internal * n_external; ++i)
          internal_external_solve(e, i) = Real(0);
        failure_pivot(e) = -1;
        failure_pivot_abs(e) = Real(0);
        failure_initial_diag(e) = Real(0);
        failure_initial_row_abs(e) = Real(0);

        if (n_internal > max_internal || n_external > max_external ||
            n_internal + n_external > max_cols)
        {
          block_status(e) = 2;
          failure_pivot(e) = static_cast<int>(n_internal);
          return;
        }

        Real aug_blocks[max_internal * max_cols];
        Real aug_inverse[max_internal * max_internal];
        Real block_solve[max_internal * max_external];

        for (unsigned int i = 0; i != n_internal; ++i)
        {
          for (unsigned int j = 0; j != n_internal; ++j)
          {
            const Real value = blocks(e, i * layout.n_dofs + j);
            aug_blocks[i * max_cols + j] = value;
            aug_inverse[i * max_internal + j] = i == j ? Real(1) : Real(0);
          }
          for (unsigned int j = 0; j != n_external; ++j)
            aug_blocks[i * max_cols + n_internal + j] =
                blocks(e, i * layout.n_dofs + layout.off_p + j);
        }

        block_status(e) = 0;
        for (unsigned int k = 0; k != n_internal; ++k)
        {
          unsigned int pivot_row = k;
          Real pivot_abs = Kokkos::detail::abs_real(aug_blocks[k * max_cols + k]);
          for (unsigned int r = k + 1; r != n_internal; ++r)
          {
            const Real candidate = Kokkos::detail::abs_real(aug_blocks[r * max_cols + k]);
            if (candidate > pivot_abs)
            {
              pivot_abs = candidate;
              pivot_row = r;
            }
          }

          if (pivot_abs <= Real(1.e-20))
          {
            block_status(e) = 1;
            failure_pivot(e) = static_cast<int>(k);
            failure_pivot_abs(e) = pivot_abs;
            failure_initial_diag(e) = blocks(e, k * layout.n_dofs + k);
            Real row_abs = Real(0);
            for (unsigned int c = 0; c != n_internal; ++c)
            {
              const Real candidate = Kokkos::detail::abs_real(blocks(e, k * layout.n_dofs + c));
              if (candidate > row_abs)
                row_abs = candidate;
            }
            failure_initial_row_abs(e) = row_abs;
            return;
          }

          if (pivot_row != k)
            for (unsigned int c = 0; c != n_internal + n_external; ++c)
            {
              const Real tmp = aug_blocks[k * max_cols + c];
              aug_blocks[k * max_cols + c] = aug_blocks[pivot_row * max_cols + c];
              aug_blocks[pivot_row * max_cols + c] = tmp;
            }
          if (pivot_row != k)
            for (unsigned int c = 0; c != n_internal; ++c)
            {
              const Real tmp = aug_inverse[k * max_internal + c];
              aug_inverse[k * max_internal + c] = aug_inverse[pivot_row * max_internal + c];
              aug_inverse[pivot_row * max_internal + c] = tmp;
            }

          const Real pivot = aug_blocks[k * max_cols + k];
          for (unsigned int r = k + 1; r != n_internal; ++r)
          {
            const Real factor = aug_blocks[r * max_cols + k] / pivot;
            aug_blocks[r * max_cols + k] = Real(0);

            for (unsigned int c = k + 1; c != n_internal + n_external; ++c)
              aug_blocks[r * max_cols + c] -= factor * aug_blocks[k * max_cols + c];
            for (unsigned int c = 0; c != n_internal; ++c)
              aug_inverse[r * max_internal + c] -= factor * aug_inverse[k * max_internal + c];
          }
        }

        for (unsigned int rhs = 0; rhs != n_external; ++rhs)
          for (unsigned int rr = n_internal; rr != 0; --rr)
          {
            const unsigned int r = rr - 1;
            Real value = aug_blocks[r * max_cols + n_internal + rhs];
            for (unsigned int c = r + 1; c != n_internal; ++c)
              value -= aug_blocks[r * max_cols + c] * block_solve[c * max_external + rhs];
            block_solve[r * max_external + rhs] = value / aug_blocks[r * max_cols + r];
          }

        for (unsigned int rhs = 0; rhs != n_internal; ++rhs)
          for (unsigned int rr = n_internal; rr != 0; --rr)
          {
            const unsigned int r = rr - 1;
            Real value = aug_inverse[r * max_internal + rhs];
            for (unsigned int c = r + 1; c != n_internal; ++c)
              value -= aug_blocks[r * max_cols + c] * internal_inverse(e, c * n_internal + rhs);
            internal_inverse(e, r * n_internal + rhs) = value / aug_blocks[r * max_cols + r];
          }

        for (unsigned int i = 0; i != n_internal; ++i)
          for (unsigned int j = 0; j != n_external; ++j)
            internal_external_solve(e, i * n_external + j) = block_solve[i * max_external + j];

        for (unsigned int i = 0; i != n_external; ++i)
          for (unsigned int j = 0; j != n_external; ++j)
          {
            Real value = blocks(e, (layout.off_p + i) * layout.n_dofs + layout.off_p + j);
            for (unsigned int k = 0; k != n_internal; ++k)
              value -= blocks(e, (layout.off_p + i) * layout.n_dofs + k) *
                       block_solve[k * max_external + j];
            condensed_blocks(e, i * n_external + j) = value;
          }
      });
}

void
check_ex9_full_condense_status(const Ex9FullBatch & full,
                               const DeviceAssemblyData & component_data,
                               const DeviceReal2D & full_blocks,
                               const DeviceReal2D & u_blocks,
                               const DeviceReal2D & v_blocks,
                               const DeviceInt1D & condense_status,
                               const DeviceInt1D & failure_pivot,
                               const DeviceReal1D & failure_pivot_abs,
                               const DeviceReal1D & failure_initial_diag,
                               const DeviceReal1D & failure_initial_row_abs)
{
  int n_failures = 0;
  {
    Ex9KokkosTimer timer("full condense status reduction");
    ::Kokkos::parallel_reduce(
        "vector_fe_ex9_full_condense_status_reduce",
        ::Kokkos::RangePolicy<>(0, cast_int<int>(full.elems.size())),
        KOKKOS_LAMBDA(const int e, int & update) {
          if (condense_status(e))
            ++update;
        },
        n_failures);
  }

  if (!n_failures)
    return;

  auto h_condense_status = ::Kokkos::create_mirror_view(condense_status);
  auto h_failure_pivot = ::Kokkos::create_mirror_view(failure_pivot);
  auto h_failure_pivot_abs = ::Kokkos::create_mirror_view(failure_pivot_abs);
  auto h_failure_initial_diag = ::Kokkos::create_mirror_view(failure_initial_diag);
  auto h_failure_initial_row_abs = ::Kokkos::create_mirror_view(failure_initial_row_abs);
  {
    Ex9KokkosTimer timer("full condense failure diagnostics copy");
    ::Kokkos::deep_copy(h_condense_status, condense_status);
    ::Kokkos::deep_copy(h_failure_pivot, failure_pivot);
    ::Kokkos::deep_copy(h_failure_pivot_abs, failure_pivot_abs);
    ::Kokkos::deep_copy(h_failure_initial_diag, failure_initial_diag);
    ::Kokkos::deep_copy(h_failure_initial_row_abs, failure_initial_row_abs);
  }

  if (ex9_kokkos_timers_enabled())
  {
    const HDGLinearBlockLayout component_layout = component_data.layout;
    DeviceReal1D basis_diagnostics("vector_fe_ex9_basis_diagnostics", 8);
    DeviceReal2D write_marker("vector_fe_ex9_write_marker", 1, component_layout.n_entries);
    {
      Ex9KokkosTimer timer("full condense independent Jqq diagnostics kernel");
      const FEShapeKey vector_key = component_data.vector_key;
      const DeviceReal1D xi = component_data.xi;
      const DeviceReal1D eta = component_data.eta;
      const DeviceReal1D zeta = component_data.zeta;
      const DeviceReal1D jxw = component_data.jxw;
      const unsigned int n_qp = component_data.n_qp;
      ::Kokkos::parallel_for(
          "vector_fe_ex9_independent_jqq_diagnostics",
          ::Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(const int) {
            Real jqq00 = Real(0);
            Real q0_phi0_norm2 = Real(0);
            Real q0_phi0_component0 = Real(0);
            Real q0_phi0_component1 = Real(0);
            Real q0_jxw = n_qp ? jxw(0) : Real(0);

            for (unsigned int q = 0; q != n_qp; ++q)
            {
              const libMesh::Kokkos::RealVector phi0 =
                  libMesh::Kokkos::vector_shape(vector_key, 0, xi(q), eta(q), zeta(q));
              Real dot = Real(0);
              for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                dot += Kokkos::detail::component_or_zero(phi0, d) *
                       Kokkos::detail::component_or_zero(phi0, d);
              jqq00 += jxw(q) * dot;

              if (q == 0)
              {
                q0_phi0_norm2 = dot;
                q0_phi0_component0 = Kokkos::detail::component_or_zero(phi0, 0);
                q0_phi0_component1 = Kokkos::detail::component_or_zero(phi0, 1);
              }
            }

            basis_diagnostics(0) = jqq00;
            basis_diagnostics(1) = q0_phi0_norm2;
            basis_diagnostics(2) = q0_jxw;
            basis_diagnostics(3) = q0_phi0_component0;
            basis_diagnostics(4) = q0_phi0_component1;
            basis_diagnostics(5) = static_cast<Real>(vector_component_count_or_zero(vector_key.elem_type));
            basis_diagnostics(6) = static_cast<Real>(supports_vector_shape(vector_key) ? 1 : 0);
            write_marker(0, 0) = Real(123.25);
            basis_diagnostics(7) = write_marker(0, 0);
          });
    }

    auto h_full_blocks = ::Kokkos::create_mirror_view(full_blocks);
    auto h_u_blocks = ::Kokkos::create_mirror_view(u_blocks);
    auto h_v_blocks = ::Kokkos::create_mirror_view(v_blocks);
    auto h_basis_diagnostics = ::Kokkos::create_mirror_view(basis_diagnostics);
    auto h_write_marker = ::Kokkos::create_mirror_view(write_marker);
    {
      Ex9KokkosTimer timer("full condense block diagnostics copy");
      ::Kokkos::deep_copy(h_full_blocks, full_blocks);
      ::Kokkos::deep_copy(h_u_blocks, u_blocks);
      ::Kokkos::deep_copy(h_v_blocks, v_blocks);
      ::Kokkos::deep_copy(h_basis_diagnostics, basis_diagnostics);
      ::Kokkos::deep_copy(h_write_marker, write_marker);
    }

    for (std::size_t e = 0; e != full.elems.size(); ++e)
      if (h_condense_status(e))
      {
        const unsigned int row = static_cast<unsigned int>(h_failure_pivot(e) < 0 ?
                                                               0 :
                                                               h_failure_pivot(e));
        const unsigned int n_full_cols = std::min(full.layout.n_dofs, 12u);
        const unsigned int n_internal_cols = std::min(full.layout.internal_n_dofs, 12u);
        const unsigned int n_component_q_cols = std::min(component_layout.vector_n_dofs, 12u);
        const unsigned int component_q_row = std::min(row, component_layout.vector_n_dofs - 1);
        Real full_row_abs = Real(0);
        Real u_component_row_abs = Real(0);
        Real v_component_row_abs = Real(0);
        for (unsigned int c = 0; c != full.layout.internal_n_dofs; ++c)
        {
          const Real candidate =
              Kokkos::detail::abs_real(h_full_blocks(e, row * full.layout.n_dofs + c));
          if (candidate > full_row_abs)
            full_row_abs = candidate;
        }
        for (unsigned int c = 0; c != component_layout.vector_n_dofs; ++c)
        {
          const Real u_candidate = Kokkos::detail::abs_real(
              h_u_blocks(e,
                         component_layout.off_Jqq +
                             component_q_row * component_layout.vector_n_dofs + c));
          const Real v_candidate = Kokkos::detail::abs_real(
              h_v_blocks(e,
                         component_layout.off_Jqq +
                             component_q_row * component_layout.vector_n_dofs + c));
          if (u_candidate > u_component_row_abs)
            u_component_row_abs = u_candidate;
          if (v_candidate > v_component_row_abs)
            v_component_row_abs = v_candidate;
        }

        std::cout << "[ex9 kokkos diagnostic] condense failure elem=" << full.elems[e]->id()
                  << " local_elem=" << e << " pivot_row=" << row
                  << " full_row_abs=" << full_row_abs
                  << " u_component_jqq_row_abs=" << u_component_row_abs
                  << " v_component_jqq_row_abs=" << v_component_row_abs << '\n';
        std::cout << "[ex9 kokkos diagnostic] full layout n_dofs=" << full.layout.n_dofs
                  << " internal=" << full.layout.internal_n_dofs
                  << " external=" << full.layout.external_n_dofs
                  << " off_qu=" << full.layout.off_qu << " off_qv=" << full.layout.off_qv
                  << " off_u=" << full.layout.off_u << " off_v=" << full.layout.off_v
                  << " off_p=" << full.layout.off_p << " off_lm_u=" << full.layout.off_lm_u
                  << " off_lm_v=" << full.layout.off_lm_v << '\n';
        std::cout << "[ex9 kokkos diagnostic] component layout vector="
                  << component_layout.vector_n_dofs << " scalar=" << component_layout.scalar_n_dofs
                  << " lm=" << component_layout.lm_n_dofs
                  << " off_Jqq=" << component_layout.off_Jqq
                  << " off_Jqu=" << component_layout.off_Jqu
                  << " off_Juq=" << component_layout.off_Juq
                  << " off_Jup=" << component_layout.off_Jup << '\n';
        std::cout << "[ex9 kokkos diagnostic] independent Jqq00="
                  << h_basis_diagnostics(0) << " q0_phi0_norm2=" << h_basis_diagnostics(1)
                  << " q0_jxw=" << h_basis_diagnostics(2)
                  << " q0_phi0_component0=" << h_basis_diagnostics(3)
                  << " q0_phi0_component1=" << h_basis_diagnostics(4)
                  << " vector_components=" << h_basis_diagnostics(5)
                  << " supports_vector_shape=" << h_basis_diagnostics(6)
                  << " simple_marker=" << h_basis_diagnostics(7)
                  << " copied_marker=" << h_write_marker(0, 0) << '\n';

        std::cout << "[ex9 kokkos diagnostic] full pivot row first " << n_full_cols
                  << " cols:";
        for (unsigned int c = 0; c != n_full_cols; ++c)
          std::cout << ' ' << h_full_blocks(e, row * full.layout.n_dofs + c);
        std::cout << '\n';

        std::cout << "[ex9 kokkos diagnostic] full row 0 internal first " << n_internal_cols
                  << " cols:";
        for (unsigned int c = 0; c != n_internal_cols; ++c)
          std::cout << ' ' << h_full_blocks(e, c);
        std::cout << '\n';

        std::cout << "[ex9 kokkos diagnostic] u component Jqq row " << component_q_row
                  << " first " << n_component_q_cols << " cols:";
        for (unsigned int c = 0; c != n_component_q_cols; ++c)
          std::cout << ' '
                    << h_u_blocks(
                           e,
                           component_layout.off_Jqq +
                               component_q_row * component_layout.vector_n_dofs + c);
        std::cout << '\n';

        std::cout << "[ex9 kokkos diagnostic] v component Jqq row " << component_q_row
                  << " first " << n_component_q_cols << " cols:";
        for (unsigned int c = 0; c != n_component_q_cols; ++c)
          std::cout << ' '
                    << h_v_blocks(
                           e,
                           component_layout.off_Jqq +
                               component_q_row * component_layout.vector_n_dofs + c);
        std::cout << '\n';
        break;
      }
  }

  for (std::size_t e = 0; e != full.elems.size(); ++e)
    libmesh_error_msg_if(h_condense_status(e),
                         "vector_fe_ex9 Kokkos full HDG device condensation failed for elem "
                             << full.elems[e]->id() << " status=" << h_condense_status(e)
                             << " pivot=" << h_failure_pivot(e)
                             << " pivot_abs=" << h_failure_pivot_abs(e)
                             << " initial_diag=" << h_failure_initial_diag(e)
                             << " initial_row_abs=" << h_failure_initial_row_abs(e));
}

std::shared_ptr<DeviceBackSubstitutionPlan>
make_ex9_backsubstitution_plan(const Ex9DeviceAssemblyData & data)
{
  std::vector<std::vector<dof_id_type>> internal_indices(data.n_elem),
      external_indices(data.n_elem);
  for (unsigned int e = 0; e != data.n_elem; ++e)
  {
    internal_indices[e].reserve(data.layout.internal_n_dofs);
    internal_indices[e].insert(
        internal_indices[e].end(), data.qu_indices[e].begin(), data.qu_indices[e].end());
    internal_indices[e].insert(
        internal_indices[e].end(), data.qv_indices[e].begin(), data.qv_indices[e].end());
    internal_indices[e].insert(
        internal_indices[e].end(), data.u_indices[e].begin(), data.u_indices[e].end());
    internal_indices[e].insert(
        internal_indices[e].end(), data.v_indices[e].begin(), data.v_indices[e].end());
    libmesh_assert_equal_to(internal_indices[e].size(), data.layout.internal_n_dofs);

    external_indices[e].reserve(data.layout.external_n_dofs);
    external_indices[e].insert(
        external_indices[e].end(), data.p_indices[e].begin(), data.p_indices[e].end());
    external_indices[e].insert(
        external_indices[e].end(), data.lm_u_indices[e].begin(), data.lm_u_indices[e].end());
    external_indices[e].insert(
        external_indices[e].end(), data.lm_v_indices[e].begin(), data.lm_v_indices[e].end());
    libmesh_assert_equal_to(external_indices[e].size(), data.layout.external_n_dofs);
  }

  auto backsub_plan = std::make_shared<DeviceBackSubstitutionPlan>();
  backsub_plan->layout = data.layout;
  backsub_plan->n_elem = data.n_elem;
  backsub_plan->internal_dof_ids = DeviceDofId2D(
      "vector_fe_ex9_full_internal_dof_ids", data.n_elem, data.layout.internal_n_dofs);
  backsub_plan->external_dof_ids = DeviceDofId2D(
      "vector_fe_ex9_full_external_dof_ids", data.n_elem, data.layout.external_n_dofs);
  copy_dof_indices_to_view(internal_indices, backsub_plan->internal_dof_ids);
  copy_dof_indices_to_view(external_indices, backsub_plan->external_dof_ids);

  const NumericVector<Number> & solution = *data.system->solution;
  const dof_id_type first_local_dof = solution.first_local_index();
  const dof_id_type last_local_dof = solution.last_local_index();
  const dof_id_type n_local_dofs = last_local_dof - first_local_dof;
  backsub_plan->first_local_dof = first_local_dof;
  backsub_plan->last_local_dof = last_local_dof;
  backsub_plan->n_local_dofs = n_local_dofs;
  auto add_ghost_dof = [&](const dof_id_type dof)
  {
    if (dof < first_local_dof || dof >= last_local_dof)
      backsub_plan->ghost_dofs.push_back(cast_int<numeric_index_type>(dof));
  };
  for (unsigned int e = 0; e != data.n_elem; ++e)
  {
    for (const auto dof : internal_indices[e])
      add_ghost_dof(dof);
    for (const auto dof : external_indices[e])
      add_ghost_dof(dof);
  }
  std::sort(backsub_plan->ghost_dofs.begin(), backsub_plan->ghost_dofs.end());
  backsub_plan->ghost_dofs.erase(
      std::unique(backsub_plan->ghost_dofs.begin(), backsub_plan->ghost_dofs.end()),
      backsub_plan->ghost_dofs.end());

  std::vector<std::vector<dof_id_type>> internal_vector_indices(data.n_elem),
      external_vector_indices(data.n_elem);
  auto local_vector_index = [&](const dof_id_type dof)
  {
    if (dof >= first_local_dof && dof < last_local_dof)
      return dof - first_local_dof;

    const auto it =
        std::lower_bound(backsub_plan->ghost_dofs.begin(), backsub_plan->ghost_dofs.end(), dof);
    libmesh_assert(it != backsub_plan->ghost_dofs.end());
    libmesh_assert_equal_to(*it, dof);
    return n_local_dofs +
           cast_int<dof_id_type>(std::distance(backsub_plan->ghost_dofs.begin(), it));
  };
  for (unsigned int e = 0; e != data.n_elem; ++e)
  {
    internal_vector_indices[e].reserve(data.layout.internal_n_dofs);
    for (const auto dof : internal_indices[e])
      internal_vector_indices[e].push_back(local_vector_index(dof));

    external_vector_indices[e].reserve(data.layout.external_n_dofs);
    for (const auto dof : external_indices[e])
      external_vector_indices[e].push_back(local_vector_index(dof));
  }
  backsub_plan->internal_vector_ids = DeviceDofId2D(
      "vector_fe_ex9_full_internal_vector_ids", data.n_elem, data.layout.internal_n_dofs);
  backsub_plan->external_vector_ids = DeviceDofId2D(
      "vector_fe_ex9_full_external_vector_ids", data.n_elem, data.layout.external_n_dofs);
  copy_dof_indices_to_view(internal_vector_indices, backsub_plan->internal_vector_ids);
  copy_dof_indices_to_view(external_vector_indices, backsub_plan->external_vector_ids);

  backsub_plan->coo_rows.resize(data.n_elem * data.layout.internal_n_dofs);
  for (unsigned int e = 0; e != data.n_elem; ++e)
    for (unsigned int i = 0; i != data.layout.internal_n_dofs; ++i)
      backsub_plan->coo_rows[e * data.layout.internal_n_dofs + i] =
          cast_int<PetscInt>(internal_indices[e][i]);
  backsub_plan->coo_values =
      DeviceReal1D("vector_fe_ex9_backsub_coo_values", backsub_plan->coo_rows.size());

  return backsub_plan;
}

Ex9ResidualCooPlan
make_ex9_residual_coo_plan(const Ex9DeviceAssemblyData & data, const bool include_global_lm)
{
  const auto layout = data.layout;
  const std::size_t n_elem = data.elems.size();
  const std::size_t base_entries_per_elem = layout.n_dofs;
  const std::size_t cavity_entries_per_elem = include_global_lm ? layout.scalar_n_dofs + 1 : 0;
  const std::size_t base_entries = n_elem * base_entries_per_elem;
  const std::size_t n_entries = base_entries + n_elem * cavity_entries_per_elem;

  Ex9ResidualCooPlan plan;
  plan.rows.resize(n_entries);
  plan.values = DeviceReal1D("vector_fe_ex9_residual_coo_values", n_entries);

  for (std::size_t e = 0; e != n_elem; ++e)
  {
    std::vector<dof_id_type> indices;
    indices.reserve(layout.n_dofs);
    indices.insert(indices.end(), data.qu_indices[e].begin(), data.qu_indices[e].end());
    indices.insert(indices.end(), data.qv_indices[e].begin(), data.qv_indices[e].end());
    indices.insert(indices.end(), data.u_indices[e].begin(), data.u_indices[e].end());
    indices.insert(indices.end(), data.v_indices[e].begin(), data.v_indices[e].end());
    indices.insert(indices.end(), data.p_indices[e].begin(), data.p_indices[e].end());
    indices.insert(indices.end(), data.lm_u_indices[e].begin(), data.lm_u_indices[e].end());
    indices.insert(indices.end(), data.lm_v_indices[e].begin(), data.lm_v_indices[e].end());
    libmesh_assert_equal_to(indices.size(), layout.n_dofs);

    for (unsigned int i = 0; i != layout.n_dofs; ++i)
      plan.rows[e * base_entries_per_elem + i] = cast_int<PetscInt>(indices[i]);

    if (include_global_lm)
    {
      const std::size_t cavity_offset = base_entries + e * cavity_entries_per_elem;
      for (unsigned int i = 0; i != layout.scalar_n_dofs; ++i)
        plan.rows[cavity_offset + i] = cast_int<PetscInt>(data.p_indices[e][i]);
      plan.rows[cavity_offset + layout.scalar_n_dofs] =
          cast_int<PetscInt>(data.global_lm_indices[e][0]);
    }
  }

  return plan;
}

Ex9FullBatch
make_ex9_full_batch(const Ex9DeviceAssemblyData & data,
                    const DeviceAssemblyData & geom,
                    const Ex9DeviceWork & work,
                    const DeviceAssemblyData & u_data,
                    const DeviceAssemblyData & v_data,
                    const AssemblyPurpose purpose,
                    const std::shared_ptr<DeviceBackSubstitutionPlan> & backsub_plan,
                    Ex9FullBatchWork & full_work,
                    ComponentBatchWork & u_work,
                    ComponentBatchWork & v_work)
{
  Ex9FullBatch full;
  full.layout = data.layout;
  full.elems = data.elems;
  full.qu_indices = data.qu_indices;
  full.qv_indices = data.qv_indices;
  full.u_indices = data.u_indices;
  full.v_indices = data.v_indices;
  full.p_indices = data.p_indices;
  full.lm_u_indices = data.lm_u_indices;
  full.lm_v_indices = data.lm_v_indices;
  full.global_lm_indices = data.global_lm_indices;
  full.scalar_key = geom.scalar_key;
  full.n_qp = geom.n_qp;
  full.xi = geom.xi;
  full.eta = geom.eta;
  full.zeta = geom.zeta;
  full.jxw = geom.jxw;
  full.device_p_dofs = work.p_dofs;
  full.device_global_lm_dofs = work.global_lm_dofs;
  full.backsub_plan = backsub_plan;
  const bool assemble_residual = purpose == AssemblyPurpose::residual;

  DeviceReal2D device_blocks = full_work.device_blocks;
  DeviceReal2D device_residual = full_work.device_residual;
  {
    Ex9KokkosTimer timer("full zero blocks/residual");
    ::Kokkos::deep_copy(device_blocks, Real(0));
    if (assemble_residual)
      ::Kokkos::deep_copy(device_residual, Real(0));
  }

  {
    Ex9KokkosTimer timer("direct full component assembly");
    const ComponentDofViews u_dofs{work.qu_dofs,
                                   work.u_dofs,
                                   work.p_dofs,
                                   work.lm_u_dofs,
                                   work.v_dofs,
                                   work.lm_v_dofs};
    const ComponentDofViews v_dofs{work.qv_dofs,
                                   work.v_dofs,
                                   work.p_dofs,
                                   work.lm_v_dofs,
                                   work.u_dofs,
                                   work.lm_u_dofs};
    libmesh_error_msg_if(!assemble_component_pair_into_full_dispatch(u_data,
                                                                     v_data,
                                                                     u_dofs,
                                                                     v_dofs,
                                                                     full.layout,
                                                                     device_blocks,
                                                                     device_residual,
                                                                     u_work,
                                                                     v_work,
                                                                     assemble_residual),
                         "vector_fe_ex9 Kokkos direct full HDG assembly failed for the velocity "
                         "components.");
  }

  DeviceReal2D condensed_blocks = full_work.condensed_blocks;
  DeviceReal2D internal_inverse = full_work.internal_inverse;
  DeviceReal2D internal_external_solve = full_work.internal_external_solve;
  DeviceInt1D condense_status = full_work.condense_status;
  DeviceInt1D failure_pivot = full_work.failure_pivot;
  DeviceReal1D failure_pivot_abs = full_work.failure_pivot_abs;
  DeviceReal1D failure_initial_diag = full_work.failure_initial_diag;
  DeviceReal1D failure_initial_row_abs = full_work.failure_initial_row_abs;
  {
    Ex9KokkosTimer timer("condense zero work");
    ::Kokkos::deep_copy(condensed_blocks, Real(0));
    ::Kokkos::deep_copy(internal_inverse, Real(0));
    ::Kokkos::deep_copy(internal_external_solve, Real(0));
    ::Kokkos::deep_copy(condense_status, 0);
    ::Kokkos::deep_copy(failure_pivot, -1);
    ::Kokkos::deep_copy(failure_pivot_abs, Real(0));
    ::Kokkos::deep_copy(failure_initial_diag, Real(0));
    ::Kokkos::deep_copy(failure_initial_row_abs, Real(0));
  }
  {
    Ex9KokkosTimer timer("full static condensation kernel");
    condense_ex9_full_on_device(data.layout,
                                data.n_elem,
                                device_blocks,
                                condensed_blocks,
                                internal_inverse,
                                internal_external_solve,
                                condense_status,
                                failure_pivot,
                                failure_pivot_abs,
                                failure_initial_diag,
                                failure_initial_row_abs);
  }

  full.device_residual = device_residual;
  full.device_condensed_blocks = condensed_blocks;
  full.backsub_plan->full_blocks = device_blocks;
  full.backsub_plan->internal_inverse = internal_inverse;
  full.backsub_plan->internal_external_solve = internal_external_solve;
  check_ex9_full_condense_status(full,
                                 u_data,
                                 device_blocks,
                                 u_work.blocks,
                                 v_work.blocks,
                                 condense_status,
                                 failure_pivot,
                                 failure_pivot_abs,
                                 failure_initial_diag,
                                 failure_initial_row_abs);

  return full;
}

bool
is_petsc_kokkos_vec(const Parallel::Communicator & comm, Vec vec);

class Ex9DeviceOperator
{
public:
  Ex9DeviceOperator(NonlinearImplicitSystem & system, HDGProblem & problem)
    : _system(system),
      _problem(problem),
      _qu_num(system.variable_number("qu")),
      _qv_num(system.variable_number("qv")),
      _u_num(system.variable_number("vel_x")),
      _v_num(system.variable_number("vel_y")),
      _lm_u_num(system.variable_number("lm_u")),
      _lm_v_num(system.variable_number("lm_v")),
      _p_num(system.variable_number("pressure")),
      _global_lm_num(problem.cavity ? system.variable_number("global_lm") : invalid_uint)
  {
    libmesh_error_msg_if(!_problem.mesh || !_problem.system || !_problem.dof_map ||
                             !_problem.vector_fe || !_problem.vector_fe_face || !_problem.qrule ||
                             !_problem.qface || _problem.system != &_system ||
                             _problem.mesh->mesh_dimension() != 2,
                         "vector_fe_ex9 Kokkos HDG assembly received an unsupported problem "
                         "setup.");

    libmesh_error_msg_if(!collect_supported_elements(_problem, _elems, _elem_type),
                         "vector_fe_ex9 Kokkos HDG assembly supports only nonempty homogeneous "
                         "TRI6/TRI7 element batches.");

    _full_data = make_ex9_device_assembly_data(_system,
                                               _problem,
                                               _elems,
                                               _elem_type,
                                               _qu_num,
                                               _qv_num,
                                               _u_num,
                                               _v_num,
                                               _p_num,
                                               _lm_u_num,
                                               _lm_v_num,
                                               _global_lm_num);
    _u_data = make_device_assembly_data(_system,
                                        _problem,
                                        AssemblyPurpose::residual,
                                        _elems,
                                        _elem_type,
                                        _qu_num,
                                        _u_num,
                                        _lm_u_num,
                                        _p_num,
                                        _v_num,
                                        _lm_v_num,
                                        0);
    _v_data = make_device_assembly_data(_system,
                                        _problem,
                                        AssemblyPurpose::residual,
                                        _elems,
                                        _elem_type,
                                        _qv_num,
                                        _v_num,
                                        _lm_v_num,
                                        _p_num,
                                        _u_num,
                                        _lm_u_num,
                                        1);
    _work = std::make_unique<Ex9DeviceWork>(*_full_data);
    _u_work = std::make_unique<ComponentBatchWork>(*_u_data);
    _v_work = std::make_unique<ComponentBatchWork>(*_v_data);
    _full_work = std::make_unique<Ex9FullBatchWork>(*_full_data);
    _backsub_plan = make_ex9_backsubstitution_plan(*_full_data);
    _residual_coo_plan = make_ex9_residual_coo_plan(*_full_data, _problem.cavity);
  }

  void
  assemble(const NumericVector<Number> & X, const AssemblyPurpose purpose, Ex9FullBatch & batch)
  {
    Ex9KokkosTimer assemble_timer(purpose == AssemblyPurpose::jacobian ?
                                      "assemble jacobian total" :
                                      "assemble residual total");
    Ex9DeviceWork & work = *_work;
    {
      Ex9KokkosTimer timer("solution dof gather");
      gather_solution_dofs(X, work);
    }

    {
      Ex9KokkosTimer timer("make full batch total");
      batch = make_ex9_full_batch(*_full_data,
                                  *_u_data,
                                  work,
                                  *_u_data,
                                  *_v_data,
                                  purpose,
                                  _backsub_plan,
                                  *_full_work,
                                  *_u_work,
                                  *_v_work);
    }
  }

  Ex9ResidualCooPlan &
  residual_coo_plan()
  {
    return _residual_coo_plan;
  }

  Ex9CondensedJacobianCooPlan &
  condensed_jacobian_coo_plan()
  {
    return _condensed_jacobian_coo_plan;
  }

private:
  void
  gather_solution_dofs(const NumericVector<Number> & X, Ex9DeviceWork & work)
  {
    auto * petsc_x = dynamic_cast<const PetscVector<Number> *>(&X);
    libmesh_error_msg_if(!petsc_x || !is_petsc_kokkos_vec(X.comm(), petsc_x->vec()),
                         "vector_fe_ex9 Kokkos assembly requires PETSc Kokkos solution vectors. "
                         "Do not override -vec_type away from kokkos on this path.");

    if (X.comm().size() == 1)
    {
      using PetscKokkosReadView =
          ::Kokkos::View<const PetscScalar *, ::Kokkos::DefaultExecutionSpace::memory_space>;
      PetscKokkosReadView x_view;
      {
        Ex9KokkosTimer timer("VecGetKokkosView solution", false);
        LibmeshPetscCall2(X.comm(), VecGetKokkosView(petsc_x->vec(), &x_view));
      }
      {
        Ex9KokkosTimer timer("device gather kernel");
        gather_ex9_dofs_on_device(*_full_data, x_view, work);
      }
      {
        Ex9KokkosTimer timer("VecRestoreKokkosView solution", false);
        LibmeshPetscCall2(X.comm(), VecRestoreKokkosView(petsc_x->vec(), &x_view));
      }
      return;
    }

    DeviceReal1D global_x;
    {
      Ex9KokkosTimer timer("localize solution to device");
      global_x = copy_global_solution_to_device(X);
    }
    {
      Ex9KokkosTimer timer("device gather kernel");
      gather_ex9_dofs_on_device(*_full_data, global_x, work);
    }
  }

  NonlinearImplicitSystem & _system;
  HDGProblem & _problem;
  std::vector<const Elem *> _elems;
  ElemType _elem_type = INVALID_ELEM;
  const unsigned int _qu_num;
  const unsigned int _qv_num;
  const unsigned int _u_num;
  const unsigned int _v_num;
  const unsigned int _lm_u_num;
  const unsigned int _lm_v_num;
  const unsigned int _p_num;
  const unsigned int _global_lm_num;
  std::unique_ptr<Ex9DeviceAssemblyData> _full_data;
  std::unique_ptr<DeviceAssemblyData> _u_data;
  std::unique_ptr<DeviceAssemblyData> _v_data;
  std::unique_ptr<Ex9DeviceWork> _work;
  std::unique_ptr<ComponentBatchWork> _u_work;
  std::unique_ptr<ComponentBatchWork> _v_work;
  std::unique_ptr<Ex9FullBatchWork> _full_work;
  std::shared_ptr<DeviceBackSubstitutionPlan> _backsub_plan;
  Ex9ResidualCooPlan _residual_coo_plan;
  Ex9CondensedJacobianCooPlan _condensed_jacobian_coo_plan;
};

std::unordered_map<const HDGProblem *, std::unique_ptr<Ex9DeviceOperator>> &
ex9_device_operator_cache()
{
  static std::unordered_map<const HDGProblem *, std::unique_ptr<Ex9DeviceOperator>> cache;
  return cache;
}

Ex9DeviceOperator &
get_ex9_device_operator(NonlinearImplicitSystem & system, HDGProblem & problem)
{
  auto & cache = ex9_device_operator_cache();
  auto & operator_ptr = cache[&problem];
  if (!operator_ptr)
    operator_ptr = std::make_unique<Ex9DeviceOperator>(system, problem);
  return *operator_ptr;
}

void
clear_ex9_device_operator_cache(const HDGProblem & problem)
{
  ex9_device_operator_cache().erase(&problem);
}

std::vector<dof_id_type>
ex9_internal_indices(const Ex9FullBatch & batch, const std::size_t e)
{
  const auto layout = batch.layout;
  std::vector<dof_id_type> internal_indices;
  internal_indices.reserve(layout.internal_n_dofs);
  internal_indices.insert(
      internal_indices.end(), batch.qu_indices[e].begin(), batch.qu_indices[e].end());
  internal_indices.insert(
      internal_indices.end(), batch.qv_indices[e].begin(), batch.qv_indices[e].end());
  internal_indices.insert(
      internal_indices.end(), batch.u_indices[e].begin(), batch.u_indices[e].end());
  internal_indices.insert(
      internal_indices.end(), batch.v_indices[e].begin(), batch.v_indices[e].end());
  libmesh_assert_equal_to(internal_indices.size(), layout.internal_n_dofs);
  return internal_indices;
}

std::vector<dof_id_type>
ex9_external_indices(const Ex9FullBatch & batch, const std::size_t e)
{
  const auto layout = batch.layout;
  std::vector<dof_id_type> external_indices;
  external_indices.reserve(layout.external_n_dofs);
  external_indices.insert(
      external_indices.end(), batch.p_indices[e].begin(), batch.p_indices[e].end());
  external_indices.insert(
      external_indices.end(), batch.lm_u_indices[e].begin(), batch.lm_u_indices[e].end());
  external_indices.insert(
      external_indices.end(), batch.lm_v_indices[e].begin(), batch.lm_v_indices[e].end());
  libmesh_assert_equal_to(external_indices.size(), layout.external_n_dofs);
  return external_indices;
}

std::vector<dof_id_type>
ex9_full_indices(const Ex9FullBatch & batch, const std::size_t e)
{
  const auto layout = batch.layout;
  std::vector<dof_id_type> indices;
  indices.reserve(layout.n_dofs);
  indices.insert(indices.end(), batch.qu_indices[e].begin(), batch.qu_indices[e].end());
  indices.insert(indices.end(), batch.qv_indices[e].begin(), batch.qv_indices[e].end());
  indices.insert(indices.end(), batch.u_indices[e].begin(), batch.u_indices[e].end());
  indices.insert(indices.end(), batch.v_indices[e].begin(), batch.v_indices[e].end());
  indices.insert(indices.end(), batch.p_indices[e].begin(), batch.p_indices[e].end());
  indices.insert(indices.end(), batch.lm_u_indices[e].begin(), batch.lm_u_indices[e].end());
  indices.insert(indices.end(), batch.lm_v_indices[e].begin(), batch.lm_v_indices[e].end());
  libmesh_assert_equal_to(indices.size(), layout.n_dofs);
  return indices;
}

bool
is_petsc_kokkos_vec(const Parallel::Communicator & comm, Vec vec)
{
  const char * vec_type = nullptr;
  LibmeshPetscCall2(comm, VecGetType(vec, &vec_type));

  PetscBool is_seq_kokkos = PETSC_FALSE;
  PetscBool is_mpi_kokkos = PETSC_FALSE;
  LibmeshPetscCall2(comm, PetscStrcmp(vec_type, "seqkokkos", &is_seq_kokkos));
  LibmeshPetscCall2(comm, PetscStrcmp(vec_type, "mpikokkos", &is_mpi_kokkos));
  return is_seq_kokkos || is_mpi_kokkos;
}

bool
is_petsc_kokkos_mat(const Parallel::Communicator & comm, Mat mat)
{
  const char * mat_type = nullptr;
  LibmeshPetscCall2(comm, MatGetType(mat, &mat_type));

  PetscBool is_seq_aijkokkos = PETSC_FALSE;
  PetscBool is_mpi_aijkokkos = PETSC_FALSE;
  LibmeshPetscCall2(comm, PetscStrcmp(mat_type, MATSEQAIJKOKKOS, &is_seq_aijkokkos));
  LibmeshPetscCall2(comm, PetscStrcmp(mat_type, MATMPIAIJKOKKOS, &is_mpi_aijkokkos));
  return is_seq_aijkokkos || is_mpi_aijkokkos;
}

bool
run_device_backwards_substitution(DeviceBackSubstitutionPlan & plan,
                                  const NumericVector<Number> & full_rhs,
                                  NumericVector<Number> & full_sol)
{
  auto * petsc_rhs = dynamic_cast<const PetscVector<Number> *>(&full_rhs);
  auto * petsc_sol = dynamic_cast<PetscVector<Number> *>(&full_sol);
  libmesh_error_msg_if(
      !petsc_rhs || !petsc_sol || !is_petsc_kokkos_vec(full_rhs.comm(), petsc_rhs->vec()) ||
          !is_petsc_kokkos_vec(full_sol.comm(), petsc_sol->vec()),
      "vector_fe_ex9 Kokkos back-substitution requires PETSc Kokkos vectors. Do not override "
      "-vec_type away from kokkos on this path.");

  const auto layout = plan.layout;
  const std::size_t n_elem = plan.n_elem;
  const std::size_t n_entries = n_elem * layout.internal_n_dofs;
  libmesh_assert_equal_to(plan.coo_rows.size(), n_entries);

  DeviceReal1D coo_values = plan.coo_values;

  if (full_sol.comm().size() == 1)
  {
    using PetscKokkosReadView =
        ::Kokkos::View<const PetscScalar *, ::Kokkos::DefaultExecutionSpace::memory_space>;
    PetscKokkosReadView rhs_view;
    PetscKokkosReadView sol_view;
    LibmeshPetscCall2(full_rhs.comm(), VecGetKokkosView(petsc_rhs->vec(), &rhs_view));
    LibmeshPetscCall2(full_sol.comm(), VecGetKokkosView(petsc_sol->vec(), &sol_view));
    fill_back_substitution_coo_values(plan, rhs_view, sol_view);
    LibmeshPetscCall2(full_rhs.comm(), VecRestoreKokkosView(petsc_rhs->vec(), &rhs_view));
    LibmeshPetscCall2(full_sol.comm(), VecRestoreKokkosView(petsc_sol->vec(), &sol_view));
  }
  else
  {
    if (!plan.ghosted_rhs)
    {
      plan.ghosted_rhs = std::make_unique<PetscVector<Number>>(full_rhs.comm());
      plan.ghosted_rhs->init(
          full_rhs.size(), full_rhs.local_size(), plan.ghost_dofs, true, GHOSTED);
    }
    if (!plan.ghosted_sol)
    {
      plan.ghosted_sol = std::make_unique<PetscVector<Number>>(full_sol.comm());
      plan.ghosted_sol->init(
          full_sol.size(), full_sol.local_size(), plan.ghost_dofs, true, GHOSTED);
    }

    libmesh_error_msg_if(
        !is_petsc_kokkos_vec(full_rhs.comm(), plan.ghosted_rhs->vec()) ||
            !is_petsc_kokkos_vec(full_sol.comm(), plan.ghosted_sol->vec()),
        "vector_fe_ex9 Kokkos MPI back-substitution requires PETSc Kokkos ghost vectors. "
        "Do not override -ghost_vec_type away from kokkos on this path.");

    *plan.ghosted_rhs = *petsc_rhs;
    *plan.ghosted_sol = *petsc_sol;

    auto rhs_guard = plan.ghosted_rhs->make_kokkos_read_view_guard();
    auto sol_guard = plan.ghosted_sol->make_kokkos_read_view_guard();
    fill_back_substitution_coo_values(plan, rhs_guard.view(), sol_guard.view());
  }

  if (plan.coo_target != petsc_sol->vec())
  {
    Ex9KokkosTimer timer("VecSetPreallocationCOO back-substitution", false);
    LibmeshPetscCall2(
        full_sol.comm(),
        VecSetPreallocationCOO(
            petsc_sol->vec(), cast_int<PetscCount>(n_entries), plan.coo_rows.data()));
    plan.coo_target = petsc_sol->vec();
  }
  {
    Ex9KokkosTimer timer("VecSetValuesCOO back-substitution", false);
    LibmeshPetscCall2(full_sol.comm(),
                      VecSetValuesCOO(petsc_sol->vec(), coo_values.data(), ADD_VALUES));
  }
  {
    Ex9KokkosTimer timer("full solution close after back-substitution", false);
    full_sol.close();
  }
  return true;
}

bool
run_device_forward_elimination(DeviceBackSubstitutionPlan & plan,
                               const NumericVector<Number> & full_rhs,
                               NumericVector<Number> & reduced_rhs)
{
  auto * petsc_full_rhs = dynamic_cast<const PetscVector<Number> *>(&full_rhs);
  auto * petsc_reduced_rhs = dynamic_cast<PetscVector<Number> *>(&reduced_rhs);
  libmesh_error_msg_if(
      !petsc_full_rhs || !petsc_reduced_rhs ||
          !is_petsc_kokkos_vec(full_rhs.comm(), petsc_full_rhs->vec()) ||
          !is_petsc_kokkos_vec(reduced_rhs.comm(), petsc_reduced_rhs->vec()),
      "vector_fe_ex9 Kokkos forward elimination requires PETSc Kokkos vectors. Do not override "
      "-vec_type away from kokkos on this path.");

  const auto layout = plan.layout;
  const std::size_t n_base_entries = plan.forward_base_n_entries;
  const std::size_t n_correction_entries = plan.n_elem * layout.external_n_dofs;
  const std::size_t n_entries = n_base_entries + n_correction_entries;
  libmesh_assert_equal_to(plan.forward_coo_rows.size(), n_entries);
  libmesh_assert_equal_to(plan.forward_base_vector_ids.extent(0), n_base_entries);

  DeviceReal1D coo_values = plan.forward_coo_values;

  if (full_rhs.comm().size() == 1)
  {
    using PetscKokkosReadView =
        ::Kokkos::View<const PetscScalar *, ::Kokkos::DefaultExecutionSpace::memory_space>;
    PetscKokkosReadView rhs_view;
    LibmeshPetscCall2(full_rhs.comm(), VecGetKokkosView(petsc_full_rhs->vec(), &rhs_view));
    fill_forward_elimination_coo_values(plan, rhs_view);
    LibmeshPetscCall2(full_rhs.comm(), VecRestoreKokkosView(petsc_full_rhs->vec(), &rhs_view));
  }
  else
  {
    if (!plan.ghosted_rhs)
    {
      plan.ghosted_rhs = std::make_unique<PetscVector<Number>>(full_rhs.comm());
      plan.ghosted_rhs->init(
          full_rhs.size(), full_rhs.local_size(), plan.ghost_dofs, true, GHOSTED);
    }

    libmesh_error_msg_if(!is_petsc_kokkos_vec(full_rhs.comm(), plan.ghosted_rhs->vec()),
                         "vector_fe_ex9 Kokkos MPI forward elimination requires PETSc Kokkos "
                         "ghost vectors. Do not override -ghost_vec_type away from kokkos on this "
                         "path.");

    *plan.ghosted_rhs = *petsc_full_rhs;
    auto rhs_guard = plan.ghosted_rhs->make_kokkos_read_view_guard();
    fill_forward_elimination_coo_values(plan, rhs_guard.view());
  }

  {
    Ex9KokkosTimer timer("reduced RHS zero before forward elimination", false);
    reduced_rhs.zero();
  }
  if (plan.forward_coo_target != petsc_reduced_rhs->vec())
  {
    Ex9KokkosTimer timer("VecSetPreallocationCOO forward elimination", false);
    LibmeshPetscCall2(
        reduced_rhs.comm(),
        VecSetPreallocationCOO(petsc_reduced_rhs->vec(),
                               cast_int<PetscCount>(n_entries),
                               plan.forward_coo_rows.data()));
    plan.forward_coo_target = petsc_reduced_rhs->vec();
  }
  {
    Ex9KokkosTimer timer("VecSetValuesCOO forward elimination", false);
    LibmeshPetscCall2(reduced_rhs.comm(),
                      VecSetValuesCOO(petsc_reduced_rhs->vec(), coo_values.data(), ADD_VALUES));
  }

  return true;
}

void
add_device_full_residual_to_petsc_coo(const Ex9FullBatch & batch,
                                      NumericVector<Number> & vector,
                                      const bool include_global_lm,
                                      Ex9ResidualCooPlan & coo_plan)
{
  auto * petsc_vector = dynamic_cast<PetscVector<Number> *>(&vector);
  libmesh_error_msg_if(!petsc_vector || !is_petsc_kokkos_vec(vector.comm(), petsc_vector->vec()),
                       "vector_fe_ex9 Kokkos residual requires PETSc Kokkos vectors. Do not "
                       "override -vec_type away from kokkos on this path.");

  const auto layout = batch.layout;
  const std::size_t n_elem = batch.elems.size();
  const std::size_t base_entries_per_elem = layout.n_dofs;
  const std::size_t cavity_entries_per_elem = include_global_lm ? layout.scalar_n_dofs + 1 : 0;
  const std::size_t base_entries = n_elem * base_entries_per_elem;
  const std::size_t n_entries = base_entries + n_elem * cavity_entries_per_elem;
  libmesh_assert_equal_to(coo_plan.rows.size(), n_entries);

  DeviceReal1D coo_values = coo_plan.values;
  const auto residual = batch.device_residual;
  const auto p_dofs = batch.device_p_dofs;
  const auto global_lm_dofs = batch.device_global_lm_dofs;
  const auto scalar_key = batch.scalar_key;
  const auto xi = batch.xi;
  const auto eta = batch.eta;
  const auto zeta = batch.zeta;
  const auto jxw = batch.jxw;
  const unsigned int n_dofs = layout.n_dofs;
  const unsigned int scalar_n_dofs = layout.scalar_n_dofs;
  const unsigned int n_qp = batch.n_qp;
  const std::size_t base_n_entries = base_entries;
  const std::size_t cavity_n_entries_per_elem = cavity_entries_per_elem;
  {
    Ex9KokkosTimer timer("residual COO values kernel");
    ::Kokkos::parallel_for(
        "vector_fe_ex9_residual_coo_values",
        ::Kokkos::RangePolicy<>(0, cast_int<int>(n_entries)),
        KOKKOS_LAMBDA(const int entry) {
          if (static_cast<std::size_t>(entry) < base_n_entries)
          {
            const int elem = entry / n_dofs;
            const int local = entry - elem * n_dofs;
            coo_values(entry) = residual(elem, local);
            return;
          }

          const int cavity_entry = entry - static_cast<int>(base_n_entries);
          const int elem = cavity_entry / static_cast<int>(cavity_n_entries_per_elem);
          const int local = cavity_entry - elem * static_cast<int>(cavity_n_entries_per_elem);
          if (local < static_cast<int>(scalar_n_dofs))
          {
            Real value = Real(0);
            for (unsigned int q = 0; q != n_qp; ++q)
              value -= jxw(q) * Kokkos::shape(scalar_key, local, xi(q), eta(q), zeta(q)) *
                       global_lm_dofs(elem, 0);
            coo_values(entry) = value;
          }
          else
          {
            Real value = Real(0);
            for (unsigned int q = 0; q != n_qp; ++q)
            {
              Real p_value = Real(0);
              for (unsigned int j = 0; j != scalar_n_dofs; ++j)
                p_value += Kokkos::shape(scalar_key, j, xi(q), eta(q), zeta(q)) * p_dofs(elem, j);
              value -= jxw(q) * p_value;
            }
            coo_values(entry) = value;
          }
        });
  }

  if (coo_plan.target != petsc_vector->vec())
  {
    Ex9KokkosTimer timer("VecSetPreallocationCOO residual", false);
    LibmeshPetscCall2(vector.comm(),
                      VecSetPreallocationCOO(petsc_vector->vec(),
                                             cast_int<PetscCount>(n_entries),
                                             coo_plan.rows.data()));
    coo_plan.target = petsc_vector->vec();
  }
  {
    Ex9KokkosTimer timer("VecSetValuesCOO residual", false);
    LibmeshPetscCall2(vector.comm(),
                      VecSetValuesCOO(petsc_vector->vec(), coo_values.data(), ADD_VALUES));
  }
}

void
add_device_condensed_jacobian_to_petsc_coo(const Ex9FullBatch & batch,
                                           StaticCondensation & sc,
                                           const bool include_global_lm,
                                           Ex9CondensedJacobianCooPlan & coo_plan)
{
  auto * petsc_matrix = dynamic_cast<PetscMatrixBase<Number> *>(&sc.get_condensed_mat());
  libmesh_error_msg_if(!petsc_matrix || !is_petsc_kokkos_mat(sc.comm(), petsc_matrix->mat()),
                       "vector_fe_ex9 Kokkos static-condensation Jacobian requires PETSc "
                       "AIJKokkos reduced matrices. Run with -mat_type aijkokkos.");

  const auto layout = batch.layout;
  const std::size_t n_elem = batch.elems.size();
  const std::size_t entries_per_elem = layout.condensed_n_entries;
  const std::size_t base_entries = n_elem * entries_per_elem;
  const std::size_t cavity_entries_per_elem = include_global_lm ? 2 * layout.scalar_n_dofs + 1 : 0;
  const std::size_t n_entries = base_entries + n_elem * cavity_entries_per_elem;

  if (!coo_plan.initialized)
  {
    Ex9KokkosTimer timer("condensed Jacobian COO symbolic setup", false);
    coo_plan.rows.resize(n_entries);
    coo_plan.cols.resize(n_entries);
    coo_plan.values = DeviceReal1D("vector_fe_ex9_condensed_coo_values", n_entries);

    for (std::size_t e = 0; e != n_elem; ++e)
    {
      const auto external_indices = ex9_external_indices(batch, e);
      std::vector<dof_id_type> reduced_indices;
      reduced_indices.reserve(external_indices.size());
      for (const auto full_dof : external_indices)
        reduced_indices.push_back(sc.reduced_dof_for_uncondensed_dof(*batch.elems[e], full_dof));

      for (unsigned int i = 0; i != layout.external_n_dofs; ++i)
        for (unsigned int j = 0; j != layout.external_n_dofs; ++j)
        {
          const std::size_t entry = e * entries_per_elem + i * layout.external_n_dofs + j;
          coo_plan.rows[entry] = cast_int<PetscInt>(reduced_indices[i]);
          coo_plan.cols[entry] = cast_int<PetscInt>(reduced_indices[j]);
        }

      if (include_global_lm)
      {
        const dof_id_type reduced_global_lm =
            sc.reduced_dof_for_uncondensed_dof(*batch.elems[e], batch.global_lm_indices[e][0]);
        const std::size_t cavity_offset = base_entries + e * cavity_entries_per_elem;
        for (unsigned int i = 0; i != layout.scalar_n_dofs; ++i)
        {
          const dof_id_type reduced_p =
              sc.reduced_dof_for_uncondensed_dof(*batch.elems[e], batch.p_indices[e][i]);
          coo_plan.rows[cavity_offset + i] = cast_int<PetscInt>(reduced_p);
          coo_plan.cols[cavity_offset + i] = cast_int<PetscInt>(reduced_global_lm);
          coo_plan.rows[cavity_offset + layout.scalar_n_dofs + i] =
              cast_int<PetscInt>(reduced_global_lm);
          coo_plan.cols[cavity_offset + layout.scalar_n_dofs + i] =
              cast_int<PetscInt>(reduced_p);
        }
        coo_plan.rows[cavity_offset + 2 * layout.scalar_n_dofs] =
            cast_int<PetscInt>(reduced_global_lm);
        coo_plan.cols[cavity_offset + 2 * layout.scalar_n_dofs] =
            cast_int<PetscInt>(reduced_global_lm);
      }
    }

    coo_plan.initialized = true;
  }

  libmesh_assert_equal_to(coo_plan.rows.size(), n_entries);
  libmesh_assert_equal_to(coo_plan.cols.size(), n_entries);

  DeviceReal1D coo_values = coo_plan.values;
  const auto condensed_blocks = batch.device_condensed_blocks;
  const auto scalar_key = batch.scalar_key;
  const auto xi = batch.xi;
  const auto eta = batch.eta;
  const auto zeta = batch.zeta;
  const auto jxw = batch.jxw;
  const unsigned int condensed_n_entries = layout.condensed_n_entries;
  const unsigned int scalar_n_dofs = layout.scalar_n_dofs;
  const unsigned int n_qp = batch.n_qp;
  const std::size_t base_n_entries = base_entries;
  const std::size_t cavity_n_entries_per_elem = cavity_entries_per_elem;
  {
    Ex9KokkosTimer timer("condensed Jacobian COO values kernel");
    ::Kokkos::parallel_for(
        "vector_fe_ex9_condensed_coo_values",
        ::Kokkos::RangePolicy<>(0, cast_int<int>(n_entries)),
        KOKKOS_LAMBDA(const int entry) {
          if (static_cast<std::size_t>(entry) < base_n_entries)
          {
            const int elem = entry / condensed_n_entries;
            const int local = entry - elem * condensed_n_entries;
            coo_values(entry) = condensed_blocks(elem, local);
            return;
          }

          const int cavity_entry = entry - static_cast<int>(base_n_entries);
          const int elem = cavity_entry / static_cast<int>(cavity_n_entries_per_elem);
          const int local = (cavity_entry - elem * static_cast<int>(cavity_n_entries_per_elem)) %
                            static_cast<int>(scalar_n_dofs);
          if ((cavity_entry - elem * static_cast<int>(cavity_n_entries_per_elem)) >=
              static_cast<int>(2 * scalar_n_dofs))
          {
            coo_values(entry) = Real(0);
            return;
          }
          Real value = Real(0);
          for (unsigned int q = 0; q != n_qp; ++q)
            value -= jxw(q) * Kokkos::shape(scalar_key, local, xi(q), eta(q), zeta(q));
          coo_values(entry) = value;
        });
  }

  if (coo_plan.target != petsc_matrix->mat())
  {
    Ex9KokkosTimer timer("MatSetPreallocationCOO condensed Jacobian", false);
    LibmeshPetscCall2(sc.comm(),
                      MatSetPreallocationCOO(petsc_matrix->mat(),
                                             cast_int<PetscCount>(n_entries),
                                             coo_plan.rows.data(),
                                             coo_plan.cols.data()));
    coo_plan.target = petsc_matrix->mat();
  }
  {
    Ex9KokkosTimer timer("MatSetValuesCOO condensed Jacobian", false);
    LibmeshPetscCall2(sc.comm(),
                      MatSetValuesCOO(petsc_matrix->mat(), coo_values.data(), ADD_VALUES));
  }
}

void
prepare_device_forward_elimination_plan(const Ex9FullBatch & batch, StaticCondensation & sc)
{
  auto plan = batch.backsub_plan;
  libmesh_assert(plan);
  const auto layout = batch.layout;
  std::vector<dof_id_type> base_full_dofs;
  std::vector<dof_id_type> base_reduced_dofs;
  sc.local_uncondensed_dof_map(base_full_dofs, base_reduced_dofs);
  libmesh_assert_equal_to(base_full_dofs.size(), base_reduced_dofs.size());

  const std::size_t n_base_entries = base_full_dofs.size();
  const std::size_t n_correction_entries = batch.elems.size() * layout.external_n_dofs;
  const std::size_t n_entries = n_base_entries + n_correction_entries;
  if (plan->forward_base_n_entries == n_base_entries &&
      plan->forward_coo_rows.size() == n_entries &&
      plan->forward_base_vector_ids.extent(0) == n_base_entries &&
      plan->forward_coo_values.extent(0) == n_entries)
    return;

  auto local_vector_index = [plan](const dof_id_type dof)
  {
    if (dof >= plan->first_local_dof && dof < plan->last_local_dof)
      return dof - plan->first_local_dof;

    const auto it = std::lower_bound(plan->ghost_dofs.begin(), plan->ghost_dofs.end(), dof);
    libmesh_assert(it != plan->ghost_dofs.end());
    libmesh_assert_equal_to(*it, dof);
    return plan->n_local_dofs +
           cast_int<dof_id_type>(std::distance(plan->ghost_dofs.begin(), it));
  };

  plan->forward_base_n_entries = n_base_entries;
  plan->forward_coo_rows.resize(n_entries);
  std::vector<dof_id_type> base_vector_ids(n_base_entries);
  for (std::size_t i = 0; i != n_base_entries; ++i)
  {
    plan->forward_coo_rows[i] = cast_int<PetscInt>(base_reduced_dofs[i]);
    base_vector_ids[i] = local_vector_index(base_full_dofs[i]);
  }
  plan->forward_base_vector_ids =
      DeviceDofId1D("vector_fe_ex9_forward_base_vector_ids", n_base_entries);
  copy_dof_indices_to_view(base_vector_ids, plan->forward_base_vector_ids);

  for (std::size_t e = 0; e != batch.elems.size(); ++e)
  {
    const auto external_indices = ex9_external_indices(batch, e);
    for (unsigned int i = 0; i != layout.external_n_dofs; ++i)
      plan->forward_coo_rows[n_base_entries + e * layout.external_n_dofs + i] =
          cast_int<PetscInt>(sc.reduced_dof_for_uncondensed_dof(*batch.elems[e],
                                                                external_indices[i]));
  }
  plan->forward_coo_values =
      DeviceReal1D("vector_fe_ex9_forward_elimination_coo_values", n_entries);
  plan->forward_coo_target = nullptr;
}

void
add_full_jacobian(Ex9FullBatch & batch,
                  SparseMatrix<Number> &,
                  StaticCondensation * sc,
                  const bool include_global_lm,
                  Ex9CondensedJacobianCooPlan & coo_plan)
{
  libmesh_error_msg_if(!sc,
                       "vector_fe_ex9 Kokkos Jacobian only supports the static-condensation "
                       "PETSc COO path.");
  {
    Ex9KokkosTimer timer("add condensed Jacobian to PETSc COO");
    add_device_condensed_jacobian_to_petsc_coo(batch, *sc, include_global_lm, coo_plan);
  }
  prepare_device_forward_elimination_plan(batch, *sc);

  auto backsub_plan = batch.backsub_plan;
  libmesh_assert(backsub_plan);
  sc->set_forward_elimination_callback(
      [backsub_plan](const NumericVector<Number> & full_rhs, NumericVector<Number> & reduced_rhs)
      { return run_device_forward_elimination(*backsub_plan, full_rhs, reduced_rhs); });
  sc->set_backwards_substitution_callback(
      [backsub_plan](const NumericVector<Number> & full_rhs, NumericVector<Number> & full_sol)
      { return run_device_backwards_substitution(*backsub_plan, full_rhs, full_sol); });
}

void
assemble_ex9_full_batch(const NumericVector<Number> & X,
                        NonlinearImplicitSystem & system,
                        HDGProblem & problem,
                        const AssemblyPurpose purpose,
                        Ex9FullBatch & batch)
{
  get_ex9_device_operator(system, problem).assemble(X, purpose, batch);
}

#endif

} // namespace

#if defined(LIBMESH_HAVE_KOKKOS) && defined(LIBMESH_HAVE_PETSC) &&                                 \
    !defined(LIBMESH_USE_COMPLEX_NUMBERS)
void
HDGProblem::kokkos_set_petsc_defaults(const bool need_ghost_vectors)
{
  PetscBool has_vec_type = PETSC_FALSE;
  PetscBool has_ghost_vec_type = PETSC_FALSE;
  PetscBool has_mat_type = PETSC_FALSE;
  LibmeshPetscCallA(PETSC_COMM_WORLD,
                    PetscOptionsHasName(nullptr, nullptr, "-vec_type", &has_vec_type));
  LibmeshPetscCallA(PETSC_COMM_WORLD,
                    PetscOptionsHasName(nullptr, nullptr, "-ghost_vec_type", &has_ghost_vec_type));
  LibmeshPetscCallA(PETSC_COMM_WORLD,
                    PetscOptionsHasName(nullptr, nullptr, "-mat_type", &has_mat_type));

  if (!has_vec_type)
    LibmeshPetscCallA(PETSC_COMM_WORLD, PetscOptionsSetValue(nullptr, "-vec_type", "kokkos"));
  if (need_ghost_vectors && !has_ghost_vec_type)
    LibmeshPetscCallA(PETSC_COMM_WORLD, PetscOptionsSetValue(nullptr, "-ghost_vec_type", "kokkos"));
  if (!has_mat_type)
    LibmeshPetscCallA(PETSC_COMM_WORLD, PetscOptionsSetValue(nullptr, "-mat_type", "aijkokkos"));
}

void
HDGProblem::kokkos_residual(const NumericVector<Number> & X,
                            NumericVector<Number> & R,
                            NonlinearImplicitSystem & S)
{
  Ex9KokkosTimer total_timer("kokkos_residual callback total");
  Ex9FullBatch batch;
  assemble_ex9_full_batch(X, S, *this, AssemblyPurpose::residual, batch);

  {
    Ex9KokkosTimer timer("residual vector zero", false);
    R.zero();
  }
  {
    Ex9KokkosTimer timer("add residual to PETSc COO");
    add_device_full_residual_to_petsc_coo(
        batch, R, this->cavity, get_ex9_device_operator(S, *this).residual_coo_plan());
  }
  {
    Ex9KokkosTimer timer("residual vector close", false);
    R.close();
  }
}

void
HDGProblem::kokkos_jacobian(const NumericVector<Number> & X,
                            SparseMatrix<Number> & J,
                            NonlinearImplicitSystem & S)
{
  Ex9KokkosTimer total_timer("kokkos_jacobian callback total");
  Ex9FullBatch batch;
  libmesh_error_msg_if(!this->sc,
                       "vector_fe_ex9 Kokkos Jacobian requires HDG static condensation with the "
                       "PETSc COO path.");
  assemble_ex9_full_batch(X, S, *this, AssemblyPurpose::jacobian, batch);

  add_full_jacobian(batch,
                    J,
                    this->sc,
                    this->cavity,
                    get_ex9_device_operator(S, *this).condensed_jacobian_coo_plan());
  {
    Ex9KokkosTimer timer("Jacobian close", false);
    J.close();
  }
}

void
HDGProblem::kokkos_clear_cache()
{
  clear_ex9_device_operator_cache(*this);
}
#endif

} // namespace libMesh
