// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

#ifndef LIBMESH_DOF_MAP_KOKKOS_H
#define LIBMESH_DOF_MAP_KOKKOS_H

#include "libmesh/dof_map.h"

#ifdef LIBMESH_HAVE_KOKKOS

#include "libmesh/kokkos_storage_policy.h"

#include <cstddef>
#include <vector>

namespace libMesh
{

struct DofMap::KokkosDofIndexCache
{
  using memory_space = typename ::Kokkos::DefaultExecutionSpace::memory_space;
  using elem_id_view = ::Kokkos::View<dof_id_type *, memory_space>;
  using elem_dof_id_view = ::Kokkos::View<dof_id_type **, memory_space>;
  using elem_dof_count_view = ::Kokkos::View<unsigned int *, memory_space>;
  using elem_subdomain_view = ::Kokkos::View<subdomain_id_type *, memory_space>;

  elem_id_view element_ids;
  elem_dof_id_view element_dof_indices;
  elem_dof_count_view element_n_dofs;
  elem_subdomain_view element_subdomains;
  std::vector<dof_id_type> host_element_ids;
  std::vector<dof_id_type> host_element_dof_indices;
  std::vector<unsigned int> host_element_n_dofs;
  std::vector<subdomain_id_type> host_element_subdomains;
  unsigned int max_dofs = 0;
};

struct DofMap::KokkosLocalIndexCache
{
  using memory_space = typename ::Kokkos::DefaultExecutionSpace::memory_space;
  using elem_local_index_view = ::Kokkos::View<unsigned int **, memory_space>;

  elem_local_index_view element_local_indices;
  unsigned int max_dofs = 0;
};

#if defined(LIBMESH_HAVE_PETSC) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)

template <typename T> class PetscMatrixBase;
template <typename T> class PetscVector;

/**
 * Controls how a Kokkos element assembly plan applies DoF constraints.
 */
struct KokkosConstraintAssemblyOptions
{
  bool asymmetric_constraint_rows = true;
  bool heterogeneous = false;
  int qoi_index = -1;
};

/**
 * A reusable mapping from unconstrained element matrices and vectors to a
 * constrained global COO stream.
 *
 * Physics kernels fill element_matrices and element_rhs.  A subsequent
 * device kernel applies C^T K C and C^T(F-KH), including libMesh's constraint
 * equations, into matrix_values and rhs_values.  The same plan and storage can
 * be reused until the associated KokkosDofIndexCache or DofMap constraints
 * change.
 */
struct KokkosConstraintAssemblyPlan
{
  using memory_space = typename ::Kokkos::DefaultExecutionSpace::memory_space;
  using real_view = ::Kokkos::View<Real *, memory_space>;
  using element_real_view = ::Kokkos::View<Real **, memory_space>;
  using size_view = ::Kokkos::View<std::size_t *, memory_space>;
  using uint_view = ::Kokkos::View<unsigned int *, memory_space>;

  // Retaining this shallow view both identifies the source cache and prevents
  // its allocation address from being recycled while the plan is alive.
  DofMap::KokkosDofIndexCache::elem_dof_id_view dof_index_identity;
  unsigned int n_elements = 0;
  unsigned int max_element_dofs = 0;
  KokkosConstraintAssemblyOptions options;

  uint_view expanded_dof_counts;
  size_view rhs_offsets;
  size_view matrix_offsets;
  size_view constraint_matrix_offsets;
  real_view constraint_matrix_values;
  real_view constraint_shifts;
  size_view constraint_row_offsets;
  uint_view constraint_row_columns;
  real_view constraint_row_values;
  real_view constraint_row_rhs_values;

  element_real_view element_matrices;
  element_real_view element_rhs;
  real_view rhs_values;
  real_view matrix_values;
  std::vector<dof_id_type> host_rhs_rows;
  std::vector<dof_id_type> host_matrix_rows;
  std::vector<dof_id_type> host_matrix_columns;

  const void * matrix_target = nullptr;
  const void * rhs_target = nullptr;
};

bool kokkos_constraint_assembly_plan_matches(
    const KokkosConstraintAssemblyPlan & plan,
    const DofMap::KokkosDofIndexCache & dof_index_cache,
    const KokkosConstraintAssemblyOptions & options = {});

void build_kokkos_constraint_assembly_plan(
    const DofMap & dof_map,
    const DofMap::KokkosDofIndexCache & dof_index_cache,
    KokkosConstraintAssemblyPlan & plan,
    const KokkosConstraintAssemblyOptions & options = {});

/**
 * Transform values previously written to plan.element_matrices and
 * plan.element_rhs.  Set transform_matrix=false to retain already-computed COO
 * matrix values while refreshing only the RHS.
 */
void transform_kokkos_element_values(
    KokkosConstraintAssemblyPlan & plan,
    const DofMap::KokkosDofIndexCache & dof_index_cache,
    bool transform_matrix = true);

void prepare_kokkos_petsc_coo(const Parallel::Communicator & comm,
                              KokkosConstraintAssemblyPlan & plan,
                              PetscMatrixBase<Number> & matrix,
                              PetscVector<Number> & rhs);

void add_kokkos_petsc_coo_values(const Parallel::Communicator & comm,
                                 const KokkosConstraintAssemblyPlan & plan,
                                 PetscMatrixBase<Number> & matrix,
                                 PetscVector<Number> & rhs);

#endif

} // namespace libMesh

#endif

#endif
