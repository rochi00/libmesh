// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public  License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

#include "libmesh/dof_map_kokkos.h"

#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"
#include "libmesh/mesh_base_kokkos.h"

#ifdef LIBMESH_HAVE_PETSC
#include "libmesh/petsc_matrix_base.h"
#include "libmesh/petsc_macro.h"
#include "libmesh/petsc_vector.h"
#endif

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace libMesh
{

#ifdef LIBMESH_HAVE_KOKKOS
void DofMap::KokkosDofIndexCacheDeleter::operator()(KokkosDofIndexCache * cache) const
{
  delete cache;
}

void DofMap::KokkosLocalIndexCacheDeleter::operator()(KokkosLocalIndexCache * cache) const
{
  delete cache;
}

const DofMap::KokkosDofIndexCache *
DofMap::get_kokkos_dof_index_cache(const unsigned int vn) const
{
  if (auto it = _kokkos_dof_index_caches.find(vn); it != _kokkos_dof_index_caches.end())
    return it->second.get();

  const auto & geometry_cache = _mesh.get_kokkos_geometry_cache();
  auto cache = kokkos_dof_index_cache_ptr(new KokkosDofIndexCache());
  std::vector<dof_id_type> dof_indices;
  cache->host_element_ids = geometry_cache.host_element_ids;

  for (const auto elem_id : cache->host_element_ids)
    {
      const Elem * elem = _mesh.query_elem_ptr(elem_id);
      libmesh_assert(elem);
      this->dof_indices(elem, dof_indices, vn);
      cache->max_dofs = std::max(cache->max_dofs, cast_int<unsigned int>(dof_indices.size()));
    }

  cache->element_ids =
    KokkosDofIndexCache::elem_id_view("dof_map_kokkos_element_ids",
                                      cache->host_element_ids.size());
  cache->element_dof_indices =
    KokkosDofIndexCache::elem_dof_id_view("dof_map_kokkos_dof_indices",
                                          cache->host_element_ids.size(),
                                          cache->max_dofs);
  cache->element_n_dofs =
    KokkosDofIndexCache::elem_dof_count_view("dof_map_kokkos_n_dofs",
                                             cache->host_element_ids.size());
  cache->element_subdomains =
    KokkosDofIndexCache::elem_subdomain_view("dof_map_kokkos_elem_subdomains",
                                             cache->host_element_ids.size());

  auto h_element_ids = ::Kokkos::create_mirror_view(cache->element_ids);
  auto h_dof_indices = ::Kokkos::create_mirror_view(cache->element_dof_indices);
  auto h_n_dofs = ::Kokkos::create_mirror_view(cache->element_n_dofs);
  auto h_subdomains = ::Kokkos::create_mirror_view(cache->element_subdomains);
  cache->host_element_dof_indices.resize(cache->host_element_ids.size() * cache->max_dofs,
                                         DofObject::invalid_id);
  cache->host_element_n_dofs.resize(cache->host_element_ids.size(), 0);
  cache->host_element_subdomains.resize(cache->host_element_ids.size(),
                                        static_cast<subdomain_id_type>(Elem::invalid_subdomain_id));

  for (auto elem_index : index_range(cache->host_element_ids))
    {
      const dof_id_type elem_id = cache->host_element_ids[elem_index];
      const Elem * elem = _mesh.query_elem_ptr(elem_id);
      libmesh_assert(elem);
      h_element_ids(cast_int<unsigned int>(elem_index)) = elem_id;
      h_subdomains(cast_int<unsigned int>(elem_index)) = elem->subdomain_id();
      cache->host_element_subdomains[elem_index] = elem->subdomain_id();
      this->dof_indices(elem, dof_indices, vn);
      h_n_dofs(cast_int<unsigned int>(elem_index)) = cast_int<unsigned int>(dof_indices.size());
      cache->host_element_n_dofs[elem_index] = cast_int<unsigned int>(dof_indices.size());
      for (auto i : index_range(dof_indices))
        {
          h_dof_indices(cast_int<unsigned int>(elem_index), cast_int<unsigned int>(i)) =
            dof_indices[i];
          cache->host_element_dof_indices[elem_index * cache->max_dofs + i] = dof_indices[i];
        }
    }

  ::Kokkos::deep_copy(cache->element_ids, h_element_ids);
  ::Kokkos::deep_copy(cache->element_dof_indices, h_dof_indices);
  ::Kokkos::deep_copy(cache->element_n_dofs, h_n_dofs);
  ::Kokkos::deep_copy(cache->element_subdomains, h_subdomains);

  auto [it, inserted] = _kokkos_dof_index_caches.emplace(vn, std::move(cache));
  libmesh_ignore(inserted);
  return it->second.get();
}

const DofMap::KokkosLocalIndexCache *
DofMap::get_kokkos_local_index_cache(const NumericVector<Number> & local_vector,
                                     const unsigned int vn) const
{
#ifdef LIBMESH_HAVE_PETSC
  const auto key = std::make_pair(vn, &local_vector);
  if (auto it = _kokkos_local_index_caches.find(key); it != _kokkos_local_index_caches.end())
    return it->second.get();

  const auto * petsc_vector = dynamic_cast<const PetscVector<Number> *>(&local_vector);
  if (!petsc_vector)
    return nullptr;

  const auto * dof_index_cache = this->get_kokkos_dof_index_cache(vn);
  if (!dof_index_cache)
    return nullptr;

  const auto & geometry_cache = _mesh.get_kokkos_geometry_cache();
  auto cache = kokkos_local_index_cache_ptr(new KokkosLocalIndexCache());
  cache->max_dofs = dof_index_cache->max_dofs;

  cache->element_local_indices =
    KokkosLocalIndexCache::elem_local_index_view("dof_map_kokkos_local_indices",
                                                 geometry_cache.host_element_ids.size(),
                                                 cache->max_dofs);
  auto h_local_indices = ::Kokkos::create_mirror_view(cache->element_local_indices);

  for (auto elem_index : index_range(geometry_cache.host_element_ids))
    {
      const unsigned int n_dofs = dof_index_cache->host_element_n_dofs[elem_index];
      for (unsigned int i = 0; i != n_dofs; ++i)
        h_local_indices(cast_int<unsigned int>(elem_index), i) =
          cast_int<unsigned int>(
            petsc_vector->map_global_to_local_index(
              dof_index_cache->host_element_dof_indices[elem_index * cache->max_dofs + i]));
    }

  ::Kokkos::deep_copy(cache->element_local_indices, h_local_indices);
  auto [it, inserted] = _kokkos_local_index_caches.emplace(key, std::move(cache));
  libmesh_ignore(inserted);
  return it->second.get();
#else
  libmesh_ignore(local_vector, vn);
  return nullptr;
#endif
}

const DofMap::KokkosLocalIndexCache *
DofMap::require_kokkos_local_index_cache(const NumericVector<Number> & local_vector,
                                         const unsigned int vn) const
{
  this->prepare_kokkos_local_index_cache(local_vector, vn);
  return this->get_kokkos_local_index_cache(local_vector, vn);
}

void DofMap::prepare_kokkos_dof_index_caches() const
{
  for (auto vn : make_range(this->n_variables()))
    libmesh_ignore(this->get_kokkos_dof_index_cache(vn));
}

void
DofMap::prepare_kokkos_local_index_cache(const NumericVector<Number> & local_vector,
                                         const unsigned int vn) const
{
  libmesh_ignore(this->get_kokkos_local_index_cache(local_vector, vn));
}

#if defined(LIBMESH_HAVE_PETSC) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)
namespace
{

template <typename View>
void copy_vector_to_kokkos_view(
    const std::vector<typename View::non_const_value_type> & source,
    const View & destination)
{
  libmesh_assert_equal_to(source.size(), destination.extent(0));
  auto host = ::Kokkos::create_mirror_view(destination);
  for (std::size_t i = 0; i != source.size(); ++i)
    host(i) = source[i];
  ::Kokkos::deep_copy(destination, host);
}

void append_constraint_dependencies(const DofConstraints & constraints,
                                    std::vector<dof_id_type> & expanded_dofs)
{
  while (true)
    {
      std::set<dof_id_type> new_dofs;
      for (const dof_id_type dof : expanded_dofs)
        if (const auto row = constraints.find(dof); row != constraints.end())
          for (const auto & [dependency, coefficient] : row->second)
            {
              libmesh_ignore(coefficient);
              if (std::find(expanded_dofs.begin(), expanded_dofs.end(), dependency) ==
                  expanded_dofs.end())
                new_dofs.insert(dependency);
            }

      if (new_dofs.empty())
        return;

      expanded_dofs.insert(expanded_dofs.end(), new_dofs.begin(), new_dofs.end());
    }
}

Real direct_constraint_shift(const DofMap & dof_map,
                             const dof_id_type dof,
                             const KokkosConstraintAssemblyOptions & options)
{
  if (!options.heterogeneous)
    return 0.;

  if (options.qoi_index >= 0)
    return libmesh_real(dof_map.has_heterogeneous_adjoint_constraint(
        cast_int<unsigned int>(options.qoi_index), dof));

  // The legacy accessor is non-const even though reading the map is harmless.
  const auto & values = const_cast<DofMap &>(dof_map).get_primal_constraint_values();
  if (const auto value = values.find(dof); value != values.end())
    return libmesh_real(value->second);
  return 0.;
}

Real expand_constraint_dof(const DofMap & dof_map,
                           const DofConstraints & constraints,
                           const KokkosConstraintAssemblyOptions & options,
                           const std::map<dof_id_type, unsigned int> & positions,
                           const dof_id_type dof,
                           const Real scale,
                           std::vector<Real> & coefficients,
                           std::set<dof_id_type> & active_constraints)
{
  const auto row = constraints.find(dof);
  if (row == constraints.end())
    {
      const auto position = positions.find(dof);
      libmesh_assert(position != positions.end());
      coefficients[position->second] += scale;
      return 0.;
    }

  libmesh_error_msg_if(!active_constraints.insert(dof).second,
                       "Cyclic DoF constraint encountered while building a Kokkos "
                       "constraint assembly plan");
  Real shift = scale * direct_constraint_shift(dof_map, dof, options);
  for (const auto & [dependency, coefficient] : row->second)
    shift += expand_constraint_dof(dof_map,
                                   constraints,
                                   options,
                                   positions,
                                   dependency,
                                   scale * coefficient,
                                   coefficients,
                                   active_constraints);
  active_constraints.erase(dof);
  return shift;
}

void validate_constraint_transform(
    const DofMap & dof_map,
    const KokkosConstraintAssemblyOptions & options,
    const std::vector<dof_id_type> & raw_dofs,
    const std::vector<dof_id_type> & expanded_dofs,
    const std::vector<Real> & constraint_matrix,
    const std::vector<Real> & constraint_shifts,
    const std::vector<std::size_t> & row_offsets,
    const std::vector<unsigned int> & row_columns,
    const std::vector<Real> & row_values,
    const std::vector<Real> & row_rhs_values)
{
#ifndef NDEBUG
  const unsigned int n = cast_int<unsigned int>(raw_dofs.size());
  const unsigned int m = cast_int<unsigned int>(expanded_dofs.size());
  DenseMatrix<Number> raw_matrix(n, n);
  DenseVector<Number> raw_rhs(n);
  for (unsigned int i = 0; i != n; ++i)
    {
      raw_rhs(i) = 0.5 + 0.11 * i;
      for (unsigned int j = 0; j != n; ++j)
        raw_matrix(i, j) = 1. + 0.17 * i + 0.013 * j + 0.001 * i * j;
    }

  DenseMatrix<Number> expected_matrix(raw_matrix);
  DenseVector<Number> expected_rhs(raw_rhs);
  std::vector<dof_id_type> expected_dofs = raw_dofs;
  if (options.heterogeneous)
    dof_map.heterogeneously_constrain_element_matrix_and_vector(
        expected_matrix,
        expected_rhs,
        expected_dofs,
        options.asymmetric_constraint_rows,
        options.qoi_index);
  else
    dof_map.constrain_element_matrix_and_vector(expected_matrix,
                                                expected_rhs,
                                                expected_dofs,
                                                options.asymmetric_constraint_rows);

  libmesh_assert_equal_to(expected_dofs.size(), m);
  for (unsigned int a = 0; a != m; ++a)
    libmesh_assert_equal_to(expected_dofs[a], expanded_dofs[a]);

  for (unsigned int a = 0; a != m; ++a)
    {
      Real transformed_rhs = 0.;
      if (row_offsets[a] != row_offsets[a + 1])
        transformed_rhs = row_rhs_values[a];
      else
        for (unsigned int i = 0; i != n; ++i)
          {
            Real shifted_rhs = raw_rhs(i);
            for (unsigned int j = 0; j != n; ++j)
              shifted_rhs -= raw_matrix(i, j) * constraint_shifts[j];
            transformed_rhs += constraint_matrix[i * m + a] * shifted_rhs;
          }
      libmesh_assert_less(std::abs(transformed_rhs - expected_rhs(a)), 1.e-12);

      for (unsigned int b = 0; b != m; ++b)
        {
          Real transformed_matrix = 0.;
          if (row_offsets[a] != row_offsets[a + 1])
            {
              for (std::size_t entry = row_offsets[a]; entry != row_offsets[a + 1]; ++entry)
                if (row_columns[entry] == b)
                  transformed_matrix = row_values[entry];
            }
          else
            for (unsigned int i = 0; i != n; ++i)
              for (unsigned int j = 0; j != n; ++j)
                transformed_matrix += constraint_matrix[i * m + a] * raw_matrix(i, j) *
                                      constraint_matrix[j * m + b];
          libmesh_assert_less(std::abs(transformed_matrix - expected_matrix(a, b)), 1.e-12);
        }
    }
#else
  libmesh_ignore(dof_map,
                 options,
                 raw_dofs,
                 expanded_dofs,
                 constraint_matrix,
                 constraint_shifts,
                 row_offsets,
                 row_columns,
                 row_values,
                 row_rhs_values);
#endif
}

bool petsc_vector_uses_kokkos(const Parallel::Communicator & comm, Vec vector)
{
  const char * type = nullptr;
  LibmeshPetscCall2(comm, VecGetType(vector, &type));
  PetscBool is_seq_kokkos = PETSC_FALSE;
  PetscBool is_mpi_kokkos = PETSC_FALSE;
  LibmeshPetscCall2(comm, PetscStrcmp(type, "seqkokkos", &is_seq_kokkos));
  LibmeshPetscCall2(comm, PetscStrcmp(type, "mpikokkos", &is_mpi_kokkos));
  return is_seq_kokkos || is_mpi_kokkos;
}

bool petsc_matrix_uses_kokkos(const Parallel::Communicator & comm, Mat matrix)
{
  const char * type = nullptr;
  LibmeshPetscCall2(comm, MatGetType(matrix, &type));
  PetscBool is_seq_kokkos = PETSC_FALSE;
  PetscBool is_mpi_kokkos = PETSC_FALSE;
  LibmeshPetscCall2(comm, PetscStrcmp(type, MATSEQAIJKOKKOS, &is_seq_kokkos));
  LibmeshPetscCall2(comm, PetscStrcmp(type, MATMPIAIJKOKKOS, &is_mpi_kokkos));
  return is_seq_kokkos || is_mpi_kokkos;
}

} // anonymous namespace

bool kokkos_constraint_assembly_plan_matches(
    const KokkosConstraintAssemblyPlan & plan,
    const DofMap::KokkosDofIndexCache & dof_index_cache,
    const KokkosConstraintAssemblyOptions & options)
{
  return plan.dof_index_identity.data() == dof_index_cache.element_dof_indices.data() &&
         plan.n_elements == dof_index_cache.host_element_ids.size() &&
         plan.max_element_dofs == dof_index_cache.max_dofs &&
         plan.options.asymmetric_constraint_rows == options.asymmetric_constraint_rows &&
         plan.options.heterogeneous == options.heterogeneous &&
         plan.options.qoi_index == options.qoi_index;
}

void build_kokkos_constraint_assembly_plan(
    const DofMap & dof_map,
    const DofMap::KokkosDofIndexCache & dof_index_cache,
    KokkosConstraintAssemblyPlan & plan,
    const KokkosConstraintAssemblyOptions & options)
{
  ::Kokkos::Profiling::ScopedRegion region("libMesh build constrained COO assembly plan");
  plan = KokkosConstraintAssemblyPlan{};
  plan.dof_index_identity = dof_index_cache.element_dof_indices;
  plan.n_elements = cast_int<unsigned int>(dof_index_cache.host_element_ids.size());
  plan.max_element_dofs = dof_index_cache.max_dofs;
  plan.options = options;

  const DofConstraints & constraints = dof_map.get_dof_constraints();
  std::vector<unsigned int> expanded_counts;
  std::vector<std::size_t> rhs_offsets(1, 0);
  std::vector<std::size_t> matrix_offsets(1, 0);
  std::vector<std::size_t> constraint_matrix_offsets(1, 0);
  std::vector<Real> constraint_matrix_values;
  std::vector<Real> constraint_shifts(plan.n_elements * plan.max_element_dofs, 0.);
  std::vector<std::size_t> constraint_row_offsets(1, 0);
  std::vector<unsigned int> constraint_row_columns;
  std::vector<Real> constraint_row_values;
  std::vector<Real> constraint_row_rhs_values;

  expanded_counts.reserve(plan.n_elements);
  for (unsigned int e = 0; e != plan.n_elements; ++e)
    {
      const unsigned int n = dof_index_cache.host_element_n_dofs[e];
      const auto dof_begin = dof_index_cache.host_element_dof_indices.begin() +
                             e * dof_index_cache.max_dofs;
      const std::vector<dof_id_type> raw_dofs(dof_begin, dof_begin + n);
      std::vector<dof_id_type> expanded_dofs = raw_dofs;
      append_constraint_dependencies(constraints, expanded_dofs);
      const unsigned int m = cast_int<unsigned int>(expanded_dofs.size());
      expanded_counts.push_back(m);

      std::map<dof_id_type, unsigned int> positions;
      for (unsigned int a = 0; a != m; ++a)
        positions.emplace(expanded_dofs[a], a);

      std::vector<Real> element_constraint_matrix(n * m, 0.);
      std::vector<Real> element_constraint_shifts(n, 0.);
      for (unsigned int i = 0; i != n; ++i)
        {
          std::set<dof_id_type> active_constraints;
          std::vector<Real> coefficients(m, 0.);
          element_constraint_shifts[i] =
              expand_constraint_dof(dof_map,
                                    constraints,
                                    options,
                                    positions,
                                    raw_dofs[i],
                                    1.,
                                    coefficients,
                                    active_constraints);
          std::copy(coefficients.begin(),
                    coefficients.end(),
                    element_constraint_matrix.begin() + i * m);
        }
      constraint_matrix_values.insert(constraint_matrix_values.end(),
                                      element_constraint_matrix.begin(),
                                      element_constraint_matrix.end());
      constraint_matrix_offsets.push_back(constraint_matrix_values.size());
      std::copy(element_constraint_shifts.begin(),
                element_constraint_shifts.end(),
                constraint_shifts.begin() + e * plan.max_element_dofs);

      const std::size_t row_entry_begin = constraint_row_values.size();
      std::vector<std::size_t> element_row_offsets(1, 0);
      std::vector<Real> element_row_rhs_values;
      for (unsigned int a = 0; a != m; ++a)
        {
          const auto row = constraints.find(expanded_dofs[a]);
          if (row != constraints.end())
            {
              std::map<unsigned int, Real> row_entries;
              row_entries[a] = 1.;
              if (options.asymmetric_constraint_rows)
                for (const auto & [dependency, coefficient] : row->second)
                  {
                    const auto position = positions.find(dependency);
                    libmesh_assert(position != positions.end());
                    row_entries[position->second] = -coefficient;
                  }
              for (const auto & [column, value] : row_entries)
                {
                  constraint_row_columns.push_back(column);
                  constraint_row_values.push_back(value);
                }
            }
          element_row_offsets.push_back(constraint_row_values.size() - row_entry_begin);
          const Real row_rhs = row != constraints.end() && options.heterogeneous &&
                                       options.asymmetric_constraint_rows
                                   ? direct_constraint_shift(dof_map, expanded_dofs[a], options)
                                   : 0.;
          element_row_rhs_values.push_back(row_rhs);
          constraint_row_rhs_values.push_back(row_rhs);
          constraint_row_offsets.push_back(constraint_row_values.size());
        }

      validate_constraint_transform(
          dof_map,
          options,
          raw_dofs,
          expanded_dofs,
          element_constraint_matrix,
          element_constraint_shifts,
          element_row_offsets,
          std::vector<unsigned int>(constraint_row_columns.begin() + row_entry_begin,
                                    constraint_row_columns.end()),
          std::vector<Real>(constraint_row_values.begin() + row_entry_begin,
                            constraint_row_values.end()),
          element_row_rhs_values);

      for (const dof_id_type dof : expanded_dofs)
        plan.host_rhs_rows.push_back(dof);
      for (const dof_id_type row : expanded_dofs)
        for (const dof_id_type column : expanded_dofs)
          {
            plan.host_matrix_rows.push_back(row);
            plan.host_matrix_columns.push_back(column);
          }

      rhs_offsets.push_back(plan.host_rhs_rows.size());
      matrix_offsets.push_back(plan.host_matrix_rows.size());
    }

  plan.expanded_dof_counts = KokkosConstraintAssemblyPlan::uint_view(
      "libmesh constrained COO expanded DoF counts", plan.n_elements);
  plan.rhs_offsets = KokkosConstraintAssemblyPlan::size_view(
      "libmesh constrained COO RHS offsets", plan.n_elements + 1);
  plan.matrix_offsets = KokkosConstraintAssemblyPlan::size_view(
      "libmesh constrained COO matrix offsets", plan.n_elements + 1);
  plan.constraint_matrix_offsets = KokkosConstraintAssemblyPlan::size_view(
      "libmesh constrained COO transform offsets", plan.n_elements + 1);
  plan.constraint_matrix_values = KokkosConstraintAssemblyPlan::real_view(
      "libmesh constrained COO transform values", constraint_matrix_values.size());
  plan.constraint_shifts = KokkosConstraintAssemblyPlan::real_view(
      "libmesh constrained COO shifts", constraint_shifts.size());
  plan.constraint_row_offsets = KokkosConstraintAssemblyPlan::size_view(
      "libmesh constrained COO row offsets", constraint_row_offsets.size());
  plan.constraint_row_columns = KokkosConstraintAssemblyPlan::uint_view(
      "libmesh constrained COO row columns", constraint_row_columns.size());
  plan.constraint_row_values = KokkosConstraintAssemblyPlan::real_view(
      "libmesh constrained COO row values", constraint_row_values.size());
  plan.constraint_row_rhs_values = KokkosConstraintAssemblyPlan::real_view(
      "libmesh constrained COO row RHS values", constraint_row_rhs_values.size());

  copy_vector_to_kokkos_view(expanded_counts, plan.expanded_dof_counts);
  copy_vector_to_kokkos_view(rhs_offsets, plan.rhs_offsets);
  copy_vector_to_kokkos_view(matrix_offsets, plan.matrix_offsets);
  copy_vector_to_kokkos_view(constraint_matrix_offsets, plan.constraint_matrix_offsets);
  copy_vector_to_kokkos_view(constraint_matrix_values, plan.constraint_matrix_values);
  copy_vector_to_kokkos_view(constraint_shifts, plan.constraint_shifts);
  copy_vector_to_kokkos_view(constraint_row_offsets, plan.constraint_row_offsets);
  copy_vector_to_kokkos_view(constraint_row_columns, plan.constraint_row_columns);
  copy_vector_to_kokkos_view(constraint_row_values, plan.constraint_row_values);
  copy_vector_to_kokkos_view(constraint_row_rhs_values, plan.constraint_row_rhs_values);

  plan.element_matrices = KokkosConstraintAssemblyPlan::element_real_view(
      "libmesh unconstrained element matrices",
      plan.n_elements,
      plan.max_element_dofs * plan.max_element_dofs);
  plan.element_rhs = KokkosConstraintAssemblyPlan::element_real_view(
      "libmesh unconstrained element RHS", plan.n_elements, plan.max_element_dofs);
  plan.rhs_values = KokkosConstraintAssemblyPlan::real_view(
      "libmesh constrained COO RHS values", plan.host_rhs_rows.size());
  plan.matrix_values = KokkosConstraintAssemblyPlan::real_view(
      "libmesh constrained COO matrix values", plan.host_matrix_rows.size());
}

void transform_kokkos_element_values(
    KokkosConstraintAssemblyPlan & plan,
    const DofMap::KokkosDofIndexCache & dof_index_cache,
    const bool transform_matrix)
{
  libmesh_assert(kokkos_constraint_assembly_plan_matches(plan,
                                                        dof_index_cache,
                                                        plan.options));
  const auto n_dofs = dof_index_cache.element_n_dofs;
  const auto expanded_dof_counts = plan.expanded_dof_counts;
  const auto rhs_offsets = plan.rhs_offsets;
  const auto matrix_offsets = plan.matrix_offsets;
  const auto constraint_matrix_offsets = plan.constraint_matrix_offsets;
  const auto constraint_matrix_values = plan.constraint_matrix_values;
  const auto constraint_shifts = plan.constraint_shifts;
  const auto constraint_row_offsets = plan.constraint_row_offsets;
  const auto constraint_row_columns = plan.constraint_row_columns;
  const auto constraint_row_values = plan.constraint_row_values;
  const auto constraint_row_rhs_values = plan.constraint_row_rhs_values;
  const auto element_matrices = plan.element_matrices;
  const auto element_rhs = plan.element_rhs;
  const auto rhs_values = plan.rhs_values;
  const auto matrix_values = plan.matrix_values;
  const unsigned int matrix_stride = plan.max_element_dofs;
  const unsigned int n_elements = plan.n_elements;
  const bool heterogeneous = plan.options.heterogeneous;
  using ExecutionSpace = typename KokkosConstraintAssemblyPlan::real_view::execution_space;

  ::Kokkos::parallel_for(
      "libMesh constrain element COO values",
      ::Kokkos::RangePolicy<ExecutionSpace>(0, cast_int<int>(n_elements)),
      KOKKOS_LAMBDA(const int raw_elem) {
        const unsigned int e = static_cast<unsigned int>(raw_elem);
        const unsigned int n = n_dofs(e);
        const unsigned int m = expanded_dof_counts(e);
        const std::size_t rhs_entry_offset = rhs_offsets(e);
        const std::size_t matrix_entry_offset = matrix_offsets(e);
        const std::size_t transform_offset = constraint_matrix_offsets(e);
        const std::size_t shift_offset = e * matrix_stride;

        for (unsigned int a = 0; a != m; ++a)
          {
            const std::size_t constraint_row = rhs_entry_offset + a;
            const std::size_t row_begin = constraint_row_offsets(constraint_row);
            const std::size_t row_end = constraint_row_offsets(constraint_row + 1);
            Real value = 0.;
            if (row_begin != row_end)
              value = constraint_row_rhs_values(constraint_row);
            else
              for (unsigned int i = 0; i != n; ++i)
                {
                  Real shifted_rhs = element_rhs(e, i);
                  if (heterogeneous)
                    for (unsigned int j = 0; j != n; ++j)
                      shifted_rhs -= element_matrices(e, i * matrix_stride + j) *
                                     constraint_shifts(shift_offset + j);
                  value += constraint_matrix_values(transform_offset + i * m + a) *
                           shifted_rhs;
                }
            rhs_values(rhs_entry_offset + a) = value;
          }

        if (transform_matrix)
          for (unsigned int a = 0; a != m; ++a)
            {
              const std::size_t constraint_row = rhs_entry_offset + a;
              const std::size_t row_begin = constraint_row_offsets(constraint_row);
              const std::size_t row_end = constraint_row_offsets(constraint_row + 1);
              for (unsigned int b = 0; b != m; ++b)
                {
                  Real value = 0.;
                  if (row_begin != row_end)
                    {
                      for (std::size_t entry = row_begin; entry != row_end; ++entry)
                        if (constraint_row_columns(entry) == b)
                          value = constraint_row_values(entry);
                    }
                  else
                    for (unsigned int i = 0; i != n; ++i)
                      {
                        const Real row_coefficient =
                            constraint_matrix_values(transform_offset + i * m + a);
                        for (unsigned int j = 0; j != n; ++j)
                          value += row_coefficient *
                                   element_matrices(e, i * matrix_stride + j) *
                                   constraint_matrix_values(transform_offset + j * m + b);
                      }
                  matrix_values(matrix_entry_offset + a * m + b) = value;
                }
            }
      });
  ::Kokkos::fence();
}

void prepare_kokkos_petsc_coo(const Parallel::Communicator & comm,
                              KokkosConstraintAssemblyPlan & plan,
                              PetscMatrixBase<Number> & matrix,
                              PetscVector<Number> & rhs)
{
  if (plan.matrix_target != matrix.mat())
    {
      std::vector<PetscInt> rows(plan.host_matrix_rows.size());
      std::vector<PetscInt> columns(plan.host_matrix_columns.size());
      for (auto i : index_range(rows))
        {
          rows[i] = cast_int<PetscInt>(plan.host_matrix_rows[i]);
          columns[i] = cast_int<PetscInt>(plan.host_matrix_columns[i]);
        }
      LibmeshPetscCall2(comm,
                        MatSetPreallocationCOO(matrix.mat(),
                                               cast_int<PetscCount>(rows.size()),
                                               rows.data(),
                                               columns.data()));
      LibmeshPetscCall2(
          comm, MatSetOption(matrix.mat(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
      plan.matrix_target = matrix.mat();
    }

  if (plan.rhs_target != rhs.vec())
    {
      std::vector<PetscInt> rows(plan.host_rhs_rows.size());
      for (auto i : index_range(rows))
        rows[i] = cast_int<PetscInt>(plan.host_rhs_rows[i]);
      LibmeshPetscCall2(comm,
                        VecSetPreallocationCOO(
                            rhs.vec(), cast_int<PetscCount>(rows.size()), rows.data()));
      plan.rhs_target = rhs.vec();
    }
}

void add_kokkos_petsc_coo_values(const Parallel::Communicator & comm,
                                 const KokkosConstraintAssemblyPlan & plan,
                                 PetscMatrixBase<Number> & matrix,
                                 PetscVector<Number> & rhs)
{
  libmesh_assert_equal_to(plan.matrix_target, matrix.mat());
  libmesh_assert_equal_to(plan.rhs_target, rhs.vec());
  constexpr bool host_accessible =
      ::Kokkos::SpaceAccessibility<
          ::Kokkos::DefaultHostExecutionSpace,
          typename KokkosConstraintAssemblyPlan::real_view::memory_space>::accessible;

  if constexpr (host_accessible)
    {
      LibmeshPetscCall2(comm, VecSetValuesCOO(rhs.vec(), plan.rhs_values.data(), ADD_VALUES));
      LibmeshPetscCall2(
          comm, MatSetValuesCOO(matrix.mat(), plan.matrix_values.data(), ADD_VALUES));
    }
  else
    {
      if (petsc_vector_uses_kokkos(comm, rhs.vec()))
        LibmeshPetscCall2(comm, VecSetValuesCOO(rhs.vec(), plan.rhs_values.data(), ADD_VALUES));
      else
        {
          auto host_rhs = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(),
                                                                plan.rhs_values);
          LibmeshPetscCall2(comm, VecSetValuesCOO(rhs.vec(), host_rhs.data(), ADD_VALUES));
        }

      if (petsc_matrix_uses_kokkos(comm, matrix.mat()))
        LibmeshPetscCall2(
            comm, MatSetValuesCOO(matrix.mat(), plan.matrix_values.data(), ADD_VALUES));
      else
        {
          auto host_matrix = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(),
                                                                   plan.matrix_values);
          LibmeshPetscCall2(
              comm, MatSetValuesCOO(matrix.mat(), host_matrix.data(), ADD_VALUES));
        }
    }
}
#endif
#endif

} // namespace libMesh
