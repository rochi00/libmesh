// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

#ifndef LIBMESH_KOKKOS_HDG_ASSEMBLY_H
#define LIBMESH_KOKKOS_HDG_ASSEMBLY_H

#include "kokkos_fe_evaluator.h"
#include "kokkos_fe_map.h"

#define PETSC_SKIP_CXX_COMPLEX_FIX 1
#include <Kokkos_Core.hpp>
#undef __CUDACC_VER__

namespace libMesh::Kokkos::detail
{

constexpr unsigned int hdg_linear_condense_max_internal = 32;
constexpr unsigned int hdg_linear_condense_max_external = 32;
constexpr unsigned int hdg_linear_condense_max_cols =
    hdg_linear_condense_max_internal + hdg_linear_condense_max_external;
constexpr unsigned int hdg_linear_condense_aug_size =
    hdg_linear_condense_max_internal * hdg_linear_condense_max_cols;
constexpr unsigned int hdg_linear_condense_aux_size =
    hdg_linear_condense_max_internal * hdg_linear_condense_max_external;

constexpr int hdg_linear_side_interior = 0;
constexpr int hdg_linear_side_outlet = 1;
constexpr int hdg_linear_side_dirichlet = 2;

struct HDGLinearBlockLayout
{
  unsigned int vector_n_dofs = 0;
  unsigned int scalar_n_dofs = 0;
  unsigned int lm_n_dofs = 0;
  unsigned int off_Jqq = 0;
  unsigned int off_Jqu = 0;
  unsigned int off_Juq = 0;
  unsigned int off_Jup = 0;
  unsigned int off_Jpu = 0;
  unsigned int off_Jplm = 0;
  unsigned int off_Jqlm = 0;
  unsigned int off_Jlmq = 0;
  unsigned int off_Jlmp = 0;
  unsigned int off_Jlms = 0;
  unsigned int off_Jlmlm = 0;
  unsigned int off_Juu = 0;
  unsigned int off_Julm = 0;
  unsigned int n_entries = 0;
};

inline HDGLinearBlockLayout
make_hdg_linear_block_layout(const unsigned int vector_n_dofs,
                             const unsigned int scalar_n_dofs,
                             const unsigned int lm_n_dofs)
{
  HDGLinearBlockLayout layout;
  layout.vector_n_dofs = vector_n_dofs;
  layout.scalar_n_dofs = scalar_n_dofs;
  layout.lm_n_dofs = lm_n_dofs;
  layout.off_Jqu = layout.off_Jqq + vector_n_dofs * vector_n_dofs;
  layout.off_Juq = layout.off_Jqu + vector_n_dofs * scalar_n_dofs;
  layout.off_Jup = layout.off_Juq + scalar_n_dofs * vector_n_dofs;
  layout.off_Jpu = layout.off_Jup + scalar_n_dofs * scalar_n_dofs;
  layout.off_Jplm = layout.off_Jpu + scalar_n_dofs * scalar_n_dofs;
  layout.off_Jqlm = layout.off_Jplm + scalar_n_dofs * lm_n_dofs;
  layout.off_Jlmq = layout.off_Jqlm + vector_n_dofs * lm_n_dofs;
  layout.off_Jlmp = layout.off_Jlmq + lm_n_dofs * vector_n_dofs;
  layout.off_Jlms = layout.off_Jlmp + lm_n_dofs * scalar_n_dofs;
  layout.off_Jlmlm = layout.off_Jlms + lm_n_dofs * scalar_n_dofs;
  layout.off_Juu = layout.off_Jlmlm + lm_n_dofs * lm_n_dofs;
  layout.off_Julm = layout.off_Juu + scalar_n_dofs * scalar_n_dofs;
  layout.n_entries = layout.off_Julm + scalar_n_dofs * lm_n_dofs;
  return layout;
}

LIBMESH_DEVICE_INLINE Real
component_or_zero(const RealVector & v, const unsigned int component)
{
  switch (component)
  {
    case 0:
      return v(0);
    case 1:
#if LIBMESH_DIM > 1
      return v(1);
#else
      return Real(0);
#endif
    case 2:
#if LIBMESH_DIM > 2
      return v(2);
#else
      return Real(0);
#endif
    default:
      return Real(0);
  }
}

LIBMESH_DEVICE_INLINE Real
component_from_xyz(const Real x, const Real y, const Real z, const unsigned int component)
{
  switch (component)
  {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    default:
      return Real(0);
  }
}

LIBMESH_DEVICE_INLINE RealVector
hdg_velocity_from_components(const Real active_value,
                             const Real other_value,
                             const unsigned int active_component)
{
  RealVector value;
  if (active_component == 0)
  {
    value(0) = active_value;
#if LIBMESH_DIM > 1
    value(1) = other_value;
#endif
  }
  else
  {
    value(0) = other_value;
#if LIBMESH_DIM > 1
    value(1) = active_value;
#endif
  }

  return value;
}

LIBMESH_DEVICE_INLINE Real
abs_real(const Real value)
{
  return value < Real(0) ? -value : value;
}

inline unsigned int
hdg_linear_condensed_internal_n_dofs(const HDGLinearBlockLayout layout)
{
  return layout.vector_n_dofs + layout.scalar_n_dofs;
}

inline unsigned int
hdg_linear_condensed_external_n_dofs(const HDGLinearBlockLayout layout)
{
  return layout.scalar_n_dofs + layout.lm_n_dofs;
}

inline unsigned int
hdg_linear_n_dofs(const HDGLinearBlockLayout layout)
{
  return layout.vector_n_dofs + 2 * layout.scalar_n_dofs + layout.lm_n_dofs;
}

template <typename DofStorage>
LIBMESH_DEVICE_INLINE Real
hdg_eval_scalar_dofs(const libMesh::FEShapeKey key,
                     const DofStorage & dofs,
                     const unsigned int n_dofs,
                     const Real xi,
                     const Real eta,
                     const Real zeta)
{
  Real value = Real(0);

  for (unsigned int j = 0; j != n_dofs; ++j)
    value += libMesh::Kokkos::shape(key, j, xi, eta, zeta) * dofs(j);

  return value;
}

template <typename DofStorage>
LIBMESH_DEVICE_INLINE RealVector
hdg_eval_vector_dofs(const libMesh::FEShapeKey key,
                     const DofStorage & dofs,
                     const unsigned int n_dofs,
                     const Real xi,
                     const Real eta,
                     const Real zeta)
{
  RealVector value;

  for (unsigned int j = 0; j != n_dofs; ++j)
    value += libMesh::Kokkos::vector_shape(key, j, xi, eta, zeta) * dofs(j);

  return value;
}

template <typename DofStorage>
LIBMESH_DEVICE_INLINE Real
hdg_eval_trace_dofs(const libMesh::FEShapeKey key,
                    const DofStorage & dofs,
                    const unsigned int n_dofs,
                    const unsigned int side_id,
                    const bool positive_edge_orientation,
                    const Real xi,
                    const Real eta,
                    const Real zeta)
{
  Real value = Real(0);

  for (unsigned int j = 0; j != n_dofs; ++j)
    value += libMesh::Kokkos::side_trace_shape(
                 key, j, side_id, positive_edge_orientation, xi, eta, zeta) *
             dofs(j);

  return value;
}

template <typename Storage>
struct HDGElementConstView
{
  Storage storage;
  unsigned int elem = 0;

  LIBMESH_DEVICE_INLINE Real operator()(const unsigned int i) const { return storage(elem, i); }
};

template <typename Storage>
struct HDGElementView
{
  Storage storage;
  unsigned int elem = 0;

  LIBMESH_DEVICE_INLINE typename Storage::reference_type operator()(const unsigned int i) const
  {
    return storage(elem, i);
  }
};

template <typename Storage>
struct HDGElementNodeView
{
  Storage storage;
  unsigned int elem = 0;

  LIBMESH_DEVICE_INLINE Real operator()(const unsigned int i, const unsigned int component) const
  {
    return storage(elem, i, component);
  }
};

template <typename BlockStorage>
LIBMESH_DEVICE_INLINE Real
hdg_linear_cc_entry(const HDGLinearBlockLayout layout,
                    const BlockStorage & blocks,
                    const unsigned int i,
                    const unsigned int j)
{
  if (i < layout.vector_n_dofs)
  {
    if (j < layout.vector_n_dofs)
      return blocks(layout.off_Jqq + i * layout.vector_n_dofs + j);

    return blocks(layout.off_Jqu + i * layout.scalar_n_dofs + (j - layout.vector_n_dofs));
  }

  if (j < layout.vector_n_dofs)
    return blocks(layout.off_Juq + (i - layout.vector_n_dofs) * layout.vector_n_dofs + j);

  return blocks(layout.off_Juu + (i - layout.vector_n_dofs) * layout.scalar_n_dofs +
                (j - layout.vector_n_dofs));
}

template <typename BlockStorage>
LIBMESH_DEVICE_INLINE Real
hdg_linear_ce_entry(const HDGLinearBlockLayout layout,
                    const BlockStorage & blocks,
                    const unsigned int i,
                    const unsigned int j)
{
  if (i < layout.vector_n_dofs)
  {
    if (j < layout.scalar_n_dofs)
      return Real(0);

    return blocks(layout.off_Jqlm + i * layout.lm_n_dofs + (j - layout.scalar_n_dofs));
  }

  if (j < layout.scalar_n_dofs)
    return blocks(layout.off_Jup + (i - layout.vector_n_dofs) * layout.scalar_n_dofs + j);

  return blocks(layout.off_Julm + (i - layout.vector_n_dofs) * layout.lm_n_dofs +
                (j - layout.scalar_n_dofs));
}

template <typename BlockStorage>
LIBMESH_DEVICE_INLINE Real
hdg_linear_ec_entry(const HDGLinearBlockLayout layout,
                    const BlockStorage & blocks,
                    const unsigned int i,
                    const unsigned int j)
{
  if (i < layout.scalar_n_dofs)
  {
    if (j < layout.vector_n_dofs)
      return Real(0);

    return blocks(layout.off_Jpu + i * layout.scalar_n_dofs + (j - layout.vector_n_dofs));
  }

  if (j < layout.vector_n_dofs)
    return blocks(layout.off_Jlmq + (i - layout.scalar_n_dofs) * layout.vector_n_dofs + j);

  return blocks(layout.off_Jlms + (i - layout.scalar_n_dofs) * layout.scalar_n_dofs +
                (j - layout.vector_n_dofs));
}

template <typename BlockStorage>
LIBMESH_DEVICE_INLINE Real
hdg_linear_ee_entry(const HDGLinearBlockLayout layout,
                    const BlockStorage & blocks,
                    const unsigned int i,
                    const unsigned int j)
{
  if (i < layout.scalar_n_dofs)
  {
    if (j < layout.scalar_n_dofs)
      return Real(0);

    return blocks(layout.off_Jplm + i * layout.lm_n_dofs + (j - layout.scalar_n_dofs));
  }

  if (j < layout.scalar_n_dofs)
    return blocks(layout.off_Jlmp + (i - layout.scalar_n_dofs) * layout.scalar_n_dofs + j);

  return blocks(layout.off_Jlmlm + (i - layout.scalar_n_dofs) * layout.lm_n_dofs +
                (j - layout.scalar_n_dofs));
}

template <typename BlockStorage, typename CondensedStorage>
LIBMESH_DEVICE_INLINE void
condense_hdg_linear_qu_blocks_device(const HDGLinearBlockLayout layout,
                                     const BlockStorage & blocks,
                                     const CondensedStorage & condensed,
                                     int & status)
{
  constexpr unsigned int max_internal = hdg_linear_condense_max_internal;
  constexpr unsigned int max_external = hdg_linear_condense_max_external;
  constexpr unsigned int max_cols = hdg_linear_condense_max_cols;

  const unsigned int n_internal = layout.vector_n_dofs + layout.scalar_n_dofs;
  const unsigned int n_external = layout.scalar_n_dofs + layout.lm_n_dofs;

  for (unsigned int i = 0; i != n_external * n_external; ++i)
    condensed(i) = Real(0);

  if (n_internal > max_internal || n_external > max_external)
  {
    status = 2;
    return;
  }

  Real aug[max_internal * max_cols];
  Real x[max_internal * max_external];

  for (unsigned int i = 0; i != n_internal; ++i)
  {
    for (unsigned int j = 0; j != n_internal; ++j)
      aug[i * max_cols + j] = hdg_linear_cc_entry(layout, blocks, i, j);

    for (unsigned int j = 0; j != n_external; ++j)
      aug[i * max_cols + n_internal + j] = hdg_linear_ce_entry(layout, blocks, i, j);
  }

  for (unsigned int k = 0; k != n_internal; ++k)
  {
    unsigned int pivot_row = k;
    Real pivot_abs = abs_real(aug[k * max_cols + k]);

    for (unsigned int r = k + 1; r != n_internal; ++r)
    {
      const Real candidate = abs_real(aug[r * max_cols + k]);
      if (candidate > pivot_abs)
      {
        pivot_abs = candidate;
        pivot_row = r;
      }
    }

    if (pivot_abs <= Real(1.e-20))
    {
      status = 1;
      return;
    }

    if (pivot_row != k)
      for (unsigned int c = 0; c != n_internal + n_external; ++c)
      {
        const Real tmp = aug[k * max_cols + c];
        aug[k * max_cols + c] = aug[pivot_row * max_cols + c];
        aug[pivot_row * max_cols + c] = tmp;
      }

    const Real pivot = aug[k * max_cols + k];
    for (unsigned int r = k + 1; r != n_internal; ++r)
    {
      const Real factor = aug[r * max_cols + k] / pivot;
      aug[r * max_cols + k] = Real(0);

      for (unsigned int c = k + 1; c != n_internal + n_external; ++c)
        aug[r * max_cols + c] -= factor * aug[k * max_cols + c];
    }
  }

  for (unsigned int rhs = 0; rhs != n_external; ++rhs)
    for (unsigned int rr = n_internal; rr != 0; --rr)
    {
      const unsigned int r = rr - 1;
      Real value = aug[r * max_cols + n_internal + rhs];

      for (unsigned int c = r + 1; c != n_internal; ++c)
        value -= aug[r * max_cols + c] * x[c * max_external + rhs];

      x[r * max_external + rhs] = value / aug[r * max_cols + r];
    }

  for (unsigned int i = 0; i != n_external; ++i)
    for (unsigned int j = 0; j != n_external; ++j)
    {
      Real value = hdg_linear_ee_entry(layout, blocks, i, j);

      for (unsigned int k = 0; k != n_internal; ++k)
        value -= hdg_linear_ec_entry(layout, blocks, i, k) * x[k * max_external + j];

      condensed(i * n_external + j) = value;
    }

  status = 0;
}

template <typename MemberType,
          typename BlockStorage,
          typename CondensedStorage,
          typename StatusStorage,
          typename ScratchStorage>
LIBMESH_DEVICE_INLINE void
condense_hdg_linear_qu_blocks_team(const MemberType & team,
                                   const HDGLinearBlockLayout layout,
                                   const BlockStorage & blocks,
                                   const CondensedStorage & condensed,
                                   const StatusStorage & status,
                                   const unsigned int elem,
                                   const ScratchStorage & aug,
                                   const ScratchStorage & x)
{
  constexpr unsigned int max_internal = hdg_linear_condense_max_internal;
  constexpr unsigned int max_external = hdg_linear_condense_max_external;
  constexpr unsigned int max_cols = hdg_linear_condense_max_cols;

  const unsigned int n_internal = layout.vector_n_dofs + layout.scalar_n_dofs;
  const unsigned int n_external = layout.scalar_n_dofs + layout.lm_n_dofs;
  const unsigned int n_cols = n_internal + n_external;

  ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_external * n_external),
                         [&](const int raw_i)
                         {
                           const unsigned int i = static_cast<unsigned int>(raw_i);
                           condensed(i) = Real(0);
                         });

  if (n_internal > max_internal || n_external > max_external)
  {
    if (team.team_rank() == 0)
      status(elem) = 2;
    return;
  }

  if (team.team_rank() == 0)
    status(elem) = 0;
  team.team_barrier();

  ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_internal * n_cols),
                         [&](const int raw_entry)
                         {
                           const unsigned int entry = static_cast<unsigned int>(raw_entry);
                           const unsigned int i = entry / n_cols;
                           const unsigned int j = entry - i * n_cols;
                           aug(i * max_cols + j) =
                               j < n_internal
                                   ? hdg_linear_cc_entry(layout, blocks, i, j)
                                   : hdg_linear_ce_entry(layout, blocks, i, j - n_internal);
                         });
  team.team_barrier();

  for (unsigned int k = 0; k != n_internal; ++k)
  {
    if (team.team_rank() == 0)
    {
      unsigned int pivot_row = k;
      Real pivot_abs = abs_real(aug(k * max_cols + k));

      for (unsigned int r = k + 1; r != n_internal; ++r)
      {
        const Real candidate = abs_real(aug(r * max_cols + k));
        if (candidate > pivot_abs)
        {
          pivot_abs = candidate;
          pivot_row = r;
        }
      }

      if (pivot_abs <= Real(1.e-20))
        status(elem) = 1;
      else if (pivot_row != k)
        for (unsigned int c = 0; c != n_cols; ++c)
        {
          const Real tmp = aug(k * max_cols + c);
          aug(k * max_cols + c) = aug(pivot_row * max_cols + c);
          aug(pivot_row * max_cols + c) = tmp;
        }
    }
    team.team_barrier();

    if (status(elem))
      return;

    const Real pivot = aug(k * max_cols + k);
    ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_internal - k - 1),
                           [&](const int raw_r)
                           {
                             const unsigned int r = k + 1 + static_cast<unsigned int>(raw_r);
                             const Real factor = aug(r * max_cols + k) / pivot;
                             aug(r * max_cols + k) = Real(0);

                             for (unsigned int c = k + 1; c != n_cols; ++c)
                               aug(r * max_cols + c) -= factor * aug(k * max_cols + c);
                           });
    team.team_barrier();
  }

  ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_external),
                         [&](const int raw_rhs)
                         {
                           const unsigned int rhs = static_cast<unsigned int>(raw_rhs);
                           for (unsigned int rr = n_internal; rr != 0; --rr)
                           {
                             const unsigned int r = rr - 1;
                             Real value = aug(r * max_cols + n_internal + rhs);

                             for (unsigned int c = r + 1; c != n_internal; ++c)
                               value -= aug(r * max_cols + c) * x(c * max_external + rhs);

                             x(r * max_external + rhs) = value / aug(r * max_cols + r);
                           }
                         });
  team.team_barrier();

  ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_external * n_external),
                         [&](const int raw_entry)
                         {
                           const unsigned int entry = static_cast<unsigned int>(raw_entry);
                           const unsigned int i = entry / n_external;
                           const unsigned int j = entry - i * n_external;
                           Real value = hdg_linear_ee_entry(layout, blocks, i, j);

                           for (unsigned int k = 0; k != n_internal; ++k)
                             value -= hdg_linear_ec_entry(layout, blocks, i, k) *
                                      x(k * max_external + j);

                           condensed(entry) = value;
                         });
}

template <typename BlockStorage, typename ResidualStorage, typename CondensedStorage>
LIBMESH_DEVICE_INLINE void
condense_hdg_linear_qu_residual_device(const HDGLinearBlockLayout layout,
                                       const BlockStorage & blocks,
                                       const ResidualStorage & residual,
                                       const CondensedStorage & condensed,
                                       int & status)
{
  constexpr unsigned int max_internal = hdg_linear_condense_max_internal;
  constexpr unsigned int max_external = hdg_linear_condense_max_external;
  constexpr unsigned int max_cols = max_internal + 1;

  const unsigned int n_internal = layout.vector_n_dofs + layout.scalar_n_dofs;
  const unsigned int n_external = layout.scalar_n_dofs + layout.lm_n_dofs;

  for (unsigned int i = 0; i != n_external; ++i)
    condensed(i) = Real(0);

  if (n_internal > max_internal || n_external > max_external)
  {
    status = 2;
    return;
  }

  Real aug[max_internal * max_cols];
  Real y[max_internal];

  for (unsigned int i = 0; i != n_internal; ++i)
  {
    for (unsigned int j = 0; j != n_internal; ++j)
      aug[i * max_cols + j] = hdg_linear_cc_entry(layout, blocks, i, j);

    aug[i * max_cols + n_internal] = residual(i);
  }

  for (unsigned int k = 0; k != n_internal; ++k)
  {
    unsigned int pivot_row = k;
    Real pivot_abs = abs_real(aug[k * max_cols + k]);

    for (unsigned int r = k + 1; r != n_internal; ++r)
    {
      const Real candidate = abs_real(aug[r * max_cols + k]);
      if (candidate > pivot_abs)
      {
        pivot_abs = candidate;
        pivot_row = r;
      }
    }

    if (pivot_abs <= Real(1.e-20))
    {
      status = 1;
      return;
    }

    if (pivot_row != k)
      for (unsigned int c = 0; c != n_internal + 1; ++c)
      {
        const Real tmp = aug[k * max_cols + c];
        aug[k * max_cols + c] = aug[pivot_row * max_cols + c];
        aug[pivot_row * max_cols + c] = tmp;
      }

    const Real pivot = aug[k * max_cols + k];
    for (unsigned int r = k + 1; r != n_internal; ++r)
    {
      const Real factor = aug[r * max_cols + k] / pivot;
      aug[r * max_cols + k] = Real(0);

      for (unsigned int c = k + 1; c != n_internal + 1; ++c)
        aug[r * max_cols + c] -= factor * aug[k * max_cols + c];
    }
  }

  for (unsigned int rr = n_internal; rr != 0; --rr)
  {
    const unsigned int r = rr - 1;
    Real value = aug[r * max_cols + n_internal];

    for (unsigned int c = r + 1; c != n_internal; ++c)
      value -= aug[r * max_cols + c] * y[c];

    y[r] = value / aug[r * max_cols + r];
  }

  for (unsigned int i = 0; i != n_external; ++i)
  {
    Real value = residual(n_internal + i);

    for (unsigned int k = 0; k != n_internal; ++k)
      value -= hdg_linear_ec_entry(layout, blocks, i, k) * y[k];

    condensed(i) = value;
  }

  status = 0;
}

template <typename MemberType,
          typename BlockStorage,
          typename ResidualStorage,
          typename CondensedStorage,
          typename StatusStorage,
          typename ScratchStorage>
LIBMESH_DEVICE_INLINE void
condense_hdg_linear_qu_residual_team(const MemberType & team,
                                     const HDGLinearBlockLayout layout,
                                     const BlockStorage & blocks,
                                     const ResidualStorage & residual,
                                     const CondensedStorage & condensed,
                                     const StatusStorage & status,
                                     const unsigned int elem,
                                     const ScratchStorage & aug,
                                     const ScratchStorage & y)
{
  constexpr unsigned int max_internal = hdg_linear_condense_max_internal;
  constexpr unsigned int max_external = hdg_linear_condense_max_external;
  constexpr unsigned int max_cols = hdg_linear_condense_max_cols;

  const unsigned int n_internal = layout.vector_n_dofs + layout.scalar_n_dofs;
  const unsigned int n_external = layout.scalar_n_dofs + layout.lm_n_dofs;
  const unsigned int n_cols = n_internal + 1;

  ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_external),
                         [&](const int raw_i)
                         {
                           const unsigned int i = static_cast<unsigned int>(raw_i);
                           condensed(i) = Real(0);
                         });

  if (n_internal > max_internal || n_external > max_external)
  {
    if (team.team_rank() == 0)
      status(elem) = 2;
    return;
  }

  if (team.team_rank() == 0)
    status(elem) = 0;
  team.team_barrier();

  ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_internal * n_cols),
                         [&](const int raw_entry)
                         {
                           const unsigned int entry = static_cast<unsigned int>(raw_entry);
                           const unsigned int i = entry / n_cols;
                           const unsigned int j = entry - i * n_cols;
                           aug(i * max_cols + j) = j < n_internal
                                                       ? hdg_linear_cc_entry(layout, blocks, i, j)
                                                       : residual(i);
                         });
  team.team_barrier();

  for (unsigned int k = 0; k != n_internal; ++k)
  {
    if (team.team_rank() == 0)
    {
      unsigned int pivot_row = k;
      Real pivot_abs = abs_real(aug(k * max_cols + k));

      for (unsigned int r = k + 1; r != n_internal; ++r)
      {
        const Real candidate = abs_real(aug(r * max_cols + k));
        if (candidate > pivot_abs)
        {
          pivot_abs = candidate;
          pivot_row = r;
        }
      }

      if (pivot_abs <= Real(1.e-20))
        status(elem) = 1;
      else if (pivot_row != k)
        for (unsigned int c = 0; c != n_cols; ++c)
        {
          const Real tmp = aug(k * max_cols + c);
          aug(k * max_cols + c) = aug(pivot_row * max_cols + c);
          aug(pivot_row * max_cols + c) = tmp;
        }
    }
    team.team_barrier();

    if (status(elem))
      return;

    const Real pivot = aug(k * max_cols + k);
    ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_internal - k - 1),
                           [&](const int raw_r)
                           {
                             const unsigned int r = k + 1 + static_cast<unsigned int>(raw_r);
                             const Real factor = aug(r * max_cols + k) / pivot;
                             aug(r * max_cols + k) = Real(0);

                             for (unsigned int c = k + 1; c != n_cols; ++c)
                               aug(r * max_cols + c) -= factor * aug(k * max_cols + c);
                           });
    team.team_barrier();
  }

  if (team.team_rank() == 0)
    for (unsigned int rr = n_internal; rr != 0; --rr)
    {
      const unsigned int r = rr - 1;
      Real value = aug(r * max_cols + n_internal);

      for (unsigned int c = r + 1; c != n_internal; ++c)
        value -= aug(r * max_cols + c) * y(c);

      y(r) = value / aug(r * max_cols + r);
    }
  team.team_barrier();

  ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_external),
                         [&](const int raw_i)
                         {
                           const unsigned int i = static_cast<unsigned int>(raw_i);
                           Real value = residual(n_internal + i);

                           for (unsigned int k = 0; k != n_internal; ++k)
                             value -= hdg_linear_ec_entry(layout, blocks, i, k) * y(k);

                           condensed(i) = value;
                         });
}

template <typename BlockStorage, typename CondensedStorage, typename StatusStorage>
void
condense_hdg_linear_qu_blocks(const HDGLinearBlockLayout layout,
                              const BlockStorage & blocks,
                              const CondensedStorage & condensed,
                              const StatusStorage & status,
                              const char * const kernel_name)
{
  ::Kokkos::parallel_for(
      kernel_name, ::Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
        int local_status = 0;
        condense_hdg_linear_qu_blocks_device(layout, blocks, condensed, local_status);
        status() = local_status;
      });
}

template <typename BlockStorage,
          typename ResidualStorage,
          typename CondensedStorage,
          typename StatusStorage>
void
condense_hdg_linear_qu_residual(const HDGLinearBlockLayout layout,
                                const BlockStorage & blocks,
                                const ResidualStorage & residual,
                                const CondensedStorage & condensed,
                                const StatusStorage & status,
                                const char * const kernel_name)
{
  ::Kokkos::parallel_for(
      kernel_name, ::Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
        int local_status = 0;
        condense_hdg_linear_qu_residual_device(layout, blocks, residual, condensed, local_status);
        status() = local_status;
      });
}

template <libMesh::ElemType ExactTopo,
          typename NodeStorage,
          typename QpStorage,
          typename BlockStorage>
void
assemble_hdg_linear_volume_blocks(const libMesh::FEShapeKey vector_key,
                                  const libMesh::FEShapeKey scalar_key,
                                  const HDGLinearBlockLayout layout,
                                  const NodeStorage & coords,
                                  const unsigned int n_geom_nodes,
                                  const QpStorage & xi,
                                  const QpStorage & eta,
                                  const QpStorage & zeta,
                                  const QpStorage & jxw,
                                  const unsigned int n_qpoints,
                                  const Real nu,
                                  const BlockStorage & blocks,
                                  const char * const kernel_name)
{
  ::Kokkos::parallel_for(
      kernel_name, ::Kokkos::RangePolicy<>(0, layout.off_Jplm), KOKKOS_LAMBDA(const int raw_idx) {
        const unsigned int idx = static_cast<unsigned int>(raw_idx);
        Real value = Real(0);

        if (idx < layout.off_Jqu)
        {
          const unsigned int local = idx - layout.off_Jqq;
          const unsigned int i = local / layout.vector_n_dofs;
          const unsigned int j = local % layout.vector_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
            value +=
                jxw(q) * (libMesh::Kokkos::vector_shape(vector_key, i, xi(q), eta(q), zeta(q)) *
                          libMesh::Kokkos::vector_shape(vector_key, j, xi(q), eta(q), zeta(q)));
        }
        else if (idx < layout.off_Juq)
        {
          const unsigned int local = idx - layout.off_Jqu;
          const unsigned int i = local / layout.scalar_n_dofs;
          const unsigned int j = local % layout.scalar_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const unsigned int scalar_i = i / 2;
            const unsigned int component = i - scalar_i * 2;
            const RealTensor J = libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                coords, n_geom_nodes, xi(q), eta(q), zeta(q));
            const RealVector grad_phys =
                libMesh::Kokkos::inverse<RealTensor>(J, 2) *
                libMesh::Kokkos::grad_shape(scalar_key, scalar_i, xi(q), eta(q), zeta(q));

            value += jxw(q) * component_or_zero(grad_phys, component) *
                     libMesh::Kokkos::shape(scalar_key, j, xi(q), eta(q), zeta(q));
          }
        }
        else if (idx < layout.off_Jup)
        {
          const unsigned int local = idx - layout.off_Juq;
          const unsigned int i = local / layout.vector_n_dofs;
          const unsigned int j = local % layout.vector_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealTensor J = libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                coords, n_geom_nodes, xi(q), eta(q), zeta(q));
            const RealVector grad_phys =
                libMesh::Kokkos::inverse<RealTensor>(J, 2) *
                libMesh::Kokkos::grad_shape(scalar_key, i, xi(q), eta(q), zeta(q));
            value +=
                jxw(q) * nu *
                (grad_phys * libMesh::Kokkos::vector_shape(vector_key, j, xi(q), eta(q), zeta(q)));
          }
        }
        else
        {
          const unsigned int local =
              idx < layout.off_Jpu ? idx - layout.off_Jup : idx - layout.off_Jpu;
          const unsigned int i = local / layout.scalar_n_dofs;
          const unsigned int j = local % layout.scalar_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealTensor J = libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                coords, n_geom_nodes, xi(q), eta(q), zeta(q));
            const RealVector grad_phys =
                libMesh::Kokkos::inverse<RealTensor>(J, 2) *
                libMesh::Kokkos::grad_shape(scalar_key, i, xi(q), eta(q), zeta(q));
            value -= jxw(q) * component_or_zero(grad_phys, 0) *
                     libMesh::Kokkos::shape(scalar_key, j, xi(q), eta(q), zeta(q));
          }
        }

        blocks(idx) = value;
      });
}

template <libMesh::ElemType ExactTopo,
          typename NodeStorage,
          typename QpStorage,
          typename DofStorage,
          typename ResidualStorage>
void
assemble_hdg_linear_volume_residual(const libMesh::FEShapeKey vector_key,
                                    const libMesh::FEShapeKey scalar_key,
                                    const HDGLinearBlockLayout layout,
                                    const NodeStorage & coords,
                                    const unsigned int n_geom_nodes,
                                    const QpStorage & xi,
                                    const QpStorage & eta,
                                    const QpStorage & zeta,
                                    const QpStorage & jxw,
                                    const unsigned int n_qpoints,
                                    const Real nu,
                                    const DofStorage & q_dofs,
                                    const DofStorage & u_dofs,
                                    const DofStorage & p_dofs,
                                    const ResidualStorage & residual,
                                    const char * const kernel_name)
{
  ::Kokkos::parallel_for(
      kernel_name,
      ::Kokkos::RangePolicy<>(0, layout.vector_n_dofs + 2 * layout.scalar_n_dofs),
      KOKKOS_LAMBDA(const int raw_idx) {
        const unsigned int idx = static_cast<unsigned int>(raw_idx);
        const unsigned int u0 = layout.vector_n_dofs;
        const unsigned int p0 = u0 + layout.scalar_n_dofs;
        Real value = Real(0);

        if (idx < u0)
        {
          const unsigned int i = idx;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const unsigned int scalar_i = i / 2;
            const unsigned int component = i - scalar_i * 2;
            const RealTensor J = libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                coords, n_geom_nodes, xi(q), eta(q), zeta(q));
            const RealVector grad_phys =
                libMesh::Kokkos::inverse<RealTensor>(J, 2) *
                libMesh::Kokkos::grad_shape(scalar_key, scalar_i, xi(q), eta(q), zeta(q));
            const RealVector q_value = hdg_eval_vector_dofs(
                vector_key, q_dofs, layout.vector_n_dofs, xi(q), eta(q), zeta(q));
            const Real u_value = hdg_eval_scalar_dofs(
                scalar_key, u_dofs, layout.scalar_n_dofs, xi(q), eta(q), zeta(q));

            value +=
                jxw(q) *
                ((libMesh::Kokkos::vector_shape(vector_key, i, xi(q), eta(q), zeta(q)) * q_value) +
                 component_or_zero(grad_phys, component) * u_value);
          }
        }
        else if (idx < p0)
        {
          const unsigned int i = idx - u0;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealTensor J = libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                coords, n_geom_nodes, xi(q), eta(q), zeta(q));
            const RealVector grad_phys =
                libMesh::Kokkos::inverse<RealTensor>(J, 2) *
                libMesh::Kokkos::grad_shape(scalar_key, i, xi(q), eta(q), zeta(q));
            RealVector stress =
                hdg_eval_vector_dofs(
                    vector_key, q_dofs, layout.vector_n_dofs, xi(q), eta(q), zeta(q)) *
                nu;
            stress(0) -= hdg_eval_scalar_dofs(
                scalar_key, p_dofs, layout.scalar_n_dofs, xi(q), eta(q), zeta(q));

            value += jxw(q) * (grad_phys * stress);
          }
        }
        else
        {
          const unsigned int i = idx - p0;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealTensor J = libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                coords, n_geom_nodes, xi(q), eta(q), zeta(q));
            const RealVector grad_phys =
                libMesh::Kokkos::inverse<RealTensor>(J, 2) *
                libMesh::Kokkos::grad_shape(scalar_key, i, xi(q), eta(q), zeta(q));
            const Real u_value = hdg_eval_scalar_dofs(
                scalar_key, u_dofs, layout.scalar_n_dofs, xi(q), eta(q), zeta(q));

            value -= jxw(q) * component_or_zero(grad_phys, 0) * u_value;
          }
        }

        residual(idx) += value;
      });
}

template <typename QpStorage, typename BlockStorage>
void
add_hdg_linear_side_blocks(const libMesh::FEShapeKey vector_key,
                           const libMesh::FEShapeKey scalar_key,
                           const libMesh::FEShapeKey lm_key,
                           const HDGLinearBlockLayout layout,
                           const unsigned int side_id,
                           const bool positive_edge_orientation,
                           const QpStorage & parent_xi,
                           const QpStorage & parent_eta,
                           const QpStorage & parent_zeta,
                           const QpStorage & side_jxw,
                           const QpStorage & normal_x,
                           const QpStorage & normal_y,
                           const QpStorage & normal_z,
                           const unsigned int n_qpoints,
                           const Real nu,
                           const Real tau,
                           const BlockStorage & blocks,
                           const char * const kernel_name)
{
  ::Kokkos::parallel_for(
      kernel_name,
      ::Kokkos::RangePolicy<>(0, layout.n_entries - layout.off_Juq),
      KOKKOS_LAMBDA(const int raw_face_idx) {
        const unsigned int idx = static_cast<unsigned int>(raw_face_idx) + layout.off_Juq;
        Real value = Real(0);

        if (idx < layout.off_Jup)
        {
          const unsigned int local = idx - layout.off_Juq;
          const unsigned int i = local / layout.vector_n_dofs;
          const unsigned int j = local % layout.vector_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealVector normal =
                libMesh::Kokkos::make_vector(normal_x(q), normal_y(q), normal_z(q));
            value -=
                side_jxw(q) * nu *
                libMesh::Kokkos::shape(scalar_key, i, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                (libMesh::Kokkos::vector_shape(
                     vector_key, j, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                 normal);
          }
        }
        else if (idx < layout.off_Jpu)
        {
          const unsigned int local = idx - layout.off_Jup;
          const unsigned int i = local / layout.scalar_n_dofs;
          const unsigned int j = local % layout.scalar_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
            value +=
                side_jxw(q) *
                libMesh::Kokkos::shape(scalar_key, i, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                libMesh::Kokkos::shape(scalar_key, j, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                normal_x(q);
        }
        else if (idx < layout.off_Jplm)
        {
          // Jpu has no side contribution in this linear HDG component block.
        }
        else if (idx < layout.off_Jqlm)
        {
          const unsigned int local = idx - layout.off_Jplm;
          const unsigned int i = local / layout.lm_n_dofs;
          const unsigned int j = local % layout.lm_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
            value +=
                side_jxw(q) *
                libMesh::Kokkos::shape(scalar_key, i, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                libMesh::Kokkos::side_trace_shape(lm_key,
                                                  j,
                                                  side_id,
                                                  positive_edge_orientation,
                                                  parent_xi(q),
                                                  parent_eta(q),
                                                  parent_zeta(q)) *
                normal_x(q);
        }
        else if (idx < layout.off_Jlmq)
        {
          const unsigned int local = idx - layout.off_Jqlm;
          const unsigned int i = local / layout.lm_n_dofs;
          const unsigned int j = local % layout.lm_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealVector normal =
                libMesh::Kokkos::make_vector(normal_x(q), normal_y(q), normal_z(q));
            value -= side_jxw(q) *
                     (libMesh::Kokkos::vector_shape(
                          vector_key, i, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                      normal) *
                     libMesh::Kokkos::side_trace_shape(lm_key,
                                                       j,
                                                       side_id,
                                                       positive_edge_orientation,
                                                       parent_xi(q),
                                                       parent_eta(q),
                                                       parent_zeta(q));
          }
        }
        else if (idx < layout.off_Jlmp)
        {
          const unsigned int local = idx - layout.off_Jlmq;
          const unsigned int i = local / layout.vector_n_dofs;
          const unsigned int j = local % layout.vector_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealVector normal =
                libMesh::Kokkos::make_vector(normal_x(q), normal_y(q), normal_z(q));
            value -= side_jxw(q) * nu *
                     libMesh::Kokkos::side_trace_shape(lm_key,
                                                       i,
                                                       side_id,
                                                       positive_edge_orientation,
                                                       parent_xi(q),
                                                       parent_eta(q),
                                                       parent_zeta(q)) *
                     (libMesh::Kokkos::vector_shape(
                          vector_key, j, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                      normal);
          }
        }
        else if (idx < layout.off_Jlms)
        {
          const unsigned int local = idx - layout.off_Jlmp;
          const unsigned int i = local / layout.scalar_n_dofs;
          const unsigned int j = local % layout.scalar_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
            value +=
                side_jxw(q) *
                libMesh::Kokkos::side_trace_shape(lm_key,
                                                  i,
                                                  side_id,
                                                  positive_edge_orientation,
                                                  parent_xi(q),
                                                  parent_eta(q),
                                                  parent_zeta(q)) *
                libMesh::Kokkos::shape(scalar_key, j, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                normal_x(q);
        }
        else if (idx < layout.off_Jlmlm)
        {
          const unsigned int local = idx - layout.off_Jlms;
          const unsigned int i = local / layout.scalar_n_dofs;
          const unsigned int j = local % layout.scalar_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const Real normal_norm2 =
                normal_x(q) * normal_x(q) + normal_y(q) * normal_y(q) + normal_z(q) * normal_z(q);
            value +=
                side_jxw(q) *
                libMesh::Kokkos::side_trace_shape(lm_key,
                                                  i,
                                                  side_id,
                                                  positive_edge_orientation,
                                                  parent_xi(q),
                                                  parent_eta(q),
                                                  parent_zeta(q)) *
                tau *
                libMesh::Kokkos::shape(scalar_key, j, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                normal_norm2;
          }
        }
        else if (idx < layout.off_Juu)
        {
          const unsigned int local = idx - layout.off_Jlmlm;
          const unsigned int i = local / layout.lm_n_dofs;
          const unsigned int j = local % layout.lm_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const Real normal_norm2 =
                normal_x(q) * normal_x(q) + normal_y(q) * normal_y(q) + normal_z(q) * normal_z(q);
            value -= side_jxw(q) *
                     libMesh::Kokkos::side_trace_shape(lm_key,
                                                       i,
                                                       side_id,
                                                       positive_edge_orientation,
                                                       parent_xi(q),
                                                       parent_eta(q),
                                                       parent_zeta(q)) *
                     tau *
                     libMesh::Kokkos::side_trace_shape(lm_key,
                                                       j,
                                                       side_id,
                                                       positive_edge_orientation,
                                                       parent_xi(q),
                                                       parent_eta(q),
                                                       parent_zeta(q)) *
                     normal_norm2;
          }
        }
        else if (idx < layout.off_Julm)
        {
          const unsigned int local = idx - layout.off_Juu;
          const unsigned int i = local / layout.scalar_n_dofs;
          const unsigned int j = local % layout.scalar_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const Real normal_norm2 =
                normal_x(q) * normal_x(q) + normal_y(q) * normal_y(q) + normal_z(q) * normal_z(q);
            value +=
                side_jxw(q) *
                libMesh::Kokkos::shape(scalar_key, i, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                tau *
                libMesh::Kokkos::shape(scalar_key, j, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                normal_norm2;
          }
        }
        else
        {
          const unsigned int local = idx - layout.off_Julm;
          const unsigned int i = local / layout.lm_n_dofs;
          const unsigned int j = local % layout.lm_n_dofs;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const Real normal_norm2 =
                normal_x(q) * normal_x(q) + normal_y(q) * normal_y(q) + normal_z(q) * normal_z(q);
            value -=
                side_jxw(q) *
                libMesh::Kokkos::shape(scalar_key, i, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                tau *
                libMesh::Kokkos::side_trace_shape(lm_key,
                                                  j,
                                                  side_id,
                                                  positive_edge_orientation,
                                                  parent_xi(q),
                                                  parent_eta(q),
                                                  parent_zeta(q)) *
                normal_norm2;
          }
        }

        blocks(idx) += value;
      });
}

template <typename QpStorage, typename DofStorage, typename ResidualStorage>
void
add_hdg_linear_side_residual(const libMesh::FEShapeKey vector_key,
                             const libMesh::FEShapeKey scalar_key,
                             const libMesh::FEShapeKey lm_key,
                             const HDGLinearBlockLayout layout,
                             const unsigned int side_id,
                             const bool positive_edge_orientation,
                             const QpStorage & parent_xi,
                             const QpStorage & parent_eta,
                             const QpStorage & parent_zeta,
                             const QpStorage & side_jxw,
                             const QpStorage & normal_x,
                             const QpStorage & normal_y,
                             const QpStorage & normal_z,
                             const unsigned int n_qpoints,
                             const Real nu,
                             const Real tau,
                             const DofStorage & q_dofs,
                             const DofStorage & u_dofs,
                             const DofStorage & p_dofs,
                             const DofStorage & lm_dofs,
                             const ResidualStorage & residual,
                             const char * const kernel_name)
{
  ::Kokkos::parallel_for(
      kernel_name,
      ::Kokkos::RangePolicy<>(0, hdg_linear_n_dofs(layout)),
      KOKKOS_LAMBDA(const int raw_idx) {
        const unsigned int idx = static_cast<unsigned int>(raw_idx);
        const unsigned int u0 = layout.vector_n_dofs;
        const unsigned int p0 = u0 + layout.scalar_n_dofs;
        const unsigned int lm0 = p0 + layout.scalar_n_dofs;
        Real value = Real(0);

        if (idx < u0)
        {
          const unsigned int i = idx;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealVector normal =
                libMesh::Kokkos::make_vector(normal_x(q), normal_y(q), normal_z(q));
            const Real lm_value = hdg_eval_trace_dofs(lm_key,
                                                      lm_dofs,
                                                      layout.lm_n_dofs,
                                                      side_id,
                                                      positive_edge_orientation,
                                                      parent_xi(q),
                                                      parent_eta(q),
                                                      parent_zeta(q));
            value -= side_jxw(q) *
                     (libMesh::Kokkos::vector_shape(
                          vector_key, i, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                      normal) *
                     lm_value;
          }
        }
        else if (idx < p0)
        {
          const unsigned int i = idx - u0;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealVector normal =
                libMesh::Kokkos::make_vector(normal_x(q), normal_y(q), normal_z(q));
            const Real normal_norm2 = normal * normal;
            const Real q_normal = hdg_eval_vector_dofs(vector_key,
                                                       q_dofs,
                                                       layout.vector_n_dofs,
                                                       parent_xi(q),
                                                       parent_eta(q),
                                                       parent_zeta(q)) *
                                  normal;
            const Real u_value = hdg_eval_scalar_dofs(scalar_key,
                                                      u_dofs,
                                                      layout.scalar_n_dofs,
                                                      parent_xi(q),
                                                      parent_eta(q),
                                                      parent_zeta(q));
            const Real p_value = hdg_eval_scalar_dofs(scalar_key,
                                                      p_dofs,
                                                      layout.scalar_n_dofs,
                                                      parent_xi(q),
                                                      parent_eta(q),
                                                      parent_zeta(q));
            const Real lm_value = hdg_eval_trace_dofs(lm_key,
                                                      lm_dofs,
                                                      layout.lm_n_dofs,
                                                      side_id,
                                                      positive_edge_orientation,
                                                      parent_xi(q),
                                                      parent_eta(q),
                                                      parent_zeta(q));
            const Real phi_i =
                libMesh::Kokkos::shape(scalar_key, i, parent_xi(q), parent_eta(q), parent_zeta(q));

            value += side_jxw(q) * phi_i *
                     (-nu * q_normal + p_value * normal_x(q) +
                      tau * (u_value - lm_value) * normal_norm2);
          }
        }
        else if (idx < lm0)
        {
          const unsigned int i = idx - p0;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const Real lm_value = hdg_eval_trace_dofs(lm_key,
                                                      lm_dofs,
                                                      layout.lm_n_dofs,
                                                      side_id,
                                                      positive_edge_orientation,
                                                      parent_xi(q),
                                                      parent_eta(q),
                                                      parent_zeta(q));
            value +=
                side_jxw(q) *
                libMesh::Kokkos::shape(scalar_key, i, parent_xi(q), parent_eta(q), parent_zeta(q)) *
                lm_value * normal_x(q);
          }
        }
        else
        {
          const unsigned int i = idx - lm0;
          for (unsigned int q = 0; q != n_qpoints; ++q)
          {
            const RealVector normal =
                libMesh::Kokkos::make_vector(normal_x(q), normal_y(q), normal_z(q));
            const Real normal_norm2 = normal * normal;
            const Real q_normal = hdg_eval_vector_dofs(vector_key,
                                                       q_dofs,
                                                       layout.vector_n_dofs,
                                                       parent_xi(q),
                                                       parent_eta(q),
                                                       parent_zeta(q)) *
                                  normal;
            const Real u_value = hdg_eval_scalar_dofs(scalar_key,
                                                      u_dofs,
                                                      layout.scalar_n_dofs,
                                                      parent_xi(q),
                                                      parent_eta(q),
                                                      parent_zeta(q));
            const Real p_value = hdg_eval_scalar_dofs(scalar_key,
                                                      p_dofs,
                                                      layout.scalar_n_dofs,
                                                      parent_xi(q),
                                                      parent_eta(q),
                                                      parent_zeta(q));
            const Real lm_value = hdg_eval_trace_dofs(lm_key,
                                                      lm_dofs,
                                                      layout.lm_n_dofs,
                                                      side_id,
                                                      positive_edge_orientation,
                                                      parent_xi(q),
                                                      parent_eta(q),
                                                      parent_zeta(q));
            const Real phi_i = libMesh::Kokkos::side_trace_shape(lm_key,
                                                                 i,
                                                                 side_id,
                                                                 positive_edge_orientation,
                                                                 parent_xi(q),
                                                                 parent_eta(q),
                                                                 parent_zeta(q));

            value += side_jxw(q) * phi_i *
                     (-nu * q_normal + p_value * normal_x(q) +
                      tau * (u_value - lm_value) * normal_norm2);
          }
        }

        residual(idx) += value;
      });
}

template <libMesh::ElemType ExactTopo,
          typename NodeStorage,
          typename VolumeQpStorage,
          typename VolumeForceStorage,
          typename SideQpStorage,
          typename OrientationStorage,
          typename SideFlagStorage,
          typename DirichletQpStorage,
          typename DofStorage,
          typename BlockStorage,
          typename ResidualStorage,
          typename CondensedStorage,
          typename StatusStorage>
void
assemble_hdg_linear_batch_boundary(const libMesh::FEShapeKey vector_key,
                                   const libMesh::FEShapeKey scalar_key,
                                   const libMesh::FEShapeKey lm_key,
                                   const HDGLinearBlockLayout layout,
                                   const NodeStorage & coords,
                                   const unsigned int n_elements,
                                   const unsigned int n_geom_nodes,
                                   const VolumeQpStorage & xi,
                                   const VolumeQpStorage & eta,
                                   const VolumeQpStorage & zeta,
                                   const VolumeQpStorage & jxw,
                                   const VolumeForceStorage & scalar_forcing,
                                   const VolumeForceStorage & pressure_forcing,
                                   const unsigned int n_qpoints,
                                   const SideQpStorage & parent_xi,
                                   const SideQpStorage & parent_eta,
                                   const SideQpStorage & parent_zeta,
                                   const SideQpStorage & side_jxw,
                                   const SideQpStorage & normal_x,
                                   const SideQpStorage & normal_y,
                                   const SideQpStorage & normal_z,
                                   const OrientationStorage & positive_edge_orientations,
                                   const SideFlagStorage & side_types,
                                   const DirichletQpStorage & dirichlet_component,
                                   const DirichletQpStorage & dirichlet_vdotn,
                                   const unsigned int n_sides,
                                   const unsigned int n_side_qpoints,
                                   const Real nu,
                                   const Real tau,
                                   const bool include_convection,
                                   const DofStorage & q_dofs,
                                   const DofStorage & u_dofs,
                                   const DofStorage & p_dofs,
                                   const DofStorage & lm_dofs,
                                   const DofStorage & other_u_dofs,
                                   const DofStorage & other_lm_dofs,
                                   const BlockStorage & blocks,
                                   const ResidualStorage & residual,
                                   const CondensedStorage & condensed,
                                   const CondensedStorage & condensed_residual,
                                   const StatusStorage & condense_status,
                                   const StatusStorage & residual_status,
                                   const char * const kernel_name,
                                   const unsigned int velocity_component = 0,
                                   const bool condense_outputs = true,
                                   const unsigned int requested_team_size = 0)
{
  using ExecutionSpace = typename BlockStorage::execution_space;
  using TeamPolicy = ::Kokkos::TeamPolicy<ExecutionSpace>;
  using MemberType = typename TeamPolicy::member_type;
  using ScratchSpace = typename MemberType::scratch_memory_space;
  using ScratchRealView =
      ::Kokkos::View<Real *, ScratchSpace, ::Kokkos::MemoryTraits<::Kokkos::Unmanaged>>;

  const std::size_t volume_scalar_phi_size = n_qpoints * layout.scalar_n_dofs;
  const std::size_t volume_grad_phys_size = n_qpoints * layout.scalar_n_dofs * LIBMESH_DIM;
  const std::size_t volume_vector_phi_size = n_qpoints * layout.vector_n_dofs * LIBMESH_DIM;
  const std::size_t volume_q_value_size = n_qpoints * LIBMESH_DIM;
  const std::size_t volume_scalar_values_size = 3 * n_qpoints;
  const std::size_t volume_scratch_size = volume_scalar_phi_size + volume_grad_phys_size +
                                          volume_vector_phi_size + volume_q_value_size +
                                          volume_scalar_values_size;
  const std::size_t face_scalar_phi_size = n_side_qpoints * layout.scalar_n_dofs;
  const std::size_t face_vector_phi_size = n_side_qpoints * layout.vector_n_dofs * LIBMESH_DIM;
  const std::size_t face_trace_phi_size = n_side_qpoints * layout.lm_n_dofs;
  const std::size_t face_q_value_size = n_side_qpoints * LIBMESH_DIM;
  const std::size_t face_normals_size = n_side_qpoints * LIBMESH_DIM;
  const std::size_t face_scalar_values_size = 4 * n_side_qpoints;
  const std::size_t face_scratch_size = face_scalar_phi_size + face_vector_phi_size +
                                        face_trace_phi_size + face_q_value_size +
                                        face_normals_size + face_scalar_values_size;
  const std::size_t scratch_bytes = ScratchRealView::shmem_size(hdg_linear_condense_aug_size) +
                                    ScratchRealView::shmem_size(hdg_linear_condense_aux_size) +
                                    ScratchRealView::shmem_size(volume_scratch_size) +
                                    ScratchRealView::shmem_size(face_scratch_size);

  TeamPolicy policy = requested_team_size ?
                          TeamPolicy(static_cast<int>(n_elements),
                                     static_cast<int>(requested_team_size)) :
                          TeamPolicy(static_cast<int>(n_elements), ::Kokkos::AUTO());
  policy = policy.set_scratch_size(0, ::Kokkos::PerTeam(scratch_bytes));

  ::Kokkos::parallel_for(
      kernel_name, policy, KOKKOS_LAMBDA(const MemberType & team) {
        const unsigned int elem = static_cast<unsigned int>(team.league_rank());
        const HDGElementNodeView<NodeStorage> elem_coords{coords, elem};
        const HDGElementConstView<DofStorage> elem_q_dofs{q_dofs, elem};
        const HDGElementConstView<DofStorage> elem_u_dofs{u_dofs, elem};
        const HDGElementConstView<DofStorage> elem_p_dofs{p_dofs, elem};
        const HDGElementConstView<DofStorage> elem_lm_dofs{lm_dofs, elem};
        const HDGElementConstView<DofStorage> elem_other_u_dofs{other_u_dofs, elem};
        const HDGElementConstView<DofStorage> elem_other_lm_dofs{other_lm_dofs, elem};
        const HDGElementView<BlockStorage> elem_blocks{blocks, elem};
        const HDGElementView<ResidualStorage> elem_residual{residual, elem};
        const HDGElementView<CondensedStorage> elem_condensed{condensed, elem};
        const HDGElementView<CondensedStorage> elem_condensed_residual{condensed_residual, elem};
        ScratchRealView condense_aug(team.team_scratch(0), hdg_linear_condense_aug_size);
        ScratchRealView condense_aux(team.team_scratch(0), hdg_linear_condense_aux_size);
        ScratchRealView volume_scratch(team.team_scratch(0), volume_scratch_size);
        ScratchRealView face_scratch(team.team_scratch(0), face_scratch_size);

        const unsigned int q0 = 0;
        const unsigned int u0 = q0 + layout.vector_n_dofs;
        const unsigned int p0 = u0 + layout.scalar_n_dofs;
        const unsigned int lm0 = p0 + layout.scalar_n_dofs;
        const unsigned int residual_n_dofs = hdg_linear_n_dofs(layout);
        const unsigned int volume_scalar_phi_offset = 0;
        const unsigned int volume_grad_phys_offset =
            volume_scalar_phi_offset + volume_scalar_phi_size;
        const unsigned int volume_vector_phi_offset =
            volume_grad_phys_offset + volume_grad_phys_size;
        const unsigned int volume_q_value_offset =
            volume_vector_phi_offset + volume_vector_phi_size;
        const unsigned int volume_u_value_offset = volume_q_value_offset + volume_q_value_size;
        const unsigned int volume_p_value_offset = volume_u_value_offset + n_qpoints;
        const unsigned int volume_other_u_value_offset = volume_p_value_offset + n_qpoints;

        auto scalar_phi_at = [&](const unsigned int q, const unsigned int i) -> Real &
        { return volume_scratch(volume_scalar_phi_offset + q * layout.scalar_n_dofs + i); };
        auto grad_phys_at = [&](const unsigned int q,
                                const unsigned int i,
                                const unsigned int d) -> Real &
        {
          return volume_scratch(volume_grad_phys_offset +
                                (q * layout.scalar_n_dofs + i) * LIBMESH_DIM + d);
        };
        auto vector_phi_at = [&](const unsigned int q,
                                 const unsigned int i,
                                 const unsigned int d) -> Real &
        {
          return volume_scratch(volume_vector_phi_offset +
                                (q * layout.vector_n_dofs + i) * LIBMESH_DIM + d);
        };
        auto q_value_at = [&](const unsigned int q, const unsigned int d) -> Real &
        { return volume_scratch(volume_q_value_offset + q * LIBMESH_DIM + d); };
        auto u_value_at = [&](const unsigned int q) -> Real &
        { return volume_scratch(volume_u_value_offset + q); };
        auto p_value_at = [&](const unsigned int q) -> Real &
        { return volume_scratch(volume_p_value_offset + q); };
        auto other_u_value_at = [&](const unsigned int q) -> Real &
        { return volume_scratch(volume_other_u_value_offset + q); };

        const unsigned int face_scalar_phi_offset = 0;
        const unsigned int face_vector_phi_offset = face_scalar_phi_offset + face_scalar_phi_size;
        const unsigned int face_trace_phi_offset = face_vector_phi_offset + face_vector_phi_size;
        const unsigned int face_q_value_offset = face_trace_phi_offset + face_trace_phi_size;
        const unsigned int face_normals_offset = face_q_value_offset + face_q_value_size;
        const unsigned int face_u_value_offset = face_normals_offset + face_normals_size;
        const unsigned int face_p_value_offset = face_u_value_offset + n_side_qpoints;
        const unsigned int face_lm_value_offset = face_p_value_offset + n_side_qpoints;
        const unsigned int face_other_lm_value_offset = face_lm_value_offset + n_side_qpoints;

        auto face_scalar_phi_at = [&](const unsigned int q, const unsigned int i) -> Real &
        { return face_scratch(face_scalar_phi_offset + q * layout.scalar_n_dofs + i); };
        auto face_vector_phi_at = [&](const unsigned int q,
                                      const unsigned int i,
                                      const unsigned int d) -> Real &
        {
          return face_scratch(face_vector_phi_offset +
                              (q * layout.vector_n_dofs + i) * LIBMESH_DIM + d);
        };
        auto face_trace_phi_at = [&](const unsigned int q, const unsigned int i) -> Real &
        { return face_scratch(face_trace_phi_offset + q * layout.lm_n_dofs + i); };
        auto face_q_value_at = [&](const unsigned int q, const unsigned int d) -> Real &
        { return face_scratch(face_q_value_offset + q * LIBMESH_DIM + d); };
        auto face_normal_at = [&](const unsigned int q, const unsigned int d) -> Real &
        { return face_scratch(face_normals_offset + q * LIBMESH_DIM + d); };
        auto face_u_value_at = [&](const unsigned int q) -> Real &
        { return face_scratch(face_u_value_offset + q); };
        auto face_p_value_at = [&](const unsigned int q) -> Real &
        { return face_scratch(face_p_value_offset + q); };
        auto face_lm_value_at = [&](const unsigned int q) -> Real &
        { return face_scratch(face_lm_value_offset + q); };
        auto face_other_lm_value_at = [&](const unsigned int q) -> Real &
        { return face_scratch(face_other_lm_value_offset + q); };

        ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, layout.n_entries),
                               [&](const int raw_i)
                               {
                                 const unsigned int i = static_cast<unsigned int>(raw_i);
                                 elem_blocks(i) = Real(0);
                               });
        ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, residual_n_dofs),
                               [&](const int raw_i)
                               {
                                 const unsigned int i = static_cast<unsigned int>(raw_i);
                                 elem_residual(i) = Real(0);
                               });
        team.team_barrier();

        ::Kokkos::parallel_for(
            ::Kokkos::TeamThreadRange(team, n_qpoints),
            [&](const int raw_q)
            {
              const unsigned int q = static_cast<unsigned int>(raw_q);
              const Real xiq = xi(q);
              const Real etaq = eta(q);
              const Real zetaq = zeta(q);
              const RealTensor J =
                  libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                      elem_coords, n_geom_nodes, xiq, etaq, zetaq);
              const RealTensor Jinv = libMesh::Kokkos::inverse<RealTensor>(J, 2);

              for (unsigned int i = 0; i != layout.scalar_n_dofs; ++i)
              {
                scalar_phi_at(q, i) = libMesh::Kokkos::shape(scalar_key, i, xiq, etaq, zetaq);
                const RealVector grad_phys =
                    Jinv * libMesh::Kokkos::grad_shape(scalar_key, i, xiq, etaq, zetaq);
                for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  grad_phys_at(q, i, d) = component_or_zero(grad_phys, d);
              }

              for (unsigned int i = 0; i != layout.vector_n_dofs; ++i)
              {
                const RealVector phi =
                    libMesh::Kokkos::vector_shape(vector_key, i, xiq, etaq, zetaq);
                for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  vector_phi_at(q, i, d) = component_or_zero(phi, d);
              }

              for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                q_value_at(q, d) = Real(0);
              u_value_at(q) = Real(0);
              p_value_at(q) = Real(0);
              other_u_value_at(q) = Real(0);

              for (unsigned int i = 0; i != layout.vector_n_dofs; ++i)
                for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  q_value_at(q, d) += vector_phi_at(q, i, d) * elem_q_dofs(i);

              for (unsigned int i = 0; i != layout.scalar_n_dofs; ++i)
              {
                const Real phi = scalar_phi_at(q, i);
                u_value_at(q) += phi * elem_u_dofs(i);
                p_value_at(q) += phi * elem_p_dofs(i);
                other_u_value_at(q) += phi * elem_other_u_dofs(i);
              }
            });
        team.team_barrier();

        ::Kokkos::parallel_for(
            ::Kokkos::TeamThreadRange(team, layout.off_Jplm),
            [&](const int raw_idx)
            {
              const unsigned int idx = static_cast<unsigned int>(raw_idx);
              Real value = Real(0);

              if (idx < layout.off_Jqu)
              {
                const unsigned int local = idx - layout.off_Jqq;
                const unsigned int i = local / layout.vector_n_dofs;
                const unsigned int j = local % layout.vector_n_dofs;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  Real dot = Real(0);
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    dot += vector_phi_at(q, i, d) * vector_phi_at(q, j, d);
                  value += jxw(q) * dot;
                }
              }
              else if (idx < layout.off_Juq)
              {
                const unsigned int local = idx - layout.off_Jqu;
                const unsigned int i = local / layout.scalar_n_dofs;
                const unsigned int j = local % layout.scalar_n_dofs;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  const unsigned int scalar_i = i / 2;
                  const unsigned int component = i - scalar_i * 2;
                  value += jxw(q) * grad_phys_at(q, scalar_i, component) *
                           scalar_phi_at(q, j);
                }
              }
              else if (idx < layout.off_Jup)
              {
                const unsigned int local = idx - layout.off_Juq;
                const unsigned int i = local / layout.vector_n_dofs;
                const unsigned int j = local % layout.vector_n_dofs;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  Real dot = Real(0);
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    dot += grad_phys_at(q, i, d) * vector_phi_at(q, j, d);
                  value += jxw(q) * nu * dot;
                }
              }
              else
              {
                const unsigned int local =
                    idx < layout.off_Jpu ? idx - layout.off_Jup : idx - layout.off_Jpu;
                const unsigned int i = local / layout.scalar_n_dofs;
                const unsigned int j = local % layout.scalar_n_dofs;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                  value -= jxw(q) * grad_phys_at(q, i, velocity_component) *
                           scalar_phi_at(q, j);
              }

              elem_blocks(idx) += value;
            });
        team.team_barrier();

        ::Kokkos::parallel_for(
            ::Kokkos::TeamThreadRange(team, layout.vector_n_dofs + 2 * layout.scalar_n_dofs),
            [&](const int raw_idx)
            {
              const unsigned int idx = static_cast<unsigned int>(raw_idx);
              Real value = Real(0);

              if (idx < u0)
              {
                const unsigned int i = idx;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  const unsigned int scalar_i = i / 2;
                  const unsigned int component = i - scalar_i * 2;
                  Real phi_dot_q = Real(0);
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    phi_dot_q += vector_phi_at(q, i, d) * q_value_at(q, d);

                  value += jxw(q) *
                           (phi_dot_q + grad_phys_at(q, scalar_i, component) * u_value_at(q));
                }
              }
              else if (idx < p0)
              {
                const unsigned int i = idx - u0;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  Real grad_dot_stress = Real(0);
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  {
                    Real stress_component = nu * q_value_at(q, d);
                    if (d == velocity_component)
                      stress_component -= p_value_at(q);
                    grad_dot_stress += grad_phys_at(q, i, d) * stress_component;
                  }
                  const Real active_u = u_value_at(q);
                  const Real other_u = other_u_value_at(q);
                  const RealVector convective_flux =
                      include_convection
                          ? hdg_velocity_from_components(active_u, other_u, velocity_component) *
                                active_u
                          : RealVector();
                  Real grad_dot_convective_flux = Real(0);
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    grad_dot_convective_flux +=
                        grad_phys_at(q, i, d) * component_or_zero(convective_flux, d);

                  value += jxw(q) * (grad_dot_stress - grad_dot_convective_flux -
                                     scalar_phi_at(q, i) * scalar_forcing(elem, q));
                }
              }
              else
              {
                const unsigned int i = idx - p0;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  value -=
                      jxw(q) * grad_phys_at(q, i, velocity_component) * u_value_at(q);
                  if (velocity_component == 0)
                    value -= jxw(q) * scalar_phi_at(q, i) *
                             pressure_forcing(elem, q);
                }
              }

              elem_residual(idx) += value;
            });
        team.team_barrier();

        for (unsigned int side = 0; side != n_sides; ++side)
        {
          const bool positive_edge_orientation =
              static_cast<bool>(positive_edge_orientations(elem, side));
          const int side_type = side_types(elem, side);
          const bool is_outlet = side_type == hdg_linear_side_outlet;
          const bool is_dirichlet = side_type == hdg_linear_side_dirichlet;

          ::Kokkos::parallel_for(
              ::Kokkos::TeamThreadRange(team, n_side_qpoints),
              [&](const int raw_q)
              {
                const unsigned int q = static_cast<unsigned int>(raw_q);
                const Real px = parent_xi(elem, side, q);
                const Real py = parent_eta(elem, side, q);
                const Real pz = parent_zeta(elem, side, q);

                face_normal_at(q, 0) = normal_x(elem, side, q);
#if LIBMESH_DIM > 1
                face_normal_at(q, 1) = normal_y(elem, side, q);
#endif
#if LIBMESH_DIM > 2
                face_normal_at(q, 2) = normal_z(elem, side, q);
#endif

                for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  face_q_value_at(q, d) = Real(0);
                face_u_value_at(q) = Real(0);
                face_p_value_at(q) = Real(0);
                face_lm_value_at(q) = Real(0);
                face_other_lm_value_at(q) = Real(0);

                for (unsigned int i = 0; i != layout.scalar_n_dofs; ++i)
                {
                  const Real phi = libMesh::Kokkos::shape(scalar_key, i, px, py, pz);
                  face_scalar_phi_at(q, i) = phi;
                  face_u_value_at(q) += phi * elem_u_dofs(i);
                  face_p_value_at(q) += phi * elem_p_dofs(i);
                }

                for (unsigned int i = 0; i != layout.vector_n_dofs; ++i)
                {
                  const RealVector phi =
                      libMesh::Kokkos::vector_shape(vector_key, i, px, py, pz);
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  {
                    const Real phi_d = component_or_zero(phi, d);
                    face_vector_phi_at(q, i, d) = phi_d;
                    face_q_value_at(q, d) += phi_d * elem_q_dofs(i);
                  }
                }

                for (unsigned int i = 0; i != layout.lm_n_dofs; ++i)
                {
                  const Real phi = libMesh::Kokkos::side_trace_shape(
                      lm_key, i, side, positive_edge_orientation, px, py, pz);
                  face_trace_phi_at(q, i) = phi;
                  face_lm_value_at(q) += phi * elem_lm_dofs(i);
                  face_other_lm_value_at(q) += phi * elem_other_lm_dofs(i);
                }
              });
          team.team_barrier();

          ::Kokkos::parallel_for(
              ::Kokkos::TeamThreadRange(team, layout.n_entries - layout.off_Juq),
              [&](const int raw_face_idx)
              {
                const unsigned int idx = static_cast<unsigned int>(raw_face_idx) + layout.off_Juq;
                Real value = Real(0);

                if (idx < layout.off_Jup)
                {
                  if (is_outlet)
                  {
                    elem_blocks(idx) += value;
                    return;
                  }

                  const unsigned int local = idx - layout.off_Juq;
                  const unsigned int i = local / layout.vector_n_dofs;
                  const unsigned int j = local % layout.vector_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real vector_normal = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      vector_normal += face_vector_phi_at(q, j, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) * nu *
                             face_scalar_phi_at(q, i) * vector_normal;
                  }
                }
                else if (idx < layout.off_Jpu)
                {
                  if (is_outlet)
                  {
                    elem_blocks(idx) += value;
                    return;
                  }

                  const unsigned int local = idx - layout.off_Jup;
                  const unsigned int i = local / layout.scalar_n_dofs;
                  const unsigned int j = local % layout.scalar_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    value += side_jxw(elem, side, q) * face_scalar_phi_at(q, i) *
                             face_scalar_phi_at(q, j) *
                             face_normal_at(q, velocity_component);
                }
                else if (idx < layout.off_Jplm)
                {
                }
                else if (idx < layout.off_Jqlm)
                {
                  if (is_dirichlet)
                  {
                    elem_blocks(idx) += value;
                    return;
                  }

                  const unsigned int local = idx - layout.off_Jplm;
                  const unsigned int i = local / layout.lm_n_dofs;
                  const unsigned int j = local % layout.lm_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    value += side_jxw(elem, side, q) * face_scalar_phi_at(q, i) *
                             face_trace_phi_at(q, j) *
                             face_normal_at(q, velocity_component);
                }
                else if (idx < layout.off_Jlmq)
                {
                  if (is_dirichlet)
                  {
                    elem_blocks(idx) += value;
                    return;
                  }

                  const unsigned int local = idx - layout.off_Jqlm;
                  const unsigned int i = local / layout.lm_n_dofs;
                  const unsigned int j = local % layout.lm_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real vector_normal = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      vector_normal += face_vector_phi_at(q, i, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) *
                             vector_normal * face_trace_phi_at(q, j);
                  }
                }
                else if (idx < layout.off_Jlmp)
                {
                  if (is_dirichlet)
                  {
                    elem_blocks(idx) += value;
                    return;
                  }

                  const unsigned int local = idx - layout.off_Jlmq;
                  const unsigned int i = local / layout.vector_n_dofs;
                  const unsigned int j = local % layout.vector_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real vector_normal = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      vector_normal += face_vector_phi_at(q, j, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) * nu *
                             face_trace_phi_at(q, i) * vector_normal;
                  }
                }
                else if (idx < layout.off_Jlms)
                {
                  if (is_dirichlet)
                  {
                    elem_blocks(idx) += value;
                    return;
                  }

                  const unsigned int local = idx - layout.off_Jlmp;
                  const unsigned int i = local / layout.scalar_n_dofs;
                  const unsigned int j = local % layout.scalar_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    value += side_jxw(elem, side, q) * face_trace_phi_at(q, i) *
                             face_scalar_phi_at(q, j) *
                             face_normal_at(q, velocity_component);
                }
                else if (idx < layout.off_Jlmlm)
                {
                  if (is_dirichlet)
                  {
                    elem_blocks(idx) += value;
                    return;
                  }

                  const unsigned int local = idx - layout.off_Jlms;
                  const unsigned int i = local / layout.scalar_n_dofs;
                  const unsigned int j = local % layout.scalar_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real normal_norm2 = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                    value += side_jxw(elem, side, q) *
                             face_trace_phi_at(q, i) * tau * face_scalar_phi_at(q, j) *
                             normal_norm2;
                  }
                }
                else if (idx < layout.off_Juu)
                {
                  const unsigned int local = idx - layout.off_Jlmlm;
                  const unsigned int i = local / layout.lm_n_dofs;
                  const unsigned int j = local % layout.lm_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    if (is_dirichlet)
                    {
                      value -= side_jxw(elem, side, q) * face_trace_phi_at(q, i) *
                               face_trace_phi_at(q, j);
                      continue;
                    }

                    Real normal_norm2 = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) * face_trace_phi_at(q, i) * tau *
                             face_trace_phi_at(q, j) * normal_norm2;
                  }
                }
                else if (idx < layout.off_Julm)
                {
                  if (is_outlet)
                  {
                    elem_blocks(idx) += value;
                    return;
                  }

                  const unsigned int local = idx - layout.off_Juu;
                  const unsigned int i = local / layout.scalar_n_dofs;
                  const unsigned int j = local % layout.scalar_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real normal_norm2 = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                    value += side_jxw(elem, side, q) *
                             face_scalar_phi_at(q, i) * tau * face_scalar_phi_at(q, j) *
                             normal_norm2;
                  }
                }
                else
                {
                  if (is_outlet || is_dirichlet)
                  {
                    elem_blocks(idx) += value;
                    return;
                  }

                  const unsigned int local = idx - layout.off_Julm;
                  const unsigned int i = local / layout.lm_n_dofs;
                  const unsigned int j = local % layout.lm_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real normal_norm2 = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) *
                             face_scalar_phi_at(q, i) * tau * face_trace_phi_at(q, j) *
                             normal_norm2;
                  }
                }

                elem_blocks(idx) += value;
              });
          team.team_barrier();

          ::Kokkos::parallel_for(
              ::Kokkos::TeamThreadRange(team, residual_n_dofs),
              [&](const int raw_idx)
              {
                const unsigned int idx = static_cast<unsigned int>(raw_idx);
                Real value = Real(0);

                if (idx < u0)
                {
                  const unsigned int i = idx;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    const Real boundary_value =
                        is_dirichlet ? dirichlet_component(elem, side, q)
                                     : face_lm_value_at(q);
                    Real vector_normal = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      vector_normal += face_vector_phi_at(q, i, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) * vector_normal * boundary_value;
                  }
                }
                else if (idx < p0)
                {
                  const unsigned int i = idx - u0;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real normal_norm2 = Real(0);
                    Real q_normal = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    {
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                      q_normal += face_q_value_at(q, d) * face_normal_at(q, d);
                    }
                    const Real u_value = face_u_value_at(q);
                    const Real p_value = face_p_value_at(q);
                    const Real lm_value = face_lm_value_at(q);
                    const Real other_lm_value = face_other_lm_value_at(q);
                    const Real boundary_value =
                        is_dirichlet ? dirichlet_component(elem, side, q) : lm_value;
                    const Real advective_vdotn =
                        velocity_component == 0
                            ? face_normal_at(q, 0) * lm_value + face_normal_at(q, 1) * other_lm_value
                            : face_normal_at(q, 0) * other_lm_value + face_normal_at(q, 1) * lm_value;
                    const Real advective_flux =
                        !include_convection ? Real(0)
                        : is_dirichlet      ? dirichlet_vdotn(elem, side, q) * boundary_value
                                            : advective_vdotn * lm_value;
                    const Real phi_i = face_scalar_phi_at(q, i);

                    const Real linear_flux =
                        is_outlet
                            ? Real(0)
                            : -nu * q_normal +
                                  p_value * face_normal_at(q, velocity_component) +
                                  tau * (u_value - boundary_value) * normal_norm2;
                    value += side_jxw(elem, side, q) * phi_i * (linear_flux + advective_flux);
                  }
                }
                else if (idx < lm0)
                {
                  const unsigned int i = idx - p0;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    const Real boundary_value =
                        is_dirichlet ? dirichlet_component(elem, side, q)
                                     : face_lm_value_at(q);
                    value += side_jxw(elem, side, q) * face_scalar_phi_at(q, i) *
                             boundary_value * face_normal_at(q, velocity_component);
                  }
                }
                else
                {
                  const unsigned int i = idx - lm0;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    const Real phi_i = face_trace_phi_at(q, i);
                    if (is_dirichlet)
                    {
                      const Real lm_value = face_lm_value_at(q);
                      value -= side_jxw(elem, side, q) * phi_i * lm_value;
                      continue;
                    }

                    Real normal_norm2 = Real(0);
                    Real q_normal = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    {
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                      q_normal += face_q_value_at(q, d) * face_normal_at(q, d);
                    }
                    const Real u_value = face_u_value_at(q);
                    const Real p_value = face_p_value_at(q);
                    const Real lm_value = face_lm_value_at(q);
                    const Real other_lm_value = face_other_lm_value_at(q);
                    const Real advective_vdotn =
                        velocity_component == 0
                            ? face_normal_at(q, 0) * lm_value + face_normal_at(q, 1) * other_lm_value
                            : face_normal_at(q, 0) * other_lm_value + face_normal_at(q, 1) * lm_value;
                    const Real advective_flux =
                        (!include_convection || is_outlet)
                            ? Real(0)
                            : advective_vdotn * lm_value;
                    value += side_jxw(elem, side, q) * phi_i *
                             (-nu * q_normal +
                              p_value * face_normal_at(q, velocity_component) +
                              tau * (u_value - lm_value) * normal_norm2 + advective_flux);
                  }
                }

                elem_residual(idx) += value;
              });
          team.team_barrier();
        }

        if (condense_outputs)
        {
          condense_hdg_linear_qu_blocks_team(team,
                                             layout,
                                             elem_blocks,
                                             elem_condensed,
                                             condense_status,
                                             elem,
                                             condense_aug,
                                             condense_aux);
          team.team_barrier();
          condense_hdg_linear_qu_residual_team(team,
                                               layout,
                                               elem_blocks,
                                               elem_residual,
                                               elem_condensed_residual,
                                               residual_status,
                                               elem,
                                               condense_aug,
                                               condense_aux);
        }
        else if (team.team_rank() == 0)
        {
          condense_status(elem) = 0;
          residual_status(elem) = 0;
        }
      });
}

template <libMesh::ElemType ExactTopo,
          typename NodeStorage,
          typename VolumeQpStorage,
          typename SideQpStorage,
          typename OrientationStorage,
          typename SideTypeStorage,
          typename SideValueStorage,
          typename ForcingStorage,
          typename DofStorage,
          typename BlockStorage,
          typename FullBlockStorage,
          typename ResidualStorage,
          typename StatusStorage>
void
assemble_hdg_linear_batch_boundary_pair(const libMesh::FEShapeKey vector_key,
                                        const libMesh::FEShapeKey scalar_key,
                                        const libMesh::FEShapeKey lm_key,
                                        const HDGLinearBlockLayout layout,
                                        const NodeStorage & coords,
                                        const unsigned int n_elements,
                                        const unsigned int n_geom_nodes,
                                        const VolumeQpStorage & xi,
                                        const VolumeQpStorage & eta,
                                        const VolumeQpStorage & zeta,
                                        const VolumeQpStorage & jxw,
                                        const ForcingStorage & scalar_forcing_0,
                                        const ForcingStorage & scalar_forcing_1,
                                        const ForcingStorage & pressure_forcing,
                                        const unsigned int n_qpoints,
                                        const SideQpStorage & parent_xi,
                                        const SideQpStorage & parent_eta,
                                        const SideQpStorage & parent_zeta,
                                        const SideQpStorage & side_jxw,
                                        const SideQpStorage & normal_x,
                                        const SideQpStorage & normal_y,
                                        const SideQpStorage & normal_z,
                                        const OrientationStorage & positive_edge_orientations,
                                        const SideTypeStorage & side_types,
                                        const SideValueStorage & dirichlet_component_0,
                                        const SideValueStorage & dirichlet_component_1,
                                        const SideValueStorage & dirichlet_vdotn,
                                        const unsigned int n_sides,
                                        const unsigned int n_side_qpoints,
                                        const Real nu,
                                        const Real tau,
                                        const bool include_convection,
                                        const DofStorage & q_dofs_0,
                                        const DofStorage & u_dofs_0,
                                        const DofStorage & lm_dofs_0,
                                        const DofStorage & other_u_dofs_0,
                                        const DofStorage & other_lm_dofs_0,
                                        const DofStorage & q_dofs_1,
                                        const DofStorage & u_dofs_1,
                                        const DofStorage & p_dofs,
                                        const DofStorage & lm_dofs_1,
                                        const DofStorage & other_u_dofs_1,
                                        const DofStorage & other_lm_dofs_1,
                                        const BlockStorage & blocks_0,
                                        const ResidualStorage & residual_0,
                                        const BlockStorage & blocks_1,
                                        const ResidualStorage & residual_1,
                                        const FullBlockStorage & full_convective_blocks,
                                        const bool add_full_convective_jacobian,
                                        const unsigned int full_n_dofs,
                                        const unsigned int full_off_u,
                                        const unsigned int full_off_v,
                                        const unsigned int full_off_lm_u,
                                        const unsigned int full_off_lm_v,
                                        const StatusStorage & condense_status_0,
                                        const StatusStorage & residual_status_0,
                                        const StatusStorage & condense_status_1,
                                        const StatusStorage & residual_status_1,
                                        const char * const kernel_name,
                                        const unsigned int requested_team_size = 0,
                                        const bool assemble_residual = true)
{
  using ExecutionSpace = typename BlockStorage::execution_space;
  using TeamPolicy = ::Kokkos::TeamPolicy<ExecutionSpace>;
  using MemberType = typename TeamPolicy::member_type;
  using ScratchSpace = typename MemberType::scratch_memory_space;
  using ScratchRealView =
      ::Kokkos::View<Real *, ScratchSpace, ::Kokkos::MemoryTraits<::Kokkos::Unmanaged>>;

  const std::size_t volume_scalar_phi_size = n_qpoints * layout.scalar_n_dofs;
  const std::size_t volume_grad_phys_size = n_qpoints * layout.scalar_n_dofs * LIBMESH_DIM;
  const std::size_t volume_vector_phi_size = n_qpoints * layout.vector_n_dofs * LIBMESH_DIM;
  const std::size_t volume_q_value_size = 2 * n_qpoints * LIBMESH_DIM;
  const std::size_t volume_scalar_values_size = 5 * n_qpoints;
  const std::size_t volume_scratch_size = volume_scalar_phi_size + volume_grad_phys_size +
                                          volume_vector_phi_size + volume_q_value_size +
                                          volume_scalar_values_size;
  const std::size_t face_scalar_phi_size = n_side_qpoints * layout.scalar_n_dofs;
  const std::size_t face_vector_phi_size = n_side_qpoints * layout.vector_n_dofs * LIBMESH_DIM;
  const std::size_t face_trace_phi_size = n_side_qpoints * layout.lm_n_dofs;
  const std::size_t face_q_value_size = 2 * n_side_qpoints * LIBMESH_DIM;
  const std::size_t face_normals_size = n_side_qpoints * LIBMESH_DIM;
  const std::size_t face_scalar_values_size = 7 * n_side_qpoints;
  const std::size_t face_scratch_size = face_scalar_phi_size + face_vector_phi_size +
                                        face_trace_phi_size + face_q_value_size +
                                        face_normals_size + face_scalar_values_size;
  const std::size_t scratch_bytes = ScratchRealView::shmem_size(volume_scratch_size) +
                                    ScratchRealView::shmem_size(face_scratch_size);

  TeamPolicy policy = requested_team_size ?
                          TeamPolicy(static_cast<int>(n_elements),
                                     static_cast<int>(requested_team_size)) :
                          TeamPolicy(static_cast<int>(n_elements), ::Kokkos::AUTO());
  policy = policy.set_scratch_size(0, ::Kokkos::PerTeam(scratch_bytes));

  ::Kokkos::parallel_for(
      kernel_name, policy, KOKKOS_LAMBDA(const MemberType & team) {
        const unsigned int elem = static_cast<unsigned int>(team.league_rank());
        const HDGElementNodeView<NodeStorage> elem_coords{coords, elem};
        const HDGElementConstView<DofStorage> elem_q_dofs_0{q_dofs_0, elem};
        const HDGElementConstView<DofStorage> elem_u_dofs_0{u_dofs_0, elem};
        const HDGElementConstView<DofStorage> elem_lm_dofs_0{lm_dofs_0, elem};
        const HDGElementConstView<DofStorage> elem_other_u_dofs_0{other_u_dofs_0, elem};
        const HDGElementConstView<DofStorage> elem_other_lm_dofs_0{other_lm_dofs_0, elem};
        const HDGElementConstView<DofStorage> elem_q_dofs_1{q_dofs_1, elem};
        const HDGElementConstView<DofStorage> elem_u_dofs_1{u_dofs_1, elem};
        const HDGElementConstView<DofStorage> elem_p_dofs{p_dofs, elem};
        const HDGElementConstView<DofStorage> elem_lm_dofs_1{lm_dofs_1, elem};
        const HDGElementConstView<DofStorage> elem_other_u_dofs_1{other_u_dofs_1, elem};
        const HDGElementConstView<DofStorage> elem_other_lm_dofs_1{other_lm_dofs_1, elem};
        ScratchRealView volume_scratch(team.team_scratch(0), volume_scratch_size);
        ScratchRealView face_scratch(team.team_scratch(0), face_scratch_size);

        const unsigned int q0 = 0;
        const unsigned int u0 = q0 + layout.vector_n_dofs;
        const unsigned int p0 = u0 + layout.scalar_n_dofs;
        const unsigned int lm0 = p0 + layout.scalar_n_dofs;
        const unsigned int residual_n_dofs = hdg_linear_n_dofs(layout);

        const unsigned int volume_scalar_phi_offset = 0;
        const unsigned int volume_grad_phys_offset =
            volume_scalar_phi_offset + volume_scalar_phi_size;
        const unsigned int volume_vector_phi_offset =
            volume_grad_phys_offset + volume_grad_phys_size;
        const unsigned int volume_q_value_offset =
            volume_vector_phi_offset + volume_vector_phi_size;
        const unsigned int volume_u_value_offset = volume_q_value_offset + volume_q_value_size;
        const unsigned int volume_p_value_offset = volume_u_value_offset + 2 * n_qpoints;
        const unsigned int volume_other_u_value_offset = volume_p_value_offset + n_qpoints;

        auto scalar_phi_at = [&](const unsigned int q, const unsigned int i) -> Real &
        { return volume_scratch(volume_scalar_phi_offset + q * layout.scalar_n_dofs + i); };
        auto grad_phys_at = [&](const unsigned int q,
                                const unsigned int i,
                                const unsigned int d) -> Real &
        {
          return volume_scratch(volume_grad_phys_offset +
                                (q * layout.scalar_n_dofs + i) * LIBMESH_DIM + d);
        };
        auto vector_phi_at = [&](const unsigned int q,
                                 const unsigned int i,
                                 const unsigned int d) -> Real &
        {
          return volume_scratch(volume_vector_phi_offset +
                                (q * layout.vector_n_dofs + i) * LIBMESH_DIM + d);
        };
        auto q_value_at = [&](const unsigned int c,
                              const unsigned int q,
                              const unsigned int d) -> Real &
        { return volume_scratch(volume_q_value_offset + (c * n_qpoints + q) * LIBMESH_DIM + d); };
        auto u_value_at = [&](const unsigned int c, const unsigned int q) -> Real &
        { return volume_scratch(volume_u_value_offset + c * n_qpoints + q); };
        auto p_value_at = [&](const unsigned int q) -> Real &
        { return volume_scratch(volume_p_value_offset + q); };
        auto other_u_value_at = [&](const unsigned int c, const unsigned int q) -> Real &
        { return volume_scratch(volume_other_u_value_offset + c * n_qpoints + q); };

        const unsigned int face_scalar_phi_offset = 0;
        const unsigned int face_vector_phi_offset = face_scalar_phi_offset + face_scalar_phi_size;
        const unsigned int face_trace_phi_offset = face_vector_phi_offset + face_vector_phi_size;
        const unsigned int face_q_value_offset = face_trace_phi_offset + face_trace_phi_size;
        const unsigned int face_normals_offset = face_q_value_offset + face_q_value_size;
        const unsigned int face_u_value_offset = face_normals_offset + face_normals_size;
        const unsigned int face_p_value_offset = face_u_value_offset + 2 * n_side_qpoints;
        const unsigned int face_lm_value_offset = face_p_value_offset + n_side_qpoints;
        const unsigned int face_other_lm_value_offset =
            face_lm_value_offset + 2 * n_side_qpoints;

        auto face_scalar_phi_at = [&](const unsigned int q, const unsigned int i) -> Real &
        { return face_scratch(face_scalar_phi_offset + q * layout.scalar_n_dofs + i); };
        auto face_vector_phi_at = [&](const unsigned int q,
                                      const unsigned int i,
                                      const unsigned int d) -> Real &
        {
          return face_scratch(face_vector_phi_offset +
                              (q * layout.vector_n_dofs + i) * LIBMESH_DIM + d);
        };
        auto face_trace_phi_at = [&](const unsigned int q, const unsigned int i) -> Real &
        { return face_scratch(face_trace_phi_offset + q * layout.lm_n_dofs + i); };
        auto face_q_value_at = [&](const unsigned int c,
                                   const unsigned int q,
                                   const unsigned int d) -> Real &
        { return face_scratch(face_q_value_offset + (c * n_side_qpoints + q) * LIBMESH_DIM + d); };
        auto face_normal_at = [&](const unsigned int q, const unsigned int d) -> Real &
        { return face_scratch(face_normals_offset + q * LIBMESH_DIM + d); };
        auto face_u_value_at = [&](const unsigned int c, const unsigned int q) -> Real &
        { return face_scratch(face_u_value_offset + c * n_side_qpoints + q); };
        auto face_p_value_at = [&](const unsigned int q) -> Real &
        { return face_scratch(face_p_value_offset + q); };
        auto face_lm_value_at = [&](const unsigned int c, const unsigned int q) -> Real &
        { return face_scratch(face_lm_value_offset + c * n_side_qpoints + q); };
        auto face_other_lm_value_at = [&](const unsigned int c, const unsigned int q) -> Real &
        { return face_scratch(face_other_lm_value_offset + c * n_side_qpoints + q); };
        auto full_convective_block = [&](const unsigned int row, const unsigned int col) -> Real &
        { return full_convective_blocks(elem, row * full_n_dofs + col); };

        ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, layout.n_entries),
                               [&](const int raw_i)
                               {
                                 const unsigned int i = static_cast<unsigned int>(raw_i);
                                 blocks_0(elem, i) = Real(0);
                                 blocks_1(elem, i) = Real(0);
                               });
        if (assemble_residual)
          ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, residual_n_dofs),
                                 [&](const int raw_i)
                                 {
                                   const unsigned int i = static_cast<unsigned int>(raw_i);
                                   residual_0(elem, i) = Real(0);
                                   residual_1(elem, i) = Real(0);
                                 });
        team.team_barrier();

        ::Kokkos::parallel_for(
            ::Kokkos::TeamThreadRange(team, n_qpoints),
            [&](const int raw_q)
            {
              const unsigned int q = static_cast<unsigned int>(raw_q);
              const Real xiq = xi(q);
              const Real etaq = eta(q);
              const Real zetaq = zeta(q);
              const RealTensor J =
                  libMesh::Kokkos::jacobian<libMesh::LAGRANGE, ExactTopo>(
                      elem_coords, n_geom_nodes, xiq, etaq, zetaq);
              const RealTensor Jinv = libMesh::Kokkos::inverse<RealTensor>(J, 2);

              for (unsigned int i = 0; i != layout.scalar_n_dofs; ++i)
              {
                scalar_phi_at(q, i) = libMesh::Kokkos::shape(scalar_key, i, xiq, etaq, zetaq);
                const RealVector grad_phys =
                    Jinv * libMesh::Kokkos::grad_shape(scalar_key, i, xiq, etaq, zetaq);
                for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  grad_phys_at(q, i, d) = component_or_zero(grad_phys, d);
              }

              for (unsigned int i = 0; i != layout.vector_n_dofs; ++i)
              {
                const RealVector phi =
                    libMesh::Kokkos::vector_shape(vector_key, i, xiq, etaq, zetaq);
                for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  vector_phi_at(q, i, d) = component_or_zero(phi, d);
              }

              for (unsigned int c = 0; c != 2; ++c)
                for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  q_value_at(c, q, d) = Real(0);
              u_value_at(0, q) = Real(0);
              u_value_at(1, q) = Real(0);
              p_value_at(q) = Real(0);
              other_u_value_at(0, q) = Real(0);
              other_u_value_at(1, q) = Real(0);

              for (unsigned int i = 0; i != layout.vector_n_dofs; ++i)
                for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                {
                  q_value_at(0, q, d) += vector_phi_at(q, i, d) * elem_q_dofs_0(i);
                  q_value_at(1, q, d) += vector_phi_at(q, i, d) * elem_q_dofs_1(i);
                }

              for (unsigned int i = 0; i != layout.scalar_n_dofs; ++i)
              {
                const Real phi = scalar_phi_at(q, i);
                u_value_at(0, q) += phi * elem_u_dofs_0(i);
                u_value_at(1, q) += phi * elem_u_dofs_1(i);
                p_value_at(q) += phi * elem_p_dofs(i);
                other_u_value_at(0, q) += phi * elem_other_u_dofs_0(i);
                other_u_value_at(1, q) += phi * elem_other_u_dofs_1(i);
              }
            });
        team.team_barrier();

        ::Kokkos::parallel_for(
            ::Kokkos::TeamThreadRange(team, 2 * layout.off_Jplm),
            [&](const int raw_component_idx)
            {
              const unsigned int component_idx = static_cast<unsigned int>(raw_component_idx);
              const unsigned int c = component_idx / layout.off_Jplm;
              const unsigned int idx = component_idx - c * layout.off_Jplm;
              const unsigned int velocity_component = c;
              Real value = Real(0);

              if (idx < layout.off_Jqu)
              {
                const unsigned int local = idx - layout.off_Jqq;
                const unsigned int i = local / layout.vector_n_dofs;
                const unsigned int j = local % layout.vector_n_dofs;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  Real dot = Real(0);
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    dot += vector_phi_at(q, i, d) * vector_phi_at(q, j, d);
                  value += jxw(q) * dot;
                }
              }
              else if (idx < layout.off_Juq)
              {
                const unsigned int local = idx - layout.off_Jqu;
                const unsigned int i = local / layout.scalar_n_dofs;
                const unsigned int j = local % layout.scalar_n_dofs;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  const unsigned int scalar_i = i / 2;
                  const unsigned int component = i - scalar_i * 2;
                  value += jxw(q) * grad_phys_at(q, scalar_i, component) *
                           scalar_phi_at(q, j);
                }
              }
              else if (idx < layout.off_Jup)
              {
                const unsigned int local = idx - layout.off_Juq;
                const unsigned int i = local / layout.vector_n_dofs;
                const unsigned int j = local % layout.vector_n_dofs;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  Real dot = Real(0);
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    dot += grad_phys_at(q, i, d) * vector_phi_at(q, j, d);
                  value += jxw(q) * nu * dot;
                }
              }
              else
              {
                const unsigned int local =
                    idx < layout.off_Jpu ? idx - layout.off_Jup : idx - layout.off_Jpu;
                const unsigned int i = local / layout.scalar_n_dofs;
                const unsigned int j = local % layout.scalar_n_dofs;
                for (unsigned int q = 0; q != n_qpoints; ++q)
                  value -= jxw(q) * grad_phys_at(q, i, velocity_component) *
                           scalar_phi_at(q, j);
              }

              if (c == 0)
                blocks_0(elem, idx) += value;
              else
                blocks_1(elem, idx) += value;
            });
        team.team_barrier();

        if (assemble_residual)
        {
          const unsigned int volume_residual_n_dofs =
              layout.vector_n_dofs + 2 * layout.scalar_n_dofs;
          ::Kokkos::parallel_for(
              ::Kokkos::TeamThreadRange(team, 2 * volume_residual_n_dofs),
              [&](const int raw_component_idx)
              {
                const unsigned int component_idx = static_cast<unsigned int>(raw_component_idx);
                const unsigned int c = component_idx / volume_residual_n_dofs;
                const unsigned int idx = component_idx - c * volume_residual_n_dofs;
                const unsigned int velocity_component = c;
                Real value = Real(0);

                if (idx < u0)
                {
                  const unsigned int i = idx;
                  for (unsigned int q = 0; q != n_qpoints; ++q)
                  {
                    const unsigned int scalar_i = i / 2;
                    const unsigned int component = i - scalar_i * 2;
                    Real phi_dot_q = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      phi_dot_q += vector_phi_at(q, i, d) * q_value_at(c, q, d);

                    value += jxw(q) *
                             (phi_dot_q + grad_phys_at(q, scalar_i, component) *
                                              u_value_at(c, q));
                  }
                }
                else if (idx < p0)
                {
                  const unsigned int i = idx - u0;
                  for (unsigned int q = 0; q != n_qpoints; ++q)
                  {
                    Real grad_dot_stress = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    {
                      Real stress_component = nu * q_value_at(c, q, d);
                      if (d == velocity_component)
                        stress_component -= p_value_at(q);
                      grad_dot_stress += grad_phys_at(q, i, d) * stress_component;
                    }
                    const Real active_u = u_value_at(c, q);
                    const Real other_u = other_u_value_at(c, q);
                    const RealVector convective_flux =
                        include_convection
                            ? hdg_velocity_from_components(active_u, other_u, velocity_component) *
                                  active_u
                            : RealVector();
                    Real grad_dot_convective_flux = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      grad_dot_convective_flux +=
                          grad_phys_at(q, i, d) * component_or_zero(convective_flux, d);
                    const Real forcing =
                        c == 0 ? scalar_forcing_0(elem, q) : scalar_forcing_1(elem, q);

                    value += jxw(q) * (grad_dot_stress - grad_dot_convective_flux -
                                       scalar_phi_at(q, i) * forcing);
                  }
                }
                else
                {
                  const unsigned int i = idx - p0;
                  for (unsigned int q = 0; q != n_qpoints; ++q)
                  {
                    value -= jxw(q) * grad_phys_at(q, i, velocity_component) *
                             u_value_at(c, q);
                    if (velocity_component == 0)
                      value -= jxw(q) * scalar_phi_at(q, i) * pressure_forcing(elem, q);
                  }
                }

                if (c == 0)
                  residual_0(elem, idx) += value;
                else
                  residual_1(elem, idx) += value;
              });
          team.team_barrier();
        }

        if (add_full_convective_jacobian && include_convection)
        {
          ::Kokkos::parallel_for(
              ::Kokkos::TeamThreadRange(team, layout.scalar_n_dofs * layout.scalar_n_dofs),
              [&](const int raw_idx)
              {
                const unsigned int idx = static_cast<unsigned int>(raw_idx);
                const unsigned int i = idx / layout.scalar_n_dofs;
                const unsigned int j = idx - i * layout.scalar_n_dofs;
                Real u_u_col = Real(0);
                Real ux_v_col = Real(0);
                Real vy_u_col = Real(0);
                Real v_v_col = Real(0);

                for (unsigned int q = 0; q != n_qpoints; ++q)
                {
                  const Real phi_j = scalar_phi_at(q, j);
                  const Real grad_x = grad_phys_at(q, i, 0);
                  const Real grad_y = grad_phys_at(q, i, 1);
                  const Real u_value = u_value_at(0, q);
                  const Real v_value = u_value_at(1, q);

                  u_u_col -= jxw(q) * phi_j *
                             (Real(2) * grad_x * u_value + grad_y * v_value);
                  ux_v_col -= jxw(q) * grad_y * phi_j * u_value;
                  vy_u_col -= jxw(q) * grad_x * phi_j * v_value;
                  v_v_col -= jxw(q) * phi_j *
                             (grad_x * u_value + Real(2) * grad_y * v_value);
                }

                full_convective_block(full_off_u + i, full_off_u + j) += u_u_col;
                full_convective_block(full_off_u + i, full_off_v + j) += ux_v_col;
                full_convective_block(full_off_v + i, full_off_u + j) += vy_u_col;
                full_convective_block(full_off_v + i, full_off_v + j) += v_v_col;
              });
          team.team_barrier();
        }

        for (unsigned int side = 0; side != n_sides; ++side)
        {
          const bool positive_edge_orientation =
              static_cast<bool>(positive_edge_orientations(elem, side));
          const int side_type = side_types(elem, side);
          const bool is_outlet = side_type == hdg_linear_side_outlet;
          const bool is_dirichlet = side_type == hdg_linear_side_dirichlet;

          ::Kokkos::parallel_for(
              ::Kokkos::TeamThreadRange(team, n_side_qpoints),
              [&](const int raw_q)
              {
                const unsigned int q = static_cast<unsigned int>(raw_q);
                const Real px = parent_xi(elem, side, q);
                const Real py = parent_eta(elem, side, q);
                const Real pz = parent_zeta(elem, side, q);

                face_normal_at(q, 0) = normal_x(elem, side, q);
#if LIBMESH_DIM > 1
                face_normal_at(q, 1) = normal_y(elem, side, q);
#endif
#if LIBMESH_DIM > 2
                face_normal_at(q, 2) = normal_z(elem, side, q);
#endif

                for (unsigned int c = 0; c != 2; ++c)
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                    face_q_value_at(c, q, d) = Real(0);
                face_u_value_at(0, q) = Real(0);
                face_u_value_at(1, q) = Real(0);
                face_p_value_at(q) = Real(0);
                face_lm_value_at(0, q) = Real(0);
                face_lm_value_at(1, q) = Real(0);
                face_other_lm_value_at(0, q) = Real(0);
                face_other_lm_value_at(1, q) = Real(0);

                for (unsigned int i = 0; i != layout.scalar_n_dofs; ++i)
                {
                  const Real phi = libMesh::Kokkos::shape(scalar_key, i, px, py, pz);
                  face_scalar_phi_at(q, i) = phi;
                  face_u_value_at(0, q) += phi * elem_u_dofs_0(i);
                  face_u_value_at(1, q) += phi * elem_u_dofs_1(i);
                  face_p_value_at(q) += phi * elem_p_dofs(i);
                }

                for (unsigned int i = 0; i != layout.vector_n_dofs; ++i)
                {
                  const RealVector phi =
                      libMesh::Kokkos::vector_shape(vector_key, i, px, py, pz);
                  for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  {
                    const Real phi_d = component_or_zero(phi, d);
                    face_vector_phi_at(q, i, d) = phi_d;
                    face_q_value_at(0, q, d) += phi_d * elem_q_dofs_0(i);
                    face_q_value_at(1, q, d) += phi_d * elem_q_dofs_1(i);
                  }
                }

                for (unsigned int i = 0; i != layout.lm_n_dofs; ++i)
                {
                  const Real phi = libMesh::Kokkos::side_trace_shape(
                      lm_key, i, side, positive_edge_orientation, px, py, pz);
                  face_trace_phi_at(q, i) = phi;
                  face_lm_value_at(0, q) += phi * elem_lm_dofs_0(i);
                  face_lm_value_at(1, q) += phi * elem_lm_dofs_1(i);
                  face_other_lm_value_at(0, q) += phi * elem_other_lm_dofs_0(i);
                  face_other_lm_value_at(1, q) += phi * elem_other_lm_dofs_1(i);
                }
              });
          team.team_barrier();

          const unsigned int side_block_n_entries = layout.n_entries - layout.off_Juq;
          ::Kokkos::parallel_for(
              ::Kokkos::TeamThreadRange(team, 2 * side_block_n_entries),
              [&](const int raw_component_idx)
              {
                const unsigned int component_idx = static_cast<unsigned int>(raw_component_idx);
                const unsigned int c = component_idx / side_block_n_entries;
                const unsigned int idx =
                    component_idx - c * side_block_n_entries + layout.off_Juq;
                const unsigned int velocity_component = c;
                Real value = Real(0);

                if (idx < layout.off_Jup)
                {
                  if (is_outlet)
                  {
                    if (c == 0)
                      blocks_0(elem, idx) += value;
                    else
                      blocks_1(elem, idx) += value;
                    return;
                  }
                  const unsigned int local = idx - layout.off_Juq;
                  const unsigned int i = local / layout.vector_n_dofs;
                  const unsigned int j = local % layout.vector_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real vector_normal = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      vector_normal += face_vector_phi_at(q, j, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) * nu *
                             face_scalar_phi_at(q, i) * vector_normal;
                  }
                }
                else if (idx < layout.off_Jpu)
                {
                  if (is_outlet)
                  {
                    if (c == 0)
                      blocks_0(elem, idx) += value;
                    else
                      blocks_1(elem, idx) += value;
                    return;
                  }
                  const unsigned int local = idx - layout.off_Jup;
                  const unsigned int i = local / layout.scalar_n_dofs;
                  const unsigned int j = local % layout.scalar_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    value += side_jxw(elem, side, q) * face_scalar_phi_at(q, i) *
                             face_scalar_phi_at(q, j) *
                             face_normal_at(q, velocity_component);
                }
                else if (idx < layout.off_Jplm)
                {
                }
                else if (idx < layout.off_Jqlm)
                {
                  if (is_dirichlet)
                  {
                    if (c == 0)
                      blocks_0(elem, idx) += value;
                    else
                      blocks_1(elem, idx) += value;
                    return;
                  }
                  const unsigned int local = idx - layout.off_Jplm;
                  const unsigned int i = local / layout.lm_n_dofs;
                  const unsigned int j = local % layout.lm_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    value += side_jxw(elem, side, q) * face_scalar_phi_at(q, i) *
                             face_trace_phi_at(q, j) *
                             face_normal_at(q, velocity_component);
                }
                else if (idx < layout.off_Jlmq)
                {
                  if (is_dirichlet)
                  {
                    if (c == 0)
                      blocks_0(elem, idx) += value;
                    else
                      blocks_1(elem, idx) += value;
                    return;
                  }
                  const unsigned int local = idx - layout.off_Jqlm;
                  const unsigned int i = local / layout.lm_n_dofs;
                  const unsigned int j = local % layout.lm_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real vector_normal = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      vector_normal += face_vector_phi_at(q, i, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) * vector_normal * face_trace_phi_at(q, j);
                  }
                }
                else if (idx < layout.off_Jlmp)
                {
                  if (is_dirichlet)
                  {
                    if (c == 0)
                      blocks_0(elem, idx) += value;
                    else
                      blocks_1(elem, idx) += value;
                    return;
                  }
                  const unsigned int local = idx - layout.off_Jlmq;
                  const unsigned int i = local / layout.vector_n_dofs;
                  const unsigned int j = local % layout.vector_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real vector_normal = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      vector_normal += face_vector_phi_at(q, j, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) * nu *
                             face_trace_phi_at(q, i) * vector_normal;
                  }
                }
                else if (idx < layout.off_Jlms)
                {
                  if (is_dirichlet)
                  {
                    if (c == 0)
                      blocks_0(elem, idx) += value;
                    else
                      blocks_1(elem, idx) += value;
                    return;
                  }
                  const unsigned int local = idx - layout.off_Jlmp;
                  const unsigned int i = local / layout.scalar_n_dofs;
                  const unsigned int j = local % layout.scalar_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    value += side_jxw(elem, side, q) * face_trace_phi_at(q, i) *
                             face_scalar_phi_at(q, j) *
                             face_normal_at(q, velocity_component);
                }
                else if (idx < layout.off_Jlmlm)
                {
                  if (is_dirichlet)
                  {
                    if (c == 0)
                      blocks_0(elem, idx) += value;
                    else
                      blocks_1(elem, idx) += value;
                    return;
                  }
                  const unsigned int local = idx - layout.off_Jlms;
                  const unsigned int i = local / layout.scalar_n_dofs;
                  const unsigned int j = local % layout.scalar_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real normal_norm2 = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                    value += side_jxw(elem, side, q) * face_trace_phi_at(q, i) * tau *
                             face_scalar_phi_at(q, j) * normal_norm2;
                  }
                }
                else if (idx < layout.off_Juu)
                {
                  const unsigned int local = idx - layout.off_Jlmlm;
                  const unsigned int i = local / layout.lm_n_dofs;
                  const unsigned int j = local % layout.lm_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    if (is_dirichlet)
                    {
                      value -= side_jxw(elem, side, q) * face_trace_phi_at(q, i) *
                               face_trace_phi_at(q, j);
                      continue;
                    }
                    Real normal_norm2 = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) * face_trace_phi_at(q, i) * tau *
                             face_trace_phi_at(q, j) * normal_norm2;
                  }
                }
                else if (idx < layout.off_Julm)
                {
                  if (is_outlet)
                  {
                    if (c == 0)
                      blocks_0(elem, idx) += value;
                    else
                      blocks_1(elem, idx) += value;
                    return;
                  }
                  const unsigned int local = idx - layout.off_Juu;
                  const unsigned int i = local / layout.scalar_n_dofs;
                  const unsigned int j = local % layout.scalar_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real normal_norm2 = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                    value += side_jxw(elem, side, q) * face_scalar_phi_at(q, i) * tau *
                             face_scalar_phi_at(q, j) * normal_norm2;
                  }
                }
                else
                {
                  if (is_outlet || is_dirichlet)
                  {
                    if (c == 0)
                      blocks_0(elem, idx) += value;
                    else
                      blocks_1(elem, idx) += value;
                    return;
                  }
                  const unsigned int local = idx - layout.off_Julm;
                  const unsigned int i = local / layout.lm_n_dofs;
                  const unsigned int j = local % layout.lm_n_dofs;
                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    Real normal_norm2 = Real(0);
                    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                    value -= side_jxw(elem, side, q) * face_scalar_phi_at(q, i) * tau *
                             face_trace_phi_at(q, j) * normal_norm2;
                  }
                }

                if (c == 0)
                  blocks_0(elem, idx) += value;
                else
                  blocks_1(elem, idx) += value;
              });
          team.team_barrier();

          if (assemble_residual)
          {
            ::Kokkos::parallel_for(
                ::Kokkos::TeamThreadRange(team, 2 * residual_n_dofs),
                [&](const int raw_component_idx)
                {
                  const unsigned int component_idx = static_cast<unsigned int>(raw_component_idx);
                  const unsigned int c = component_idx / residual_n_dofs;
                  const unsigned int idx = component_idx - c * residual_n_dofs;
                  const unsigned int velocity_component = c;
                  Real value = Real(0);

                  if (idx < u0)
                  {
                    const unsigned int i = idx;
                    for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    {
                      const Real boundary_value =
                          is_dirichlet
                              ? (c == 0 ? dirichlet_component_0(elem, side, q)
                                        : dirichlet_component_1(elem, side, q))
                              : face_lm_value_at(c, q);
                      Real vector_normal = Real(0);
                      for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                        vector_normal += face_vector_phi_at(q, i, d) * face_normal_at(q, d);
                      value -= side_jxw(elem, side, q) * vector_normal * boundary_value;
                    }
                  }
                  else if (idx < p0)
                  {
                    const unsigned int i = idx - u0;
                    for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    {
                      Real normal_norm2 = Real(0);
                      Real q_normal = Real(0);
                      for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      {
                        normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                        q_normal += face_q_value_at(c, q, d) * face_normal_at(q, d);
                      }
                      const Real u_value = face_u_value_at(c, q);
                      const Real p_value = face_p_value_at(q);
                      const Real lm_value = face_lm_value_at(c, q);
                      const Real other_lm_value = face_other_lm_value_at(c, q);
                      const Real boundary_value =
                          is_dirichlet
                              ? (c == 0 ? dirichlet_component_0(elem, side, q)
                                        : dirichlet_component_1(elem, side, q))
                              : lm_value;
                      const Real advective_vdotn =
                          velocity_component == 0
                              ? face_normal_at(q, 0) * lm_value +
                                    face_normal_at(q, 1) * other_lm_value
                              : face_normal_at(q, 0) * other_lm_value +
                                    face_normal_at(q, 1) * lm_value;
                      const Real advective_flux =
                          !include_convection ? Real(0)
                          : is_dirichlet      ? dirichlet_vdotn(elem, side, q) * boundary_value
                                              : advective_vdotn * lm_value;
                      const Real linear_flux =
                          is_outlet ? Real(0)
                                    : -nu * q_normal +
                                          p_value * face_normal_at(q, velocity_component) +
                                          tau * (u_value - boundary_value) * normal_norm2;
                      value += side_jxw(elem, side, q) * face_scalar_phi_at(q, i) *
                               (linear_flux + advective_flux);
                    }
                  }
                  else if (idx < lm0)
                  {
                    const unsigned int i = idx - p0;
                    for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    {
                      const Real boundary_value =
                          is_dirichlet
                              ? (c == 0 ? dirichlet_component_0(elem, side, q)
                                        : dirichlet_component_1(elem, side, q))
                              : face_lm_value_at(c, q);
                      value += side_jxw(elem, side, q) * face_scalar_phi_at(q, i) *
                               boundary_value * face_normal_at(q, velocity_component);
                    }
                  }
                  else
                  {
                    const unsigned int i = idx - lm0;
                    for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    {
                      const Real phi_i = face_trace_phi_at(q, i);
                      if (is_dirichlet)
                      {
                        value -= side_jxw(elem, side, q) * phi_i * face_lm_value_at(c, q);
                        continue;
                      }

                      Real normal_norm2 = Real(0);
                      Real q_normal = Real(0);
                      for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                      {
                        normal_norm2 += face_normal_at(q, d) * face_normal_at(q, d);
                        q_normal += face_q_value_at(c, q, d) * face_normal_at(q, d);
                      }
                      const Real lm_value = face_lm_value_at(c, q);
                      const Real other_lm_value = face_other_lm_value_at(c, q);
                      const Real advective_vdotn =
                          velocity_component == 0
                              ? face_normal_at(q, 0) * lm_value +
                                    face_normal_at(q, 1) * other_lm_value
                              : face_normal_at(q, 0) * other_lm_value +
                                    face_normal_at(q, 1) * lm_value;
                      const Real advective_flux =
                          (!include_convection || is_outlet) ? Real(0) : advective_vdotn * lm_value;
                      value += side_jxw(elem, side, q) * phi_i *
                               (-nu * q_normal +
                                face_p_value_at(q) * face_normal_at(q, velocity_component) +
                                tau * (face_u_value_at(c, q) - lm_value) * normal_norm2 +
                                advective_flux);
                    }
                  }

                  if (c == 0)
                    residual_0(elem, idx) += value;
                  else
                    residual_1(elem, idx) += value;
                });
            team.team_barrier();
          }

          if (add_full_convective_jacobian && include_convection && !is_dirichlet)
          {
            ::Kokkos::parallel_for(
                ::Kokkos::TeamThreadRange(team, layout.scalar_n_dofs * layout.lm_n_dofs),
                [&](const int raw_idx)
                {
                  const unsigned int idx = static_cast<unsigned int>(raw_idx);
                  const unsigned int i = idx / layout.lm_n_dofs;
                  const unsigned int j = idx - i * layout.lm_n_dofs;
                  Real u_lmu = Real(0);
                  Real u_lmv = Real(0);
                  Real v_lmu = Real(0);
                  Real v_lmv = Real(0);

                  for (unsigned int q = 0; q != n_side_qpoints; ++q)
                  {
                    const Real phi_i = face_scalar_phi_at(q, i);
                    const Real trace_j = face_trace_phi_at(q, j);
                    const Real lm_u_value = face_lm_value_at(0, q);
                    const Real lm_v_value = face_lm_value_at(1, q);
                    const Real normal_x_q = face_normal_at(q, 0);
                    const Real normal_y_q = face_normal_at(q, 1);

                    u_lmu += side_jxw(elem, side, q) * phi_i *
                             (Real(2) * normal_x_q * lm_u_value +
                              normal_y_q * lm_v_value) *
                             trace_j;
                    u_lmv += side_jxw(elem, side, q) * phi_i * normal_y_q * lm_u_value *
                             trace_j;
                    v_lmu += side_jxw(elem, side, q) * phi_i * normal_x_q * lm_v_value *
                             trace_j;
                    v_lmv += side_jxw(elem, side, q) * phi_i *
                             (normal_x_q * lm_u_value +
                              Real(2) * normal_y_q * lm_v_value) *
                             trace_j;
                  }

                  full_convective_block(full_off_u + i, full_off_lm_u + j) += u_lmu;
                  full_convective_block(full_off_u + i, full_off_lm_v + j) += u_lmv;
                  full_convective_block(full_off_v + i, full_off_lm_u + j) += v_lmu;
                  full_convective_block(full_off_v + i, full_off_lm_v + j) += v_lmv;
                });
            team.team_barrier();

            if (!is_outlet)
            {
              ::Kokkos::parallel_for(
                  ::Kokkos::TeamThreadRange(team, layout.lm_n_dofs * layout.lm_n_dofs),
                  [&](const int raw_idx)
                  {
                    const unsigned int idx = static_cast<unsigned int>(raw_idx);
                    const unsigned int i = idx / layout.lm_n_dofs;
                    const unsigned int j = idx - i * layout.lm_n_dofs;
                    Real lmu_lmu = Real(0);
                    Real lmu_lmv = Real(0);
                    Real lmv_lmu = Real(0);
                    Real lmv_lmv = Real(0);

                    for (unsigned int q = 0; q != n_side_qpoints; ++q)
                    {
                      const Real trace_i = face_trace_phi_at(q, i);
                      const Real trace_j = face_trace_phi_at(q, j);
                      const Real lm_u_value = face_lm_value_at(0, q);
                      const Real lm_v_value = face_lm_value_at(1, q);
                      const Real normal_x_q = face_normal_at(q, 0);
                      const Real normal_y_q = face_normal_at(q, 1);

                      lmu_lmu += side_jxw(elem, side, q) * trace_i *
                                 (Real(2) * normal_x_q * lm_u_value +
                                  normal_y_q * lm_v_value) *
                                 trace_j;
                      lmu_lmv += side_jxw(elem, side, q) * trace_i * normal_y_q *
                                 lm_u_value * trace_j;
                      lmv_lmu += side_jxw(elem, side, q) * trace_i * normal_x_q *
                                 lm_v_value * trace_j;
                      lmv_lmv += side_jxw(elem, side, q) * trace_i *
                                 (normal_x_q * lm_u_value +
                                  Real(2) * normal_y_q * lm_v_value) *
                                 trace_j;
                    }

                    full_convective_block(full_off_lm_u + i, full_off_lm_u + j) += lmu_lmu;
                    full_convective_block(full_off_lm_u + i, full_off_lm_v + j) += lmu_lmv;
                    full_convective_block(full_off_lm_v + i, full_off_lm_u + j) += lmv_lmu;
                    full_convective_block(full_off_lm_v + i, full_off_lm_v + j) += lmv_lmv;
                  });
              team.team_barrier();
            }
          }
        }

        if (team.team_rank() == 0)
        {
          condense_status_0(elem) = 0;
          residual_status_0(elem) = 0;
          condense_status_1(elem) = 0;
          residual_status_1(elem) = 0;
        }
      });
}

template <libMesh::ElemType ExactTopo,
          typename NodeStorage,
          typename VolumeQpStorage,
          typename SideQpStorage,
          typename OrientationStorage,
          typename DofStorage,
          typename BlockStorage,
          typename ResidualStorage,
          typename CondensedStorage,
          typename StatusStorage>
void
assemble_hdg_linear_batch(const libMesh::FEShapeKey vector_key,
                          const libMesh::FEShapeKey scalar_key,
                          const libMesh::FEShapeKey lm_key,
                          const HDGLinearBlockLayout layout,
                          const NodeStorage & coords,
                          const unsigned int n_elements,
                          const unsigned int n_geom_nodes,
                          const VolumeQpStorage & xi,
                          const VolumeQpStorage & eta,
                          const VolumeQpStorage & zeta,
                          const VolumeQpStorage & jxw,
                          const unsigned int n_qpoints,
                          const SideQpStorage & parent_xi,
                          const SideQpStorage & parent_eta,
                          const SideQpStorage & parent_zeta,
                          const SideQpStorage & side_jxw,
                          const SideQpStorage & normal_x,
                          const SideQpStorage & normal_y,
                          const SideQpStorage & normal_z,
                          const OrientationStorage & positive_edge_orientations,
                          const unsigned int n_sides,
                          const unsigned int n_side_qpoints,
                          const Real nu,
                          const Real tau,
                          const DofStorage & q_dofs,
                          const DofStorage & u_dofs,
                          const DofStorage & p_dofs,
                          const DofStorage & lm_dofs,
                          const BlockStorage & blocks,
                          const ResidualStorage & residual,
                          const CondensedStorage & condensed,
                          const CondensedStorage & condensed_residual,
                          const StatusStorage & condense_status,
                          const StatusStorage & residual_status,
                          const char * const kernel_name,
                          const unsigned int velocity_component = 0)
{
  using ExecutionSpace = typename BlockStorage::execution_space;
  ::Kokkos::View<int **, ExecutionSpace> side_types(
      "hdg_linear_batch_default_side_types", n_elements, n_sides);
  ::Kokkos::View<Real ***, ExecutionSpace> dirichlet_component(
      "hdg_linear_batch_default_dirichlet_component", n_elements, n_sides, n_side_qpoints);
  ::Kokkos::View<Real ***, ExecutionSpace> dirichlet_vdotn(
      "hdg_linear_batch_default_dirichlet_vdotn", n_elements, n_sides, n_side_qpoints);
  ::Kokkos::View<Real **, ExecutionSpace> scalar_forcing(
      "hdg_linear_batch_default_scalar_forcing", n_elements, n_qpoints);
  ::Kokkos::View<Real **, ExecutionSpace> pressure_forcing(
      "hdg_linear_batch_default_pressure_forcing", n_elements, n_qpoints);
  ::Kokkos::deep_copy(side_types, hdg_linear_side_interior);
  ::Kokkos::deep_copy(dirichlet_component, Real(0));
  ::Kokkos::deep_copy(dirichlet_vdotn, Real(0));
  ::Kokkos::deep_copy(scalar_forcing, Real(0));
  ::Kokkos::deep_copy(pressure_forcing, Real(0));

  assemble_hdg_linear_batch_boundary<ExactTopo>(vector_key,
                                                scalar_key,
                                                lm_key,
                                                layout,
                                                coords,
                                                n_elements,
                                                n_geom_nodes,
                                                xi,
                                                eta,
                                                zeta,
                                                jxw,
                                                scalar_forcing,
                                                pressure_forcing,
                                                n_qpoints,
                                                parent_xi,
                                                parent_eta,
                                                parent_zeta,
                                                side_jxw,
                                                normal_x,
                                                normal_y,
                                                normal_z,
                                                positive_edge_orientations,
                                                side_types,
                                                dirichlet_component,
                                                dirichlet_vdotn,
                                                n_sides,
                                                n_side_qpoints,
                                                nu,
                                                tau,
                                                false,
                                                q_dofs,
                                                u_dofs,
                                                p_dofs,
                                                lm_dofs,
                                                u_dofs,
                                                lm_dofs,
                                                blocks,
                                                residual,
                                                condensed,
                                                condensed_residual,
                                                condense_status,
                                                residual_status,
                                                kernel_name,
                                                velocity_component);
}

} // namespace libMesh::Kokkos::detail

#endif // LIBMESH_KOKKOS_HDG_ASSEMBLY_H
