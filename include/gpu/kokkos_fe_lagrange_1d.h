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

// Kokkos FEEvaluator specializations for 1-D Lagrange elements.
//
// Covers EDGE2 (linear), EDGE3 (quadratic), and EDGE4 (cubic).
// Reference-element coordinate convention (libMesh-compatible):
//   EDGE2/EDGE3: xi in [-1, 1]
//
// EDGE3 node ordering (libMesh non-sequential):
//   index 0 -> xi = -1   (left node)
//   index 1 -> xi = +1   (right node)
//   index 2 -> xi =  0   (midpoint)

#ifndef LIBMESH_KOKKOS_FE_LAGRANGE_1D_H
#define LIBMESH_KOKKOS_FE_LAGRANGE_1D_H

#include "kokkos_fe_base.h"
#include "libmesh/fe_lagrange_shape_1D.h"

namespace libMesh::Kokkos
{

// ── EDGE2 (linear edge, 2 nodes) ─────────────────────────────────────────────

template <>
struct FEEvaluator<libMesh::LAGRANGE, libMesh::EDGE2>
{
  static constexpr unsigned int n_dofs() { return 2; }

#ifdef LIBMESH_HAVE_KOKKOS
  LIBMESH_DEVICE_INLINE static Real
  shape(unsigned int i, Real xi, Real /*eta*/, Real /*zeta*/)
  {
    return libMesh::fe_lagrange_1D_linear_shape(i, xi);
  }

  LIBMESH_DEVICE_INLINE static RealVector
  grad_shape(unsigned int i, Real xi, Real /*eta*/, Real /*zeta*/)
  {
    return make_vector(libMesh::fe_lagrange_1D_linear_shape_deriv(i, 0, xi), 0.0, 0.0);
  }
#endif
};

// ── EDGE3 (quadratic edge, 3 nodes) ──────────────────────────────────────────
// Node ordering matches libMesh: 0->left(-1), 1->right(+1), 2->mid(0)
//   L_0(xi) = 0.5*xi*(xi-1)   dL_0/dxi = xi - 0.5
//   L_1(xi) = 0.5*xi*(xi+1)   dL_1/dxi = xi + 0.5
//   L_2(xi) = 1 - xi²         dL_2/dxi = -2*xi

template <>
struct FEEvaluator<libMesh::LAGRANGE, libMesh::EDGE3>
{
  static constexpr unsigned int n_dofs() { return 3; }

#ifdef LIBMESH_HAVE_KOKKOS
  LIBMESH_DEVICE_INLINE static Real
  shape(unsigned int i, Real xi, Real /*eta*/, Real /*zeta*/)
  {
    return libMesh::fe_lagrange_1D_quadratic_shape(i, xi);
  }

  LIBMESH_DEVICE_INLINE static RealVector
  grad_shape(unsigned int i, Real xi, Real /*eta*/, Real /*zeta*/)
  {
    return make_vector(libMesh::fe_lagrange_1D_quadratic_shape_deriv(i, 0, xi), 0.0, 0.0);
  }
#endif
};

// ── EDGE4 (cubic edge, 4 nodes) ──────────────────────────────────────────────
// Node ordering matches libMesh: 0->left(-1), 1->right(+1), 2->(-1/3), 3->(+1/3)

template <>
struct FEEvaluator<libMesh::LAGRANGE, libMesh::EDGE4>
{
  static constexpr unsigned int n_dofs() { return 4; }

#ifdef LIBMESH_HAVE_KOKKOS
  LIBMESH_DEVICE_INLINE static Real
  shape(unsigned int i, Real xi, Real /*eta*/, Real /*zeta*/)
  {
    return libMesh::fe_lagrange_1D_cubic_shape(i, xi);
  }

  LIBMESH_DEVICE_INLINE static RealVector
  grad_shape(unsigned int i, Real xi, Real /*eta*/, Real /*zeta*/)
  {
    return make_vector(libMesh::fe_lagrange_1D_cubic_shape_deriv(i, 0, xi), 0.0, 0.0);
  }
#endif
};

} // namespace libMesh::Kokkos

#endif // LIBMESH_KOKKOS_FE_LAGRANGE_1D_H
