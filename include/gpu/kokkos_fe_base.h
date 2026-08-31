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

// Primary FEEvaluator template for Kokkos device-compatible shape functions.
//
// Uses libMesh's own ElemType and FEFamily enums as non-type template
// parameters — no separate tag structs are needed.
//
// All uses must be explicit specializations defined in the kokkos_fe_lagrange_*.h
// and kokkos_fe_monomial.h headers.  Every specialization must provide:
//
//   static constexpr unsigned int n_dofs()
//
//   LIBMESH_DEVICE_INLINE
//   static Real shape(unsigned int i, Real xi, Real eta, Real zeta)
//
//   LIBMESH_DEVICE_INLINE
//   static RealVector grad_shape(unsigned int i, Real xi, Real eta, Real zeta)
//
// Reference-element coordinate conventions (matching libMesh):
//   Edge:  xi  in [-1, 1]
//   Quad:  (xi, eta)          in [-1,1]^2
//   Hex:   (xi, eta, zeta)    in [-1,1]^3
//   Tri:   (xi, eta)          in unit triangle, xi >= 0, eta >= 0, xi+eta <= 1
//   Tet:   (xi, eta, zeta)    in unit tetrahedron
//
// Unused coordinate arguments (e.g. zeta on a 2D element) are accepted but
// ignored, so call sites can always pass all three without special-casing.
//
#ifndef LIBMESH_KOKKOS_FE_BASE_H
#define LIBMESH_KOKKOS_FE_BASE_H

#include "libmesh/libmesh_device.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/kokkos_tensor_ops.h"
#include "libmesh/kokkos_vector_ops.h"
#include "libmesh/type_tensor.h"
#include "libmesh/type_vector.h"

namespace libMesh::Kokkos
{

using Real = libMesh::Real;
using RealVector = libMesh::TypeVector<Real>;
using RealTensor = libMesh::TypeTensor<Real>;

LIBMESH_DEVICE_INLINE
RealVector zero_vector()
{
  return RealVector();
}

LIBMESH_DEVICE_INLINE
RealVector make_vector(const Real x, const Real y = 0, const Real z = 0)
{
  RealVector v = zero_vector();

  v(0) = x;

#if LIBMESH_DIM > 1
  v(1) = y;
#else
  libmesh_assert_equal_to(y, Real(0));
#endif

#if LIBMESH_DIM > 2
  v(2) = z;
#else
  libmesh_assert_equal_to(z, Real(0));
#endif

  return v;
}

LIBMESH_DEVICE_INLINE
RealTensor zero_tensor()
{
  return RealTensor();
}

template <libMesh::FEFamily family, libMesh::ElemType elem_type, unsigned int Order = 0>
struct FEEvaluator; // forward declaration only; instantiation requires a specialization

} // namespace libMesh::Kokkos

#endif // LIBMESH_KOKKOS_FE_BASE_H
