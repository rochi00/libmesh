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

// (family, element class, order) shape-key queries shared between host FE
// code and Kokkos kernels: which basis topology evaluates a key, how many
// dofs it carries, and which keys the device evaluators support.

#ifndef LIBMESH_FE_SHAPE_TRAITS_H
#define LIBMESH_FE_SHAPE_TRAITS_H

#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_fe_elem_class.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/fe_reference_element_traits.h"
#include "libmesh/libmesh_device.h"

namespace libMesh
{

struct FEShapeKey
{
  FEFamily family;
  ElemType elem_type;
  Order    order;
};

LIBMESH_DEVICE_INLINE bool
is_monomial_2d_elem_type(ElemType elem_type)
{
  switch (elem_type)
  {
    case C0POLYGON:
    case TRI3:
    case TRISHELL3:
    case TRI6:
    case TRI7:
    case QUAD4:
    case QUADSHELL4:
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
      return true;
    default:
      return false;
  }
}

LIBMESH_DEVICE_INLINE bool
is_monomial_3d_elem_type(ElemType elem_type,
                         bool include_pyramid18 = true)
{
  switch (elem_type)
  {
    case TET4:
    case TET10:
    case TET14:
    case HEX8:
    case HEX20:
    case HEX27:
    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case C0POLYHEDRON:
      return true;
    case PYRAMID18:
      return include_pyramid18;
    default:
      return false;
  }
}

// The single topology shared by every side of a uniform-sided element;
// INVALID_ELEM for the mixed-face prisms and pyramids.
LIBMESH_DEVICE_INLINE ElemType
side_topology_or_invalid(ElemType parent)
{
  return requires_side_specific_topology(parent)
    ? INVALID_ELEM
    : side_topology_or_invalid(parent, 0);
}

LIBMESH_DEVICE_INLINE FEElemClass
class_from_topology_or_invalid(ElemType topo)
{
  switch (topo)
  {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      return FEElemClass::EDGE;

    case TRI3:
    case TRI6:
    case TRI7:
      return FEElemClass::TRI;

    case QUAD4:
    case QUAD8:
    case QUAD9:
      return FEElemClass::QUAD;

    case TET4:
    case TET10:
    case TET14:
      return FEElemClass::TET;

    case HEX8:
    case HEX20:
    case HEX27:
      return FEElemClass::HEX;

    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
      return FEElemClass::PRISM;

    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      return FEElemClass::PYRAMID;

    default:
      return FEElemClass::N_CLASSES;
  }
}

LIBMESH_DEVICE_INLINE unsigned int
elem_class_dim_or_zero(FEElemClass cls)
{
  switch (cls)
  {
    case FEElemClass::EDGE:
      return 1;
    case FEElemClass::TRI:
    case FEElemClass::QUAD:
      return 2;
    case FEElemClass::TET:
    case FEElemClass::HEX:
    case FEElemClass::PRISM:
    case FEElemClass::PYRAMID:
      return 3;
    default:
      return 0;
  }
}

LIBMESH_DEVICE_INLINE unsigned int
topology_dim_or_zero(ElemType topo)
{
  return elem_class_dim_or_zero(class_from_topology_or_invalid(topo));
}

LIBMESH_DEVICE_INLINE constexpr ElemType
lagrange_shape_topology_or_invalid(FEShapeKey key)
{
  switch (key.order)
  {
    case CONSTANT:
    case FIRST:
      switch (key.elem_type)
      {
        case EDGE2:
        case EDGE3:
        case EDGE4:
          return EDGE2;

        case TRI3:
        case TRI6:
        case TRI7:
          return TRI3;

        case QUAD4:
        case QUAD8:
        case QUAD9:
          return QUAD4;

        case TET4:
        case TET10:
        case TET14:
          return TET4;

        case HEX8:
        case HEX20:
        case HEX27:
          return HEX8;

        default:
          return INVALID_ELEM;
      }

    case SECOND:
      switch (key.elem_type)
      {
        case EDGE3:
          return EDGE3;

        case TRI6:
        case TRI7:
          return TRI6;

        case QUAD8:
          return QUAD8;

        case QUAD9:
          return QUAD9;

        case TET10:
        case TET14:
          return TET10;

        case HEX20:
          return HEX20;

        case HEX27:
          return HEX27;

        default:
          return INVALID_ELEM;
      }

    case THIRD:
      switch (key.elem_type)
      {
        case EDGE4:
          return EDGE4;

        default:
          return INVALID_ELEM;
      }

    default:
      return INVALID_ELEM;
  }
}

LIBMESH_DEVICE_INLINE unsigned int
lagrange_exact_n_dofs_or_zero(ElemType elem_type,
                              Order order)
{
  switch (order)
  {
    case CONSTANT:
      return (elem_type == NODEELEM) ? 1u : 0u;

    case FIRST:
      switch (elem_type)
      {
        case NODEELEM:
          return 1;
        case TRISHELL3:
        case QUADSHELL4:
        case QUADSHELL8:
        case QUADSHELL9:
          return 0;  // no shell support in the Lagrange evaluators
        default:
          return reference_vertex_count(elem_type);
      }

    case SECOND:
      switch (elem_type)
      {
        case NODEELEM:
          return 1;

        case EDGE3:
          return 3;

        case TRI6:
        case TRI7:
          return 6;

        case QUAD8:
          return 8;

        case QUAD9:
          return 9;

        case TET10:
        case TET14:
          return 10;

        case HEX20:
          return 20;

        case HEX27:
          return 27;

        case PRISM15:
          return 15;

        case PRISM18:
        case PRISM20:
        case PRISM21:
          return 18;

        case PYRAMID13:
          return 13;

        case PYRAMID14:
        case PYRAMID18:
          return 14;

        default:
          return 0;
      }

    case THIRD:
      switch (elem_type)
      {
        case NODEELEM:
          return 1;

        case EDGE4:
          return 4;

        case TRI7:
          return 7;

        case TET14:
          return 14;

        case PRISM20:
          return 20;

        case PRISM21:
          return 21;

        case PYRAMID18:
          return 18;

        default:
          return 0;
      }

    default:
      return 0;
  }
}

// The monomial basis is the complete polynomial space, so the count is the
// dimension-d binomial; EDGE4 and PYRAMID18 evaluators stop at THIRD.
LIBMESH_DEVICE_INLINE unsigned int
monomial_exact_n_dofs_or_zero(ElemType elem_type,
                              Order order)
{
  if (elem_type == INVALID_ELEM || order < CONSTANT)
    return 0;
  if (order == CONSTANT)
    return 1;

  const unsigned int p = static_cast<unsigned int>(order);
  switch (elem_type)
  {
    case NODEELEM:
      return 1;
    case EDGE2:
    case EDGE3:
      return p + 1;
    case EDGE4:
      return order <= THIRD ? p + 1 : 0;
    default:
      break;
  }
  if (is_monomial_2d_elem_type(elem_type))
    return (p + 1) * (p + 2) / 2;
  if (is_monomial_3d_elem_type(elem_type, order <= THIRD))
    return (p + 1) * (p + 2) * (p + 3) / 6;
  return 0;
}

LIBMESH_DEVICE_INLINE constexpr unsigned int
monomial_evaluator_dim_or_zero(ElemType elem_type)
{
  switch (elem_type)
  {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      return 1;

    case TRI3:
    case TRI6:
    case TRI7:
    case QUAD4:
    case QUAD8:
    case QUAD9:
      return 2;

    case TET4:
    case TET10:
    case TET14:
    case HEX8:
    case HEX20:
    case HEX27:
    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      return 3;

    default:
      return 0;
  }
}

LIBMESH_DEVICE_INLINE unsigned int
side_hierarchic_trace_n_dofs_or_zero(ElemType elem_type,
                                     Order order)
{
  if (order != FIRST)
    return 0;

  switch (elem_type)
  {
    case TRI6:
    case TRI7:
      return 6;

    case QUAD8:
    case QUAD9:
      return 8;

    default:
      return 0;
  }
}

LIBMESH_DEVICE_INLINE bool
supports_shape(FEShapeKey key);

LIBMESH_DEVICE_INLINE bool
supports_vector_shape(FEShapeKey key);

LIBMESH_DEVICE_INLINE bool
supports_side_trace_shape(FEShapeKey key)
{
  return key.family == SIDE_HIERARCHIC &&
         side_hierarchic_trace_n_dofs_or_zero(key.elem_type, key.order) != 0;
}

LIBMESH_DEVICE_INLINE unsigned int
vector_component_count_or_zero(ElemType elem_type)
{
  if (elem_type == NODEELEM)
    return 1;

  return topology_dim_or_zero(elem_type);
}

LIBMESH_DEVICE_INLINE bool
supports_lagrange_map_topology(ElemType topo)
{
  switch (topo)
  {
    case EDGE2:
    case EDGE3:
    case EDGE4:
    case TRI3:
    case TRI6:
    case TRI7:
    case QUAD4:
    case QUAD8:
    case QUAD9:
    case TET4:
    case TET10:
    case HEX8:
    case HEX20:
    case HEX27:
      return true;

    default:
      return false;
  }
}

LIBMESH_DEVICE_INLINE bool
supports_lagrange_face_map_topology(ElemType topo)
{
  return supports_lagrange_map_topology(topo);
}

template <typename Op, typename Unsupported>
LIBMESH_DEVICE_INLINE auto
dispatch_lagrange_map_topology_or(ElemType topo,
                                  const Op & op,
                                  const Unsupported & unsupported)
  -> decltype(op.template operator()<EDGE2>())
{
  switch (topo)
  {
    case EDGE2:
      return op.template operator()<EDGE2>();
    case EDGE3:
      return op.template operator()<EDGE3>();
    case EDGE4:
      return op.template operator()<EDGE4>();
    case TRI3:
      return op.template operator()<TRI3>();
    case TRI6:
      return op.template operator()<TRI6>();
    case TRI7:
      return op.template operator()<TRI7>();
    case QUAD4:
      return op.template operator()<QUAD4>();
    case QUAD8:
      return op.template operator()<QUAD8>();
    case QUAD9:
      return op.template operator()<QUAD9>();
    case TET4:
      return op.template operator()<TET4>();
    case TET10:
      return op.template operator()<TET10>();
    case HEX8:
      return op.template operator()<HEX8>();
    case HEX20:
      return op.template operator()<HEX20>();
    case HEX27:
      return op.template operator()<HEX27>();
    default:
      return unsupported(topo);
  }
}

LIBMESH_DEVICE_INLINE bool
supports_shape_with_lagrange_map(FEShapeKey key)
{
  return supports_shape(key) &&
         supports_lagrange_map_topology(key.elem_type);
}

LIBMESH_DEVICE_INLINE bool
supports_vector_shape_with_lagrange_map(FEShapeKey key)
{
  return supports_vector_shape(key) &&
         supports_lagrange_map_topology(key.elem_type);
}

LIBMESH_DEVICE_INLINE bool
supports_shape(FEShapeKey key)
{
  switch (key.family)
  {
    case LAGRANGE:
    case L2_LAGRANGE:
      return lagrange_exact_n_dofs_or_zero(key.elem_type, key.order) != 0 &&
             lagrange_shape_topology_or_invalid(key) != INVALID_ELEM;

    case MONOMIAL:
      return monomial_exact_n_dofs_or_zero(key.elem_type, key.order) != 0 &&
             monomial_evaluator_dim_or_zero(key.elem_type) != 0 &&
             key.order >= CONSTANT &&
             key.order <= FIFTH;

    default:
      return false;
  }
}

LIBMESH_DEVICE_INLINE bool
supports_vector_shape(FEShapeKey key)
{
  switch (key.family)
  {
    case LAGRANGE_VEC:
    case L2_LAGRANGE_VEC:
      return vector_component_count_or_zero(key.elem_type) != 0 &&
             lagrange_exact_n_dofs_or_zero(key.elem_type, key.order) != 0 &&
             lagrange_shape_topology_or_invalid(key) != INVALID_ELEM;

    default:
      return false;
  }
}

LIBMESH_DEVICE_INLINE bool
supports_grad_shape(FEShapeKey key)
{
  return supports_shape(key);
}

LIBMESH_DEVICE_INLINE bool
supports_vector_shape_deriv(FEShapeKey key)
{
  return supports_vector_shape(key);
}

LIBMESH_DEVICE_INLINE bool
supports_n_dofs(FEShapeKey key)
{
  return supports_shape(key) || supports_vector_shape(key) || supports_side_trace_shape(key);
}

LIBMESH_DEVICE_INLINE unsigned int
n_dofs_or_zero(FEShapeKey key)
{
  switch (key.family)
  {
    case LAGRANGE:
    case L2_LAGRANGE:
      return lagrange_exact_n_dofs_or_zero(key.elem_type, key.order);

    case LAGRANGE_VEC:
    case L2_LAGRANGE_VEC:
      return vector_component_count_or_zero(key.elem_type) *
             lagrange_exact_n_dofs_or_zero(key.elem_type, key.order);

    case MONOMIAL:
      return monomial_exact_n_dofs_or_zero(key.elem_type, key.order);

    case SIDE_HIERARCHIC:
      return side_hierarchic_trace_n_dofs_or_zero(key.elem_type, key.order);

    default:
      return 0;
  }
}

} // namespace libMesh

#endif // LIBMESH_FE_SHAPE_TRAITS_H
