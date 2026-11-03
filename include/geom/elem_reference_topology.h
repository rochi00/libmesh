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

#ifndef LIBMESH_ELEM_REFERENCE_TOPOLOGY_H
#define LIBMESH_ELEM_REFERENCE_TOPOLOGY_H

// Elem's static lookups by element type read the element classes'
// constexpr tables, so they are defined here, after those classes are
// complete, rather than in elem.h, which the classes include.  They are
// inline so that device code, which has only an element type, can call
// them too.

#include "libmesh/libmesh_device.h"
#include "libmesh/elem.h"
#include "libmesh/face_tri.h"
#include "libmesh/face_quad.h"
#include "libmesh/cell_tet.h"
#include "libmesh/cell_hex.h"
#include "libmesh/cell_prism.h"
#include "libmesh/cell_pyramid.h"

namespace libMesh
{

/**
 * A copy of a constexpr table that device code can index.
 *
 * CUDA lets device code read a host static constexpr array only in a
 * constant expression, not at an index known only at run time.
 * Copying the array into a constexpr local with copy_table() is a
 * constant expression, and the local is then indexed like any other.
 */
template <typename V, unsigned int N>
struct TableCopy
{
  V values[N];
};

template <typename V, unsigned int N>
LIBMESH_DEVICE_INLINE constexpr
TableCopy<V, N> copy_table (const V (&a)[N])
{
  TableCopy<V, N> t {};
  for (unsigned int i = 0; i != N; ++i)
    t.values[i] = a[i];
  return t;
}

// Arrays can't be assigned whole, so a table of rows is copied entry by
// entry.  Partial ordering prefers this overload for two-dimensional
// arrays.
template <typename V, unsigned int Rows, unsigned int Cols>
LIBMESH_DEVICE_INLINE constexpr
TableCopy<V[Cols], Rows> copy_table (const V (&a)[Rows][Cols])
{
  TableCopy<V[Cols], Rows> t {};
  for (unsigned int i = 0; i != Rows; ++i)
    for (unsigned int j = 0; j != Cols; ++j)
      t.values[i][j] = a[i][j];
  return t;
}



LIBMESH_DEVICE_INLINE
Point Elem::master_point (const ElemType t,
                          const unsigned int i)
{
#ifndef NDEBUG
  constexpr TableCopy<unsigned int, INVALID_ELEM> n_nodes =
    copy_table(type_to_n_nodes_map);
  libmesh_assert_less (i, n_nodes.values[t]);
#endif

  // The element types that share a reference element share its table,
  // whose rows are numbered as the most nodes any of them has
  switch (t)
    {
    case NODEELEM:
      return Point(0, 0, 0);
      // As Edge2::master_point(), Edge3::master_point() and
      // Edge4::master_point(): the vertices at -1 and 1, then the
      // midpoint or the thirds
    case EDGE2:
    case EDGE3:
    case EDGE4:
      if (i < 2)
        return Point(2.0f*Real(i)-1.0f, 0, 0);
      if (t == EDGE3)
        return Point(0, 0, 0);
      return Point((Real(2)*Real(i)-5)/3, 0, 0);
    case TRI3:
    case TRISHELL3:
    case TRI6:
    case TRI7:
      {
        constexpr TableCopy<Real[3], 7> p = copy_table(Tri::_master_points);
        return Point(p.values[i][0], p.values[i][1], p.values[i][2]);
      }
    case QUAD4:
    case QUADSHELL4:
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
      {
        constexpr TableCopy<Real[3], 9> p = copy_table(Quad::_master_points);
        return Point(p.values[i][0], p.values[i][1], p.values[i][2]);
      }
    case TET4:
    case TET10:
    case TET14:
      {
        constexpr TableCopy<Real[3], 14> p = copy_table(Tet::_master_points);
        return Point(p.values[i][0], p.values[i][1], p.values[i][2]);
      }
    case HEX8:
    case HEX20:
    case HEX27:
      {
        constexpr TableCopy<Real[3], 27> p = copy_table(Hex::_master_points);
        return Point(p.values[i][0], p.values[i][1], p.values[i][2]);
      }
    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
      {
        constexpr TableCopy<Real[3], 21> p = copy_table(Prism::_master_points);
        return Point(p.values[i][0], p.values[i][1], p.values[i][2]);
      }
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      {
        constexpr TableCopy<Real[3], 18> p = copy_table(Pyramid::_master_points);
        return Point(p.values[i][0], p.values[i][1], p.values[i][2]);
      }
    default:
      libmesh_error_msg("ERROR: Unknown element type " << Utility::enum_to_string(t));
    }

  return Point();
}

} // namespace libMesh

#endif // LIBMESH_ELEM_REFERENCE_TOPOLOGY_H
