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
#include "libmesh/face_tri3.h"
#include "libmesh/face_tri6.h"
#include "libmesh/face_tri7.h"
#include "libmesh/face_quad4.h"
#include "libmesh/face_quad8.h"
#include "libmesh/face_quad9.h"
#include "libmesh/cell_tet4.h"
#include "libmesh/cell_tet10.h"
#include "libmesh/cell_tet14.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/cell_hex20.h"
#include "libmesh/cell_hex27.h"
#include "libmesh/cell_prism6.h"
#include "libmesh/cell_prism15.h"
#include "libmesh/cell_prism18.h"
#include "libmesh/cell_prism20.h"
#include "libmesh/cell_prism21.h"
#include "libmesh/cell_pyramid5.h"
#include "libmesh/cell_pyramid13.h"
#include "libmesh/cell_pyramid14.h"
#include "libmesh/cell_pyramid18.h"

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



// Most of our elements have the same topology on every side, but the
// prisms and pyramids have both triangular and quadrilateral faces, and
// a 2D or 3D infinite element's side 0 is its finite base while the rest
// are infinite.  The polygons and polyhedra have one side type but no
// fixed side count, so they can answer here too while their side index
// goes unchecked; ask an actual Elem when you have one.
LIBMESH_DEVICE_INLINE
ElemType Elem::side_type (const ElemType t,
                          const unsigned int s)
{
#ifndef NDEBUG
  constexpr TableCopy<unsigned int, INVALID_ELEM> n_sides =
    copy_table(type_to_n_sides_map);
  libmesh_assert_less (s, n_sides.values[t]);
#endif

  switch (t)
    {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      return NODEELEM;
    case TRI3:
    case TRISHELL3:
    case QUAD4:
    case QUADSHELL4:
      // A polygon's sides are all edges and a polyhedron's are all
      // polygons, however many a given element turns out to have
    case C0POLYGON:
      return EDGE2;
    case TRI6:
    case TRI7:
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
      return EDGE3;
    case TET4:
      return TRI3;
    case TET10:
      return TRI6;
    case TET14:
      return TRI7;
    case HEX8:
      return QUAD4;
    case HEX20:
      return QUAD8;
    case HEX27:
      return QUAD9;
      // Sides 0 and 4 of a prism are its triangles
    case PRISM6:
      return (s == 0 || s == 4) ? TRI3 : QUAD4;
    case PRISM15:
      return (s == 0 || s == 4) ? TRI6 : QUAD8;
    case PRISM18:
      return (s == 0 || s == 4) ? TRI6 : QUAD9;
    case PRISM20:
    case PRISM21:
      return (s == 0 || s == 4) ? TRI7 : QUAD9;
      // Side 4 of a pyramid is its quadrilateral base
    case PYRAMID5:
      return (s < 4) ? TRI3 : QUAD4;
    case PYRAMID13:
      return (s < 4) ? TRI6 : QUAD8;
    case PYRAMID14:
      return (s < 4) ? TRI6 : QUAD9;
    case PYRAMID18:
      return (s < 4) ? TRI7 : QUAD9;
    case C0POLYHEDRON:
      return C0POLYGON;
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
      // An InfEdge2's sides are its end nodes, like any other 1D element's
    case INFEDGE2:
      return NODEELEM;
      // A 2D or 3D infinite element's side 0 is the finite base it was
      // built from; its remaining sides run out to infinity with it
    case INFQUAD4:
      return (s == 0) ? EDGE2 : INFEDGE2;
    case INFQUAD6:
      return (s == 0) ? EDGE3 : INFEDGE2;
    case INFHEX8:
      return (s == 0) ? QUAD4 : INFQUAD4;
    case INFHEX16:
      return (s == 0) ? QUAD8 : INFQUAD6;
    case INFHEX18:
      return (s == 0) ? QUAD9 : INFQUAD6;
    case INFPRISM6:
      return (s == 0) ? TRI3 : INFQUAD4;
    case INFPRISM12:
      return (s == 0) ? TRI6 : INFQUAD6;
#endif
    default:
      libmesh_error_msg("No side type for element type " << Utility::enum_to_string(t));
    }

  return INVALID_ELEM;
}



// This reads the same constexpr side_nodes_map tables the element
// classes use for their virtual local_side_node(), so that code which
// knows an element type but has no Elem to call a virtual function on --
// building a side of an element that doesn't exist yet, or a device
// kernel that has only the type -- can still get at the reference
// element topology.
//
// The polygons and polyhedra have no such table to read, and the
// infinite elements' maps have no caller here yet, so both error out:
// asking for a topology we can't answer is a programming error rather
// than something to signal with a return value.
LIBMESH_DEVICE_INLINE
unsigned int Elem::local_side_node (const ElemType t,
                                    const unsigned int side,
                                    const unsigned int side_node)
{
  // The nodes on a side are the nodes of the side's own element type, so
  // we compose the two lookups here rather than tabulating side node
  // counts a second time.  That also keeps us inside the meaningful part
  // of a row: the prisms' and pyramids' triangular sides carry fewer
  // nodes than their quadrilateral ones, and their rows are padded out
  // to the longer length
#ifndef NDEBUG
  constexpr TableCopy<unsigned int, INVALID_ELEM> n_sides =
    copy_table(type_to_n_sides_map);
  constexpr TableCopy<unsigned int, INVALID_ELEM> n_nodes =
    copy_table(type_to_n_nodes_map);
  libmesh_assert_less (side, n_sides.values[t]);
  libmesh_assert_less (side_node, n_nodes.values[side_type(t, side)]);
#endif

  // Each table is copied as the comment on copy_table() explains, then
  // indexed
  switch (t)
    {
      // A 1D element's sides are its end nodes
    case EDGE2:
    case EDGE3:
    case EDGE4:
      return side;
      // The shell elements are numbered like the elements they shadow
    case TRI3:
    case TRISHELL3:
      { constexpr auto m = copy_table(Tri3::side_nodes_map); return m.values[side][side_node]; }
    case TRI6:
      { constexpr auto m = copy_table(Tri6::side_nodes_map); return m.values[side][side_node]; }
    case TRI7:
      { constexpr auto m = copy_table(Tri7::side_nodes_map); return m.values[side][side_node]; }
    case QUAD4:
    case QUADSHELL4:
      { constexpr auto m = copy_table(Quad4::side_nodes_map); return m.values[side][side_node]; }
    case QUAD8:
    case QUADSHELL8:
      { constexpr auto m = copy_table(Quad8::side_nodes_map); return m.values[side][side_node]; }
    case QUAD9:
    case QUADSHELL9:
      { constexpr auto m = copy_table(Quad9::side_nodes_map); return m.values[side][side_node]; }
    case TET4:
      { constexpr auto m = copy_table(Tet4::side_nodes_map); return m.values[side][side_node]; }
    case TET10:
      { constexpr auto m = copy_table(Tet10::side_nodes_map); return m.values[side][side_node]; }
    case TET14:
      { constexpr auto m = copy_table(Tet14::side_nodes_map); return m.values[side][side_node]; }
    case HEX8:
      { constexpr auto m = copy_table(Hex8::side_nodes_map); return m.values[side][side_node]; }
    case HEX20:
      { constexpr auto m = copy_table(Hex20::side_nodes_map); return m.values[side][side_node]; }
    case HEX27:
      { constexpr auto m = copy_table(Hex27::side_nodes_map); return m.values[side][side_node]; }
    case PRISM6:
      { constexpr auto m = copy_table(Prism6::side_nodes_map); return m.values[side][side_node]; }
    case PRISM15:
      { constexpr auto m = copy_table(Prism15::side_nodes_map); return m.values[side][side_node]; }
    case PRISM18:
      { constexpr auto m = copy_table(Prism18::side_nodes_map); return m.values[side][side_node]; }
    case PRISM20:
      { constexpr auto m = copy_table(Prism20::side_nodes_map); return m.values[side][side_node]; }
    case PRISM21:
      { constexpr auto m = copy_table(Prism21::side_nodes_map); return m.values[side][side_node]; }
    case PYRAMID5:
      { constexpr auto m = copy_table(Pyramid5::side_nodes_map); return m.values[side][side_node]; }
    case PYRAMID13:
      { constexpr auto m = copy_table(Pyramid13::side_nodes_map); return m.values[side][side_node]; }
    case PYRAMID14:
      { constexpr auto m = copy_table(Pyramid14::side_nodes_map); return m.values[side][side_node]; }
    case PYRAMID18:
      { constexpr auto m = copy_table(Pyramid18::side_nodes_map); return m.values[side][side_node]; }
    default:
      libmesh_error_msg("No static side node map for element type " << Utility::enum_to_string(t));
    }

  return invalid_uint;
}

} // namespace libMesh

#endif // LIBMESH_ELEM_REFERENCE_TOPOLOGY_H
