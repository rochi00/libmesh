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

// Reference-element topology and node locations shared between the host
// element classes and Kokkos device code: constexpr side/edge tables and
// lookups, with second-order side rows and higher-order node coordinates
// derived from the stored linear facts.

#ifndef LIBMESH_FE_REFERENCE_ELEMENT_TRAITS_H
#define LIBMESH_FE_REFERENCE_ELEMENT_TRAITS_H

#include "libmesh/enum_elem_type.h"
#include "libmesh/libmesh.h"
#include "libmesh/libmesh_device.h"
#include "libmesh/point.h"

namespace libMesh
{

template <unsigned int N>
struct ReferenceElementVector
{
  unsigned int values[N];

  LIBMESH_DEVICE_INLINE constexpr unsigned int operator[](unsigned int i) const
  { return values[i]; }
};

template <unsigned int Rows, unsigned int Cols>
struct ReferenceElementTable
{
  unsigned int values[Rows][Cols];

  LIBMESH_DEVICE_INLINE constexpr const unsigned int * operator[](unsigned int i) const
  { return values[i]; }

  LIBMESH_DEVICE_INLINE constexpr unsigned int operator()(unsigned int i, unsigned int j) const
  { return values[i][j]; }
};


LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<5, 4>
prism6_side_nodes()
{
  return {{
    {0, 2, 1, 99},
    {0, 1, 4, 3},
    {1, 2, 5, 4},
    {2, 0, 3, 5},
    {3, 4, 5, 99}
  }};
}

LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<5, 4>
pyramid5_side_nodes()
{
  return {{
    {0, 1, 4, 99},
    {1, 2, 4, 99},
    {2, 3, 4, 99},
    {3, 0, 4, 99},
    {0, 3, 2, 1}
  }};
}

LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<3, 2>
tri3_side_nodes()
{
  return {{
    {0, 1},
    {1, 2},
    {2, 0}
  }};
}

LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<4, 2>
quad4_side_nodes()
{
  return {{
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 0}
  }};
}

LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<4, 3>
tet4_side_nodes()
{
  return {{
    {0, 2, 1},
    {0, 1, 3},
    {1, 2, 3},
    {2, 0, 3}
  }};
}

LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<6, 4>
hex8_side_nodes()
{
  return {{
    {0, 3, 2, 1},
    {0, 1, 5, 4},
    {1, 2, 6, 5},
    {2, 3, 7, 6},
    {3, 0, 4, 7},
    {4, 5, 6, 7}
  }};
}

LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<6, 3>
tet_edge_nodes()
{
  return {{
    {0, 1, 4},
    {1, 2, 5},
    {0, 2, 6},
    {0, 3, 7},
    {1, 3, 8},
    {2, 3, 9}
  }};
}

LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<12, 3>
hex_edge_nodes()
{
  return {{
    {0, 1, 8},
    {1, 2, 9},
    {2, 3, 10},
    {0, 3, 11},
    {0, 4, 12},
    {1, 5, 13},
    {2, 6, 14},
    {3, 7, 15},
    {4, 5, 16},
    {5, 6, 17},
    {6, 7, 18},
    {4, 7, 19}
  }};
}

LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<9, 3>
prism_edge_nodes()
{
  return {{
    {0, 1, 6},
    {1, 2, 7},
    {0, 2, 8},
    {0, 3, 9},
    {1, 4, 10},
    {2, 5, 11},
    {3, 4, 12},
    {4, 5, 13},
    {3, 5, 14}
  }};
}

LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<8, 3>
pyramid_edge_nodes()
{
  return {{
    {0, 1, 5},
    {1, 2, 6},
    {2, 3, 7},
    {0, 3, 8},
    {0, 4, 9},
    {1, 4, 10},
    {2, 4, 11},
    {3, 4, 12}
  }};
}

LIBMESH_DEVICE_INLINE bool
requires_side_specific_topology(ElemType parent)
{
  switch (parent)
  {
    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      return true;
    default:
      return false;
  }
}

LIBMESH_DEVICE_INLINE ElemType
side_topology_or_invalid(ElemType parent,
                         unsigned int side)
{
  if (side > 4)
    return INVALID_ELEM;

  // Prism sides 0 and 4 are the triangles; pyramid side 4 is the quad base.
  // Every other supported element has the same topology on all its sides.
  const bool prism_tri = (side == 0 || side == 4);
  const bool pyramid_tri = (side != 4);

  switch (parent)
  {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      return NODEELEM;
    case TRI3:
    case QUAD4:
      return EDGE2;
    case TRI6:
    case TRI7:
    case QUAD8:
    case QUAD9:
      return EDGE3;
    case TET4:
      return TRI3;
    case HEX8:
      return QUAD4;
    case TET10:
      return TRI6;
    case TET14:
      return TRI7;
    case HEX20:
      return QUAD8;
    case HEX27:
      return QUAD9;
    case PRISM6:
      return prism_tri ? TRI3 : QUAD4;
    case PRISM15:
      return prism_tri ? TRI6 : QUAD8;
    case PRISM18:
      return prism_tri ? TRI6 : QUAD9;
    case PRISM20:
    case PRISM21:
      return prism_tri ? TRI7 : QUAD9;
    case PYRAMID5:
      return pyramid_tri ? TRI3 : QUAD4;
    case PYRAMID13:
      return pyramid_tri ? TRI6 : QUAD8;
    case PYRAMID14:
      return pyramid_tri ? TRI6 : QUAD9;
    case PYRAMID18:
      return pyramid_tri ? TRI7 : QUAD9;
    default:
      return INVALID_ELEM;
  }
}

// Mid-side node counts are uniform per element type except for the
// prisms and pyramids, whose triangular and quadrilateral faces differ.
LIBMESH_DEVICE_INLINE constexpr unsigned int
side_node_count_or_zero(ElemType parent,
                        unsigned int side)
{
  switch (parent)
  {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      return side < 2 ? 1 : 0;
    case TRI3:
    case TRISHELL3:
      return side < 3 ? 2 : 0;
    case TRI6:
    case TRI7:
      return side < 3 ? 3 : 0;
    case QUAD4:
    case QUADSHELL4:
      return side < 4 ? 2 : 0;
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
      return side < 4 ? 3 : 0;
    case TET4:
      return side < 4 ? 3 : 0;
    case TET10:
      return side < 4 ? 6 : 0;
    case TET14:
      return side < 4 ? 7 : 0;
    case HEX8:
      return side < 6 ? 4 : 0;
    case HEX20:
      return side < 6 ? 8 : 0;
    case HEX27:
      return side < 6 ? 9 : 0;
    case PRISM6:
      return side < 5 ? ((side == 0 || side == 4) ? 3 : 4) : 0;
    case PRISM15:
      return side < 5 ? ((side == 0 || side == 4) ? 6 : 8) : 0;
    case PRISM18:
      return side < 5 ? ((side == 0 || side == 4) ? 6 : 9) : 0;
    case PRISM20:
    case PRISM21:
      return side < 5 ? ((side == 0 || side == 4) ? 7 : 9) : 0;
    case PYRAMID5:
      return side < 5 ? (side == 4 ? 4 : 3) : 0;
    case PYRAMID13:
      return side < 5 ? (side == 4 ? 8 : 6) : 0;
    case PYRAMID14:
      return side < 5 ? (side == 4 ? 9 : 6) : 0;
    case PYRAMID18:
      return side < 5 ? (side == 4 ? 9 : 7) : 0;
    default:
      return 0;
  }
}

// Every element type with edge nodes has 3 nodes (two vertices plus a
// midpoint) on each of its edges; only the edge count varies.  The edge
// tables cover the second-order families only: the linear elements'
// vertex-pair edges stay with their classes, which device code never
// queries.
LIBMESH_DEVICE_INLINE constexpr unsigned int
edge_node_count_or_zero(ElemType parent,
                        unsigned int edge)
{
  switch (parent)
  {
    case TET10:
    case TET14:
      return edge < 6 ? 3 : 0;
    case HEX20:
    case HEX27:
      return edge < 12 ? 3 : 0;
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
      return edge < 9 ? 3 : 0;
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      return edge < 8 ? 3 : 0;
    default:
      return 0;
  }
}

LIBMESH_DEVICE_INLINE constexpr bool
try_local_edge_node(ElemType parent,
                    unsigned int edge,
                    unsigned int edge_node,
                    unsigned int & node)
{
  const unsigned int count = edge_node_count_or_zero(parent, edge);
  if (!count || edge_node >= count)
    return false;

  switch (parent)
  {
    case TET10:
    case TET14:
      node = tet_edge_nodes()(edge, edge_node);
      return true;
    case HEX20:
    case HEX27:
      node = hex_edge_nodes()(edge, edge_node);
      return true;
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
      node = prism_edge_nodes()(edge, edge_node);
      return true;
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      node = pyramid_edge_nodes()(edge, edge_node);
      return true;
    default:
      return false;
  }
}

LIBMESH_DEVICE_INLINE constexpr ElemType
linear_sibling_or_invalid(ElemType type)
{
  switch (type)
  {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      return EDGE2;
    case TRI3:
    case TRISHELL3:
    case TRI6:
    case TRI7:
      return TRI3;
    case QUAD4:
    case QUADSHELL4:
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
      return QUAD4;
    case TET4:
    case TET10:
    case TET14:
      return TET4;
    case HEX8:
    case HEX20:
    case HEX27:
      return HEX8;
    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
      return PRISM6;
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      return PYRAMID5;
    default:
      return INVALID_ELEM;
  }
}

// The node id of a face's center node, for the element types that have them.
LIBMESH_DEVICE_INLINE constexpr unsigned int
face_center_node_or_invalid(ElemType type,
                            unsigned int side)
{
  switch (type)
  {
    case TET14:
      return side < 4 ? 10 + side : invalid_uint;
    case HEX27:
      return side < 6 ? 20 + side : invalid_uint;
    case PRISM18:
      return (side >= 1 && side <= 3) ? 14 + side : invalid_uint;
    case PRISM20:
    case PRISM21:
      return side == 0 ? 18 :
             side == 4 ? 19 :
             side <= 3 ? 14 + side : invalid_uint;
    case PYRAMID14:
      return side == 4 ? 13 : invalid_uint;
    case PYRAMID18:
      return side == 4 ? 13 :
             side < 4 ? 14 + side : invalid_uint;
    default:
      return invalid_uint;
  }
}

// Corner k of a linear element's side, from the stored linear tables.
LIBMESH_DEVICE_INLINE constexpr bool
try_linear_corner(ElemType linear,
                  unsigned int side,
                  unsigned int k,
                  unsigned int & node)
{
  switch (linear)
  {
    case EDGE2:
      node = side;
      return true;
    case TRI3:
      node = tri3_side_nodes()(side, k);
      return true;
    case QUAD4:
      node = quad4_side_nodes()(side, k);
      return true;
    case TET4:
      node = tet4_side_nodes()(side, k);
      return true;
    case HEX8:
      node = hex8_side_nodes()(side, k);
      return true;
    case PRISM6:
      node = prism6_side_nodes()(side, k);
      return true;
    case PYRAMID5:
      node = pyramid5_side_nodes()(side, k);
      return true;
    default:
      return false;
  }
}

// The mid-edge node of the edge joining vertices a and b, if any.
LIBMESH_DEVICE_INLINE constexpr bool
try_edge_mid_between(ElemType parent,
                     unsigned int a,
                     unsigned int b,
                     unsigned int & node)
{
  for (unsigned int e = 0; edge_node_count_or_zero(parent, e); ++e)
  {
    unsigned int v0 = 0, v1 = 0;
    if (try_local_edge_node(parent, e, 0, v0) &&
        try_local_edge_node(parent, e, 1, v1) &&
        ((v0 == a && v1 == b) || (v0 == b && v1 == a)))
      return try_local_edge_node(parent, e, 2, node);
  }
  return false;
}

// A second-order side row is fully determined by the linear sibling's
// corner row plus the element's edge table: the corners come first, then
// the midpoint of each consecutive corner pair (wrapping), then the face's
// center node where one exists.  Only the linear tables are stored; the
// second-order rows are derived, so the side and edge topologies cannot
// disagree.
LIBMESH_DEVICE_INLINE constexpr bool
derive_local_side_node(ElemType parent,
                       unsigned int side,
                       unsigned int side_node,
                       unsigned int & node)
{
  const unsigned int count = side_node_count_or_zero(parent, side);
  if (!count || side_node >= count)
    return false;

  const ElemType linear = linear_sibling_or_invalid(parent);
  const unsigned int corners = side_node_count_or_zero(linear, side);

  if (side_node < corners)
    return try_linear_corner(linear, side, side_node, node);

  // 2D second-order elements: the single mid-side node
  if (count == 3 && corners == 2)
  {
    node = (linear == TRI3 ? 3u : 4u) + side;
    return true;
  }

  // 3D mid-edge nodes
  if (side_node < 2 * corners)
  {
    unsigned int a = 0, b = 0;
    if (!try_linear_corner(linear, side, side_node - corners, a) ||
        !try_linear_corner(linear, side, (side_node - corners + 1) % corners, b))
      return false;
    return try_edge_mid_between(parent, a, b, node);
  }

  // face center
  node = face_center_node_or_invalid(parent, side);
  return node != invalid_uint;
}

// Materialize a type's full side-node table at compile time from the
// derivation above, so runtime lookups are direct indexing while the
// derivation remains the only authority (99 pads short rows, matching the
// stored linear tables).
template <unsigned int Rows, unsigned int Cols>
LIBMESH_DEVICE_INLINE constexpr ReferenceElementTable<Rows, Cols>
build_side_nodes(ElemType parent)
{
  ReferenceElementTable<Rows, Cols> t {};
  for (unsigned int r = 0; r != Rows; ++r)
    for (unsigned int c = 0; c != Cols; ++c)
      {
        unsigned int n = 99;
        derive_local_side_node(parent, r, c, n);
        t.values[r][c] = n;
      }
  return t;
}

LIBMESH_DEVICE_INLINE bool
try_local_side_node(ElemType parent,
                    unsigned int side,
                    unsigned int side_node,
                    unsigned int & node)
{
  const unsigned int count = side_node_count_or_zero(parent, side);
  if (!count || side_node >= count)
    return false;

  switch (parent)
  {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      node = side;
      return true;
    case TRI3:
    case TRISHELL3:
      node = tri3_side_nodes()(side, side_node);
      return true;
    case QUAD4:
    case QUADSHELL4:
      node = quad4_side_nodes()(side, side_node);
      return true;
    case TET4:
      node = tet4_side_nodes()(side, side_node);
      return true;
    case HEX8:
      node = hex8_side_nodes()(side, side_node);
      return true;
    case PRISM6:
      node = prism6_side_nodes()(side, side_node);
      return true;
    case PYRAMID5:
      node = pyramid5_side_nodes()(side, side_node);
      return true;
    case TRI6:
    case TRI7:
    {
      constexpr auto t = build_side_nodes<3, 3>(TRI6);
      node = t(side, side_node);
      return true;
    }
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
    {
      constexpr auto t = build_side_nodes<4, 3>(QUAD8);
      node = t(side, side_node);
      return true;
    }
    case TET10:
    {
      constexpr auto t = build_side_nodes<4, 6>(TET10);
      node = t(side, side_node);
      return true;
    }
    case TET14:
    {
      constexpr auto t = build_side_nodes<4, 7>(TET14);
      node = t(side, side_node);
      return true;
    }
    case HEX20:
    {
      constexpr auto t = build_side_nodes<6, 8>(HEX20);
      node = t(side, side_node);
      return true;
    }
    case HEX27:
    {
      constexpr auto t = build_side_nodes<6, 9>(HEX27);
      node = t(side, side_node);
      return true;
    }
    case PRISM15:
    {
      constexpr auto t = build_side_nodes<5, 8>(PRISM15);
      node = t(side, side_node);
      return true;
    }
    case PRISM18:
    {
      constexpr auto t = build_side_nodes<5, 9>(PRISM18);
      node = t(side, side_node);
      return true;
    }
    case PRISM20:
    case PRISM21:
    {
      constexpr auto t = build_side_nodes<5, 9>(PRISM20);
      node = t(side, side_node);
      return true;
    }
    case PYRAMID13:
    {
      constexpr auto t = build_side_nodes<5, 8>(PYRAMID13);
      node = t(side, side_node);
      return true;
    }
    case PYRAMID14:
    {
      constexpr auto t = build_side_nodes<5, 9>(PYRAMID14);
      node = t(side, side_node);
      return true;
    }
    case PYRAMID18:
    {
      constexpr auto t = build_side_nodes<5, 9>(PYRAMID18);
      node = t(side, side_node);
      return true;
    }
    default:
      return false;
  }
}

// Reference-element vertex locations, per family (higher-order family
// members share their linear sibling's vertices).
LIBMESH_DEVICE_INLINE unsigned int
reference_vertex_count(ElemType type)
{
  switch (type)
  {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      return 2;
    case TRI3:
    case TRISHELL3:
    case TRI6:
    case TRI7:
      return 3;
    case QUAD4:
    case QUADSHELL4:
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
    case TET4:
    case TET10:
    case TET14:
      return 4;
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      return 5;
    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
      return 6;
    case HEX8:
    case HEX20:
    case HEX27:
      return 8;
    default:
      return 0;
  }
}

LIBMESH_DEVICE_INLINE bool
reference_vertex(ElemType type,
                 unsigned int v,
                 Point & pt)
{
  if (v >= reference_vertex_count(type))
    return false;

  switch (type)
  {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      pt = Point(v == 0 ? -1.0 : 1.0);
      return true;
    case TRI3:
    case TRISHELL3:
    case TRI6:
    case TRI7:
      pt = Point(v == 1 ? 1.0 : 0.0, v == 2 ? 1.0 : 0.0);
      return true;
    case QUAD4:
    case QUADSHELL4:
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
      pt = Point((v == 1 || v == 2) ? 1.0 : -1.0,
                 (v == 2 || v == 3) ? 1.0 : -1.0);
      return true;
    case TET4:
    case TET10:
    case TET14:
      pt = Point(v == 1 ? 1.0 : 0.0, v == 2 ? 1.0 : 0.0, v == 3 ? 1.0 : 0.0);
      return true;
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      pt = v == 4 ? Point(0.0, 0.0, 1.0)
                  : Point((v == 1 || v == 2) ? 1.0 : -1.0,
                          (v == 2 || v == 3) ? 1.0 : -1.0,
                          0.0);
      return true;
    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
      pt = Point(v % 3 == 1 ? 1.0 : 0.0,
                 v % 3 == 2 ? 1.0 : 0.0,
                 v < 3 ? -1.0 : 1.0);
      return true;
    case HEX8:
    case HEX20:
    case HEX27:
      pt = Point((v % 4 == 1 || v % 4 == 2) ? 1.0 : -1.0,
                 (v % 4 == 2 || v % 4 == 3) ? 1.0 : -1.0,
                 v < 4 ? -1.0 : 1.0);
      return true;
    default:
      return false;
  }
}

// The node holding an element's vertex-centroid, if it has one.
LIBMESH_DEVICE_INLINE unsigned int
centroid_node_or_invalid(ElemType type)
{
  switch (type)
  {
    case EDGE3:
      return 2;
    case TRI7:
      return 6;
    case QUAD9:
    case QUADSHELL9:
      return 8;
    case HEX27:
      return 26;
    case PRISM21:
      return 20;
    default:
      return invalid_uint;
  }
}

// Every higher-order reference node sits at the centroid of its
// subentity's vertices -- mid-edge nodes at edge midpoints, face nodes at
// face-corner centroids, interior nodes at the vertex centroid -- so only
// the vertices are tabulated and the rest is derived through the same
// side/edge topology tables everything else uses.  The one exception is
// the cubic EDGE4, whose two interior nodes trisect the edge.
LIBMESH_DEVICE_INLINE bool
try_reference_node(ElemType type,
                   unsigned int node,
                   Point & pt)
{
  const unsigned int nv = reference_vertex_count(type);
  if (node < nv)
    return reference_vertex(type, node, pt);

  if (type == EDGE4)
    {
      if (node > 3)
        return false;
      pt = Point(node == 2 ? Real(-1) / 3 : Real(1) / 3);
      return true;
    }

  if (node == centroid_node_or_invalid(type))
    {
      Point sum;
      for (unsigned int v = 0; v != nv; ++v)
        {
          Point pv;
          reference_vertex(type, v, pv);
          sum += pv;
        }
      pt = sum / Real(nv);
      return true;
    }

  // Mid-edge nodes: the third entry of an edge-table row {v0, v1, mid}.
  for (unsigned int e = 0; edge_node_count_or_zero(type, e); ++e)
    {
      unsigned int mid, v0, v1;
      if (try_local_edge_node(type, e, 2, mid) && mid == node &&
          try_local_edge_node(type, e, 0, v0) &&
          try_local_edge_node(type, e, 1, v1))
        {
          Point p0, p1;
          reference_vertex(type, v0, p0);
          reference_vertex(type, v1, p1);
          pt = (p0 + p1) / 2;
          return true;
        }
    }

  // 2D mid-side nodes ({v0, v1, mid} side rows) and 3D face-center nodes
  // (the last entry of a 7- or 9-node side row, behind 3 or 4 corners).
  for (unsigned int s = 0; ; ++s)
    {
      const unsigned int count = side_node_count_or_zero(type, s);
      if (!count)
        break;
      const unsigned int corners =
        count == 3 ? 2 : count == 7 ? 3 : count == 9 ? 4 : 0;
      unsigned int last;
      if (!corners ||
          !try_local_side_node(type, s, count - 1, last) || last != node)
        continue;
      Point sum;
      for (unsigned int k = 0; k != corners; ++k)
        {
          unsigned int v;
          try_local_side_node(type, s, k, v);
          Point pv;
          reference_vertex(type, v, pv);
          sum += pv;
        }
      pt = sum / Real(corners);
      return true;
    }

  return false;
}
LIBMESH_DEVICE_INLINE bool
try_refspace_node(ElemType type,
                  unsigned int node,
                  Point & pt)
{
  switch (type)
  {
    case NODEELEM:
      if (!node)
      {
        pt = Point(0.0, 0.0, 0.0);
        return true;
      }
      return false;

    case TRISHELL3:
      return try_reference_node(TRI3, node, pt);

    case QUADSHELL4:
      return try_reference_node(QUAD4, node, pt);

    case QUADSHELL8:
      return try_reference_node(QUAD8, node, pt);

    case QUADSHELL9:
      return try_reference_node(QUAD9, node, pt);

    default:
      return try_reference_node(type, node, pt);
  }
}

LIBMESH_DEVICE_INLINE bool
try_reference_side_node(ElemType parent,
                        unsigned int side,
                        unsigned int side_node,
                        Point & pt)
{
  unsigned int node = libMesh::invalid_uint;
  if (!try_local_side_node(parent, side, side_node, node))
    return false;

  return try_reference_node(parent, node, pt);
}

} // namespace libMesh

#endif // LIBMESH_FE_REFERENCE_ELEMENT_TRAITS_H
