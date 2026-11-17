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

// Checks MeshMirror against the mesh it copies: every active local
// element and every ghost's bucket, id, subdomain, p level and nodes,
// exactly; every side's boundary ids, queried in a kernel; the
// nodes of every boundary id; every side's neighbor and the side facing
// back, against the host's links and, in a kernel, against the
// neighbor's own link; the elements of every node, against the host's
// and, in a kernel, against the elements' nodes; each lower-dimensional
// element's interior parent and side, against the host's and, in a
// kernel, against the parent's side nodes; and, in a kernel reading
// each element through its bucket's node accessor, the quadrature
// points and JxW FE::reinit() gives at the element's p level.  On a
// distorted HEX27 cube with several ids on some sides and QUAD9 faces
// on one boundary, on a mixed TRI3 / QUAD4 square with EDGE2 sides on
// its boundary, before and after some elements are refined, on a
// distributed HEX8 cube, and on a mesh with one element of every
// Lagrange type, before and after its nodes move.  The kernel must
// give the host's points and JxW to within rounding.  On several processors
// each checks its own mirror, ghosts included.

#ifndef KOKKOS_MESH_MIRROR_ORACLE_H
#define KOKKOS_MESH_MIRROR_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"

#include "libmesh/kokkos_mesh_mirror.h"

#include "libmesh/boundary_info.h"
#include "libmesh/distributed_mesh.h"
#include "libmesh/elem.h"
#include "libmesh/elem_reference_topology.h"
#include "libmesh/face_tri3.h"
#include "libmesh/fe_base.h"
#include "libmesh/int_range.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_modification.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/parallel_implementation.h"
#include "libmesh/quadrature_gauss_3D.h"
#include "libmesh/reference_elem.h"
#include "libmesh/remote_elem.h"
#include "libmesh/replicated_mesh.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

namespace libMeshTest
{
namespace KokkosMeshMirrorOracle
{

using namespace libMesh;
using libMesh::Kokkos::MeshMirror;
using index_type = MeshMirror::index_type;
using Link = MeshMirror::Link;

// Room for the rule of any type at THIRD order raised by p level 1
constexpr unsigned int max_qp = 64;

// What the checks found: the mismatches, and how many of each thing
// they checked, for the summary
struct Counts
{
  int n_fail = 0;

  // Local elements and ghosts, sides, neighbor links, (node, element)
  // entries, lower-dimensional elements and quadrature points checked
  std::size_t n_elem = 0, n_ghosts = 0, n_sides = 0, n_links = 0, n_node_elems = 0,
    n_lower_d = 0, n_points = 0;
  // Links by kind: valid with a side, valid without one, none, remote,
  // refined; for neighbors [0] and interior parents [1]
  std::size_t kinds[2][5] = {};
};

// Counts a mismatch in the Counts c in scope, printing the first 10
#define MIRROR_FAIL(...)                                        \
  do {                                                          \
    if (c.n_fail < 10)                                          \
      std::printf("[mesh_mirror_oracle] mismatch: " __VA_ARGS__); \
    ++c.n_fail;                                                 \
  } while (0)

// Each mirrored element against the mesh's, and the bucket's order
inline void
check_structure(const MeshBase & mesh, const MeshMirror & mirror, Counts & c)
{
  auto h_points = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), mirror.points());

  // Each bucket's views, on the host
  struct HostBucket
  {
    decltype(::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), MeshMirror::Bucket().nodes)) nodes;
    decltype(::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), MeshMirror::Bucket().id)) id;
    decltype(::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), MeshMirror::Bucket().subdomain)) sid;
    decltype(::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), MeshMirror::Bucket().p_level)) p;
  };
  std::vector<HostBucket> hb;

  std::size_t n_mirrored = 0;
  for (const auto k : make_range(mirror.n_buckets()))
    {
      const MeshMirror::Bucket & b = mirror.bucket(k);
      n_mirrored += b.n_elem;
      hb.push_back(HostBucket{::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.nodes),
                              ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.id),
                              ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.subdomain),
                              ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.p_level)});
      const auto & h_id = hb.back().id;
      const auto & h_sid = hb.back().sid;
      // Each part, local and ghost, sorted, and in its subdomain's range
      for (index_type e = 0; e + 1 < b.n_elem; ++e)
        if (e + 1 != b.n_local &&
            std::make_pair(h_sid(e), h_id(e)) >= std::make_pair(h_sid(e+1), h_id(e+1)))
          MIRROR_FAIL("bucket %u not sorted at %u\n", k, e);
      for (const auto e : make_range(b.n_elem))
        {
          const Elem * elem = mesh.elem_ptr(h_id(e));
          if ((elem->processor_id() == mesh.processor_id()) != (e < b.n_local))
            MIRROR_FAIL("bucket %u elem %u local %d, n_local %u\n", k, e,
                        int(elem->processor_id() == mesh.processor_id()), b.n_local);
          if (b.lower_d != bool(elem->dim() < LIBMESH_DIM && elem->interior_parent()))
            MIRROR_FAIL("bucket %u elem %u: lower_d\n", k, e);
          const auto [first, last] = mirror.subdomain_range(k, h_sid(e), e >= b.n_local);
          if (e < first || e >= last)
            MIRROR_FAIL("bucket %u elem %u outside its subdomain's range\n", k, e);
        }
      for (const auto k2 : make_range(k))
        if (mirror.bucket(k2).type == b.type && mirror.bucket(k2).lower_d == b.lower_d)
          MIRROR_FAIL("buckets %u and %u have the same type\n", k2, k);
      if (mirror.bucket_of_type(b.type, b.lower_d) != k)
        MIRROR_FAIL("bucket_of_type for bucket %u\n", k);
    }

  // The local elements and the ghosts the mirror should hold: the
  // active elements sharing a node with a local one
  std::set<dof_id_type> local_nodes;
  for (const Elem * elem : mesh.active_local_element_ptr_range())
    for (const Node & node : elem->node_ref_range())
      local_nodes.insert(node.id());
  std::size_t n_expected = 0, n_ghosts = 0;
  for (const Elem * elem : mesh.active_element_ptr_range())
    {
      const bool local = elem->processor_id() == mesh.processor_id();
      bool ghost = false;
      for (const Node & node : elem->node_ref_range())
        ghost = ghost || (!local && local_nodes.count(node.id()));
      const auto [k, e] = mirror.elem_index(elem->id());
      if (!local && !ghost)
        {
          if (k != invalid_uint)
            MIRROR_FAIL("elem %lu mirrored but not a ghost\n", (unsigned long)elem->id());
          continue;
        }
      ++n_expected;
      n_ghosts += ghost;
      if (k == invalid_uint)
        {
          MIRROR_FAIL("elem %lu not mirrored\n", (unsigned long)elem->id());
          continue;
        }
      const MeshMirror::Bucket & b = mirror.bucket(k);
      const auto & h_nodes = hb[k].nodes;
      const auto & h_id = hb[k].id;
      const auto & h_sid = hb[k].sid;
      const auto & h_p = hb[k].p;
      if (b.type != elem->type() || h_id(e) != elem->id() ||
          h_sid(e) != elem->subdomain_id() || h_p(e) != elem->p_level())
        MIRROR_FAIL("elem %lu: type, id, subdomain or p level\n", (unsigned long)elem->id());
      for (const auto i : make_range(elem->n_nodes()))
        {
          const index_type n = h_nodes(e, i);
          Point p;
          for (const auto d : make_range(LIBMESH_DIM))
            p(d) = h_points(n, d);
          if (mirror.node_ids()[n] != elem->node_id(i) ||
              !KokkosOracle::same_bits(p, elem->point(i)) ||
              (local && n >= mirror.n_local_nodes()))
            MIRROR_FAIL("elem %lu node %u\n", (unsigned long)elem->id(), i);
        }
    }
  if (n_expected != n_mirrored)
    MIRROR_FAIL("%zu local elements and ghosts, %zu mirrored\n", n_expected, n_mirrored);
  if (local_nodes.size() != mirror.n_local_nodes())
    MIRROR_FAIL("%zu nodes of local elements, mirror says %u\n", local_nodes.size(),
                mirror.n_local_nodes());
  c.n_elem += n_expected - n_ghosts;
  c.n_ghosts += n_ghosts;
}

// The sorted vertex ids of side s of elem, or of the whole element if
// s is invalid_uint
inline std::vector<dof_id_type>
vertex_ids(const Elem * elem, const unsigned int s)
{
  std::vector<dof_id_type> v;
  if (s == invalid_uint)
    for (const auto i : make_range(elem->n_vertices()))
      v.push_back(elem->node_id(i));
  else
    for (const unsigned int i : elem->nodes_on_side(s))
      if (elem->is_vertex(i))
        v.push_back(elem->node_id(i));
  std::sort(v.begin(), v.end());
  return v;
}

// The link the mirror should hold to other, the neighbor or interior
// parent of elem, with other's side found by vertex ids for a neighbor
// on elem's level or an interior parent, and by which_neighbor_am_i()
// for a coarser neighbor
inline Link
expected_link(const MeshMirror & mirror, const Elem * elem, const Elem * other,
              const unsigned int s, const bool interior_parent)
{
  Link l;
  if (!other)
    return l;
  if (other == remote_elem)
    {
      l.bucket = Link::remote;
      return l;
    }
  if (!other->active())
    {
      l.bucket = Link::refined;
      return l;
    }
  const auto [k, e] = mirror.elem_index(other->id());
  if (k == invalid_uint)
    {
      l.bucket = Link::remote;
      return l;
    }
  l.bucket = (unsigned char)k;
  l.elem = e;
  const std::vector<dof_id_type> mine = vertex_ids(elem, interior_parent ? invalid_uint : s);
  if (interior_parent || other->level() == elem->level())
    {
      for (auto t : other->side_index_range())
        if (vertex_ids(other, t) == mine)
          l.side = (unsigned char)t;
    }
  else
    l.side = (unsigned char)other->which_neighbor_am_i(elem);
  return l;
}

// Whether a and b link to the same element and side
inline bool
same_link(const Link & a, const Link & b)
{ return a.elem == b.elem && a.bucket == b.bucket && a.side == b.side; }

// The index of l's kind in Counts::kinds
inline unsigned int
link_kind(const Link & l)
{
  if (l.valid())
    return l.side == Link::invalid_side;
  return l.bucket == Link::none ? 2 : l.bucket == Link::remote ? 3 : 4;
}

// Every side's neighbor against the host's; each link to a neighbor
// on the same level against the neighbor's link back, on the host
// across buckets and in a kernel within one
inline void
check_neighbors(const MeshBase & mesh, const MeshMirror & mirror, Counts & c)
{
  std::vector<decltype(::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(),
                                                             MeshMirror::Bucket().neighbors))> h_nb;
  for (const auto k : make_range(mirror.n_buckets()))
    h_nb.push_back(::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), mirror.bucket(k).neighbors));

  for (const auto k : make_range(mirror.n_buckets()))
    {
      const MeshMirror::Bucket b = mirror.bucket(k);
      auto h_id = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.id);
      for (const auto e : make_range(b.n_elem))
        {
          const Elem * elem = mesh.elem_ptr(h_id(e));
          for (const auto s : make_range(b.n_sides))
            {
              const Elem * nb = elem->neighbor_ptr(s);
              const Link want = expected_link(mirror, elem, nb, s, false);
              const Link got = h_nb[k](e, s);
              ++c.n_links;
              ++c.kinds[0][link_kind(got)];
              if (!same_link(want, got))
                MIRROR_FAIL("elem %lu side %u: neighbor (%u, %u, %u), want (%u, %u, %u)\n",
                            (unsigned long)elem->id(), s, unsigned(got.bucket), got.elem,
                            unsigned(got.side), unsigned(want.bucket), want.elem, unsigned(want.side));
              // A local element's neighbors are mirrored
              if (e < b.n_local && got.bucket == Link::remote)
                MIRROR_FAIL("local elem %lu side %u: remote neighbor\n", (unsigned long)elem->id(), s);
              if (got.valid() && nb->level() == elem->level())
                {
                  const Link back = h_nb[got.bucket](got.elem, got.side);
                  if (back.bucket != k || back.elem != e || back.side != s)
                    MIRROR_FAIL("elem %lu side %u: neighbor's link back\n", (unsigned long)elem->id(), s);
                }
            }
        }

      // In a kernel: a valid link back from a neighbor in this bucket
      // is to this side
      if (!b.n_sides)
        continue;
      int n_bad = 0;
      ::Kokkos::parallel_reduce(
        "mesh_mirror_neighbors", b.n_elem, KOKKOS_LAMBDA(const index_type e, int & bad) {
          for (unsigned int s = 0; s != b.n_sides; ++s)
            {
              const Link l = b.neighbor(e, s);
              if (!l.valid() || l.bucket != k)
                continue;
              if (l.elem >= b.n_elem || l.side >= b.n_sides)
                {
                  ++bad;
                  continue;
                }
              const Link back = b.neighbor(l.elem, l.side);
              if (back.valid() && (back.bucket != k || back.elem != e || back.side != s))
                ++bad;
            }
        }, n_bad);
      if (n_bad)
        MIRROR_FAIL("bucket %u: %d links back wrong in a kernel\n", k, n_bad);
    }
}

// Each node's elements against the host's: all active elements of a
// node of a local element, the mirrored ones of any other node; and,
// in a kernel, that each entry's element has the node there
inline void
check_node_elems(const MeshBase & mesh, const MeshMirror & mirror, Counts & c)
{
  const MeshMirror::NodeElems ne = mirror.node_elems();
  auto h_offsets = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), ne.offsets);
  auto h_entries = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), ne.entries);
  const index_type n_nodes = mirror.n_nodes();
  if (h_offsets.extent(0) != std::size_t(n_nodes) + 1)
    {
      MIRROR_FAIL("node elem offsets have %zu entries for %u nodes\n", h_offsets.extent(0), n_nodes);
      return;
    }

  // The host's: every active element of the mesh by mirrored node
  std::map<index_type, std::vector<std::tuple<unsigned int, index_type, unsigned int>>> host;
  for (const Elem * elem : mesh.active_element_ptr_range())
    {
      const auto [k, e] = mirror.elem_index(elem->id());
      for (const auto i : make_range(elem->n_nodes()))
        {
          const index_type n = mirror.node_index(elem->node_id(i));
          if (n == MeshMirror::invalid_index)
            continue;
          if (k == invalid_uint)
            {
              if (n < mirror.n_local_nodes())
                MIRROR_FAIL("elem %lu has node %u of a local element but isn't mirrored\n",
                            (unsigned long)elem->id(), n);
              continue;
            }
          host[n].emplace_back(k, e, i);
        }
    }

  for (const auto n : make_range(n_nodes))
    {
      auto & want = host[n];
      std::sort(want.begin(), want.end());
      std::vector<std::tuple<unsigned int, index_type, unsigned int>> got;
      for (const auto i : make_range(h_offsets(n), h_offsets(n+1)))
        got.emplace_back(h_entries(i).bucket, h_entries(i).elem, h_entries(i).node);
      if (got != want)
        MIRROR_FAIL("node %u (id %lu): %zu elements, want %zu\n", n,
                    (unsigned long)mirror.node_ids()[n], got.size(), want.size());
      c.n_node_elems += got.size();
    }

  for (const auto k : make_range(mirror.n_buckets()))
    {
      const MeshMirror::Bucket b = mirror.bucket(k);
      const unsigned int nn = Elem::type_to_n_nodes_map[b.type];
      int n_bad = 0;
      ::Kokkos::parallel_reduce(
        "mesh_mirror_node_elems", n_nodes, KOKKOS_LAMBDA(const index_type n, int & bad) {
          for (index_type i = 0; i != ne.n_elems(n); ++i)
            {
              const MeshMirror::NodeElem x = ne(n, i);
              if (i && !(ne(n, i-1).order() < x.order()))
                ++bad;
              if (x.bucket == k && (x.elem >= b.n_elem || x.node >= nn || b.nodes(x.elem, x.node) != n))
                ++bad;
            }
        }, n_bad);
      if (n_bad)
        MIRROR_FAIL("bucket %u: %d node elements wrong in a kernel\n", k, n_bad);
    }
}

// Each lower-dimensional element's interior parent against the host's;
// and, in a kernel per pair of buckets, that the parent's side has the
// element's type and nodes, in any order.
inline void
check_lower_d(const MeshBase & mesh, const MeshMirror & mirror, Counts & c)
{
  for (const auto k : make_range(mirror.n_buckets()))
    {
      const MeshMirror::Bucket b = mirror.bucket(k);
      if (!b.lower_d)
        {
          if (b.interior_parent.extent(0))
            MIRROR_FAIL("bucket %u has interior parents but isn't lower_d\n", k);
          continue;
        }
      auto h_id = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.id);
      auto h_ip = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.interior_parent);
      for (const auto e : make_range(b.n_elem))
        {
          const Elem * elem = mesh.elem_ptr(h_id(e));
          const Link want = expected_link(mirror, elem, elem->interior_parent(), invalid_uint, true);
          if (!same_link(want, h_ip(e)))
            MIRROR_FAIL("elem %lu: interior parent (%u, %u, %u), want (%u, %u, %u)\n",
                        (unsigned long)elem->id(), unsigned(h_ip(e).bucket), h_ip(e).elem,
                        unsigned(h_ip(e).side), unsigned(want.bucket), want.elem, unsigned(want.side));
          ++c.n_lower_d;
          ++c.kinds[1][link_kind(h_ip(e))];
        }

      const unsigned int nn = Elem::type_to_n_nodes_map[b.type];
      for (const auto kp : make_range(mirror.n_buckets()))
        {
          const MeshMirror::Bucket pb = mirror.bucket(kp);
          int n_bad = 0;
          ::Kokkos::parallel_reduce(
            "mesh_mirror_interior_parents", b.n_elem, KOKKOS_LAMBDA(const index_type e, int & bad) {
              const Link ip = b.interior_parent(e);
              if (ip.bucket != kp || ip.side == Link::invalid_side)
                return;
              if (ip.elem >= pb.n_elem || ip.side >= pb.n_sides ||
                  Elem::side_type(pb.type, ip.side) != b.type)
                {
                  ++bad;
                  return;
                }
              // The element's nodes are the side's, in any order; with
              // the side's type, every side node is then one of them
              for (unsigned int j = 0; j != nn; ++j)
                {
                  bool found = false;
                  for (unsigned int i = 0; i != nn; ++i)
                    found = found ||
                      b.nodes(e, j) == pb.nodes(ip.elem, Elem::local_side_node(pb.type, ip.side, i));
                  bad += !found;
                }
            }, n_bad);
          if (n_bad)
            MIRROR_FAIL("buckets %u, %u: %d interior parent sides wrong in a kernel\n", k, kp, n_bad);
        }
    }
}

// Every side's boundary ids, from the device's search and from the
// device's list of each id's sides; then each id's nodes
inline void
check_boundary(const MeshBase & mesh, const MeshMirror & mirror, Counts & c)
{
  const BoundaryInfo & bi = mesh.get_boundary_info();
  const unsigned int n_bids = cast_int<unsigned int>(mirror.boundary_ids().size());
  std::vector<boundary_id_type> ids;

  for (const auto k : make_range(mirror.n_buckets()))
    {
      const MeshMirror::Bucket b = mirror.bucket(k);
      if (!b.n_sides || !n_bids)
        continue;

      ::Kokkos::View<unsigned char *> d_has("mirror_has", std::size_t(b.n_elem) * b.n_sides * n_bids);
      ::Kokkos::parallel_for(
        "mesh_mirror_has_boundary", b.n_elem, KOKKOS_LAMBDA(const index_type e) {
          for (unsigned int s = 0; s != b.n_sides; ++s)
            for (unsigned int bid = 0; bid != n_bids; ++bid)
              d_has((std::size_t(e) * b.n_sides + s) * n_bids + bid) = b.has_boundary_index(e, s, bid);
        });

      // Decode every listed side on the device
      auto h_offsets = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.boundary_offsets);
      const index_type n_listed = h_offsets(n_bids);
      ::Kokkos::View<index_type *> d_side_elem("mirror_side_elem", n_listed);
      ::Kokkos::View<unsigned int *> d_side_side("mirror_side_side", n_listed);
      ::Kokkos::parallel_for(
        "mesh_mirror_boundary_sides", n_bids, KOKKOS_LAMBDA(const unsigned int bid) {
          for (index_type i = 0; i != b.n_boundary_sides(bid); ++i)
            {
              index_type e;
              unsigned int s;
              b.boundary_side(bid, i, e, s);
              d_side_elem(b.boundary_offsets(bid) + i) = e;
              d_side_side(b.boundary_offsets(bid) + i) = s;
            }
        });

      auto h_has = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_has);
      auto h_side_elem = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_side_elem);
      auto h_side_side = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_side_side);
      auto h_id = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.id);

      std::size_t n_host_listed = 0;
      for (const auto e : make_range(b.n_elem))
        {
          const Elem * elem = mesh.elem_ptr(h_id(e));
          for (const auto s : make_range(b.n_sides))
            {
              bi.boundary_ids(elem, s, ids);
              n_host_listed += ids.size();
              ++c.n_sides;
              const std::set<boundary_id_type> host(ids.begin(), ids.end());
              for (const auto bid : make_range(n_bids))
                if (bool(h_has((std::size_t(e) * b.n_sides + s) * n_bids + bid)) !=
                    bool(host.count(mirror.boundary_ids()[bid])))
                  MIRROR_FAIL("elem %lu side %u boundary %d\n", (unsigned long)elem->id(), s,
                              int(mirror.boundary_ids()[bid]));
            }
        }
      if (n_host_listed != n_listed)
        MIRROR_FAIL("bucket %u lists %u boundary sides, the mesh has %zu\n", k, n_listed, n_host_listed);
      for (const auto bid : make_range(n_bids))
        for (const auto i : make_range(h_offsets(bid), h_offsets(bid+1)))
          {
            bi.boundary_ids(mesh.elem_ptr(h_id(h_side_elem(i))), h_side_side(i), ids);
            if (std::find(ids.begin(), ids.end(), mirror.boundary_ids()[bid]) == ids.end() ||
                (i > h_offsets(bid) &&
                 std::make_pair(h_side_elem(i-1), h_side_side(i-1)) >=
                 std::make_pair(h_side_elem(i), h_side_side(i))))
              MIRROR_FAIL("bucket %u boundary side %u\n", k, i);
          }
    }

  auto h_node_offsets = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), mirror.boundary_node_offsets());
  auto h_bnodes = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), mirror.boundary_nodes());
  for (const auto bid : make_range(n_bids))
    {
      std::set<index_type> host;
      for (const auto n : make_range(mirror.n_nodes()))
        {
          bi.boundary_ids(mesh.node_ptr(mirror.node_ids()[n]), ids);
          if (std::find(ids.begin(), ids.end(), mirror.boundary_ids()[bid]) != ids.end())
            host.insert(n);
        }
      const std::set<index_type> listed(h_bnodes.data() + h_node_offsets(bid),
                                        h_bnodes.data() + h_node_offsets(bid+1));
      if (host != listed || listed.size() != h_node_offsets(bid+1) - h_node_offsets(bid))
        MIRROR_FAIL("nodes of boundary %d\n", int(mirror.boundary_ids()[bid]));
    }
}

// A quadrature point's physical point and JxW
struct MappedPoint
{
  Point xyz;
  Real JxW = 0;
};

// The map of element e of bucket b on its reference nodes
struct ReferenceMapOf
{
  KOKKOS_INLINE_FUNCTION
  LagrangeMapEval<MeshMirror::ReferenceNodes>
  operator()(const MeshMirror::Bucket & b, const index_type e) const
  { return b.reference_map(e); }
};

// The same map, compiled for the bucket's type T
template <ElemType T>
struct TypedReferenceMapOf
{
  KOKKOS_INLINE_FUNCTION
  LagrangeMapEval<MeshMirror::ReferenceNodes>
  operator()(const MeshMirror::Bucket & b, const index_type e) const
  { return b.reference_map<T>(e); }
};

// The rule at each element's p level of bucket b, mapped by
// map_of(b, e), the element's map through its bucket's node
// accessor, against FE::reinit() on mesh's element of the same id
template <typename MapOf>
inline void
check_bucket_geometry(const MeshBase & mesh, const MeshMirror::Bucket & b, Counts & c,
                      const MapOf & map_of)
{
  constexpr Order order = THIRD;
  const unsigned int dim = Elem::type_to_dim_map[b.type];

  // The host's points and JxW per element, and where each element's
  // start
  auto h_id = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.id);
  std::vector<MappedPoint> host;
  ::Kokkos::View<index_type *> d_first("mirror_first_qp", b.n_elem + 1);
  auto h_first = ::Kokkos::create_mirror_view(d_first);
  h_first(0) = 0;
  std::unique_ptr<FEBase> fe = FEBase::build(dim, FEType(FIRST, LAGRANGE));
  fe->add_p_level_in_reinit(false);
  QGauss qrule(dim, order);
  fe->attach_quadrature_rule(&qrule);
  const auto & xyz = fe->get_xyz();
  const auto & JxW = fe->get_JxW();
  for (const auto e : make_range(b.n_elem))
    {
      const Elem * elem = mesh.elem_ptr(h_id(e));
      fe->reinit(elem);
      libmesh_error_msg_if(qrule.n_points() > max_qp, "mesh mirror oracle: raise max_qp");
      for (const auto q : make_range(qrule.n_points()))
        host.push_back({xyz[q], JxW[q]});
      h_first(e+1) = h_first(e) + qrule.n_points();
    }
  ::Kokkos::deep_copy(d_first, h_first);
  const bool negative_weights = qrule.allow_rules_with_negative_weights;

  ::Kokkos::View<MappedPoint *> d_out("mirror_out", host.size());
  ::Kokkos::View<unsigned int *> d_n("mirror_n_qp", b.n_elem);
  ::Kokkos::parallel_for(
    "mesh_mirror_geometry", b.n_elem, KOKKOS_LAMBDA(const index_type e) {
      const Order o = QBase::order_for_p_level(order, b.p_level(e));
      const unsigned int n_qp = QGauss::n_points_elem(b.type, o, negative_weights);
      d_n(e) = n_qp;
      if (n_qp != d_first(e+1) - d_first(e) || n_qp > max_qp)
        return;
      Point qp[max_qp];
      Real w[max_qp];
      QGauss::rule_elem(b.type, o, negative_weights, qp, w);

      const auto map = map_of(b, e);
      const auto no_check = [](const Real) {};
      for (unsigned int q = 0; q != n_qp; ++q)
        {
          Point dxyz[3];
          for (unsigned int j = 0; j != map.dim; ++j)
            dxyz[j] = map.map_deriv(j, qp[q]);
          FEMapPoint<> m;
          if (map.dim == 1)
            fe_map_1D(dxyz[0], m, no_check);
          else if (map.dim == 2)
            fe_map_2D(dxyz[0], dxyz[1], m, no_check);
          else
            fe_map_3D(dxyz[0], dxyz[1], dxyz[2], m, no_check);

          MappedPoint & out = d_out(d_first(e) + q);
          out.xyz = map.map(qp[q]);
          out.JxW = m.jac * w[q];
        }
    });

  auto h_n = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_n);
  auto h_out = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_out);
  for (const auto e : make_range(b.n_elem))
    {
      if (h_n(e) != h_first(e+1) - h_first(e))
        {
          MIRROR_FAIL("elem %lu: host %u points, kernel %u\n", (unsigned long)h_id(e),
                      h_first(e+1) - h_first(e), h_n(e));
          continue;
        }
      for (const auto q : make_range(h_first(e), h_first(e+1)))
        {
          const auto compare = [&](const char * name, const Real h, const Real kernel)
            {
              if (!KokkosOracle::same_bits(Point(h), Point(kernel)) &&
                  std::abs(kernel - h) > 1e-10 * (1 + std::abs(h)))
                MIRROR_FAIL("type %d elem %lu qp %u %s: host %.17g kernel %.17g\n",
                            int(b.type), (unsigned long)h_id(e), q - h_first(e),
                            name, double(h), double(kernel));
            };
          // 1e-10 relative to 1 + |host|: far above the few ulp a
          // contracted map evaluation changes, far below the change
          // from a wrong node
          for (const auto d : make_range(LIBMESH_DIM))
            compare("point", host[q].xyz(d), h_out(q).xyz(d));
          // FEMap rescales an affine element's JxW from its first
          // point's, which agrees with the kernel's to rounding
          compare("JxW", host[q].JxW, h_out(q).JxW);
        }
      c.n_points += h_first(e+1) - h_first(e);
    }
}

// check_bucket_geometry() on each bucket of mirror, through
// map_of(b, e)
template <typename MapOf = ReferenceMapOf>
inline void
check_geometry(const MeshBase & mesh, const MeshMirror & mirror, Counts & c,
               const MapOf & map_of = MapOf())
{
  for (const auto k : make_range(mirror.n_buckets()))
    if (Elem::type_to_dim_map[mirror.bucket(k).type])
      check_bucket_geometry(mesh, mirror.bucket(k), c, map_of);
}

// check_bucket_geometry() on each bucket of mirror, through the map
// compiled for its type: the kernel dispatch_elem_type() picks
inline void
check_typed_geometry(const MeshBase & mesh, const MeshMirror & mirror, Counts & c)
{
  for (const auto k : make_range(mirror.n_buckets()))
    {
      const MeshMirror::Bucket & b = mirror.bucket(k);
      if (!Elem::type_to_dim_map[b.type])
        continue;
      if (!libMesh::Kokkos::LagrangeElemTypes::contains(b.type))
        {
          MIRROR_FAIL("type %d has no typed map\n", int(b.type));
          continue;
        }
      libMesh::Kokkos::dispatch_elem_type<libMesh::Kokkos::LagrangeElemTypes>(b.type, [&](auto traits) {
          check_bucket_geometry(mesh, b, c, TypedReferenceMapOf<decltype(traits)::type>());
        });
    }
}

// Every check of mirror against mesh
inline void
check_mesh(const MeshBase & mesh, const MeshMirror & mirror, Counts & c)
{
  check_structure(mesh, mirror, c);
  check_boundary(mesh, mirror, c);
  check_neighbors(mesh, mirror, c);
  check_node_elems(mesh, mirror, c);
  check_lower_d(mesh, mirror, c);
  check_geometry(mesh, mirror, c);
  check_typed_geometry(mesh, mirror, c);
}

#undef MIRROR_FAIL

// Renumbers the nodes of side: keeps them if how is 0, rotates them
// if 1, reverses the orientation if 2
inline void
permute_nodes(Elem & side, const unsigned int how)
{
  std::vector<unsigned int> perm;
  switch (side.type())
    {
    case EDGE2: perm = {1, 0}; break;
    case EDGE3: perm = {1, 0, 2}; break;
    case TRI3:  perm = how == 1 ? std::vector<unsigned int>{1, 2, 0} : std::vector<unsigned int>{0, 2, 1}; break;
    case QUAD4: perm = how == 1 ? std::vector<unsigned int>{1, 2, 3, 0} : std::vector<unsigned int>{0, 3, 2, 1}; break;
    case QUAD9: perm = how == 1 ? std::vector<unsigned int>{1, 2, 3, 0, 5, 6, 7, 4, 8}
                                : std::vector<unsigned int>{0, 3, 2, 1, 7, 6, 5, 4, 8}; break;
    default: libmesh_error_msg("permute_nodes: no permutation for " << Utility::enum_to_string(side.type()));
    }
  if (!how)
    return;
  std::vector<Node *> old(side.n_nodes());
  for (const auto i : make_range(side.n_nodes()))
    old[i] = side.node_ptr(i);
  for (const auto i : make_range(side.n_nodes()))
    side.set_node(i, old[perm[i]]);
}

// Adds a lower-dimensional element, in subdomain sid, on each side of
// an active element with no neighbor there and, if bid isn't
// invalid_id, boundary id bid; the side's interior parent is the
// element.  If permute, the sides' nodes are kept, rotated and
// reversed in turn.
inline void
add_lower_d(MeshBase & mesh, const boundary_id_type bid, const subdomain_id_type sid,
            const bool permute = false)
{
  const BoundaryInfo & bi = mesh.get_boundary_info();
  std::vector<std::pair<Elem *, unsigned int>> sides;
  for (Elem * elem : mesh.active_element_ptr_range())
    for (auto s : elem->side_index_range())
      if (!elem->neighbor_ptr(s) &&
          (bid == BoundaryInfo::invalid_id || bi.has_boundary_id(elem, s, bid)))
        sides.emplace_back(elem, s);
  unsigned int n = 0;
  for (const auto & [elem, s] : sides)
    {
      std::unique_ptr<Elem> side = elem->build_side_ptr(s);
      side->set_interior_parent(elem);
      side->subdomain_id() = sid;
      if (permute)
        permute_nodes(*side, n++ % 3);
      mesh.add_elem(std::move(side));
    }
  mesh.prepare_for_use();
}

// A 4 x 3 QUAD4 square with every other quad split into two TRI3s,
// in subdomain 1
inline void
build_mixed_square(UnstructuredMesh & mesh)
{
  MeshTools::Generation::build_square(mesh, 4, 3, 0., 1., 0., 1., QUAD4);
  std::vector<Elem *> split;
  for (Elem * elem : mesh.element_ptr_range())
    if (elem->id() % 2 == 0)
      split.push_back(elem);
  for (Elem * quad : split)
    {
      Node * n[4];
      for (const auto i : make_range(std::size(n)))
        n[i] = quad->node_ptr(i);
      mesh.delete_elem(quad);
      for (const auto & tri : {std::array<unsigned int, 3>{0, 1, 2}, std::array<unsigned int, 3>{0, 2, 3}})
        {
          std::unique_ptr<Elem> t = Elem::build(TRI3);
          for (const auto i : index_range(tri))
            t->set_node(i, n[tri[i]]);
          t->subdomain_id() = 1;
          mesh.add_elem(std::move(t));
        }
    }
  mesh.prepare_for_use();
}

// Every type with a Lagrange map the device code supports, as
// libMesh::Kokkos::LagrangeElemTypes lists them
constexpr ElemType lagrange_types[] = {EDGE2, EDGE3, EDGE4,
                                       TRI3, TRI6, TRI7, QUAD4, QUAD8, QUAD9,
                                       TET4, TET10, TET14, HEX8, HEX20, HEX27,
                                       PRISM6, PRISM15, PRISM18, PRISM20, PRISM21,
                                       PYRAMID5, PYRAMID13, PYRAMID14, PYRAMID18};

// Node coordinates of one element of type t.  The distortion, at most
// 0.03 per coordinate, keeps every element valid but breaks the
// reference element's symmetries; its sines' coefficients are
// arbitrary, chosen only so no two nodes or directions move alike.
// If tilt, a 1D or 2D element is then tilted into 3D: the tilt shears
// z by x and y, so the element leaves its plane, and then moves each
// node out of the sheared plane by up to 0.04, so a 2D element is
// curved in 3D; an edge is sheared in y too.
inline std::vector<Point>
element_nodes(const ElemType t, const bool tilt)
{
  const Elem & ref = ReferenceElem::get(t);
  const unsigned int dim = ref.dim();
  std::vector<Point> pts;
  for (const auto n : make_range(ref.n_nodes()))
    {
      Point x = ref.point(n);
      for (const auto d : make_range(dim))
        x(d) += 0.03 * std::sin(1.7 * n + 0.9 * d + 0.3);
      if (tilt && dim < 3)
        {
          x(2) += 0.3 * x(0) - 0.2 * (dim > 1 ? x(1) : 0.) + 0.04 * std::cos(2.1 * n);
          if (dim == 1)
            x(1) += 0.5 * x(0);
        }
      pts.push_back(x);
    }
  return pts;
}

// Adds to mesh one element of each of types, with the nodes
// element_nodes(t, tilt) gives, each element 3 apart in x, farther
// than any element reaches, so no two share a node or a side.  The
// caller prepares the mesh.
template <typename Types>
inline void
add_lagrange_elements(MeshBase & mesh, const Types & types, const bool tilt)
{
  dof_id_type node_id = mesh.max_node_id(), elem_id = mesh.max_elem_id();
  for (const ElemType t : types)
    {
      std::unique_ptr<Elem> owned = Elem::build(t);
      owned->set_id(elem_id);
      Elem * elem = mesh.add_elem(std::move(owned));
      unsigned int n = 0;
      for (Point p : element_nodes(t, tilt))
        {
          p(0) += 3. * elem_id;
          elem->set_node(n++, mesh.add_point(p, node_id++));
        }
      ++elem_id;
    }
}

inline int
run_all_oracles(const Parallel::Communicator & comm)
{
  Counts c;

  // A distorted HEX27 cube in three subdomains, with a second id on
  // the sides of boundary 0, a node set from the side sets, and
  // alternating p levels
  {
    ReplicatedMesh mesh(comm);
    // Unequal numbers of elements per direction, so no two directions
    // look alike; distort() moves each interior node by up to 0.15 of
    // the smallest element size around it, enough to make each map
    // nonaffine and little enough to leave every element valid
    MeshTools::Generation::build_cube(mesh, 3, 2, 2, 0., 1., 0., 1., 0., 1., HEX27);
    MeshTools::Modification::distort(mesh, 0.15, /*perturb_boundary=*/false);
    BoundaryInfo & bi = mesh.get_boundary_info();
    std::vector<boundary_id_type> ids;
    for (Elem * elem : mesh.active_element_ptr_range())
      {
        elem->subdomain_id() = elem->id() % 3;
#ifdef LIBMESH_ENABLE_AMR
        elem->hack_p_level(elem->id() % 2);
#endif
        for (auto s : elem->side_index_range())
          {
            bi.boundary_ids(elem, s, ids);
            if (std::find(ids.begin(), ids.end(), 0) != ids.end())
              bi.add_side(elem, s, 7);
          }
      }
    bi.build_node_list_from_side_list();

    MeshMirror mirror(mesh);
    check_mesh(mesh, mirror, c);

    // Then QUAD9 faces on boundary 7, in subdomain 10, and on boundary
    // 2, in subdomain 11, rotated and reversed
    add_lower_d(mesh, 7, 10);
    add_lower_d(mesh, 2, 11, /*permute=*/true);
    MeshMirror with_faces(mesh);
    check_mesh(mesh, with_faces, c);
  }

  // EDGE2 elements on edges of a HEX8 cube, the hexes their interior
  // parents: a parent with no side that is the element
  {
    ReplicatedMesh mesh(comm);
    MeshTools::Generation::build_cube(mesh, 2, 2, 1, 0., 1., 0., 1., 0., 1., HEX8);
    std::vector<std::unique_ptr<Elem>> edges;
    // Every third edge of each hex, so some hex edges have elements
    // and some don't
    for (Elem * elem : mesh.active_element_ptr_range())
      for (unsigned int e = 0; e != elem->n_edges(); e += 3)
        {
          edges.push_back(elem->build_edge_ptr(e));
          edges.back()->set_interior_parent(elem);
          edges.back()->subdomain_id() = 3;
        }
    for (auto & edge : edges)
      mesh.add_elem(std::move(edge));
    mesh.prepare_for_use();
    MeshMirror mirror(mesh);
    check_mesh(mesh, mirror, c);
  }

  // A mixed TRI3 / QUAD4 square: neighbors across buckets; then with
  // EDGE2 sides on its boundary
  {
    ReplicatedMesh mesh(comm);
    build_mixed_square(mesh);
    MeshMirror mirror(mesh);
    check_mesh(mesh, mirror, c);

    add_lower_d(mesh, BoundaryInfo::invalid_id, 5);
    MeshMirror with_sides(mesh);
    check_mesh(mesh, with_sides, c);
  }

#ifdef LIBMESH_ENABLE_AMR
  // The mixed square with some elements refined: sides facing finer
  // elements, and neighbors on a coarser level
  {
    ReplicatedMesh mesh(comm);
    build_mixed_square(mesh);
    MeshRefinement refinement(mesh);
    // Every fifth element, scattered so refined elements border
    // unrefined ones of both types
    for (Elem * elem : mesh.active_element_ptr_range())
      if (elem->id() % 5 == 0)
        elem->set_refinement_flag(Elem::REFINE);
    refinement.refine_elements();
    MeshMirror mirror(mesh);
    check_mesh(mesh, mirror, c);
  }

  // The mixed square with EDGE2 sides, then some elements refined:
  // MeshRefinement refines a side with its interior parent, and gives
  // the children interior parents on their own level
  {
    ReplicatedMesh mesh(comm);
    build_mixed_square(mesh);
    add_lower_d(mesh, BoundaryInfo::invalid_id, 5);
    MeshRefinement refinement(mesh);
    // Every third 2D element, scattered as above; the sides follow
    for (Elem * elem : mesh.active_element_ptr_range())
      if (elem->dim() == 2 && elem->id() % 3 == 0)
        elem->set_refinement_flag(Elem::REFINE);
    refinement.refine_elements();
    MeshMirror mirror(mesh);
    check_mesh(mesh, mirror, c);
  }

  // A square with one quad refined, then EDGE2 sides on the boundary
  // sides of the level 0 quads: the refined quad's sides have an
  // inactive interior parent, which MeshRefinement never leaves
  {
    ReplicatedMesh mesh(comm);
    MeshTools::Generation::build_square(mesh, 3, 2, 0., 1., 0., 1., QUAD4);
    MeshRefinement refinement(mesh);
    mesh.elem_ref(0).set_refinement_flag(Elem::REFINE);
    refinement.refine_elements();
    std::vector<std::pair<Elem *, unsigned int>> sides;
    for (Elem * elem : mesh.element_ptr_range())
      if (elem->level() == 0)
        for (auto s : elem->side_index_range())
          if (!elem->neighbor_ptr(s))
            sides.emplace_back(elem, s);
    for (const auto & [elem, s] : sides)
      {
        std::unique_ptr<Elem> side = elem->build_side_ptr(s);
        side->set_interior_parent(elem);
        side->subdomain_id() = 5;
        mesh.add_elem(std::move(side));
      }
    mesh.prepare_for_use();
    MeshMirror mirror(mesh);
    check_mesh(mesh, mirror, c);
  }
#endif

  // A distributed HEX8 cube: on several processors, ghosts whose
  // neighbors are remote.  4 x 3 x 3, so that on a few processors each
  // owns interior elements as well as ghosts.
  {
    DistributedMesh mesh(comm);
    MeshTools::Generation::build_cube(mesh, 4, 3, 3, 0., 1., 0., 1., 0., 1., HEX8);
    MeshMirror mirror(mesh);
    check_mesh(mesh, mirror, c);
  }

  // One element of every Lagrange type, apart from each other, the 1D
  // and 2D ones tilted into 3D; two ids on side 0, another on the
  // last side; then the nodes moved
  {
    ReplicatedMesh mesh(comm, 3);
    add_lagrange_elements(mesh, lagrange_types, /*tilt=*/true);
    for (Elem * elem : mesh.element_ptr_range())
      elem->subdomain_id() = elem->id() % 4;
    mesh.prepare_for_use();
    BoundaryInfo & bi = mesh.get_boundary_info();
    for (Elem * elem : mesh.active_element_ptr_range())
      {
        bi.add_side(elem, 0, 1);
        bi.add_side(elem, 0, 2);
        bi.add_side(elem, elem->n_sides() - 1, 5);
        bi.add_node(elem->node_ptr(0), 3);
      }

    MeshMirror mirror(mesh);
    check_mesh(mesh, mirror, c);

    // A different factor per direction, so every point and JxW
    // changes and a point not updated is caught
    MeshTools::Modification::scale(mesh, 2., 0.5, 1.5);
    mirror.update_points(mesh);
    check_mesh(mesh, mirror, c);
  }

  // Each processor checked its own mirror
  comm.sum(c.n_fail);
  for (std::size_t * n : {&c.n_elem, &c.n_ghosts, &c.n_sides, &c.n_links, &c.n_node_elems,
                          &c.n_lower_d, &c.n_points})
    comm.sum(*n);
  for (auto & row : c.kinds)
    for (std::size_t & n : row)
      comm.sum(n);
  if (comm.rank() == 0)
    std::printf("[mesh_mirror_oracle] %s  (%d failures on %u processors; %zu elements, "
                "%zu ghosts, %zu sides, %zu neighbor links, %zu node elements, "
                "%zu lower-d elements, %zu points)\n",
                c.n_fail ? "FAIL" : "PASS", c.n_fail,
                unsigned(comm.size()), c.n_elem, c.n_ghosts, c.n_sides, c.n_links,
                c.n_node_elems, c.n_lower_d, c.n_points);
  if (comm.rank() == 0)
    for (const auto i : make_range(std::size(c.kinds)))
      std::printf("[mesh_mirror_oracle] %s links: %zu with a side, %zu without, %zu none, "
                  "%zu remote, %zu refined\n", i ? "interior parent" : "neighbor",
                  c.kinds[i][0], c.kinds[i][1], c.kinds[i][2], c.kinds[i][3], c.kinds[i][4]);
  return c.n_fail;
}

} // namespace KokkosMeshMirrorOracle
} // namespace libMeshTest

#endif // KOKKOS_MESH_MIRROR_ORACLE_H
