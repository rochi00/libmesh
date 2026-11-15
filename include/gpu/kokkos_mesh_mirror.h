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



#ifndef LIBMESH_KOKKOS_MESH_MIRROR_H
#define LIBMESH_KOKKOS_MESH_MIRROR_H

// Local includes
#include "libmesh/libmesh_common.h"
#include "libmesh/boundary_info.h"
#include "libmesh/elem.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/mesh_base.h"
#include "libmesh/remote_elem.h"

#define PETSC_SKIP_CXX_COMPLEX_FIX 1
#include "libmesh/ignore_warnings.h"
#include <Kokkos_Core.hpp>
#include "libmesh/restore_warnings.h"
#undef __CUDACC_VER__

// C++ includes
#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace libMesh::Kokkos
{

/**
 * A copy in device memory of the active local elements of a mesh and
 * the ghost elements around them: the nodes' coordinates, each
 * element's nodes, subdomain, p level and id, the boundary ids of its
 * sides and nodes, its neighbor across each side, the interior parent
 * of each lower-dimensional element, and the elements of each node.
 *
 * Elements are grouped in buckets by type, so that every element a
 * kernel launched on a bucket visits has the same topology; elements
 * with an interior parent (lower-dimensional elements, as mortar
 * builds) have buckets of their own.  Nothing the type determines is
 * stored per element: kernels get the number of nodes, the sides and
 * their nodes from Elem's static tables.  Nor is any geometry stored:
 * kernels compute it from the coordinates, with the map functions
 * shared with FEMap, through the node accessor a bucket gives each
 * element.
 *
 * A bucket holds its local elements first, then its ghosts: the
 * active elements of other processors that share a node with an
 * active local element, which a mesh with the default ghosting holds.
 * Each of the two parts is sorted by subdomain, then by id.  Nodes are
 * numbered in the order the local elements of the buckets, then their
 * ghosts, first reach them, so the nodes of local elements come first.
 * Indices into the mirror are 32 bit.  Ids and the maps from ids to
 * indices stay on the host.
 *
 * The mirror is a snapshot of the mesh it was built from.  If the
 * mesh's nodes move, update_points() copies their coordinates again;
 * any other change of the mesh needs a new mirror.
 */
class MeshMirror
{
public:
  using memory_space = ::Kokkos::DefaultExecutionSpace::memory_space;
  using index_type = std::uint32_t;
  using points_view = ::Kokkos::View<const Real * [LIBMESH_DIM], ::Kokkos::LayoutRight, memory_space>;

  static constexpr index_type invalid_index = std::numeric_limits<index_type>::max();

  /**
   * A mirrored element, side of an element, or the reason there is
   * none: what lies across a side, and the interior parent of a
   * lower-dimensional element.
   */
  struct Link
  {
    // bucket when no element is there: a side on the mesh's boundary,
    // or an element with no interior parent
    static constexpr unsigned char none = 0xFF;
    // bucket when the element there isn't mirrored: remote, or a
    // ghost's neighbor beyond the ghost layer
    static constexpr unsigned char remote = 0xFE;
    // bucket when several finer elements are there: the host's link
    // is to an inactive element
    static constexpr unsigned char refined = 0xFD;
    // side when the linked element has no side that is this element:
    // an interior parent that is coarser, or of two more dimensions
    static constexpr unsigned char invalid_side = 0xFF;

    index_type elem = invalid_index;
    unsigned char bucket = none;
    unsigned char side = invalid_side;

    KOKKOS_INLINE_FUNCTION
    bool valid () const { return bucket < refined; }
  };

  /**
   * Node \p node of element \p elem of bucket \p bucket: one entry of
   * the elements of a node.
   */
  struct NodeElem
  {
    index_type elem;
    unsigned char bucket;
    unsigned char node;

    // The order of a node's entries
    KOKKOS_INLINE_FUNCTION
    std::uint64_t order () const
    { return (std::uint64_t(bucket) << 40) | (std::uint64_t(elem) << 8) | node; }
  };

  /**
   * The elements of each node, as a kernel captures them: those of
   * node n are entries(i) for i in [offsets(n), offsets(n+1)), in
   * increasing (bucket, elem, node) order.  The list of a node of a
   * local element is complete; that of a node only ghosts reach holds
   * just the mirrored elements.
   */
  struct NodeElems
  {
    ::Kokkos::View<const index_type *, memory_space> offsets;
    ::Kokkos::View<const NodeElem *, memory_space> entries;

    KOKKOS_INLINE_FUNCTION
    index_type n_elems (const index_type n) const
    { return offsets(n+1) - offsets(n); }

    KOKKOS_INLINE_FUNCTION
    NodeElem operator() (const index_type n, const index_type i) const
    { return entries(offsets(n) + i); }
  };

  /**
   * The point of node i of an element of a bucket, in the reference
   * configuration: the accessor LagrangeMapEval and the map functions
   * read the element's nodes through.
   */
  struct ReferenceNodes
  {
    points_view points;
    ::Kokkos::View<const index_type **, ::Kokkos::LayoutRight, memory_space> nodes;
    index_type elem;

    KOKKOS_INLINE_FUNCTION
    Point operator() (const unsigned int i) const
    {
      const index_type n = nodes(elem, i);
      Point p;
      for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
        p(d) = points(n, d);
      return p;
    }
  };

  /**
   * The elements of one type.  It holds only views, so a kernel
   * captures it by value.
   */
  struct Bucket
  {
    ElemType type = INVALID_ELEM;
    // Whether the elements have interior parents
    bool lower_d = false;
    // Elements [0, n_local) are local, [n_local, n_elem) ghosts
    index_type n_elem = 0;
    index_type n_local = 0;
    // Elem::type_to_n_sides_map[type], which the boundary keys use
    unsigned int n_sides = 0;

    points_view points;

    // The mirror's node index of each element's nodes, [elem][node]
    ::Kokkos::View<const index_type **, ::Kokkos::LayoutRight, memory_space> nodes;
    ::Kokkos::View<const subdomain_id_type *, memory_space> subdomain;
    ::Kokkos::View<const unsigned char *, memory_space> p_level;
    ::Kokkos::View<const dof_id_type *, memory_space> id;

    // What lies across each side, [elem][side], and the side of the
    // neighbor that faces back
    ::Kokkos::View<const Link **, ::Kokkos::LayoutRight, memory_space> neighbors;

    // Each element's interior parent and the side of it the element
    // is; empty unless lower_d
    ::Kokkos::View<const Link *, memory_space> interior_parent;

    // The sides with boundary index b are boundary_keys(i) for i in
    // [boundary_offsets(b), boundary_offsets(b+1)), each the key
    // elem * n_sides + side, in increasing order.  A side with
    // several ids appears once under each.
    ::Kokkos::View<const index_type *, memory_space> boundary_offsets;
    ::Kokkos::View<const index_type *, memory_space> boundary_keys;

    KOKKOS_INLINE_FUNCTION
    bool is_ghost (const index_type e) const { return e >= n_local; }

    KOKKOS_INLINE_FUNCTION
    Link neighbor (const index_type e, const unsigned int s) const
    { return neighbors(e, s); }

    KOKKOS_INLINE_FUNCTION
    ReferenceNodes reference_nodes (const index_type e) const
    { return ReferenceNodes{points, nodes, e}; }

    /**
     * The Lagrange map of element \p e, on its nodes' coordinates.
     */
    KOKKOS_INLINE_FUNCTION
    LagrangeMapEval<ReferenceNodes> reference_map (const index_type e) const
    { return lagrange_map_eval(type, this->reference_nodes(e)); }

    KOKKOS_INLINE_FUNCTION
    index_type n_boundary_sides (const unsigned int b) const
    { return boundary_offsets(b+1) - boundary_offsets(b); }

    /**
     * The element and side of the \p i th side with boundary index
     * \p b.
     */
    KOKKOS_INLINE_FUNCTION
    void boundary_side (const unsigned int b, const index_type i,
                        index_type & e, unsigned int & s) const
    {
      const index_type key = boundary_keys(boundary_offsets(b) + i);
      e = key / n_sides;
      s = key % n_sides;
    }

    /**
     * Whether side \p s of element \p e has boundary index \p b: a
     * binary search of the bucket's sides with that index.
     */
    KOKKOS_INLINE_FUNCTION
    bool has_boundary_index (const index_type e, const unsigned int s,
                             const unsigned int b) const
    {
      const index_type key = e * n_sides + s;
      index_type lo = boundary_offsets(b), hi = boundary_offsets(b+1);
      while (lo < hi)
        {
          const index_type mid = lo + (hi - lo) / 2;
          if (boundary_keys(mid) < key)
            lo = mid + 1;
          else
            hi = mid;
        }
      return lo != boundary_offsets(b+1) && boundary_keys(lo) == key;
    }
  };

  /**
   * Copies the active local elements of \p mesh, the ghosts around
   * them, their nodes, and the boundary ids \p mesh's BoundaryInfo
   * gives their sides and nodes.  Every element must have a Lagrange
   * map onto a type Elem::master_point() knows, or be a NodeElem.
   */
  explicit MeshMirror (const MeshBase & mesh);

  /**
   * Copies the coordinates of the mirrored nodes of \p mesh again,
   * after they moved.  The mesh must otherwise be unchanged.
   */
  void update_points (const MeshBase & mesh);

  unsigned int n_buckets () const { return cast_int<unsigned int>(_buckets.size()); }

  const Bucket & bucket (const unsigned int k) const { return _buckets[k]; }

  index_type n_nodes () const { return cast_int<index_type>(_node_ids.size()); }

  /**
   * The number of nodes of local elements, which are nodes
   * [0, n_local_nodes())
   */
  index_type n_local_nodes () const { return _n_local_nodes; }

  const points_view & points () const { return _points; }

  /**
   * The ids of the mirrored nodes, by node index, on the host
   */
  const std::vector<dof_id_type> & node_ids () const { return _node_ids; }

  /**
   * The elements of each node
   */
  const NodeElems & node_elems () const { return _node_elems; }

  /**
   * The bucket of element type \p t, of elements with interior parents
   * if \p lower_d, or invalid_uint if there is none
   */
  unsigned int bucket_of_type (const ElemType t, const bool lower_d = false) const;

  /**
   * The bucket and index of the element with id \p id, or
   * (invalid_uint, invalid_index) if it isn't mirrored
   */
  std::pair<unsigned int, index_type> elem_index (const dof_id_type id) const;

  /**
   * The index of the node with id \p id, or invalid_index if it isn't
   * mirrored
   */
  index_type node_index (const dof_id_type id) const;

  /**
   * The range [first, last) of the local elements, or the ghosts if
   * \p ghosts, of subdomain \p sid in bucket \p k; empty if it has none
   */
  std::pair<index_type, index_type> subdomain_range (const unsigned int k,
                                                     const subdomain_id_type sid,
                                                     const bool ghosts = false) const;

  /**
   * The boundary ids the mirror indexes, in increasing order: boundary
   * index b stands for boundary_ids()[b]
   */
  const std::vector<boundary_id_type> & boundary_ids () const { return _boundary_ids; }

  /**
   * The boundary index of boundary id \p id, or invalid_uint if no
   * mirrored side or node has it
   */
  unsigned int boundary_index (const boundary_id_type id) const;

  /**
   * The nodes with boundary index b are boundary_nodes()(i) for i in
   * [boundary_node_offsets()(b), boundary_node_offsets()(b+1)), in
   * increasing order.
   */
  const ::Kokkos::View<const index_type *, memory_space> & boundary_node_offsets () const
  { return _boundary_node_offsets; }

  const ::Kokkos::View<const index_type *, memory_space> & boundary_nodes () const
  { return _boundary_nodes; }

  /**
   * The link to \p other, a neighbor or interior parent, and its side
   * \p side (invalid_uint if none): which element of the mirror it is,
   * or why there is none.
   */
  Link link (const Elem * other, const unsigned int side) const;

  /**
   * Builds the elements of each of \p n_nodes nodes, on the device,
   * from the nodes of the elements of \p buckets.  Public only because
   * device lambdas can't be defined in a constructor.
   */
  static NodeElems build_node_elems (const std::vector<Bucket> & buckets,
                                     const index_type n_nodes);

private:
  template <typename T>
  static index_type checked_index (const T i)
  {
    libmesh_error_msg_if(i >= T(invalid_index),
                         "MeshMirror indices are 32 bit; this mesh is too big for one");
    return index_type(i);
  }

  // The subdomains of the local elements, or the ghosts, of a bucket in
  // increasing order, and where each one's elements start; the last
  // offset is where the part ends
  struct SubdomainRanges
  {
    std::vector<subdomain_id_type> ids;
    std::vector<index_type> offsets;
  };

  std::vector<Bucket> _buckets;
  std::vector<SubdomainRanges> _local_subdomains, _ghost_subdomains;

  points_view _points;
  index_type _n_local_nodes = 0;
  std::vector<dof_id_type> _node_ids;
  std::unordered_map<dof_id_type, index_type> _node_index;
  std::unordered_map<dof_id_type, std::pair<unsigned int, index_type>> _elem_index;
  NodeElems _node_elems;

  std::vector<boundary_id_type> _boundary_ids;
  ::Kokkos::View<const index_type *, memory_space> _boundary_node_offsets;
  ::Kokkos::View<const index_type *, memory_space> _boundary_nodes;
};



inline
MeshMirror::MeshMirror (const MeshBase & mesh)
{
  const auto check = [](const Elem * elem)
  {
    const ElemType t = elem->type();
    libmesh_error_msg_if(t == C0POLYGON || t == C0POLYHEDRON || elem->infinite() ||
                         t == REMOTEELEM || t == INVALID_ELEM,
                         "MeshMirror does not support elements of type " << Utility::enum_to_string(t));
    libmesh_error_msg_if(t != NODEELEM && elem->mapping_type() != LAGRANGE_MAP,
                         "MeshMirror supports only Lagrange maps");
    libmesh_error_msg_if(elem->p_level() > std::numeric_limits<unsigned char>::max(),
                         "MeshMirror stores p levels in 8 bits");
  };
  const auto is_lower_d = [](const Elem * elem)
  { return elem->dim() < LIBMESH_DIM && elem->interior_parent(); };
  // Buckets of lower-dimensional elements come after the others
  const auto key = [&is_lower_d](const Elem * elem)
  { return is_lower_d(elem) * int(INVALID_ELEM) + elem->type(); };

  // The local elements, by bucket, and their nodes
  std::vector<std::vector<const Elem *>> locals(2 * INVALID_ELEM), ghosts(2 * INVALID_ELEM);
  std::unordered_set<dof_id_type> local_nodes;
  for (const Elem * elem : mesh.active_local_element_ptr_range())
    {
      check(elem);
      locals[key(elem)].push_back(elem);
      for (const Node & node : elem->node_ref_range())
        local_nodes.insert(node.id());
    }

  // The ghosts: active elements of other processors sharing a node
  // with a local one
  for (const Elem * elem : mesh.active_element_ptr_range())
    if (elem->processor_id() != mesh.processor_id())
      for (const Node & node : elem->node_ref_range())
        if (local_nodes.count(node.id()))
          {
            check(elem);
            ghosts[key(elem)].push_back(elem);
            break;
          }

  // The buckets' elements, local then ghost, each part sorted by
  // subdomain then id
  const auto by_subdomain_id = [](const Elem * a, const Elem * b)
  { return std::make_pair(a->subdomain_id(), a->id()) <
           std::make_pair(b->subdomain_id(), b->id()); };
  std::vector<std::vector<const Elem *>> bucket_elems;
  for (int i = 0; i != 2 * INVALID_ELEM; ++i)
    {
      if (locals[i].empty() && ghosts[i].empty())
        continue;
      const unsigned int k = cast_int<unsigned int>(_buckets.size());
      libmesh_error_msg_if(k >= Link::refined, "MeshMirror stores bucket numbers in 8 bits");

      std::sort(locals[i].begin(), locals[i].end(), by_subdomain_id);
      std::sort(ghosts[i].begin(), ghosts[i].end(), by_subdomain_id);

      Bucket bucket;
      bucket.type = ElemType(i % INVALID_ELEM);
      bucket.lower_d = i >= INVALID_ELEM;
      bucket.n_local = checked_index(locals[i].size());
      bucket.n_elem = checked_index(locals[i].size() + ghosts[i].size());
      bucket.n_sides = Elem::type_to_n_sides_map[bucket.type];
      checked_index(std::size_t(bucket.n_elem) * std::max(bucket.n_sides, 1u));
      _buckets.push_back(bucket);

      std::vector<const Elem *> elems = std::move(locals[i]);
      elems.insert(elems.end(), ghosts[i].begin(), ghosts[i].end());
      for (index_type e = 0; e != elems.size(); ++e)
        _elem_index.emplace(elems[e]->id(), std::make_pair(k, e));

      // Each part's subdomain ranges
      SubdomainRanges local_ranges, ghost_ranges;
      for (index_type e = 0; e != bucket.n_elem; ++e)
        {
          SubdomainRanges & r = e < bucket.n_local ? local_ranges : ghost_ranges;
          if (r.ids.empty() || r.ids.back() != elems[e]->subdomain_id())
            {
              r.ids.push_back(elems[e]->subdomain_id());
              r.offsets.push_back(e);
            }
        }
      local_ranges.offsets.push_back(bucket.n_local);
      ghost_ranges.offsets.push_back(bucket.n_elem);
      _local_subdomains.push_back(std::move(local_ranges));
      _ghost_subdomains.push_back(std::move(ghost_ranges));

      bucket_elems.push_back(std::move(elems));
    }

  // Number the nodes of the local elements, then those only ghosts
  // reach
  for (const bool ghost : {false, true})
    {
      for (unsigned int k = 0; k != _buckets.size(); ++k)
        {
          const index_type first = ghost ? _buckets[k].n_local : 0;
          const index_type last = ghost ? _buckets[k].n_elem : _buckets[k].n_local;
          for (index_type e = first; e != last; ++e)
            for (const Node & node : bucket_elems[k][e]->node_ref_range())
              if (_node_index.emplace(node.id(), checked_index(_node_ids.size())).second)
                _node_ids.push_back(node.id());
        }
      if (!ghost)
        _n_local_nodes = checked_index(_node_ids.size());
    }

  // The mirror's boundary ids: those of the mirrored sides and nodes
  const BoundaryInfo & bi = mesh.get_boundary_info();
  std::vector<boundary_id_type> ids;
  for (const auto & elems : bucket_elems)
    for (const Elem * elem : elems)
      for (auto s : elem->side_index_range())
        {
          bi.boundary_ids(elem, s, ids);
          _boundary_ids.insert(_boundary_ids.end(), ids.begin(), ids.end());
        }
  for (const dof_id_type node_id : _node_ids)
    {
      bi.boundary_ids(mesh.node_ptr(node_id), ids);
      _boundary_ids.insert(_boundary_ids.end(), ids.begin(), ids.end());
    }
  std::sort(_boundary_ids.begin(), _boundary_ids.end());
  _boundary_ids.erase(std::unique(_boundary_ids.begin(), _boundary_ids.end()), _boundary_ids.end());
  const unsigned int n_bids = cast_int<unsigned int>(_boundary_ids.size());

  for (unsigned int k = 0; k != _buckets.size(); ++k)
    {
      Bucket & bucket = _buckets[k];
      const std::vector<const Elem *> & elems = bucket_elems[k];
      const index_type n_elem = bucket.n_elem;
      const unsigned int n_nodes = Elem::type_to_n_nodes_map[bucket.type];
      const unsigned int n_sides = bucket.n_sides;

      ::Kokkos::View<index_type **, ::Kokkos::LayoutRight, memory_space>
        nodes("MeshMirror::nodes", n_elem, n_nodes);
      ::Kokkos::View<subdomain_id_type *, memory_space> subdomain("MeshMirror::subdomain", n_elem);
      ::Kokkos::View<unsigned char *, memory_space> p_level("MeshMirror::p_level", n_elem);
      ::Kokkos::View<dof_id_type *, memory_space> id("MeshMirror::id", n_elem);
      ::Kokkos::View<Link **, ::Kokkos::LayoutRight, memory_space>
        neighbors("MeshMirror::neighbors", n_elem, n_sides);
      ::Kokkos::View<Link *, memory_space>
        interior_parent("MeshMirror::interior_parent", bucket.lower_d ? n_elem : 0);
      auto h_nodes = ::Kokkos::create_mirror_view(nodes);
      auto h_subdomain = ::Kokkos::create_mirror_view(subdomain);
      auto h_p_level = ::Kokkos::create_mirror_view(p_level);
      auto h_id = ::Kokkos::create_mirror_view(id);
      auto h_neighbors = ::Kokkos::create_mirror_view(neighbors);
      auto h_interior_parent = ::Kokkos::create_mirror_view(interior_parent);

      // The bucket's sides by boundary index, as (index, key) pairs
      std::vector<std::pair<unsigned int, index_type>> sides;

      for (index_type e = 0; e != n_elem; ++e)
        {
          const Elem * elem = elems[e];
          h_subdomain(e) = elem->subdomain_id();
          h_p_level(e) = cast_int<unsigned char>(elem->p_level());
          h_id(e) = elem->id();

          for (unsigned int i = 0; i != n_nodes; ++i)
            h_nodes(e, i) = _node_index[elem->node_id(i)];

          for (auto s : elem->side_index_range())
            {
              bi.boundary_ids(elem, s, ids);
              for (const boundary_id_type b : ids)
                sides.emplace_back(this->boundary_index(b), e * n_sides + s);

              const Elem * nb = elem->neighbor_ptr(s);
              const bool linked = nb && nb != remote_elem && nb->active();
              h_neighbors(e, s) = this->link(nb, linked ? nb->which_neighbor_am_i(elem) : invalid_uint);
            }

          if (bucket.lower_d)
            {
              // Only a parent of one more dimension has a side that is
              // the element: an edge lies on two faces of a cell, which
              // which_side_am_i() asserts against
              const Elem * ip = elem->interior_parent();
              const bool side_of = ip != remote_elem && ip->active() && ip->dim() == elem->dim() + 1;
              h_interior_parent(e) = this->link(ip, side_of ? ip->which_side_am_i(elem) : invalid_uint);
            }
        }

      std::sort(sides.begin(), sides.end());
      ::Kokkos::View<index_type *, memory_space> offsets("MeshMirror::boundary_offsets", n_bids + 1);
      ::Kokkos::View<index_type *, memory_space> keys("MeshMirror::boundary_keys", sides.size());
      auto h_offsets = ::Kokkos::create_mirror_view(offsets);
      auto h_keys = ::Kokkos::create_mirror_view(keys);
      for (unsigned int b = 0; b != n_bids + 1; ++b)
        h_offsets(b) = 0;
      for (std::size_t i = 0; i != sides.size(); ++i)
        {
          ++h_offsets(sides[i].first + 1);
          h_keys(i) = sides[i].second;
        }
      for (unsigned int b = 0; b != n_bids; ++b)
        h_offsets(b+1) += h_offsets(b);

      ::Kokkos::deep_copy(nodes, h_nodes);
      ::Kokkos::deep_copy(subdomain, h_subdomain);
      ::Kokkos::deep_copy(p_level, h_p_level);
      ::Kokkos::deep_copy(id, h_id);
      ::Kokkos::deep_copy(neighbors, h_neighbors);
      ::Kokkos::deep_copy(interior_parent, h_interior_parent);
      ::Kokkos::deep_copy(offsets, h_offsets);
      ::Kokkos::deep_copy(keys, h_keys);

      bucket.nodes = nodes;
      bucket.subdomain = subdomain;
      bucket.p_level = p_level;
      bucket.id = id;
      bucket.neighbors = neighbors;
      bucket.interior_parent = interior_parent;
      bucket.boundary_offsets = offsets;
      bucket.boundary_keys = keys;
    }

  // The nodes by boundary index
  std::vector<std::vector<index_type>> nodes_of(n_bids);
  for (index_type n = 0; n != _node_ids.size(); ++n)
    {
      bi.boundary_ids(mesh.node_ptr(_node_ids[n]), ids);
      for (const boundary_id_type b : ids)
        nodes_of[this->boundary_index(b)].push_back(n);
    }
  ::Kokkos::View<index_type *, memory_space> node_offsets("MeshMirror::boundary_node_offsets", n_bids + 1);
  auto h_node_offsets = ::Kokkos::create_mirror_view(node_offsets);
  h_node_offsets(0) = 0;
  for (unsigned int b = 0; b != n_bids; ++b)
    h_node_offsets(b+1) = h_node_offsets(b) + checked_index(nodes_of[b].size());
  ::Kokkos::View<index_type *, memory_space> bnodes("MeshMirror::boundary_nodes", h_node_offsets(n_bids));
  auto h_bnodes = ::Kokkos::create_mirror_view(bnodes);
  for (unsigned int b = 0; b != n_bids; ++b)
    std::copy(nodes_of[b].begin(), nodes_of[b].end(), h_bnodes.data() + h_node_offsets(b));
  ::Kokkos::deep_copy(node_offsets, h_node_offsets);
  ::Kokkos::deep_copy(bnodes, h_bnodes);
  _boundary_node_offsets = node_offsets;
  _boundary_nodes = bnodes;

  _node_elems = build_node_elems(_buckets, this->n_nodes());

  this->update_points(mesh);
}



inline
MeshMirror::NodeElems
MeshMirror::build_node_elems (const std::vector<Bucket> & buckets,
                              const index_type n_nodes)
{
  using exec_space = ::Kokkos::DefaultExecutionSpace;

  // Count each node's elements into offsets(n+1), then sum
  ::Kokkos::View<index_type *, memory_space> offsets("MeshMirror::node_elem_offsets", n_nodes + 1);
  for (const Bucket & b : buckets)
    {
      const unsigned int nn = Elem::type_to_n_nodes_map[b.type];
      const auto nodes = b.nodes;
      ::Kokkos::parallel_for(
        "MeshMirror::count_node_elems",
        ::Kokkos::RangePolicy<exec_space>(0, std::size_t(b.n_elem) * nn),
        KOKKOS_LAMBDA(const std::size_t i) {
          ::Kokkos::atomic_add(&offsets(nodes(i / nn, i % nn) + 1), index_type(1));
        });
    }
  ::Kokkos::parallel_scan(
    "MeshMirror::sum_node_elems",
    ::Kokkos::RangePolicy<exec_space>(0, n_nodes + 1),
    KOKKOS_LAMBDA(const index_type n, index_type & sum, const bool final) {
      const index_type count = offsets(n);
      sum += count;
      if (final)
        offsets(n) = sum;
    });

  index_type n_entries = 0;
  ::Kokkos::deep_copy(n_entries, ::Kokkos::subview(offsets, n_nodes));
  ::Kokkos::View<NodeElem *, memory_space> entries("MeshMirror::node_elems", n_entries);

  // Fill each node's entries in whatever order the atomics give
  ::Kokkos::View<index_type *, memory_space> next("MeshMirror::node_elem_next", n_nodes);
  ::Kokkos::deep_copy(next, ::Kokkos::subview(offsets, ::Kokkos::make_pair(index_type(0), n_nodes)));
  for (unsigned int k = 0; k != buckets.size(); ++k)
    {
      const Bucket & b = buckets[k];
      const unsigned int nn = Elem::type_to_n_nodes_map[b.type];
      const auto nodes = b.nodes;
      const unsigned char bucket = cast_int<unsigned char>(k);
      ::Kokkos::parallel_for(
        "MeshMirror::fill_node_elems",
        ::Kokkos::RangePolicy<exec_space>(0, std::size_t(b.n_elem) * nn),
        KOKKOS_LAMBDA(const std::size_t i) {
          const index_type e = index_type(i / nn);
          const unsigned int j = i % nn;
          const index_type slot = ::Kokkos::atomic_fetch_add(&next(nodes(e, j)), index_type(1));
          entries(slot) = NodeElem{e, bucket, (unsigned char)j};
        });
    }

  // Then sort each node's entries, which are few
  ::Kokkos::parallel_for(
    "MeshMirror::sort_node_elems",
    ::Kokkos::RangePolicy<exec_space>(0, n_nodes),
    KOKKOS_LAMBDA(const index_type n) {
      for (index_type i = offsets(n) + 1; i < offsets(n+1); ++i)
        {
          const NodeElem x = entries(i);
          index_type j = i;
          for (; j > offsets(n) && entries(j-1).order() > x.order(); --j)
            entries(j) = entries(j-1);
          entries(j) = x;
        }
    });

  return NodeElems{offsets, entries};
}



inline
void MeshMirror::update_points (const MeshBase & mesh)
{
  ::Kokkos::View<Real * [LIBMESH_DIM], ::Kokkos::LayoutRight, memory_space>
    points("MeshMirror::points", _node_ids.size());
  auto h_points = ::Kokkos::create_mirror_view(points);
  for (std::size_t n = 0; n != _node_ids.size(); ++n)
    {
      const Point & p = mesh.point(_node_ids[n]);
      for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
        h_points(n, d) = p(d);
    }
  ::Kokkos::deep_copy(points, h_points);

  // Every bucket reads the new coordinates; the old view goes when
  // the last kernel holding it does
  _points = points;
  for (Bucket & bucket : _buckets)
    bucket.points = _points;
}



inline
MeshMirror::Link MeshMirror::link (const Elem * other, const unsigned int side) const
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
  const auto [k, e] = this->elem_index(other->id());
  if (k == invalid_uint)
    {
      l.bucket = Link::remote;
      return l;
    }
  l.elem = e;
  l.bucket = cast_int<unsigned char>(k);
  l.side = side == invalid_uint ? Link::invalid_side : cast_int<unsigned char>(side);
  return l;
}



inline
unsigned int MeshMirror::bucket_of_type (const ElemType t, const bool lower_d) const
{
  for (unsigned int k = 0; k != _buckets.size(); ++k)
    if (_buckets[k].type == t && _buckets[k].lower_d == lower_d)
      return k;
  return invalid_uint;
}



inline
std::pair<unsigned int, MeshMirror::index_type>
MeshMirror::elem_index (const dof_id_type id) const
{
  const auto it = _elem_index.find(id);
  return it == _elem_index.end() ? std::make_pair(invalid_uint, invalid_index) : it->second;
}



inline
MeshMirror::index_type MeshMirror::node_index (const dof_id_type id) const
{
  const auto it = _node_index.find(id);
  return it == _node_index.end() ? invalid_index : it->second;
}



inline
std::pair<MeshMirror::index_type, MeshMirror::index_type>
MeshMirror::subdomain_range (const unsigned int k, const subdomain_id_type sid,
                             const bool ghosts) const
{
  const SubdomainRanges & r = ghosts ? _ghost_subdomains[k] : _local_subdomains[k];
  const auto it = std::lower_bound(r.ids.begin(), r.ids.end(), sid);
  if (it == r.ids.end() || *it != sid)
    return {0, 0};
  const std::size_t i = it - r.ids.begin();
  return {r.offsets[i], r.offsets[i+1]};
}



inline
unsigned int MeshMirror::boundary_index (const boundary_id_type id) const
{
  const auto it = std::lower_bound(_boundary_ids.begin(), _boundary_ids.end(), id);
  return (it == _boundary_ids.end() || *it != id) ? invalid_uint :
    cast_int<unsigned int>(it - _boundary_ids.begin());
}

} // namespace libMesh::Kokkos

#endif // LIBMESH_KOKKOS_MESH_MIRROR_H
