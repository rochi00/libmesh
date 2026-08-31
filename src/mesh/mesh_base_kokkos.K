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

#include "libmesh/mesh_base_kokkos.h"

#include "libmesh/elem.h"
#include "libmesh/utility.h"

#include <algorithm>

namespace libMesh
{

#ifdef LIBMESH_HAVE_KOKKOS
void MeshBase::KokkosGeometryCacheDeleter::operator()(KokkosGeometryCache * cache) const
{
  delete cache;
}

const MeshBase::KokkosGeometryCache &
MeshBase::get_kokkos_geometry_cache() const
{
  if (_kokkos_geometry_cache)
    return *_kokkos_geometry_cache;

  auto cache = kokkos_geometry_cache_ptr(new KokkosGeometryCache());
  cache->host_element_ids.reserve(this->n_active_local_elem());

  for (const auto & elem : this->active_local_element_ptr_range())
    {
      cache->element_lookup.emplace(elem->id(), cast_int<unsigned int>(cache->host_element_ids.size()));
      cache->host_element_ids.push_back(elem->id());
      cache->max_nodes = std::max(cache->max_nodes, elem->n_nodes());
      cache->max_sides = std::max(cache->max_sides, elem->n_sides());
      cache->max_edges = std::max(cache->max_edges, elem->n_edges());
      cache->max_faces = std::max(cache->max_faces, elem->n_faces());

      for (unsigned int s = 0; s != elem->n_sides(); ++s)
        cache->max_side_nodes =
          std::max(cache->max_side_nodes, cast_int<unsigned int>(elem->nodes_on_side(s).size()));

      for (unsigned int n = 0; n != elem->n_nodes(); ++n)
        {
          const dof_id_type node_id = elem->node_id(n);
          if (!cache->node_lookup.count(node_id))
            {
              cache->node_lookup.emplace(node_id, cast_int<unsigned int>(cache->host_node_ids.size()));
              cache->host_node_ids.push_back(node_id);
            }
        }
    }

  cache->node_ids_dual =
    KokkosGeometryCache::node_id_dual_view("mesh_kokkos_node_ids", cache->host_node_ids.size());
  cache->element_ids_dual =
    KokkosGeometryCache::elem_id_dual_view("mesh_kokkos_element_ids", cache->host_element_ids.size());
  cache->node_coordinates_dual =
    KokkosGeometryCache::node_coord_dual_view("mesh_kokkos_node_coordinates",
                                              cache->host_node_ids.size(),
                                              LIBMESH_DIM);
  cache->element_node_ids_dual =
    KokkosGeometryCache::elem_node_id_dual_view("mesh_kokkos_element_node_ids",
                                                cache->host_element_ids.size(),
                                                cache->max_nodes);
  cache->element_types_dual =
    KokkosGeometryCache::elem_type_dual_view("mesh_kokkos_element_types",
                                             cache->host_element_ids.size());
  cache->element_mapping_types_dual =
    KokkosGeometryCache::elem_mapping_type_dual_view("mesh_kokkos_element_mapping_types",
                                                     cache->host_element_ids.size());
  cache->element_n_nodes_dual =
    KokkosGeometryCache::elem_n_nodes_dual_view("mesh_kokkos_element_n_nodes",
                                                cache->host_element_ids.size());
  cache->element_n_sides_dual =
    KokkosGeometryCache::elem_n_sides_dual_view("mesh_kokkos_element_n_sides",
                                                cache->host_element_ids.size());
  cache->element_n_edges_dual =
    KokkosGeometryCache::elem_n_edges_dual_view("mesh_kokkos_element_n_edges",
                                                cache->host_element_ids.size());
  cache->element_n_faces_dual =
    KokkosGeometryCache::elem_n_faces_dual_view("mesh_kokkos_element_n_faces",
                                                cache->host_element_ids.size());
  cache->element_side_types_dual =
    KokkosGeometryCache::elem_side_type_dual_view("mesh_kokkos_element_side_types",
                                                  cache->host_element_ids.size(),
                                                  cache->max_sides);
  cache->element_side_n_nodes_dual =
    KokkosGeometryCache::elem_side_n_nodes_dual_view("mesh_kokkos_element_side_n_nodes",
                                                     cache->host_element_ids.size(),
                                                     cache->max_sides);
  cache->element_side_local_nodes_dual =
    KokkosGeometryCache::elem_side_local_node_dual_view("mesh_kokkos_element_side_local_nodes",
                                                        cache->host_element_ids.size(),
                                                        cache->max_sides,
                                                        cache->max_side_nodes);
  cache->element_side_is_boundary_dual =
    KokkosGeometryCache::elem_side_boundary_dual_view("mesh_kokkos_element_side_is_boundary",
                                                      cache->host_element_ids.size(),
                                                      cache->max_sides);
  cache->element_edge_positive_orientations_dual =
    KokkosGeometryCache::elem_edge_orientation_dual_view("mesh_kokkos_element_edge_orientations",
                                                         cache->host_element_ids.size(),
                                                         cache->max_edges);
  cache->element_face_positive_orientations_dual =
    KokkosGeometryCache::elem_face_orientation_dual_view("mesh_kokkos_element_face_orientations",
                                                         cache->host_element_ids.size(),
                                                         cache->max_faces);
  cache->element_p_levels_dual =
    KokkosGeometryCache::elem_p_level_dual_view("mesh_kokkos_element_p_levels",
                                                cache->host_element_ids.size());
  cache->element_subdomains_dual =
    KokkosGeometryCache::elem_subdomain_dual_view("mesh_kokkos_element_subdomains",
                                                  cache->host_element_ids.size());

  auto h_node_ids = cache->node_ids_dual.view_host();
  auto h_element_ids = cache->element_ids_dual.view_host();
  auto h_node_coordinates = cache->node_coordinates_dual.view_host();
  auto h_element_node_ids = cache->element_node_ids_dual.view_host();
  auto h_element_types = cache->element_types_dual.view_host();
  auto h_element_mapping_types = cache->element_mapping_types_dual.view_host();
  auto h_element_n_nodes = cache->element_n_nodes_dual.view_host();
  auto h_element_n_sides = cache->element_n_sides_dual.view_host();
  auto h_element_n_edges = cache->element_n_edges_dual.view_host();
  auto h_element_n_faces = cache->element_n_faces_dual.view_host();
  auto h_element_side_types = cache->element_side_types_dual.view_host();
  auto h_element_side_n_nodes = cache->element_side_n_nodes_dual.view_host();
  auto h_element_side_local_nodes = cache->element_side_local_nodes_dual.view_host();
  auto h_element_side_is_boundary = cache->element_side_is_boundary_dual.view_host();
  auto h_element_edge_positive_orientations =
    cache->element_edge_positive_orientations_dual.view_host();
  auto h_element_face_positive_orientations =
    cache->element_face_positive_orientations_dual.view_host();
  auto h_element_p_levels = cache->element_p_levels_dual.view_host();
  auto h_element_subdomains = cache->element_subdomains_dual.view_host();
  cache->host_element_types.resize(cache->host_element_ids.size());
  cache->host_element_mapping_types.resize(cache->host_element_ids.size());
  cache->host_element_n_nodes.resize(cache->host_element_ids.size());
  cache->host_element_p_levels.resize(cache->host_element_ids.size());
  cache->host_element_subdomains.resize(cache->host_element_ids.size(),
                                        static_cast<subdomain_id_type>(Elem::invalid_subdomain_id));

  ::Kokkos::deep_copy(h_element_side_types, INVALID_ELEM);
  ::Kokkos::deep_copy(h_element_side_n_nodes, 0);
  ::Kokkos::deep_copy(h_element_side_local_nodes, libMesh::invalid_uint);
  ::Kokkos::deep_copy(h_element_side_is_boundary, static_cast<unsigned char>(0));
  ::Kokkos::deep_copy(h_element_edge_positive_orientations, static_cast<unsigned char>(0));
  ::Kokkos::deep_copy(h_element_face_positive_orientations, static_cast<unsigned char>(0));

  for (auto node_index : index_range(cache->host_node_ids))
    {
      const dof_id_type node_id = cache->host_node_ids[node_index];
      const Node & node = *this->query_node_ptr(node_id);
      h_node_ids(cast_int<unsigned int>(node_index)) = node_id;
      for (unsigned int component = 0; component != LIBMESH_DIM; ++component)
        h_node_coordinates(cast_int<unsigned int>(node_index), component) = node(component);
    }

  for (auto elem_index : index_range(cache->host_element_ids))
    {
      const unsigned int elem_index_uint = cast_int<unsigned int>(elem_index);
      const dof_id_type elem_id = cache->host_element_ids[elem_index];
      const Elem & elem = *this->query_elem_ptr(elem_id);
      h_element_ids(elem_index_uint) = elem_id;
      h_element_types(elem_index_uint) = elem.type();
      h_element_mapping_types(elem_index_uint) = elem.mapping_type();
      h_element_n_nodes(elem_index_uint) = elem.n_nodes();
      h_element_n_sides(elem_index_uint) = elem.n_sides();
      h_element_n_edges(elem_index_uint) = elem.n_edges();
      h_element_n_faces(elem_index_uint) = elem.n_faces();
      h_element_p_levels(elem_index_uint) = elem.p_level();
      h_element_subdomains(elem_index_uint) = elem.subdomain_id();
      cache->host_element_types[elem_index] = elem.type();
      cache->host_element_mapping_types[elem_index] = elem.mapping_type();
      cache->host_element_n_nodes[elem_index] = elem.n_nodes();
      cache->host_element_p_levels[elem_index] = elem.p_level();
      cache->host_element_subdomains[elem_index] = elem.subdomain_id();
      for (unsigned int n = 0; n != elem.n_nodes(); ++n)
        h_element_node_ids(elem_index_uint, n) =
          libmesh_map_find(cache->node_lookup, elem.node_id(n));

      for (unsigned int s = 0; s != elem.n_sides(); ++s)
        {
          const auto side_nodes = elem.nodes_on_side(s);
          h_element_side_types(elem_index_uint, s) = elem.side_type(s);
          h_element_side_n_nodes(elem_index_uint, s) =
            cast_int<unsigned int>(side_nodes.size());
          h_element_side_is_boundary(elem_index_uint, s) =
            elem.neighbor_ptr(s) == nullptr ? 1 : 0;
          for (auto local_side_node_index : index_range(side_nodes))
            h_element_side_local_nodes(elem_index_uint,
                                       s,
                                       cast_int<unsigned int>(local_side_node_index)) =
              side_nodes[local_side_node_index];
        }

      for (unsigned int edge = 0; edge != elem.n_edges(); ++edge)
        h_element_edge_positive_orientations(elem_index_uint, edge) =
          elem.positive_edge_orientation(edge) ? 1 : 0;

      for (unsigned int face = 0; face != elem.n_faces(); ++face)
        h_element_face_positive_orientations(elem_index_uint, face) =
          elem.positive_face_orientation(face) ? 1 : 0;
    }

  cache->node_ids_dual.modify_host();
  cache->element_ids_dual.modify_host();
  cache->node_coordinates_dual.modify_host();
  cache->element_node_ids_dual.modify_host();
  cache->element_types_dual.modify_host();
  cache->element_mapping_types_dual.modify_host();
  cache->element_n_nodes_dual.modify_host();
  cache->element_n_sides_dual.modify_host();
  cache->element_n_edges_dual.modify_host();
  cache->element_n_faces_dual.modify_host();
  cache->element_side_types_dual.modify_host();
  cache->element_side_n_nodes_dual.modify_host();
  cache->element_side_local_nodes_dual.modify_host();
  cache->element_side_is_boundary_dual.modify_host();
  cache->element_edge_positive_orientations_dual.modify_host();
  cache->element_face_positive_orientations_dual.modify_host();
  cache->element_p_levels_dual.modify_host();
  cache->element_subdomains_dual.modify_host();

  cache->node_ids_dual.sync_device();
  cache->element_ids_dual.sync_device();
  cache->node_coordinates_dual.sync_device();
  cache->element_node_ids_dual.sync_device();
  cache->element_types_dual.sync_device();
  cache->element_mapping_types_dual.sync_device();
  cache->element_n_nodes_dual.sync_device();
  cache->element_n_sides_dual.sync_device();
  cache->element_n_edges_dual.sync_device();
  cache->element_n_faces_dual.sync_device();
  cache->element_side_types_dual.sync_device();
  cache->element_side_n_nodes_dual.sync_device();
  cache->element_side_local_nodes_dual.sync_device();
  cache->element_side_is_boundary_dual.sync_device();
  cache->element_edge_positive_orientations_dual.sync_device();
  cache->element_face_positive_orientations_dual.sync_device();
  cache->element_p_levels_dual.sync_device();
  cache->element_subdomains_dual.sync_device();

  cache->node_ids = cache->node_ids_dual.view_device();
  cache->element_ids = cache->element_ids_dual.view_device();
  cache->node_coordinates = cache->node_coordinates_dual.view_device();
  cache->element_node_ids = cache->element_node_ids_dual.view_device();
  cache->element_types = cache->element_types_dual.view_device();
  cache->element_mapping_types = cache->element_mapping_types_dual.view_device();
  cache->element_n_nodes = cache->element_n_nodes_dual.view_device();
  cache->element_n_sides = cache->element_n_sides_dual.view_device();
  cache->element_n_edges = cache->element_n_edges_dual.view_device();
  cache->element_n_faces = cache->element_n_faces_dual.view_device();
  cache->element_side_types = cache->element_side_types_dual.view_device();
  cache->element_side_n_nodes = cache->element_side_n_nodes_dual.view_device();
  cache->element_side_local_nodes = cache->element_side_local_nodes_dual.view_device();
  cache->element_side_is_boundary = cache->element_side_is_boundary_dual.view_device();
  cache->element_edge_positive_orientations =
    cache->element_edge_positive_orientations_dual.view_device();
  cache->element_face_positive_orientations =
    cache->element_face_positive_orientations_dual.view_device();
  cache->element_p_levels = cache->element_p_levels_dual.view_device();
  cache->element_subdomains = cache->element_subdomains_dual.view_device();
  _kokkos_geometry_cache = std::move(cache);
  return *_kokkos_geometry_cache;
}

void MeshBase::prepare_kokkos_geometry_cache() const
{
  libmesh_ignore(this->get_kokkos_geometry_cache());
}

unsigned int MeshBase::get_kokkos_elem_index(const Elem & elem) const
{
  const auto & cache = this->get_kokkos_geometry_cache();
  if (auto it = cache.element_lookup.find(elem.id()); it != cache.element_lookup.end())
    return it->second;

  return libMesh::invalid_uint;
}
#endif

} // namespace libMesh
