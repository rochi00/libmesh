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



#ifndef LIBMESH_KOKKOS_NODE_DOFS_H
#define LIBMESH_KOKKOS_NODE_DOFS_H

// Local includes
#include "libmesh/kokkos_mesh_mirror.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/mesh_base.h"
#include "libmesh/node.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/system.h"

// C++ includes
#include <sstream>
#include <string>
#include <vector>

namespace libMesh::Kokkos
{

/**
 * The displacement of each node of a MeshMirror, as a kernel captures
 * it: u(n) is node n's, with a zero component in each direction no
 * variable displaces.  NodeDofs::gather() fills it from a solution
 * vector, and it plugs into MeshMirror::Bucket::current_map().
 */
struct NodalDisplacement
{
  ::Kokkos::View<const Real * [LIBMESH_DIM], ::Kokkos::LayoutRight,
                 MeshMirror::memory_space> values;

  KOKKOS_INLINE_FUNCTION
  Point operator() (const MeshMirror::index_type n) const
  {
    Point u;
    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
      u(d) = values(n, d);
    return u;
  }
};



/**
 * The degrees of freedom of the variables of a System that displace
 * the nodes of a MeshMirror, one per direction: dofs()(n, d) is the
 * global index of the dof of node n of the variable displacing it in
 * direction d, or DofObject::invalid_id if no variable does.  It is
 * the first piece of a mirror of the DofMap.
 *
 * The displacements must be in the space of the elements' own maps:
 * each variable LAGRANGE, with one dof at every node of every
 * mirrored element and no other dofs on the elements it is active
 * on, so that each element's displaced map is its Lagrange map on its
 * displaced nodes, exactly.  A variable of another order or family
 * needs its own shapes added to the map, which the table does not
 * support yet.
 *
 * The table is a snapshot of the System's numbering for the mirror's
 * nodes; after the mesh or the dofs change it needs building again.
 */
class NodeDofs
{
public:
  using memory_space = MeshMirror::memory_space;
  using index_type = MeshMirror::index_type;
  using dofs_view = ::Kokkos::View<const dof_id_type * [LIBMESH_DIM], ::Kokkos::LayoutRight, memory_space>;

  /**
   * The dofs of the variables \p vars of \p system at the nodes of \p
   * mirror, which must have been built from \p system's mesh:
   * vars[d] displaces direction d, and is invalid_uint, or missing, if
   * none does.  Collective: every processor checks its own nodes, and
   * all throw if any finds a variable that isn't in the maps' space.
   */
  NodeDofs (const MeshMirror & mirror,
            const System & system,
            const std::vector<unsigned int> & vars);

  const dofs_view & dofs () const { return _dofs; }

  /**
   * The variable displacing each direction, or invalid_uint
   */
  const std::vector<unsigned int> & variables () const { return _vars; }

  /**
   * The dofs of the table on the host, by node then direction,
   * leaving out directions no variable displaces
   */
  const std::vector<dof_id_type> & host_dofs () const { return _host_dofs; }

  /**
   * The values of \p v at each mirrored node's displacement dofs,
   * copied to the device: the displacement of the mirror's nodes that
   * \p v holds.  \p v needn't hold the dofs of ghosts, since they are
   * gathered from their owners.  Collective.
   */
  NodalDisplacement gather (const NumericVector<Number> & v) const;

private:
  std::vector<unsigned int> _vars;
  index_type _n_nodes = 0;
  std::vector<dof_id_type> _host_dofs;
  dofs_view _dofs;
};



inline
NodeDofs::NodeDofs (const MeshMirror & mirror,
                    const System & system,
                    const std::vector<unsigned int> & vars) :
  _vars(LIBMESH_DIM, invalid_uint),
  _n_nodes(mirror.n_nodes())
{
  libmesh_error_msg_if(vars.size() > LIBMESH_DIM,
                       "NodeDofs takes at most " << LIBMESH_DIM << " displacement variables");
  std::copy(vars.begin(), vars.end(), _vars.begin());

  const MeshBase & mesh = system.get_mesh();
  const DofMap & dof_map = system.get_dof_map();
  const unsigned int sys = system.number();

  // The first reason this processor's nodes can't be displaced in
  // the maps' space, if any
  std::string bad;
  const auto fail = [&bad](const auto & ... what)
  {
    if (bad.empty())
      {
        std::ostringstream os;
        (os << ... << what);
        bad = os.str();
      }
  };

  for (const unsigned int var : _vars)
    if (var != invalid_uint)
      {
        libmesh_error_msg_if(var >= system.n_vars(),
                             "NodeDofs: system " << system.name() << " has no variable " << var);
        const FEType & t = system.variable_type(var);
        if (t.family != LAGRANGE)
          fail("variable ", system.variable_name(var), " is not LAGRANGE");
      }

  // Each mirrored element: a LAGRANGE variable active on it has a dof
  // at each node and none elsewhere
  std::vector<dof_id_type> di;
  for (const Elem * elem : mesh.active_element_ptr_range())
    {
      if (mirror.elem_index(elem->id()).first == invalid_uint)
        continue;
      for (const unsigned int var : _vars)
        if (var != invalid_uint &&
            dof_map.variable(var).active_on_subdomain(elem->subdomain_id()))
          {
            dof_map.dof_indices(elem, di, var);
            if (di.size() != elem->n_nodes())
              fail("variable ", system.variable_name(var), " has ", di.size(),
                   " dofs on element ", elem->id(), " of type ",
                   Utility::enum_to_string(elem->type()), ", not one per node");
          }
    }

  // Each node's dofs
  ::Kokkos::View<dof_id_type * [LIBMESH_DIM], ::Kokkos::LayoutRight, memory_space>
    dofs("NodeDofs::dofs", _n_nodes);
  auto h_dofs = ::Kokkos::create_mirror_view(dofs);
  for (index_type n = 0; n != _n_nodes; ++n)
    {
      const Node & node = mesh.node_ref(mirror.node_ids()[n]);
      for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
        {
          h_dofs(n, d) = DofObject::invalid_id;
          const unsigned int var = _vars[d];
          if (var == invalid_uint)
            continue;
          if (node.n_comp(sys, var) != 1)
            {
              fail("node ", node.id(), " has ", node.n_comp(sys, var), " dofs of variable ",
                   system.variable_name(var), ", not one");
              continue;
            }
          h_dofs(n, d) = node.dof_number(sys, var, 0);
          _host_dofs.push_back(h_dofs(n, d));
        }
    }

  // Every processor throws, so none waits for the others in gather()
  bool ok = bad.empty();
  mesh.comm().min(ok);
  libmesh_error_msg_if(!ok, "NodeDofs: displacements must be in the maps' space: " <<
                       (bad.empty() ? std::string("another processor's nodes aren't") : bad));

  ::Kokkos::deep_copy(dofs, h_dofs);
  _dofs = dofs;
}



inline
NodalDisplacement NodeDofs::gather (const NumericVector<Number> & v) const
{
  std::vector<numeric_index_type> indices(_host_dofs.begin(), _host_dofs.end());
  std::vector<Number> local;
  v.localize(local, indices);

  ::Kokkos::View<Real * [LIBMESH_DIM], ::Kokkos::LayoutRight, memory_space>
    values("NodeDofs::values", _n_nodes);
  auto h_values = ::Kokkos::create_mirror_view(values);
  std::size_t i = 0;
  for (index_type n = 0; n != _n_nodes; ++n)
    for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
      h_values(n, d) = _vars[d] == invalid_uint ? Real(0) : libmesh_real(local[i++]);
  libmesh_assert_equal_to(i, local.size());
  ::Kokkos::deep_copy(values, h_values);
  return NodalDisplacement{values};
}

} // namespace libMesh::Kokkos

#endif // LIBMESH_KOKKOS_NODE_DOFS_H
