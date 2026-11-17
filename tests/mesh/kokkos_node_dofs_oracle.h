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

// Checks NodeDofs and the displaced maps of a MeshMirror against the
// host.  The table's dofs at every mirrored node, ghosts' included,
// against the nodes' own; then, with displacement variables set to a
// smooth field at every node, the quadrature points and JxW a kernel
// computes from the reference coordinates and the gathered
// displacements, through MeshMirror::Bucket::current_map(), against
// FE::reinit() on a copy of the mesh whose nodes were moved by the
// same field; then the same after the field changes, gathered again
// with the same table.  On distorted HEX27, TET10 and PRISM18 cubes
// with QUAD9 faces on the HEX27's boundary, a mixed TRI3 / QUAD4
// square with EDGE2 sides and two displacement components, the square
// partly refined (hanging nodes), QUAD9 and TRI6 squares and an EDGE3
// line tilted into 3D and bent out of their plane or line by three
// components, and a distributed HEX8 cube.  Displacements not in the
// maps' space (FIRST on QUAD9, a variable missing from some nodes)
// must throw on every processor.  The kernel must give the host's
// points and JxW to within rounding.

#ifndef KOKKOS_NODE_DOFS_ORACLE_H
#define KOKKOS_NODE_DOFS_ORACLE_H

#include "mesh/kokkos_mesh_mirror_oracle.h"

#include "libmesh/kokkos_node_dofs.h"

#include "libmesh/equation_systems.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/system.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace libMeshTest
{
namespace KokkosNodeDofsOracle
{

using namespace libMesh;
using libMesh::Kokkos::MeshMirror;
using libMesh::Kokkos::NodeDofs;
using libMesh::Kokkos::NodalDisplacement;
using index_type = MeshMirror::index_type;
using KokkosMeshMirrorOracle::Counts;

// Counts a mismatch in the Counts c in scope, printing the first 10
#define NODE_DOFS_FAIL(...)                                    \
  do {                                                         \
    if (c.n_fail < 10)                                         \
      std::printf("[node_dofs_oracle] mismatch: " __VA_ARGS__); \
    ++c.n_fail;                                                \
  } while (0)

// The map of element e of bucket b on its nodes displaced by u
struct DisplacedMapOf
{
  NodalDisplacement u;

  KOKKOS_INLINE_FUNCTION
  LagrangeMapEval<MeshMirror::CurrentNodes<NodalDisplacement>>
  operator()(const MeshMirror::Bucket & b, const index_type e) const
  { return b.current_map(e, u); }
};

// The same map, compiled for the bucket's type T
template <ElemType T>
struct TypedDisplacedMapOf
{
  NodalDisplacement u;

  KOKKOS_INLINE_FUNCTION
  LagrangeMapEval<MeshMirror::CurrentNodes<NodalDisplacement>>
  operator()(const MeshMirror::Bucket & b, const index_type e) const
  { return b.current_map<T>(e, u); }
};

// Component d of the displacement at reference point p: smooth, not
// polynomial, different in each direction and each phase, and large
// enough to change the geometry well beyond rounding.  Its gradient
// G has a norm less than 0.6 on these meshes, so the displaced map's
// Jacobian, (I + G) times the original's, stays nonsingular and every
// element valid.  The sine's coefficients are arbitrary, chosen only
// so no two directions or phases move alike.
inline Real
displacement(const Point & p, const unsigned int d, const unsigned int phase)
{
  return 0.05 * (1 + d) * std::sin(2.1 * p(0) - 1.3 * p(1) + 0.7 * p(2) + d + 2. * phase) +
         0.02 * phase * p(d) * p(d);
}

// The variables' dofs at every node this processor owns set to the
// displacement of its point
inline void
set_displacement(System & sys, const std::vector<unsigned int> & vars, const unsigned int phase)
{
  const MeshBase & mesh = sys.get_mesh();
  for (const Node * node : mesh.local_node_ptr_range())
    for (const auto d : index_range(vars))
      if (node->n_comp(sys.number(), vars[d]))
        sys.solution->set(node->dof_number(sys.number(), vars[d], 0),
                          displacement(*node, d, phase));
  sys.solution->close();
  sys.update();
}

// A copy of mesh with each node that has the variables' dofs moved by
// the displacement of its point
inline std::unique_ptr<MeshBase>
moved_copy(const MeshBase & mesh, const System & sys, const std::vector<unsigned int> & vars,
           const unsigned int phase)
{
  std::unique_ptr<MeshBase> moved = mesh.clone();
  for (Node * node : moved->node_ptr_range())
    {
      const Node & original = mesh.node_ref(node->id());
      for (const auto d : index_range(vars))
        if (original.n_comp(sys.number(), vars[d]))
          (*node)(d) = original(d) + displacement(original, d, phase);
    }
  return moved;
}

// The table's dofs at each mirrored node against the node's own; each
// node counts in c.n_node_elems
inline void
check_table(const MeshBase & mesh, const System & sys, const MeshMirror & mirror,
            const NodeDofs & table, Counts & c)
{
  auto h_dofs = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), table.dofs());
  std::size_t i = 0;
  for (const auto n : make_range(mirror.n_nodes()))
    {
      const Node & node = mesh.node_ref(mirror.node_ids()[n]);
      for (const auto d : make_range(LIBMESH_DIM))
        {
          const unsigned int var = table.variables()[d];
          const dof_id_type expected =
            var == invalid_uint ? DofObject::invalid_id : node.dof_number(sys.number(), var, 0);
          if (h_dofs(n, d) != expected)
            NODE_DOFS_FAIL("node %lu direction %u: dof %lu, expected %lu\n",
                           (unsigned long)node.id(), d, (unsigned long)h_dofs(n, d),
                           (unsigned long)expected);
          if (var != invalid_uint &&
              (i >= table.host_dofs().size() || table.host_dofs()[i++] != expected))
            NODE_DOFS_FAIL("node %lu direction %u: host dof out of order\n",
                           (unsigned long)node.id(), d);
        }
    }
  if (i != table.host_dofs().size())
    NODE_DOFS_FAIL("%zu host dofs, expected %zu\n", table.host_dofs().size(), i);
  c.n_node_elems += mirror.n_nodes();
}

// n_vars LAGRANGE displacement variables of order o on mesh: the
// table, then the displaced geometry with two fields
inline void
check_displaced(MeshBase & mesh, const Order o, const unsigned int n_vars, Counts & c)
{
  // Distorting, rotating and refining leave the mesh unprepared
  if (!mesh.is_prepared())
    mesh.prepare_for_use();

  EquationSystems es(mesh);
  System & sys = es.add_system<System>("displacement");
  std::vector<unsigned int> vars;
  for (const auto d : make_range(n_vars))
    vars.push_back(sys.add_variable("u" + std::to_string(d), FEType(o, LAGRANGE)));
  es.init();

  const MeshMirror mirror(mesh);
  const NodeDofs table(mirror, sys, vars);
  check_table(mesh, sys, mirror, table, c);

  for (const unsigned int phase : {0u, 1u})
    {
      set_displacement(sys, vars, phase);
      // The parallel solution first, then the ghosted copy
      const NodalDisplacement u = table.gather(phase ? *sys.current_local_solution : *sys.solution);
      const std::unique_ptr<MeshBase> moved = moved_copy(mesh, sys, vars, phase);
      KokkosMeshMirrorOracle::check_geometry(*moved, mirror, c, DisplacedMapOf{u});
      for (const auto k : make_range(mirror.n_buckets()))
        {
          const MeshMirror::Bucket & b = mirror.bucket(k);
          if (Elem::type_to_dim_map[b.type])
            libMesh::Kokkos::dispatch_elem_type<libMesh::Kokkos::LagrangeElemTypes>(b.type, [&](auto traits) {
                KokkosMeshMirrorOracle::check_bucket_geometry
                  (*moved, b, c, TypedDisplacedMapOf<decltype(traits)::type>{u});
              });
        }
    }

  for (const auto k : make_range(mirror.n_buckets()))
    {
      c.n_elem += mirror.bucket(k).n_local;
      c.n_ghosts += mirror.bucket(k).n_elem - mirror.bucket(k).n_local;
    }
}

// Whether building the table of vars, active on subdomains (all if
// empty), throws on every processor
inline void
check_throws(MeshBase & mesh, const Order o, const std::set<subdomain_id_type> & subdomains,
             const char * what, std::size_t & n_throws, Counts & c)
{
#ifdef LIBMESH_ENABLE_EXCEPTIONS
  EquationSystems es(mesh);
  System & sys = es.add_system<System>("displacement");
  std::vector<unsigned int> vars;
  for (const auto d : make_range(mesh.mesh_dimension()))
    vars.push_back(subdomains.empty() ?
                   sys.add_variable("u" + std::to_string(d), FEType(o, LAGRANGE)) :
                   sys.add_variable("u" + std::to_string(d), FEType(o, LAGRANGE), &subdomains));
  es.init();

  const MeshMirror mirror(mesh);
  bool threw = false;
  try
    {
      const NodeDofs table(mirror, sys, vars);
    }
  catch (const std::exception &)
    {
      threw = true;
    }
  if (!threw)
    NODE_DOFS_FAIL("%s: NodeDofs didn't throw\n", what);
  n_throws += threw;
#else
  libmesh_ignore(mesh, o, subdomains, what, n_throws, c);
#endif
}

inline int
run_all_oracles(const Parallel::Communicator & comm)
{
  Counts c;
  std::size_t n_throws = 0;

  // A distorted HEX27 cube in three subdomains, with QUAD9 faces on
  // boundary 2 in subdomain 10, rotated and reversed
  {
    ReplicatedMesh mesh(comm);
    // Distorted as in the mesh mirror oracle
    MeshTools::Generation::build_cube(mesh, 3, 2, 2, 0., 1., 0., 1., 0., 1., HEX27);
    MeshTools::Modification::distort(mesh, 0.15, /*perturb_boundary=*/false);
    for (Elem * elem : mesh.active_element_ptr_range())
      elem->subdomain_id() = elem->id() % 3;
    KokkosMeshMirrorOracle::add_lower_d(mesh, 2, 10, /*permute=*/true);
    check_displaced(mesh, SECOND, 3, c);
  }

  // Distorted TET10 and PRISM18 cubes
  for (const ElemType t : {TET10, PRISM18})
    {
      ReplicatedMesh mesh(comm);
      // 0.1 of the smallest element size around each interior node,
      // less than the hexes' 0.15, for these thinner elements
      MeshTools::Generation::build_cube(mesh, 2, 2, 2, 0., 1., 0., 1., 0., 1., t);
      MeshTools::Modification::distort(mesh, 0.1, /*perturb_boundary=*/false);
      check_displaced(mesh, SECOND, 3, c);
    }

  // A mixed TRI3 / QUAD4 square with EDGE2 sides, displaced in its
  // plane: no variable in z
  {
    ReplicatedMesh mesh(comm);
    KokkosMeshMirrorOracle::build_mixed_square(mesh);
    KokkosMeshMirrorOracle::add_lower_d(mesh, BoundaryInfo::invalid_id, 5);
    check_displaced(mesh, FIRST, 2, c);
  }

#ifdef LIBMESH_ENABLE_AMR
  // The mixed square partly refined: hanging nodes, displaced by their
  // own (unconstrained) dofs on both sides
  {
    ReplicatedMesh mesh(comm);
    KokkosMeshMirrorOracle::build_mixed_square(mesh);
    MeshRefinement refinement(mesh);
    for (Elem * elem : mesh.active_element_ptr_range())
      if (elem->id() % 5 == 0)
        elem->set_refinement_flag(Elem::REFINE);
    refinement.refine_elements();
    check_displaced(mesh, FIRST, 2, c);
  }
#endif

  // QUAD9 and TRI6 squares and an EDGE3 line tilted into 3D, each
  // displaced in all three directions, so out of its plane or line
  for (const ElemType t : {QUAD9, TRI6})
    {
      ReplicatedMesh mesh(comm);
      // Euler angles that move the square off every coordinate plane
      MeshTools::Generation::build_square(mesh, 3, 3, 0., 1., 0., 1., t);
      MeshTools::Modification::rotate(mesh, 30., 40., 10.);
      check_displaced(mesh, SECOND, 3, c);
    }
  {
    ReplicatedMesh mesh(comm);
    MeshTools::Generation::build_line(mesh, 6, 0., 1., EDGE3);
    MeshTools::Modification::rotate(mesh, 30., 40., 10.);
    check_displaced(mesh, SECOND, 3, c);
  }

  // A distributed HEX8 cube: on several processors, ghosts' dofs
  // owned by others, gathered from them
  {
    DistributedMesh mesh(comm);
    MeshTools::Generation::build_cube(mesh, 4, 3, 3, 0., 1., 0., 1., 0., 1., HEX8);
    check_displaced(mesh, FIRST, 3, c);
  }

  // Displacements outside the maps' space: FIRST on QUAD9, where the
  // map is SECOND; and variables on one subdomain, leaving the nodes
  // only the other's elements reach without dofs
  {
    ReplicatedMesh mesh(comm);
    MeshTools::Generation::build_square(mesh, 2, 2, 0., 1., 0., 1., QUAD9);
    check_throws(mesh, FIRST, {}, "FIRST on QUAD9", n_throws, c);
  }
  {
    ReplicatedMesh mesh(comm);
    MeshTools::Generation::build_square(mesh, 4, 1, 0., 1., 0., 1., QUAD4);
    for (Elem * elem : mesh.active_element_ptr_range())
      elem->subdomain_id() = elem->id() >= 2;
    mesh.prepare_for_use();
    check_throws(mesh, FIRST, {0}, "a variable on one subdomain", n_throws, c);
  }

  comm.sum(c.n_fail);
  for (std::size_t * n : {&c.n_elem, &c.n_ghosts, &c.n_node_elems, &c.n_points})
    comm.sum(*n);
  if (comm.rank() == 0)
    std::printf("[node_dofs_oracle] %s  (%d failures on %u processors; %zu elements, "
                "%zu ghosts, %zu nodes, %zu displaced points, %zu throws on processor 0)\n",
                c.n_fail ? "FAIL" : "PASS", c.n_fail,
                unsigned(comm.size()), c.n_elem, c.n_ghosts, c.n_node_elems, c.n_points,
                n_throws);
  return c.n_fail;
}

#undef NODE_DOFS_FAIL

} // namespace KokkosNodeDofsOracle
} // namespace libMeshTest

#endif // KOKKOS_NODE_DOFS_ORACLE_H
