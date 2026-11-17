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



// <h1>Kokkos Example 2 - Evaluating the Element Map on the Device</h1>
//
// This example builds on Kokkos example 1 by mapping the reference
// element to the physical one on the device.  It meshes a line,
// square or cube with elements of any Lagrange type, or of several
// types side by side, and curves it with a smooth displacement, a
// Lagrange variable of a System, so that the map's Jacobian varies
// from point to point.  Then, at every quadrature point of every
// element, it computes with libMesh's shared host/device map code
//
//   jacobian : the map's derivatives dx/dxi from the node coordinates
//              (LagrangeMapEval::map_derivs()), and from them the
//              Jacobian, its determinant and inverse (fe_map_*D()); it
//              writes JxW
//   gradient : as jacobian, then also the physical gradient of
//              u_h = sum_i u_i phi_i (fe_physical_derivs()); it writes
//              JxW and grad u_h
//
// each reading the curved geometry in one of three ways:
//
//   copied    : each element's curved node coordinates copied to the
//               device per element, in a row
//   mirror    : a Kokkos::MeshMirror of the curved mesh: each node's
//               coordinates once, read through the connectivity
//   displaced : a mirror of the straight mesh, with the displacement
//               gathered to its nodes by Kokkos::NodeDofs and added
//               to each node as the kernel reads it, so the geometry
//               follows the solution with nothing rebuilt
//
// The mirror groups the elements in buckets by type.  The example
// launches each kernel once per bucket, through dispatch_elem_type(),
// which instantiates it for every type in a compile-time list and
// picks the bucket's on the host.  So the kernels are written once,
// for any type, as a host assembly loop is, yet each is compiled for
// one type, with the number of nodes, the shapes and the rule as
// constants.
//
// u_h is one coefficient per node, read through the connectivity, or
// from a per-element copy for the copied layout.  It uses the map's
// own Lagrange shapes, so the gradient variant reuses the shape
// derivatives the map computes: in the same pass over the nodes it
// sums u_h's reference gradient, then applies the chain rule to that
// sum once.  Host FE::reinit() instead applies it to each shape's
// gradient, and sums those, so the two agree only to rounding; the
// jacobian variant's JxW, from the same fe_map_*D() as host FEMap,
// matches exactly.  Every variant is checked against FE::reinit() on
// a copy of the mesh moved by the displacement.  On several
// processors each checks its own local elements.
//
// Options:
//   -n <elements per side>
//   -elem_type <type>[,<type>...]  the types, all of one dimension and
//                                  default order, each meshing its own
//                                  slab of the domain (default HEX27)


// libMesh includes
#include "libmesh/libmesh.h"
#include "libmesh/elem.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_lagrange_n_dofs.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/kokkos_mesh_mirror.h"
#include "libmesh/kokkos_node_dofs.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/quadrature_gauss_3D.h" // QGauss::rule_elem()
#include "libmesh/string_to_enum.h"
#include "libmesh/system.h"

// Kokkos
#include <Kokkos_Core.hpp>

// C++ includes
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// The kernels are in a named namespace: nvcc does not allow device
// lambdas in some functions of unnamed ones.  libMesh's names are
// brought in only here, since libMesh::Kokkos would make the code nvcc
// generates at global scope, which names ::Kokkos as Kokkos,
// ambiguous.
namespace kokkos_ex2
{

using namespace libMesh;
using libMesh::Kokkos::ElemTraits;
using libMesh::Kokkos::LagrangeElemTypes;
using libMesh::Kokkos::MeshMirror;
using libMesh::Kokkos::NodalDisplacement;
using libMesh::Kokkos::NodeDofs;
using libMesh::Kokkos::dispatch_elem_type;

using ExecSpace = ::Kokkos::DefaultExecutionSpace;
using RealView = ::Kokkos::View<Real *>;
using PointView = ::Kokkos::View<Point *>;
using index_type = MeshMirror::index_type;
using NodesView = ::Kokkos::View<const index_type **, ::Kokkos::LayoutRight, MeshMirror::memory_space>;

// Per element, its node coordinates or its coefficients together
using CopiedPointsView = ::Kokkos::View<Point **, ::Kokkos::LayoutRight>;
using CopiedView = ::Kokkos::View<Real **, ::Kokkos::LayoutRight>;

// What the variants compute at one point: JxW, and grad u_h for the
// gradient variant
struct MapResult
{
  Real JxW = 0;
  RealVectorValue grad;
};

// Per (element, point)
using ResultsView = ::Kokkos::View<MapResult **, ::Kokkos::LayoutRight>;

// The types the kernels are compiled for: each is one more
// instantiation of every kernel
using Types = LagrangeElemTypes;

// The rule: Gauss, of order 2p+1 for maps of order p, without
// negative weights
constexpr bool negative_weights = false;
constexpr Order rule_order (const Order p) { return static_cast<Order>(2 * p + 1); }

// What each kernel needs to know about its type at compile time
template <typename Traits>
struct Shape : Traits
{
  static constexpr Order q_order = rule_order(Traits::order);
  static constexpr unsigned int n_qp = QGauss::n_points_elem(Traits::type, q_order, negative_weights);
};

// The curving displacement at point p of the straight domain.  Its
// gradient stays well below 1, so the map remains invertible
// everywhere.  On a line or a square it also moves the domain out of
// its plane, which the maps of fe_map_1D() and fe_map_2D() allow.
inline Point displacement(const Point & p)
{
  const Real s = 0.05 * std::sin(2 * libMesh::pi * p(0)) * std::cos(libMesh::pi * p(1)) *
    std::cos(libMesh::pi * p(2));
  return Point(s, 0.5 * s, 0.25 * s);
}

// The maps of each layout: element e's map, as an object with
// map_derivs(), from its own node accessor, compiled for one type.
struct CopiedNodes
{
  CopiedPointsView coords;
  std::size_t e;

  KOKKOS_INLINE_FUNCTION Point operator()(const unsigned int i) const { return coords(e, i); }
};

template <typename S>
struct CopiedMaps
{
  CopiedPointsView coords;
  KOKKOS_INLINE_FUNCTION auto operator()(const std::size_t e) const
  { return lagrange_map_eval<S::type>(CopiedNodes{coords, e}); }
};

template <typename S>
struct MirrorMaps
{
  MeshMirror::Bucket b;
  KOKKOS_INLINE_FUNCTION auto operator()(const std::size_t e) const
  { return b.reference_map<S::type>(e); }
};

template <typename S>
struct DisplacedMaps
{
  MeshMirror::Bucket b;
  NodalDisplacement u;
  KOKKOS_INLINE_FUNCTION auto operator()(const std::size_t e) const
  { return b.current_map<S::type>(e, u); }
};

// Element e's coefficient i of u_h, through the connectivity or from
// a per-element copy
struct MirrorCoefficients
{
  NodesView nodes;
  RealView u;
  KOKKOS_INLINE_FUNCTION Real operator()(const std::size_t e, const unsigned int i) const
  { return u(nodes(e, i)); }
};

struct CopiedCoefficients
{
  CopiedView u;
  KOKKOS_INLINE_FUNCTION Real operator()(const std::size_t e, const unsigned int i) const
  { return u(e, i); }
};

// The map at reference point p of element e, in one pass over the
// element's nodes.  Each shape derivative it uses is also passed to
// f(i, j, dphi).  The Jacobian check counts points where host FEMap
// would throw, since device code cannot.
template <typename S, typename Map, typename F>
KOKKOS_INLINE_FUNCTION void
map_point(const Map & map, const ::Kokkos::View<unsigned int> & n_degenerate,
          const Point & p, FEMapPoint<> & m, const F & f)
{
  Point dxyz[S::dim];
  map.map_derivs(p, dxyz, f);
  const auto check = [=](const Real det) {
    if (!(det > 0))
      ::Kokkos::atomic_add(&n_degenerate(), 1u);
  };
  if constexpr (S::dim == 1)
    fe_map_1D(dxyz[0], m, check);
  else if constexpr (S::dim == 2)
    fe_map_2D(dxyz[0], dxyz[1], m, check);
  else
    fe_map_3D(dxyz[0], dxyz[1], dxyz[2], m, check);
}

// One bucket's rule, its data per layout and its output.  The two
// mirrors number elements and nodes alike, so their k-th buckets
// hold the same elements.
struct BucketData
{
  MeshMirror::Bucket straight, curved;
  std::size_t n_elem;
  unsigned int n_qp;
  PointView qp;
  RealView qw;
  CopiedPointsView coords;
  CopiedView u_copied;
  ResultsView out;
};

// The jacobian variant: JxW only
template <typename S, typename Maps>
void run_jacobian(const Maps maps, const BucketData & d,
                  const ::Kokkos::View<unsigned int> n_degenerate)
{
  const PointView qp = d.qp;
  const RealView qw = d.qw;
  const ResultsView out = d.out;
  ::Kokkos::parallel_for("map_jacobian", d.n_elem * S::n_qp, KOKKOS_LAMBDA(const std::size_t k) {
      const std::size_t e = k / S::n_qp;
      const unsigned int q = k % S::n_qp;
      FEMapPoint<> m;
      map_point<S>(maps(e), n_degenerate, qp(q), m, [](const unsigned int, const unsigned int, const Real) {});
      out(e, q).JxW = m.jac * qw(q);
    });
}

// The gradient variant: JxW and grad u_h.  The map's shape
// derivatives are u_h's, so u_h's reference gradient is summed in
// the map's pass over the nodes, and the chain rule, through the
// inverse map derivatives in m, is applied to it once.
template <typename S, typename Maps, typename Coefficients>
void run_gradient(const Maps maps, const Coefficients u, const BucketData & d,
                  const ::Kokkos::View<unsigned int> n_degenerate)
{
  const PointView qp = d.qp;
  const RealView qw = d.qw;
  const ResultsView out = d.out;
  ::Kokkos::parallel_for("map_gradient", d.n_elem * S::n_qp, KOKKOS_LAMBDA(const std::size_t k) {
      const std::size_t e = k / S::n_qp;
      const unsigned int q = k % S::n_qp;
      Real du[S::dim] = {};
      FEMapPoint<> m;
      map_point<S>(maps(e), n_degenerate, qp(q), m, [&](const unsigned int i, const unsigned int j, const Real dphi) {
          du[j] += u(e, i) * dphi;
        });
      Real grad[LIBMESH_DIM];
      fe_physical_derivs(S::dim, du, m, grad);
      MapResult & r = out(e, q);
      r.JxW = m.jac * qw(q);
      for (unsigned int c = 0; c != LIBMESH_DIM; ++c)
        r.grad(c) = grad[c];
    });
}

// The bucket's rule, its per-element copies of the curved coordinates
// and of u_h's coefficients, and room for its output
template <typename S>
void fill_bucket(BucketData & d, const MeshMirror & curved, const RealView u)
{
  constexpr unsigned int n_nodes = S::n_nodes, n_qp = S::n_qp;
  d.n_qp = n_qp;
  d.qp = PointView("qp", n_qp);
  d.qw = RealView("qw", n_qp);
  d.coords = CopiedPointsView("coords", d.n_elem, n_nodes);
  d.u_copied = CopiedView("u_copied", d.n_elem, n_nodes);
  d.out = ResultsView("out", d.n_elem, n_qp);

  const PointView qp = d.qp;
  const RealView qw = d.qw;
  const CopiedPointsView coords = d.coords;
  const CopiedView u_copied = d.u_copied;
  const auto points = curved.points();
  const auto nodes = d.curved.nodes;
  using Range2D = ::Kokkos::MDRangePolicy<::Kokkos::Rank<2>>;
  ::Kokkos::parallel_for("copy", Range2D({0, 0}, {d.n_elem, std::size_t(n_nodes)}),
    KOKKOS_LAMBDA(const std::size_t e, const std::size_t i) {
      u_copied(e, i) = u(nodes(e, i));
      Point & x = coords(e, i);
      for (unsigned int c = 0; c != LIBMESH_DIM; ++c)
        x(c) = points(nodes(e, i), c);
    });
  ::Kokkos::parallel_for("rule", 1, KOKKOS_LAMBDA(const int) {
      QGauss::rule_elem(S::type, S::q_order, negative_weights, qp.data(), qw.data());
    });
}

// Calls f(Shape<T>(), d) for each bucket's data d, with T its type
template <typename F>
void for_each_bucket(const std::vector<BucketData> & data, F && f)
{
  for (const BucketData & d : data)
    if (d.n_elem)
      dispatch_elem_type<Types>(d.straight.type, [&](auto traits) {
          f(Shape<decltype(traits)>(), d);
        });
}

// The domain, as a slab of elements of each of the types along x,
// stitched where they meet, n elements per side in all
void build_domain(UnstructuredMesh & mesh, const std::vector<ElemType> & types, const unsigned int n)
{
  const unsigned int dim = Elem::type_to_dim_map[types[0]];
  const unsigned int n_types = types.size();
  const unsigned int nx = std::max(1u, n / n_types);
  // The boundary ids of the sides at x = min and x = max
  const boundary_id_type left = (dim == 1) ? 0 : (dim == 2) ? 3 : 4;
  const boundary_id_type right = (dim == 1) ? 1 : (dim == 2) ? 1 : 2;
  for (unsigned int k = 0; k != n_types; ++k)
    {
      const Real x0 = Real(k) / n_types, x1 = Real(k + 1) / n_types;
      ReplicatedMesh slab(mesh.comm());
      UnstructuredMesh & m = k ? static_cast<UnstructuredMesh &>(slab) : mesh;
      if (dim == 1)
        MeshTools::Generation::build_line(m, nx, x0, x1, types[k]);
      else if (dim == 2)
        MeshTools::Generation::build_square(m, nx, n, x0, x1, 0., 1., types[k]);
      else
        MeshTools::Generation::build_cube(m, nx, n, n, x0, x1, 0., 1., 0., 1., types[k]);
      if (k)
        mesh.stitch_meshes(slab, right, left, TOLERANCE, /*clear_stitched_boundary_ids=*/true,
                           /*verbose=*/false);
    }
}

// The example itself, in a function of its own rather than in main():
// nvcc takes the address of the function enclosing each device lambda,
// which C++ forbids for main().
int run_example (LibMeshInit & init)
{
  int n_fail = 0;

  const unsigned int n = libMesh::command_line_next("-n", 8);
  std::vector<ElemType> types;
  {
    std::istringstream is(libMesh::command_line_next("-elem_type", std::string("HEX27")));
    for (std::string t; std::getline(is, t, ',');)
      types.push_back(Utility::string_to_enum<ElemType>(t));
  }
  libmesh_error_msg_if(types.empty(), "Kokkos example 2 needs an element type");
  for (const ElemType t : types)
    {
      libmesh_error_msg_if(!Types::contains(t), "Kokkos example 2 has no kernels for "
                           << Utility::enum_to_string(t));
      libmesh_error_msg_if(Elem::type_to_dim_map[t] != Elem::type_to_dim_map[types[0]] ||
                           Elem::type_to_default_order_map[t] != Elem::type_to_default_order_map[types[0]],
                           "Kokkos example 2: the types must have one dimension and order");
    }
  const unsigned int dim = Elem::type_to_dim_map[types[0]];
  const Order order = Elem::type_to_default_order_map[types[0]];

  // The straight domain, and the displacement that curves it: one
  // Lagrange variable per direction, of the maps' order, set at every
  // node this processor owns
  Mesh mesh(init.comm());
  build_domain(mesh, types, n);
  EquationSystems es(mesh);
  System & sys = es.add_system<System>("displacement");
  std::vector<unsigned int> vars;
  for (const char * name : {"dx", "dy", "dz"})
    vars.push_back(sys.add_variable(name, FEType(order, LAGRANGE)));
  es.init();
  for (const Node * node : mesh.local_node_ptr_range())
    {
      const Point d = displacement(*node);
      for (unsigned int c = 0; c != 3; ++c)
        sys.solution->set(node->dof_number(sys.number(), vars[c], 0), d(c));
    }
  sys.solution->close();

  // A copy of the mesh with its nodes moved by the displacement: what
  // the host checks against, and what the copied and mirror layouts
  // read
  std::unique_ptr<MeshBase> moved = mesh.clone();
  for (Node * node : moved->node_ptr_range())
    {
      const Node & original = mesh.node_ref(node->id());
      const Point d = displacement(original);
      for (unsigned int c = 0; c != 3; ++c)
        (*node)(c) = original(c) + d(c);
    }

  // The mirrors of the straight mesh and of the moved one, which number
  // elements and nodes alike, and the displacement at the straight
  // mirror's nodes, gathered from the solution
  const MeshMirror straight(mesh), curved(*moved);
  libmesh_error_msg_if(curved.node_ids() != straight.node_ids() ||
                       curved.n_buckets() != straight.n_buckets(),
                       "Kokkos example 2: the mirrors of the mesh and its copy differ");
  const NodeDofs node_dofs(straight, sys, vars);
  const NodalDisplacement u_disp = node_dofs.gather(*sys.solution);
  const index_type n_mirror_nodes = straight.n_nodes();

  // u_h's coefficients, one per node, and a count of points with a
  // non-positive Jacobian
  RealView u("u", n_mirror_nodes);
  ::Kokkos::View<unsigned int> n_degenerate("n_degenerate");
  ::Kokkos::parallel_for("fill_u", n_mirror_nodes, KOKKOS_LAMBDA(const index_type k) {
      u(k) = std::sin(0.1 * Real(k) + 0.37 * Real(k % 7));
    });

  // Each bucket's data
  std::vector<BucketData> data;
  libMesh::out << "Kokkos example 2: on processor 0 of " << init.comm().size() << ",";
  for (unsigned int k = 0; k != straight.n_buckets(); ++k)
    {
      BucketData d {straight.bucket(k), curved.bucket(k), straight.bucket(k).n_local, 0, {}, {}, {}, {}, {}};
      libmesh_error_msg_if(d.curved.type != d.straight.type,
                           "Kokkos example 2: the mirrors of the mesh and its copy differ");
      libmesh_error_msg_if(FECounts::lagrange_n_dofs(d.straight.type, order) !=
                           Elem::type_to_n_nodes_map[d.straight.type],
                           "Kokkos example 2: u_h must use the maps' own shapes, one per node");
      if (d.n_elem)
        dispatch_elem_type<Types>(d.straight.type, [&](auto traits) {
            using S = Shape<decltype(traits)>;
            fill_bucket<S>(d, curved, u);
            libMesh::out << " " << d.n_elem << " " << Utility::enum_to_string(S::type)
                         << " (QGauss " << Utility::enum_to_string(S::q_order) << ", "
                         << S::n_qp << " points)";
          });
      data.push_back(d);
    }
  libMesh::out << " elements; " << n_mirror_nodes << " nodes, LAGRANGE "
               << Utility::enum_to_string(order) << ", on "
               << ExecSpace::name() << std::endl;
  ::Kokkos::fence();

  // The host reference: FE::reinit() on a few elements of each bucket
  // of the moved mesh, with the coefficients copied back from the
  // device
  const FEType fe_type(order, LAGRANGE);
  std::unique_ptr<FEBase> fe = FEBase::build(dim, fe_type);
  QGauss qrule(dim, rule_order(order));
  qrule.allow_rules_with_negative_weights = negative_weights;
  fe->attach_quadrature_rule(&qrule);
  const std::vector<Real> & JxW = fe->get_JxW();
  const std::vector<std::vector<RealGradient>> & dphi = fe->get_dphi();
  auto h_u = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), u);

  // Errors are relative to |JxW| and to |grad u_h| at each point
  auto check = [&](const std::string & name, const bool with_gradient) {
    ::Kokkos::fence();
    unsigned int n_bad = 0;
    Real worst = 0;
    for (const BucketData & d : data)
      {
        if (!d.n_elem)
          continue;
        auto h_out = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d.out);
        auto h_nodes = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d.curved.nodes);
        auto h_id = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d.curved.id);
        for (const std::size_t e : {std::size_t(0), d.n_elem / 3, d.n_elem / 2, d.n_elem - 1})
          {
            fe->reinit(moved->elem_ptr(h_id(e)));
            libmesh_error_msg_if(qrule.n_points() != d.n_qp, "Kokkos example 2: the rules differ");
            for (unsigned int q = 0; q != d.n_qp; ++q)
              {
                const MapResult & kernel = h_out(e, q);
                std::vector<Real> err = {std::abs(kernel.JxW - JxW[q]) / std::abs(JxW[q])};
                if (with_gradient)
                  {
                    RealGradient grad;
                    for (unsigned int i = 0; i != dphi.size(); ++i)
                      grad.add_scaled(dphi[i][q], h_u(h_nodes(e, i)));
                    for (unsigned int c = 0; c != LIBMESH_DIM; ++c)
                      err.push_back(std::abs(kernel.grad(c) - grad(c)) / grad.norm());
                  }
                for (const Real r : err)
                  {
                    worst = std::max(worst, r);
                    n_bad += (r > 1e-12);
                  }
              }
          }
      }
    libMesh::out << "  check " << std::setw(19) << std::left << name << ": "
                 << (n_bad ? "FAIL" : "ok") << " (max relative error " << std::scientific
                 << std::setprecision(2) << worst << std::defaultfloat << ")" << std::endl;
    n_fail += n_bad;
  };

  // Each variant on each layout, launched on every bucket
  using Data = const BucketData &;
  struct Variant
  {
    std::string name;
    std::function<void()> run;
    bool with_gradient;
  };
  const std::vector<Variant> variants =
    {{"jacobian/copied", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            using S = decltype(s);
            run_jacobian<S>(CopiedMaps<S>{d.coords}, d, n_degenerate); }); }, false},
     {"jacobian/mirror", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            using S = decltype(s);
            run_jacobian<S>(MirrorMaps<S>{d.curved}, d, n_degenerate); }); }, false},
     {"jacobian/displaced", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            using S = decltype(s);
            run_jacobian<S>(DisplacedMaps<S>{d.straight, u_disp}, d, n_degenerate); }); }, false},
     {"gradient/copied", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            using S = decltype(s);
            run_gradient<S>(CopiedMaps<S>{d.coords}, CopiedCoefficients{d.u_copied}, d,
                            n_degenerate); }); }, true},
     {"gradient/mirror", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            using S = decltype(s);
            run_gradient<S>(MirrorMaps<S>{d.curved}, MirrorCoefficients{d.curved.nodes, u}, d,
                            n_degenerate); }); }, true},
     {"gradient/displaced", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            using S = decltype(s);
            run_gradient<S>(DisplacedMaps<S>{d.straight, u_disp},
                            MirrorCoefficients{d.straight.nodes, u}, d, n_degenerate); }); }, true}};

  // Each checked on cleared output
  for (const Variant & v : variants)
    {
      for (const BucketData & d : data)
        ::Kokkos::deep_copy(d.out, MapResult());
      v.run();
      check(v.name, v.with_gradient);
    }

  unsigned int h_n_degenerate = 0;
  ::Kokkos::deep_copy(h_n_degenerate, n_degenerate);
  if (h_n_degenerate)
    {
      libMesh::out << "  " << h_n_degenerate << " points with a non-positive Jacobian" << std::endl;
      ++n_fail;
    }

  return n_fail;
}

} // namespace kokkos_ex2



int main (int argc, char ** argv)
{
  // Kokkos is initialized before libMesh and finalized after it, so
  // every View is destroyed before Kokkos shuts down.
  ::Kokkos::initialize(argc, argv);
  int n_fail = 0;
  {
    libMesh::LibMeshInit init (argc, argv);
    n_fail = kokkos_ex2::run_example(init);
  }
  ::Kokkos::finalize();
  return n_fail ? 1 : 0;
}
