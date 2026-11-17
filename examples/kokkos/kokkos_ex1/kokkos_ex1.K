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



// <h1>Kokkos Example 1 - Evaluating Shape Functions on the Device</h1>
//
// libMesh's shape functions and quadrature rules are callable from
// Kokkos device code.  This example meshes a line, square or cube
// with elements of any Lagrange type, or of several types side by
// side, copies the mesh to the device as a Kokkos::MeshMirror, and
// evaluates a finite element function u_h = sum_i u_i phi_i and its
// reference gradient at every quadrature point of every element, with
// the shapes computed on the device, in three ways:
//
//   point : one thread per (element, quadrature point), recomputing
//           phi and dphi with fe_shape() and fe_shape_deriv()
//   team  : one team per element; the team builds the quadrature rule
//           in scratch memory with QGauss::rule_elem(), and its threads
//           loop over the points, recomputing phi and dphi
//   table : one thread per (element, quadrature point), reading phi and
//           dphi from a table built once on the device
//
// u_h is a continuous field, one coefficient per mesh node, of the
// elements' own Lagrange shapes, so each element reads its
// coefficients through the mirror's connectivity: u_i is
// u(nodes(e, i)), node i of element e.  Each variant also runs on a
// copy of the coefficients laid out per element, u(e, i), as a code
// that gathered them once beforehand would read them.  The two layouts
// hold the same values and give the same results.
//
// The mirror groups the elements in buckets by type, and each kernel
// is launched once per bucket, through dispatch_elem_type(): it is
// compiled for every type in a compile-time list, and the bucket's is
// picked on the host.  So the kernels are written once, for any type,
// yet inside each the type, FE order and rule are constants, and the
// compiler can inline and simplify the type's shape code.  Each
// variant is checked against libMesh's host FEInterface::shape() and
// shape_deriv().  On several processors each checks its own local
// elements.
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
#include "libmesh/fe_interface.h"
#include "libmesh/fe_lagrange_n_dofs.h"
#include "libmesh/fe_shape.h"
#include "libmesh/fe_type.h"
#include "libmesh/kokkos_mesh_mirror.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/quadrature_gauss_3D.h" // QGauss::rule_elem()
#include "libmesh/string_to_enum.h"

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
namespace kokkos_ex1
{

using namespace libMesh;
using libMesh::Kokkos::LagrangeElemTypes;
using libMesh::Kokkos::MeshMirror;
using libMesh::Kokkos::dispatch_elem_type;

using ExecSpace = ::Kokkos::DefaultExecutionSpace;
using RealView = ::Kokkos::View<Real *>;
using PointView = ::Kokkos::View<Point *>;
using index_type = MeshMirror::index_type;
using NodesView = ::Kokkos::View<const index_type **, ::Kokkos::LayoutRight, MeshMirror::memory_space>;

// A value and its reference gradient at one point, of u_h or of one
// shape; the gradient's components past the element's dimension are
// zero
struct ValueAndGradient
{
  Real value = 0;
  RealVectorValue grad;
};

// Per (element, point) or (point, shape), each row's values together
using ValuesView = ::Kokkos::View<ValueAndGradient **, ::Kokkos::LayoutRight>;

// Per element, its coefficients together
using CopiedView = ::Kokkos::View<Real **, ::Kokkos::LayoutRight>;

// The types the kernels are compiled for: each is one more
// instantiation of every kernel
using Types = LagrangeElemTypes;

// u_h's shapes: Lagrange, of each type's default order, on a Gauss
// rule of order 2p+1 without negative weights
constexpr FEFamily family = LAGRANGE;
constexpr bool negative_weights = false;
constexpr Order rule_order (const Order p) { return static_cast<Order>(2 * p + 1); }

// What each kernel needs to know about its type at compile time
template <typename Traits>
struct Shape : Traits
{
  static constexpr Order q_order = rule_order(Traits::order);
  static constexpr unsigned int n_qp = QGauss::n_points_elem(Traits::type, q_order, negative_weights);
  static constexpr unsigned int n_dof = FECounts::lagrange_n_dofs(Traits::type, Traits::order);
  // dof i is node i's
  static_assert(n_dof == Traits::n_nodes, "one coefficient per node");
};

// Element e's coefficient i, read through the mirror's connectivity
struct MirrorCoefficients
{
  NodesView nodes;
  RealView u;

  KOKKOS_INLINE_FUNCTION
  Real operator()(const std::size_t e, const unsigned int i) const { return u(nodes(e, i)); }
};

// Element e's coefficient i, from a per-element copy
struct CopiedCoefficients
{
  CopiedView u;

  KOKKOS_INLINE_FUNCTION
  Real operator()(const std::size_t e, const unsigned int i) const { return u(e, i); }
};

// u_h and its reference gradient at reference point p, from element
// e's coefficients, computing the shapes there
template <typename S, typename Coefficients>
KOKKOS_INLINE_FUNCTION ValueAndGradient
evaluate(const Coefficients & u, const std::size_t e, const Point & p)
{
  ValueAndGradient r;
  for (unsigned int i = 0; i != S::n_dof; ++i)
    {
      const Real ui = u(e, i);
      r.value += ui * fe_shape(S::dim, family, S::type, S::order, i, p);
      for (unsigned int j = 0; j != S::dim; ++j)
        r.grad(j) += ui * fe_shape_deriv(S::dim, family, S::type, S::order, i, j, p);
    }
  return r;
}

// One bucket's rule; the table of its shapes there, per (point, shape);
// its per-element copy of the coefficients; and its output, u_h and
// its reference gradient per (element, point)
struct BucketData
{
  MeshMirror::Bucket bucket;
  std::size_t n_elem;
  unsigned int n_qp, n_dof;
  PointView qp;
  ValuesView table;
  CopiedView u_copied;
  ValuesView out;
};

// The point variant: a flat range over (element, point) pairs
template <typename S, typename Coefficients>
void run_point(const Coefficients u, const BucketData & d)
{
  const PointView qp = d.qp;
  const ValuesView out = d.out;
  ::Kokkos::parallel_for("shape_point", d.n_elem * S::n_qp, KOKKOS_LAMBDA(const std::size_t k) {
      const std::size_t e = k / S::n_qp;
      const unsigned int q = k % S::n_qp;
      out(e, q) = evaluate<S>(u, e, qp(q));
    });
}

// The team variant: the rule goes into the team's scratch memory,
// so it is computed once per element rather than read from global
// memory
template <typename S, typename Coefficients>
void run_team(const Coefficients u, const BucketData & d)
{
  using TeamPolicy = ::Kokkos::TeamPolicy<ExecSpace>;
  using Member = TeamPolicy::member_type;
  using ScratchView = ::Kokkos::View<Point *, ExecSpace::scratch_memory_space,
                                     ::Kokkos::MemoryTraits<::Kokkos::Unmanaged>>;
  constexpr unsigned int n_qp = S::n_qp;
  const std::size_t scratch_bytes = ScratchView::shmem_size(n_qp);
  const ValuesView out = d.out;

  ::Kokkos::parallel_for("shape_team",
    TeamPolicy(d.n_elem, ::Kokkos::AUTO).set_scratch_size(0, ::Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const Member & team) {
      const std::size_t e = team.league_rank();
      ScratchView s(team.team_scratch(0), n_qp);
      ::Kokkos::single(::Kokkos::PerTeam(team), [&]() {
          Real wts[n_qp];
          QGauss::rule_elem(S::type, S::q_order, negative_weights, s.data(), wts);
        });
      team.team_barrier();
      ::Kokkos::parallel_for(::Kokkos::TeamThreadRange(team, n_qp), [&](const unsigned int q) {
          out(e, q) = evaluate<S>(u, e, s(q));
        });
    });
}

// The table variant: as the point variant, but reading the shapes
template <typename S, typename Coefficients>
void run_table(const Coefficients u, const BucketData & d)
{
  const ValuesView table = d.table, out = d.out;
  ::Kokkos::parallel_for("shape_table", d.n_elem * S::n_qp, KOKKOS_LAMBDA(const std::size_t k) {
      const std::size_t e = k / S::n_qp;
      const unsigned int q = k % S::n_qp;
      ValueAndGradient r;
      for (unsigned int i = 0; i != S::n_dof; ++i)
        {
          const Real ui = u(e, i);
          const ValueAndGradient & shape = table(q, i);
          r.value += ui * shape.value;
          r.grad.add_scaled(shape.grad, ui);
        }
      out(e, q) = r;
    });
}

// The bucket's rule, built on the device by the same code the host's
// QGauss uses, the table of its shapes and derivatives there, its
// copy of the coefficients, and room for its output
template <typename S>
void fill_bucket(BucketData & d, const RealView u)
{
  constexpr unsigned int n_qp = S::n_qp, n_dof = S::n_dof;
  d.n_qp = n_qp;
  d.n_dof = n_dof;
  d.qp = PointView("qp", n_qp);
  d.table = ValuesView("table", n_qp, n_dof);
  d.u_copied = CopiedView("u_copied", d.n_elem, n_dof);
  d.out = ValuesView("out", d.n_elem, n_qp);

  const PointView qp = d.qp;
  const ValuesView table = d.table;
  const CopiedView u_copied = d.u_copied;
  const MirrorCoefficients mirrored {d.bucket.nodes, u};
  using Range2D = ::Kokkos::MDRangePolicy<::Kokkos::Rank<2>>;
  ::Kokkos::parallel_for("copy_u", Range2D({0, 0}, {d.n_elem, std::size_t(n_dof)}),
    KOKKOS_LAMBDA(const std::size_t e, const std::size_t i) {
      u_copied(e, i) = mirrored(e, i);
    });
  ::Kokkos::parallel_for("rule", 1, KOKKOS_LAMBDA(const int) {
      Real wts[n_qp];
      QGauss::rule_elem(S::type, S::q_order, negative_weights, qp.data(), wts);
    });
  ::Kokkos::parallel_for("tabulate", Range2D({0, 0}, {std::size_t(n_qp), std::size_t(n_dof)}),
    KOKKOS_LAMBDA(const std::size_t q, const std::size_t i) {
      ValueAndGradient & shape = table(q, i);
      shape.value = fe_shape(S::dim, family, S::type, S::order, i, qp(q));
      for (unsigned int j = 0; j != S::dim; ++j)
        shape.grad(j) = fe_shape_deriv(S::dim, family, S::type, S::order, i, j, qp(q));
    });
}

// Calls f(Shape<T>(), d) for each bucket's data d, with T its type
template <typename F>
void for_each_bucket(const std::vector<BucketData> & data, F && f)
{
  for (const BucketData & d : data)
    if (d.n_elem)
      dispatch_elem_type<Types>(d.bucket.type, [&](auto traits) {
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

  const unsigned int n = libMesh::command_line_next("-n", 16);
  std::vector<ElemType> types;
  {
    std::istringstream is(libMesh::command_line_next("-elem_type", std::string("HEX27")));
    for (std::string t; std::getline(is, t, ',');)
      types.push_back(Utility::string_to_enum<ElemType>(t));
  }
  libmesh_error_msg_if(types.empty(), "Kokkos example 1 needs an element type");
  for (const ElemType t : types)
    {
      libmesh_error_msg_if(!Types::contains(t), "Kokkos example 1 has no kernels for "
                           << Utility::enum_to_string(t));
      libmesh_error_msg_if(Elem::type_to_dim_map[t] != Elem::type_to_dim_map[types[0]] ||
                           Elem::type_to_default_order_map[t] != Elem::type_to_default_order_map[types[0]],
                           "Kokkos example 1: the types must have one dimension and order");
    }
  const unsigned int dim = Elem::type_to_dim_map[types[0]];
  const Order order = Elem::type_to_default_order_map[types[0]];

  // The domain and its mirror on the device.  Only the connectivity
  // matters here: the shapes are evaluated on the reference element.
  Mesh mesh(init.comm());
  build_domain(mesh, types, n);
  const MeshMirror mirror(mesh);
  const index_type n_nodes = mirror.n_nodes();

  // The coefficients, one per mirror node
  RealView u("u", n_nodes);
  ::Kokkos::parallel_for("fill_u", n_nodes, KOKKOS_LAMBDA(const index_type k) {
      u(k) = std::sin(0.1 * Real(k) + 0.37 * Real(k % 7));
    });

  // Each bucket's data
  std::vector<BucketData> data;
  libMesh::out << "Kokkos example 1: on processor 0 of " << init.comm().size() << ",";
  for (unsigned int k = 0; k != mirror.n_buckets(); ++k)
    {
      BucketData d {mirror.bucket(k), mirror.bucket(k).n_local, 0, 0, {}, {}, {}, {}};
      if (d.n_elem)
        dispatch_elem_type<Types>(d.bucket.type, [&](auto traits) {
            using S = Shape<decltype(traits)>;
            fill_bucket<S>(d, u);
            libMesh::out << " " << d.n_elem << " " << Utility::enum_to_string(S::type)
                         << " (QGauss " << Utility::enum_to_string(S::q_order) << ", "
                         << S::n_qp << " points)";
          });
      data.push_back(d);
    }
  libMesh::out << " elements; " << n_nodes << " nodes, " << Utility::enum_to_string(family) << " "
               << Utility::enum_to_string(order) << ", on "
               << ExecSpace::name() << std::endl;
  ::Kokkos::fence();

  // The host reference: libMesh's FEInterface at the host QGauss
  // rule's points, with the coefficients copied back from the device
  // and each element's gathered through the mirror's connectivity on
  // the host.  (Recomputing them on the host could differ in the last
  // bit, since the device may contract 0.1*k + 0.37*j into a fused
  // multiply-add.)
  const FEType fe_type(order, family);
  auto h_u = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), u);

  auto check = [&](const std::string & name) {
    ::Kokkos::fence();
    unsigned int n_bad = 0;
    Real worst = 0;
    for (const BucketData & d : data)
      {
        if (!d.n_elem)
          continue;
        std::unique_ptr<Elem> elem = Elem::build(d.bucket.type);
        QGauss qrule(dim, rule_order(order));
        qrule.allow_rules_with_negative_weights = negative_weights;
        qrule.init(*elem);
        libmesh_error_msg_if(qrule.n_points() != d.n_qp, "Kokkos example 1: the rules differ");
        auto h_out = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d.out);
        auto h_nodes = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d.bucket.nodes);
        for (const std::size_t e : {std::size_t(0), d.n_elem / 3, d.n_elem / 2, d.n_elem - 1})
          for (unsigned int q = 0; q != d.n_qp; ++q)
            {
              ValueAndGradient ref;
              for (unsigned int i = 0; i != d.n_dof; ++i)
                {
                  const Real ui = h_u(h_nodes(e, i));
                  ref.value += ui * FEInterface::shape(fe_type, elem.get(), i, qrule.qp(q));
                  for (unsigned int j = 0; j != dim; ++j)
                    ref.grad(j) += ui * FEInterface::shape_deriv(fe_type, elem.get(), i, j, qrule.qp(q));
                }
              const ValueAndGradient & kernel = h_out(e, q);
              const auto compare = [&](const Real k, const Real r) {
                const Real err = std::abs(k - r) / (1 + std::abs(r));
                worst = std::max(worst, err);
                n_bad += (err > 1e-12);
              };
              compare(kernel.value, ref.value);
              for (unsigned int j = 0; j != LIBMESH_DIM; ++j)
                compare(kernel.grad(j), ref.grad(j));
            }
      }
    libMesh::out << "  check " << std::setw(13) << std::left << name << ": "
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
  };
  const std::vector<Variant> variants =
    {{"point/mirror", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            run_point<decltype(s)>(MirrorCoefficients{d.bucket.nodes, u}, d); }); }},
     {"point/copied", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            run_point<decltype(s)>(CopiedCoefficients{d.u_copied}, d); }); }},
     {"team/mirror", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            run_team<decltype(s)>(MirrorCoefficients{d.bucket.nodes, u}, d); }); }},
     {"team/copied", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            run_team<decltype(s)>(CopiedCoefficients{d.u_copied}, d); }); }},
     {"table/mirror", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            run_table<decltype(s)>(MirrorCoefficients{d.bucket.nodes, u}, d); }); }},
     {"table/copied", [&]() { for_each_bucket(data, [&](auto s, Data d) {
            run_table<decltype(s)>(CopiedCoefficients{d.u_copied}, d); }); }}};

  // Each checked on cleared output
  for (const Variant & v : variants)
    {
      for (const BucketData & d : data)
        ::Kokkos::deep_copy(d.out, ValueAndGradient());
      v.run();
      check(v.name);
    }

  return n_fail;
}

} // namespace kokkos_ex1



int main (int argc, char ** argv)
{
  // Kokkos is initialized before libMesh and finalized after it, so
  // every View is destroyed before Kokkos shuts down.
  ::Kokkos::initialize(argc, argv);
  int n_fail = 0;
  {
    libMesh::LibMeshInit init (argc, argv);
    n_fail = kokkos_ex1::run_example(init);
  }
  ::Kokkos::finalize();
  return n_fail ? 1 : 0;
}
