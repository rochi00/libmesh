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

// Checks a side's quadrature mapped onto its element, done entirely in
// a Kokkos kernel from the element's mirrored nodes, its side, the rule
// order and the p level, against FE::reinit(elem, s) on the host.  The
// kernel finds the side's type and node numbers, builds the side's
// rule with QGauss::rule_elem() at the order QBase raises it to for the
// p level, maps each rule point to the element's reference element
// with fe_side_reference_point(), and there evaluates the element's
// map, from MeshMirror::Bucket::reference_map<T>(), for the physical
// point, and the side's map for JxW and the normal.  Every side of one
// element of every 1D, 2D and 3D Lagrange type, distorted and, for 1D
// and 2D, tilted into 3D.  Each bucket's kernel is compiled for its
// type, through dispatch_elem_type(); the side's type varies with the
// side on prisms and pyramids, so its rule and map take the type at
// run time.  The kernel must give the host's rule, points, JxW and
// normals to within rounding.  On several processors
// each checks the elements of its own mirror.

#ifndef KOKKOS_SIDE_REINIT_ORACLE_H
#define KOKKOS_SIDE_REINIT_ORACLE_H

#include "mesh/kokkos_mesh_mirror_oracle.h"

#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/int_range.h"
#include "libmesh/quadrature_gauss_3D.h"
#include "libmesh/replicated_mesh.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace libMeshTest
{
namespace KokkosSideReinitOracle
{

using namespace libMesh;
using libMesh::Kokkos::MeshMirror;
using index_type = MeshMirror::index_type;

// One side of an element of the bucket at one rule order and p level
struct Input
{
  index_type e;
  unsigned int side;

  // The rule's order before the p level raises it
  Order order;
  unsigned int p_level;

  // Where the side's points start in refs, and how many the host's
  // rule has
  unsigned int first_qp, n_qp;
};

// What a side's rule point maps to on the element: the physical point,
// JxW and the normal
struct Mapped
{
  Point xyz;
  Real JxW = 0;
  Point normal;
};

// The host's rule point and weight, and what FE mapped it to
struct Reference
{
  // The point on the side's reference element, and its weight
  Point qp;
  Real w;
  Mapped mapped;
};

// What the checks found, for the summary
struct Counts
{
  // Rules with the wrong number of points, whose points aren't checked
  int n_count_fail = 0;

  // Sides and rules, and their points, checked
  std::size_t n_cases = 0, n_points = 0;
};

// Every side of bucket b, of type T, at each rule order and p level,
// against FE on the elements of mesh they mirror
template <ElemType T>
inline void
check_bucket(MeshBase & mesh, const MeshMirror::Bucket & b,
             KokkosOracle::ValueCheck & check, Counts & c)
{
  constexpr unsigned int dim = Elem::type_to_dim_map[T];
  const Order orders[] = {FIRST, THIRD, FIFTH};
#ifdef LIBMESH_ENABLE_AMR
  const unsigned int p_levels[] = {0, 1, 2};
#else
  const unsigned int p_levels[] = {0};
#endif

  auto h_id = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.id);
  std::vector<Input> inputs;
  // Every quadrature point of every input, in order
  std::vector<Reference> refs;
  for (const auto e : make_range(b.n_elem))
    {
      Elem & elem = mesh.elem_ref(h_id(e));
      for (const auto s : make_range(elem.n_sides()))
        for (const Order o : orders)
          for (const unsigned int p : p_levels)
            {
#ifdef LIBMESH_ENABLE_AMR
              elem.hack_p_level(p);
#endif
              // The p level raises the side's rule order, as it would
              // a FE's own order unless asked not to; FIRST LAGRANGE
              // has no higher order on the first order types
              std::unique_ptr<FEBase> fe = FEBase::build(dim, FEType(FIRST, LAGRANGE));
              fe->add_p_level_in_reinit(false);
              QGauss qface(dim - 1, o);
              fe->attach_quadrature_rule(&qface);
              const auto & xyz = fe->get_xyz();
              const auto & JxW = fe->get_JxW();
              const auto & normals = fe->get_normals();
              fe->reinit(&elem, s);

              const unsigned int n_qp = qface.n_points();
              libmesh_assert_equal_to(xyz.size(), n_qp);
              inputs.push_back({e, s, o, p, cast_int<unsigned int>(refs.size()), n_qp});
              for (const auto q : make_range(n_qp))
                refs.push_back({qface.qp(q), qface.w(q), {xyz[q], JxW[q], normals[q]}});
            }
#ifdef LIBMESH_ENABLE_AMR
      elem.hack_p_level(0);
#endif
    }

  // Upload what each case is: the kernel gets nothing of the host's
  // side, rule or map
  const std::size_t n_cases = inputs.size(), n_points = refs.size();
  const auto d_in = KokkosOracle::upload_objects(inputs, "side_reinit_in");
  ::Kokkos::View<unsigned int *> d_n("side_reinit_n", n_cases);
  ::Kokkos::View<Point *> d_qp("side_reinit_qp", n_points);
  ::Kokkos::View<Real *> d_w("side_reinit_w", n_points);
  ::Kokkos::View<Mapped *> d_mapped("side_reinit_mapped", n_points);

  // One thread per side and rule, which writes its rule in the host
  // rule's place only if it has the same number of points
  ::Kokkos::parallel_for(
    "side_reinit_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const Input in = d_in(k);
      const unsigned int s = in.side;

      const auto elem_eval = b.reference_map<T>(in.e);
      const auto elem_nodes = b.reference_nodes(in.e);
      const auto side_nodes = [=](const unsigned int i)
        { return elem_nodes(Elem::local_side_node(T, s, i)); };
      const ElemType side_type = Elem::side_type(T, s);
      const auto side_eval = lagrange_map_eval(side_type, side_nodes);

      // The side's rule, as FE::reinit() has QBase::init() build it
      const Order order = QBase::order_for_p_level(in.order, in.p_level);
      const unsigned int n_qp = QGauss::n_points_elem(side_type, order, false);
      d_n(k) = n_qp;
      if (n_qp != in.n_qp)
        return;
      Point * qp = &d_qp(in.first_qp);
      Real * w = &d_w(in.first_qp);
      QGauss::rule_elem(side_type, order, false, qp, w);

      // FEMap asserts each Jacobian is positive; every element here
      // is valid, so the kernel skips that check
      const auto no_check = [](const Real) {};
      for (unsigned int q = 0; q != n_qp; ++q)
        {
          // FE::side_map(), then FE::reinit() at the mapped point
          const Point ref = fe_side_reference_point(T, s, qp[q]);
          const Point xyz = elem_eval.map(ref);

          // FEMap::compute_face_map(), on the side's own map
          Mapped & out = d_mapped(in.first_qp + q);
          out.xyz = xyz;
          switch (dim)
            {
            case 1:
              {
                // The element's tangent at its end, pointing out
                const Point dx_dxi = elem_eval.map_deriv(0, ref);
                out.normal = (s == 0) ? Point(-dx_dxi.unit()) : Point(dx_dxi.unit());
                // FEMap gives a point side a Jacobian of 1
                out.JxW = 1.0 * w[q];
                break;
              }
            case 2:
              {
                // The element's map derivatives at the side's point,
                // found on the element by inverse mapping it, at the
                // tolerance FEMap::compute_face_map() uses
                FEFaceMapPoint<> f;
                const auto parent_derivs = [&](RealVectorValue & dx_dxi, RealVectorValue & dx_deta)
                  {
                    const NothrowInverseMapPolicy policy;
                    bool converged = false;
                    const Point p = fe_inverse_map(2, side_eval.map(qp[q]), TOLERANCE, true,
                                                   elem_eval, policy, converged);
                    dx_dxi  = elem_eval.map_deriv(0, p);
                    dx_deta = elem_eval.map_deriv(1, p);
                  };
                fe_face_map_2D(side_eval.map_deriv(0, qp[q]), parent_derivs, f, no_check);
                out.JxW = f.jac * w[q];
                out.normal = f.normal;
                break;
              }
            case 3:
              {
                FEFaceMapPoint<> f;
                fe_face_map_3D(side_eval.map_deriv(0, qp[q]), side_eval.map_deriv(1, qp[q]),
                               f, no_check);
                out.JxW = f.jac * w[q];
                out.normal = f.normal;
                break;
              }
            default:
              break;
            }
        }
    });

  auto h_n = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_n);
  auto h_qp = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_qp);
  auto h_w = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_w);
  auto h_mapped = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_mapped);

  for (const auto k : make_range(n_cases))
    {
      const Input & in = inputs[k];
      const auto describe = [&in, &h_id]()
        {
          std::ostringstream os;
          os << "type " << int(T) << " elem " << h_id(in.e) << " side " << in.side
             << " order " << int(in.order) << " p " << in.p_level;
          return os.str();
        };
      if (h_n(k) != in.n_qp)
        {
          if (check.n_fail + c.n_count_fail < 10)
            std::printf("[side_reinit_oracle] mismatch: %s: host %u points, kernel %u\n",
                        describe().c_str(), in.n_qp, h_n(k));
          ++c.n_count_fail;
          continue;
        }
      for (const auto q : make_range(in.n_qp))
        {
          const std::size_t i = in.first_qp + q;
          const Reference & r = refs[i];
          const Mapped & kernel = h_mapped(i);

          // Which value of this point mismatched, for check() to print
          const auto value = [&](const std::string & name)
            {
              return [&describe, q, name]()
                { return describe() + " qp " + std::to_string(q) + " " + name; };
            };
          const auto check_vector = [&](const std::string & name, const Point & host_value,
                                        const Point & kernel_value)
            {
              for (const auto d : make_range(LIBMESH_DIM))
                check(host_value(d), kernel_value(d), value(name + "(" + std::to_string(d) + ")"));
            };

          check_vector("rule point", r.qp, h_qp(i));
          check(r.w, h_w(i), value("weight"));
          check_vector("point", r.mapped.xyz, kernel.xyz);
          check(r.mapped.JxW, kernel.JxW, value("JxW"));
          check_vector("normal", r.mapped.normal, kernel.normal);
        }
    }
  c.n_cases += n_cases;
  c.n_points += n_points;
}

// check_bucket() on each bucket of a mirror of mesh, compiled for its
// type
inline void
check_mesh(MeshBase & mesh, KokkosOracle::ValueCheck & check, Counts & c)
{
  const MeshMirror mirror(mesh);
  for (const auto k : make_range(mirror.n_buckets()))
    {
      const MeshMirror::Bucket & b = mirror.bucket(k);
      libMesh::Kokkos::dispatch_elem_type<libMesh::Kokkos::LagrangeElemTypes>(b.type, [&](auto traits) {
          check_bucket<decltype(traits)::type>(mesh, b, check, c);
        });
    }
}

inline int
run_all_oracles(const Parallel::Communicator & comm)
{
  using KokkosMeshMirrorOracle::add_lagrange_elements;
  using KokkosMeshMirrorOracle::lagrange_types;

  // 1e-10 relative to 1 + |host|: far above the few ulp a contracted
  // evaluation of these formulas changes, far below the change from
  // any wrong term
  KokkosOracle::ValueCheck check{"side_reinit_oracle", 1e-10};
  Counts c;

  // Every type, distorted
  {
    ReplicatedMesh mesh(comm, 3);
    add_lagrange_elements(mesh, lagrange_types, /*tilt=*/false);
    mesh.prepare_for_use();
    check_mesh(mesh, check, c);
  }

  // The 1D and 2D types, tilted into 3D
  {
    std::vector<ElemType> lower_dim;
    for (const ElemType t : lagrange_types)
      if (Elem::type_to_dim_map[t] < 3)
        lower_dim.push_back(t);
    ReplicatedMesh mesh(comm, 3);
    add_lagrange_elements(mesh, lower_dim, /*tilt=*/true);
    mesh.prepare_for_use();
    check_mesh(mesh, check, c);
  }

  // Each processor checked its own mirror
  int n_fail = check.n_fail + c.n_count_fail;
  comm.sum(n_fail);
  for (std::size_t * n : {&c.n_cases, &c.n_points})
    comm.sum(*n);
  if (comm.rank() == 0)
    std::printf("[side_reinit_oracle] %s  (%d failures on %u processors; "
                "%zu sides and rules, %zu points)\n",
                n_fail ? "FAIL" : "PASS",
                n_fail, unsigned(comm.size()), c.n_cases, c.n_points);
  return n_fail;
}

} // namespace KokkosSideReinitOracle
} // namespace libMeshTest

#endif // KOKKOS_SIDE_REINIT_ORACLE_H
