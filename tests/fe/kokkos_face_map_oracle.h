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

// Checks fe_face_map_*D() and fe_face_curvature_*D(), run in a Kokkos
// kernel on side map derivatives from LagrangeMapEval, against
// FEMap::compute_face_map() on the host: every side of one element of
// every 2D and 3D Lagrange type, distorted and, for 2D, tilted into
// 3D, mirrored to the device, at each face quadrature point.  Each
// bucket's kernel is compiled for its type, through
// dispatch_elem_type(), and reads the element's map from
// MeshMirror::Bucket::reference_map<T>() and its side's nodes through
// the element's; the side's type varies with the side on prisms and
// pyramids, so its map takes the type at run time.  A 2D element's
// side in 3D finds the element's own map derivatives by
// fe_inverse_map(), as FEMap does by FEMap::inverse_map().  The kernel
// must give the side's JxW, normals, tangents and curvatures to within
// rounding.  (Not its points: FE::reinit() replaces
// those with the element's own map at the side's points.)  On several
// processors each checks the elements of its own mirror.

#ifndef KOKKOS_FACE_MAP_ORACLE_H
#define KOKKOS_FACE_MAP_ORACLE_H

#include "mesh/kokkos_mesh_mirror_oracle.h"

#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/int_range.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/replicated_mesh.h"

#include <cmath>
#include <cstdio>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace libMeshTest
{
namespace KokkosFaceMapOracle
{

using namespace libMesh;
using libMesh::Kokkos::MeshMirror;
using index_type = MeshMirror::index_type;

// One face quadrature point: an element of the bucket, its side, and
// the point on the side's reference element and its weight
struct Input
{
  index_type e;
  unsigned int side;
  Point qp;
  Real w;
};

// The kernel's results at one face quadrature point, and FEMap's.
// FEMap doesn't expose the side's Jacobian, only JxW, so face.jac is
// left out of the comparison.
struct Output
{
  Real JxW = 0;

  // The side's normal and tangents
  FEFaceMapPoint<> face;

  // The curvature of a 2D element's side or a 3D element's face
  Real curvature = 0;
};

// What FEMap::compute_face_map() computed at one face quadrature point,
// with how many of the tangents and whether the curvature it set
struct Reference
{
  // The element's id, to print a mismatch
  dof_id_type id;

  Output host;

  // How many of host.face.tangents FEMap set: one on a 2D element's
  // side, two on a 3D element's
  unsigned int n_tangents;

  // Whether FEMap set host.curvature, which it does only with second
  // derivatives enabled
  bool has_curvature;
};

// What the checks found, for the summary
struct Counts
{
  std::size_t n_points = 0;
};

// Every side's face quadrature points of bucket b, of type T, against
// FEMap on the elements of mesh they mirror
template <ElemType T>
inline void
check_bucket(const MeshBase & mesh, const MeshMirror::Bucket & b,
             KokkosOracle::ValueCheck & check, Counts & c)
{
  constexpr unsigned int dim = Elem::type_to_dim_map[T];

  auto h_id = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.id);
  std::vector<Input> inputs;
  std::vector<Reference> refs;
  for (const auto e : make_range(b.n_elem))
    {
      const Elem & elem = mesh.elem_ref(h_id(e));
      for (const auto s : make_range(elem.n_sides()))
        {
          // The FE type only decides which shapes FE computes; the map
          // is the element's own.  A FIFTH order rule has interior
          // points on every side, as many as 3 x 3 on a quad.
          std::unique_ptr<FEBase> fe = FEBase::build(dim, FEType(FIRST, LAGRANGE));
          QGauss qface(dim - 1, FIFTH);
          fe->attach_quadrature_rule(&qface);
          const auto & JxW = fe->get_JxW();
          const auto & normals = fe->get_normals();
          const auto & tangents = fe->get_tangents();
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          const auto & curvatures = fe->get_curvatures();
#endif
          fe->reinit(&elem, s);

          for (const auto q : index_range(JxW))
            {
              inputs.push_back({e, s, qface.qp(q), qface.w(q)});

              Reference r{elem.id(), {}, 0, false};
              r.host.JxW = JxW[q];
              r.host.face.normal = normals[q];
              if (q < tangents.size())
                {
                  r.n_tangents = tangents[q].size();
                  libmesh_assert_less_equal(r.n_tangents, std::size(r.host.face.tangents));
                  for (const auto t : make_range(r.n_tangents))
                    r.host.face.tangents[t] = tangents[q][t];
                }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
              if (q < curvatures.size())
                {
                  r.has_curvature = true;
                  r.host.curvature = curvatures[q];
                }
#endif
              refs.push_back(r);
            }
        }
    }

  const auto d_in = KokkosOracle::upload_objects(inputs, "face_map_in");
  const std::size_t n_cases = inputs.size();
  ::Kokkos::View<Output *> d_out("face_map_out", n_cases);

  ::Kokkos::parallel_for(
    "face_map_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const Input in = d_in(k);
      const unsigned int s = in.side;
      const auto elem_eval = b.reference_map<T>(in.e);
      const auto elem_nodes = b.reference_nodes(in.e);
      const auto side_nodes = [=](const unsigned int i)
        { return elem_nodes(Elem::local_side_node(T, s, i)); };
      const auto side_eval = lagrange_map_eval(Elem::side_type(T, s), side_nodes);

      const Point & qp = in.qp;

      // FEMap asserts each Jacobian is positive; every element here
      // is valid, so the kernel skips that check
      const auto no_check = [](const Real) {};

      FEFaceMapPoint<> f;
      Real curvature = 0;
      switch (dim)
        {
        case 2:
          {
            // The element's map derivatives at the side's point, found
            // on the element by inverse mapping it
            const Point xyz = side_eval.map(qp);
            const auto parent_derivs = [&](RealVectorValue & dx_dxi, RealVectorValue & dx_deta)
              {
                const NothrowInverseMapPolicy policy;
                bool converged = false;
                const Point p = fe_inverse_map(dim, xyz, TOLERANCE, true, elem_eval, policy, converged);
                dx_dxi  = elem_eval.map_deriv(0, p);
                dx_deta = elem_eval.map_deriv(1, p);
              };
            const Point dxyzdxi = side_eval.map_deriv(0, qp);
            fe_face_map_2D(dxyzdxi, parent_derivs, f, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            curvature = fe_face_curvature_2D(dxyzdxi, side_eval.map_second_deriv(0, qp), f.normal);
#endif
            break;
          }
        case 3:
          {
            const Point dxyzdxi = side_eval.map_deriv(0, qp);
            const Point dxyzdeta = side_eval.map_deriv(1, qp);
            fe_face_map_3D(dxyzdxi, dxyzdeta, f, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            curvature = fe_face_curvature_3D(dxyzdxi, dxyzdeta,
                                             side_eval.map_second_deriv(0, qp),
                                             side_eval.map_second_deriv(1, qp),
                                             side_eval.map_second_deriv(2, qp),
                                             f.normal);
#endif
            break;
          }
        default:
          break;
        }

      Output & out = d_out(k);
      out.JxW = f.jac * in.w;
      out.face = f;
      out.curvature = curvature;
    });

  auto h_out = ::Kokkos::create_mirror_view(d_out);
  ::Kokkos::deep_copy(h_out, d_out);

  for (const auto k : make_range(n_cases))
    {
      const Input & in = inputs[k];
      const Reference & r = refs[k];
      const Output & kernel = h_out(k);

      // Which value of this case mismatched, for check() to print
      const auto value = [&](const std::string & name)
        {
          return [&in, &r, name]()
            {
              std::ostringstream os;
              os << "type " << int(T) << " elem " << r.id << " side " << in.side
                 << " qp " << in.qp << " " << name;
              return os.str();
            };
        };
      const auto check_vector = [&](const std::string & name, const RealVectorValue & host_value,
                                    const RealVectorValue & kernel_value)
        {
          for (const auto d : make_range(LIBMESH_DIM))
            check(host_value(d), kernel_value(d), value(name + "(" + std::to_string(d) + ")"));
        };

      check(r.host.JxW, kernel.JxW, value("JxW"));
      check_vector("normal", r.host.face.normal, kernel.face.normal);
      for (const auto t : make_range(r.n_tangents))
        check_vector("tangent " + std::to_string(t), r.host.face.tangents[t], kernel.face.tangents[t]);
      if (r.has_curvature)
        check(r.host.curvature, kernel.curvature, value("curvature"));
    }
  c.n_points += n_cases;
}

// check_bucket() on each bucket of a mirror of mesh, compiled for its
// type
inline void
check_mesh(const MeshBase & mesh, KokkosOracle::ValueCheck & check, Counts & c)
{
  const MeshMirror mirror(mesh);
  for (const auto k : make_range(mirror.n_buckets()))
    {
      const MeshMirror::Bucket & b = mirror.bucket(k);
      const auto check_typed = [&](auto traits)
        { check_bucket<decltype(traits)::type>(mesh, b, check, c); };
      if (Elem::type_to_dim_map[b.type] == 2)
        libMesh::Kokkos::dispatch_elem_type<libMesh::Kokkos::LagrangeElemTypes2D>(b.type, check_typed);
      else
        libMesh::Kokkos::dispatch_elem_type<libMesh::Kokkos::LagrangeElemTypes3D>(b.type, check_typed);
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
  KokkosOracle::ValueCheck check{"face_map_oracle", 1e-10};
  Counts c;

  // The 2D and 3D types, distorted; then the 2D ones tilted into 3D
  std::vector<ElemType> types_2D_3D, types_2D;
  for (const ElemType t : lagrange_types)
    {
      if (Elem::type_to_dim_map[t] >= 2)
        types_2D_3D.push_back(t);
      if (Elem::type_to_dim_map[t] == 2)
        types_2D.push_back(t);
    }
  {
    ReplicatedMesh mesh(comm, 3);
    add_lagrange_elements(mesh, types_2D_3D, /*tilt=*/false);
    mesh.prepare_for_use();
    check_mesh(mesh, check, c);
  }
  {
    ReplicatedMesh mesh(comm, 3);
    add_lagrange_elements(mesh, types_2D, /*tilt=*/true);
    mesh.prepare_for_use();
    check_mesh(mesh, check, c);
  }

  // Each processor checked its own mirror
  comm.sum(check.n_fail);
  comm.sum(check.n_values);
  comm.sum(c.n_points);
  if (comm.rank() == 0)
    std::printf("[face_map_oracle] %s  (%d failures on %u processors; %zu points, %d values)\n",
                check.n_fail ? "FAIL" : "PASS",
                check.n_fail, unsigned(comm.size()), c.n_points, check.n_values);
  return check.n_fail;
}

} // namespace KokkosFaceMapOracle
} // namespace libMeshTest

#endif // KOKKOS_FACE_MAP_ORACLE_H
