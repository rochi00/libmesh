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

// Checks fe_map_*D(), fe_map_second_derivs_*(), fe_physical_derivs()
// and fe_physical_second_derivs_*D(), run in a Kokkos kernel on map
// derivatives from MeshMirror::Bucket::reference_map<T>() and shape
// derivatives from fe_shape_*(), against FEMap::compute_map() and
// FE::reinit() on the host: one element of every Lagrange type,
// distorted and, for 1D and 2D, tilted into 3D, mirrored to the
// device, at each quadrature point.  Each bucket's kernel is compiled
// for its type, through dispatch_elem_type(), as library kernels are.
// The kernel must give JxW, the inverse map's first and second
// derivatives and the Lagrange shapes' physical first and second
// derivatives to within rounding.  On elements with an
// affine map FEMap computes neither JxW, which it rescales from the
// first point's, nor second derivatives at the other points, so those
// aren't checked there.  On several processors each checks the
// elements of its own mirror.

#ifndef KOKKOS_VOLUME_MAP_ORACLE_H
#define KOKKOS_VOLUME_MAP_ORACLE_H

#include "mesh/kokkos_mesh_mirror_oracle.h"

#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/fe_shape.h"
#include "libmesh/int_range.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/replicated_mesh.h"

#include <cmath>
#include <cstdio>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace libMeshTest
{
namespace KokkosVolumeMapOracle
{

using namespace libMesh;
using libMesh::Kokkos::MeshMirror;
using index_type = MeshMirror::index_type;

// The most shapes an element of a Lagrange type has, a HEX27's
constexpr unsigned int max_shapes = 27;

// The (row, column) of each of the xx, xy, xz, yy, yz and zz second
// derivatives, the order FEMapPoint and fe_physical_second_derivs_*D()
// store them in, in a full Hessian
constexpr std::pair<unsigned int, unsigned int> hessian_entries[] =
  {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

// One quadrature point: an element of the bucket, and the point on the
// reference element and its weight
struct Input
{
  index_type e;
  Point qp;
  Real w;
};

// The kernel's results at one quadrature point, and FEMap's and FE's:
// JxW, the inverse map's derivatives, and each shape's physical
// derivatives.  FEMap doesn't expose map.det, map.jac or (J^T J)^-1,
// so those are left out of the comparison.
struct Output
{
  Real JxW = 0;

  // The inverse map's first and second derivatives
  FEMapPoint<> map;

  // Each shape's physical gradient
  RealVectorValue dphi[max_shapes];
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  // In the order of hessian_entries
  Real d2phi[max_shapes][std::size(hessian_entries)] = {};
#endif
};

// What FEMap and FE computed at one quadrature point, with which of
// the map's values FEMap computed there and how many shapes' first and
// second derivatives FE did
struct Reference
{
  // The element's id, to print a mismatch
  dof_id_type id;

  Output host;

  // Whether FEMap computed JxW at this point, not rescaled it
  bool has_JxW;
  // Whether FEMap computed d(xi, eta, zeta)/d(x, y, z), indexed by the
  // reference direction and then the physical one
  bool has_dxi[3][3] = {};
  // Whether FEMap computed all six second derivatives of xi, eta and
  // zeta
  bool has_d2xi[3] = {};

  // How many shapes' gradients and Hessians FE computed
  unsigned int n_grads, n_hessians;
};

// What the checks found, for the summary
struct Counts
{
  std::size_t n_points = 0;
};

// The gradient of reference coordinate j, xi, eta or zeta, in m
inline const RealVectorValue &
ref_grad(const FEMapPoint<> & m, const unsigned int j)
{
  return (j == 0) ? m.dxi : (j == 1) ? m.deta : m.dzeta;
}

// The same, to set
inline RealVectorValue &
ref_grad(FEMapPoint<> & m, const unsigned int j)
{
  return (j == 0) ? m.dxi : (j == 1) ? m.deta : m.dzeta;
}

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
// The second derivatives of reference coordinate j in m
inline const Real *
ref_hessian(const FEMapPoint<> & m, const unsigned int j)
{
  return (j == 0) ? m.d2xi : (j == 1) ? m.d2eta : m.d2zeta;
}

// The same, to set
inline Real *
ref_hessian(FEMapPoint<> & m, const unsigned int j)
{
  return (j == 0) ? m.d2xi : (j == 1) ? m.d2eta : m.d2zeta;
}
#endif

// The quadrature points of bucket b, of type T, against FEMap and FE
// on the elements of mesh they mirror
template <ElemType T>
inline void
check_bucket(const MeshBase & mesh, const MeshMirror::Bucket & b,
             KokkosOracle::ValueCheck & check, Counts & c)
{
  constexpr unsigned int dim = Elem::type_to_dim_map[T];
  constexpr Order order = Elem::type_to_default_order_map[T];
  static_assert(Elem::type_to_n_nodes_map[T] <= max_shapes);

  // The element's own LAGRANGE shapes, whose physical derivatives the
  // kernel computes too; a FIFTH order rule has interior points, as
  // many as 3 x 3 x 3 on a hex
  std::unique_ptr<FEBase> fe = FEBase::build(dim, FEType(order, LAGRANGE));
  QGauss qrule(dim, FIFTH);
  fe->attach_quadrature_rule(&qrule);
  // FEMap's derivative of each reference coordinate in each physical
  // direction, in the order of Reference::has_dxi
  const FEMap & map = fe->get_fe_map();
  const std::vector<Real> * first[3][3] =
    {{&map.get_dxidx(), &map.get_dxidy(), &map.get_dxidz()},
     {&map.get_detadx(), &map.get_detady(), &map.get_detadz()},
     {&map.get_dzetadx(), &map.get_dzetady(), &map.get_dzetadz()}};
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  const auto & d2phi = fe->get_d2phi();
  const std::vector<std::vector<Real>> * second[] =
    {&map.get_d2xidxyz2(), &map.get_d2etadxyz2(), &map.get_d2zetadxyz2()};
#endif
  const auto & dphi = fe->get_dphi();
  const auto & JxW = fe->get_JxW();

  auto h_id = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.id);
  std::vector<Input> inputs;
  std::vector<Reference> refs;
  for (const auto e : make_range(b.n_elem))
    {
      const Elem & elem = mesh.elem_ref(h_id(e));
      fe->reinit(&elem);
      const bool affine = elem.has_affine_map();

      for (const auto q : index_range(JxW))
        {
          inputs.push_back({e, qrule.qp(q), qrule.w(q)});

          Reference r;
          r.id = elem.id();
          r.has_JxW = !affine || q == 0;
          r.host.JxW = JxW[q];
          r.n_grads = dphi.size();
          r.n_hessians = 0;
          for (const auto i : make_range(r.n_grads))
            r.host.dphi[i] = dphi[i][q];
          // FEMap sizes only the derivatives it computes, of the
          // element's own reference coordinates
          for (const auto j : make_range(std::size(first)))
            for (const auto d : make_range(std::size(first[j])))
              if (q < first[j][d]->size())
                {
                  ref_grad(r.host.map, j)(d) = (*first[j][d])[q];
                  r.has_dxi[j][d] = true;
                }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          // FEMap computes an affine map's second derivatives at its
          // first point only, so skip them, and the shapes' FE
          // computes from them, on an affine element
          if (!affine)
            {
              for (const auto j : make_range(std::size(second)))
                if (q < second[j]->size() &&
                    (*second[j])[q].size() == std::size(hessian_entries))
                  {
                    for (const auto h : make_range(std::size(hessian_entries)))
                      ref_hessian(r.host.map, j)[h] = (*second[j])[q][h];
                    r.has_d2xi[j] = true;
                  }

              r.n_hessians = d2phi.size();
              for (const auto i : make_range(r.n_hessians))
                for (const auto h : make_range(std::size(hessian_entries)))
                  r.host.d2phi[i][h] = d2phi[i][q](hessian_entries[h].first,
                                                   hessian_entries[h].second);
            }
#endif
          refs.push_back(r);
        }
    }

  const auto d_in = KokkosOracle::upload_objects(inputs, "volume_map_in");
  const std::size_t n_cases = inputs.size();
  ::Kokkos::View<Output *> d_out("volume_map_out", n_cases);

  ::Kokkos::parallel_for(
    "volume_map_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const Input in = d_in(k);
      const auto eval = b.reference_map<T>(in.e);

      const Point & qp = in.qp;

      // The map's derivatives in each reference direction, and its
      // second derivatives, xixi, xieta, etaeta, xizeta, etazeta,
      // zetazeta
      Point dxyz[3];
      for (unsigned int j = 0; j != dim; ++j)
        dxyz[j] = eval.map_deriv(j, qp);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
      RealVectorValue d2xyz[6];
      for (unsigned int j = 0; j != dim*(dim+1)/2; ++j)
        d2xyz[j] = eval.map_second_deriv(j, qp);
#endif

      // FEMap asserts each Jacobian is positive; every element here
      // is valid, so the kernel skips that check
      const auto no_check = [](const Real) {};

      Output & out = d_out(k);
      FEMapPoint<> & m = out.map;
      switch (dim)
        {
        case 1:
          fe_map_1D(dxyz[0], m, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          fe_map_second_derivs_1D(dxyz[0], d2xyz, m);
#endif
          break;
        case 2:
          fe_map_2D(dxyz[0], dxyz[1], m, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          fe_map_second_derivs_2D(dxyz[0], dxyz[1], d2xyz, m);
#endif
          break;
        case 3:
          fe_map_3D(dxyz[0], dxyz[1], dxyz[2], m, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          fe_map_second_derivs_square(d2xyz, m);
#endif
          break;
        default:
          break;
        }
      out.JxW = m.jac * in.w;

      // A Lagrange element of its default order has a shape per node
      for (unsigned int i = 0; i != eval.n_nodes; ++i)
        {
          Real dphi[3] = {}, d[3] = {};
          for (unsigned int j = 0; j != dim; ++j)
            dphi[j] = fe_shape_deriv(dim, LAGRANGE, T, order, i, j, qp);
          fe_physical_derivs(dim, dphi, m, d);
          out.dphi[i] = RealVectorValue(d[0], d[1], d[2]);

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          Real d2phi[6] = {};
          for (unsigned int j = 0; j != dim*(dim+1)/2; ++j)
            d2phi[j] = fe_shape_second_deriv(dim, LAGRANGE, T, order, i, j, qp);
          if (dim == 1)
            fe_physical_second_derivs_1D(dphi, d2phi, m, out.d2phi[i]);
          else if (dim == 2)
            fe_physical_second_derivs_2D(dphi, d2phi, m, out.d2phi[i]);
          else
            fe_physical_second_derivs_3D(dphi, d2phi, m, out.d2phi[i]);
#endif
        }
    });

  auto h_out = ::Kokkos::create_mirror_view(d_out);
  ::Kokkos::deep_copy(h_out, d_out);

  const char * const ref_names[] = {"xi", "eta", "zeta"};
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
              os << "type " << int(T) << " elem " << r.id << " qp " << in.qp << " " << name;
              return os.str();
            };
        };

      if (r.has_JxW)
        check(r.host.JxW, kernel.JxW, value("JxW"));
      for (const auto j : make_range(std::size(r.has_dxi)))
        for (const auto d : make_range(std::size(r.has_dxi[j])))
          if (r.has_dxi[j][d])
            check(ref_grad(r.host.map, j)(d), ref_grad(kernel.map, j)(d),
                  value(std::string("d") + ref_names[j] + "(" + std::to_string(d) + ")"));
      for (const auto i : make_range(r.n_grads))
        for (const auto d : make_range(LIBMESH_DIM))
          check(r.host.dphi[i](d), kernel.dphi[i](d),
                value("dphi " + std::to_string(i) + "(" + std::to_string(d) + ")"));
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
      for (const auto j : make_range(std::size(r.has_d2xi)))
        if (r.has_d2xi[j])
          for (const auto h : make_range(std::size(hessian_entries)))
            check(ref_hessian(r.host.map, j)[h], ref_hessian(kernel.map, j)[h],
                  value(std::string("d2") + ref_names[j] + "[" + std::to_string(h) + "]"));
      for (const auto i : make_range(r.n_hessians))
        for (const auto h : make_range(std::size(hessian_entries)))
          check(r.host.d2phi[i][h], kernel.d2phi[i][h],
                value("d2phi " + std::to_string(i) + "[" + std::to_string(h) + "]"));
#endif
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
  KokkosOracle::ValueCheck check{"volume_map_oracle", 1e-10};
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
  comm.sum(check.n_fail);
  comm.sum(check.n_values);
  comm.sum(c.n_points);
  if (comm.rank() == 0)
    std::printf("[volume_map_oracle] %s  (%d failures on %u processors; %zu points, %d values)\n",
                check.n_fail ? "FAIL" : "PASS",
                check.n_fail, unsigned(comm.size()), c.n_points, check.n_values);
  return check.n_fail;
}

} // namespace KokkosVolumeMapOracle
} // namespace libMeshTest

#endif // KOKKOS_VOLUME_MAP_ORACLE_H
