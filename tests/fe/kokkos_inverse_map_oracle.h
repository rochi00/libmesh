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

// Checks fe_inverse_map(), run in a Kokkos kernel on the map
// MeshMirror::Bucket::reference_map<T>() gives and with
// NothrowInverseMapPolicy, against FEMap::inverse_map() on the host:
// one element of every Lagrange type, distorted, tilted into 3D and
// collapsed, mirrored to the device, at points inside, on, outside and
// far from each, secure or not.  Each bucket's kernel is compiled for
// its type, through dispatch_elem_type(), as library kernels are.
// Where FEMap returns a point, a singular node included, the kernel
// must return the same one, to within rounding; where FEMap throws,
// the kernel must report Newton hasn't converged.  Where the kernel's
// rounding differs from the host's, as on a device, it may decide
// whether Newton converges or stalls: a case where one side stalls and
// the other converges is dropped.
// Where Newton converges to a singular node, such as a pyramid's apex,
// it determines the point only to within its tolerance, so there the
// kernel's point must be within the tolerance of the host's.  On
// several processors each checks the elements of its own mirror.

#ifndef KOKKOS_INVERSE_MAP_ORACLE_H
#define KOKKOS_INVERSE_MAP_ORACLE_H

#include "mesh/kokkos_mesh_mirror_oracle.h"

#include "libmesh/elem.h"
#include "libmesh/fe_map.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/int_range.h"
#include "libmesh/libmesh_exceptions.h"
#include "libmesh/replicated_mesh.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

namespace libMeshTest
{
namespace KokkosInverseMapOracle
{

using namespace libMesh;
using libMesh::Kokkos::MeshMirror;
using index_type = MeshMirror::index_type;

// One case: an element of the bucket, and the point to inverse map
struct Input
{
  index_type e;
  Point x;

  // Newton's tolerance on the reference point, and whether a failure
  // to converge throws, as FEMap::inverse_map() takes them
  Real tol;
  bool secure;
};

// What fe_inverse_map() returned in the kernel
struct Output
{
  // The reference point, and whether Newton converged to it
  Point p;
  bool converged = false;

  // Whether Newton met a singular Jacobian
  bool singular = false;
};

// What FEMap::inverse_map() did
struct Reference
{
  // The element's id, to print a mismatch
  dof_id_type id;

  // Whether FEMap threw DegenerateMap, and the point it returned if not
  bool threw = false;
  Point p;

  // Whether that point is a singular node, which FEMap returns for a
  // physical point at it
  bool at_singular_node = false;

  // Whether FEMap returned a point within the tolerance of a singular node
  bool near_singular_node = false;

  // Whether Newton met a singular Jacobian on the host
  bool singular = false;
};

// What the checks found, for the summary
struct Counts
{
  int n_fail = 0;

  // Cases where FEMap returned a point, threw, or returned a singular
  // node, and cases dropped where one side stalled
  std::size_t n_point = 0, n_threw = 0, n_singular_node = 0, n_stalled = 0;
};

// Whether p is the far away point fe_inverse_map() returns when Newton
// gives up
inline bool
is_far_point(const unsigned int dim, const Point & p)
{
  Point far = p;
  fe_inverse_map_far_point(dim, far);
  return far == p;
}

// A map evaluator and a policy for fe_inverse_map() that record whether
// Newton met a singular Jacobian: fe_inverse_map() calls singular_node()
// on every singular 3D one, and singular() on a singular 1D or 2D one
// when secure.  Newton that stops without either has stalled.
template <typename MapEval>
struct SingularRecordingEval
{
  MapEval eval;
  bool * met_singular;

  LIBMESH_DEVICE_INLINE
  Point map(const Point & p) const
  { return eval.map(p); }

  LIBMESH_DEVICE_INLINE
  Point map_deriv(const unsigned int j, const Point & p) const
  { return eval.map_deriv(j, p); }

  LIBMESH_DEVICE_INLINE
  bool singular_node(const Point & physical_point, const Real tolerance, Point & p) const
  {
    *met_singular = true;
    return eval.singular_node(physical_point, tolerance, p);
  }
};

// NothrowInverseMapPolicy, recording in *met_singular whether Newton
// met a singular Jacobian
struct SingularRecordingPolicy
{
  NothrowInverseMapPolicy policy;
  bool * met_singular;

  LIBMESH_DEVICE_INLINE
  bool singular(const Point & p) const
  {
    *met_singular = true;
    return policy.singular(p);
  }

  LIBMESH_DEVICE_INLINE
  bool not_converged(const unsigned int cnt,
                     const unsigned int max_cnt,
                     const Point & physical_guess,
                     const Point & dp,
                     const Point & p,
                     const Real inverse_map_error) const
  { return policy.not_converged(cnt, max_cnt, physical_guess, dp, p, inverse_map_error); }
};

// Whether the host's Newton converged to a point
inline bool
host_converged(const unsigned int dim, const Reference & r)
{
  return !r.threw && !r.at_singular_node && !r.singular && !is_far_point(dim, r.p);
}

// The cases of bucket b, of type T, against FEMap on the elements of
// mesh they mirror; collapsed if every element of mesh has no extent
template <ElemType T>
inline void
check_bucket(const MeshBase & mesh, const MeshMirror::Bucket & b, const bool collapsed, Counts & c)
{
  constexpr unsigned int dim = Elem::type_to_dim_map[T];

  // The square of FEMap::inverse_map()'s default tolerance, close
  // enough to rounding that Newton may stall, and the default itself,
  // TOLERANCE in double precision
  const Real tols[] = {TOLERANCE * TOLERANCE, 1.e-6};

  auto h_id = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), b.id);
  std::vector<Input> inputs;
  std::vector<Reference> refs;
  for (const auto e : make_range(b.n_elem))
    {
      const Elem & elem = mesh.elem_ref(h_id(e));

      // The images of points inside and outside the element; the
      // vertices, singular nodes such as a pyramid's apex among
      // them; and a point so far away that Newton fails
      std::vector<Point> targets;
      for (const Point & r : {Point(0.2, 0.15, 0.1), Point(1./3, 1./3, 0.25),
                              Point(-0.4, 0.3, 0.2), Point(1.4, -0.3, 0.6)})
        targets.push_back(FEMap::map(dim, &elem, r));
      for (const auto n : make_range(elem.n_vertices()))
        targets.push_back(elem.point(n));
      targets.push_back(Point(100., -50., 25.));

      for (const Point & x : targets)
        for (const Real tol : tols)
          for (const bool secure : {false, true})
            {
              // Unless secure, the 1D and 2D Newton steps don't check
              // for a singular Jacobian: on an element of no extent
              // they divide zero by zero, and the result is undefined
              if (collapsed && !secure && dim < 3)
                continue;

              inputs.push_back({e, x, tol, secure});
              Reference r;
              r.id = elem.id();
              try
                {
                  r.p = FEMap::inverse_map(dim, &elem, x, tol, secure);
                  for (const auto n : make_range(elem.n_nodes()))
                    if (elem.is_singular_node(n))
                      {
                        if (r.p == elem.master_point(n))
                          r.at_singular_node = true;
                        if ((r.p - elem.master_point(n)).norm() <= tol)
                          r.near_singular_node = true;
                      }
                }
              catch (const DegenerateMap &)
                {
                  r.threw = true;
                }

              // Whether FEMap met a singular Jacobian: the same Newton
              // with the same map, as LagrangeMapEval evaluates it
              const auto host_nodes = [&elem](const unsigned int i) { return elem.point(i); };
              const auto host_eval = lagrange_map_eval<T>(host_nodes);
              // Whether it converged is already in r
              bool host_converged_too = false;
              fe_inverse_map(dim, x, tol, secure,
                             SingularRecordingEval<decltype(host_eval)>{host_eval, &r.singular},
                             SingularRecordingPolicy{{}, &r.singular}, host_converged_too);

              refs.push_back(r);
            }
    }

  const auto d_in = KokkosOracle::upload_objects(inputs, "inverse_map_in");
  const std::size_t n_cases = inputs.size();
  ::Kokkos::View<Output *> d_out("inverse_map_out", n_cases);

  ::Kokkos::parallel_for(
    "inverse_map_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const Input in = d_in(k);
      const auto eval = b.reference_map<T>(in.e);
      Output out;
      const SingularRecordingEval<decltype(eval)> recording_eval{eval, &out.singular};
      const SingularRecordingPolicy policy{{}, &out.singular};
      out.p = fe_inverse_map(dim, in.x, in.tol, in.secure, recording_eval, policy, out.converged);
      d_out(k) = out;
    });

  auto h_out = ::Kokkos::create_mirror_view(d_out);
  ::Kokkos::deep_copy(h_out, d_out);

  for (const auto k : make_range(n_cases))
    {
      const Input & in = inputs[k];
      const Reference & r = refs[k];
      const Output & kernel = h_out(k);
      const bool converged = kernel.converged;
      const Point & p = kernel.p;

      // 1e-10: Newton converges quadratically, so even at the
      // looser tol a converged point is accurate to near rounding,
      // and the host's and device's rounding move it far less.
      // Relative to the point's size: far outside an element the
      // reference point is large, and rounding scales with it.
      const bool close = (p - r.p).norm() <= 1e-10 * std::max(Real(1), r.p.norm());

      bool ok;
      if (r.threw)
        {
          // FEMap threw DegenerateMap: a singular Jacobian or a failed Newton
          ok = !converged;
          ++c.n_threw;
        }
      else if (r.at_singular_node)
        {
          // FEMap returns the singular node, reporting Newton hasn't converged
          ok = !converged && close;
          ++c.n_singular_node;
        }
      else
        {
          ok = close;

          // Near a singular node the map's Jacobian depends on the
          // direction the point is approached from, so rounding moves
          // the last Newton step's direction, and the point Newton
          // converges to is determined only to within its tolerance.
          if (!ok && converged && r.near_singular_node)
            ok = (p - r.p).norm() <= in.tol;
          ++c.n_point;
        }

      // Where the kernel's rounding differs from the host's
      if (!ok && !r.at_singular_node)
        {
          const bool kernel_stalled = !converged && !kernel.singular;
          const bool host_stalled = !host_converged(dim, r) && !r.singular;
          if ((host_stalled && converged) || (host_converged(dim, r) && kernel_stalled))
            {
              // Newton steps that rounding keeps above a tight tolerance,
              // as far outside an element, make Newton stall before
              // max_cnt or not, so one side may converge where the other
              // gives up.  There is nothing to compare: drop the case.
              ok = true;
              ++c.n_stalled;
            }
        }

      if (!ok)
        {
          if (c.n_fail < 10)
            std::printf("[inverse_map_oracle] mismatch: type %d elem %lu secure %d x (%g %g %g): "
                        "host %s (%.17g %.17g %.17g), kernel converged %d (%.17g %.17g %.17g)\n",
                        int(T), (unsigned long)r.id, int(in.secure),
                        double(in.x(0)), double(in.x(1)), double(in.x(2)),
                        r.threw ? "threw" : "returned",
                        double(r.p(0)), double(r.p(1)), double(r.p(2)),
                        int(converged), double(p(0)), double(p(1)), double(p(2)));
          ++c.n_fail;
        }
    }
}

// check_bucket() on each bucket of a mirror of mesh, compiled for its
// type
inline void
check_mesh(const MeshBase & mesh, const bool collapsed, Counts & c)
{
  const MeshMirror mirror(mesh);
  for (const auto k : make_range(mirror.n_buckets()))
    {
      const MeshMirror::Bucket & b = mirror.bucket(k);
      libMesh::Kokkos::dispatch_elem_type<libMesh::Kokkos::LagrangeElemTypes>(b.type, [&](auto traits) {
          check_bucket<decltype(traits)::type>(mesh, b, collapsed, c);
        });
    }
}

inline int
run_all_oracles(const Parallel::Communicator & comm)
{
  using KokkosMeshMirrorOracle::add_lagrange_elements;
  using KokkosMeshMirrorOracle::lagrange_types;

  // FEMap warns on libMesh::err when Newton struggles; keep that out of
  // the test output.
  std::ostringstream quiet;
  libMesh::err.reset(quiet);

  Counts c;

  // Every type, distorted
  {
    ReplicatedMesh mesh(comm, 3);
    add_lagrange_elements(mesh, lagrange_types, /*tilt=*/false);
    mesh.prepare_for_use();
    check_mesh(mesh, /*collapsed=*/false, c);
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
    check_mesh(mesh, /*collapsed=*/false, c);
  }

  // Every type collapsed, every node at one, arbitrary, point; after
  // preparing the mesh, since partitioning a mesh of no extent on
  // several processors fails
  {
    ReplicatedMesh mesh(comm, 3);
    add_lagrange_elements(mesh, lagrange_types, /*tilt=*/false);
    mesh.prepare_for_use();
    for (Node * node : mesh.node_ptr_range())
      *node = Point(0.25, 0.5, 0.75);
    check_mesh(mesh, /*collapsed=*/true, c);
  }

  libMesh::err.reset(std::cerr);

  // Each processor checked its own mirror
  comm.sum(c.n_fail);
  for (std::size_t * n : {&c.n_point, &c.n_threw, &c.n_singular_node, &c.n_stalled})
    comm.sum(*n);
  if (comm.rank() == 0)
    std::printf("[inverse_map_oracle] %s  (%d failures on %u processors; %zu points, "
                "%zu throws, %zu singular nodes, %zu dropped where one side stalled)\n",
                c.n_fail ? "FAIL" : "PASS", c.n_fail,
                unsigned(comm.size()), c.n_point, c.n_threw, c.n_singular_node, c.n_stalled);
  return c.n_fail;
}

} // namespace KokkosInverseMapOracle
} // namespace libMeshTest

#endif // KOKKOS_INVERSE_MAP_ORACLE_H
