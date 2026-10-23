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

#ifndef LIBMESH_FE_MAP_POINT_H
#define LIBMESH_FE_MAP_POINT_H

// Local includes
#include "libmesh/libmesh_device.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/fe_shape.h"
#include "libmesh/point.h"
#include "libmesh/tensor_value.h"
#include "libmesh/vector_value.h"

// C++ includes
#include <cmath> // std::sqrt
#include <new> // std::nothrow_t

// The Jacobian of the reference-to-physical map at one point, and the
// inverse map's first derivatives, from the map's first derivatives
// there.  Each calls check(det) where FEMap::compute_single_point_map()
// checks for a degenerate map, before dividing by it: FEMap passes its
// check, which throws, and device code one that records the failure.
// FEMap calls these; they are inline here so that device code maps
// points the same way, and templated on the coordinate type, as the
// shared shape functions are, so that automatic differentiation does.

namespace libMesh
{

/**
 * One point of the map.  \p det is the quantity checked for a
 * degenerate map: \p jac itself, except for a 2D element in 3D space,
 * where it is det(J^T J) and \p jac is its square root.
 */
template <typename T = Real>
struct FEMapPoint
{
  T det = 0;
  T jac = 0;

  // The gradients of xi, eta and zeta with respect to (x, y, z)
  VectorValue<T> dxi, deta, dzeta;

#if LIBMESH_DIM == 3
  // (J^T J)^-1 for a 2D element in 3D space, which its inverse map
  // second derivatives also need.
  T g11inv = 0, g12inv = 0, g21inv = 0, g22inv = 0;
#endif
};



template <typename T, typename DegenerateCheck>
LIBMESH_DEVICE_INLINE
void fe_map_1D(const TypeVector<T> & dxyzdxi,
               FEMapPoint<T> & m,
               const DegenerateCheck & check)
{
  // 1D elements can live in 2D or 3D space.
  // The transformation matrix from local->global
  // coordinates is
  //
  // T = | dx/dxi |
  //     | dy/dxi |
  //     | dz/dxi |
  //
  // The generalized determinant of T (from the
  // so-called "normal" eqns.) is
  // jac = "det(T)" = sqrt(det(T'T))
  //
  // where T'= transpose of T, so
  //
  // jac = sqrt( (dx/dxi)^2 + (dy/dxi)^2 + (dz/dxi)^2 )
  m.det = m.jac = dxyzdxi.norm();

  check(m.det);

  // The inverse Jacobian entries also come from the
  // generalized inverse of T (see also the 2D element
  // living in 3D code).
  const T jacm2 = 1./m.jac/m.jac;
  m.dxi(0) = jacm2*dxyzdxi(0);
#if LIBMESH_DIM > 1
  m.dxi(1) = jacm2*dxyzdxi(1);
#endif
#if LIBMESH_DIM > 2
  m.dxi(2) = jacm2*dxyzdxi(2);
#endif
}



template <typename T, typename DegenerateCheck>
LIBMESH_DEVICE_INLINE
void fe_map_2D(const TypeVector<T> & dxyzdxi,
               const TypeVector<T> & dxyzdeta,
               FEMapPoint<T> & m,
               const DegenerateCheck & check)
{
  // compute the jacobian once
  const T dx_dxi = dxyzdxi(0),
    dx_deta = dxyzdeta(0),
    dy_dxi = dxyzdxi(1),
    dy_deta = dxyzdeta(1);

#if LIBMESH_DIM == 2
  // Compute the Jacobian.  This assumes the 2D face
  // lives in 2D space
  //
  // Symbolically, the matrix determinant is
  //
  //         | dx/dxi  dx/deta |
  // jac =   | dy/dxi  dy/deta |
  //
  // jac = dx/dxi*dy/deta - dx/deta*dy/dxi
  m.det = m.jac = (dx_dxi*dy_deta - dx_deta*dy_dxi);

  check(m.det);

  // Compute the shape function derivatives wrt x,y at the
  // quadrature points
  const T inv_jac = 1./m.jac;

  m.dxi(0)  =  dy_deta*inv_jac; //dxi/dx  =  (1/J)*dy/deta
  m.dxi(1)  = -dx_deta*inv_jac; //dxi/dy  = -(1/J)*dx/deta
  m.deta(0) = -dy_dxi* inv_jac; //deta/dx = -(1/J)*dy/dxi
  m.deta(1) =  dx_dxi* inv_jac; //deta/dy =  (1/J)*dx/dxi
#else // LIBMESH_DIM == 3

  const T dz_dxi = dxyzdxi(2),
    dz_deta = dxyzdeta(2);

  // Compute the Jacobian.  This assumes a 2D face in
  // 3D space.
  //
  // The transformation matrix T from local to global
  // coordinates is
  //
  //         | dx/dxi  dx/deta |
  //     T = | dy/dxi  dy/deta |
  //         | dz/dxi  dz/deta |
  // note det(T' T) = det(T')det(T) = det(T)det(T)
  // so det(T) = std::sqrt(det(T' T))
  //
  //----------------------------------------------
  // Notes:
  //
  //       dX = R dXi -> R'dX = R'R dXi
  // (R^-1)dX =   dXi    [(R'R)^-1 R']dX = dXi
  //
  // so R^-1 = (R'R)^-1 R'
  //
  // and R^-1 R = (R'R)^-1 R'R = I.
  //
  const T g11 = (dx_dxi*dx_dxi +
                 dy_dxi*dy_dxi +
                 dz_dxi*dz_dxi);

  const T g12 = (dx_dxi*dx_deta +
                 dy_dxi*dy_deta +
                 dz_dxi*dz_deta);

  const T g21 = g12;

  const T g22 = (dx_deta*dx_deta +
                 dy_deta*dy_deta +
                 dz_deta*dz_deta);

  m.det = (g11*g22 - g12*g21);

  check(m.det);

  const T inv_det = 1./m.det;
  using std::sqrt;
  m.jac = sqrt(m.det);

  m.g11inv =  g22*inv_det;
  m.g12inv = -g12*inv_det;
  m.g21inv = -g21*inv_det;
  m.g22inv =  g11*inv_det;

  m.dxi(0)  = m.g11inv*dx_dxi + m.g12inv*dx_deta;
  m.dxi(1)  = m.g11inv*dy_dxi + m.g12inv*dy_deta;
  m.dxi(2)  = m.g11inv*dz_dxi + m.g12inv*dz_deta;

  m.deta(0) = m.g21inv*dx_dxi + m.g22inv*dx_deta;
  m.deta(1) = m.g21inv*dy_dxi + m.g22inv*dy_deta;
  m.deta(2) = m.g21inv*dz_dxi + m.g22inv*dz_deta;
#endif // LIBMESH_DIM == 3
}



template <typename T, typename DegenerateCheck>
LIBMESH_DEVICE_INLINE
void fe_map_3D(const TypeVector<T> & dxyzdxi,
               const TypeVector<T> & dxyzdeta,
               const TypeVector<T> & dxyzdzeta,
               FEMapPoint<T> & m,
               const DegenerateCheck & check)
{
  // compute the jacobian
  const T
    dx_dxi   = dxyzdxi(0),   dy_dxi   = dxyzdxi(1),   dz_dxi   = dxyzdxi(2),
    dx_deta  = dxyzdeta(0),  dy_deta  = dxyzdeta(1),  dz_deta  = dxyzdeta(2),
    dx_dzeta = dxyzdzeta(0), dy_dzeta = dxyzdzeta(1), dz_dzeta = dxyzdzeta(2);

  // Symbolically, the matrix determinant is
  //
  //         | dx/dxi   dy/dxi   dz/dxi   |
  // jac =   | dx/deta  dy/deta  dz/deta  |
  //         | dx/dzeta dy/dzeta dz/dzeta |
  //
  // jac = dx/dxi*(dy/deta*dz/dzeta - dz/deta*dy/dzeta) +
  //       dy/dxi*(dz/deta*dx/dzeta - dx/deta*dz/dzeta) +
  //       dz/dxi*(dx/deta*dy/dzeta - dy/deta*dx/dzeta)

  m.det = m.jac = (dx_dxi*(dy_deta*dz_dzeta - dz_deta*dy_dzeta)  +
                   dy_dxi*(dz_deta*dx_dzeta - dx_deta*dz_dzeta)  +
                   dz_dxi*(dx_deta*dy_dzeta - dy_deta*dx_dzeta));

  check(m.det);

  // Compute the shape function derivatives wrt x,y at the
  // quadrature points
  const T inv_jac  = 1./m.jac;

  m.dxi(0)   = (dy_deta*dz_dzeta - dz_deta*dy_dzeta)*inv_jac;
  m.dxi(1)   = (dz_deta*dx_dzeta - dx_deta*dz_dzeta)*inv_jac;
  m.dxi(2)   = (dx_deta*dy_dzeta - dy_deta*dx_dzeta)*inv_jac;

  m.deta(0)  = (dz_dxi*dy_dzeta  - dy_dxi*dz_dzeta )*inv_jac;
  m.deta(1)  = (dx_dxi*dz_dzeta  - dz_dxi*dx_dzeta )*inv_jac;
  m.deta(2)  = (dy_dxi*dx_dzeta  - dx_dxi*dy_dzeta )*inv_jac;

  m.dzeta(0) = (dy_dxi*dz_deta   - dz_dxi*dy_deta  )*inv_jac;
  m.dzeta(1) = (dz_dxi*dx_deta   - dx_dxi*dz_deta  )*inv_jac;
  m.dzeta(2) = (dx_dxi*dy_deta   - dy_dxi*dx_deta  )*inv_jac;
}



/**
 * Sets the first \p dim coordinates of \p p to the far away point
 * fe_inverse_map() and its policies return when Newton gives up.
 */
LIBMESH_DEVICE_INLINE
void fe_inverse_map_far_point(const unsigned int dim, Point & p)
{
  for (unsigned int i=0; i != dim; ++i)
    p(i) = 1e6;
}



/**
 * The Newton iteration of FEMap::inverse_map(): \returns the reference
 * point that \p eval maps to \p physical_point, to within \p
 * tolerance.  \p eval.map(p) and \p eval.map_deriv(j, p) evaluate the
 * map and its derivative in reference direction \p j.  \p policy
 * decides what a failure does, so that FEMap can throw and warn as it
 * always has and device code can record the failure instead:
 * \p policy.singular(p) is called on a singular 1D or 2D Jacobian when
 * \p secure, \p policy.solve(J, delta, dp, p) takes the 3D step, and
 * \p policy.not_converged(...) is called when \p secure and Newton has
 * taken too many steps.  singular() and not_converged() return true,
 * and solve() false, to stop and return \p p.  \p converged is set
 * false when Newton stops early like that, or gives up and returns a
 * far away point, and true when it converges.
 */
template <typename MapEval, typename InverseMapPolicy>
LIBMESH_DEVICE_INLINE
Point fe_inverse_map(const unsigned int dim,
                     const Point & physical_point,
                     const Real tolerance,
                     const bool secure,
                     const MapEval & eval,
                     const InverseMapPolicy & policy,
                     bool & converged)
{
  // How much did the point on the reference
  // element change by in this Newton step?
  Real inverse_map_error = 0.;

  //  The point on the reference element.  This is
  //  the "initial guess" for Newton's method.  The
  //  centroid seems like a good idea, but computing
  //  it is a little more intensive than, say taking
  //  the zero point.
  //
  //  Convergence should be insensitive of this choice
  //  for "good" elements.
  Point p; // the zero point.  No computation required

  //  The number of iterations in the map inversion process.
  unsigned int cnt = 0;

  //  The number of iterations after which we give up and declare
  //  divergence
  const unsigned int max_cnt = 10;

  //  The distance (in master element space) beyond which we give up
  //  and declare divergence.  This is no longer used...
  // Real max_step_length = 4.;



  //  Newton iteration loop.
  do
    {
      //  Where our current iterate \p p maps to.
      const Point physical_guess = eval.map(p);

      //  How far our current iterate is from the actual point.
      const Point delta = physical_point - physical_guess;

      //  Increment in current iterate \p p, will be computed.
      Point dp;


      //  The form of the map and how we invert it depends
      //  on the dimension that we are in.
      switch (dim)
        {
          // ------------------------------------------------------------------
          //  0D map inversion is trivial
        case 0:
          {
            break;
          }

          // ------------------------------------------------------------------
          //  1D map inversion
          //
          //  Here we find the point on a 1D reference element that maps to
          //  the point \p physical_point in the domain.  This is a bit tricky
          //  since we do not want to assume that the point \p physical_point
          //  is also in a 1D domain.  In particular, this method might get
          //  called on the edge of a 3D element, in which case
          //  \p physical_point actually lives in 3D.
        case 1:
          {
            const Point dxi = eval.map_deriv(0, p);

            //  Newton's method in this case looks like
            //
            //  {X} - {X_n} = [J]*dp
            //
            //  Where {X}, {X_n} are 3x1 vectors, [J] is a 3x1 matrix
            //  d(x,y,z)/dxi, and we seek dp, a scalar.  Since the above
            //  system is either overdetermined or rank-deficient, we will
            //  solve the normal equations for this system
            //
            //  [J]^T ({X} - {X_n}) = [J]^T [J] {dp}
            //
            //  which involves the trivial inversion of the scalar
            //  G = [J]^T [J]
            const Real G = dxi*dxi;

            if (secure && G <= 0 && policy.singular(p))
              {
                converged = false;
                return p;
              }

            const Real Ginv = 1./G;

            const Real  dxidelta = dxi*delta;

            dp(0) = Ginv*dxidelta;

            // No master elements have radius > 4, but sometimes we
            // can take a step that big while still converging
            // if (secure)
            // libmesh_assert_less (dp.size(), max_step_length);

            break;
          }



          // ------------------------------------------------------------------
          //  2D map inversion
          //
          //  Here we find the point on a 2D reference element that maps to
          //  the point \p physical_point in the domain.  This is a bit tricky
          //  since we do not want to assume that the point \p physical_point
          //  is also in a 2D domain.  In particular, this method might get
          //  called on the face of a 3D element, in which case
          //  \p physical_point actually lives in 3D.
        case 2:
          {
            const Point dxi  = eval.map_deriv(0, p);
            const Point deta = eval.map_deriv(1, p);

            //  Newton's method in this case looks like
            //
            //  {X} - {X_n} = [J]*{dp}
            //
            //  Where {X}, {X_n} are 3x1 vectors, [J] is a 3x2 matrix
            //  d(x,y,z)/d(xi,eta), and we seek {dp}, a 2x1 vector.  Since
            //  the above system is either over-determined or rank-deficient,
            //  we will solve the normal equations for this system
            //
            //  [J]^T ({X} - {X_n}) = [J]^T [J] {dp}
            //
            //  which involves the inversion of the 2x2 matrix
            //  [G] = [J]^T [J]
            const Real
              G11 = dxi*dxi,  G12 = dxi*deta,
              G21 = dxi*deta, G22 = deta*deta;


            const Real det = (G11*G22 - G12*G21);

            if (secure && det == 0 && policy.singular(p))
              {
                converged = false;
                return p;
              }

            const Real inv_det = 1./det;

            const Real
              Ginv11 =  G22*inv_det,
              Ginv12 = -G12*inv_det,

              Ginv21 = -G21*inv_det,
              Ginv22 =  G11*inv_det;


            const Real  dxidelta  = dxi*delta;
            const Real  detadelta = deta*delta;

            dp(0) = (Ginv11*dxidelta + Ginv12*detadelta);
            dp(1) = (Ginv21*dxidelta + Ginv22*detadelta);

            // No master elements have radius > 4, but sometimes we
            // can take a step that big while still converging
            // if (secure)
            // libmesh_assert_less (dp.size(), max_step_length);

            break;
          }



          // ------------------------------------------------------------------
          //  3D map inversion
          //
          //  Here we find the point in a 3D reference element that maps to
          //  the point \p physical_point in a 3D domain. Nothing special
          //  has to happen here, since (unless the map is singular because
          //  you have a BAD element) the map will be invertible and we can
          //  apply Newton's method directly.
        case 3:
          {
            const Point dxi   = eval.map_deriv(0, p);
            const Point deta  = eval.map_deriv(1, p);
            const Point dzeta = eval.map_deriv(2, p);

            //  Newton's method in this case looks like
            //
            //  {X} = {X_n} + [J]*{dp}
            //
            //  Where {X}, {X_n} are 3x1 vectors, [J] is a 3x3 matrix
            //  d(x,y,z)/d(xi,eta,zeta), and we seek {dp}, a 3x1 vector.
            //  Since the above system is nonsingular for invertible maps
            //  we will solve
            //
            //  {dp} = [J]^-1 ({X} - {X_n})
            //
            //  which involves the inversion of the 3x3 matrix [J]
            if (!policy.solve(RealTensorValue(dxi(0), deta(0), dzeta(0),
                                              dxi(1), deta(1), dzeta(1),
                                              dxi(2), deta(2), dzeta(2)),
                              delta, dp, p))
              {
                converged = false;
                return p;
              }

            // No master elements have radius > 4, but sometimes we
            // can take a step that big while still converging
            // if (secure)
            // libmesh_assert_less (dp.size(), max_step_length);

            break;
          }


          //  Some other dimension?
        default:
          libmesh_error_msg("Invalid dim = " << dim);
        } // end switch(Dim), dp now computed



      //  ||P_n+1 - P_n||
      inverse_map_error = dp.norm();

      //  P_n+1 = P_n + dp
      p.add (dp);

      //  Increment the iteration count.
      cnt++;

      //  Watch for divergence of Newton's
      //  method.  Here's how it goes:
      //  (1) For good elements, we expect convergence in 10
      //      iterations, with no too-large steps.
      //      - If called with (secure == true) and we have not yet converged
      //        print out a warning message.
      //      - If called with (secure == true) and we have not converged in
      //        20 iterations abort
      //  (2) This method may be called in cases when the target point is not
      //      inside the element and we have no business expecting convergence.
      //      For these cases if we have not converged in 10 iterations forget
      //      about it.
      if (cnt > max_cnt)
        {
          //  Warn about divergence when secure is true - this
          //  shouldn't happen
          if (secure)
            {
              if (policy.not_converged(cnt, max_cnt, physical_guess, dp, p,
                                       inverse_map_error))
                {
                  converged = false;
                  return p;
                }
            }
          //  Return a far off point when secure is false - this
          //  should only happen when we're trying to map a point
          //  that's outside the element
          else
            {
              fe_inverse_map_far_point(dim, p);

              converged = false;
              return p;
            }
        }
    }
  while (inverse_map_error > tolerance);

  converged = true;
  return p;
}



/**
 * A Lagrange map for fe_inverse_map(), evaluated as FEMap::map() and
 * FEMap::map_deriv() evaluate it: the order \p order shapes on \p type,
 * weighting \p nodes(i), the element's node points, in node order.
 * \p nodes is anything that returns node i's point, so device code can
 * read node coordinates from wherever it keeps them.
 */
template <typename NodeAccessor>
struct LagrangeMapEval
{
  unsigned int dim;
  ElemType type;
  Order order;
  unsigned int n_nodes;
  NodeAccessor nodes;

  LIBMESH_DEVICE_INLINE
  Point map(const Point & p) const
  { return this->sum(invalid_uint, p); }

  LIBMESH_DEVICE_INLINE
  Point map_deriv(const unsigned int j, const Point & p) const
  { return this->sum(j, p); }

  // The node points weighted by the shapes, or by their derivatives in
  // direction j unless j is invalid_uint
  LIBMESH_DEVICE_INLINE
  Point sum(const unsigned int j, const Point & p) const
  {
    Point x;
    for (unsigned int i = 0; i != n_nodes; ++i)
      x.add_scaled(nodes(i), this->shape(i, j, p));
    return x;
  }

  LIBMESH_DEVICE_INLINE
  Real shape(const unsigned int i, const unsigned int j, const Point & p) const
  {
    return (j == invalid_uint) ?
      fe_lagrange_shape<LAGRANGE>(dim, type, order, i, p) :
      fe_lagrange_shape_deriv<LAGRANGE>(dim, type, order, i, j, p);
  }
};



/**
 * A policy for fe_inverse_map() in device code, which can neither throw
 * nor print: it stops, leaving fe_inverse_map() to report Newton hasn't
 * converged, wherever FEMap's policy would throw.  Unlike FEMap, it
 * doesn't return a singular node, such as a pyramid's apex, for a
 * singular 3D Jacobian, but the far away point FEMap returns when not
 * \p secure.
 */
struct NothrowInverseMapPolicy
{
  unsigned int dim;

  LIBMESH_DEVICE_INLINE
  bool singular(const Point &) const
  { return true; }

  LIBMESH_DEVICE_INLINE
  bool solve(const RealTensorValue & J,
             const Point & delta,
             Point & dp,
             Point & p) const
  {
    if (J.solve(delta, dp, std::nothrow_t()))
      return true;

    fe_inverse_map_far_point(dim, p);
    return false;
  }

  LIBMESH_DEVICE_INLINE
  bool not_converged(const unsigned int cnt,
                     const unsigned int max_cnt,
                     const Point &,
                     const Point &,
                     const Point &,
                     const Real) const
  { return cnt > 2*max_cnt; }
};

} // namespace libMesh

#endif // LIBMESH_FE_MAP_POINT_H
