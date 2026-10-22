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
#include "libmesh/point.h"

// C++ includes
#include <cmath> // std::sqrt

// The Jacobian of the reference-to-physical map at one point, and the
// inverse map's first derivatives, from the map's first derivatives
// there.  Each calls check(det) where FEMap::compute_single_point_map()
// checks for a degenerate map, before dividing by it: FEMap passes its
// check, which throws, and device code one that records the failure.
// FEMap calls these; they are inline here so that device code maps
// points the same way.

namespace libMesh
{

/**
 * One point of the map.  \p det is the quantity checked for a
 * degenerate map: \p jac itself, except for a 2D element in 3D space,
 * where it is det(J^T J) and \p jac is its square root.
 */
struct FEMapPoint
{
  Real det = 0;
  Real jac = 0;

  // The gradients of xi, eta and zeta with respect to (x, y, z)
  Point dxi, deta, dzeta;

#if LIBMESH_DIM == 3
  // (J^T J)^-1 for a 2D element in 3D space, which its inverse map
  // second derivatives also need.
  Real g11inv = 0, g12inv = 0, g21inv = 0, g22inv = 0;
#endif
};



template <typename DegenerateCheck>
LIBMESH_DEVICE_INLINE
void fe_map_1D(const Point & dxyzdxi,
               FEMapPoint & m,
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
  const Real jacm2 = 1./m.jac/m.jac;
  m.dxi(0) = jacm2*dxyzdxi(0);
#if LIBMESH_DIM > 1
  m.dxi(1) = jacm2*dxyzdxi(1);
#endif
#if LIBMESH_DIM > 2
  m.dxi(2) = jacm2*dxyzdxi(2);
#endif
}



template <typename DegenerateCheck>
LIBMESH_DEVICE_INLINE
void fe_map_2D(const Point & dxyzdxi,
               const Point & dxyzdeta,
               FEMapPoint & m,
               const DegenerateCheck & check)
{
  // compute the jacobian once
  const Real dx_dxi = dxyzdxi(0),
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
  const Real inv_jac = 1./m.jac;

  m.dxi(0)  =  dy_deta*inv_jac; //dxi/dx  =  (1/J)*dy/deta
  m.dxi(1)  = -dx_deta*inv_jac; //dxi/dy  = -(1/J)*dx/deta
  m.deta(0) = -dy_dxi* inv_jac; //deta/dx = -(1/J)*dy/dxi
  m.deta(1) =  dx_dxi* inv_jac; //deta/dy =  (1/J)*dx/dxi
#else // LIBMESH_DIM == 3

  const Real dz_dxi = dxyzdxi(2),
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
  const Real g11 = (dx_dxi*dx_dxi +
                    dy_dxi*dy_dxi +
                    dz_dxi*dz_dxi);

  const Real g12 = (dx_dxi*dx_deta +
                    dy_dxi*dy_deta +
                    dz_dxi*dz_deta);

  const Real g21 = g12;

  const Real g22 = (dx_deta*dx_deta +
                    dy_deta*dy_deta +
                    dz_deta*dz_deta);

  m.det = (g11*g22 - g12*g21);

  check(m.det);

  const Real inv_det = 1./m.det;
  m.jac = std::sqrt(m.det);

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



template <typename DegenerateCheck>
LIBMESH_DEVICE_INLINE
void fe_map_3D(const Point & dxyzdxi,
               const Point & dxyzdeta,
               const Point & dxyzdzeta,
               FEMapPoint & m,
               const DegenerateCheck & check)
{
  // compute the jacobian
  const Real
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
  const Real inv_jac  = 1./m.jac;

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

} // namespace libMesh

#endif // LIBMESH_FE_MAP_POINT_H
