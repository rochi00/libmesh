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
// Likewise a side's Jacobian, normal, tangents and curvature, from the
// side's map derivatives there.  FEMap calls these; they are inline
// here so that device code maps points the same way, and templated on
// the coordinate type, as the shared shape functions are, so that
// automatic differentiation does.

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

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  // The second derivatives of xi, eta and zeta with respect to (x, y,
  // z), in the order xx, xy, xz, yy, yz, zz, which
  // fe_map_second_derivs_*() set from the map's second derivatives.
  T d2xi[6] = {}, d2eta[6] = {}, d2zeta[6] = {};
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



#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

// The inverse map's second derivatives at a point m from fe_map_*D(),
// from the map's second derivatives there, d2xyz[j], numbered as the
// shapes' are: xi-xi, xi-eta, eta-eta, xi-zeta, eta-zeta, zeta-zeta.

/**
 * For an element whose dimension is LIBMESH_DIM, such as a 3D
 * element, whose Jacobian is square.
 */
template <typename T>
LIBMESH_DEVICE_INLINE
void fe_map_second_derivs_square(const VectorValue<T> * d2xyz,
                                 FEMapPoint<T> & m)
{
  // Construct J^{-1}, A, and B matrices (see JWP's notes for details)
  // for cases in which the element dimension matches LIBMESH_DIM.
#if LIBMESH_DIM==1
  const TensorValue<T>
    Jinv(m.dxi(0), 0., 0.,
         0.,       0., 0.,
         0.,       0., 0.),

    A(d2xyz[0](0), 0., 0.,
      0.,          0., 0.,
      0.,          0., 0.),

    B(0., 0., 0.,
      0., 0., 0.,
      0., 0., 0.);

#elif LIBMESH_DIM==2
  const TensorValue<T>
    Jinv(m.dxi(0),  m.dxi(1),  0.,
         m.deta(0), m.deta(1), 0.,
         0.,        0.,        0.),

    A(d2xyz[0](0), d2xyz[2](0), 0.,
      d2xyz[0](1), d2xyz[2](1), 0.,
      0.,          0.,          0.),

    B(d2xyz[1](0), 0., 0.,
      d2xyz[1](1), 0., 0.,
      0.,          0., 0.);

#elif LIBMESH_DIM==3
  const TensorValue<T>
    Jinv(m.dxi(0),   m.dxi(1),   m.dxi(2),
         m.deta(0),  m.deta(1),  m.deta(2),
         m.dzeta(0), m.dzeta(1), m.dzeta(2)),

    A(d2xyz[0](0), d2xyz[2](0), d2xyz[5](0),
      d2xyz[0](1), d2xyz[2](1), d2xyz[5](1),
      d2xyz[0](2), d2xyz[2](2), d2xyz[5](2)),

    B(d2xyz[1](0), d2xyz[3](0), d2xyz[4](0),
      d2xyz[1](1), d2xyz[3](1), d2xyz[4](1),
      d2xyz[1](2), d2xyz[3](2), d2xyz[4](2));
#endif

  // For (s,t) in {(x,x), (x,y), (x,z), (y,y), (y,z), (z,z)}, compute the
  // vector of inverse map second derivatives [xi_{s t}, eta_{s t}, zeta_{s t}],
  // for the directions s, t the space has
  unsigned int ctr=0;
  for (unsigned int s=0; s<3; ++s)
    for (unsigned int t=s; t<3; ++t, ++ctr)
      if (t < LIBMESH_DIM)
        {
          const VectorValue<T>
            v1(m.dxi(s)*m.dxi(t),
               m.deta(s)*m.deta(t),
               m.dzeta(s)*m.dzeta(t)),

            v2(m.dxi(s)*m.deta(t) + m.deta(s)*m.dxi(t),
               m.dxi(s)*m.dzeta(t) + m.dzeta(s)*m.dxi(t),
               m.deta(s)*m.dzeta(t) + m.dzeta(s)*m.deta(t));

          // Compute the inverse map second derivatives
          const VectorValue<T> v3 = -Jinv*(A*v1 + B*v2);

          m.d2xi[ctr] = v3(0);

          if (LIBMESH_DIM > 1)
            m.d2eta[ctr] = v3(1);

          if (LIBMESH_DIM > 2)
            m.d2zeta[ctr] = v3(2);
        }
}



/**
 * For a 1D element.  In 2D or 3D space the caller checks first that
 * dxyzdxi isn't zero, which this divides by the square of.
 */
template <typename T>
LIBMESH_DEVICE_INLINE
void fe_map_second_derivs_1D(const TypeVector<T> & dxyzdxi,
                             const VectorValue<T> * d2xyz,
                             FEMapPoint<T> & m)
{
#if LIBMESH_DIM == 1
  libmesh_ignore(dxyzdxi);
  fe_map_second_derivs_square(d2xyz, m);
#else
  // Compute inverse map second derivatives for 1D-element-living-in-2D
  // or 3D case.  See JWP notes for details

  // numer = x_xi*x_{xi xi} + y_xi*y_{xi xi} + z_xi*z_{xi xi}
  const T numer = dxyzdxi * d2xyz[0];

  // denom = (x_xi)^2 + (y_xi)^2 + (z_xi)^2 must be >= 0.0
  const T denom = dxyzdxi * dxyzdxi;

  // xi_{s t} for the directions s, t the space has
  unsigned int ctr=0;
  for (unsigned int s=0; s<3; ++s)
    for (unsigned int t=s; t<3; ++t, ++ctr)
      if (t < LIBMESH_DIM)
        m.d2xi[ctr] = -numer * m.dxi(s)*m.dxi(t) / denom;
#endif
}



/**
 * For a 2D element, whose Jacobian in 3D space isn't square.
 */
template <typename T>
LIBMESH_DEVICE_INLINE
void fe_map_second_derivs_2D(const TypeVector<T> & dxyzdxi,
                             const TypeVector<T> & dxyzdeta,
                             const VectorValue<T> * d2xyz,
                             FEMapPoint<T> & m)
{
#if LIBMESH_DIM < 3
  libmesh_ignore(dxyzdxi, dxyzdeta);
  fe_map_second_derivs_square(d2xyz, m);
#else
  // Compute inverse map second derivative values for
  // 2D-element-living-in-3D case.  We pursue a least-squares
  // solution approach for this "non-square" case, see JWP notes
  // for details: with A = [x_{xi xi} x_{eta eta}],
  //
  // [xi_{s t}, eta_{s t}] = -(J^T J)^{-1} J^T (A [xi_s*xi_t, eta_s*eta_t]
  //                                            + alpha x_{xi eta})
  //
  // where alpha = xi_s*eta_t + eta_s*xi_t, and (J^T J)^{-1} is m's g*inv.
  unsigned int ctr=0;
  for (unsigned int s=0; s<3; ++s)
    for (unsigned int t=s; t<3; ++t, ++ctr)
      {
        const T alpha = m.dxi(s)*m.deta(t) + m.deta(s)*m.dxi(t);
        const T xi_st = m.dxi(s)*m.dxi(t), eta_st = m.deta(s)*m.deta(t);

        VectorValue<T> tmp2;
        for (unsigned int i=0; i<3; ++i)
          tmp2(i) = d2xyz[0](i)*xi_st + d2xyz[2](i)*eta_st + alpha*d2xyz[1](i);

        // J^T tmp2
        const T tmp3_0 = dxyzdxi * tmp2, tmp3_1 = dxyzdeta * tmp2;

        // Don't forget to multiply by -1!
        m.d2xi[ctr]  = -(m.g11inv*tmp3_0 + m.g12inv*tmp3_1);
        m.d2eta[ctr] = -(m.g21inv*tmp3_0 + m.g22inv*tmp3_1);
      }
#endif
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES



// The derivatives of a shape with respect to (x, y, z) at a point m
// from fe_map_*D(), from its derivatives dphi[j] there in reference
// direction j < dim, by the chain rule.  S is the shape's type, which
// H1 vector shapes make a vector.
template <typename S, typename T>
LIBMESH_DEVICE_INLINE
void fe_physical_derivs(const unsigned int dim,
                        const S * dphi,
                        const FEMapPoint<T> & m,
                        S * dphidxyz)
{
  for (unsigned int k=0; k != LIBMESH_DIM; ++k)
    if (dim == 1)
      dphidxyz[k] = dphi[0]*m.dxi(k);
    else if (dim == 2)
      dphidxyz[k] = dphi[0]*m.dxi(k) + dphi[1]*m.deta(k);
    else
      dphidxyz[k] = dphi[0]*m.dxi(k) + dphi[1]*m.deta(k) + dphi[2]*m.dzeta(k);
}



#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

// The second derivatives of a shape with respect to (x, y, z), in the
// order xx, xy, xz, yy, yz, zz, at a point m from fe_map_*D() and
// fe_map_second_derivs_*(), from the shape's derivatives there with
// respect to the reference coordinates: dphi[j] in direction j, and
// d2phi[j] numbered as fe_shape_second_deriv() numbers them.  S is
// the shape's type, which H1 vector shapes make a vector.

template <typename S, typename T>
LIBMESH_DEVICE_INLINE
void fe_physical_second_derivs_1D(const S * dphi,
                                  const S * d2phi,
                                  const FEMapPoint<T> & m,
                                  S * d2phidxyz2)
{
  const S & dphidxi = dphi[0];
  const S & d2phidxi2 = d2phi[0];
  const T & dxidx = m.dxi(0);
#if LIBMESH_DIM>1
  const T & dxidy = m.dxi(1);
#endif
#if LIBMESH_DIM>2
  const T & dxidz = m.dxi(2);
#endif

  // phi_{x x}
  d2phidxyz2[0] =
    d2phidxi2*dxidx*dxidx + // (xi_x)^2 * phi_{xi xi}
    m.d2xi[0]*dphidxi;      // xi_{x x} * phi_{xi}

#if LIBMESH_DIM>1
  // phi_{x y}
  d2phidxyz2[1] =
    d2phidxi2*dxidx*dxidy + // xi_x * xi_y * phi_{xi xi}
    m.d2xi[1]*dphidxi;      // xi_{x y} * phi_{xi}
#endif

#if LIBMESH_DIM>2
  // phi_{x z}
  d2phidxyz2[2] =
    d2phidxi2*dxidx*dxidz + // xi_x * xi_z * phi_{xi xi}
    m.d2xi[2]*dphidxi;      // xi_{x z} * phi_{xi}
#endif


#if LIBMESH_DIM>1
  // phi_{y y}
  d2phidxyz2[3] =
    d2phidxi2*dxidy*dxidy + // (xi_y)^2 * phi_{xi xi}
    m.d2xi[3]*dphidxi;      // xi_{y y} * phi_{xi}
#endif

#if LIBMESH_DIM>2
  // phi_{y z}
  d2phidxyz2[4] =
    d2phidxi2*dxidy*dxidz + // xi_y * xi_z * phi_{xi xi}
    m.d2xi[4]*dphidxi;      // xi_{y z} * phi_{xi}

  // phi_{z z}
  d2phidxyz2[5] =
    d2phidxi2*dxidz*dxidz + // (xi_z)^2 * phi_{xi xi}
    m.d2xi[5]*dphidxi;      // xi_{z z} * phi_{xi}
#endif
}



template <typename S, typename T>
LIBMESH_DEVICE_INLINE
void fe_physical_second_derivs_2D(const S * dphi,
                                  const S * d2phi,
                                  const FEMapPoint<T> & m,
                                  S * d2phidxyz2)
{
  const S & dphidxi = dphi[0], & dphideta = dphi[1];
  const S & d2phidxi2 = d2phi[0], & d2phidxideta = d2phi[1], & d2phideta2 = d2phi[2];
  const T & dxidx = m.dxi(0), & dxidy = m.dxi(1);
  const T & detadx = m.deta(0), & detady = m.deta(1);
#if LIBMESH_DIM > 2
  const T & dxidz = m.dxi(2), & detadz = m.deta(2);
#endif

  // phi_{x x}
  d2phidxyz2[0] =
    d2phidxi2*dxidx*dxidx +       // (xi_x)^2 * phi_{xi xi}
    d2phideta2*detadx*detadx +    // (eta_x)^2 * phi_{eta eta}
    2*d2phidxideta*dxidx*detadx + // 2 * xi_x * eta_x * phi_{xi eta}
    m.d2xi[0]*dphidxi +           // xi_{x x} * phi_{xi}
    m.d2eta[0]*dphideta;          // eta_{x x} * phi_{eta}

  // phi_{x y}
  d2phidxyz2[1] =
    d2phidxi2*dxidx*dxidy +                      // xi_x * xi_y * phi_{xi xi}
    d2phideta2*detadx*detady +                   // eta_x * eta_y * phi_{eta eta}
    d2phidxideta*(dxidx*detady + detadx*dxidy) + // (xi_x*eta_y + eta_x*xi_y) * phi_{xi eta}
    m.d2xi[1]*dphidxi +                          // xi_{x y} * phi_{xi}
    m.d2eta[1]*dphideta;                         // eta_{x y} * phi_{eta}

#if LIBMESH_DIM > 2
  // phi_{x z}
  d2phidxyz2[2] =
    d2phidxi2*dxidx*dxidz +                      // xi_x * xi_z * phi_{xi xi}
    d2phideta2*detadx*detadz +                   // eta_x * eta_z * phi_{eta eta}
    d2phidxideta*(dxidx*detadz + detadx*dxidz) + // (xi_x*eta_z + eta_x*xi_z) * phi_{xi eta}
    m.d2xi[2]*dphidxi +                          // xi_{x z} * phi_{xi}
    m.d2eta[2]*dphideta;                         // eta_{x z} * phi_{eta}
#endif

  // phi_{y y}
  d2phidxyz2[3] =
    d2phidxi2*dxidy*dxidy +       // (xi_y)^2 * phi_{xi xi}
    d2phideta2*detady*detady +    // (eta_y)^2 * phi_{eta eta}
    2*d2phidxideta*dxidy*detady + // 2 * xi_y * eta_y * phi_{xi eta}
    m.d2xi[3]*dphidxi +           // xi_{y y} * phi_{xi}
    m.d2eta[3]*dphideta;          // eta_{y y} * phi_{eta}

#if LIBMESH_DIM > 2
  // phi_{y z}
  d2phidxyz2[4] =
    d2phidxi2*dxidy*dxidz +                      // xi_y * xi_z * phi_{xi xi}
    d2phideta2*detady*detadz +                   // eta_y * eta_z * phi_{eta eta}
    d2phidxideta*(dxidy*detadz + detady*dxidz) + // (xi_y*eta_z + eta_y*xi_z) * phi_{xi eta}
    m.d2xi[4]*dphidxi +                          // xi_{y z} * phi_{xi}
    m.d2eta[4]*dphideta;                         // eta_{y z} * phi_{eta}

  // phi_{z z}
  d2phidxyz2[5] =
    d2phidxi2*dxidz*dxidz +       // (xi_z)^2 * phi_{xi xi}
    d2phideta2*detadz*detadz +    // (eta_z)^2 * phi_{eta eta}
    2*d2phidxideta*dxidz*detadz + // 2 * xi_z * eta_z * phi_{xi eta}
    m.d2xi[5]*dphidxi +           // xi_{z z} * phi_{xi}
    m.d2eta[5]*dphideta;          // eta_{z z} * phi_{eta}
#endif
}



template <typename S, typename T>
LIBMESH_DEVICE_INLINE
void fe_physical_second_derivs_3D(const S * dphi,
                                  const S * d2phi,
                                  const FEMapPoint<T> & m,
                                  S * d2phidxyz2)
{
  const S & dphidxi = dphi[0], & dphideta = dphi[1], & dphidzeta = dphi[2];
  const S & d2phidxi2 = d2phi[0], & d2phidxideta = d2phi[1], & d2phideta2 = d2phi[2],
    & d2phidxidzeta = d2phi[3], & d2phidetadzeta = d2phi[4], & d2phidzeta2 = d2phi[5];
  const T & dxidx = m.dxi(0), & dxidy = m.dxi(1), & dxidz = m.dxi(2);
  const T & detadx = m.deta(0), & detady = m.deta(1), & detadz = m.deta(2);
  const T & dzetadx = m.dzeta(0), & dzetady = m.dzeta(1), & dzetadz = m.dzeta(2);

  // phi_{x x}
  d2phidxyz2[0] =
    d2phidxi2*dxidx*dxidx +           // (xi_x)^2 * phi_{xi xi}
    d2phideta2*detadx*detadx +        // (eta_x)^2 * phi_{eta eta}
    d2phidzeta2*dzetadx*dzetadx +     // (zeta_x)^2 * phi_{zeta zeta}
    2*d2phidxideta*dxidx*detadx +     // 2 * xi_x * eta_x * phi_{xi eta}
    2*d2phidxidzeta*dxidx*dzetadx +   // 2 * xi_x * zeta_x * phi_{xi zeta}
    2*d2phidetadzeta*detadx*dzetadx + // 2 * eta_x * zeta_x * phi_{eta zeta}
    m.d2xi[0]*dphidxi +               // xi_{x x} * phi_{xi}
    m.d2eta[0]*dphideta +             // eta_{x x} * phi_{eta}
    m.d2zeta[0]*dphidzeta;            // zeta_{x x} * phi_{zeta}

  // phi_{x y}
  d2phidxyz2[1] =
    d2phidxi2*dxidx*dxidy +                            // xi_x * xi_y * phi_{xi xi}
    d2phideta2*detadx*detady +                         // eta_x * eta_y * phi_{eta eta}
    d2phidzeta2*dzetadx*dzetady +                      // zeta_x * zeta_y * phi_{zeta zeta}
    d2phidxideta*(dxidx*detady + detadx*dxidy) +       // (xi_x*eta_y + eta_x*xi_y) * phi_{xi eta}
    d2phidxidzeta*(dxidx*dzetady + dzetadx*dxidy) +    // (zeta_x*xi_y + xi_x*zeta_y) * phi_{xi zeta}
    d2phidetadzeta*(detadx*dzetady + dzetadx*detady) + // (zeta_x*eta_y + eta_x*zeta_y) * phi_{eta zeta}
    m.d2xi[1]*dphidxi +                                // xi_{x y} * phi_{xi}
    m.d2eta[1]*dphideta +                              // eta_{x y} * phi_{eta}
    m.d2zeta[1]*dphidzeta;                             // zeta_{x y} * phi_{zeta}

  // phi_{x z}
  d2phidxyz2[2] =
    d2phidxi2*dxidx*dxidz +                            // xi_x * xi_z * phi_{xi xi}
    d2phideta2*detadx*detadz +                         // eta_x * eta_z * phi_{eta eta}
    d2phidzeta2*dzetadx*dzetadz +                      // zeta_x * zeta_z * phi_{zeta zeta}
    d2phidxideta*(dxidx*detadz + detadx*dxidz) +       // (xi_x*eta_z + eta_x*xi_z) * phi_{xi eta}
    d2phidxidzeta*(dxidx*dzetadz + dzetadx*dxidz) +    // (zeta_x*xi_z + xi_x*zeta_z) * phi_{xi zeta}
    d2phidetadzeta*(detadx*dzetadz + dzetadx*detadz) + // (zeta_x*eta_z + eta_x*zeta_z) * phi_{eta zeta}
    m.d2xi[2]*dphidxi +                                // xi_{x z} * phi_{xi}
    m.d2eta[2]*dphideta +                              // eta_{x z} * phi_{eta}
    m.d2zeta[2]*dphidzeta;                             // zeta_{x z} * phi_{zeta}

  // phi_{y y}
  d2phidxyz2[3] =
    d2phidxi2*dxidy*dxidy +           // (xi_y)^2 * phi_{xi xi}
    d2phideta2*detady*detady +        // (eta_y)^2 * phi_{eta eta}
    d2phidzeta2*dzetady*dzetady +     // (zeta_y)^2 * phi_{zeta zeta}
    2*d2phidxideta*dxidy*detady +     // 2 * xi_y * eta_y * phi_{xi eta}
    2*d2phidxidzeta*dxidy*dzetady +   // 2 * xi_y * zeta_y * phi_{xi zeta}
    2*d2phidetadzeta*detady*dzetady + // 2 * eta_y * zeta_y * phi_{eta zeta}
    m.d2xi[3]*dphidxi +               // xi_{y y} * phi_{xi}
    m.d2eta[3]*dphideta +             // eta_{y y} * phi_{eta}
    m.d2zeta[3]*dphidzeta;            // zeta_{y y} * phi_{zeta}

  // phi_{y z}
  d2phidxyz2[4] =
    d2phidxi2*dxidy*dxidz +                            // xi_y * xi_z * phi_{xi xi}
    d2phideta2*detady*detadz +                         // eta_y * eta_z * phi_{eta eta}
    d2phidzeta2*dzetady*dzetadz +                      // zeta_y * zeta_z * phi_{zeta zeta}
    d2phidxideta*(dxidy*detadz + detady*dxidz) +       // (xi_y*eta_z + eta_y*xi_z) * phi_{xi eta}
    d2phidxidzeta*(dxidy*dzetadz + dzetady*dxidz) +    // (zeta_y*xi_z + xi_y*zeta_z) * phi_{xi zeta}
    d2phidetadzeta*(detady*dzetadz + dzetady*detadz) + // (zeta_y*eta_z + eta_y*zeta_z) * phi_{eta zeta}
    m.d2xi[4]*dphidxi +                                // xi_{y z} * phi_{xi}
    m.d2eta[4]*dphideta +                              // eta_{y z} * phi_{eta}
    m.d2zeta[4]*dphidzeta;                             // zeta_{y z} * phi_{zeta}

  // phi_{z z}
  d2phidxyz2[5] =
    d2phidxi2*dxidz*dxidz +           // (xi_z)^2 * phi_{xi xi}
    d2phideta2*detadz*detadz +        // (eta_z)^2 * phi_{eta eta}
    d2phidzeta2*dzetadz*dzetadz +     // (zeta_z)^2 * phi_{zeta zeta}
    2*d2phidxideta*dxidz*detadz +     // 2 * xi_z * eta_z * phi_{xi eta}
    2*d2phidxidzeta*dxidz*dzetadz +   // 2 * xi_z * zeta_z * phi_{xi zeta}
    2*d2phidetadzeta*detadz*dzetadz + // 2 * eta_z * zeta_z * phi_{eta zeta}
    m.d2xi[5]*dphidxi +               // xi_{z z} * phi_{xi}
    m.d2eta[5]*dphideta +             // eta_{z z} * phi_{eta}
    m.d2zeta[5]*dphidzeta;            // zeta_{z z} * phi_{zeta}
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES



/**
 * One point of a side's map, which FEMap::compute_face_map() computes
 * from the side's own map: \p jac, which scales the side's quadrature
 * weights, and the outward unit normal and unit tangents.  A side of a
 * 2D element has one tangent, plus the element's unit normal when it
 * lives in 3D space; a side of a 3D element has two.
 */
template <typename T = Real>
struct FEFaceMapPoint
{
  T jac = 0;
  VectorValue<T> normal;
  VectorValue<T> tangents[2];
};



/**
 * \p dxyzdxi is the side's map derivative.  In 3D space the side alone
 * doesn't say which way is out, so parent_derivs(dx_dxi, dx_deta) sets
 * the element's own map derivatives at the point: FEMap inverse maps
 * the point onto the element, and device code can tabulate them.
 */
template <typename T, typename ParentDerivs, typename DegenerateCheck>
LIBMESH_DEVICE_INLINE
void fe_face_map_2D(const TypeVector<T> & dxyzdxi,
                    const ParentDerivs & parent_derivs,
                    FEFaceMapPoint<T> & f,
                    const DegenerateCheck & check)
{
  // The first tangent comes from just the edge's Jacobian
  f.tangents[0] = dxyzdxi.unit();

#if LIBMESH_DIM < 3
  libmesh_ignore(parent_derivs);

  // For a 2D element living in 2D, the normal is given directly
  // from the entries in the edge Jacobian.
  f.normal = (VectorValue<T>(dxyzdxi(1), -dxyzdxi(0), T(0))).unit();

#else
  // For a 2D element living in 3D, there is a second tangent.
  // For the second tangent, we need to refer to the full
  // element's (not just the edge's) Jacobian.
  VectorValue<T> dx_dxi, dx_deta;
  parent_derivs(dx_dxi, dx_deta);

  // The second tangent vector is formed by crossing these vectors.
  f.tangents[1] = dx_dxi.cross(dx_deta).unit();

  // Finally, the normal in this case is given by crossing these
  // two tangents.
  f.normal = f.tangents[0].cross(f.tangents[1]).unit();
#endif

  f.jac = dxyzdxi.norm();

  check(f.jac);
}



template <typename T, typename DegenerateCheck>
LIBMESH_DEVICE_INLINE
void fe_face_map_3D(const TypeVector<T> & dxyzdxi,
                    const TypeVector<T> & dxyzdeta,
                    FEFaceMapPoint<T> & f,
                    const DegenerateCheck & check)
{
  const VectorValue<T> n = dxyzdxi.cross(dxyzdeta);
  f.normal      = n.unit();
  f.tangents[0] = dxyzdxi.unit();
  f.tangents[1] = n.cross(dxyzdxi).unit();

  // The side is a 2D element in 3D space, whose Jacobian, and the
  // check on it, fe_map_2D() computes; the compiler drops the inverse
  // map it computes too, which we don't use.
  FEMapPoint<T> m;
  fe_map_2D(dxyzdxi, dxyzdeta, m, check);
  f.jac = m.jac;
}



#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

template <typename T>
LIBMESH_DEVICE_INLINE
T fe_face_curvature_2D(const TypeVector<T> & dxyzdxi,
                       const TypeVector<T> & d2xyzdxi2,
                       const TypeVector<T> & normal)
{
  // The curvature is computed via the familiar Frenet formula:
  // curvature = [d^2(x) / d (xi)^2] dot [normal]
  // For a reference, see:
  // F.S. Merritt, Mathematics Manual, 1962, McGraw-Hill, p. 310
  //
  // Note: The sign convention here is different from the
  // 3D case.  Concave-upward curves (smiles) have a positive
  // curvature.  Concave-downward curves (frowns) have a
  // negative curvature.  Be sure to take that into account!
  const T numerator   = d2xyzdxi2 * normal;
  const T denominator = dxyzdxi.norm_sq();
  libmesh_assert_not_equal_to (denominator, 0);
  return numerator / denominator;
}



template <typename T>
LIBMESH_DEVICE_INLINE
T fe_face_curvature_3D(const TypeVector<T> & dxyzdxi,
                       const TypeVector<T> & dxyzdeta,
                       const TypeVector<T> & d2xyzdxi2,
                       const TypeVector<T> & d2xyzdxideta,
                       const TypeVector<T> & d2xyzdeta2,
                       const TypeVector<T> & normal)
{
  // Compute curvature using the typical nomenclature
  // of the first and second fundamental forms.
  // For reference, see:
  // 1) http://mathworld.wolfram.com/MeanCurvature.html
  //    (note -- they are using inward normal)
  // 2) F.S. Merritt, Mathematics Manual, 1962, McGraw-Hill
  const T L  = -d2xyzdxi2    * normal;
  const T M  = -d2xyzdxideta * normal;
  const T N  = -d2xyzdeta2   * normal;
  const T E  =  dxyzdxi.norm_sq();
  const T F  =  dxyzdxi      * dxyzdeta;
  const T G  =  dxyzdeta.norm_sq();

  const T numerator   = E*N -2.*F*M + G*L;
  const T denominator = E*G - F*F;
  libmesh_assert_not_equal_to (denominator, 0.);
  return 0.5*numerator/denominator;
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES



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

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  // The map's second derivative j, numbered as the shapes' are: xi-xi,
  // xi-eta, eta-eta, xi-zeta, eta-zeta, zeta-zeta.  Summed in node
  // order, as FEMap sums its d2xyz, so the two agree bit for bit.
  LIBMESH_DEVICE_INLINE
  Point map_second_deriv(const unsigned int j, const Point & p) const
  {
    Point x;
    for (unsigned int i = 0; i != n_nodes; ++i)
      x.add_scaled(nodes(i), fe_lagrange_shape_second_deriv<LAGRANGE>(dim, type, order, i, j, p));
    return x;
  }
#endif
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
