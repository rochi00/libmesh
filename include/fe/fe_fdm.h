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

#ifndef LIBMESH_FE_FDM_H
#define LIBMESH_FE_FDM_H

// Local includes
#include "libmesh/libmesh_device.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_order.h"
#include "libmesh/point.h"

// Central finite differences of shape functions, for the element
// types whose derivatives haven't been worked out analytically.  The
// fe_fdm_deriv() and fe_fdm_second_deriv() overloads declared in
// fe.h take the shape function as a function pointer; these take it
// as any callable, so that device code, which cannot call through a
// host function pointer, takes the same differences.  They take any
// point type, so that the shared shapes, which are templated on their
// coordinate type, difference their own points.

namespace libMesh
{

namespace detail
{

// Plain aggregates rather than std::tuple: std::make_tuple() of a
// point type, whose copy constructor isn't constexpr, can't be called
// from CUDA device code.
template <typename PointType>
struct FDMPoints
{
  PointType pp, pm;
  Real eps;
};

template <typename PointType>
struct FDMSecondPoints
{
  PointType pp, pm;
  Real eps;
  unsigned int deriv_j;
};

template <typename PointType>
LIBMESH_DEVICE_INLINE
FDMPoints<PointType>
fdm_points(const unsigned int j, const PointType & p)
{
  libmesh_assert_less (j, LIBMESH_DIM);

  // cheat by using finite difference approximations:
  const Real eps = 1.e-6;
  PointType pp = p, pm = p;

  switch (j)
    {
      // d()/dxi
    case 0:
      {
        pp(0) += eps;
        pm(0) -= eps;
        break;
      }

      // d()/deta
    case 1:
      {
        pp(1) += eps;
        pm(1) -= eps;
        break;
      }

      // d()/dzeta
    case 2:
      {
        pp(2) += eps;
        pm(2) -= eps;
        break;
      }

    default:
      libmesh_error_msg("Invalid derivative index j = " << j);
    }

  return {pp, pm, eps};
}

template <typename PointType>
LIBMESH_DEVICE_INLINE
FDMSecondPoints<PointType>
fdm_second_points(const unsigned int j, const PointType & p)
{
  // cheat by using finite difference approximations:
  const Real eps = 1.e-5;
  PointType pp = p, pm = p;
  unsigned int deriv_j = 0;

  switch (j)
    {
      //  d^2() / dxi^2
    case 0:
      {
        pp(0) += eps;
        pm(0) -= eps;
        deriv_j = 0;
        break;
      }

      // d^2() / dxi deta
    case 1:
      {
        pp(1) += eps;
        pm(1) -= eps;
        deriv_j = 0;
        break;
      }

      // d^2() / deta^2
    case 2:
      {
        pp(1) += eps;
        pm(1) -= eps;
        deriv_j = 1;
        break;
      }

      // d^2()/dxidzeta
    case 3:
      {
        pp(2) += eps;
        pm(2) -= eps;
        deriv_j = 0;
        break;
      }                  // d^2()/deta^2

      // d^2()/detadzeta
    case 4:
      {
        pp(2) += eps;
        pm(2) -= eps;
        deriv_j = 1;
        break;
      }

      // d^2()/dzeta^2
    case 5:
      {
        pp(2) += eps;
        pm(2) -= eps;
        deriv_j = 2;
        break;
      }

    default:
      libmesh_error_msg("Invalid shape function derivative j = " << j);
    }

  return {pp, pm, eps, deriv_j};
}

/**
 * The derivative in direction \p j of shape function \p i, where \p
 * shape_func(type, order, i, p) evaluates the shape functions.
 */
template <typename PointType, typename ShapeFunc>
LIBMESH_DEVICE_INLINE
auto fe_fdm_deriv(const ElemType type,
                  const Order order,
                  const unsigned int i,
                  const unsigned int j,
                  const PointType & p,
                  const ShapeFunc & shape_func)
{
  auto [pp, pm, eps] = fdm_points(j, p);

  return (shape_func(type, order, i, pp) -
          shape_func(type, order, i, pm))/2./eps;
}

/**
 * The second derivative \p j of shape function \p i, where \p
 * deriv_func(type, order, i, j, p) evaluates the shape functions'
 * first derivatives.
 */
template <typename PointType, typename DerivFunc>
LIBMESH_DEVICE_INLINE
auto fe_fdm_second_deriv(const ElemType type,
                         const Order order,
                         const unsigned int i,
                         const unsigned int j,
                         const PointType & p,
                         const DerivFunc & deriv_func)
{
  auto [pp, pm, eps, deriv_j] = fdm_second_points(j, p);

  return (deriv_func(type, order, i, deriv_j, pp) -
          deriv_func(type, order, i, deriv_j, pm))/2./eps;
}

} // namespace detail

} // namespace libMesh

#endif // LIBMESH_FE_FDM_H
