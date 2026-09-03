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

#ifndef LIBMESH_FE_TENSOR_PRODUCT_LAGRANGE_H
#define LIBMESH_FE_TENSOR_PRODUCT_LAGRANGE_H

// Lagrange shape functions and derivatives for the tensor-product
// topologies (quads and hexes), including their reduced-node
// "serendipity" members QUAD8 and HEX20, as device-inlinable functions
// shared between the host FE classes and Kokkos kernels.

#include "libmesh/libmesh_device.h"
#include "libmesh/fe_lagrange_shape_1D.h"

namespace libMesh
{
namespace detail
{

LIBMESH_DEVICE_INLINE
unsigned int quad4_i0(const unsigned int i)
{
  libmesh_assert_less(i, 4);
  return (i == 0 || i == 3) ? 0u : 1u;
}

LIBMESH_DEVICE_INLINE
unsigned int quad4_i1(const unsigned int i)
{
  libmesh_assert_less(i, 4);
  return i < 2 ? 0u : 1u;
}

LIBMESH_DEVICE_INLINE
unsigned int quad9_i0(const unsigned int i)
{
  libmesh_assert_less(i, 9);

  switch (i)
    {
    case 0:
    case 3:
    case 7:
      return 0;
    case 1:
    case 2:
    case 5:
      return 1;
    default:
      return 2;
    }
}

LIBMESH_DEVICE_INLINE
unsigned int quad9_i1(const unsigned int i)
{
  libmesh_assert_less(i, 9);

  switch (i)
    {
    case 0:
    case 1:
    case 4:
      return 0;
    case 2:
    case 3:
    case 6:
      return 1;
    default:
      return 2;
    }
}

LIBMESH_DEVICE_INLINE
unsigned int hex8_i0(const unsigned int i)
{
  libmesh_assert_less(i, 8);
  return (i == 0 || i == 3 || i == 4 || i == 7) ? 0u : 1u;
}

LIBMESH_DEVICE_INLINE
unsigned int hex8_i1(const unsigned int i)
{
  libmesh_assert_less(i, 8);
  return (i == 0 || i == 1 || i == 4 || i == 5) ? 0u : 1u;
}

LIBMESH_DEVICE_INLINE
unsigned int hex8_i2(const unsigned int i)
{
  libmesh_assert_less(i, 8);
  return i < 4 ? 0u : 1u;
}

LIBMESH_DEVICE_INLINE
unsigned int hex27_i0(const unsigned int i)
{
  libmesh_assert_less(i, 27);

  switch (i)
    {
    case 0:
    case 3:
    case 4:
    case 7:
    case 11:
    case 12:
    case 15:
    case 19:
    case 24:
      return 0;
    case 1:
    case 2:
    case 5:
    case 6:
    case 9:
    case 13:
    case 14:
    case 17:
    case 22:
      return 1;
    default:
      return 2;
    }
}

LIBMESH_DEVICE_INLINE
unsigned int hex27_i1(const unsigned int i)
{
  libmesh_assert_less(i, 27);

  switch (i)
    {
    case 0:
    case 1:
    case 4:
    case 5:
    case 8:
    case 12:
    case 13:
    case 16:
    case 21:
      return 0;
    case 2:
    case 3:
    case 6:
    case 7:
    case 10:
    case 14:
    case 15:
    case 18:
    case 23:
      return 1;
    default:
      return 2;
    }
}

LIBMESH_DEVICE_INLINE
unsigned int hex27_i2(const unsigned int i)
{
  libmesh_assert_less(i, 27);

  switch (i)
    {
    case 0:
    case 1:
    case 2:
    case 3:
    case 8:
    case 9:
    case 10:
    case 11:
    case 20:
      return 0;
    case 4:
    case 5:
    case 6:
    case 7:
    case 16:
    case 17:
    case 18:
    case 19:
    case 25:
      return 1;
    default:
      return 2;
    }
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_quad4_shape(const unsigned int i,
                             const Real xi,
                             const Real eta)
{
  libmesh_assert_less(i, 4);

  return fe_lagrange_1D_linear_shape(quad4_i0(i), xi) *
         fe_lagrange_1D_linear_shape(quad4_i1(i), eta);
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_quad4_shape_deriv(const unsigned int i,
                                   const unsigned int j,
                                   const Real xi,
                                   const Real eta)
{
  libmesh_assert_less(i, 4);
  libmesh_assert_less(j, 2);

  switch (j)
    {
    case 0:
      return fe_lagrange_1D_linear_shape_deriv(quad4_i0(i), 0, xi) *
             fe_lagrange_1D_linear_shape(quad4_i1(i), eta);

    default:
      return fe_lagrange_1D_linear_shape(quad4_i0(i), xi) *
             fe_lagrange_1D_linear_shape_deriv(quad4_i1(i), 0, eta);
    }
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_quad9_shape(const unsigned int i,
                             const Real xi,
                             const Real eta)
{
  libmesh_assert_less(i, 9);

  return fe_lagrange_1D_quadratic_shape(quad9_i0(i), xi) *
         fe_lagrange_1D_quadratic_shape(quad9_i1(i), eta);
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_quad9_shape_deriv(const unsigned int i,
                                   const unsigned int j,
                                   const Real xi,
                                   const Real eta)
{
  libmesh_assert_less(i, 9);
  libmesh_assert_less(j, 2);

  switch (j)
    {
    case 0:
      return fe_lagrange_1D_quadratic_shape_deriv(quad9_i0(i), 0, xi) *
             fe_lagrange_1D_quadratic_shape(quad9_i1(i), eta);

    default:
      return fe_lagrange_1D_quadratic_shape(quad9_i0(i), xi) *
             fe_lagrange_1D_quadratic_shape_deriv(quad9_i1(i), 0, eta);
    }
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_hex8_shape(const unsigned int i,
                            const Real xi,
                            const Real eta,
                            const Real zeta)
{
  libmesh_assert_less(i, 8);

  return fe_lagrange_1D_linear_shape(hex8_i0(i), xi) *
         fe_lagrange_1D_linear_shape(hex8_i1(i), eta) *
         fe_lagrange_1D_linear_shape(hex8_i2(i), zeta);
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_hex8_shape_deriv(const unsigned int i,
                                  const unsigned int j,
                                  const Real xi,
                                  const Real eta,
                                  const Real zeta)
{
  libmesh_assert_less(i, 8);
  libmesh_assert_less(j, 3);

  switch (j)
    {
    case 0:
      return fe_lagrange_1D_linear_shape_deriv(hex8_i0(i), 0, xi) *
             fe_lagrange_1D_linear_shape(hex8_i1(i), eta) *
             fe_lagrange_1D_linear_shape(hex8_i2(i), zeta);

    case 1:
      return fe_lagrange_1D_linear_shape(hex8_i0(i), xi) *
             fe_lagrange_1D_linear_shape_deriv(hex8_i1(i), 0, eta) *
             fe_lagrange_1D_linear_shape(hex8_i2(i), zeta);

    default:
      return fe_lagrange_1D_linear_shape(hex8_i0(i), xi) *
             fe_lagrange_1D_linear_shape(hex8_i1(i), eta) *
             fe_lagrange_1D_linear_shape_deriv(hex8_i2(i), 0, zeta);
    }
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_hex27_shape(const unsigned int i,
                             const Real xi,
                             const Real eta,
                             const Real zeta)
{
  libmesh_assert_less(i, 27);

  return fe_lagrange_1D_quadratic_shape(hex27_i0(i), xi) *
         fe_lagrange_1D_quadratic_shape(hex27_i1(i), eta) *
         fe_lagrange_1D_quadratic_shape(hex27_i2(i), zeta);
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_hex27_shape_deriv(const unsigned int i,
                                   const unsigned int j,
                                   const Real xi,
                                   const Real eta,
                                   const Real zeta)
{
  libmesh_assert_less(i, 27);
  libmesh_assert_less(j, 3);

  switch (j)
    {
    case 0:
      return fe_lagrange_1D_quadratic_shape_deriv(hex27_i0(i), 0, xi) *
             fe_lagrange_1D_quadratic_shape(hex27_i1(i), eta) *
             fe_lagrange_1D_quadratic_shape(hex27_i2(i), zeta);

    case 1:
      return fe_lagrange_1D_quadratic_shape(hex27_i0(i), xi) *
             fe_lagrange_1D_quadratic_shape_deriv(hex27_i1(i), 0, eta) *
             fe_lagrange_1D_quadratic_shape(hex27_i2(i), zeta);

    default:
      return fe_lagrange_1D_quadratic_shape(hex27_i0(i), xi) *
             fe_lagrange_1D_quadratic_shape(hex27_i1(i), eta) *
             fe_lagrange_1D_quadratic_shape_deriv(hex27_i2(i), 0, zeta);
    }
}

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

LIBMESH_DEVICE_INLINE
Real fe_lagrange_quad4_shape_second_deriv(const unsigned int i,
                                          const unsigned int j,
                                          const Real xi,
                                          const Real eta)
{
  libmesh_assert_less(i, 4);
  libmesh_assert_less(j, 3);

  switch (j)
    {
    case 0:
    case 2:
      return 0.;

    default:
      return fe_lagrange_1D_linear_shape_deriv(quad4_i0(i), 0, xi) *
             fe_lagrange_1D_linear_shape_deriv(quad4_i1(i), 0, eta);
    }
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_quad9_shape_second_deriv(const unsigned int i,
                                          const unsigned int j,
                                          const Real xi,
                                          const Real eta)
{
  libmesh_assert_less(i, 9);
  libmesh_assert_less(j, 3);

  switch (j)
    {
    case 0:
      return fe_lagrange_1D_quadratic_shape_second_deriv(quad9_i0(i), 0, xi) *
             fe_lagrange_1D_quadratic_shape(quad9_i1(i), eta);

    case 1:
      return fe_lagrange_1D_quadratic_shape_deriv(quad9_i0(i), 0, xi) *
             fe_lagrange_1D_quadratic_shape_deriv(quad9_i1(i), 0, eta);

    default:
      return fe_lagrange_1D_quadratic_shape(quad9_i0(i), xi) *
             fe_lagrange_1D_quadratic_shape_second_deriv(quad9_i1(i), 0, eta);
    }
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_hex8_shape_second_deriv(const unsigned int i,
                                         const unsigned int j,
                                         const Real xi,
                                         const Real eta,
                                         const Real zeta)
{
  libmesh_assert_less(i, 8);
  libmesh_assert_less(j, 6);

  switch (j)
    {
    case 0:
    case 2:
    case 5:
      return 0.;

    case 1:
      return fe_lagrange_1D_linear_shape_deriv(hex8_i0(i), 0, xi) *
             fe_lagrange_1D_linear_shape_deriv(hex8_i1(i), 0, eta) *
             fe_lagrange_1D_linear_shape(hex8_i2(i), zeta);

    case 3:
      return fe_lagrange_1D_linear_shape_deriv(hex8_i0(i), 0, xi) *
             fe_lagrange_1D_linear_shape(hex8_i1(i), eta) *
             fe_lagrange_1D_linear_shape_deriv(hex8_i2(i), 0, zeta);

    default:
      return fe_lagrange_1D_linear_shape(hex8_i0(i), xi) *
             fe_lagrange_1D_linear_shape_deriv(hex8_i1(i), 0, eta) *
             fe_lagrange_1D_linear_shape_deriv(hex8_i2(i), 0, zeta);
    }
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_hex27_shape_second_deriv(const unsigned int i,
                                          const unsigned int j,
                                          const Real xi,
                                          const Real eta,
                                          const Real zeta)
{
  libmesh_assert_less(i, 27);
  libmesh_assert_less(j, 6);

  switch (j)
    {
    case 0:
      return fe_lagrange_1D_quadratic_shape_second_deriv(hex27_i0(i), 0, xi) *
             fe_lagrange_1D_quadratic_shape(hex27_i1(i), eta) *
             fe_lagrange_1D_quadratic_shape(hex27_i2(i), zeta);

    case 1:
      return fe_lagrange_1D_quadratic_shape_deriv(hex27_i0(i), 0, xi) *
             fe_lagrange_1D_quadratic_shape_deriv(hex27_i1(i), 0, eta) *
             fe_lagrange_1D_quadratic_shape(hex27_i2(i), zeta);

    case 2:
      return fe_lagrange_1D_quadratic_shape(hex27_i0(i), xi) *
             fe_lagrange_1D_quadratic_shape_second_deriv(hex27_i1(i), 0, eta) *
             fe_lagrange_1D_quadratic_shape(hex27_i2(i), zeta);

    case 3:
      return fe_lagrange_1D_quadratic_shape_deriv(hex27_i0(i), 0, xi) *
             fe_lagrange_1D_quadratic_shape(hex27_i1(i), eta) *
             fe_lagrange_1D_quadratic_shape_deriv(hex27_i2(i), 0, zeta);

    case 4:
      return fe_lagrange_1D_quadratic_shape(hex27_i0(i), xi) *
             fe_lagrange_1D_quadratic_shape_deriv(hex27_i1(i), 0, eta) *
             fe_lagrange_1D_quadratic_shape_deriv(hex27_i2(i), 0, zeta);

    default:
      return fe_lagrange_1D_quadratic_shape(hex27_i0(i), xi) *
             fe_lagrange_1D_quadratic_shape(hex27_i1(i), eta) *
             fe_lagrange_1D_quadratic_shape_second_deriv(hex27_i2(i), 0, zeta);
    }
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES


// The reduced-node ("serendipity") members of the tensor-product
// topologies: QUAD8 and HEX20 drop the interior/face nodes, so their
// bases are not products of 1D factors, but they live on the same
// reference domains and dispatch through the same element classes.

LIBMESH_DEVICE_INLINE
Real fe_lagrange_quad8_shape(const unsigned int i,
                             const Real xi,
                             const Real eta)
{
  libmesh_assert_less(i, 8);

  switch (i)
    {
    case 0: return 0.25 * (1.0 - xi) * (1.0 - eta) * (-1.0 - xi - eta);
    case 1: return 0.25 * (1.0 + xi) * (1.0 - eta) * (-1.0 + xi - eta);
    case 2: return 0.25 * (1.0 + xi) * (1.0 + eta) * (-1.0 + xi + eta);
    case 3: return 0.25 * (1.0 - xi) * (1.0 + eta) * (-1.0 - xi + eta);
    case 4: return 0.5  * (1.0 - xi * xi) * (1.0 - eta);
    case 5: return 0.5  * (1.0 + xi) * (1.0 - eta * eta);
    case 6: return 0.5  * (1.0 - xi * xi) * (1.0 + eta);
    default: return 0.5 * (1.0 - xi) * (1.0 - eta * eta);
    }
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_quad8_shape_deriv(const unsigned int i,
                                   const unsigned int j,
                                   const Real xi,
                                   const Real eta)
{
  libmesh_assert_less(i, 8);
  libmesh_assert_less(j, 2);

  switch (j)
    {
    case 0:
      switch (i)
        {
        case 0: return 0.25 * (1.0 - eta) * (2.0 * xi + eta);
        case 1: return 0.25 * (1.0 - eta) * (2.0 * xi - eta);
        case 2: return 0.25 * (1.0 + eta) * (2.0 * xi + eta);
        case 3: return 0.25 * (1.0 + eta) * (2.0 * xi - eta);
        case 4: return -xi * (1.0 - eta);
        case 5: return 0.5 * (1.0 - eta * eta);
        case 6: return -xi * (1.0 + eta);
        default: return -0.5 * (1.0 - eta * eta);
        }

    default:
      switch (i)
        {
        case 0: return 0.25 * (1.0 - xi) * (xi + 2.0 * eta);
        case 1: return 0.25 * (1.0 + xi) * (2.0 * eta - xi);
        case 2: return 0.25 * (1.0 + xi) * (xi + 2.0 * eta);
        case 3: return 0.25 * (1.0 - xi) * (2.0 * eta - xi);
        case 4: return -0.5 * (1.0 - xi * xi);
        case 5: return -eta * (1.0 + xi);
        case 6: return 0.5 * (1.0 - xi * xi);
        default: return -eta * (1.0 - xi);
        }
    }
}

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
LIBMESH_DEVICE_INLINE
Real fe_lagrange_quad8_shape_second_deriv(const unsigned int i,
                                          const unsigned int j,
                                          const Real xi,
                                          const Real eta)
{
  libmesh_assert_less(i, 8);
  libmesh_assert_less(j, 3);

  switch (j)
    {
    case 0:
      switch (i)
        {
        case 0:
        case 1:
          return 0.5 * (1.0 - eta);
        case 2:
        case 3:
          return 0.5 * (1.0 + eta);
        case 4:
          return eta - 1.0;
        case 6:
          return -1.0 - eta;
        default:
          return 0.0;
        }

    case 1:
      switch (i)
        {
        case 0: return 0.25 * (1.0 - 2.0 * xi - 2.0 * eta);
        case 1: return 0.25 * (-1.0 - 2.0 * xi + 2.0 * eta);
        case 2: return 0.25 * (1.0 + 2.0 * xi + 2.0 * eta);
        case 3: return 0.25 * (-1.0 + 2.0 * xi - 2.0 * eta);
        case 4: return xi;
        case 5: return -eta;
        case 6: return -xi;
        default: return eta;
        }

    default:
      switch (i)
        {
        case 0:
        case 3:
          return 0.5 * (1.0 - xi);
        case 1:
        case 2:
          return 0.5 * (1.0 + xi);
        case 5:
          return -1.0 - xi;
        case 7:
          return xi - 1.0;
        default:
          return 0.0;
        }
    }
}
#endif

LIBMESH_DEVICE_INLINE
Real fe_lagrange_hex20_shape(const unsigned int i,
                             const Real xi,
                             const Real eta,
                             const Real zeta)
{
  libmesh_assert_less(i, 20);

  const Real x = 0.5 * (xi + 1.0);
  const Real y = 0.5 * (eta + 1.0);
  const Real z = 0.5 * (zeta + 1.0);

  switch (i)
    {
    case 0: return (1.0 - x) * (1.0 - y) * (1.0 - z) * (1.0 - 2.0 * x - 2.0 * y - 2.0 * z);
    case 1: return x * (1.0 - y) * (1.0 - z) * (2.0 * x - 2.0 * y - 2.0 * z - 1.0);
    case 2: return x * y * (1.0 - z) * (2.0 * x + 2.0 * y - 2.0 * z - 3.0);
    case 3: return (1.0 - x) * y * (1.0 - z) * (2.0 * y - 2.0 * x - 2.0 * z - 1.0);
    case 4: return (1.0 - x) * (1.0 - y) * z * (2.0 * z - 2.0 * x - 2.0 * y - 1.0);
    case 5: return x * (1.0 - y) * z * (2.0 * x - 2.0 * y + 2.0 * z - 3.0);
    case 6: return x * y * z * (2.0 * x + 2.0 * y + 2.0 * z - 5.0);
    case 7: return (1.0 - x) * y * z * (2.0 * y - 2.0 * x + 2.0 * z - 3.0);
    case 8: return 4.0 * x * (1.0 - x) * (1.0 - y) * (1.0 - z);
    case 9: return 4.0 * x * y * (1.0 - y) * (1.0 - z);
    case 10: return 4.0 * x * (1.0 - x) * y * (1.0 - z);
    case 11: return 4.0 * (1.0 - x) * y * (1.0 - y) * (1.0 - z);
    case 12: return 4.0 * (1.0 - x) * (1.0 - y) * z * (1.0 - z);
    case 13: return 4.0 * x * (1.0 - y) * z * (1.0 - z);
    case 14: return 4.0 * x * y * z * (1.0 - z);
    case 15: return 4.0 * (1.0 - x) * y * z * (1.0 - z);
    case 16: return 4.0 * x * (1.0 - x) * (1.0 - y) * z;
    case 17: return 4.0 * x * y * (1.0 - y) * z;
    case 18: return 4.0 * x * (1.0 - x) * y * z;
    default: return 4.0 * (1.0 - x) * y * (1.0 - y) * z;
    }
}

LIBMESH_DEVICE_INLINE
Real fe_lagrange_hex20_shape_deriv(const unsigned int i,
                                   const unsigned int j,
                                   const Real xi,
                                   const Real eta,
                                   const Real zeta)
{
  libmesh_assert_less(i, 20);
  libmesh_assert_less(j, 3);

  const Real x = 0.5 * (xi + 1.0);
  const Real y = 0.5 * (eta + 1.0);
  const Real z = 0.5 * (zeta + 1.0);

  switch (j)
    {
    case 0:
      switch (i)
        {
        case 0: return 0.5 * (1.0 - y) * (1.0 - z) * ((1.0 - x) * (-2.0) + (-1.0) * (1.0 - 2.0 * x - 2.0 * y - 2.0 * z));
        case 1: return 0.5 * (1.0 - y) * (1.0 - z) * (x * 2.0 + (2.0 * x - 2.0 * y - 2.0 * z - 1.0));
        case 2: return 0.5 * y * (1.0 - z) * (x * 2.0 + (2.0 * x + 2.0 * y - 2.0 * z - 3.0));
        case 3: return 0.5 * y * (1.0 - z) * ((1.0 - x) * (-2.0) + (-1.0) * (2.0 * y - 2.0 * x - 2.0 * z - 1.0));
        case 4: return 0.5 * (1.0 - y) * z * ((1.0 - x) * (-2.0) + (-1.0) * (2.0 * z - 2.0 * x - 2.0 * y - 1.0));
        case 5: return 0.5 * (1.0 - y) * z * (x * 2.0 + (2.0 * x - 2.0 * y + 2.0 * z - 3.0));
        case 6: return 0.5 * y * z * (x * 2.0 + (2.0 * x + 2.0 * y + 2.0 * z - 5.0));
        case 7: return 0.5 * y * z * ((1.0 - x) * (-2.0) + (-1.0) * (2.0 * y - 2.0 * x + 2.0 * z - 3.0));
        case 8: return 2.0 * (1.0 - y) * (1.0 - z) * (1.0 - 2.0 * x);
        case 9: return 2.0 * y * (1.0 - y) * (1.0 - z);
        case 10: return 2.0 * y * (1.0 - z) * (1.0 - 2.0 * x);
        case 11: return -2.0 * y * (1.0 - y) * (1.0 - z);
        case 12: return -2.0 * (1.0 - y) * z * (1.0 - z);
        case 13: return 2.0 * (1.0 - y) * z * (1.0 - z);
        case 14: return 2.0 * y * z * (1.0 - z);
        case 15: return -2.0 * y * z * (1.0 - z);
        case 16: return 2.0 * (1.0 - y) * z * (1.0 - 2.0 * x);
        case 17: return 2.0 * y * (1.0 - y) * z;
        case 18: return 2.0 * y * z * (1.0 - 2.0 * x);
        default: return -2.0 * y * (1.0 - y) * z;
        }

    case 1:
      switch (i)
        {
        case 0: return 0.5 * (1.0 - x) * (1.0 - z) * ((1.0 - y) * (-2.0) + (-1.0) * (1.0 - 2.0 * x - 2.0 * y - 2.0 * z));
        case 1: return 0.5 * x * (1.0 - z) * ((1.0 - y) * (-2.0) + (-1.0) * (2.0 * x - 2.0 * y - 2.0 * z - 1.0));
        case 2: return 0.5 * x * (1.0 - z) * (y * 2.0 + (2.0 * x + 2.0 * y - 2.0 * z - 3.0));
        case 3: return 0.5 * (1.0 - x) * (1.0 - z) * (y * 2.0 + (2.0 * y - 2.0 * x - 2.0 * z - 1.0));
        case 4: return 0.5 * (1.0 - x) * z * ((1.0 - y) * (-2.0) + (-1.0) * (2.0 * z - 2.0 * x - 2.0 * y - 1.0));
        case 5: return 0.5 * x * z * ((1.0 - y) * (-2.0) + (-1.0) * (2.0 * x - 2.0 * y + 2.0 * z - 3.0));
        case 6: return 0.5 * x * z * (y * 2.0 + (2.0 * x + 2.0 * y + 2.0 * z - 5.0));
        case 7: return 0.5 * (1.0 - x) * z * (y * 2.0 + (2.0 * y - 2.0 * x + 2.0 * z - 3.0));
        case 8: return -2.0 * x * (1.0 - x) * (1.0 - z);
        case 9: return 2.0 * x * (1.0 - z) * (1.0 - 2.0 * y);
        case 10: return 2.0 * x * (1.0 - x) * (1.0 - z);
        case 11: return 2.0 * (1.0 - x) * (1.0 - z) * (1.0 - 2.0 * y);
        case 12: return -2.0 * (1.0 - x) * z * (1.0 - z);
        case 13: return -2.0 * x * z * (1.0 - z);
        case 14: return 2.0 * x * z * (1.0 - z);
        case 15: return 2.0 * (1.0 - x) * z * (1.0 - z);
        case 16: return -2.0 * x * (1.0 - x) * z;
        case 17: return 2.0 * x * z * (1.0 - 2.0 * y);
        case 18: return 2.0 * x * (1.0 - x) * z;
        default: return 2.0 * (1.0 - x) * z * (1.0 - 2.0 * y);
        }

    default:
      switch (i)
        {
        case 0: return 0.5 * (1.0 - x) * (1.0 - y) * ((1.0 - z) * (-2.0) + (-1.0) * (1.0 - 2.0 * x - 2.0 * y - 2.0 * z));
        case 1: return 0.5 * x * (1.0 - y) * ((1.0 - z) * (-2.0) + (-1.0) * (2.0 * x - 2.0 * y - 2.0 * z - 1.0));
        case 2: return 0.5 * x * y * ((1.0 - z) * (-2.0) + (-1.0) * (2.0 * x + 2.0 * y - 2.0 * z - 3.0));
        case 3: return 0.5 * (1.0 - x) * y * ((1.0 - z) * (-2.0) + (-1.0) * (2.0 * y - 2.0 * x - 2.0 * z - 1.0));
        case 4: return 0.5 * (1.0 - x) * (1.0 - y) * (z * 2.0 + (2.0 * z - 2.0 * x - 2.0 * y - 1.0));
        case 5: return 0.5 * x * (1.0 - y) * (z * 2.0 + (2.0 * x - 2.0 * y + 2.0 * z - 3.0));
        case 6: return 0.5 * x * y * (z * 2.0 + (2.0 * x + 2.0 * y + 2.0 * z - 5.0));
        case 7: return 0.5 * (1.0 - x) * y * (z * 2.0 + (2.0 * y - 2.0 * x + 2.0 * z - 3.0));
        case 8: return -2.0 * x * (1.0 - x) * (1.0 - y);
        case 9: return -2.0 * x * y * (1.0 - y);
        case 10: return -2.0 * x * (1.0 - x) * y;
        case 11: return -2.0 * (1.0 - x) * y * (1.0 - y);
        case 12: return 2.0 * (1.0 - x) * (1.0 - y) * (1.0 - 2.0 * z);
        case 13: return 2.0 * x * (1.0 - y) * (1.0 - 2.0 * z);
        case 14: return 2.0 * x * y * (1.0 - 2.0 * z);
        case 15: return 2.0 * (1.0 - x) * y * (1.0 - 2.0 * z);
        case 16: return 2.0 * x * (1.0 - x) * (1.0 - y);
        case 17: return 2.0 * x * y * (1.0 - y);
        case 18: return 2.0 * x * (1.0 - x) * y;
        default: return 2.0 * (1.0 - x) * y * (1.0 - y);
        }
    }
}

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
LIBMESH_DEVICE_INLINE
Real fe_lagrange_hex20_shape_second_deriv(const unsigned int i,
                                          const unsigned int j,
                                          const Real xi,
                                          const Real eta,
                                          const Real zeta)
{
  libmesh_assert_less(i, 20);
  libmesh_assert_less(j, 6);

  const Real x = 0.5 * (xi + 1.0);
  const Real y = 0.5 * (eta + 1.0);
  const Real z = 0.5 * (zeta + 1.0);

  switch (j)
    {
    case 0:
      switch (i)
        {
        case 0:
        case 1: return (1.0 - y) * (1.0 - z);
        case 2:
        case 3: return y * (1.0 - z);
        case 4:
        case 5: return (1.0 - y) * z;
        case 6:
        case 7: return y * z;
        case 8: return -2.0 * (1.0 - y) * (1.0 - z);
        case 10: return -2.0 * y * (1.0 - z);
        case 16: return -2.0 * (1.0 - y) * z;
        case 18: return -2.0 * y * z;
        default: return 0.0;
        }

    case 1:
      switch (i)
        {
        case 0: return (1.25 - x - y - 0.5 * z) * (1.0 - z);
        case 1: return (-x + y + 0.5 * z - 0.25) * (1.0 - z);
        case 2: return (x + y - 0.5 * z - 0.75) * (1.0 - z);
        case 3: return (-y + x + 0.5 * z - 0.25) * (1.0 - z);
        case 4: return -0.25 * z * (4.0 * x + 4.0 * y - 2.0 * z - 3.0);
        case 5: return -0.25 * z * (-4.0 * y + 4.0 * x + 2.0 * z - 1.0);
        case 6: return 0.25 * z * (-5.0 + 4.0 * x + 4.0 * y + 2.0 * z);
        case 7: return 0.25 * z * (4.0 * x - 4.0 * y - 2.0 * z + 1.0);
        case 8: return (-1.0 + 2.0 * x) * (1.0 - z);
        case 9: return (1.0 - 2.0 * y) * (1.0 - z);
        case 10: return (1.0 - 2.0 * x) * (1.0 - z);
        case 11: return (-1.0 + 2.0 * y) * (1.0 - z);
        case 12: return z * (1.0 - z);
        case 13: return -z * (1.0 - z);
        case 14: return z * (1.0 - z);
        case 15: return -z * (1.0 - z);
        case 16: return (-1.0 + 2.0 * x) * z;
        case 17: return (1.0 - 2.0 * y) * z;
        case 18: return (1.0 - 2.0 * x) * z;
        default: return (-1.0 + 2.0 * y) * z;
        }

    case 2:
      switch (i)
        {
        case 0:
        case 3: return (1.0 - x) * (1.0 - z);
        case 1:
        case 2: return x * (1.0 - z);
        case 4:
        case 7: return (1.0 - x) * z;
        case 5:
        case 6: return x * z;
        case 9: return -2.0 * x * (1.0 - z);
        case 11: return -2.0 * (1.0 - x) * (1.0 - z);
        case 17: return -2.0 * x * z;
        case 19: return -2.0 * (1.0 - x) * z;
        default: return 0.0;
        }

    case 3:
      switch (i)
        {
        case 0: return (1.25 - x - 0.5 * y - z) * (1.0 - y);
        case 1: return (-x + 0.5 * y + z - 0.25) * (1.0 - y);
        case 2: return -0.25 * y * (2.0 * y + 4.0 * x - 4.0 * z - 1.0);
        case 3: return -0.25 * y * (-2.0 * y + 4.0 * x + 4.0 * z - 3.0);
        case 4: return (-z + x + 0.5 * y - 0.25) * (1.0 - y);
        case 5: return (x - 0.5 * y + z - 0.75) * (1.0 - y);
        case 6: return 0.25 * y * (2.0 * y + 4.0 * x + 4.0 * z - 5.0);
        case 7: return 0.25 * y * (-2.0 * y + 4.0 * x - 4.0 * z + 1.0);
        case 8: return (-1.0 + 2.0 * x) * (1.0 - y);
        case 9: return -y * (1.0 - y);
        case 10: return (-1.0 + 2.0 * x) * y;
        case 11: return y * (1.0 - y);
        case 12: return (-1.0 + 2.0 * z) * (1.0 - y);
        case 13: return (1.0 - 2.0 * z) * (1.0 - y);
        case 14: return (1.0 - 2.0 * z) * y;
        case 15: return (-1.0 + 2.0 * z) * y;
        case 16: return (1.0 - 2.0 * x) * (1.0 - y);
        case 17: return y * (1.0 - y);
        case 18: return (1.0 - 2.0 * x) * y;
        default: return -y * (1.0 - y);
        }

    case 4:
      switch (i)
        {
        case 0: return (1.25 - 0.5 * x - y - z) * (1.0 - x);
        case 1: return 0.25 * x * (2.0 * x - 4.0 * y - 4.0 * z + 3.0);
        case 2: return -0.25 * x * (2.0 * x + 4.0 * y - 4.0 * z - 1.0);
        case 3: return (-y + 0.5 * x + z - 0.25) * (1.0 - x);
        case 4: return (-z + 0.5 * x + y - 0.25) * (1.0 - x);
        case 5: return -0.25 * x * (2.0 * x - 4.0 * y + 4.0 * z - 1.0);
        case 6: return 0.25 * x * (2.0 * x + 4.0 * y + 4.0 * z - 5.0);
        case 7: return (y - 0.5 * x + z - 0.75) * (1.0 - x);
        case 8: return x * (1.0 - x);
        case 9: return (-1.0 + 2.0 * y) * x;
        case 10: return -x * (1.0 - x);
        case 11: return (-1.0 + 2.0 * y) * (1.0 - x);
        case 12: return (-1.0 + 2.0 * z) * (1.0 - x);
        case 13: return (-1.0 + 2.0 * z) * x;
        case 14: return (1.0 - 2.0 * z) * x;
        case 15: return (1.0 - 2.0 * z) * (1.0 - x);
        case 16: return -x * (1.0 - x);
        case 17: return (1.0 - 2.0 * y) * x;
        case 18: return x * (1.0 - x);
        default: return (1.0 - 2.0 * y) * (1.0 - x);
        }

    default:
      switch (i)
        {
        case 0:
        case 4: return (1.0 - x) * (1.0 - y);
        case 1:
        case 5: return x * (1.0 - y);
        case 2:
        case 6: return x * y;
        case 3:
        case 7: return (1.0 - x) * y;
        case 12: return -2.0 * (1.0 - x) * (1.0 - y);
        case 13: return -2.0 * x * (1.0 - y);
        case 14: return -2.0 * x * y;
        case 15: return -2.0 * (1.0 - x) * y;
        default: return 0.0;
        }
    }
}
#endif

} // namespace detail
} // namespace libMesh

#endif // LIBMESH_FE_TENSOR_PRODUCT_LAGRANGE_H
