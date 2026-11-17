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

#ifndef LIBMESH_FE_LAGRANGE_SHAPE_3D_H
#define LIBMESH_FE_LAGRANGE_SHAPE_3D_H

// Local includes
#include "libmesh/libmesh_device.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/fe_lagrange_shape_1D.h"
#include "libmesh/fe_lagrange_shape_2D.h"
#include "libmesh/fe_simplex_lagrange.h"
#include "libmesh/fe_tensor_product_lagrange.h"
#include "libmesh/point.h"
#include "libmesh/vector_value.h"

// C++ includes
#include <cstddef>

// The 3D LAGRANGE and L2_LAGRANGE shape functions, and their first
// and second derivatives, for every element type that defines them
// by its ElemType alone: everything but C0Polyhedron, whose shapes
// depend on the polyhedron itself.  FE<3,LAGRANGE> and
// FE<3,L2_LAGRANGE> call these, and they are inline here so that
// device code evaluates the same functions.  As in
// fe_lagrange_shape_1D.h, they are templated on the coordinate type.

namespace libMesh
{

#if LIBMESH_DIM == 3

namespace detail
{

/**
 * A table of small unsigned values packed Bits bits each into one
 * word: t[i] is a shift and a mask.  A shape function indexing a
 * local array with its run-time shape number i makes device code put
 * the array in local memory, as a stack frame each thread reads; the
 * packed word is a constant the compiler keeps in a register.
 */
template <unsigned int Bits>
struct PackedTable
{
  unsigned long long word;

  LIBMESH_DEVICE_INLINE
  constexpr unsigned int operator[] (const unsigned int i) const
  { return static_cast<unsigned int>(word >> (Bits * i)) & ((1u << Bits) - 1); }
};

template <unsigned int Bits, std::size_t N>
LIBMESH_DEVICE_INLINE
constexpr PackedTable<Bits> pack_table (const unsigned int (&values)[N])
{
  static_assert(Bits * N <= 64, "the table must fit in one word");
  unsigned long long word = 0;
  for (std::size_t i = 0; i != N; ++i)
    word |= static_cast<unsigned long long>(values[i]) << (Bits * i);
  return PackedTable<Bits>{word};
}

/**
 * A two-dimensional table of small unsigned values packed into one
 * word, as PackedTable is: t(row, col) is a shift and a mask.
 */
template <unsigned int Bits, unsigned int Cols>
struct PackedMatrix
{
  PackedTable<Bits> table;

  LIBMESH_DEVICE_INLINE
  constexpr unsigned int operator() (const unsigned int row, const unsigned int col) const
  { return table[row * Cols + col]; }
};

template <unsigned int Bits, std::size_t Rows, std::size_t Cols>
LIBMESH_DEVICE_INLINE
constexpr PackedMatrix<Bits, Cols> pack_table (const unsigned int (&values)[Rows][Cols])
{
  unsigned int flat[Rows * Cols] = {};
  for (std::size_t r = 0; r != Rows; ++r)
    for (std::size_t c = 0; c != Cols; ++c)
      flat[r * Cols + c] = values[r][c];
  return PackedMatrix<Bits, Cols>{pack_table<Bits>(flat)};
}


// The PRISM21 cubic shapes, a tensor product of TRI7 and EDGE3 shapes,
// and their derivatives.  PRISM20's shapes are PRISM21's with the
// central bubble redistributed, so both call these, rather than the
// PRISM20 case calling the dispatch function it is in: device code
// can't inline a recursive call, so the call would carry the stack
// frame of the whole dispatch.

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_prism21_shape(const unsigned int i,
                                 const TypeVector<Scalar> & p)
{
  // Compute prism shape functions as a tensor-product
  // of a triangle and an edge

  VectorValue<Scalar> p2d(p(0),p(1));
  Scalar p1d = p(2);

  //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
  constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 1, 2});
  constexpr auto i1 = detail::pack_table<3>({0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 6, 6});

  return (fe_lagrange_2D_shape<LAGRANGE>(TRI7, THIRD, i1[i], p2d)*
          fe_lagrange_1D_quadratic_shape(i0[i], p1d));
}

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_prism21_shape_deriv(const unsigned int i,
                                       const unsigned int j,
                                       const TypeVector<Scalar> & p)
{
  // Compute prism shape functions as a tensor-product
  // of a triangle and an edge

  VectorValue<Scalar> p2d(p(0),p(1));
  Scalar p1d = p(2);

  //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
  constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 1, 2});
  constexpr auto i1 = detail::pack_table<3>({0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 6, 6});

  switch (j)
    {
      // d()/dxi
    case 0:
      return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI7, THIRD, i1[i], 0, p2d)*
              fe_lagrange_1D_quadratic_shape(i0[i], p1d));

      // d()/deta
    case 1:
      return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI7, THIRD, i1[i], 1, p2d)*
              fe_lagrange_1D_quadratic_shape(i0[i], p1d));

      // d()/dzeta
    case 2:
      return (fe_lagrange_2D_shape<LAGRANGE>(TRI7, THIRD, i1[i], p2d)*
              fe_lagrange_1D_quadratic_shape_deriv(i0[i], 0, p1d));

    default:
      libmesh_error_msg("Invalid shape function derivative j = " << j);
    }
}

// The derivatives of the PYRAMID14 quadratic shapes, defined in
// R. Graglia, "Higher order bases on pyramidal elements", IEEE Trans
// Antennas and Propagation, vol 47, no 5, May 1999.  The PYRAMID18
// cubic shapes add face bubbles to these, so both call them, for the
// same reason as the PRISM21 functions above.

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_pyramid14_shape_deriv(const unsigned int i,
                                         const unsigned int j,
                                         const TypeVector<Scalar> & p)
{
  libmesh_assert_less (i, 14);

  const Scalar xi   = p(0);
  const Scalar eta  = p(1);
  const Scalar zeta = p(2);
  const Real eps  = 1.e-35;

  // The "normalized coordinates" defined by Graglia.  These are
  // the planes which define the faces of the pyramid.
  Scalar
    p1 = 0.5*(1. - eta - zeta), // back
    p2 = 0.5*(1. + xi  - zeta), // left
    p3 = 0.5*(1. + eta - zeta), // front
    p4 = 0.5*(1. - xi  - zeta); // right

  // Denominators are perturbed by epsilon to avoid
  // divide-by-zero issues.
  Scalar
    den = (-1. + zeta + eps),
    den2 = den*den,
    den3 = den2*den;

  switch (j)
    {
      // d/dxi
    case 0:
      switch(i)
        {
        case 0:
          return 0.5*p1*(-xi*eta + zeta - zeta*zeta + 2.*p4*eta)/den2;

        case 1:
          return -0.5*p1*(xi*eta + zeta - zeta*zeta + 2.*p2*eta)/den2;

        case 2:
          return 0.5*p3*(xi*eta - zeta + zeta*zeta + 2.*p2*eta)/den2;

        case 3:
          return -0.5*p3*(-xi*eta - zeta + zeta*zeta + 2.*p4*eta)/den2;

        case 4:
          return 0.;

        case 5:
          return 2.*p1*eta*xi/den2;

        case 6:
          return 2.*p1*p3*(xi + 2.*p2)/den2;

        case 7:
          return -2.*p3*eta*xi/den2;

        case 8:
          return -2.*p1*p3*(-xi + 2.*p4)/den2;

        case 9:
          return 2.*p1*zeta/den;

        case 10:
          return -2.*p1*zeta/den;

        case 11:
          return -2.*p3*zeta/den;

        case 12:
          return 2.*p3*zeta/den;

        case 13:
          return -8.*p1*p3*xi/den2;

        default:
          libmesh_error_msg("Invalid i = " << i);
        }

      // d/deta
    case 1:
      switch(i)
        {
        case 0:
          return -0.5*p4*(xi*eta - zeta + zeta*zeta - 2.*p1*xi)/den2;

        case 1:
          return 0.5*p2*(xi*eta + zeta - zeta*zeta - 2.*p1*xi)/den2;

        case 2:
          return 0.5*p2*(xi*eta - zeta + zeta*zeta + 2.*p3*xi)/den2;

        case 3:
          return -0.5*p4*(xi*eta + zeta - zeta*zeta + 2.*p3*xi)/den2;

        case 4:
          return 0.;

        case 5:
          return 2.*p2*p4*(eta - 2.*p1)/den2;

        case 6:
          return -2.*p2*xi*eta/den2;

        case 7:
          return 2.*p2*p4*(eta + 2.*p3)/den2;

        case 8:
          return 2.*p4*xi*eta/den2;

        case 9:
          return 2.*p4*zeta/den;

        case 10:
          return 2.*p2*zeta/den;

        case 11:
          return -2.*p2*zeta/den;

        case 12:
          return -2.*p4*zeta/den;

        case 13:
          return -8.*p2*p4*eta/den2;

        default:
          libmesh_error_msg("Invalid i = " << i);
        }


      // d/dzeta
    case 2:
      {
        switch(i)
          {
          case 0:
            return -0.5*p1*(xi*eta - zeta + zeta*zeta)/den2
              - 0.5*p4*(xi*eta - zeta + zeta*zeta)/den2
              + p4*p1*(2.*zeta - 1)/den2
              - 2.*p4*p1*(xi*eta - zeta + zeta*zeta)/den3;

          case 1:
            return 0.5*p2*(xi*eta + zeta - zeta*zeta)/den2
              + 0.5*p1*(xi*eta + zeta - zeta*zeta)/den2
              - p1*p2*(1 - 2.*zeta)/den2
              + 2.*p1*p2*(xi*eta + zeta - zeta*zeta)/den3;

          case 2:
            return -0.5*p3*(xi*eta - zeta + zeta*zeta)/den2
              - 0.5*p2*(xi*eta - zeta + zeta*zeta)/den2
              + p2*p3*(2.*zeta - 1)/den2
              - 2.*p2*p3*(xi*eta - zeta + zeta*zeta)/den3;

          case 3:
            return 0.5*p4*(xi*eta + zeta - zeta*zeta)/den2
              + 0.5*p3*(xi*eta + zeta - zeta*zeta)/den2
              - p3*p4*(1 - 2.*zeta)/den2
              + 2.*p3*p4*(xi*eta + zeta - zeta*zeta)/den3;

          case 4:
            return 4.*zeta - 1.;

          case 5:
            return 2.*p4*p1*eta/den2
              + 2.*p2*p4*eta/den2
              + 2.*p1*p2*eta/den2
              + 8.*p2*p1*p4*eta/den3;

          case 6:
            return -2.*p2*p3*xi/den2
              - 2.*p1*p3*xi/den2
              - 2.*p1*p2*xi/den2
              - 8.*p1*p2*p3*xi/den3;

          case 7:
            return -2.*p3*p4*eta/den2
              - 2.*p2*p4*eta/den2
              - 2.*p2*p3*eta/den2
              - 8.*p2*p3*p4*eta/den3;

          case 8:
            return 2.*p4*p1*xi/den2
              + 2.*p1*p3*xi/den2
              + 2.*p3*p4*xi/den2
              + 8.*p3*p4*p1*xi/den3;

          case 9:
            return 2.*p4*zeta/den
              + 2.*p1*zeta/den
              - 4.*p1*p4/den
              + 4.*p1*p4*zeta/den2;

          case 10:
            return 2.*p1*zeta/den
              + 2.*p2*zeta/den
              - 4.*p2*p1/den
              + 4.*p2*p1*zeta/den2;

          case 11:
            return 2.*p2*zeta/den
              + 2.*p3*zeta/den
              - 4.*p3*p2/den
              + 4.*p3*p2*zeta/den2;

          case 12:
            return 2.*p3*zeta/den
              + 2.*p4*zeta/den
              - 4.*p4*p3/den
              + 4.*p4*p3*zeta/den2;

          case 13:
            return -8.*p2*p3*p4/den2
              - 8.*p3*p4*p1/den2
              - 8.*p2*p1*p4/den2
              - 8.*p1*p2*p3/den2
              - 32.*p1*p2*p3*p4/den3;

          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }

    default:
      libmesh_error_msg("Invalid j = " << j);
    }
}

// The PYRAMID18 cubic shapes are the PYRAMID14 shapes plus multiples
// of four bubbles on the triangular sides, bub1..bub4 in
// fe_lagrange_3D_shape().  Each bubble is a product of five linear
// factors (its weight bub_f, three of Graglia's normalized coordinates
// and zeta) over den^2, den = zeta - 1 + eps, so its derivatives
// follow from the product rule, with the factors' constant gradients,
// and from d(den)/dzeta = 1.

// The bubble's factors are indexed only by loop counters with
// constant bounds, and their gradients are computed from k rather
// than read from tables, so that device code, which puts an array
// indexed at run time in local memory, keeps them in registers.
template <typename Scalar>
struct Pyramid18Bubble
{
  unsigned int k;
  // The linear factors at the point
  Scalar f[5];
  Scalar den;

  // Component d of the gradient of factor a
  LIBMESH_DEVICE_INLINE
  Real grad(const unsigned int a, const unsigned int d) const;
};

namespace pyramid18
{
constexpr Real alpha = 0.5;

// The pyramid's side c (back, left, front, right) varies as t_c =
// -eta, xi, eta, -xi; Graglia's p_c is (1 + t_c - zeta)/2, and bubble
// k's weight varies as t_k.  dt_c/d(direction d):
LIBMESH_DEVICE_INLINE
Real t_grad(const unsigned int c, const unsigned int d)
{
  if (d == 0)
    return (c == 1) ? 1 : (c == 3) ? -1 : 0;
  if (d == 1)
    return (c == 2) ? 1 : (c == 0) ? -1 : 0;
  return 0;
}

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar t(const unsigned int c, const TypeVector<Scalar> & p)
{
  return (c == 0) ? -p(1) : (c == 1) ? p(0) : (c == 2) ? p(1) : -p(0);
}

// The m-th of the three sides whose p_c are bubble k's factors
LIBMESH_DEVICE_INLINE
unsigned int side_of(const unsigned int k, const unsigned int m)
{
  constexpr auto sides = detail::pack_table<2>({{0, 1, 3}, {0, 1, 2}, {1, 2, 3}, {0, 2, 3}});
  return sides(k, m);
}
}

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Real Pyramid18Bubble<Scalar>::grad(const unsigned int a, const unsigned int d) const
{
  using namespace pyramid18;
  if (a == 0)
    return alpha*t_grad(k, d) - (d == 2 ? 1-alpha : 0);
  if (a == 4)
    return (d == 2);
  return (d == 2) ? Real(-0.5) : 0.5*t_grad(side_of(k, a-1), d);
}

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Pyramid18Bubble<Scalar> fe_lagrange_pyramid18_bubble(const unsigned int k,
                                                     const TypeVector<Scalar> & p)
{
  libmesh_assert_less (k, 4);
  using namespace pyramid18;

  const Scalar zeta = p(2);
  const Real eps  = 1.e-35;

  // bub_f = (1-alpha)*(1-zeta) + alpha*t_k, then Graglia's p_c of the
  // bubble's three sides, then zeta
  Pyramid18Bubble<Scalar> b;
  b.k = k;
  b.f[0] = (1-alpha)*(1-zeta) + alpha*t(k, p);
  LIBMESH_DEVICE_UNROLL
  for (unsigned int m = 0; m != 3; ++m)
    b.f[m+1] = 0.5*(1. + t(side_of(k, m), p) - zeta);
  b.f[4] = zeta;
  b.den = -1. + zeta + eps;
  return b;
}

// The multiple of bubble k in PYRAMID18 shape i, as in
// fe_lagrange_3D_shape():
//
//   vertices 0-3      3 on bubbles i and i-1 (mod 4)
//   apex 4            3 on every bubble
//   base edges 5-8    -12 on bubble i-5
//   side edges 9-12   -12 on bubbles i-9 and i-10 (mod 4)
//   base 13           none
//   sides 14-17       27 on bubble i-14
//
// computed rather than read from a table: device code would put a
// table indexed by the run-time shape number in local memory.
LIBMESH_DEVICE_INLINE
int fe_lagrange_pyramid18_bubble_weight(const unsigned int i,
                                        const unsigned int k)
{
  libmesh_assert_less (i, 18);
  libmesh_assert_less (k, 4);
  if (i < 4)
    return (k == i || k == (i + 3) % 4) ? 3 : 0;
  if (i == 4)
    return 3;
  if (i < 9)
    return (k == i - 5) ? -12 : 0;
  if (i < 13)
    return (k == i - 9 || k == (i - 9 + 3) % 4) ? -12 : 0;
  if (i == 13)
    return 0;
  return (k == i - 14) ? 27 : 0;
}

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_pyramid18_bubble_deriv(const unsigned int k,
                                          const unsigned int j,
                                          const TypeVector<Scalar> & p)
{
  libmesh_assert_less (j, 3);
  const Pyramid18Bubble<Scalar> b = fe_lagrange_pyramid18_bubble(k, p);

  // The product F of the factors and its derivative F_j
  Scalar F = 1, F_j = 0;
  for (unsigned int a = 0; a != 5; ++a)
    {
      Scalar term = b.grad(a, j);
      for (unsigned int c = 0; c != 5; ++c)
        if (c != a)
          term *= b.f[c];
      F_j += term;
      F *= b.f[a];
    }

  const Scalar den2 = b.den*b.den;
  Scalar val = F_j/den2;
  if (j == 2)
    val -= 2.*F/(den2*b.den);
  return val;
}

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_pyramid18_shape_deriv(const unsigned int i,
                                         const unsigned int j,
                                         const TypeVector<Scalar> & p)
{
  libmesh_assert_less (i, 18);

  Scalar val = 0;
  if (i < 14)
    val = fe_lagrange_pyramid14_shape_deriv(i, j, p);
  LIBMESH_DEVICE_UNROLL
  for (unsigned int k = 0; k != 4; ++k)
    if (const int w = fe_lagrange_pyramid18_bubble_weight(i, k))
      val += Real(w) * fe_lagrange_pyramid18_bubble_deriv(k, j, p);
  return val;
}

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_prism21_shape_second_deriv(const unsigned int i,
                                              const unsigned int j,
                                              const TypeVector<Scalar> & p)
{
  // Compute prism shape functions as a tensor-product
  // of a triangle and an edge

  VectorValue<Scalar> p2d(p(0),p(1));
  Scalar p1d = p(2);

  //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
  constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 1, 2});
  constexpr auto i1 = detail::pack_table<3>({0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 6, 6});

  switch (j)
    {
      // d^2()/dxi^2
    case 0:
      return (fe_lagrange_2D_shape_second_deriv<LAGRANGE>(TRI7, THIRD, i1[i], 0, p2d)*
              fe_lagrange_1D_quadratic_shape(i0[i], p1d));

      // d^2()/dxideta
    case 1:
      return (fe_lagrange_2D_shape_second_deriv<LAGRANGE>(TRI7, THIRD, i1[i], 1, p2d)*
              fe_lagrange_1D_quadratic_shape(i0[i], p1d));

      // d^2()/deta^2
    case 2:
      return (fe_lagrange_2D_shape_second_deriv<LAGRANGE>(TRI7, THIRD, i1[i], 2, p2d)*
              fe_lagrange_1D_quadratic_shape(i0[i], p1d));

      // d^2()/dxidzeta
    case 3:
      return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI7, THIRD, i1[i], 0, p2d)*
              fe_lagrange_1D_quadratic_shape_deriv(i0[i], 0, p1d));

      // d^2()/detadzeta
    case 4:
      return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI7, THIRD, i1[i], 1, p2d)*
              fe_lagrange_1D_quadratic_shape_deriv(i0[i], 0, p1d));

      // d^2()/dzeta^2
    case 5:
      return (fe_lagrange_2D_shape<LAGRANGE>(TRI7, THIRD, i1[i], p2d)*
              fe_lagrange_1D_quadratic_shape_second_deriv(i0[i], 0, p1d));

    default:
      libmesh_error_msg("Invalid shape function derivative j = " << j);
    }
}


template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_pyramid14_shape_second_deriv(const unsigned int i,
                                                const unsigned int j,
                                                const TypeVector<Scalar> & p)
{
  libmesh_assert_less (i, 14);

  const Scalar xi   = p(0);
  const Scalar eta  = p(1);
  const Scalar zeta = p(2);
  const Real eps  = 1.e-35;

  // The "normalized coordinates" defined by Graglia.  These are
  // the planes which define the faces of the pyramid.
  Scalar
    p1 = 0.5*(1. - eta - zeta), // back
    p2 = 0.5*(1. + xi  - zeta), // left
    p3 = 0.5*(1. + eta - zeta), // front
    p4 = 0.5*(1. - xi  - zeta); // right

  // Denominators are perturbed by epsilon to avoid
  // divide-by-zero issues.
  Scalar
    den = (-1. + zeta + eps),
    den2 = den*den,
    den3 = den2*den,
    den4 = den2*den2;

  // These terms are used in several of the derivatives
  Scalar
    numer_mp = xi*eta - zeta + zeta*zeta,
    numer_pm = xi*eta + zeta - zeta*zeta;

  switch (j)
    {
    case 0: // d^2()/dxi^2
      {
        switch(i)
          {
          case 0:
          case 1:
            return -p1*eta/den2;
          case 2:
          case 3:
            return p3*eta/den2;
          case 4:
          case 9:
          case 10:
          case 11:
          case 12:
            return 0.;
          case 5:
            return 2.*p1*eta/den2;
          case 6:
          case 8:
            return 4.*p1*p3/den2;
          case 7:
            return -2.*p3*eta/den2;
          case 13:
            return -8.*p1*p3/den2;
          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }

    case 1: // d^2()/dxideta
      {
        switch(i)
          {
          case 0:
            return 0.25*numer_mp/den2
              - 0.5*p1*xi/den2
              - 0.5*p4*eta/den2
              + p4*p1/den2;

          case 1:
            return 0.25*numer_pm/den2
              - 0.5*p1*xi/den2
              + 0.5*p2*eta/den2
              - p1*p2/den2;

          case 2:
            return 0.25*numer_mp/den2
              + 0.5*p3*xi/den2
              + 0.5*p2*eta/den2
              + p2*p3/den2;

          case 3:
            return 0.25*numer_pm/den2
              + 0.5*p3*xi/den2
              - 0.5*p4*eta/den2
              - p3*p4/den2;

          case 4:
            return 0.;

          case 5:
            return p4*eta/den2
              - 2.*p4*p1/den2
              - p2*eta/den2
              + 2.*p1*p2/den2;

          case 6:
            return -p3*xi/den2
              + p1*xi/den2
              - 2.*p2*p3/den2
              + 2.*p1*p2/den2;

          case 7:
            return p4*eta/den2
              + 2.*p3*p4/den2
              - p2*eta/den2
              - 2.*p2*p3/den2;

          case 8:
            return -p3*xi/den2
              + p1*xi/den2
              - 2.*p4*p1/den2
              + 2.*p3*p4/den2;

          case 9:
          case 11:
            return -zeta/den;

          case 10:
          case 12:
            return zeta/den;

          case 13:
            return 4.*p4*p1/den2
              - 4.*p3*p4/den2
              + 4.*p2*p3/den2
              - 4.*p1*p2/den2;

          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }


    case 2: // d^2()/deta^2
      {
        switch(i)
          {
          case 0:
          case 3:
            return -p4*xi/den2;
          case 1:
          case 2:
            return p2*xi/den2;
          case 4:
          case 9:
          case 10:
          case 11:
          case 12:
            return 0.;
          case 5:
          case 7:
            return 4.*p2*p4/den2;
          case 6:
            return -2.*p2*xi/den2;
          case 8:
            return 2.*p4*xi/den2;
          case 13:
            return -8.*p2*p4/den2;
          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }


    case 3: // d^2()/dxidzeta
      {
        switch(i)
          {
          case 0:
            return 0.25*numer_mp/den2
              - 0.5*p1*(2.*zeta - 1.)/den2
              + p1*numer_mp/den3
              - 0.5*p1*eta/den2
              - 0.5*p4*eta/den2
              - 2.*p4*p1*eta/den3;

          case 1:
            return 0.25*numer_pm/den2
              - 0.5*p1*(1 - 2.*zeta)/den2
              + p1*numer_pm/den3
              + 0.5*p2*eta/den2
              + 0.5*p1*eta/den2
              + 2.*p1*p2*eta/den3;

          case 2:
            return -0.25*numer_mp/den2
              + 0.5*p3*(2.*zeta - 1.)/den2
              - p3*numer_mp/den3
              - 0.5*p3*eta/den2
              - 0.5*p2*eta/den2
              - 2.*p2*p3*eta/den3;

          case 3:
            return -0.25*numer_pm/den2
              + 0.5*p3*(1 - 2.*zeta)/den2
              - p3*numer_pm/den3
              + 0.5*p4*eta/den2
              + 0.5*p3*eta/den2
              + 2.*p3*p4*eta/den3;

          case 4:
            return 0.;

          case 5:
            return p4*eta/den2
              + 4.*p4*p1*eta/den3
              - p2*eta/den2
              - 4.*p1*p2*eta/den3;

          case 6:
            return -p3*xi/den2
              - p1*xi/den2
              - 4.*p1*p3*xi/den3
              - 2.*p2*p3/den2
              - 2.*p1*p3/den2
              - 2.*p1*p2/den2
              - 8.*p1*p2*p3/den3;

          case 7:
            return -p4*eta/den2
              - 4.*p3*p4*eta/den3
              + p2*eta/den2
              + 4.*p2*p3*eta/den3;

          case 8:
            return -p3*xi/den2
              - p1*xi/den2
              - 4.*p1*p3*xi/den3
              + 2.*p4*p1/den2
              + 2.*p1*p3/den2
              + 2.*p3*p4/den2
              + 8.*p3*p4*p1/den3;

          case 9:
            return -zeta/den
              + 2.*p1/den
              - 2.*p1*zeta/den2;

          case 10:
            return zeta/den
              - 2.*p1/den
              + 2.*p1*zeta/den2;

          case 11:
            return zeta/den
              - 2.*p3/den
              + 2.*p3*zeta/den2;

          case 12:
            return -zeta/den
              + 2.*p3/den
              - 2.*p3*zeta/den2;

          case 13:
            return -4.*p4*p1/den2
              - 4.*p3*p4/den2
              - 16.*p3*p4*p1/den3
              + 4.*p2*p3/den2
              + 4.*p1*p2/den2
              + 16.*p1*p2*p3/den3;

          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }

    case 4: // d^2()/detadzeta
      {
        switch(i)
          {
          case 0:
            return 0.25*numer_mp/den2
              - 0.5*p4*(2.*zeta - 1.)/den2
              + p4*numer_mp/den3
              - 0.5*p1*xi/den2
              - 0.5*p4*xi/den2
              - 2.*p4*p1*xi/den3;

          case 1:
            return -0.25*numer_pm/den2
              + 0.5*p2*(1. - 2.*zeta)/den2
              - p2*numer_pm/den3
              + 0.5*p2*xi/den2
              + 0.5*p1*xi/den2
              + 2.*p1*p2*xi/den3;

          case 2:
            return -0.25*numer_mp/den2
              + 0.5*p2*(2.*zeta - 1.)/den2
              - p2*numer_mp/den3
              - 0.5*p3*xi/den2
              - 0.5*p2*xi/den2
              - 2.*p2*p3*xi/den3;

          case 3:
            return 0.25*numer_pm/den2
              - 0.5*p4*(1. - 2.*zeta)/den2
              + p4*numer_pm/den3
              + 0.5*p4*xi/den2
              + 0.5*p3*xi/den2
              + 2.*p3*p4*xi/den3;

          case 4:
            return 0.;

          case 5:
            return -p4*eta/den2
              - p2*eta/den2
              - 4.*p2*p4*eta/den3
              + 2.*p4*p1/den2
              + 2.*p2*p4/den2
              + 2.*p1*p2/den2
              + 8.*p2*p1*p4/den3;

          case 6:
            return p3*xi/den2
              + 4.*p2*p3*xi/den3
              - p1*xi/den2
              - 4.*p1*p2*xi/den3;

          case 7:
            return -p4*eta/den2
              - p2*eta/den2
              - 4.*p2*p4*eta/den3
              - 2.*p3*p4/den2
              - 2.*p2*p4/den2
              - 2.*p2*p3/den2
              - 8.*p2*p3*p4/den3;

          case 8:
            return p1*xi/den2
              + 4.*p4*p1*xi/den3
              - p3*xi/den2
              - 4.*p3*p4*xi/den3;

          case 9:
            return -zeta/den
              + 2.*p4/den
              - 2.*p4*zeta/den2;

          case 10:
            return -zeta/den
              + 2.*p2/den
              - 2.*p2*zeta/den2;

          case 11:
            return zeta/den
              - 2.*p2/den
              + 2.*p2*zeta/den2;

          case 12:
            return zeta/den
              - 2.*p4/den
              + 2.*p4*zeta/den2;

          case 13:
            return 4.*p3*p4/den2
              + 4.*p2*p3/den2
              + 16.*p2*p3*p4/den3
              - 4.*p4*p1/den2
              - 4.*p1*p2/den2
              - 16.*p2*p1*p4/den3;

          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }

    case 5: // d^2()/dzeta^2
      {
        switch(i)
          {
          case 0:
            return 0.5*numer_mp/den2
              - p1*(2.*zeta - 1.)/den2
              + 2.*p1*numer_mp/den3
              - p4*(2.*zeta - 1.)/den2
              + 2.*p4*numer_mp/den3
              + 2.*p4*p1/den2
              - 4.*p4*p1*(2.*zeta - 1.)/den3
              + 6.*p4*p1*numer_mp/den4;

          case 1:
            return -0.5*numer_pm/den2
              + p2*(1 - 2.*zeta)/den2
              - 2.*p2*numer_pm/den3
              + p1*(1 - 2.*zeta)/den2
              - 2.*p1*numer_pm/den3
              + 2.*p1*p2/den2
              + 4.*p1*p2*(1 - 2.*zeta)/den3
              - 6.*p1*p2*numer_pm/den4;

          case 2:
            return 0.5*numer_mp/den2
              - p3*(2.*zeta - 1.)/den2
              + 2.*p3*numer_mp/den3
              - p2*(2.*zeta - 1.)/den2
              + 2.*p2*numer_mp/den3
              + 2.*p2*p3/den2
              - 4.*p2*p3*(2.*zeta - 1.)/den3
              + 6.*p2*p3*numer_mp/den4;

          case 3:
            return -0.5*numer_pm/den2
              + p4*(1 - 2.*zeta)/den2
              - 2.*p4*numer_pm/den3
              + p3*(1 - 2.*zeta)/den2
              - 2.*p3*numer_pm/den3
              + 2.*p3*p4/den2
              + 4.*p3*p4*(1 - 2.*zeta)/den3
              - 6.*p3*p4*numer_pm/den4;

          case 4:
            return 4.;

          case 5:
            return -2.*p1*eta/den2
              - 2.*p4*eta/den2
              - 8.*p4*p1*eta/den3
              - 2.*p2*eta/den2
              - 8.*p2*p4*eta/den3
              - 8.*p1*p2*eta/den3
              - 24.*p2*p1*p4*eta/den4;

          case 6:
            return 2.*p3*xi/den2
              + 2.*p2*xi/den2
              + 8.*p2*p3*xi/den3
              + 2.*p1*xi/den2
              + 8.*p1*p3*xi/den3
              + 8.*p1*p2*xi/den3
              + 24.*p1*p2*p3*xi/den4;

          case 7:
            return 2.*p4*eta/den2
              + 2.*p3*eta/den2
              + 8.*p3*p4*eta/den3
              + 2.*p2*eta/den2
              + 8.*p2*p4*eta/den3
              + 8.*p2*p3*eta/den3
              + 24.*p2*p3*p4*eta/den4;

          case 8:
            return -2.*p1*xi/den2
              - 2.*p4*xi/den2
              - 8.*p4*p1*xi/den3
              - 2.*p3*xi/den2
              - 8.*p1*p3*xi/den3
              - 8.*p3*p4*xi/den3
              - 24.*p3*p4*p1*xi/den4;

          case 9:
            return -2.*zeta/den
              + 4.*p4/den
              - 4.*p4*zeta/den2
              + 4.*p1/den
              - 4.*p1*zeta/den2
              + 8.*p4*p1/den2
              - 8.*p1*p4*zeta/den3;

          case 10:
            return -2.*zeta/den
              + 4.*p1/den
              - 4.*p1*zeta/den2
              + 4.*p2/den
              - 4.*p2*zeta/den2
              + 8.*p1*p2/den2
              - 8.*p2*p1*zeta/den3;

          case 11:
            return -2.*zeta/den
              + 4.*p2/den
              - 4.*p2*zeta/den2
              + 4.*p3/den
              - 4.*p3*zeta/den2
              + 8.*p2*p3/den2
              - 8.*p3*p2*zeta/den3;

          case 12:
            return -2.*zeta/den
              + 4.*p3/den
              - 4.*p3*zeta/den2
              + 4.*p4/den
              - 4.*p4*zeta/den2
              + 8.*p3*p4/den2
              - 8.*p4*p3*zeta/den3;

          case 13:
            return 8.*p3*p4/den2
              + 8.*p2*p4/den2
              + 8.*p2*p3/den2
              + 32.*p2*p3*p4/den3
              + 8.*p4*p1/den2
              + 8.*p1*p3/den2
              + 32.*p3*p4*p1/den3
              + 8.*p1*p2/den2
              + 32.*p2*p1*p4/den3
              + 32.*p1*p2*p3/den3
              + 96.*p1*p2*p3*p4/den4;

          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }

    default:
      libmesh_error_msg("Invalid j = " << j);
    }
}

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_pyramid18_bubble_second_deriv(const unsigned int k,
                                                 const unsigned int j,
                                                 const TypeVector<Scalar> & p)
{
  libmesh_assert_less (j, 6);

  // libMesh numbers second derivatives xx, xy, yy, xz, yz, zz
  constexpr auto dir0 = detail::pack_table<3>({0, 0, 1, 0, 1, 2});
  constexpr auto dir1 = detail::pack_table<3>({0, 1, 1, 2, 2, 2});
  const unsigned int r = dir0[j], s = dir1[j];

  const Pyramid18Bubble<Scalar> b = fe_lagrange_pyramid18_bubble(k, p);

  // The product F of the factors, its first derivatives F_r, F_s and
  // its second derivative F_rs (the factors' own vanish)
  Scalar F = 1, F_r = 0, F_s = 0, F_rs = 0;
  for (unsigned int a = 0; a != 5; ++a)
    {
      Scalar term_r = b.grad(a, r), term_s = b.grad(a, s);
      for (unsigned int c = 0; c != 5; ++c)
        if (c != a)
          {
            term_r *= b.f[c];
            term_s *= b.f[c];
          }
      F_r += term_r;
      F_s += term_s;

      for (unsigned int e = 0; e != 5; ++e)
        if (e != a)
          {
            Scalar term_rs = b.grad(a, r) * b.grad(e, s);
            for (unsigned int c = 0; c != 5; ++c)
              if (c != a && c != e)
                term_rs *= b.f[c];
            F_rs += term_rs;
          }

      F *= b.f[a];
    }

  const Scalar
    den2 = b.den*b.den,
    den3 = den2*b.den,
    den4 = den2*den2;

  Scalar val = F_rs/den2;
  if (s == 2)
    val -= 2.*F_r/den3;
  if (r == 2)
    val -= 2.*F_s/den3;
  if (r == 2 && s == 2)
    val += 6.*F/den4;
  return val;
}

template <typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_pyramid18_shape_second_deriv(const unsigned int i,
                                                const unsigned int j,
                                                const TypeVector<Scalar> & p)
{
  libmesh_assert_less (i, 18);

  Scalar val = 0;
  if (i < 14)
    val = fe_lagrange_pyramid14_shape_second_deriv(i, j, p);
  LIBMESH_DEVICE_UNROLL
  for (unsigned int k = 0; k != 4; ++k)
    if (const int w = fe_lagrange_pyramid18_bubble_weight(i, k))
      val += Real(w) * fe_lagrange_pyramid18_bubble_second_deriv(k, j, p);
  return val;
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES

} // namespace detail

#endif // LIBMESH_DIM == 3

template <FEFamily T, typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_3D_shape(const ElemType type,
                            const Order order,
                            const unsigned int i,
                            const TypeVector<Scalar> & p)
{
#if LIBMESH_DIM == 3

  switch (order)
    {
      // linear Lagrange shape functions
    case FIRST:
      {
        switch (type)
          {
            // trilinear hexahedral shape functions
          case HEX8:
          case HEX20:
          case HEX27:
            {
              libmesh_assert_less (i, 8);

              return libMesh::detail::fe_lagrange_hex8_shape(i, p(0), p(1), p(2));
            }

            // linear tetrahedral shape functions
          case TET4:
          case TET10:
          case TET14:
            {
              libmesh_assert_less (i, 4);
              return libMesh::detail::fe_lagrange_tet4_shape(i, p(0), p(1), p(2));
            }

            // linear prism shape functions
          case PRISM6:
          case PRISM15:
          case PRISM18:
          case PRISM20:
          case PRISM21:
            {
              libmesh_assert_less (i, 6);

              // Compute prism shape functions as a tensor-product
              // of a triangle and an edge

              VectorValue<Scalar> p2d(p(0),p(1));
              Scalar p1d = p(2);

              //                                0  1  2  3  4  5
              constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1});
              constexpr auto i1 = detail::pack_table<3>({0, 1, 2, 0, 1, 2});

              return (fe_lagrange_2D_shape<LAGRANGE>(TRI3,  FIRST, i1[i], p2d)*
                      fe_lagrange_1D_linear_shape(i0[i], p1d));
            }

            // linear pyramid shape functions
          case PYRAMID5:
          case PYRAMID13:
          case PYRAMID14:
          case PYRAMID18:
            {
              libmesh_assert_less (i, 5);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);
              const Real eps  = 1.e-35;

              switch(i)
                {
                case 0:
                  return .25*(zeta + xi - 1.)*(zeta + eta - 1.)/((1. - zeta) + eps);

                case 1:
                  return .25*(zeta - xi - 1.)*(zeta + eta - 1.)/((1. - zeta) + eps);

                case 2:
                  return .25*(zeta - xi - 1.)*(zeta - eta - 1.)/((1. - zeta) + eps);

                case 3:
                  return .25*(zeta + xi - 1.)*(zeta - eta - 1.)/((1. - zeta) + eps);

                case 4:
                  return zeta;

                default:
                  libmesh_error_msg("Invalid i = " << i);
                }
            }


          default:
            libmesh_error_msg("ERROR: Unsupported 3D element type!: " << Utility::enum_to_string(type));
          }
      }


      // quadratic Lagrange shape functions
    case SECOND:
      {
        switch (type)
          {

            // serendipity hexahedral quadratic shape functions
          case HEX20:
            {
              libmesh_assert_less (i, 20);
              return libMesh::detail::fe_lagrange_hex20_shape(i, p(0), p(1), p(2));
            }

            // triquadratic hexahedral shape functions
          case HEX8:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case HEX27:
            {
              libmesh_assert_less (i, 27);

              return libMesh::detail::fe_lagrange_hex27_shape(i, p(0), p(1), p(2));
            }

            // quadratic tetrahedral shape functions
          case TET4:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case TET10:
            libmesh_assert_less (i, 10);
            libmesh_fallthrough();
          case TET14:
            {
              libmesh_assert_less (i, 14);
              return libMesh::detail::fe_lagrange_tet10_shape(i, p(0), p(1), p(2));
            }

            // "serendipity" prism
          case PRISM15:
            {
              libmesh_assert_less (i, 15);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);

              switch(i)
                {
                case 0:
                  return (1. - zeta)*(xi + eta - 1.)*(xi + eta + 0.5*zeta);

                case 1:
                  return (1. - zeta)*xi*(xi - 1. - 0.5*zeta);

                case 2: // phi1 with xi <- eta
                  return (1. - zeta)*eta*(eta - 1. - 0.5*zeta);

                case 3: // phi0 with zeta <- (-zeta)
                  return (1. + zeta)*(xi + eta - 1.)*(xi + eta - 0.5*zeta);

                case 4: // phi1 with zeta <- (-zeta)
                  return (1. + zeta)*xi*(xi - 1. + 0.5*zeta);

                case 5: // phi4 with xi <- eta
                  return (1. + zeta)*eta*(eta - 1. + 0.5*zeta);

                case 6:
                  return 2.*(1. - zeta)*xi*(1. - xi - eta);

                case 7:
                  return 2.*(1. - zeta)*xi*eta;

                case 8:
                  return 2.*(1. - zeta)*eta*(1. - xi - eta);

                case 9:
                  return (1. - zeta)*(1. + zeta)*(1. - xi - eta);

                case 10:
                  return (1. - zeta)*(1. + zeta)*xi;

                case 11: // phi10 with xi <-> eta
                  return (1. - zeta)*(1. + zeta)*eta;

                case 12: // phi6 with zeta <- (-zeta)
                  return 2.*(1. + zeta)*xi*(1. - xi - eta);

                case 13: // phi7 with zeta <- (-zeta)
                  return 2.*(1. + zeta)*xi*eta;

                case 14: // phi8 with zeta <- (-zeta)
                  return 2.*(1. + zeta)*eta*(1. - xi - eta);

                default:
                  libmesh_error_msg("Invalid i = " << i);
                }
            }

            // quadratic prism shape functions
          case PRISM6:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case PRISM18:
          case PRISM20:
          case PRISM21:
            {
              libmesh_assert_less (i, 18);

              // Compute prism shape functions as a tensor-product
              // of a triangle and an edge

              VectorValue<Scalar> p2d(p(0),p(1));
              Scalar p1d = p(2);

              //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
              constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2});
              constexpr auto i1 = detail::pack_table<3>({0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5});

              return (fe_lagrange_2D_shape<LAGRANGE>(TRI6,  SECOND, i1[i], p2d)*
                      fe_lagrange_1D_quadratic_shape(i0[i], p1d));
            }

            // G. Bedrosian, "Shape functions and integration formulas for
            // three-dimensional finite element analysis", Int. J. Numerical
            // Methods Engineering, vol 35, p. 95-108, 1992.
          case PYRAMID13:
            {
              libmesh_assert_less (i, 13);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);
              const Real eps  = 1.e-35;

              // Denominators are perturbed by epsilon to avoid
              // divide-by-zero issues.
              Scalar den = (1. - zeta + eps);

              switch(i)
                {
                case 0:
                  return 0.25*(-xi - eta - 1.)*((1. - xi)*(1. - eta) - zeta + xi*eta*zeta/den);

                case 1:
                  return 0.25*(-eta + xi - 1.)*((1. + xi)*(1. - eta) - zeta - xi*eta*zeta/den);

                case 2:
                  return 0.25*(xi + eta - 1.)*((1. + xi)*(1. + eta) - zeta + xi*eta*zeta/den);

                case 3:
                  return 0.25*(eta - xi - 1.)*((1. - xi)*(1. + eta) - zeta - xi*eta*zeta/den);

                case 4:
                  return zeta*(2.*zeta - 1.);

                case 5:
                  return 0.5*(1. + xi - zeta)*(1. - xi - zeta)*(1. - eta - zeta)/den;

                case 6:
                  return 0.5*(1. + eta - zeta)*(1. - eta - zeta)*(1. + xi - zeta)/den;

                case 7:
                  return 0.5*(1. + xi - zeta)*(1. - xi - zeta)*(1. + eta - zeta)/den;

                case 8:
                  return 0.5*(1. + eta - zeta)*(1. - eta - zeta)*(1. - xi - zeta)/den;

                case 9:
                  return zeta*(1. - xi - zeta)*(1. - eta - zeta)/den;

                case 10:
                  return zeta*(1. + xi - zeta)*(1. - eta - zeta)/den;

                case 11:
                  return zeta*(1. + eta - zeta)*(1. + xi - zeta)/den;

                case 12:
                  return zeta*(1. - xi - zeta)*(1. + eta - zeta)/den;

                default:
                  libmesh_error_msg("Invalid i = " << i);
                }
            }

            // Quadratic shape functions, as defined in R. Graglia, "Higher order
            // bases on pyramidal elements", IEEE Trans Antennas and Propagation,
            // vol 47, no 5, May 1999.
          case PYRAMID5:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case PYRAMID14:
          case PYRAMID18:
            {
              libmesh_assert_less (i, 14);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);
              const Real eps  = 1.e-35;

              // The "normalized coordinates" defined by Graglia.  These are
              // the planes which define the faces of the pyramid.
              Scalar
                p1 = 0.5*(1. - eta - zeta), // back
                p2 = 0.5*(1. + xi  - zeta), // left
                p3 = 0.5*(1. + eta - zeta), // front
                p4 = 0.5*(1. - xi  - zeta); // right

              // Denominators are perturbed by epsilon to avoid
              // divide-by-zero issues.
              Scalar
                den = (-1. + zeta + eps),
                den2 = den*den;

              switch(i)
                {
                case 0:
                  return p4*p1*(xi*eta - zeta + zeta*zeta)/den2;

                case 1:
                  return -p1*p2*(xi*eta + zeta - zeta*zeta)/den2;

                case 2:
                  return p2*p3*(xi*eta - zeta + zeta*zeta)/den2;

                case 3:
                  return -p3*p4*(xi*eta + zeta - zeta*zeta)/den2;

                case 4:
                  return zeta*(2.*zeta - 1.);

                case 5:
                  return -4.*p2*p1*p4*eta/den2;

                case 6:
                  return 4.*p1*p2*p3*xi/den2;

                case 7:
                  return 4.*p2*p3*p4*eta/den2;

                case 8:
                  return -4.*p3*p4*p1*xi/den2;

                case 9:
                  return -4.*p1*p4*zeta/den;

                case 10:
                  return -4.*p2*p1*zeta/den;

                case 11:
                  return -4.*p3*p2*zeta/den;

                case 12:
                  return -4.*p4*p3*zeta/den;

                case 13:
                  return 16.*p1*p2*p3*p4/den2;

                default:
                  libmesh_error_msg("Invalid i = " << i);
                }
            }


          default:
            libmesh_error_msg("ERROR: Unsupported 3D element type!: " << Utility::enum_to_string(type));
          }
      }

    case THIRD:
      {
        switch (type)
          {
            // quadratic Lagrange shape functions with cubic bubbles
          case PRISM20:
            {
              libmesh_assert_less (i, 20);

              // Compute Prism21 shape functions as a tensor-product
              // of a triangle and an edge, then redistribute the
              // central bubble function over the other Tri6 nodes
              // around it (in a way consistent with the Tri7 shape
              // function definitions).
              //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
              constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 1});

              VectorValue<Scalar> p2d(p(0),p(1));
              Scalar p1d = p(2);

              const Scalar mainval = detail::fe_lagrange_prism21_shape(i, p);

              if (i0[i] != 2)
                return mainval;

              const Scalar bubbleval =
                fe_lagrange_2D_shape<LAGRANGE>(TRI7, THIRD, 6, p2d) *
                fe_lagrange_1D_quadratic_shape(2, p1d);

              if (i < 12) // vertices
                return mainval - bubbleval / 9;

              return mainval + bubbleval * (Real(4) / 9);
            }

            // quadratic Lagrange shape functions with cubic bubbles
          case PRISM21:
            {
              libmesh_assert_less (i, 21);
              return detail::fe_lagrange_prism21_shape(i, p);
            }

            // Weird rational shape functions with weirder bubbles...
          case PYRAMID18:
            {
              libmesh_assert_less (i, 18);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);
              const Real eps  = 1.e-35;

              // The "normalized coordinates" defined by Graglia.  These are
              // the planes which define the faces of the pyramid.
              const Scalar
                p1 = 0.5*(1. - eta - zeta), // back
                p2 = 0.5*(1. + xi  - zeta), // left
                p3 = 0.5*(1. + eta - zeta), // front
                p4 = 0.5*(1. - xi  - zeta); // right

              // Denominators are perturbed by epsilon to avoid
              // divide-by-zero issues.
              const Scalar
                den = (-1. + zeta + eps),
                den2 = den*den;

              // Bubble functions on triangular sides.  We actually
              // have a degree of freedom to play with here, and I'm
              // not certain how best to use it, so let's leave it
              // as a variable in case we figure that out later.
              constexpr Real alpha = 0.5;
              const Scalar
                bub_f1 = ((1-alpha)*(1-zeta) + alpha*(-eta)),
                bub_f2 = ((1-alpha)*(1-zeta) + alpha*(xi)),
                bub_f3 = ((1-alpha)*(1-zeta) + alpha*(eta)),
                bub_f4 = ((1-alpha)*(1-zeta) + alpha*(-xi));

              const Scalar
                bub1 = bub_f1*p1*p2*p4*zeta/den2,
                bub2 = bub_f2*p1*p2*p3*zeta/den2,
                bub3 = bub_f3*p2*p3*p4*zeta/den2,
                bub4 = bub_f4*p1*p3*p4*zeta/den2;

              switch(i)
                {
                case 0:
                  return p4*p1*(xi*eta - zeta + zeta*zeta)/den2 + 3*(bub1+bub4);

                case 1:
                  return -p1*p2*(xi*eta + zeta - zeta*zeta)/den2 + 3*(bub1+bub2);

                case 2:
                  return p2*p3*(xi*eta - zeta + zeta*zeta)/den2 + 3*(bub2+bub3);

                case 3:
                  return -p3*p4*(xi*eta + zeta - zeta*zeta)/den2 + 3*(bub3+bub4);

                case 4:
                  return zeta*(2.*zeta - 1.) + 3*(bub1+bub2+bub3+bub4);

                case 5:
                  return -4.*p2*p1*p4*eta/den2 - 12*bub1;

                case 6:
                  return 4.*p1*p2*p3*xi/den2 - 12*bub2;

                case 7:
                  return 4.*p2*p3*p4*eta/den2 - 12*bub3;

                case 8:
                  return -4.*p3*p4*p1*xi/den2 - 12*bub4;

                case 9:
                  return -4.*p1*p4*zeta/den - 12*(bub1+bub4);

                case 10:
                  return -4.*p2*p1*zeta/den - 12*(bub1+bub2);

                case 11:
                  return -4.*p3*p2*zeta/den - 12*(bub2+bub3);

                case 12:
                  return -4.*p4*p3*zeta/den - 12*(bub3+bub4);

                case 13:
                  return 16.*p1*p2*p3*p4/den2;

                case 14:
                  return 27*bub1;

                case 15:
                  return 27*bub2;

                case 16:
                  return 27*bub3;

                case 17:
                  return 27*bub4;

                default:
                  libmesh_error_msg("Invalid i = " << i);
                }
            }

            // quadratic Lagrange shape functions with cubic bubbles
          case TET14:
            {
              libmesh_assert_less (i, 14);
              return libMesh::detail::fe_lagrange_tet14_shape(i, p(0), p(1), p(2));
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 3D element type!: " << Utility::enum_to_string(type));
          }
      }

      // unsupported order
    default:
      libmesh_error_msg("ERROR: Unsupported 3D FE order!: " << order);
    }

#else // LIBMESH_DIM != 3
  libmesh_ignore(type, order, i, p);
  libmesh_not_implemented();
#endif
}






template <FEFamily T, typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_3D_shape_deriv(const ElemType type,
                                  const Order order,
                                  const unsigned int i,
                                  const unsigned int j,
                                  const TypeVector<Scalar> & p)
{
#if LIBMESH_DIM == 3

  libmesh_assert_less (j, 3);

  switch (order)
    {
      // linear Lagrange shape functions
    case FIRST:
      {
        switch (type)
          {
            // trilinear hexahedral shape functions
          case HEX8:
          case HEX20:
          case HEX27:
            {
              libmesh_assert_less (i, 8);

              return libMesh::detail::fe_lagrange_hex8_shape_deriv(i, j, p(0), p(1), p(2));
            }

            // linear tetrahedral shape functions
          case TET4:
          case TET10:
          case TET14:
            {
              libmesh_assert_less (i, 4);
              return libMesh::detail::fe_lagrange_tet4_shape_deriv(i, j);
            }

            // linear prism shape functions
          case PRISM6:
          case PRISM15:
          case PRISM18:
          case PRISM20:
          case PRISM21:
            {
              libmesh_assert_less (i, 6);

              // Compute prism shape functions as a tensor-product
              // of a triangle and an edge

              VectorValue<Scalar> p2d(p(0),p(1));
              Scalar p1d = p(2);

              //                                0  1  2  3  4  5
              constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1});
              constexpr auto i1 = detail::pack_table<3>({0, 1, 2, 0, 1, 2});

              switch (j)
                {
                  // d()/dxi
                case 0:
                  return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI3,  FIRST, i1[i], 0, p2d)*
                          fe_lagrange_1D_linear_shape(i0[i], p1d));

                  // d()/deta
                case 1:
                  return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI3,  FIRST, i1[i], 1, p2d)*
                          fe_lagrange_1D_linear_shape(i0[i], p1d));

                  // d()/dzeta
                case 2:
                  return (fe_lagrange_2D_shape<LAGRANGE>(TRI3,  FIRST, i1[i], p2d)*
                          fe_lagrange_1D_linear_shape_deriv(i0[i], 0, p1d));

                default:
                  libmesh_error_msg("Invalid shape function derivative j = " << j);
                }
            }

            // linear pyramid shape functions
          case PYRAMID5:
          case PYRAMID13:
          case PYRAMID14:
          case PYRAMID18:
            {
              libmesh_assert_less (i, 5);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);
              const Real eps  = 1.e-35;

              switch (j)
                {
                  // d/dxi
                case 0:
                  switch(i)
                    {
                    case 0:
                      return  .25*(zeta + eta - 1.)/((1. - zeta) + eps);

                    case 1:
                      return -.25*(zeta + eta - 1.)/((1. - zeta) + eps);

                    case 2:
                      return -.25*(zeta - eta - 1.)/((1. - zeta) + eps);

                    case 3:
                      return  .25*(zeta - eta - 1.)/((1. - zeta) + eps);

                    case 4:
                      return 0;

                    default:
                      libmesh_error_msg("Invalid i = " << i);
                    }


                  // d/deta
                case 1:
                  switch(i)
                    {
                    case 0:
                      return  .25*(zeta + xi - 1.)/((1. - zeta) + eps);

                    case 1:
                      return  .25*(zeta - xi - 1.)/((1. - zeta) + eps);

                    case 2:
                      return -.25*(zeta - xi - 1.)/((1. - zeta) + eps);

                    case 3:
                      return -.25*(zeta + xi - 1.)/((1. - zeta) + eps);

                    case 4:
                      return 0;

                    default:
                      libmesh_error_msg("Invalid i = " << i);
                    }


                  // d/dzeta
                case 2:
                  {
                    // We computed the derivatives with general eps and
                    // then let eps tend to zero in the numerators...
                    Scalar
                      num = zeta*(2. - zeta) - 1.,
                      den = (1. - zeta + eps)*(1. - zeta + eps);

                    switch(i)
                      {
                      case 0:
                      case 2:
                        return .25*(num + xi*eta)/den;

                      case 1:
                      case 3:
                        return .25*(num - xi*eta)/den;

                      case 4:
                        return 1.;

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                default:
                  libmesh_error_msg("Invalid j = " << j);
                }
            }


          default:
            libmesh_error_msg("ERROR: Unsupported 3D element type!: " << Utility::enum_to_string(type));
          }
      }


      // quadratic Lagrange shape functions
    case SECOND:
      {
        switch (type)
          {

            // serendipity hexahedral quadratic shape functions
          case HEX20:
            {
              libmesh_assert_less (i, 20);
              return libMesh::detail::fe_lagrange_hex20_shape_deriv(i, j, p(0), p(1), p(2));
            }

            // triquadratic hexahedral shape functions
          case HEX8:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case HEX27:
            {
              libmesh_assert_less (i, 27);

              return libMesh::detail::fe_lagrange_hex27_shape_deriv(i, j, p(0), p(1), p(2));
            }

            // quadratic tetrahedral shape functions
          case TET4:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case TET10:
          case TET14:
            {
              libmesh_assert_less (i, 10);
              return libMesh::detail::fe_lagrange_tet10_shape_deriv(i, j, p(0), p(1), p(2));
            }


            // "serendipity" prism
          case PRISM15:
            {
              libmesh_assert_less (i, 15);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);

              switch (j)
                {
                  // d()/dxi
                case 0:
                  {
                    switch(i)
                      {
                      case 0:
                        return (2.*xi + 2.*eta + 0.5*zeta - 1.)*(1. - zeta);
                      case 1:
                        return (2.*xi - 1. - 0.5*zeta)*(1. - zeta);
                      case 2:
                        return 0.;
                      case 3:
                        return (2.*xi + 2.*eta - 0.5*zeta - 1.)*(1. + zeta);
                      case 4:
                        return (2.*xi - 1. + 0.5*zeta)*(1. + zeta);
                      case 5:
                        return 0.;
                      case 6:
                        return (4.*xi + 2.*eta - 2.)*(zeta - 1.);
                      case 7:
                        return -2.*(zeta - 1.)*eta;
                      case 8:
                        return 2.*(zeta - 1.)*eta;
                      case 9:
                        return (zeta - 1.)*(1. + zeta);
                      case 10:
                        return (1. - zeta)*(1. + zeta);
                      case 11:
                        return 0.;
                      case 12:
                        return (-4.*xi - 2.*eta + 2.)*(1. + zeta);
                      case 13:
                        return 2.*(1. + zeta)*eta;
                      case 14:
                        return -2.*(1. + zeta)*eta;
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                  // d()/deta
                case 1:
                  {
                    switch(i)
                      {
                      case 0:
                        return (2.*xi + 2.*eta + 0.5*zeta - 1.)*(1. - zeta);
                      case 1:
                        return 0.;
                      case 2:
                        return (2.*eta - 1. - 0.5*zeta)*(1. - zeta);
                      case 3:
                        return (2.*xi + 2.*eta - 0.5*zeta - 1.)*(1. + zeta);
                      case 4:
                        return 0.;
                      case 5:
                        return (2.*eta - 1. + 0.5*zeta)*(1. + zeta);
                      case 6:
                        return 2.*(zeta - 1.)*xi;
                      case 7:
                        return 2.*(1. - zeta)*xi;
                      case 8:
                        return (2.*xi + 4.*eta - 2.)*(zeta - 1.);
                      case 9:
                        return (zeta - 1.)*(1. + zeta);
                      case 10:
                        return 0.;
                      case 11:
                        return (1. - zeta)*(1. + zeta);
                      case 12:
                        return -2.*(1. + zeta)*xi;
                      case 13:
                        return 2.*(1. + zeta)*xi;
                      case 14:
                        return (-2.*xi - 4.*eta + 2.)*(1. + zeta);

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                  // d()/dzeta
                case 2:
                  {
                    switch(i)
                      {
                      case 0:
                        return (-xi - eta - zeta + 0.5)*(xi + eta - 1.);
                      case 1:
                        return -0.5*xi*(2.*xi - 1. - 2.*zeta);
                      case 2:
                        return -0.5*eta*(2.*eta - 1. - 2.*zeta);
                      case 3:
                        return (xi + eta - zeta - 0.5)*(xi + eta - 1.);
                      case 4:
                        return 0.5*xi*(2.*xi - 1. + 2.*zeta);
                      case 5:
                        return 0.5*eta*(2.*eta - 1. + 2.*zeta);
                      case 6:
                        return 2.*xi*(xi + eta - 1.);
                      case 7:
                        return -2.*xi*eta;
                      case 8:
                        return 2.*eta*(xi + eta - 1.);
                      case 9:
                        return 2.*zeta*(xi + eta - 1.);
                      case 10:
                        return -2.*xi*zeta;
                      case 11:
                        return -2.*eta*zeta;
                      case 12:
                        return 2.*xi*(1. - xi - eta);
                      case 13:
                        return 2.*xi*eta;
                      case 14:
                        return 2.*eta*(1. - xi - eta);

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                default:
                  libmesh_error_msg("Invalid j = " << j);
                }
            }



            // quadratic prism shape functions
          case PRISM6:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case PRISM18:
          case PRISM20:
          case PRISM21:
            {
              libmesh_assert_less (i, 18);

              // Compute prism shape functions as a tensor-product
              // of a triangle and an edge

              VectorValue<Scalar> p2d(p(0),p(1));
              Scalar p1d = p(2);

              //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
              constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2});
              constexpr auto i1 = detail::pack_table<3>({0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5});

              switch (j)
                {
                  // d()/dxi
                case 0:
                  return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI6,  SECOND, i1[i], 0, p2d)*
                          fe_lagrange_1D_quadratic_shape(i0[i], p1d));

                  // d()/deta
                case 1:
                  return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI6,  SECOND, i1[i], 1, p2d)*
                          fe_lagrange_1D_quadratic_shape(i0[i], p1d));

                  // d()/dzeta
                case 2:
                  return (fe_lagrange_2D_shape<LAGRANGE>(TRI6,  SECOND, i1[i], p2d)*
                          fe_lagrange_1D_quadratic_shape_deriv(i0[i], 0, p1d));

                default:
                  libmesh_error_msg("Invalid shape function derivative j = " << j);
                }
            }

            // G. Bedrosian, "Shape functions and integration formulas for
            // three-dimensional finite element analysis", Int. J. Numerical
            // Methods Engineering, vol 35, p. 95-108, 1992.
          case PYRAMID13:
            {
              libmesh_assert_less (i, 13);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);
              const Real eps  = 1.e-35;

              // Denominators are perturbed by epsilon to avoid
              // divide-by-zero issues.
              Scalar
                den = (-1. + zeta + eps),
                den2 = den*den,
                xi2 = xi*xi,
                eta2 = eta*eta,
                zeta2 = zeta*zeta,
                zeta3 = zeta2*zeta;

              switch (j)
                {
                  // d/dxi
                case 0:
                  switch(i)
                    {
                    case 0:
                      return 0.25*(-zeta - eta + 2.*eta*zeta - 2.*xi + 2.*zeta*xi + 2.*eta*xi + zeta2 + eta2)/den;

                    case 1:
                      return -0.25*(-zeta - eta + 2.*eta*zeta + 2.*xi - 2.*zeta*xi - 2.*eta*xi + zeta2 + eta2)/den;

                    case 2:
                      return -0.25*(-zeta + eta - 2.*eta*zeta + 2.*xi - 2.*zeta*xi + 2.*eta*xi + zeta2 + eta2)/den;

                    case 3:
                      return 0.25*(-zeta + eta - 2.*eta*zeta - 2.*xi + 2.*zeta*xi - 2.*eta*xi + zeta2 + eta2)/den;

                    case 4:
                      return 0.;

                    case 5:
                      return -(-1. + eta + zeta)*xi/den;

                    case 6:
                      return 0.5*(-1. + eta + zeta)*(1. + eta - zeta)/den;

                    case 7:
                      return (1. + eta - zeta)*xi/den;

                    case 8:
                      return -0.5*(-1. + eta + zeta)*(1. + eta - zeta)/den;

                    case 9:
                      return -(-1. + eta + zeta)*zeta/den;

                    case 10:
                      return (-1. + eta + zeta)*zeta/den;

                    case 11:
                      return -(1. + eta - zeta)*zeta/den;

                    case 12:
                      return (1. + eta - zeta)*zeta/den;

                    default:
                      libmesh_error_msg("Invalid i = " << i);
                    }

                  // d/deta
                case 1:
                  switch(i)
                    {
                    case 0:
                      return 0.25*(-zeta - 2.*eta + 2.*eta*zeta - xi + 2.*zeta*xi + 2.*eta*xi + zeta2 + xi2)/den;

                    case 1:
                      return -0.25*(zeta + 2.*eta - 2.*eta*zeta - xi + 2.*zeta*xi + 2.*eta*xi - zeta2 - xi2)/den;

                    case 2:
                      return -0.25*(-zeta + 2.*eta - 2.*eta*zeta + xi - 2.*zeta*xi + 2.*eta*xi + zeta2 + xi2)/den;

                    case 3:
                      return 0.25*(zeta - 2.*eta + 2.*eta*zeta + xi - 2.*zeta*xi + 2.*eta*xi - zeta2 - xi2)/den;

                    case 4:
                      return 0.;

                    case 5:
                      return -0.5*(-1. + xi + zeta)*(1. + xi - zeta)/den;

                    case 6:
                      return (1. + xi - zeta)*eta/den;

                    case 7:
                      return 0.5*(-1. + xi + zeta)*(1. + xi - zeta)/den;

                    case 8:
                      return -(-1. + xi + zeta)*eta/den;

                    case 9:
                      return -(-1. + xi + zeta)*zeta/den;

                    case 10:
                      return (1. + xi - zeta)*zeta/den;

                    case 11:
                      return -(1. + xi - zeta)*zeta/den;

                    case 12:
                      return (-1. + xi + zeta)*zeta/den;

                    default:
                      libmesh_error_msg("Invalid i = " << i);
                    }

                  // d/dzeta
                case 2:
                  {
                    switch(i)
                      {
                      case 0:
                        return -0.25*(xi + eta + 1.)*(-1. + 2.*zeta - zeta2 + eta*xi)/den2;

                      case 1:
                        return 0.25*(eta - xi + 1.)*(1. - 2.*zeta + zeta2 + eta*xi)/den2;

                      case 2:
                        return 0.25*(xi + eta - 1.)*(-1. + 2.*zeta - zeta2 + eta*xi)/den2;

                      case 3:
                        return -0.25*(eta - xi - 1.)*(1. - 2.*zeta + zeta2 + eta*xi)/den2;

                      case 4:
                        return 4.*zeta - 1.;

                      case 5:
                        return 0.5*(-2 + eta + 6.*zeta + eta*xi2 + eta*zeta2 - 6.*zeta2 + 2.*zeta3 - 2.*eta*zeta)/den2;

                      case 6:
                        return -0.5*(2 - 6.*zeta + xi + xi*zeta2 + eta2*xi + 6.*zeta2 - 2.*zeta3 - 2.*zeta*xi)/den2;

                      case 7:
                        return -0.5*(2 + eta - 6.*zeta + eta*xi2 + eta*zeta2 + 6.*zeta2 - 2.*zeta3 - 2.*eta*zeta)/den2;

                      case 8:
                        return 0.5*(-2 + 6.*zeta + xi + xi*zeta2 + eta2*xi - 6.*zeta2 + 2.*zeta3 - 2.*zeta*xi)/den2;

                      case 9:
                        return (1. - eta - 4.*zeta - xi - xi*zeta2 - eta*zeta2 + eta*xi + 5.*zeta2 - 2.*zeta3 + 2.*eta*zeta + 2.*zeta*xi)/den2;

                      case 10:
                        return -(-1. + eta + 4.*zeta - xi - xi*zeta2 + eta*zeta2 + eta*xi - 5.*zeta2 + 2.*zeta3 - 2.*eta*zeta + 2.*zeta*xi)/den2;

                      case 11:
                        return (1. + eta - 4.*zeta + xi + xi*zeta2 + eta*zeta2 + eta*xi + 5.*zeta2 - 2.*zeta3 - 2.*eta*zeta - 2.*zeta*xi)/den2;

                      case 12:
                        return -(-1. - eta + 4.*zeta + xi + xi*zeta2 - eta*zeta2 + eta*xi - 5.*zeta2 + 2.*zeta3 + 2.*eta*zeta - 2.*zeta*xi)/den2;

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                default:
                  libmesh_error_msg("Invalid j = " << j);
                }
            }

            // Quadratic shape functions, as defined in R. Graglia, "Higher order
            // bases on pyramidal elements", IEEE Trans Antennas and Propagation,
            // vol 47, no 5, May 1999.
          case PYRAMID5:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case PYRAMID14:
          case PYRAMID18:
            return detail::fe_lagrange_pyramid14_shape_deriv(i, j, p);


          default:
            libmesh_error_msg("ERROR: Unsupported 3D element type!: " << Utility::enum_to_string(type));
          }
      }

    case THIRD:
      {
        switch (type)
          {
            // quadratic Lagrange shape functions with a cubic bubble
          case TET14:
            {
              libmesh_assert_less (i, 14);
              return libMesh::detail::fe_lagrange_tet14_shape_deriv(i, j, p(0), p(1), p(2));
            }

          case PRISM20:
            {
              libmesh_assert_less (i, 20);

              //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
              constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 1, 2});

              VectorValue<Scalar> p2d(p(0),p(1));
              Scalar p1d = p(2);

              const Scalar mainval = detail::fe_lagrange_prism21_shape_deriv(i, j, p);

              if (i0[i] != 2)
                return mainval;

              Scalar bubbleval = 0;

              switch (j)
                {
                  // d()/dxi
                case 0:
                  bubbleval =
                    fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI7, THIRD, 6, 0, p2d)*
                    fe_lagrange_1D_quadratic_shape(2, p1d);
                  break;

                  // d()/deta
                case 1:
                  bubbleval =
                    fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI7, THIRD, 6, 1, p2d)*
                    fe_lagrange_1D_quadratic_shape(2, p1d);
                  break;

                  // d()/dzeta
                case 2:
                  bubbleval =
                    fe_lagrange_2D_shape<LAGRANGE>(TRI7, THIRD, 6, p2d)*
                    fe_lagrange_1D_quadratic_shape_deriv(2, 0, p1d);
                  break;

                default:
                  libmesh_error_msg("Invalid shape function derivative j = " << j);
                }

              if (i < 12) // vertices
                return mainval - bubbleval / 9;

              return mainval + bubbleval * (Real(4) / 9);
            }

          case PRISM21:
            {
              libmesh_assert_less (i, 21);
              return detail::fe_lagrange_prism21_shape_deriv(i, j, p);
            }

          case PYRAMID18:
            return detail::fe_lagrange_pyramid18_shape_deriv(i, j, p);

          default:
            libmesh_error_msg("ERROR: Unsupported 3D element type!: " << Utility::enum_to_string(type));
          }
      }

      // unsupported order
    default:
      libmesh_error_msg("ERROR: Unsupported 3D FE order!: " << order);
    }

#else // LIBMESH_DIM != 3
  libmesh_ignore(type, order, i, j, p);
  libmesh_not_implemented();
#endif
}






#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

template <FEFamily T, typename Scalar>
LIBMESH_DEVICE_INLINE
Scalar fe_lagrange_3D_shape_second_deriv(const ElemType type,
                                         const Order order,
                                         const unsigned int i,
                                         const unsigned int j,
                                         const TypeVector<Scalar> & p)
{
#if LIBMESH_DIM == 3

  libmesh_assert_less (j, 6);

  switch (order)
    {
      // linear Lagrange shape functions
    case FIRST:
      {
        switch (type)
          {
            // Linear tets have all second derivatives = 0
          case TET4:
          case TET10:
          case TET14:
            {
              return 0.;
            }

            // The following elements use either tensor product or
            // rational basis functions, and therefore probably have
            // second derivatives, but we have not implemented them
            // yet...
          case PRISM6:
          case PRISM15:
          case PRISM18:
          case PRISM20:
          case PRISM21:
            {
              libmesh_assert_less (i, 6);

              // Compute prism shape functions as a tensor-product
              // of a triangle and an edge

              VectorValue<Scalar> p2d(p(0),p(1));
              Scalar p1d = p(2);

              //                                0  1  2  3  4  5
              constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1});
              constexpr auto i1 = detail::pack_table<3>({0, 1, 2, 0, 1, 2});

              switch (j)
                {
                  // All repeated second derivatives and the xi-eta derivative are zero on PRISMs
                case 0: // d^2()/dxi^2
                case 1: // d^2()/dxideta
                case 2: // d^2()/deta^2
                case 5: // d^2()/dzeta^2
                  {
                    return 0.;
                  }

                case 3: // d^2()/dxidzeta
                  return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI3,  FIRST, i1[i], 0, p2d)*
                          fe_lagrange_1D_linear_shape_deriv(i0[i], 0, p1d));

                case 4: // d^2()/detadzeta
                  return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI3,  FIRST, i1[i], 1, p2d)*
                          fe_lagrange_1D_linear_shape_deriv(i0[i], 0, p1d));

                default:
                  libmesh_error_msg("Invalid j = " << j);
                }
            }

          case PYRAMID5:
          case PYRAMID13:
          case PYRAMID14:
          case PYRAMID18:
            {
              libmesh_assert_less (i, 5);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);
              const Real eps  = 1.e-35;

              switch (j)
                {
                  // xi-xi and eta-eta derivatives are all zero for PYRAMID5.
                case 0: // d^2()/dxi^2
                case 2: // d^2()/deta^2
                  return 0.;

                case 1: // d^2()/dxideta
                  {
                    switch (i)
                      {
                      case 0:
                      case 2:
                        return 0.25/(1. - zeta + eps);
                      case 1:
                      case 3:
                        return -0.25/(1. - zeta + eps);
                      case 4:
                        return 0.;
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                case 3: // d^2()/dxidzeta
                  {
                    Scalar den = (1. - zeta + eps)*(1. - zeta + eps);

                    switch (i)
                      {
                      case 0:
                      case 2:
                        return 0.25*eta/den;
                      case 1:
                      case 3:
                        return -0.25*eta/den;
                      case 4:
                        return 0.;
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                case 4: // d^2()/detadzeta
                  {
                    Scalar den = (1. - zeta + eps)*(1. - zeta + eps);

                    switch (i)
                      {
                      case 0:
                      case 2:
                        return 0.25*xi/den;
                      case 1:
                      case 3:
                        return -0.25*xi/den;
                      case 4:
                        return 0.;
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                case 5: // d^2()/dzeta^2
                  {
                    Scalar den = (1. - zeta + eps)*(1. - zeta + eps)*(1. - zeta + eps);

                    switch (i)
                      {
                      case 0:
                      case 2:
                        return 0.5*xi*eta/den;
                      case 1:
                      case 3:
                        return -0.5*xi*eta/den;
                      case 4:
                        return 0.;
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                default:
                  libmesh_error_msg("Invalid j = " << j);
                }
            }

            // Trilinear shape functions on HEX8s have nonzero mixed second derivatives
          case HEX8:
          case HEX20:
          case HEX27:
            {
              libmesh_assert_less (i, 8);
              return libMesh::detail::fe_lagrange_hex8_shape_second_deriv(i, j, p(0), p(1), p(2));
            }

          // All second derivatives for piecewise-linear polyhedra are
          // zero or dirac-type distributions, but we can't put the
          // latter in a Real, so beware when integrating...
          case C0POLYHEDRON:
            {
              return 0.;
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 3D element type!: " << Utility::enum_to_string(type));
          }

      }

      // quadratic Lagrange shape functions
    case SECOND:
      {
        switch (type)
          {

            // serendipity hexahedral quadratic shape functions
          case HEX20:
            {
              libmesh_assert_less (i, 20);
              return libMesh::detail::fe_lagrange_hex20_shape_second_deriv(i, j, p(0), p(1), p(2));
            }

            // triquadratic hexahedral shape functions
          case HEX8:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case HEX27:
            {
              libmesh_assert_less (i, 27);
              return libMesh::detail::fe_lagrange_hex27_shape_second_deriv(i, j, p(0), p(1), p(2));
            }

            // quadratic tetrahedral shape functions
          case TET4:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case TET10:
          case TET14:
            {
              libmesh_assert_less (i, 10);
              return libMesh::detail::fe_lagrange_tet10_shape_second_deriv(i, j);
            }



            // "serendipity" prism
          case PRISM15:
            {
              libmesh_assert_less (i, 15);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);

              switch (j)
                {
                  // d^2()/dxi^2
                case 0:
                  {
                    switch(i)
                      {
                      case 0:
                      case 1:
                        return 2.*(1. - zeta);
                      case 2:
                      case 5:
                      case 7:
                      case 8:
                      case 9:
                      case 10:
                      case 11:
                      case 13:
                      case 14:
                        return 0.;
                      case 3:
                      case 4:
                        return 2.*(1. + zeta);
                      case 6:
                        return 4.*(zeta - 1);
                      case 12:
                        return -4.*(1. + zeta);
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                  // d^2()/dxideta
                case 1:
                  {
                    switch(i)
                      {
                      case 0:
                      case 7:
                        return 2.*(1. - zeta);
                      case 1:
                      case 2:
                      case 4:
                      case 5:
                      case 9:
                      case 10:
                      case 11:
                        return 0.;
                      case 3:
                      case 13:
                        return 2.*(1. + zeta);
                      case 6:
                      case 8:
                        return 2.*(zeta - 1.);
                      case 12:
                      case 14:
                        return -2.*(1. + zeta);
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                  // d^2()/deta^2
                case 2:
                  {
                    switch(i)
                      {
                      case 0:
                      case 2:
                        return 2.*(1. - zeta);
                      case 1:
                      case 4:
                      case 6:
                      case 7:
                      case 9:
                      case 10:
                      case 11:
                      case 12:
                      case 13:
                        return 0.;
                      case 3:
                      case 5:
                        return 2.*(1. + zeta);
                      case 8:
                        return 4.*(zeta - 1.);
                      case 14:
                        return -4.*(1. + zeta);
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                  // d^2()/dxidzeta
                case 3:
                  {
                    switch(i)
                      {
                      case 0:
                        return 1.5 - zeta - 2.*xi - 2.*eta;
                      case 1:
                        return 0.5 + zeta - 2.*xi;
                      case 2:
                      case 5:
                      case 11:
                        return 0.;
                      case 3:
                        return -1.5 - zeta + 2.*xi + 2.*eta;
                      case 4:
                        return -0.5 + zeta + 2.*xi;
                      case 6:
                        return 4.*xi + 2.*eta - 2.;
                      case 7:
                        return -2.*eta;
                      case 8:
                        return 2.*eta;
                      case 9:
                        return 2.*zeta;
                      case 10:
                        return -2.*zeta;
                      case 12:
                        return -4.*xi - 2.*eta + 2.;
                      case 13:
                        return 2.*eta;
                      case 14:
                        return -2.*eta;
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                  // d^2()/detadzeta
                case 4:
                  {
                    switch(i)
                      {
                      case 0:
                        return 1.5 - zeta - 2.*xi - 2.*eta;
                      case 1:
                      case 4:
                      case 10:
                        return 0.;
                      case 2:
                        return .5 + zeta - 2.*eta;
                      case 3:
                        return -1.5 - zeta + 2.*xi + 2.*eta;
                      case 5:
                        return -.5 + zeta + 2.*eta;
                      case 6:
                        return 2.*xi;
                      case 7:
                        return -2.*xi;
                      case 8:
                        return 2.*xi + 4.*eta - 2.;
                      case 9:
                        return 2.*zeta;
                      case 11:
                        return -2.*zeta;
                      case 12:
                        return -2.*xi;
                      case 13:
                        return 2.*xi;
                      case 14:
                        return -2.*xi - 4.*eta + 2.;
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                  // d^2()/dzeta^2
                case 5:
                  {
                    switch(i)
                      {
                      case 0:
                      case 3:
                        return 1. - xi - eta;
                      case 1:
                      case 4:
                        return xi;
                      case 2:
                      case 5:
                        return eta;
                      case 6:
                      case 7:
                      case 8:
                      case 12:
                      case 13:
                      case 14:
                        return 0.;
                      case 9:
                        return 2.*xi + 2.*eta - 2.;
                      case 10:
                        return -2.*xi;
                      case 11:
                        return -2.*eta;
                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                default:
                  libmesh_error_msg("Invalid j = " << j);
                }
            }



            // quadratic prism shape functions
          case PRISM6:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case PRISM18:
          case PRISM20:
          case PRISM21:
            {
              libmesh_assert_less (i, 18);

              // Compute prism shape functions as a tensor-product
              // of a triangle and an edge

              VectorValue<Scalar> p2d(p(0),p(1));
              Scalar p1d = p(2);

              //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
              constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2});
              constexpr auto i1 = detail::pack_table<3>({0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5});

              switch (j)
                {
                  // d^2()/dxi^2
                case 0:
                  return (fe_lagrange_2D_shape_second_deriv<LAGRANGE>(TRI6, SECOND, i1[i], 0, p2d)*
                          fe_lagrange_1D_quadratic_shape(i0[i], p1d));

                  // d^2()/dxideta
                case 1:
                  return (fe_lagrange_2D_shape_second_deriv<LAGRANGE>(TRI6, SECOND, i1[i], 1, p2d)*
                          fe_lagrange_1D_quadratic_shape(i0[i], p1d));

                  // d^2()/deta^2
                case 2:
                  return (fe_lagrange_2D_shape_second_deriv<LAGRANGE>(TRI6, SECOND, i1[i], 2, p2d)*
                          fe_lagrange_1D_quadratic_shape(i0[i], p1d));

                  // d^2()/dxidzeta
                case 3:
                  return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI6,  SECOND, i1[i], 0, p2d)*
                          fe_lagrange_1D_quadratic_shape_deriv(i0[i], 0, p1d));

                  // d^2()/detadzeta
                case 4:
                  return (fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI6,  SECOND, i1[i], 1, p2d)*
                          fe_lagrange_1D_quadratic_shape_deriv(i0[i], 0, p1d));

                  // d^2()/dzeta^2
                case 5:
                  return (fe_lagrange_2D_shape<LAGRANGE>(TRI6,  SECOND, i1[i], p2d)*
                          fe_lagrange_1D_quadratic_shape_second_deriv(i0[i], 0, p1d));

                default:
                  libmesh_error_msg("Invalid shape function derivative j = " << j);
                }
            }


            // Quadratic shape functions, as defined in R. Graglia, "Higher order
            // bases on pyramidal elements", IEEE Trans Antennas and Propagation,
            // vol 47, no 5, May 1999.
          case PYRAMID5:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case PYRAMID14:
          case PYRAMID18:
            return detail::fe_lagrange_pyramid14_shape_second_deriv(i, j, p);

            // G. Bedrosian, "Shape functions and integration formulas for
            // three-dimensional finite element analysis", Int. J. Numerical
            // Methods Engineering, vol 35, p. 95-108, 1992.
          case PYRAMID13:
            {
              libmesh_assert_less (i, 13);

              const Scalar xi   = p(0);
              const Scalar eta  = p(1);
              const Scalar zeta = p(2);
              const Real eps  = 1.e-35;

              // Denominators are perturbed by epsilon to avoid
              // divide-by-zero issues.
              Scalar
                den = (-1. + zeta + eps),
                den2 = den*den,
                den3 = den2*den,
                xi2 = xi*xi,
                eta2 = eta*eta,
                zeta2 = zeta*zeta,
                zeta3 = zeta2*zeta;

              switch (j)
                {
                case 0: // d^2()/dxi^2
                  {
                    switch(i)
                      {
                      case 0:
                      case 1:
                        return 0.5*(-1. + zeta + eta)/den;

                      case 2:
                      case 3:
                        return 0.5*(-1. + zeta - eta)/den;

                      case 4:
                      case 6:
                      case 8:
                      case 9:
                      case 10:
                      case 11:
                      case 12:
                        return 0.;

                      case 5:
                        return (1. - eta - zeta)/den;

                      case 7:
                        return (1. + eta - zeta)/den;

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                case 1: // d^2()/dxideta
                  {
                    switch(i)
                      {
                      case 0:
                        return  0.25*(-1. + 2.*zeta + 2.*xi + 2.*eta)/den;

                      case 1:
                        return -0.25*(-1. + 2.*zeta - 2.*xi + 2.*eta)/den;

                      case 2:
                        return -0.25*(1. - 2.*zeta + 2.*xi + 2.*eta)/den;

                      case 3:
                        return  0.25*(1. - 2.*zeta - 2.*xi + 2.*eta)/den;

                      case 4:
                        return 0.;

                      case 5:
                        return -xi/den;

                      case 6:
                        return eta/den;

                      case 7:
                        return xi/den;

                      case 8:
                        return -eta/den;

                      case 9:
                        return -zeta/den;

                      case 10:
                        return zeta/den;

                      case 11:
                        return -zeta/den;

                      case 12:
                        return zeta/den;

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }


                case 2: // d^2()/deta^2
                  {
                    switch(i)
                      {
                      case 0:
                      case 3:
                        return 0.5*(-1. + zeta + xi)/den;

                      case 1:
                      case 2:
                        return 0.5*(-1. + zeta - xi)/den;

                      case 4:
                      case 5:
                      case 7:
                      case 9:
                      case 10:
                      case 11:
                      case 12:
                        return 0.;

                      case 6:
                        return (1. + xi - zeta)/den;

                      case 8:
                        return (1. - xi - zeta)/den;

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }


                case 3: // d^2()/dxidzeta
                  {
                    switch(i)
                      {
                      case 0:
                        return -0.25*(-1. + 2.*zeta - zeta2 + eta + 2.*eta*xi + eta2)/den2;

                      case 1:
                        return 0.25*(-1. + 2.*zeta - zeta2 + eta - 2.*eta*xi + eta2)/den2;

                      case 2:
                        return 0.25*(-1. + 2.*zeta - zeta2 - eta + 2.*eta*xi + eta2)/den2;

                      case 3:
                        return -0.25*(-1. + 2.*zeta - zeta2 - eta - 2.*eta*xi + eta2)/den2;

                      case 4:
                        return 0.;

                      case 5:
                        return eta*xi/den2;

                      case 6:
                        return -0.5*(1. + zeta2 + eta2 - 2.*zeta)/den2;

                      case 7:
                        return -eta*xi/den2;

                      case 8:
                        return 0.5*(1. + zeta2 + eta2 - 2.*zeta)/den2;

                      case 9:
                        return (-1. - zeta2 + eta + 2.*zeta)/den2;

                      case 10:
                        return -(-1. - zeta2 + eta + 2.*zeta)/den2;

                      case 11:
                        return (1. + zeta2 + eta - 2.*zeta)/den2;

                      case 12:
                        return -(1. + zeta2 + eta - 2.*zeta)/den2;

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                case 4: // d^2()/detadzeta
                  {
                    switch(i)
                      {
                      case 0:
                        return -0.25*(-1. + 2.*zeta - zeta2 + xi + 2.*eta*xi + xi2)/den2;

                      case 1:
                        return 0.25*(1. - 2.*zeta + zeta2 + xi + 2.*eta*xi - xi2)/den2;

                      case 2:
                        return 0.25*(-1. + 2.*zeta - zeta2 - xi + 2.*eta*xi + xi2)/den2;

                      case 3:
                        return -0.25*(1. - 2.*zeta + zeta2 - xi + 2.*eta*xi - xi2)/den2;

                      case 4:
                        return 0.;

                      case 5:
                        return 0.5*(1. + xi2 + zeta2 - 2.*zeta)/den2;

                      case 6:
                        return -eta*xi/den2;

                      case 7:
                        return -0.5*(1. + xi2 + zeta2 - 2.*zeta)/den2;

                      case 8:
                        return eta*xi/den2;

                      case 9:
                        return (-1. - zeta2 + xi + 2.*zeta)/den2;

                      case 10:
                        return -(1. + zeta2 + xi - 2.*zeta)/den2;

                      case 11:
                        return (1. + zeta2 + xi - 2.*zeta)/den2;

                      case 12:
                        return -(-1. - zeta2 + xi + 2.*zeta)/den2;

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                case 5: // d^2()/dzeta^2
                  {
                    switch(i)
                      {
                      case 0:
                        return 0.5*(xi + eta + 1.)*eta*xi/den3;

                      case 1:
                        return -0.5*(eta - xi + 1.)*eta*xi/den3;

                      case 2:
                        return -0.5*(xi + eta - 1.)*eta*xi/den3;

                      case 3:
                        return 0.5*(eta - xi - 1.)*eta*xi/den3;

                      case 4:
                        return 4.;

                      case 5:
                        return -(1. - 3.*zeta + 3.*zeta2 - zeta3 + eta*xi2)/den3;

                      case 6:
                        return (-1. + 3.*zeta - 3.*zeta2 + zeta3 + eta2*xi)/den3;

                      case 7:
                        return (-1. + 3.*zeta - 3.*zeta2 + zeta3 + eta*xi2)/den3;

                      case 8:
                        return -(1. - 3.*zeta + 3.*zeta2 - zeta3 + eta2*xi)/den3;

                      case 9:
                        return -2.*(-1. + 3.*zeta - 3.*zeta2 + zeta3 + eta*xi)/den3;

                      case 10:
                        return 2.*(1. - 3.*zeta + 3.*zeta2 - zeta3 + eta*xi)/den3;

                      case 11:
                        return -2.*(-1. + 3.*zeta - 3.*zeta2 + zeta3 + eta*xi)/den3;

                      case 12:
                        return 2.*(1. - 3.*zeta + 3.*zeta2 - zeta3 + eta*xi)/den3;

                      default:
                        libmesh_error_msg("Invalid i = " << i);
                      }
                  }

                default:
                  libmesh_error_msg("Invalid j = " << j);
                }
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 3D element type!: " << Utility::enum_to_string(type));
          }
      }

    case THIRD:
      {
        switch (type)
          {
            // quadratic Lagrange shape functions with a cubic bubble
          case TET14:
            {
              libmesh_assert_less (i, 14);
              return libMesh::detail::fe_lagrange_tet14_shape_second_deriv(i, j, p(0), p(1), p(2));
            }

          case PRISM20:
            {
              libmesh_assert_less (i, 20);

              // Compute prism shape functions as a tensor-product
              // of a triangle and an edge

              VectorValue<Scalar> p2d(p(0),p(1));
              Scalar p1d = p(2);

              //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
              constexpr auto i0 = detail::pack_table<3>({0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 1});

              const Scalar mainval = detail::fe_lagrange_prism21_shape_second_deriv(i, j, p);

              if (i0[i] != 2)
                return mainval;

              Scalar bubbleval = 0;

              switch (j)
                {
                  // d^2()/dxi^2
                case 0:
                  bubbleval =
                    fe_lagrange_2D_shape_second_deriv<LAGRANGE>(TRI7, THIRD, 6, 0, p2d)*
                    fe_lagrange_1D_quadratic_shape(2, p1d);
                  break;

                  // d^2()/dxideta
                case 1:
                  bubbleval =
                    fe_lagrange_2D_shape_second_deriv<LAGRANGE>(TRI7, THIRD, 6, 1, p2d)*
                    fe_lagrange_1D_quadratic_shape(2, p1d);
                  break;

                  // d^2()/deta^2
                case 2:
                  bubbleval =
                    fe_lagrange_2D_shape_second_deriv<LAGRANGE>(TRI7, THIRD, 6, 2, p2d)*
                    fe_lagrange_1D_quadratic_shape(2, p1d);
                  break;

                  // d^2()/dxidzeta
                case 3:
                  bubbleval =
                    fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI7, THIRD, 6, 0, p2d)*
                    fe_lagrange_1D_quadratic_shape_deriv(2, 0, p1d);
                  break;

                  // d^2()/detadzeta
                case 4:
                  bubbleval =
                    fe_lagrange_2D_shape_deriv<LAGRANGE>(TRI7, THIRD, 6, 1, p2d)*
                    fe_lagrange_1D_quadratic_shape_deriv(2, 0, p1d);
                  break;

                  // d^2()/dzeta^2
                case 5:
                  bubbleval =
                    fe_lagrange_2D_shape<LAGRANGE>(TRI7, THIRD, 6, p2d)*
                    fe_lagrange_1D_quadratic_shape_second_deriv(2, 0, p1d);
                  break;

                default:
                  libmesh_error_msg("Invalid shape function derivative j = " << j);
                }

              if (i < 12) // vertices
                return mainval - bubbleval / 9;

              return mainval + bubbleval * (Real(4) / 9);
            }

          case PRISM21:
            {
              libmesh_assert_less (i, 21);
              return detail::fe_lagrange_prism21_shape_second_deriv(i, j, p);
            }

          case PYRAMID18:
            return detail::fe_lagrange_pyramid18_shape_second_deriv(i, j, p);

          default:
            libmesh_error_msg("ERROR: Unsupported 3D element type!: " << Utility::enum_to_string(type));
          }
      }

      // unsupported order
    default:
      libmesh_error_msg("ERROR: Unsupported 3D FE order!: " << order);
    }

#else // LIBMESH_DIM != 3
  libmesh_ignore(type, order, i, j, p);
  libmesh_not_implemented();
#endif
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES

} // namespace libMesh

#endif // LIBMESH_FE_LAGRANGE_SHAPE_3D_H
