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

#ifndef LIBMESH_FE_LAGRANGE_SHAPE_2D_H
#define LIBMESH_FE_LAGRANGE_SHAPE_2D_H

// Local includes
#include "libmesh/libmesh_device.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/fe_simplex_lagrange.h"
#include "libmesh/fe_tensor_product_lagrange.h"
#include "libmesh/point.h"

// The 2D LAGRANGE and L2_LAGRANGE shape functions, and their first
// and second derivatives, for every element type that defines them
// by its ElemType alone: everything but C0Polygon, whose shapes
// depend on the polygon itself.  FE<2,LAGRANGE> and FE<2,L2_LAGRANGE>
// call these, and they are inline here so that device code evaluates
// the same functions.

namespace libMesh
{

template <FEFamily T>
LIBMESH_DEVICE_INLINE
Real fe_lagrange_2D_shape(const ElemType type,
                          const Order order,
                          const unsigned int i,
                          const Point & p)
{
#if LIBMESH_DIM > 1

  switch (order)
    {
      // linear Lagrange shape functions
    case FIRST:
      {
        switch (type)
          {
          case QUAD4:
          case QUADSHELL4:
          case QUAD8:
          case QUADSHELL8:
          case QUAD9:
          case QUADSHELL9:
            {
              // Compute quad shape functions as a tensor-product
              libmesh_assert_less (i, 4);
              return libMesh::detail::fe_lagrange_quad4_shape(i, p(0), p(1));
            }

          case TRI3:
          case TRISHELL3:
          case TRI6:
          case TRI7:
            {
              libmesh_assert_less (i, 3);
              return libMesh::detail::fe_lagrange_tri3_shape(i, p(0), p(1));
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 2D element type: " << Utility::enum_to_string(type));
          }
      }


      // quadratic Lagrange shape functions
    case SECOND:
      {
        switch (type)
          {
          case QUAD8:
          case QUADSHELL8:
            {
              libmesh_assert_less (i, 8);
              return libMesh::detail::fe_lagrange_quad8_shape(i, p(0), p(1));
            }

          case QUAD4:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case QUAD9:
          case QUADSHELL9:
            {
              libmesh_assert_less (i, 9);
              return libMesh::detail::fe_lagrange_quad9_shape(i, p(0), p(1));
            }

          case TRI3:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case TRI6:
          case TRI7:
            {
              libmesh_assert_less (i, 6);
              return libMesh::detail::fe_lagrange_tri6_shape(i, p(0), p(1));
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 2D element type: " << Utility::enum_to_string(type));
          }
      }



      // "cubic" (one cubic bubble) Lagrange shape functions on TRI7
    case THIRD:
      {
        switch (type)
          {
          case TRI7:
            {
              libmesh_assert_less (i, 7);
              return libMesh::detail::fe_lagrange_tri7_shape(i, p(0), p(1));
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 2D element type: " << Utility::enum_to_string(type));
          }
      } // end case THIRD



      // unsupported order
    default:
      libmesh_error_msg("ERROR: Unsupported 2D FE order: " << order);
    }
#else // LIBMESH_DIM > 1
  libmesh_ignore(type, order, i, p);
  libmesh_not_implemented();
#endif
}






template <FEFamily T>
LIBMESH_DEVICE_INLINE
Real fe_lagrange_2D_shape_deriv(const ElemType type,
                                const Order order,
                                const unsigned int i,
                                const unsigned int j,
                                const Point & p)
{
#if LIBMESH_DIM > 1

  libmesh_assert_less (j, 2);

  switch (order)
    {
      // linear Lagrange shape functions
    case FIRST:
      {
        switch (type)
          {
          case QUAD4:
          case QUADSHELL4:
          case QUAD8:
          case QUADSHELL8:
          case QUAD9:
          case QUADSHELL9:
            {
              libmesh_assert_less (i, 4);
              return libMesh::detail::fe_lagrange_quad4_shape_deriv(i, j, p(0), p(1));
            }

          case TRI3:
          case TRISHELL3:
          case TRI6:
          case TRI7:
            {
              libmesh_assert_less (i, 3);
              return libMesh::detail::fe_lagrange_tri3_shape_deriv(i, j);
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 2D element type: " << Utility::enum_to_string(type));
          }
      }


      // quadratic Lagrange shape functions
    case SECOND:
      {
        switch (type)
          {
          case QUAD8:
          case QUADSHELL8:
            {
              libmesh_assert_less (i, 8);
              return libMesh::detail::fe_lagrange_quad8_shape_deriv(i, j, p(0), p(1));
            }

          case QUAD4:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case QUAD9:
          case QUADSHELL9:
            {
              libmesh_assert_less (i, 9);
              return libMesh::detail::fe_lagrange_quad9_shape_deriv(i, j, p(0), p(1));
            }

          case TRI3:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case TRI6:
          case TRI7:
            {
              libmesh_assert_less (i, 6);
              return libMesh::detail::fe_lagrange_tri6_shape_deriv(i, j, p(0), p(1));
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 2D element type: " << Utility::enum_to_string(type));
          }
      }



      // "cubic" (one cubic bubble) Lagrange shape functions on TRI7
    case THIRD:
      {
        switch (type)
          {
          case TRI7:
            {
              libmesh_assert_less (i, 7);
              return libMesh::detail::fe_lagrange_tri7_shape_deriv(i, j, p(0), p(1));
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 2D element type: " << Utility::enum_to_string(type));
          }
      } // end case THIRD



      // unsupported order
    default:
      libmesh_error_msg("ERROR: Unsupported 2D FE order: " << order);
    }
#else // LIBMESH_DIM > 1
  libmesh_ignore(type, order, i, j, p);
  libmesh_not_implemented();
#endif
}






#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

template <FEFamily T>
LIBMESH_DEVICE_INLINE
Real fe_lagrange_2D_shape_second_deriv(const ElemType type,
                                       const Order order,
                                       const unsigned int i,
                                       const unsigned int j,
                                       const Point & p)
{
#if LIBMESH_DIM > 1

  // j = 0 ==> d^2 phi / dxi^2
  // j = 1 ==> d^2 phi / dxi deta
  // j = 2 ==> d^2 phi / deta^2
  libmesh_assert_less (j, 3);

  switch (order)
    {
      // linear Lagrange shape functions
    case FIRST:
      {
        switch (type)
          {
          case QUAD4:
          case QUADSHELL4:
          case QUAD8:
          case QUADSHELL8:
          case QUAD9:
          case QUADSHELL9:
            {
              libmesh_assert_less (i, 4);
              return libMesh::detail::fe_lagrange_quad4_shape_second_deriv(i, j, p(0), p(1));
            }

          // All second derivatives for linear triangles are zero.
          case TRI3:
          case TRISHELL3:
          case TRI6:
          case TRI7:

          // All second derivatives for piecewise-linear polygons are
          // zero or dirac-type distributions, but we can't put the
          // latter in a Real, so beware when integrating...
          case C0POLYGON:
            {
              return 0.;
            }

          default:
            libmesh_error_msg("ERROR: Unsupported 2D element type: " << Utility::enum_to_string(type));

          } // end switch (type)
      } // end case FIRST


      // quadratic Lagrange shape functions
    case SECOND:
      {
        switch (type)
          {
          case QUAD8:
          case QUADSHELL8:
            {
              libmesh_assert_less (j, 3);
              return libMesh::detail::fe_lagrange_quad8_shape_second_deriv(i, j, p(0), p(1));
            } // end case QUAD8

          case QUAD4:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case QUAD9:
          case QUADSHELL9:
            {
              libmesh_assert_less (i, 9);
              return libMesh::detail::fe_lagrange_quad9_shape_second_deriv(i, j, p(0), p(1));
            } // end case QUAD9

          case TRI3:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case TRI6:
          case TRI7:
            {
              libmesh_assert_less (j, 3);
              return libMesh::detail::fe_lagrange_tri6_shape_second_deriv(i, j);
            }  // end case TRI6+TRI7

          default:
            libmesh_error_msg("ERROR: Unsupported 2D element type: " << Utility::enum_to_string(type));
          }
      } // end case SECOND



      // "cubic" (one cubic bubble) Lagrange shape functions on TRI7
    case THIRD:
      {
        switch (type)
          {
          case TRI3:
            libmesh_assert_msg(T == L2_LAGRANGE,
                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case TRI6:
          case TRI7:
            {
              libmesh_assert_less (j, 3);
              return libMesh::detail::fe_lagrange_tri7_shape_second_deriv(i, j, p(0), p(1));
            }  // end case TRI6+TRI7

          default:
            libmesh_error_msg("ERROR: Unsupported 2D element type: " << Utility::enum_to_string(type));
          }
      } // end case THIRD

      // unsupported order
    default:
      libmesh_error_msg("ERROR: Unsupported 2D FE order: " << order);

    } // end switch (order)

#else // LIBMESH_DIM > 1
  libmesh_ignore(type, order, i, j, p);
  libmesh_not_implemented();
#endif
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES

} // namespace libMesh

#endif // LIBMESH_FE_LAGRANGE_SHAPE_2D_H
