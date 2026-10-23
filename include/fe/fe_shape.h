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

#ifndef LIBMESH_FE_SHAPE_H
#define LIBMESH_FE_SHAPE_H

// Local includes
#include "libmesh/libmesh_device.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/fe_lagrange_shape_1D.h"
#include "libmesh/fe_lagrange_shape_2D.h"
#include "libmesh/fe_lagrange_shape_3D.h"
#include "libmesh/fe_monomial_shape.h"
#include "libmesh/point.h"

// A shape function, or its derivative, of a run-time FE family,
// element type and order, for the families whose shapes are inline
// in headers.  FEInterface picks the same function through
// FE<dim,family>, which device code can't call, so device code, which
// only learns the FE type of a variable at run time, picks it here.
// Each takes the element's dimension, as FEInterface does, rather
// than looking it up from the type.

namespace libMesh
{

// FE<dim,T>'s shapes for T = LAGRANGE or L2_LAGRANGE, which share
// their shape functions; T only decides whether higher orders on
// first order elements, which only L2_LAGRANGE supports, assert.

template <FEFamily T>
LIBMESH_DEVICE_INLINE
Real fe_lagrange_shape(const unsigned int dim,
                       const ElemType type,
                       const Order order,
                       const unsigned int i,
                       const Point & p)
{
  switch (dim)
    {
    case 1:
      return fe_lagrange_1D_shape(order, i, p(0));
    case 2:
      return fe_lagrange_2D_shape<T>(type, order, i, p);
    case 3:
      return fe_lagrange_3D_shape<T>(type, order, i, p);
    default:
      libmesh_error_msg("Invalid dim = " << dim);
    }
}



template <FEFamily T>
LIBMESH_DEVICE_INLINE
Real fe_lagrange_shape_deriv(const unsigned int dim,
                             const ElemType type,
                             const Order order,
                             const unsigned int i,
                             const unsigned int j,
                             const Point & p)
{
  switch (dim)
    {
    case 1:
      return fe_lagrange_1D_shape_deriv(order, i, j, p(0));
    case 2:
      return fe_lagrange_2D_shape_deriv<T>(type, order, i, j, p);
    case 3:
      return fe_lagrange_3D_shape_deriv<T>(type, order, i, j, p);
    default:
      libmesh_error_msg("Invalid dim = " << dim);
    }
}



LIBMESH_DEVICE_INLINE
Real fe_shape(const unsigned int dim,
              const FEFamily family,
              const ElemType type,
              const Order order,
              const unsigned int i,
              const Point & p)
{
  switch (family)
    {
    case LAGRANGE:
      return fe_lagrange_shape<LAGRANGE>(dim, type, order, i, p);

    case L2_LAGRANGE:
      return fe_lagrange_shape<L2_LAGRANGE>(dim, type, order, i, p);

    case MONOMIAL:
      switch (dim)
        {
        case 1:
          return fe_monomial_1D_shape(order, i, p);
        case 2:
          return fe_monomial_2D_shape(order, i, p);
        case 3:
          return fe_monomial_3D_shape(order, i, p);
        default:
          libmesh_error_msg("Invalid dim = " << dim);
        }

    default:
      libmesh_error_msg("Unsupported family = " << Utility::enum_to_string(family));
    }
}



LIBMESH_DEVICE_INLINE
Real fe_shape_deriv(const unsigned int dim,
                    const FEFamily family,
                    const ElemType type,
                    const Order order,
                    const unsigned int i,
                    const unsigned int j,
                    const Point & p)
{
  switch (family)
    {
    case LAGRANGE:
      return fe_lagrange_shape_deriv<LAGRANGE>(dim, type, order, i, j, p);

    case L2_LAGRANGE:
      return fe_lagrange_shape_deriv<L2_LAGRANGE>(dim, type, order, i, j, p);

    case MONOMIAL:
      switch (dim)
        {
        case 1:
          return fe_monomial_1D_shape_deriv(order, i, j, p);
        case 2:
          return fe_monomial_2D_shape_deriv(order, i, j, p);
        case 3:
          return fe_monomial_3D_shape_deriv(order, i, j, p);
        default:
          libmesh_error_msg("Invalid dim = " << dim);
        }

    default:
      libmesh_error_msg("Unsupported family = " << Utility::enum_to_string(family));
    }
}



#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

template <FEFamily T>
LIBMESH_DEVICE_INLINE
Real fe_lagrange_shape_second_deriv(const unsigned int dim,
                                    const ElemType type,
                                    const Order order,
                                    const unsigned int i,
                                    const unsigned int j,
                                    const Point & p)
{
  switch (dim)
    {
    case 1:
      return fe_lagrange_1D_shape_second_deriv(order, i, j, p(0));
    case 2:
      return fe_lagrange_2D_shape_second_deriv<T>(type, order, i, j, p);
    case 3:
      return fe_lagrange_3D_shape_second_deriv<T>(type, order, i, j, p);
    default:
      libmesh_error_msg("Invalid dim = " << dim);
    }
}



LIBMESH_DEVICE_INLINE
Real fe_shape_second_deriv(const unsigned int dim,
                           const FEFamily family,
                           const ElemType type,
                           const Order order,
                           const unsigned int i,
                           const unsigned int j,
                           const Point & p)
{
  switch (family)
    {
    case LAGRANGE:
      return fe_lagrange_shape_second_deriv<LAGRANGE>(dim, type, order, i, j, p);

    case L2_LAGRANGE:
      return fe_lagrange_shape_second_deriv<L2_LAGRANGE>(dim, type, order, i, j, p);

    case MONOMIAL:
      switch (dim)
        {
        case 1:
          return fe_monomial_1D_shape_second_deriv(order, i, j, p);
        case 2:
          return fe_monomial_2D_shape_second_deriv(order, i, j, p);
        case 3:
          return fe_monomial_3D_shape_second_deriv(order, i, j, p);
        default:
          libmesh_error_msg("Invalid dim = " << dim);
        }

    default:
      libmesh_error_msg("Unsupported family = " << Utility::enum_to_string(family));
    }
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES

} // namespace libMesh

#endif // LIBMESH_FE_SHAPE_H
