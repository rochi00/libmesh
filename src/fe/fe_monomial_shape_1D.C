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


// C++ includes

// Local includes
#include "libmesh/fe.h"
#include "libmesh/fe_monomial_shape.h"
#include "libmesh/elem.h"


namespace libMesh
{


LIBMESH_DEFAULT_VECTORIZED_FE(1,MONOMIAL)


template <>
Real FE<1,MONOMIAL>::shape(const ElemType,
                           const Order order,
                           const unsigned int i,
                           const Point & p)
{
  return fe_monomial_1D_shape(order, i, p);
}



template <>
Real FE<1,MONOMIAL>::shape(const Elem * elem,
                           const Order order,
                           const unsigned int i,
                           const Point & p,
                           const bool add_p_level)
{
  libmesh_assert(elem);

  return FE<1,MONOMIAL>::shape(elem->type(), order + add_p_level*elem->p_level(), i, p);
}


template <>
Real FE<1,MONOMIAL>::shape(const FEType fet,
                           const Elem * elem,
                           const unsigned int i,
                           const Point & p,
                           const bool add_p_level)
{
  libmesh_assert(elem);
  return FE<1,MONOMIAL>::shape(elem->type(), fet.order + add_p_level*elem->p_level(), i, p);
}


template <>
Real FE<1,MONOMIAL>::shape_deriv(const ElemType,
                                 const Order order,
                                 const unsigned int i,
                                 const unsigned int j,
                                 const Point & p)
{
  return fe_monomial_1D_shape_deriv(order, i, j, p);
}



template <>
Real FE<1,MONOMIAL>::shape_deriv(const Elem * elem,
                                 const Order order,
                                 const unsigned int i,
                                 const unsigned int j,
                                 const Point & p,
                                 const bool add_p_level)
{
  libmesh_assert(elem);

  return FE<1,MONOMIAL>::shape_deriv(elem->type(),
                                     order + add_p_level*elem->p_level(), i, j, p);
}


template <>
Real FE<1,MONOMIAL>::shape_deriv(const FEType fet,
                                 const Elem * elem,
                                 const unsigned int i,
                                 const unsigned int j,
                                 const Point & p,
                                 const bool add_p_level)
{
  libmesh_assert(elem);
  return FE<1,MONOMIAL>::shape_deriv(elem->type(), fet.order + add_p_level*elem->p_level(), i, j, p);


}

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

template <>
Real FE<1,MONOMIAL>::shape_second_deriv(const ElemType,
                                        const Order order,
                                        const unsigned int i,
                                        const unsigned int j,
                                        const Point & p)
{
  return fe_monomial_1D_shape_second_deriv(order, i, j, p);
}



template <>
Real FE<1,MONOMIAL>::shape_second_deriv(const Elem * elem,
                                        const Order order,
                                        const unsigned int i,
                                        const unsigned int j,
                                        const Point & p,
                                        const bool add_p_level)
{
  libmesh_assert(elem);

  return FE<1,MONOMIAL>::shape_second_deriv(elem->type(),
                                            order + add_p_level*elem->p_level(), i, j, p);
}

template <>
Real FE<1,MONOMIAL>::shape_second_deriv(const FEType fet,
                                        const Elem * elem,
                                        const unsigned int i,
                                        const unsigned int j,
                                        const Point & p,
                                        const bool add_p_level)
{
  libmesh_assert(elem);
  return FE<1,MONOMIAL>::shape_second_deriv(elem->type(), fet.order + add_p_level*elem->p_level(), i, j, p);
}

#endif

} // namespace libMesh
