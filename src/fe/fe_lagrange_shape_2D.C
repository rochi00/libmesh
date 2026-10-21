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


// Local includes
#include "libmesh/fe.h"
#include "libmesh/elem.h"
#include "libmesh/fe_lagrange_shape_2D.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/face_c0polygon.h"

// Anonymous namespace for functions shared by LAGRANGE and
// L2_LAGRANGE implementations: these handle C0Polygon, whose shapes
// need the polygon itself, and pass every other type to the shared
// functions in fe_lagrange_shape_2D.h, which they overload.
// Implementations appear at the bottom of this file.
namespace libMesh
{
namespace
{

template <FEFamily T>
Real fe_lagrange_2D_shape(const ElemType,
                          const Elem * elem,
                          const Order order,
                          const unsigned int i,
                          const Point & p);

template <FEFamily T>
Real fe_lagrange_2D_shape_deriv(const ElemType type,
                                const Elem * elem,
                                const Order order,
                                const unsigned int i,
                                const unsigned int j,
                                const Point & p);

} // anonymous namespace




LIBMESH_DEFAULT_VECTORIZED_FE(2,LAGRANGE)
LIBMESH_DEFAULT_VECTORIZED_FE(2,L2_LAGRANGE)


template <>
Real FE<2,LAGRANGE>::shape(const ElemType type,
                           const Order order,
                           const unsigned int i,
                           const Point & p)
{
  return fe_lagrange_2D_shape<LAGRANGE>(type, nullptr, order, i, p);
}



template <>
Real FE<2,L2_LAGRANGE>::shape(const ElemType type,
                              const Order order,
                              const unsigned int i,
                              const Point & p)
{
  return fe_lagrange_2D_shape<L2_LAGRANGE>(type, nullptr, order, i, p);
}


template <>
Real FE<2,LAGRANGE>::shape(const Elem * elem,
                           const Order order,
                           const unsigned int i,
                           const Point & p,
                           const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape functions
  return fe_lagrange_2D_shape<LAGRANGE>(elem->type(), elem, order + add_p_level*elem->p_level(), i, p);
}



template <>
Real FE<2,L2_LAGRANGE>::shape(const Elem * elem,
                              const Order order,
                              const unsigned int i,
                              const Point & p,
                              const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape functions
  return fe_lagrange_2D_shape<L2_LAGRANGE>(elem->type(), elem, order + add_p_level*elem->p_level(), i, p);
}


template <>
Real FE<2,LAGRANGE>::shape(const FEType fet,
                           const Elem * elem,
                           const unsigned int i,
                           const Point & p,
                           const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_2D_shape<LAGRANGE>(elem->type(), elem, fet.order + add_p_level*elem->p_level(), i, p);
}



template <>
Real FE<2,L2_LAGRANGE>::shape(const FEType fet,
                              const Elem * elem,
                              const unsigned int i,
                              const Point & p,
                              const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_2D_shape<L2_LAGRANGE>(elem->type(), elem, fet.order + add_p_level*elem->p_level(), i, p);
}


template <>
Real FE<2,LAGRANGE>::shape_deriv(const ElemType type,
                                 const Order order,
                                 const unsigned int i,
                                 const unsigned int j,
                                 const Point & p)
{
  return fe_lagrange_2D_shape_deriv<LAGRANGE>(type, nullptr, order, i, j, p);
}



template <>
Real FE<2,L2_LAGRANGE>::shape_deriv(const ElemType type,
                                    const Order order,
                                    const unsigned int i,
                                    const unsigned int j,
                                    const Point & p)
{
  return fe_lagrange_2D_shape_deriv<L2_LAGRANGE>(type, nullptr, order, i, j, p);
}



template <>
Real FE<2,LAGRANGE>::shape_deriv(const Elem * elem,
                                 const Order order,
                                 const unsigned int i,
                                 const unsigned int j,
                                 const Point & p,
                                 const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape functions
  return fe_lagrange_2D_shape_deriv<LAGRANGE>(elem->type(), elem, order + add_p_level*elem->p_level(), i, j, p);
}



template <>
Real FE<2,L2_LAGRANGE>::shape_deriv(const Elem * elem,
                                    const Order order,
                                    const unsigned int i,
                                    const unsigned int j,
                                    const Point & p,
                                    const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape functions
  return fe_lagrange_2D_shape_deriv<L2_LAGRANGE>(elem->type(), elem, order + add_p_level*elem->p_level(), i, j, p);
}


template <>
Real FE<2,LAGRANGE>::shape_deriv(const FEType fet,
                                 const Elem * elem,
                                 const unsigned int i,
                                 const unsigned int j,
                                 const Point & p,
                                 const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_2D_shape_deriv<LAGRANGE>(elem->type(), elem, fet.order + add_p_level*elem->p_level(), i, j, p);
}



template <>
Real FE<2,L2_LAGRANGE>::shape_deriv(const FEType fet,
                                    const Elem * elem,
                                    const unsigned int i,
                                    const unsigned int j,
                                    const Point & p,
                                    const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_2D_shape_deriv<L2_LAGRANGE>(elem->type(), elem, fet.order + add_p_level*elem->p_level(), i, j, p);
}



#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

template <>
Real FE<2,LAGRANGE>::shape_second_deriv(const ElemType type,
                                        const Order order,
                                        const unsigned int i,
                                        const unsigned int j,
                                        const Point & p)
{
  return fe_lagrange_2D_shape_second_deriv<LAGRANGE>(type, order, i, j, p);
}



template <>
Real FE<2,L2_LAGRANGE>::shape_second_deriv(const ElemType type,
                                           const Order order,
                                           const unsigned int i,
                                           const unsigned int j,
                                           const Point & p)
{
  return fe_lagrange_2D_shape_second_deriv<L2_LAGRANGE>(type, order, i, j, p);
}



template <>
Real FE<2,LAGRANGE>::shape_second_deriv(const Elem * elem,
                                        const Order order,
                                        const unsigned int i,
                                        const unsigned int j,
                                        const Point & p,
                                        const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape functions
  return fe_lagrange_2D_shape_second_deriv<LAGRANGE>(elem->type(), order + add_p_level*elem->p_level(), i, j, p);
}



template <>
Real FE<2,L2_LAGRANGE>::shape_second_deriv(const Elem * elem,
                                           const Order order,
                                           const unsigned int i,
                                           const unsigned int j,
                                           const Point & p,
                                           const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape functions
  return fe_lagrange_2D_shape_second_deriv<L2_LAGRANGE>(elem->type(), order + add_p_level*elem->p_level(), i, j, p);
}


template <>
Real FE<2,LAGRANGE>::shape_second_deriv(const FEType fet,
                                        const Elem * elem,
                                        const unsigned int i,
                                        const unsigned int j,
                                        const Point & p,
                                        const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_2D_shape_second_deriv<LAGRANGE>(elem->type(), fet.order + add_p_level*elem->p_level(), i, j, p);
}



template <>
Real FE<2,L2_LAGRANGE>::shape_second_deriv(const FEType fet,
                                           const Elem * elem,
                                           const unsigned int i,
                                           const unsigned int j,
                                           const Point & p,
                                           const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_2D_shape_second_deriv<L2_LAGRANGE>(elem->type(), fet.order + add_p_level*elem->p_level(), i, j, p);
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES

// Anonymous namespace function definitions
namespace
{

template <FEFamily T>
Real fe_lagrange_2D_shape(const ElemType type,
                          const Elem * elem,
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
          case C0POLYGON:
            {
              // C0Polygon requires using newer FE APIs
              if (!elem)
                libmesh_error_msg("Code (see stack trace) used an outdated FE function overload.\n"
                                  "Shape functions on a C0Polygon are not defined by its ElemType alone.");

              libmesh_assert(elem->type() == C0POLYGON);

              const C0Polygon & poly = *cast_ptr<const C0Polygon *>(elem);

              // We can't use a small tolerance here, because in
              // inverse_map() Newton might hand us intermediate
              // iterates outside the polygon.
              const auto [s, a, b] = poly.subtriangle_coordinates(p, 100);
              if (s == invalid_uint)
                return 0;
              libmesh_assert_less(s, poly.n_subtriangles());

              const auto subtri = poly.subtriangle(s);

              // Avoid signed/unsigned comparison warnings
              const int nodei = i;
              if (nodei == subtri[0])
                return 1-a-b;
              if (nodei == subtri[1])
                return a;
              if (nodei == subtri[2])
                return b;

              // Basis function i is not supported on p's subtriangle
              return 0;
            }

          default:
            break;
          }
        break;
      }

    default:
      break;
    }

  return libMesh::fe_lagrange_2D_shape<T>(type, order, i, p);
#else // LIBMESH_DIM > 1
  libmesh_ignore(type, order, i, p);
  libmesh_not_implemented();
#endif
}



template <FEFamily T>
Real fe_lagrange_2D_shape_deriv(const ElemType type,
                                const Elem * elem,
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
          case C0POLYGON:
            {
              // C0Polygon requires using newer FE APIs
              if (!elem)
                libmesh_error_msg("Code (see stack trace) used an outdated FE function overload.\n"
                                  "Shape functions on a C0Polygon are not defined by its ElemType alone.");

              libmesh_assert(elem->type() == C0POLYGON);

              const C0Polygon & poly = *cast_ptr<const C0Polygon *>(elem);

              // We can't use a small tolerance here, because in
              // inverse_map() Newton might hand us intermediate
              // iterates outside the polygon.
              const auto [s, a, b] = poly.subtriangle_coordinates(p, 100);
              if (s == invalid_uint)
                return 0;
              libmesh_assert_less(s, poly.n_subtriangles());

              const auto subtri = poly.subtriangle(s);

              // Find derivatives w.r.t. subtriangle barycentric
              // coordinates
              Real du_da = 0, du_db = 0;

              // Avoid signed/unsigned comparison warnings
              const int nodei = i;
              if (nodei == subtri[0])
                du_da = du_db = -1;
              else if (nodei == subtri[1])
                du_da = 1;
              else if (nodei == subtri[2])
                du_db = 1;
              else
                // Basis function i is not supported on p's subtriangle
                return 0;

              // We want to return derivatives with respect to xi and
              // eta in master space for the polygon, but what we
              // calculated above are with respect to xi and eta
              // coordinates for a master *triangle*.  We need to
              // convert from one to the other.

              const auto master_points = poly.master_subtriangle(s);

              const Real dxi_da = master_points[1](0) - master_points[0](0);
              const Real dxi_db = master_points[2](0) - master_points[0](0);
              const Real deta_da = master_points[1](1) - master_points[0](1);
              const Real deta_db = master_points[2](1) - master_points[0](1);
              const Real jac = dxi_da*deta_db - dxi_db*deta_da;

              switch (j)
                {
                  // d()/dxi
                case 0:
                  {
                    const Real da_dxi = deta_db / jac;
                    const Real db_dxi = -deta_da / jac;
                    return du_da*da_dxi + du_db*db_dxi;
                  }
                  // d()/deta
                case 1:
                  {
                    const Real da_deta = -dxi_db / jac;
                    const Real db_deta = dxi_da / jac;
                    return du_da*da_deta + du_db*db_deta;
                  }
                default:
                  libmesh_error_msg("ERROR: Invalid derivative index j = " << j);
                }
            }

          default:
            break;
          }
        break;
      }

    default:
      break;
    }

  return libMesh::fe_lagrange_2D_shape_deriv<T>(type, order, i, j, p);
#else // LIBMESH_DIM > 1
  libmesh_ignore(type, order, i, j, p);
  libmesh_not_implemented();
#endif
}

} // anonymous namespace

} // namespace libMesh
