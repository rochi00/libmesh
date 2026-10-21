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
#include "libmesh/fe_lagrange_shape_3D.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/cell_c0polyhedron.h"
#include "libmesh/tensor_value.h"

// Anonymous namespace for functions shared by LAGRANGE and
// L2_LAGRANGE implementations: these handle C0Polyhedron, whose shapes
// need the polyhedron itself, and pass every other type to the shared
// functions in fe_lagrange_shape_3D.h, which they overload.
// Implementations appear at the bottom of this file.
namespace libMesh
{
namespace
{

template <FEFamily T>
Real fe_lagrange_3D_shape(const ElemType,
                          const Order order,
                          const Elem * elem,
                          const unsigned int i,
                          const Point & p);

template <FEFamily T>
Real fe_lagrange_3D_shape_deriv(const ElemType type,
                                const Order order,
                                const Elem * elem,
                                const unsigned int i,
                                const unsigned int j,
                                const Point & p);

} // anonymous namespace


// TODO: If optimizations for LAGRANGE work well we should do
// L2_LAGRANGE too...
LIBMESH_DEFAULT_VECTORIZED_FE(3,L2_LAGRANGE)


template<>
void FE<3,LAGRANGE>::all_shapes
  (const Elem * elem,
   const Order o,
   const std::vector<Point> & p,
   std::vector<std::vector<OutputShape>> & v,
   const bool add_p_level)
{
  const ElemType type = elem->type();

  // Just loop on the harder-to-optimize cases
  if (type != HEX8 && type != HEX27)
    {
      FE<3,LAGRANGE>::default_all_shapes
        (elem,o,p,v,add_p_level);
      return;
    }

#if LIBMESH_DIM == 3

  const unsigned int n_sf = v.size();

  switch (o)
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
              libmesh_assert_less_equal (n_sf, 8);

              for (auto qp : index_range(p))
                {
                  const Point & q_point = p[qp];

                  for (unsigned int i : make_range(n_sf))
                    v[i][qp] = libMesh::detail::fe_lagrange_hex8_shape(i, q_point(0), q_point(1), q_point(2));
                }
              return;
            }

          default:
            libmesh_error(); // How did we get here?
          }
      }


      // quadratic Lagrange shape functions
    case SECOND:
      {
        switch (type)
          {
            // triquadratic hexahedral shape functions
          case HEX8:
// TODO: refactor to optimize this
//            libmesh_assert_msg(T == L2_LAGRANGE,
//                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case HEX27:
            {
              libmesh_assert_less_equal (n_sf, 27);

              for (auto qp : index_range(p))
                {
                  const Point & q_point = p[qp];

                  for (unsigned int i : make_range(n_sf))
                    v[i][qp] = libMesh::detail::fe_lagrange_hex27_shape(i, q_point(0), q_point(1), q_point(2));
                }
              return;
            }

          default:
            libmesh_error(); // How did we get here?
          }
      }

      // unsupported order
    default:
      libmesh_error_msg("ERROR: Unsupported 3D FE order on HEX!: " << o);
    }
#else // LIBMESH_DIM != 3
  libmesh_ignore(elem, o, p, v, add_p_level);
  libmesh_not_implemented();
#endif // LIBMESH_DIM == 3
}

template<>
void FE<3,LAGRANGE>::shapes
  (const Elem * elem,
   const Order o,
   const unsigned int i,
   const std::vector<Point> & p,
   std::vector<OutputShape> & v,
   const bool add_p_level)
{
  FE<3,LAGRANGE>::default_shapes
    (elem,o,i,p,v,add_p_level);
}

template<>
void FE<3,LAGRANGE>::shape_derivs
  (const Elem * elem,
   const Order o,
   const unsigned int i,
   const unsigned int j,
   const std::vector<Point> & p,
   std::vector<OutputShape> & v,
   const bool add_p_level)
{
  FE<3,LAGRANGE>::default_shape_derivs
    (elem,o,i,j,p,v,add_p_level);
}

template<>
void FE<3,LAGRANGE>::all_shape_derivs
  (const Elem * elem,
   const Order o,
   const std::vector<Point> & p,
   std::vector<std::vector<OutputShape>> * comps[3],
   const bool add_p_level)
{
  const ElemType type = elem->type();

  // Just loop on the harder-to-optimize cases
  if (type != HEX8 && type != HEX27)
    {
      FE<3,LAGRANGE>::default_all_shape_derivs
        (elem,o,p,comps,add_p_level);
      return;
    }

#if LIBMESH_DIM == 3

  libmesh_assert(comps[0]);
  libmesh_assert(comps[1]);
  libmesh_assert(comps[2]);
  const unsigned int n_sf = comps[0]->size();

  switch (o)
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
              libmesh_assert_equal_to (n_sf, 8);

              for (auto qp : index_range(p))
                {
                  const Point & q_point = p[qp];
                  for (unsigned int i : make_range(n_sf))
                    {
                      (*comps[0])[i][qp] = libMesh::detail::fe_lagrange_hex8_shape_deriv(i, 0, q_point(0), q_point(1), q_point(2));
                      (*comps[1])[i][qp] = libMesh::detail::fe_lagrange_hex8_shape_deriv(i, 1, q_point(0), q_point(1), q_point(2));
                      (*comps[2])[i][qp] = libMesh::detail::fe_lagrange_hex8_shape_deriv(i, 2, q_point(0), q_point(1), q_point(2));
                    }
                }
              return;
            }

          default:
            libmesh_error(); // How did we get here?
          }
      }


      // quadratic Lagrange shape functions
    case SECOND:
      {
        switch (type)
          {
            // triquadratic hexahedral shape functions
          case HEX8:
// TODO: refactor to optimize this
//            libmesh_assert_msg(T == L2_LAGRANGE,
//                               "High order on first order elements only supported for L2 families");
            libmesh_fallthrough();
          case HEX27:
            {
              libmesh_assert_less_equal (n_sf, 27);

              for (auto qp : index_range(p))
                {
                  const Point & q_point = p[qp];
                  for (unsigned int i : make_range(n_sf))
                    {
                      (*comps[0])[i][qp] = libMesh::detail::fe_lagrange_hex27_shape_deriv(i, 0, q_point(0), q_point(1), q_point(2));
                      (*comps[1])[i][qp] = libMesh::detail::fe_lagrange_hex27_shape_deriv(i, 1, q_point(0), q_point(1), q_point(2));
                      (*comps[2])[i][qp] = libMesh::detail::fe_lagrange_hex27_shape_deriv(i, 2, q_point(0), q_point(1), q_point(2));
                    }
                }
              return;
            }

          default:
            libmesh_error(); // How did we get here?
          }
      }

      // unsupported order
    default:
      libmesh_error_msg("ERROR: Unsupported 3D FE order on HEX!: " << o);
    }
#else // LIBMESH_DIM != 3
  libmesh_ignore(elem, o, p, v, add_p_level);
  libmesh_not_implemented();
#endif // LIBMESH_DIM == 3
}




template <>
Real FE<3,LAGRANGE>::shape(const ElemType type,
                           const Order order,
                           const unsigned int i,
                           const Point & p)
{
  return fe_lagrange_3D_shape<LAGRANGE>(type, order, nullptr, i, p);
}



template <>
Real FE<3,L2_LAGRANGE>::shape(const ElemType type,
                              const Order order,
                              const unsigned int i,
                              const Point & p)
{
  return fe_lagrange_3D_shape<L2_LAGRANGE>(type, order, nullptr, i, p);
}



template <>
Real FE<3,LAGRANGE>::shape(const Elem * elem,
                           const Order order,
                           const unsigned int i,
                           const Point & p,
                           const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape functions
  return fe_lagrange_3D_shape<LAGRANGE>(elem->type(), order + add_p_level*elem->p_level(), elem, i, p);
}



template <>
Real FE<3,L2_LAGRANGE>::shape(const Elem * elem,
                              const Order order,
                              const unsigned int i,
                              const Point & p,
                              const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape functions
  return fe_lagrange_3D_shape<L2_LAGRANGE>(elem->type(), order + add_p_level*elem->p_level(), elem, i, p);
}



template <>
Real FE<3,LAGRANGE>::shape(const FEType fet,
                           const Elem * elem,
                           const unsigned int i,
                           const Point & p,
                           const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_3D_shape<LAGRANGE>(elem->type(), fet.order + add_p_level*elem->p_level(), elem, i, p);
}



template <>
Real FE<3,L2_LAGRANGE>::shape(const FEType fet,
                              const Elem * elem,
                              const unsigned int i,
                              const Point & p,
                              const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_3D_shape<L2_LAGRANGE>(elem->type(), fet.order + add_p_level*elem->p_level(), elem, i, p);
}

template <>
Real FE<3,LAGRANGE>::shape_deriv(const ElemType type,
                                 const Order order,
                                 const unsigned int i,
                                 const unsigned int j,
                                 const Point & p)
{
  return fe_lagrange_3D_shape_deriv<LAGRANGE>(type, order, nullptr, i, j, p);
}



template <>
Real FE<3,L2_LAGRANGE>::shape_deriv(const ElemType type,
                                    const Order order,
                                    const unsigned int i,
                                    const unsigned int j,
                                    const Point & p)
{
  return fe_lagrange_3D_shape_deriv<L2_LAGRANGE>(type, order, nullptr, i, j, p);
}



template <>
Real FE<3,LAGRANGE>::shape_deriv(const Elem * elem,
                                 const Order order,
                                 const unsigned int i,
                                 const unsigned int j,
                                 const Point & p,
                                 const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape function derivatives
  return fe_lagrange_3D_shape_deriv<LAGRANGE>(elem->type(), order + add_p_level*elem->p_level(), elem, i, j, p);
}


template <>
Real FE<3,L2_LAGRANGE>::shape_deriv(const Elem * elem,
                                    const Order order,
                                    const unsigned int i,
                                    const unsigned int j,
                                    const Point & p,
                                    const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape function derivatives
  return fe_lagrange_3D_shape_deriv<L2_LAGRANGE>(elem->type(), order + add_p_level*elem->p_level(), elem, i, j, p);
}


template <>
Real FE<3,LAGRANGE>::shape_deriv(const FEType fet,
                                 const Elem * elem,
                                 const unsigned int i,
                                 const unsigned int j,
                                 const Point & p,
                                 const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_3D_shape_deriv<LAGRANGE>(elem->type(), fet.order + add_p_level*elem->p_level(), elem, i, j, p);
}


template <>
Real FE<3,L2_LAGRANGE>::shape_deriv(const FEType fet,
                                    const Elem * elem,
                                    const unsigned int i,
                                    const unsigned int j,
                                    const Point & p,
                                    const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_3D_shape_deriv<L2_LAGRANGE>(elem->type(), fet.order + add_p_level*elem->p_level(), elem, i, j, p);
}

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

template <>
Real FE<3,LAGRANGE>::shape_second_deriv(const ElemType type,
                                        const Order order,
                                        const unsigned int i,
                                        const unsigned int j,
                                        const Point & p)
{
  return fe_lagrange_3D_shape_second_deriv<LAGRANGE>(type, order, i, j, p);
}



template <>
Real FE<3,L2_LAGRANGE>::shape_second_deriv(const ElemType type,
                                           const Order order,
                                           const unsigned int i,
                                           const unsigned int j,
                                           const Point & p)
{
  return fe_lagrange_3D_shape_second_deriv<L2_LAGRANGE>(type, order, i, j, p);
}



template <>
Real FE<3,LAGRANGE>::shape_second_deriv(const Elem * elem,
                                        const Order order,
                                        const unsigned int i,
                                        const unsigned int j,
                                        const Point & p,
                                        const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape function derivatives
  return fe_lagrange_3D_shape_second_deriv<LAGRANGE>
    (elem->type(), order + add_p_level*elem->p_level(), i, j, p);
}



template <>
Real FE<3,L2_LAGRANGE>::shape_second_deriv(const Elem * elem,
                                           const Order order,
                                           const unsigned int i,
                                           const unsigned int j,
                                           const Point & p,
                                           const bool add_p_level)
{
  libmesh_assert(elem);

  // call the orientation-independent shape function derivatives
  return fe_lagrange_3D_shape_second_deriv<L2_LAGRANGE>
    (elem->type(), order + add_p_level*elem->p_level(), i, j, p);
}


template <>
Real FE<3,LAGRANGE>::shape_second_deriv(const FEType fet,
                                        const Elem * elem,
                                        const unsigned int i,
                                        const unsigned int j,
                                        const Point & p,
                                        const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_3D_shape_second_deriv<LAGRANGE>
    (elem->type(), fet.order + add_p_level*elem->p_level(), i, j, p);
}



template <>
Real FE<3,L2_LAGRANGE>::shape_second_deriv(const FEType fet,
                                           const Elem * elem,
                                           const unsigned int i,
                                           const unsigned int j,
                                           const Point & p,
                                           const bool add_p_level)
{
  libmesh_assert(elem);
  return fe_lagrange_3D_shape_second_deriv<L2_LAGRANGE>
    (elem->type(), fet.order + add_p_level*elem->p_level(), i, j, p);
}


#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES



namespace
{

template <FEFamily T>
Real fe_lagrange_3D_shape(const ElemType type,
                          const Order order,
                          const Elem * elem,
                          const unsigned int i,
                          const Point & p)
{
#if LIBMESH_DIM == 3

  switch (order)
    {
      // linear Lagrange shape functions
    case FIRST:
      {
        switch (type)
          {
          case C0POLYHEDRON:
            {
              // Polyhedra require using newer FE APIs
              if (!elem)
                libmesh_error_msg("Code (see stack trace) used an outdated FE function overload.\n"
                                  "Shape functions on a polyhedron are not defined by ElemType alone.");

              libmesh_assert(elem->type() == C0POLYHEDRON);

              const C0Polyhedron & poly = *cast_ptr<const C0Polyhedron *>(elem);

              // We can't use a small tolerance here, because in
              // inverse_map() Newton might hand us intermediate
              // iterates outside the polyhedron.
              const auto [s, a, b, c] = poly.subelement_coordinates(p, 100);
              if (s == invalid_uint)
                return 0;
              libmesh_assert_less(s, poly.n_subelements());

              const auto subtet = poly.subelement(s);

              // Avoid signed/unsigned comparison warnings
              const int nodei = i;
              if (nodei == subtet[0])
                return 1-a-b-c;
              if (nodei == subtet[1])
                return a;
              if (nodei == subtet[2])
                return b;
              if (nodei == subtet[3])
                return c;

              // Basis function i is not supported on p's subtet
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

  return libMesh::fe_lagrange_3D_shape<T>(type, order, i, p);
#else // LIBMESH_DIM != 3
  libmesh_ignore(type, order, elem, i, p);
  libmesh_not_implemented();
#endif
}



template <FEFamily T>
Real fe_lagrange_3D_shape_deriv(const ElemType type,
                                const Order order,
                                const Elem * elem,
                                const unsigned int i,
                                const unsigned int j,
                                const Point & p)
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
          case C0POLYHEDRON:
            {
              // Polyhedra require using newer FE APIs
              if (!elem)
                libmesh_error_msg("Code (see stack trace) used an outdated FE function overload.\n"
                                  "Shape functions on a polyhedron are not defined by ElemType alone.");

              libmesh_assert(elem->type() == C0POLYHEDRON);

              const C0Polyhedron & poly = *cast_ptr<const C0Polyhedron *>(elem);

              // We can't use a small tolerance here, because in
              // inverse_map() Newton might hand us intermediate
              // iterates outside the polyhedron.
              const auto [s, a, b, c] = poly.subelement_coordinates(p, 100);
              if (s == invalid_uint)
                return 0;
              libmesh_assert_less(s, poly.n_subelements());

              const auto subtet = poly.subelement(s);

              // Find derivatives w.r.t. subtriangle barycentric
              // coordinates
              Real du_da = 0, du_db = 0, du_dc = 0;

              // Avoid signed/unsigned comparison warnings
              const int nodei = i;
              if (nodei == subtet[0])
                du_da = du_db = du_dc = -1;
              else if (nodei == subtet[1])
                du_da = 1;
              else if (nodei == subtet[2])
                du_db = 1;
              else if (nodei == subtet[3])
                du_dc = 1;
              else
                // Basis function i is not supported on p's subtet
                return 0;

              // We want to return derivatives with respect to
              // xi/eta/zeta for the polyhedron, but what we
              // calculated above are with respect to xi and eta
              // coordinates for a master *triangle*.  We need to
              // convert from one to the other.

              const auto master_points = poly.master_subelement(s);

              const RealTensor dXi_dA(
                master_points[1](0) - master_points[0](0), master_points[2](0) - master_points[0](0), master_points[3](0) - master_points[0](0),
                master_points[1](1) - master_points[0](1), master_points[2](1) - master_points[0](1), master_points[3](1) - master_points[0](1),
                master_points[1](2) - master_points[0](2), master_points[2](2) - master_points[0](2), master_points[3](2) - master_points[0](2));

              // When we vectorize this we'll want a full inverse, but
              // when we're querying one component at a time it's
              // cheaper to manually compute a single column.
              // const RealTensor dabc_dxietazeta_dabc = dxietazeta_dabc.inverse();
              const Real jac = dXi_dA.det();

              switch (j)
                {
                  // d()/dxi
                case 0:
                  {
                    const Real da_dxi =  (dXi_dA(2,2)*dXi_dA(1,1) - dXi_dA(2,1)*dXi_dA(1,2)) / jac;
                    const Real db_dxi = -(dXi_dA(2,2)*dXi_dA(1,0) - dXi_dA(2,0)*dXi_dA(1,2)) / jac;
                    const Real dc_dxi =  (dXi_dA(2,1)*dXi_dA(1,0) - dXi_dA(2,0)*dXi_dA(1,1)) / jac;
                    return du_da*da_dxi + du_db*db_dxi + du_dc*dc_dxi;
                  }
                  // d()/deta
                case 1:
                  {
                    const Real da_deta = -(dXi_dA(2,2)*dXi_dA(0,1) - dXi_dA(2,1)*dXi_dA(0,2) ) / jac;
                    const Real db_deta =  (dXi_dA(2,2)*dXi_dA(0,0) - dXi_dA(2,0)*dXi_dA(0,2)) / jac;
                    const Real dc_deta = -(dXi_dA(2,1)*dXi_dA(0,0) - dXi_dA(2,0)*dXi_dA(0,1)) / jac;
                    return du_da*da_deta + du_db*db_deta + du_dc*dc_deta;
                  }
                  // d()/dzeta
                case 2:
                  {
                    const Real da_dzeta =  (dXi_dA(1,2)*dXi_dA(0,1) - dXi_dA(1,1)*dXi_dA(0,2) ) / jac;
                    const Real db_dzeta = -(dXi_dA(1,2)*dXi_dA(0,0) - dXi_dA(1,0)*dXi_dA(0,2)) / jac;
                    const Real dc_dzeta =  (dXi_dA(1,1)*dXi_dA(0,0) - dXi_dA(1,0)*dXi_dA(0,1)) / jac;
                    return du_da*da_dzeta + du_db*db_dzeta + du_dc*dc_dzeta;
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

  return libMesh::fe_lagrange_3D_shape_deriv<T>(type, order, i, j, p);
#else // LIBMESH_DIM != 3
  libmesh_ignore(type, order, elem, i, j, p);
  libmesh_not_implemented();
#endif
}


} // anonymous namespace

} // namespace libMesh
