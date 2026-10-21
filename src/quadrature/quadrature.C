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
#include "libmesh/elem.h"
#include "libmesh/quadrature.h"
#include "libmesh/int_range.h"

namespace libMesh
{

QBase::QBase(unsigned int d,
             Order o) :
  allow_rules_with_negative_weights(true),
  allow_nodal_pyramid_quadrature(false),
  _dim(d),
  _order(o),
  _type(INVALID_ELEM),
  _elem(nullptr),
  _p_level(0)
{}

std::unique_ptr<QBase> QBase::clone() const
{
  return QBase::build(this->type(), this->get_dim(), this->get_order());
}

void QBase::print_info(std::ostream & os) const
{
  libmesh_assert(!_points.empty());
  libmesh_assert(!_weights.empty());

  Real summed_weights=0;
  os << "N_Q_Points=" << this->n_points() << std::endl << std::endl;
  for (auto qpoint: index_range(_points))
    {
      os << " Point " << qpoint << ":\n"
         << "  "
         << _points[qpoint]
         << "\n Weight:\n "
         << "  w=" << _weights[qpoint] << "\n" << std::endl;

      summed_weights += _weights[qpoint];
    }
  os << "Summed Weights: " << summed_weights << std::endl;
}



void QBase::init(const Elem & elem,
                 unsigned int p)
{
  libmesh_assert_equal_to(elem.dim(), _dim);

  // Default to the element p_level() value
  if (p == invalid_uint)
    p = elem.p_level();

  ElemType t = elem.type();

  // check to see if we have already
  // done the work for this quadrature rule
  //
  // If we have something like a Polygon subclass then we're going to
  // need to recompute to be safe; even if we're using the same
  // element, it might have been distorted enough that its subtriangle
  // triangulation has been changed.
  if (t == _type && p == _p_level && !elem.runtime_topology())
    return;
  else
    {
      _elem = &elem;
      _type = t;
      _p_level = p;
    }

  switch(_elem->dim())
    {
    case 0:
      this->init_0D();

      return;

    case 1:
      this->init_1D();

      return;

    case 2:
      this->init_2D();

      return;

    case 3:
      this->init_3D();

      return;

    default:
      libmesh_error_msg("Invalid dimension _dim = " << _dim);
    }
}



void QBase::init(const ElemType t,
                 unsigned int p,
                 bool simple_type_only)
{
  // Some element types require data from a specific element, so can
  // only be used with newer APIs.
  if (t == C0POLYGON || t == C0POLYHEDRON)
    libmesh_error_msg("Code (see stack trace) used an outdated quadrature function overload.\n"
                      "Quadrature rules on a C0Polygon are not defined by its ElemType alone.");

  // This API is dangerous to use on general meshes, which may include
  // element types where the desired quadrature depends on the
  // physical element, but we still want to be able to initialize
  // based on only a type for certain simple cases
  if (t != EDGE2 && !simple_type_only)
    libmesh_deprecated();

  // check to see if we have already
  // done the work for this quadrature rule
  if (t == _type && p == _p_level)
    return;
  else
    {
      _elem = nullptr;
      _type = t;
      _p_level = p;
    }

  switch(_dim)
    {
    case 0:
      this->init_0D();

      return;

    case 1:
      this->init_1D();

      return;

    case 2:
      this->init_2D();

      return;

    case 3:
      this->init_3D();

      return;

    default:
      libmesh_error_msg("Invalid dimension _dim = " << _dim);
    }
}



void QBase::init(const QBase & other_rule)
{
  if (other_rule._elem)
    this->init(*other_rule._elem, other_rule._p_level);
  else
    this->init(other_rule._type, other_rule._p_level, true);
}



void QBase::init (const Elem & elem,
                  const std::vector<Real> & /* vertex_distance_func */,
                  unsigned int p_level)
{
  // dispatch generic implementation
  this->init(elem.type(), p_level);
}



void QBase::init_0D()
{
  _points.resize(1);
  _weights.resize(1);
  _points[0] = Point(0.);
  _weights[0] = 1.0;
}



void QBase::init_2D ()
{
  libmesh_not_implemented();
}



void QBase::init_3D ()
{
  libmesh_not_implemented();
}



void QBase::scale(std::pair<Real, Real> old_range,
                  std::pair<Real, Real> new_range)
{
  // Make sure we are in 1D
  libmesh_assert_equal_to (_dim, 1);

  scale(old_range, new_range, cast_int<unsigned int>(_points.size()),
        _points.data(), _weights.data());
}




void QBase::tensor_product_quad(const QBase & q1D)
{

  const unsigned int np = q1D.n_points();

  _points.resize(np * np);

  _weights.resize(np * np);

  tensor_product_quad(np, q1D._points.data(), q1D._weights.data(),
                      _points.data(), _weights.data());
}





void QBase::tensor_product_hex(const QBase & q1D)
{
  const unsigned int np = q1D.n_points();

  _points.resize(np * np * np);

  _weights.resize(np * np * np);

  tensor_product_hex(np, q1D._points.data(), q1D._weights.data(),
                     _points.data(), _weights.data());
}




void QBase::tensor_product_prism(const QBase & q1D, const QBase & q2D)
{
  const unsigned int n_points1D = q1D.n_points();
  const unsigned int n_points2D = q2D.n_points();

  _points.resize  (n_points1D * n_points2D);
  _weights.resize (n_points1D * n_points2D);

  tensor_product_prism(n_points1D, q1D._points.data(), q1D._weights.data(),
                       n_points2D, q2D._points.data(), q2D._weights.data(),
                       _points.data(), _weights.data());
}




std::ostream & operator << (std::ostream & os, const QBase & q)
{
  q.print_info(os);
  return os;
}

} // namespace libMesh
