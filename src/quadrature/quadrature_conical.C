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


// libMesh includes
#include "libmesh/quadrature_conical.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/enum_quadrature_type.h"

namespace libMesh
{

// See also the files:
// quadrature_conical_2D.C
// quadrature_conical_3D.C
// for additional implementation.

QuadratureType QConical::type() const
{
  return QCONICAL;
}

std::unique_ptr<QBase> QConical::clone() const
{
  return std::make_unique<QConical>(*this);
}

void QConical::init_1D()
{
  QGauss gauss1D(1, get_order());
  gauss1D.init(*this);

  // Swap points and weights with the about-to-be destroyed rule.
  _points.swap(gauss1D.get_points());
  _weights.swap(gauss1D.get_weights());
}



void QConical::conical_product_tri()
{
  // Be sure the underlying rule object was built with the same dimension as the
  // rule we are about to construct.
  libmesh_assert_equal_to (this->get_dim(), 2);

  const unsigned int np = QGauss::n_points_1D(get_order());
  libmesh_error_msg_if(np == invalid_uint,
                       "Quadrature rule " << get_order() << " not supported!");

  // Resize the points and weights vectors
  _points.resize(np * np);
  _weights.resize(np * np);

  conical_product_tri(get_order(), _points.data(), _weights.data());
}



void QConical::conical_product_tet()
{
  // Be sure the underlying rule object was built with the same dimension as the
  // rule we are about to construct.
  libmesh_assert_equal_to (this->get_dim(), 3);

  const unsigned int np = QGauss::n_points_1D(get_order());
  libmesh_error_msg_if(np == invalid_uint,
                       "Quadrature rule " << get_order() << " not supported!");

  // Resize the points and weights vectors
  _points.resize(np * np * np);
  _weights.resize(np * np * np);

  conical_product_tet(get_order(), _points.data(), _weights.data());
}



void QConical::conical_product_pyramid()
{
  // Be sure the underlying rule object was built with the same dimension as the
  // rule we are about to construct.
  libmesh_assert_equal_to (this->get_dim(), 3);

  const unsigned int np = QGauss::n_points_1D(get_order());
  libmesh_error_msg_if(np == invalid_uint,
                       "Quadrature rule " << get_order() << " not supported!");

  // Resize the points and weights vectors
  _points.resize(np * np * np);
  _weights.resize(np * np * np);

  conical_product_pyramid(get_order(), _points.data(), _weights.data());
}

} // namespace libMesh
