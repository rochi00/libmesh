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
#include "libmesh/quadrature_gauss.h"
#include "libmesh/quadrature_jacobi_1D.h"

namespace libMesh
{

void QJacobi::init_1D()
{
  // rule_1D() errors on these, since device code can't warn; here we
  // keep warning and leaving the rule empty, as we always have.
  if (!((_alpha == 1 || _alpha == 2) && _beta == 0))
    {
      libMesh::err << "Unsupported combination of (alpha,beta) = ("
                   << _alpha
                   << ","
                   << _beta
                   << ") requested in Jacobi-Gauss quadrature rule."
                   << std::endl;
      return;
    }

  // Each rule here, like the Gauss rule of the same order, is the
  // n-point rule exact to degree 2n-1, so the two share point counts.
  const unsigned int n = QGauss::n_points_1D(get_order());
  libmesh_error_msg_if(n == invalid_uint,
                       "Quadrature rule " << _order << " not supported!");

  _points.resize(n);
  _weights.resize(n);
  rule_1D(_alpha, _beta, get_order(), _points.data(), _weights.data());
}

} // namespace libMesh
