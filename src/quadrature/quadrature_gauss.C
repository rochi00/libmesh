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
#include "libmesh/quadrature_gauss.h"
#include "libmesh/enum_quadrature_type.h"

namespace libMesh
{

// See the files:
// quadrature_gauss_1D.C
// quadrature_gauss_2D.C
// quadrature_gauss_3D.C
// for implementation of specific element types.


QuadratureType QGauss::type() const
{
  return QGAUSS;
}

std::unique_ptr<QBase> QGauss::clone() const
{
  return std::make_unique<QGauss>(*this);
}

void QGauss::keast_rule(const Real rule_data[][4],
                        const unsigned int n_pts)
{
  keast_rule(rule_data, n_pts, cast_int<unsigned int>(_points.size()),
             _points.data(), _weights.data());
}


void QGauss::dunavant_rule2(const Real * wts,
                            const Real * a,
                            const Real * b,
                            const unsigned int * permutation_ids,
                            unsigned int n_wts)
{
  // Figure out how many total points by summing up the entries
  // in the permutation_ids array, and resize the _points and _weights
  // vectors appropriately.
  unsigned int total_pts = 0;
  for (unsigned int p=0; p<n_wts; ++p)
    total_pts += permutation_ids[p];

  // Resize point and weight vectors appropriately.
  _points.resize(total_pts);
  _weights.resize(total_pts);

  dunavant_rule2(wts, a, b, permutation_ids, n_wts,
                 _points.data(), _weights.data());
}


void QGauss::dunavant_rule(const Real rule_data[][4],
                           const unsigned int n_pts)
{
  dunavant_rule(rule_data, n_pts, cast_int<unsigned int>(_points.size()),
                _points.data(), _weights.data());
}

} // namespace libMesh
