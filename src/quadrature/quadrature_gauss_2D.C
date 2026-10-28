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
#include "libmesh/quadrature_gauss_3D.h"
#include "libmesh/enum_to_string.h"

#include "libmesh/face_c0polygon.h"

namespace libMesh
{


void QGauss::init_2D()
{
#if LIBMESH_DIM > 1

  //-----------------------------------------------------------------------
  // 2D quadrature rules
  switch (_type)
    {


      //---------------------------------------------
      // Arbitrary polygon quadrature rules
    case C0POLYGON:
      {
        QGauss tri_rule(2, _order);
        tri_rule.init(TRI3, _p_level, true);

        std::vector<Point> & tripoints = tri_rule.get_points();
        std::vector<Real> & triweights = tri_rule.get_weights();

        std::size_t numtripts = tripoints.size();

        // C0Polygon requires the newer Quadrature API
        if (!_elem)
          libmesh_error();

        libmesh_assert(_elem->type() == C0POLYGON);

        const C0Polygon & poly = *cast_ptr<const C0Polygon *>(_elem);

        std::size_t numtris = poly.n_subtriangles();
        _points.resize(numtripts*numtris);
        _weights.resize(numtripts*numtris);
        for (std::size_t t = 0; t != numtris; ++t)
          {
            auto master_points = poly.master_subtriangle(t);

            // The factor of one half from the triweights cancels out
            // the factor of two here, so we don't need to do so
            // ourselves.
            const Real twice_master_tri_area =
              (- master_points[1](1) * master_points[2](0)
               - master_points[0](1) * master_points[1](0)
               + master_points[0](1) * master_points[2](0)
               + master_points[0](0) * master_points[1](1)
               - master_points[0](0) * master_points[2](1)
               + master_points[1](0) * master_points[2](1));

            const Point v01 = master_points[1] - master_points[0];
            const Point v02 = master_points[2] - master_points[0];

            for (std::size_t i = 0; i != numtripts; ++i)
              {
                _points[numtripts*t+i] =
                  master_points[0] +
                  v01 * tripoints[i](0) +
                  v02 * tripoints[i](1);
                _weights[numtripts*t+i] = triweights[i] *
                                          twice_master_tri_area;
              }
          }
        return;
      }

      //---------------------------------------------
      // Every other 2D type's rule; n_points_elem() rejects types
      // without one
    default:
      {
        const unsigned int n =
          n_points_elem(_type, get_order(), allow_rules_with_negative_weights);
        libmesh_error_msg_if(n == invalid_uint || Elem::type_to_dim_map[_type] != 2,
                             "Quadrature rule " << get_order() << " not supported for "
                             << Utility::enum_to_string(_type));

        _points.resize(n);
        _weights.resize(n);
        rule_elem(_type, get_order(), allow_rules_with_negative_weights,
                  _points.data(), _weights.data());
        return;
      }
    }
#endif
}

} // namespace libMesh
