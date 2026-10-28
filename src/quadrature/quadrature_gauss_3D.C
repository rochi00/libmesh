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
#include "libmesh/cell_c0polyhedron.h"

namespace libMesh
{

void QGauss::init_3D()
{
#if LIBMESH_DIM == 3

  //-----------------------------------------------------------------------
  // 3D quadrature rules
  switch (_type)
    {
      //---------------------------------------------
      // Arbitrary polyhedron quadrature rules
    case C0POLYHEDRON:
      {
        QGauss tet_rule(3, _order);
        tet_rule.init(TET4, _p_level, true);

        std::vector<Point> & tetpoints = tet_rule.get_points();
        std::vector<Real> & tetweights = tet_rule.get_weights();

        std::size_t numtetpts = tetpoints.size();

        // C0Polyhedron requires the newer Quadrature API
        if (!_elem)
          libmesh_error();

        libmesh_assert(_elem->type() == C0POLYHEDRON);

        const C0Polyhedron & poly = *cast_ptr<const C0Polyhedron *>(_elem);

        std::size_t numtets = poly.n_subelements();
        _points.resize(numtetpts*numtets);
        _weights.resize(numtetpts*numtets);

        for (std::size_t t = 0; t != numtets; ++t)
          {
            auto master_points = poly.master_subelement(t);

            const Point v01 = master_points[1] - master_points[0];
            const Point v02 = master_points[2] - master_points[0];
            const Point v03 = master_points[3] - master_points[0];

            // The factor of one sixth from the tetweights cancels out
            // the factor of six here, so we don't need to do so
            // ourselves.
            const Real six_master_tet_vol =
              triple_product(v01, v02, v03);

            for (std::size_t i = 0; i != numtetpts; ++i)
              {
                _points[numtetpts*t+i] =
                  master_points[0] +
                    v01 * tetpoints[i](0) +
                    v02 * tetpoints[i](1) +
                    v03 * tetpoints[i](2);
                _weights[numtetpts*t+i] = tetweights[i] *
                                          six_master_tet_vol;
              }
          }
        return;
      }

      //---------------------------------------------
      // Every other 3D type's rule; n_points_elem() rejects types
      // without one
    default:
      {
        const unsigned int n =
          n_points_elem(_type, get_order(), allow_rules_with_negative_weights);
        libmesh_error_msg_if(n == invalid_uint || Elem::type_to_dim_map[_type] != 3,
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
