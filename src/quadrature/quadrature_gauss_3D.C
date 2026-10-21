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
#include "libmesh/quadrature_conical.h"
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
      // Hex quadrature rules
    case HEX8:
    case HEX20:
    case HEX27:
      {
        // We compute the 3D quadrature rule as a tensor
        // product of the 1D quadrature rule.
        QGauss q1D(1, get_order());
        tensor_product_hex( q1D );
        return;
      }



      //---------------------------------------------
      // Tetrahedral quadrature rules
    case TET4:
    case TET10:
    case TET14:
      {
        const unsigned int n =
          n_points_tet(get_order(), allow_rules_with_negative_weights);
        libmesh_error_msg_if(n == invalid_uint,
                             "Quadrature rule " << get_order() << " not supported!");

        _points.resize(n);
        _weights.resize(n);
        rule_tet(get_order(), allow_rules_with_negative_weights,
                 _points.data(), _weights.data());
        return;
      } // end case TET



      //---------------------------------------------
      // Prism quadrature rules
    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
      {
        // We compute the 3D quadrature rule as a tensor
        // product of the 1D quadrature rule and a 2D
        // triangle quadrature rule

        QGauss q1D(1,get_order());
        QGauss q2D(2,_order);

        // Initialize the 2D rule (1D is pre-initialized)
        q2D.init(TRI3, _p_level, /*simple_type_only=*/true);

        tensor_product_prism(q1D, q2D);

        return;
      }



      //---------------------------------------------
      // Pyramid
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      {
        // We compute the Pyramid rule as a conical product of a
        // Jacobi rule with alpha==2 on the interval [0,1] two 1D
        // Gauss rules with suitably modified points.  The idea comes
        // from: Stroud, A.H. "Approximate Calculation of Multiple
        // Integrals."
        QConical conical_rule(3, _order);
        conical_rule.init(*this);

        // Swap points and weights with the about-to-be destroyed rule.
        _points.swap (conical_rule.get_points() );
        _weights.swap(conical_rule.get_weights());

        return;

      }


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
      // Unsupported type
    default:
      libmesh_error_msg("ERROR: Unsupported type: " << Utility::enum_to_string(_type));
    }
#endif
}

} // namespace libMesh
