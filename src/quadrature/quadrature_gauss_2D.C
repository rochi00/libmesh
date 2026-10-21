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
#include "libmesh/quadrature_gauss_2D.h"
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
      // Quadrilateral quadrature rules
    case QUAD4:
    case QUADSHELL4:
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
      {
        // We compute the 2D quadrature rule as a tensor
        // product of the 1D quadrature rule.
        //
        // For QUADs, a quadrature rule of order 'p' must be able to integrate
        // bilinear (p=1), biquadratic (p=2), bicubic (p=3), etc. polynomials of the form
        //
        // (x^p + x^{p-1} + ... + 1) * (y^p + y^{p-1} + ... + 1)
        //
        // These polynomials have terms *up to* degree 2p but they are *not* complete
        // polynomials of degree 2p. For example, when p=2 we have
        //        1
        //     x      y
        // x^2    xy     y^2
        //    yx^2   xy^2
        //       x^2y^2
        QGauss q1D(1,get_order());
        tensor_product_quad( q1D );
        return;
      }


      //---------------------------------------------
      // Triangle quadrature rules
    case TRI3:
    case TRISHELL3:
    case TRI3SUBDIVISION:
    case TRI6:
    case TRI7:
      {
        const unsigned int n = n_points_tri(get_order());
        libmesh_error_msg_if(n == invalid_uint,
                             "Quadrature rule " << get_order() << " not supported!");

        _points.resize(n);
        _weights.resize(n);
        rule_tri(get_order(), _points.data(), _weights.data());
        return;
      }


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
      // Unsupported type
    default:
      libmesh_error_msg("Element type not supported:" << Utility::enum_to_string(_type));
    }
#endif
}

} // namespace libMesh
