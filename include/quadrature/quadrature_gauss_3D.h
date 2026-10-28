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


#ifndef LIBMESH_QUADRATURE_GAUSS_3D_H
#define LIBMESH_QUADRATURE_GAUSS_3D_H

#include "libmesh/quadrature_gauss.h"
#include "libmesh/quadrature_gauss_2D.h" // QGauss::rule_tri(), for rule_elem()
#include "libmesh/quadrature_conical.h"
#include "libmesh/quadrature_gm.h"

namespace libMesh
{

LIBMESH_DEVICE_INLINE
void QGauss::rule_tet(const Order o,
                      const bool allow_negative_weights,
                      Point * points,
                      Real * weights)
{
  //-----------------------------------------------------------------------
  // Tetrahedral quadrature rules
  switch(o)
    {
      // Taken from pg. 222 of "The finite element method," vol. 1
      // ed. 5 by Zienkiewicz & Taylor
    case CONSTANT:
    case FIRST:
      {
        // Exact for linears


        points[0](0) = .25;
        points[0](1) = .25;
        points[0](2) = .25;

        weights[0] = Real(1)/6;

        return;
      }
    case SECOND:
      {
        // Exact for quadratics


        // Can't be constexpr with my version of Boost quad
        // precision
        const Real b = 0.25*(1-std::sqrt(Real(5))/5);
        const Real a = 1-3*b;

        points[0](0) = a;
        points[0](1) = b;
        points[0](2) = b;

        points[1](0) = b;
        points[1](1) = a;
        points[1](2) = b;

        points[2](0) = b;
        points[2](1) = b;
        points[2](2) = a;

        points[3](0) = b;
        points[3](1) = b;
        points[3](2) = b;



        weights[0] = Real(1)/24;
        weights[1] = weights[0];
        weights[2] = weights[0];
        weights[3] = weights[0];

        return;
      }



      // Can be found in the class notes
      // http://www.cs.rpi.edu/~flaherje/FEM/fem6.ps
      // by Flaherty.
      //
      // Caution: this rule has a negative weight and may be
      // unsuitable for some problems.
      // Exact for cubics.
      //
      // Note: Keast (see ref. elsewhere in this file) also gives
      // a third-order rule with positive weights, but it contains points
      // on the ref. elt. boundary, making it less suitable for FEM calculations.
    case THIRD:
      {
        if (allow_negative_weights)
          {


            points[0](0) = .25;
            points[0](1) = .25;
            points[0](2) = .25;

            points[1](0) = .5;
            points[1](1) = Real(1)/6;
            points[1](2) = Real(1)/6;

            points[2](0) = Real(1)/6;
            points[2](1) = .5;
            points[2](2) = Real(1)/6;

            points[3](0) = Real(1)/6;
            points[3](1) = Real(1)/6;
            points[3](2) = .5;

            points[4](0) = Real(1)/6;
            points[4](1) = Real(1)/6;
            points[4](2) = Real(1)/6;


            weights[0] = Real(-2)/15;
            weights[1] = .075;
            weights[2] = weights[1];
            weights[3] = weights[1];
            weights[4] = weights[1];

            return;
          } // end if (allow_negative_weights)
        else
          {
            // If a rule with positive weights is required, the 2x2x2 conical
            // product rule is third-order accurate and has less points than
            // the next-available positive-weight rule at FIFTH order.
            QConical::conical_product_tet(o, points, weights);

            return;
          }
        // Note: if !allow_rules_with_negative_weights, fall through to next case.
      }



      // Originally a Keast rule,
      //    Patrick Keast,
      //    Moderate Degree Tetrahedral Quadrature Formulas,
      //    Computer Methods in Applied Mechanics and Engineering,
      //    Volume 55, Number 3, May 1986, pages 339-348.
      //
      // Can also be found the class notes
      // http://www.cs.rpi.edu/~flaherje/FEM/fem6.ps
      // by Flaherty.
      //
      // Caution: this rule has a negative weight and may be
      // unsuitable for some problems.
    case FOURTH:
      {
        if (allow_negative_weights)
          {

            // The raw data for the quadrature rule.
            const Real rule_data[3][4] = {
              {0.250000000000000000e+00_R,                         0._R,                            0._R,  -0.131555555555555556e-01_R},  // 1
              {0.785714285714285714e+00_R,   0.714285714285714285e-01_R,                            0._R,   0.762222222222222222e-02_R},  // 4
              {0.399403576166799219e+00_R,                         0._R,      0.100596423833200785e+00_R,   0.248888888888888889e-01_R}   // 6
            };


            // Now call the keast routine to generate _points and _weights
            keast_rule(rule_data, 3, n_points_tet(o, allow_negative_weights), points, weights);

            return;
          } // end if (allow_negative_weights)
        // Note: if !allow_rules_with_negative_weights, fall through to next case.
      }

      libmesh_fallthrough();


      // Walkington's fifth-order 14-point rule from
      // "Quadrature on Simplices of Arbitrary Dimension"
      //
      // We originally had a Keast rule here, but this rule had
      // more points than an equivalent rule by Walkington and
      // also contained points on the boundary of the ref. elt,
      // making it less suitable for FEM calculations.
    case FIFTH:
      {

        // permutations of these points and suitably-modified versions of
        // these points are the quadrature point locations
        const Real a[3] = {0.31088591926330060980_R,    // a1 from the paper
                           0.092735250310891226402_R,   // a2 from the paper
                           0.045503704125649649492_R};  // a3 from the paper

        // weights.  a[] and wt[] are the only floating-point inputs required
        // for this rule.
        const Real wt[3] = {0.018781320953002641800_R,    // w1 from the paper
                            0.012248840519393658257_R,    // w2 from the paper
                            0.0070910034628469110730_R};  // w3 from the paper

        // The first two sets of 4 points are formed in a similar manner
        for (unsigned int i=0; i<2; ++i)
          {
            // Where we will insert values into _points and _weights
            const unsigned int offset=4*i;

            // Stuff points and weights values into their arrays
            const Real b = 1. - 3.*a[i];

            // Here are the permutations.  Order of these is not important,
            // all have the same weight
            points[offset + 0] = Point(a[i], a[i], a[i]);
            points[offset + 1] = Point(a[i],    b, a[i]);
            points[offset + 2] = Point(   b, a[i], a[i]);
            points[offset + 3] = Point(a[i], a[i],    b);

            // These 4 points all have the same weights
            for (unsigned int j=0; j<4; ++j)
              weights[offset + j] = wt[i];
          } // end for


        {
          // The third set contains 6 points and is formed a little differently
          const unsigned int offset = 8;
          const Real b = 0.5*(1. - 2.*a[2]);

          // Here are the permutations.  Order of these is not important,
          // all have the same weight
          points[offset + 0] = Point(b   ,    b, a[2]);
          points[offset + 1] = Point(b   , a[2], a[2]);
          points[offset + 2] = Point(a[2], a[2],    b);
          points[offset + 3] = Point(a[2],    b, a[2]);
          points[offset + 4] = Point(   b, a[2],    b);
          points[offset + 5] = Point(a[2],    b,    b);

          // These 6 points all have the same weights
          for (unsigned int j=0; j<6; ++j)
            weights[offset + j] = wt[2];
        }


        // Original rule by Keast, unsuitable because it has points on the
        // reference element boundary!
        //       _points.resize(15);
        //       _weights.resize(15);

        //       _points[0](0) = 0.25;
        //       _points[0](1) = 0.25;
        //       _points[0](2) = 0.25;

        //       {
        // const Real a = 0.;
        // const Real b = Real(1)/3;

        // _points[1](0) = a;
        // _points[1](1) = b;
        // _points[1](2) = b;

        // _points[2](0) = b;
        // _points[2](1) = a;
        // _points[2](2) = b;

        // _points[3](0) = b;
        // _points[3](1) = b;
        // _points[3](2) = a;

        // _points[4](0) = b;
        // _points[4](1) = b;
        // _points[4](2) = b;
        //       }
        //       {
        // const Real a = Real(8)/11;
        // const Real b = Real(1)/11;

        // _points[5](0) = a;
        // _points[5](1) = b;
        // _points[5](2) = b;

        // _points[6](0) = b;
        // _points[6](1) = a;
        // _points[6](2) = b;

        // _points[7](0) = b;
        // _points[7](1) = b;
        // _points[7](2) = a;

        // _points[8](0) = b;
        // _points[8](1) = b;
        // _points[8](2) = b;
        //       }
        //       {
        // const Real a = 0.066550153573664;
        // const Real b = 0.433449846426336;

        // _points[9](0) = b;
        // _points[9](1) = a;
        // _points[9](2) = a;

        // _points[10](0) = a;
        // _points[10](1) = a;
        // _points[10](2) = b;

        // _points[11](0) = a;
        // _points[11](1) = b;
        // _points[11](2) = b;

        // _points[12](0) = b;
        // _points[12](1) = a;
        // _points[12](2) = b;

        // _points[13](0) = b;
        // _points[13](1) = b;
        // _points[13](2) = a;

        // _points[14](0) = a;
        // _points[14](1) = b;
        // _points[14](2) = a;
        //       }

        //       _weights[0]  = 0.030283678097089;
        //       _weights[1]  = 0.006026785714286;
        //       _weights[2]  = _weights[1];
        //       _weights[3]  = _weights[1];
        //       _weights[4]  = _weights[1];
        //       _weights[5]  = 0.011645249086029;
        //       _weights[6]  = _weights[5];
        //       _weights[7]  = _weights[5];
        //       _weights[8]  = _weights[5];
        //       _weights[9]  = 0.010949141561386;
        //       _weights[10] = _weights[9];
        //       _weights[11] = _weights[9];
        //       _weights[12] = _weights[9];
        //       _weights[13] = _weights[9];
        //       _weights[14] = _weights[9];

        return;
      }




      // This rule is originally from Keast:
      //    Patrick Keast,
      //    Moderate Degree Tetrahedral Quadrature Formulas,
      //    Computer Methods in Applied Mechanics and Engineering,
      //    Volume 55, Number 3, May 1986, pages 339-348.
      //
      // It is accurate on 6th-degree polynomials and has 24 points
      // vs. 64 for the comparable conical product rule.
      //
      // Values copied 24th June 2008 from:
      // http://people.scs.fsu.edu/~burkardt/f_src/keast/keast.f90
    case SIXTH:
      {

        // The raw data for the quadrature rule.
        const Real rule_data[4][4] = {
          {0.356191386222544953e+00_R , 0.214602871259151684e+00_R ,                       0._R, 0.00665379170969464506e+00_R}, // 4
          {0.877978124396165982e+00_R , 0.0406739585346113397e+00_R,                       0._R, 0.00167953517588677620e+00_R}, // 4
          {0.0329863295731730594e+00_R, 0.322337890142275646e+00_R ,                       0._R, 0.00922619692394239843e+00_R}, // 4
          {0.0636610018750175299e+00_R, 0.269672331458315867e+00_R , 0.603005664791649076e+00_R, 0.00803571428571428248e+00_R}  // 12
        };


        // Now call the keast routine to generate _points and _weights
        keast_rule(rule_data, 4, n_points_tet(o, allow_negative_weights), points, weights);

        return;
      }


      // Keast's 31 point, 7th-order rule contains points on the reference
      // element boundary, so we've decided not to include it here.
      //
      // Keast's 8th-order rule has 45 points.  and a negative
      // weight, so if you've explicitly disallowed such rules
      // you will fall through to the conical product rules
      // below.
    case SEVENTH:
    case EIGHTH:
      {
        if (allow_negative_weights)
          {

            // The raw data for the quadrature rule.
            const Real rule_data[7][4] = {
              {0.250000000000000000e+00_R,                        0._R,                        0._R,   -0.393270066412926145e-01_R},  // 1
              {0.617587190300082967e+00_R,  0.127470936566639015e+00_R,                        0._R,    0.408131605934270525e-02_R},  // 4
              {0.903763508822103123e+00_R,  0.320788303926322960e-01_R,                        0._R,    0.658086773304341943e-03_R},  // 4
              {0.497770956432810185e-01_R,                        0._R,  0.450222904356718978e+00_R,    0.438425882512284693e-02_R},  // 6
              {0.183730447398549945e+00_R,                        0._R,  0.316269552601450060e+00_R,    0.138300638425098166e-01_R},  // 6
              {0.231901089397150906e+00_R,  0.229177878448171174e-01_R,  0.513280033360881072e+00_R,    0.424043742468372453e-02_R},  // 12
              {0.379700484718286102e-01_R,  0.730313427807538396e+00_R,  0.193746475248804382e+00_R,    0.223873973961420164e-02_R}   // 12
            };


            // Now call the keast routine to generate _points and _weights
            keast_rule(rule_data, 7, n_points_tet(o, allow_negative_weights), points, weights);

            return;
          } // end if (allow_negative_weights)
        // Note: if !allow_rules_with_negative_weights, fall through to next case.
      }

      libmesh_fallthrough();


      // Fall back on Grundmann-Moller or Conical Product rules at high orders.
    default:
      {
        if ((allow_negative_weights) && (o < 34))
          {
            // The Grundmann-Moller rules are defined to arbitrary order and
            // can have significantly fewer evaluation points than conical product
            // rules.  If you allow rules with negative weights, the GM rules
            // will be more efficient for degree up to 33 (for degree 34 and
            // higher, CP is more efficient!) but may be more susceptible
            // to round-off error.  Safest is to disallow rules with negative
            // weights, but this decision should be made on a case-by-case basis.
            // Only orders 9 and up reach here, which QGrundmann_Moller
            // builds from gm_rule() with s = order/2.
            QGrundmann_Moller::gm_rule(o/2, /*dim=*/3, points, weights);

            return;
          }

        else
          {
            // The following quadrature rules are generated as
            // conical products.  These tend to be non-optimal
            // (use too many points, cluster points in certain
            // regions of the domain) but they are quite easy to
            // automatically generate using a 1D Gauss rule on
            // [0,1] and two 1D Jacobi-Gauss rules on [0,1].
            QConical::conical_product_tet(o, points, weights);

            return;
          }
      }
    }
}



LIBMESH_DEVICE_INLINE
void QGauss::rule_elem(const ElemType t,
                       const Order o,
                       const bool allow_negative_weights,
                       Point * points,
                       Real * weights)
{
  // The 1D rule the tensor products are built from
  constexpr unsigned int max_n1D = n_points_1D(FORTYTHIRD);
  const unsigned int n1D = n_points_1D(o);
  libmesh_assert_less_equal (n1D, max_n1D);

  switch (t)
    {
    case EDGE2:
    case EDGE3:
    case EDGE4:
      rule_1D(o, points, weights);
      return;

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
    case QUAD4:
    case QUADSHELL4:
    case QUAD8:
    case QUADSHELL8:
    case QUAD9:
    case QUADSHELL9:
      // We compute the 3D quadrature rule as a tensor
      // product of the 1D quadrature rule.
    case HEX8:
    case HEX20:
    case HEX27:
      {
        Point points1D[max_n1D];
        Real weights1D[max_n1D];
        rule_1D(o, points1D, weights1D);
        if (t == HEX8 || t == HEX20 || t == HEX27)
          tensor_product_hex(n1D, points1D, weights1D, points, weights);
        else
          tensor_product_quad(n1D, points1D, weights1D, points, weights);
        return;
      }

    case TRI3:
    case TRISHELL3:
    case TRI3SUBDIVISION:
    case TRI6:
    case TRI7:
      rule_tri(o, points, weights);
      return;

    case TET4:
    case TET10:
    case TET14:
      rule_tet(o, allow_negative_weights, points, weights);
      return;

      // We compute the 3D quadrature rule as a tensor
      // product of the 1D quadrature rule and a 2D
      // triangle quadrature rule
    case PRISM6:
    case PRISM15:
    case PRISM18:
    case PRISM20:
    case PRISM21:
      {
        Point points1D[max_n1D];
        Real weights1D[max_n1D];
        rule_1D(o, points1D, weights1D);

        // The triangle rule, which can have hundreds of points, goes in
        // the last n2D entries rather than in a copy: the product
        // writes those entries last, each one only after reading the
        // triangle point it replaces.
        const unsigned int n2D = n_points_tri(o);
        Point * points2D = points + (n1D - 1) * n2D;
        Real * weights2D = weights + (n1D - 1) * n2D;
        rule_tri(o, points2D, weights2D);
        tensor_product_prism(n1D, points1D, weights1D,
                             n2D, points2D, weights2D, points, weights);
        return;
      }

      // We compute the Pyramid rule as a conical product of a
      // Jacobi rule with alpha==2 on the interval [0,1] two 1D
      // Gauss rules with suitably modified points.  The idea comes
      // from: Stroud, A.H. "Approximate Calculation of Multiple
      // Integrals."
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
    case PYRAMID18:
      QConical::conical_product_pyramid(o, points, weights);
      return;

    default:
      libmesh_error_msg("Element type not supported!");
    }
}

} // namespace libMesh

#endif // LIBMESH_QUADRATURE_GAUSS_3D_H
