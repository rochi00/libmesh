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


#ifndef LIBMESH_QUADRATURE_CONICAL_H
#define LIBMESH_QUADRATURE_CONICAL_H

// Local includes
#include "libmesh/quadrature.h"
#include "libmesh/quadrature_gauss_1D.h"
#include "libmesh/quadrature_jacobi_1D.h"

namespace libMesh
{

/**
 * This class implements the so-called conical product quadrature
 * rules for Tri and Tet elements. These rules are generally
 * non-optimal in the number of evaluation points, but have the following nice
 * properties:
 * .) All positive weights.
 * .) Easily constructed for any order given the underlying 1D Gauss
 *    and Jacobi quadrature rules.
 * The construction of these rules is given by e.g.
 * A. H. Stroud, "Approximate Calculation of Multiple Integrals.", 1972.
 *
 * \author John W. Peterson
 * \date 2008
 * \brief Conical product quadrature rules for Tri and Tet elements.
 */
class QConical final : public QBase
{
public:

  /**
   * Constructor.  Declares the order of the quadrature rule.
   */
  QConical (unsigned int d,
            Order o=INVALID_ORDER) :
    QBase(d,o)
  {
    if (d == 1)
      init(EDGE2);
  }

  /**
   * Copy/move ctor, copy/move assignment operator, and destructor are
   * all explicitly defaulted for this simple class.
   */
  QConical (const QConical &) = default;
  QConical (QConical &&) = default;
  QConical & operator= (const QConical &) = default;
  QConical & operator= (QConical &&) = default;
  virtual ~QConical() = default;

  /**
   * \returns The QuadratureType for this class.
   */
  virtual QuadratureType type() const override;

  virtual std::unique_ptr<QBase> clone() const override;

  /**
   * Writes the order \p o conical product rule for a Tri to \p points
   * and \p weights, which must have room for QGauss::n_points_1D(o)^2
   * entries, with every coordinate but the first two of each point
   * zero.  This is the rule the member conical_product_tri() fills
   * in; it is static so that device code, which cannot construct a
   * QConical, evaluates the same rule.
   */
  static LIBMESH_DEVICE_INLINE
  void conical_product_tri(const Order o,
                           Point * points,
                           Real * weights);

  /**
   * As the static conical_product_tri(), but for a Tet, writing all
   * three coordinates, with room for QGauss::n_points_1D(o)^3 entries.
   */
  static LIBMESH_DEVICE_INLINE
  void conical_product_tet(const Order o,
                           Point * points,
                           Real * weights);

  /**
   * As the static conical_product_tri(), but for a Pyramid, writing
   * all three coordinates, with room for QGauss::n_points_1D(o)^3
   * entries.
   */
  static LIBMESH_DEVICE_INLINE
  void conical_product_pyramid(const Order o,
                               Point * points,
                               Real * weights);

private:

  /**
   * The most points in any 1D factor rule: the static product rules
   * build their factors in stack arrays of this size.
   */
  static constexpr unsigned int max_np = QGauss::n_points_1D(FORTYTHIRD);

  /**
   * In 1D, use a Gauss rule.
   * In 2D, the conical product rule is only defined for Tris.
   * In 3D, the conical product rule is only defined for Tets.
   */
  virtual void init_1D () override;
  virtual void init_2D () override;
  virtual void init_3D () override;

  /**
   * Implementation of conical product rule for a Tri in 2D of
   * order get_order().
   */
  void conical_product_tri();

  /**
   * Implementation of conical product rule for a Tet in 3D of
   * order get_order().
   */
  void conical_product_tet();

  /**
   * Implementation of conical product rule for a Pyramid in 3D of
   * order get_order().
   */
  void conical_product_pyramid();
};



// Builds and scales a Gauss rule and a Jacobi rule.
// Then combines them to compute points and weights
// of a 2D conical product rule.
LIBMESH_DEVICE_INLINE
void QConical::conical_product_tri(const Order o,
                                   Point * points,
                                   Real * weights)
{
  Point gauss_points[max_np], jac_points[max_np];
  Real gauss_weights[max_np], jac_weights[max_np];

  // Both rules have the same number of points.
  const unsigned int np = QGauss::n_points_1D(o);
  libmesh_assert_less_equal (np, max_np);

  QGauss::rule_1D(o, gauss_points, gauss_weights);
  QJacobi::rule_1D(1, 0, o, jac_points, jac_weights);

  // The Gauss rule needs to be scaled to [0,1]
  std::pair<Real, Real> old_range(-1, 1);
  std::pair<Real, Real> new_range( 0, 1);
  scale(old_range,
        new_range,
        np, gauss_points, gauss_weights);

  // Now construct the points and weights for the conical product rule.

  // Both rules should be between x=0 and x=1
  libmesh_assert_greater_equal (gauss_points[0](0), 0.0);
  libmesh_assert_less_equal (gauss_points[np-1](0), 1.0);
  libmesh_assert_greater_equal (jac_points[0](0), 0.0);
  libmesh_assert_less_equal (jac_points[np-1](0), 1.0);

  // Compute the conical product
  unsigned int gp = 0;
  for (unsigned int i=0; i<np; i++)
    for (unsigned int j=0; j<np; j++)
      {
        points[gp](0) = jac_points[j](0);                             //s[j];
        points[gp](1) = gauss_points[i](0) * (1.-jac_points[j](0));  //r[i]*(1.-s[j]);
        weights[gp]   = gauss_weights[i] * jac_weights[j];            //A[i]*B[j];
        gp++;
      }
}



// Builds and scales a Gauss rule and a Jacobi rule.
// Then combines them to compute points and weights
// of a 3D conical product rule for the Tet.
LIBMESH_DEVICE_INLINE
void QConical::conical_product_tet(const Order o,
                                   Point * points,
                                   Real * weights)
{
  Point gauss_points[max_np], jacA_points[max_np], jacB_points[max_np];
  Real gauss_weights[max_np], jacA_weights[max_np], jacB_weights[max_np];

  // All rules have the same number of points.
  const unsigned int np = QGauss::n_points_1D(o);
  libmesh_assert_less_equal (np, max_np);

  QGauss::rule_1D(o, gauss_points, gauss_weights);
  QJacobi::rule_1D(/*alpha=*/1, /*beta=*/0, o, jacA_points, jacA_weights);
  QJacobi::rule_1D(/*alpha=*/2, /*beta=*/0, o, jacB_points, jacB_weights);

  // The Gauss rule needs to be scaled to [0,1]
  std::pair<Real, Real> old_range(-1, 1);
  std::pair<Real, Real> new_range( 0, 1);
  scale(old_range,
        new_range,
        np, gauss_points, gauss_weights);

  // Now construct the points and weights for the conical product rule.

  // All rules should be between x=0 and x=1
  libmesh_assert_greater_equal (gauss_points[0](0), 0.0);
  libmesh_assert_less_equal (gauss_points[np-1](0), 1.0);
  libmesh_assert_greater_equal (jacA_points[0](0), 0.0);
  libmesh_assert_less_equal (jacA_points[np-1](0), 1.0);
  libmesh_assert_greater_equal (jacB_points[0](0), 0.0);
  libmesh_assert_less_equal (jacB_points[np-1](0), 1.0);

  // Compute the conical product
  unsigned int gp = 0;
  for (unsigned int i=0; i<np; i++)
    for (unsigned int j=0; j<np; j++)
      for (unsigned int k=0; k<np; k++)
        {
          points[gp](0) = jacB_points[k](0);                                                     //t[k];
          points[gp](1) = jacA_points[j](0)  * (1.-jacB_points[k](0));                           //s[j]*(1.-t[k]);
          points[gp](2) = gauss_points[i](0) * (1.-jacA_points[j](0)) * (1.-jacB_points[k](0));  //r[i]*(1.-s[j])*(1.-t[k]);
          weights[gp]   = gauss_weights[i]   * jacA_weights[j]        * jacB_weights[k];         //A[i]*B[j]*C[k];
          gp++;
        }
}



// Builds a Gauss rule and a Jacobi rule.
// Then combines them to compute points and weights
// of a 3D conical product rule for the Pyramid.  The integral
// over the reference Pyramid can be written (in LaTeX notation) as:
//
// If := \int_0^1 dz \int_{-(1-z)}^{(1-z)} dy \int_{-(1-z)}^{(1-z)} f(x,y,z) dx (1)
//
// (Imagine a stack of infinitely thin squares which decrease in size as
//  you approach the apex.)  Under the transformation of variables:
//
// z=w
// y=(1-z)v
// x=(1-z)u,
//
// The Jacobian determinant of this transformation is |J|=(1-w)^2, and
// the integral itself is transformed to:
//
// If = \int_0^1 (1-w)^2 dw \int_{-1}^{1} dv \int_{-1}^{1} f((1-w)u, (1-w)v, w) du (2)
//
// The integral can now be approximated by the product of three 1D quadrature rules:
// A Jacobi rule with alpha==2, beta==0 in w, and Gauss rules in v and u.  In this way
// we can obtain 3D rules to any order for which the 1D rules exist.
LIBMESH_DEVICE_INLINE
void QConical::conical_product_pyramid(const Order o,
                                       Point * points,
                                       Real * weights)
{
  Point gauss_points[max_np], jac_points[max_np];
  Real gauss_weights[max_np], jac_weights[max_np];

  // These rules have the same number of points
  const unsigned int np = QGauss::n_points_1D(o);
  libmesh_assert_less_equal (np, max_np);

  QGauss::rule_1D(o, gauss_points, gauss_weights);
  QJacobi::rule_1D(2, 0, o, jac_points, jac_weights);

  // Compute the conical product
  unsigned int q = 0;
  for (unsigned int i=0; i<np; ++i)
    for (unsigned int j=0; j<np; ++j)
      for (unsigned int k=0; k<np; ++k, ++q)
        {
          const Real xi=gauss_points[i](0);
          const Real yj=gauss_points[j](0);
          const Real zk=jac_points[k](0);

          points[q](0) = (1.-zk) * xi;
          points[q](1) = (1.-zk) * yj;
          points[q](2) = zk;
          weights[q]   = gauss_weights[i] * gauss_weights[j] * jac_weights[k];
        }
}

} // namespace libMesh

#endif // LIBMESH_QUADRATURE_CONICAL_H
