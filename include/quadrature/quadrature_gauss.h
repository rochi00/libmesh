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



#ifndef LIBMESH_QUADRATURE_GAUSS_H
#define LIBMESH_QUADRATURE_GAUSS_H

// Local includes
#include "libmesh/quadrature.h"
#include "libmesh/quadrature_gm.h" // QGrundmann_Moller::n_points_gm()

namespace libMesh
{

/**
 * This class implements specific orders of Gauss quadrature.  The
 * rules of Order \p p are capable of integrating polynomials of
 * degree \p p exactly.
 *
 * \author Benjamin Kirk
 * \author John W. Peterson
 * \date 2003
 * \brief Implements 1, 2, and 3D "Gaussian" quadrature rules.
 */
class QGauss : public QBase
{
public:

  /**
   * Constructor.  Declares the order of the quadrature rule.  We
   * explicitly call the \p init function in 1D since the other
   * tensor-product rules require this one.
   *
   * \note The element type, EDGE2, will not be used internally,
   * however if we called the function with INVALID_ELEM it would try
   * to be smart and return, thinking it had already done the work.
   */
  QGauss (unsigned int dim,
          Order order=INVALID_ORDER) :
    QBase(dim, order)
  {
    if (dim == 1)
      init(EDGE2);
  }

  /**
   * Copy/move ctor, copy/move assignment operator, and destructor are
   * all explicitly defaulted for this simple class.
   */
  QGauss (const QGauss &) = default;
  QGauss (QGauss &&) = default;
  QGauss & operator= (const QGauss &) = default;
  QGauss & operator= (QGauss &&) = default;
  virtual ~QGauss() = default;

  /**
   * \returns \p QGAUSS.
   */
  virtual QuadratureType type() const override;

  virtual std::unique_ptr<QBase> clone() const override;

  /**
   * \returns The number of points in the one-dimensional rule of
   * order \p o, or \p invalid_uint above \p FORTYTHIRD, the highest
   * order tabulated.  An n-point Gauss rule integrates degree 2n-1
   * exactly, so orders 2n-2 and 2n-1 share it.
   */
  static LIBMESH_DEVICE_INLINE constexpr
  unsigned int n_points_1D (const Order o)
  { return o <= FORTYTHIRD ? o/2 + 1 : invalid_uint; }

  /**
   * Writes the points and weights of the one-dimensional rule of
   * order \p o to \p points and \p weights, which must have room for
   * n_points_1D(o) entries, with all but the first coordinate of each
   * point zero.  This is the rule init_1D() fills in; it is static,
   * and defined in quadrature_gauss_1D.h, so that device code, which
   * cannot construct a QGauss, evaluates the same rule.
   */
  static LIBMESH_DEVICE_INLINE
  void rule_1D (const Order o,
                Point * points,
                Real * weights);

  /**
   * \returns The number of points in the triangle rule of order \p
   * o, or \p invalid_uint above \p FORTYTHIRD.  Orders without a
   * tabulated rule, 21 and those above 30, use the conical product
   * rule, which has n_points_1D(o)^2 points.
   */
  static LIBMESH_DEVICE_INLINE constexpr
  unsigned int n_points_tri (const Order o)
  {
    switch (o)
      {
      case CONSTANT:
      case FIRST:        return 1;
      case SECOND:       return 3;
      case THIRD:        return 4;
      case FOURTH:       return 6;
      case FIFTH:        return 7;
      case SIXTH:
      case SEVENTH:      return 12;
      case EIGHTH:       return 16;
      case NINTH:        return 19;
      case TENTH:        return 25;
      case ELEVENTH:     return 30;
      case TWELFTH:      return 33;
      case THIRTEENTH:   return 37;
      case FOURTEENTH:   return 42;
      case FIFTEENTH:    return 49;
      case SIXTEENTH:    return 55;
      case SEVENTEENTH:  return 63;
      case EIGHTEENTH:
      case NINETEENTH:   return 73;
      case TWENTIETH:    return 85;
      case TWENTYSECOND:
      case TWENTYTHIRD:
      case TWENTYFOURTH:
      case TWENTYFIFTH:  return 126;
      case TWENTYSIXTH:
      case TWENTYSEVENTH:
      case TWENTYEIGHTH:
      case TWENTYNINTH:
      case THIRTIETH:    return 175;
      default:
        return n_points_1D(o) == invalid_uint ?
          invalid_uint : n_points_1D(o) * n_points_1D(o);
      }
  }

  /**
   * Writes the points and weights of the triangle rule of order \p o
   * to \p points and \p weights, which must have room for
   * n_points_tri(o) entries, with the third coordinate of each point
   * zero.  This is the rule init_2D() fills in for every Tri type; it
   * is static, and defined in quadrature_gauss_2D.h, so that device
   * code, which cannot construct a QGauss, evaluates the same rule.
   */
  static LIBMESH_DEVICE_INLINE
  void rule_tri (const Order o,
                 Point * points,
                 Real * weights);

  /**
   * \returns The number of points in the tet rule of order \p o, or
   * \p invalid_uint above \p FORTYTHIRD.  With \p
   * allow_negative_weights, orders 3, 4, 7 and 8 have rules with a
   * negative weight, and orders 9 through 33 use the Grundmann-Moller
   * rule of index o/2; otherwise those orders, and every order above
   * 33, use the conical product rule, which has n_points_1D(o)^3
   * points.
   */
  static LIBMESH_DEVICE_INLINE constexpr
  unsigned int n_points_tet (const Order o,
                             const bool allow_negative_weights)
  {
    const unsigned int n_conical = n_points_1D(o) == invalid_uint ?
      invalid_uint : n_points_1D(o) * n_points_1D(o) * n_points_1D(o);

    switch (o)
      {
      case CONSTANT:
      case FIRST:   return 1;
      case SECOND:  return 4;
      case THIRD:   return allow_negative_weights ? 5 : n_conical;
      case FOURTH:  return allow_negative_weights ? 11 : 14;
      case FIFTH:   return 14;
      case SIXTH:   return 24;
      case SEVENTH:
      case EIGHTH:  return allow_negative_weights ? 45 : n_conical;
      default:
        return (allow_negative_weights && o < 34) ?
          QGrundmann_Moller::n_points_gm(o/2, 3) : n_conical;
      }
  }

  /**
   * Writes the points and weights of the tet rule of order \p o to \p
   * points and \p weights, which must have room for n_points_tet(o,
   * allow_negative_weights) entries.  This is the rule init_3D()
   * fills in for every Tet type, given \p allow_negative_weights as
   * allow_rules_with_negative_weights; it is static, and defined in
   * quadrature_gauss_3D.h, so that device code, which cannot
   * construct a QGauss, evaluates the same rule.
   */
  static LIBMESH_DEVICE_INLINE
  void rule_tet (const Order o,
                 const bool allow_negative_weights,
                 Point * points,
                 Real * weights);

private:

  virtual void init_1D () override;
  virtual void init_2D () override;
  virtual void init_3D () override;

  /**
   * The Dunavant rules are for triangles. This function takes
   * permutation points and weights in a specific format as input and
   * fills the _points and _weights vectors.
   */
  void dunavant_rule(const Real rule_data[][4],
                     const unsigned int n_pts);

  void dunavant_rule2(const Real * wts,
                      const Real * a,
                      const Real * b,
                      const unsigned int * permutation_ids,
                      const unsigned int n_wts);

  /**
   * The Keast rules are for tets. This function takes permutation
   * points and weights in a specific format as input and fills the
   * _points and _weights vectors.
   */
  void keast_rule(const Real rule_data[][4],
                  const unsigned int n_pts);

  /**
   * As the members of the same names, but writing to \p points and
   * \p weights, which have room for \p n_out entries, so that the
   * static triangle and tet rules can build the same points and
   * weights in device code.  The members call these.
   */
  static LIBMESH_DEVICE_INLINE
  void dunavant_rule(const Real rule_data[][4],
                     const unsigned int n_pts,
                     const unsigned int n_out,
                     Point * points,
                     Real * weights);

  static LIBMESH_DEVICE_INLINE
  void keast_rule(const Real rule_data[][4],
                  const unsigned int n_pts,
                  const unsigned int n_out,
                  Point * points,
                  Real * weights);

  /**
   * As the member dunavant_rule2(), but writing to \p points and \p
   * weights, which must have room for the sum of \p permutation_ids.
   */
  static LIBMESH_DEVICE_INLINE
  void dunavant_rule2(const Real * wts,
                      const Real * a,
                      const Real * b,
                      const unsigned int * permutation_ids,
                      const unsigned int n_wts,
                      Point * points,
                      Real * weights);
};



LIBMESH_DEVICE_INLINE
void QGauss::keast_rule(const Real rule_data[][4],
                        const unsigned int n_pts,
                        const unsigned int n_out,
                        Point * points,
                        Real * weights)
{
  // Like the Dunavant rule, the input data should have 4 columns.  These columns
  // have the following format and implied permutations (w=weight).
  // {a, 0, 0, w} = 1-permutation  (a,a,a)
  // {a, b, 0, w} = 4-permutation  (a,b,b), (b,a,b), (b,b,a), (b,b,b)
  // {a, 0, b, w} = 6-permutation  (a,a,b), (a,b,b), (b,b,a), (b,a,b), (b,a,a), (a,b,a)
  // {a, b, c, w} = 12-permutation (a,a,b), (a,a,c), (b,a,a), (c,a,a), (a,b,a), (a,c,a)
  //                               (a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)

  // Always insert into the points & weights vector relative to the offset
  unsigned int offset=0;


  for (unsigned int p=0; p<n_pts; ++p)
    {

      // There must always be a non-zero entry to start the row
      libmesh_assert_not_equal_to (rule_data[p][0], static_cast<Real>(0.0));

      // A zero weight may imply you did not set up the raw data correctly
      libmesh_assert_not_equal_to (rule_data[p][3], static_cast<Real>(0.0));

      // What kind of point is this?
      // One non-zero entry in first 3 cols   ? 1-perm (centroid) point = 1
      // Two non-zero entries in first 3 cols ? 3-perm point            = 3
      // Three non-zero entries               ? 6-perm point            = 6
      unsigned int pointtype=1;

      if (rule_data[p][1] != static_cast<Real>(0.0))
        {
          if (rule_data[p][2] != static_cast<Real>(0.0))
            pointtype = 12;
          else
            pointtype = 4;
        }
      else
        {
          // The second entry is zero.  What about the third?
          if (rule_data[p][2] != static_cast<Real>(0.0))
            pointtype = 6;
        }


      switch (pointtype)
        {
        case 1:
          {
            // Be sure we have enough space to insert this point
            libmesh_assert_less (offset + 0, n_out);

            const Real a = rule_data[p][0];

            // The point has only a single permutation (the centroid!)
            points[offset  + 0] = Point(a,a,a);

            // The weight is always the last entry in the row.
            weights[offset + 0] = rule_data[p][3];

            offset += pointtype;
            break;
          }

        case 4:
          {
            // Be sure we have enough space to insert these points
            libmesh_assert_less (offset + 3, n_out);

            const Real a  = rule_data[p][0];
            const Real b  = rule_data[p][1];
            const Real wt = rule_data[p][3];

            // Here it's understood the second entry is to be used twice, and
            // thus there are three possible permutations.
            points[offset + 0] = Point(a,b,b);
            points[offset + 1] = Point(b,a,b);
            points[offset + 2] = Point(b,b,a);
            points[offset + 3] = Point(b,b,b);

            for (unsigned int j=0; j<pointtype; ++j)
              weights[offset + j] = wt;

            offset += pointtype;
            break;
          }

        case 6:
          {
            // Be sure we have enough space to insert these points
            libmesh_assert_less (offset + 5, n_out);

            const Real a  = rule_data[p][0];
            const Real b  = rule_data[p][2];
            const Real wt = rule_data[p][3];

            // Three individual entries with six permutations.
            points[offset + 0] = Point(a,a,b);
            points[offset + 1] = Point(a,b,b);
            points[offset + 2] = Point(b,b,a);
            points[offset + 3] = Point(b,a,b);
            points[offset + 4] = Point(b,a,a);
            points[offset + 5] = Point(a,b,a);

            for (unsigned int j=0; j<pointtype; ++j)
              weights[offset + j] = wt;

            offset += pointtype;
            break;
          }


        case 12:
          {
            // Be sure we have enough space to insert these points
            libmesh_assert_less (offset + 11, n_out);

            const Real a  = rule_data[p][0];
            const Real b  = rule_data[p][1];
            const Real c  = rule_data[p][2];
            const Real wt = rule_data[p][3];

            // Three individual entries with six permutations.
            points[offset + 0] = Point(a,a,b);  points[offset + 6]  = Point(a,b,c);
            points[offset + 1] = Point(a,a,c); points[offset + 7]  = Point(a,c,b);
            points[offset + 2] = Point(b,a,a); points[offset + 8]  = Point(b,a,c);
            points[offset + 3] = Point(c,a,a); points[offset + 9]  = Point(b,c,a);
            points[offset + 4] = Point(a,b,a); points[offset + 10] = Point(c,a,b);
            points[offset + 5] = Point(a,c,a); points[offset + 11] = Point(c,b,a);

            for (unsigned int j=0; j<pointtype; ++j)
              weights[offset + j] = wt;

            offset += pointtype;
            break;
          }

        default:
          libmesh_error_msg("Don't know what to do with this many permutation points!");
        }

    }

}



// A number of different rules for triangles can be described by
// permutations of the following types of points:
// I:   "1"-permutation, (1/3,1/3)  (single point only)
// II:   3-permutation, (a,a,1-2a)
// III:  6-permutation, (a,b,1-a-b)
// The weights for a given set of permutations are all the same.
LIBMESH_DEVICE_INLINE
void QGauss::dunavant_rule2(const Real * wts,
                            const Real * a,
                            const Real * b,
                            const unsigned int * permutation_ids,
                            const unsigned int n_wts,
                            Point * points,
                            Real * weights)
{
  // Always insert into the points & weights vector relative to the offset
  unsigned int offset=0;

  for (unsigned int p=0; p<n_wts; ++p)
    {
      switch (permutation_ids[p])
        {
        case 1:
          {
            // The point has only a single permutation (the centroid!)
            // So we don't even need to look in the a or b arrays.
            points [offset  + 0] = Point(Real(1)/3, Real(1)/3);
            weights[offset + 0] = wts[p];

            offset += 1;
            break;
          }


        case 3:
          {
            // For this type of rule, don't need to look in the b array.
            points[offset + 0] = Point(a[p],         a[p]);         // (a,a)
            points[offset + 1] = Point(a[p],         1-2*a[p]); // (a,1-2a)
            points[offset + 2] = Point(1-2*a[p], a[p]);         // (1-2a,a)

            for (unsigned int j=0; j<3; ++j)
              weights[offset + j] = wts[p];

            offset += 3;
            break;
          }

        case 6:
          {
            // This type of point uses all 3 arrays...
            points[offset + 0] = Point(a[p], b[p]);
            points[offset + 1] = Point(b[p], a[p]);
            points[offset + 2] = Point(a[p], 1-a[p]-b[p]);
            points[offset + 3] = Point(1-a[p]-b[p], a[p]);
            points[offset + 4] = Point(b[p], 1-a[p]-b[p]);
            points[offset + 5] = Point(1-a[p]-b[p], b[p]);

            for (unsigned int j=0; j<6; ++j)
              weights[offset + j] = wts[p];

            offset += 6;
            break;
          }

        default:
          // Device code can only report a literal, so this no longer
          // names the bad id.
          libmesh_error_msg("Unknown permutation id!");
        }
    }

}



LIBMESH_DEVICE_INLINE
void QGauss::dunavant_rule(const Real rule_data[][4],
                           const unsigned int n_pts,
                           const unsigned int n_out,
                           Point * points,
                           Real * weights)
{
  // The input data array has 4 columns.  The first 3 are the permutation points.
  // The last column is the weights for a given set of permutation points.  A zero
  // in two of the first 3 columns implies the point is a 1-permutation (centroid).
  // A zero in one of the first 3 columns implies the point is a 3-permutation.
  // Otherwise each point is assumed to be a 6-permutation.

  // Always insert into the points & weights vector relative to the offset
  unsigned int offset=0;


  for (unsigned int p=0; p<n_pts; ++p)
    {

      // There must always be a non-zero entry to start the row
      libmesh_assert_not_equal_to ( rule_data[p][0], static_cast<Real>(0.0) );

      // A zero weight may imply you did not set up the raw data correctly
      libmesh_assert_not_equal_to ( rule_data[p][3], static_cast<Real>(0.0) );

      // What kind of point is this?
      // One non-zero entry in first 3 cols   ? 1-perm (centroid) point = 1
      // Two non-zero entries in first 3 cols ? 3-perm point            = 3
      // Three non-zero entries               ? 6-perm point            = 6
      unsigned int pointtype=1;

      if (rule_data[p][1] != static_cast<Real>(0.0))
        {
          if (rule_data[p][2] != static_cast<Real>(0.0))
            pointtype = 6;
          else
            pointtype = 3;
        }

      switch (pointtype)
        {
        case 1:
          {
            // Be sure we have enough space to insert this point
            libmesh_assert_less (offset + 0, n_out);

            // The point has only a single permutation (the centroid!)
            points[offset  + 0] = Point(rule_data[p][0], rule_data[p][0]);

            // The weight is always the last entry in the row.
            weights[offset + 0] = rule_data[p][3];

            offset += 1;
            break;
          }

        case 3:
          {
            // Be sure we have enough space to insert these points
            libmesh_assert_less (offset + 2, n_out);

            // Here it's understood the second entry is to be used twice, and
            // thus there are three possible permutations.
            points[offset + 0] = Point(rule_data[p][0], rule_data[p][1]);
            points[offset + 1] = Point(rule_data[p][1], rule_data[p][0]);
            points[offset + 2] = Point(rule_data[p][1], rule_data[p][1]);

            for (unsigned int j=0; j<3; ++j)
              weights[offset + j] = rule_data[p][3];

            offset += 3;
            break;
          }

        case 6:
          {
            // Be sure we have enough space to insert these points
            libmesh_assert_less (offset + 5, n_out);

            // Three individual entries with six permutations.
            points[offset + 0] = Point(rule_data[p][0], rule_data[p][1]);
            points[offset + 1] = Point(rule_data[p][0], rule_data[p][2]);
            points[offset + 2] = Point(rule_data[p][1], rule_data[p][0]);
            points[offset + 3] = Point(rule_data[p][1], rule_data[p][2]);
            points[offset + 4] = Point(rule_data[p][2], rule_data[p][0]);
            points[offset + 5] = Point(rule_data[p][2], rule_data[p][1]);

            for (unsigned int j=0; j<6; ++j)
              weights[offset + j] = rule_data[p][3];

            offset += 6;
            break;
          }

        default:
          libmesh_error_msg("Don't know what to do with this many permutation points!");
        }
    }
}

} // namespace libMesh

#endif // LIBMESH_QUADRATURE_GAUSS_H
