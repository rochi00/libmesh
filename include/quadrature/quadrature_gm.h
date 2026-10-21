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



#ifndef LIBMESH_QUADRATURE_GM_H
#define LIBMESH_QUADRATURE_GM_H

// Local includes
#include "libmesh/quadrature.h"

// C++ includes
#include <algorithm> // std::max

namespace libMesh
{

/**
 * This class implements the Grundmann-Moller quadrature rules for
 * tetrahedra.  The GM rules are well-defined for simplices of
 * arbitrary dimension and to any order, but the rules by Dunavant for
 * two-dimensional simplices are in general superior.  This is primarily
 * due to the fact that the GM rules contain a significant proportion
 * of negative weights, making them susceptible to round-off error
 * at high-order.
 *
 * The GM rules are interesting in 3D because they overlap with the
 * conical product rules at higher order while having significantly
 * fewer evaluation points, making them potentially much more
 * efficient.  The table below gives a comparison between the number
 * of points in a conical product (CP) rule and the GM rule of
 * equivalent order.  The GM rules are defined to be exact for
 * polynomials of degree d=2*s+1, s=0,1,2,3,... The table also gives
 * the percentage of each GM rule's weights which are negative.
 * The percentage of negative weights appears to approach 50, and the
 * amplification factor (a measure of the effect of round-off) defined
 * as
 *
 * amp. factor = \f$ \frac{1}{V} \sum \|w_i\|, \f$
 *
 * where V is the volume of the reference element, grows like exp(C*s).
 * (A rule with all positive weights has an amplification factor of
 * 1.0 by definition.)
 *
 * \verbatim
 * s   degree  n_pts(conical)  n_pts(GM)   % neg wts  amp. factor
 * ------------------------------------------------------------------------
 * 0   1       1               1            0.00      1.00e+00
 * 1   3       8               5           20.00      2.60e+00
 * 2   5       27              15          26.67      5.63e+00
 * 3   7       64              35          31.43      1.19e+01
 * 4   9       125             70          34.29      2.54e+01
 * 5   11      216             126         36.51      5.41e+01
 * 6   13      343             210         38.10      1.16e+02
 * 7   15      512             330         39.39      2.51e+02
 * 8   17      729             495         40.40      5.45e+02
 * 9   19      1000            715         41.26      1.19e+03
 * 10  21      1331            1001        41.96      2.59e+03
 * 11  23      1728            1365        42.56      5.68e+03
 * 12  25      2197            1820        43.08      1.25e+04
 * 13  27      2744            2380        43.53      2.75e+04
 * 14  29      3375            3060        43.92      6.07e+04
 * 15  31      4096            3876        44.27      1.34e+05
 * 16  33      4913            4845        44.58      2.97e+05
 * 17  35      5832            5985        44.86      6.59e+05 <= Conical rule has fewer points for degree >= 34
 * 18  37      6859            7315        45.11      1.46e+06
 * 19  39      8000            8855        45.34      3.25e+06
 * 20  41      9261            10626       45.55      7.23e+06
 * 21  43      10648           12650       45.74      1.61e+07
 * \endverbatim
 *
 * Reference:
 *    Axel Grundmann and Michael M\"{o}ller, "Invariant Integration
 *    Formulas for the N-Simplex by Combinatorial Methods," SIAM
 *    Journal on Numerical Analysis, Volume 15, Number 2, April 1978,
 *    pages 282-290.
 *
 * Reference LGPL Fortran90 code by John Burkardt can be found here:
 * http://people.scs.fsu.edu/~burkardt/f_src/gm_rules/gm_rules.html
 *
 * \author John W. Peterson
 * \date 2008
 * \brief Implements the quadrature rules of Grundmann and Moller in 2D and 3D.
 */
class QGrundmann_Moller final : public QBase
{
public:

  /**
   * Constructor.  Declares the order of the quadrature rule.
   */
  QGrundmann_Moller (unsigned int dim,
                     Order order=INVALID_ORDER) :
    QBase(dim,order)
  {
    if (dim == 1)
      init(EDGE2);
  }

  /**
   * Copy/move ctor, copy/move assignment operator, and destructor are
   * all explicitly defaulted for this simple class.
   */
  QGrundmann_Moller (const QGrundmann_Moller &) = default;
  QGrundmann_Moller (QGrundmann_Moller &&) = default;
  QGrundmann_Moller & operator= (const QGrundmann_Moller &) = default;
  QGrundmann_Moller & operator= (QGrundmann_Moller &&) = default;
  virtual ~QGrundmann_Moller() = default;

  /**
   * \returns \p QGRUNDMANN_MOLLER.
   */
  virtual QuadratureType type() const override;

  virtual std::unique_ptr<QBase> clone() const override;

  /**
   * \returns The number of points in the rule of index \p s in
   * dimension \p dim, which is (dim+1+s)! / (dim+1)! / s!.  In 3D,
   * this is 1/24 * Pi_{i=1}^4 (s+i); in 2D, 1/6 * Pi_{i=1}^3 (s+i).
   */
  static LIBMESH_DEVICE_INLINE constexpr
  unsigned int n_points_gm (const unsigned int s,
                            const unsigned int dim)
  { return dim==2 ? (s+3)*(s+2)*(s+1) / 6 : (s+4)*(s+3)*(s+2)*(s+1) / 24; }

  /**
   * Writes the points and weights of the rule of index \p s in
   * dimension \p dim, which integrates polynomials of degree 2*s+1
   * exactly, to \p points and \p weights, which must have room for
   * n_points_gm(s, dim) entries.  This is the rule the member
   * gm_rule() fills in; it is static so that device code, which
   * cannot construct a QGrundmann_Moller, evaluates the same rule.
   */
  static LIBMESH_DEVICE_INLINE
  void gm_rule (const unsigned int s,
                const unsigned int dim,
                Point * points,
                Real * weights);

private:

  /**
   * In 1D, use a Gauss rule.
   * In 2D, the GM product rule is only defined for Tris.
   * In 3D, the GM product rule is only defined for Tets.
   */
  virtual void init_1D () override;
  virtual void init_2D () override;
  virtual void init_3D () override;

  /**
   * This routine is called from init_2D() and init_3D().  It actually
   * fills the _points and _weights vectors for a given rule index, s
   * and dimension, dim.
   */
  void gm_rule(unsigned int s, unsigned int dim);
};



LIBMESH_DEVICE_INLINE
void QGrundmann_Moller::gm_rule(const unsigned int s,
                                const unsigned int dim,
                                Point * points,
                                Real * weights)
{
  // Only dim==2 or dim==3 is allowed
  libmesh_assert(dim==2 || dim==3);

  // A GM rule of index s can integrate polynomials of degree 2*s+1 exactly
  const unsigned int degree = 2*s+1;

  // (-1)^i -> This one flips sign at each iteration of the i-loop below.
  int one_pm=1;

  // Index into the arrays where we should start adding the next round of points/weights
  unsigned int offset=0;

  // Implement the GM formula 4.1 on page 286 of the paper
  for (unsigned int i=0; i<=s; ++i)
    {
      // Compute the weight for this i, being careful to avoid overflow.
      // This technique is borrowed from Burkardt's code as well.
      // Use once for each of the points obtained from the compositions below.
      Real weight = one_pm;

      // This for loop needs to run for dim, degree, or dim+degree-i iterations,
      // whichever is largest.
      const unsigned int weight_loop_index =
        std::max(dim, std::max(degree, degree+dim-i))+1;

      for (unsigned int j=1; j<weight_loop_index; ++j)
        {
          if (j <= degree) // Accumulate (d+n-2i)^d term
            weight *= static_cast<Real>(degree+dim-2*i);

          if (j <= 2*s) // Accumulate 2^{-2s}
            weight *= 0.5;

          if (j <= i) // Accumulate (i!)^{-1}
            weight /= static_cast<Real>(j);

          if (j <= degree+dim-i) // Accumulate ( (d+n-i)! )^{-1}
            weight /= static_cast<Real>(j);
        }

      // Each ordered composition of |beta| = s-i into dim+1 parts,
      // and each permutation of one, gives a point with this weight,
      // made from its first dim entries.  We generate them one at a
      // time in workspace, rather than storing them all, with a
      // routine originally due to:
      //
      // Albert Nijenhuis, Herbert Wilf,
      // Combinatorial Algorithms for Computers and Calculators,
      // Second Edition,
      // Academic Press, 1978,
      // ISBN: 0-12-519260-6,
      // LC: QA164.N54.
      //
      // The routine is deceptively simple: I still don't understand exactly
      // why it works, but it does.
      const unsigned int comp_s = s-i;

      // The first composition is always (comp_s,0,...,0)
      unsigned int workspace[4] = {comp_s, 0, 0, 0};

      // the value of the first non-zero entry
      unsigned int head_value=comp_s;

      // When head_index=-1, it refers to "off the front" of the array.  Therefore,
      // this needs to be a regular int rather than unsigned.  I initially tried to
      // do this with head_index unsigned and an else statement below, but then there
      // is the special case: (1,0,...,0) which does not work correctly.
      int head_index = -1;

      for (;;)
        {
          // We use the first dim entries of each composition to
          // construct an integration point.
          for (unsigned int j=0; j<dim; ++j)
            points[offset](j) =
              static_cast<Real>(2.*workspace[j] + 1.) /
              static_cast<Real>(  degree + dim - 2.*i     );

          weights[offset] = weight;
          ++offset;

          // At the end, all the entries will be in the final slot of workspace
          if (workspace[dim] == comp_s)
            break;

          // If the previous head value is still larger than 1, reset the index
          // to "off the front" of the array
          if (head_value > 1)
            head_index = -1;

          // Either move the index onto the front of the array or on to
          // the next value.
          head_index++;

          // Get current value of the head entry
          head_value = workspace[head_index];

          // Put a zero into the head_index of the array.  If head_index==0,
          // this will be overwritten in the next line with head_value-1.
          workspace[head_index] = 0;

          // The initial entry gets the current head value, minus 1.
          // If head_value > 1, the next loop iteration will start back
          // at workspace[0] again.
          libmesh_assert_greater (head_value, 0);
          workspace[0] = head_value - 1;

          // Increment the head+1 value
          workspace[head_index+1] += 1;
        }

      // Change sign for next iteration
      one_pm = -one_pm;
    }
}

} // namespace libMesh

#endif // LIBMESH_QUADRATURE_GM_H
