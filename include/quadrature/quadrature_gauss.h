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
  static LIBMESH_DEVICE_INLINE
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
};

} // namespace libMesh

#endif // LIBMESH_QUADRATURE_GAUSS_H
