#include <libmesh/elem.h>
#include <libmesh/enum_fe_family.h>
#include <libmesh/enum_order.h>
#include <libmesh/fe_interface.h>
#include <libmesh/fe_shape.h>
#include <libmesh/fe_type.h>
#include <libmesh/vector_value.h>

#ifdef LIBMESH_HAVE_METAPHYSICL
#include <metaphysicl/dualnumber.h>
#include <metaphysicl/numberarray.h>
#endif

#include "libmesh_cppunit.h"

#include <cmath>
#include <cstring>
#include <memory>

using namespace libMesh;

class FEShapeTest : public CppUnit::TestCase
{
  /**
   * The goal of this test is to pin down fe_shape(), fe_shape_deriv()
   * and fe_shape_second_deriv() for every LAGRANGE, L2_LAGRANGE and
   * MONOMIAL element FEInterface evaluates: on Reals they give
   * FEInterface's values bit for bit, and on dual numbers they give the
   * same values along with derivatives that agree with the next
   * function down, which is what mortar's Newton projection relies on.
   *
   * The dual values agree only to rounding: a compiler may fuse a*b + c
   * into one multiply-add in a Real expression, but not across the
   * separate operator calls a dual number makes.
   */
public:
  LIBMESH_CPPUNIT_TEST_SUITE( FEShapeTest );

  CPPUNIT_TEST( testRealMatchesFEInterface );
#ifdef LIBMESH_HAVE_METAPHYSICL
  CPPUNIT_TEST( testDualDerivatives );
#endif

  CPPUNIT_TEST_SUITE_END();

protected:
  struct Element
  {
    unsigned int dim;
    FEFamily family;
    ElemType type;
    Order order;
    std::unique_ptr<Elem> elem;
    unsigned int n_shapes;
  };

  // Every element FEInterface evaluates, up to FIFTH order
  static std::vector<Element> elements()
  {
    const ElemType types[] = {EDGE2, EDGE3, EDGE4,
#if LIBMESH_DIM > 1
                              TRI3, TRI6, TRI7, QUAD4, QUAD8, QUAD9,
                              TRISHELL3, QUADSHELL4, QUADSHELL8, QUADSHELL9,
#endif
#if LIBMESH_DIM > 2
                              TET4, TET10, TET14, HEX8, HEX20, HEX27,
                              PRISM6, PRISM15, PRISM18, PRISM20, PRISM21,
                              PYRAMID5, PYRAMID13, PYRAMID14, PYRAMID18,
#endif
                             };
    std::vector<Element> result;
    for (const ElemType t : types)
      for (const FEFamily family : {LAGRANGE, L2_LAGRANGE, MONOMIAL})
        for (int o = CONSTANT; o <= FIFTH; ++o)
          {
            const FEType fe_t(Order(o), family);
            if (o > int(FEInterface::max_order(fe_t, t)))
              continue;

            // Lagrange and monomial shapes only need the element's
            // type, so it needs no nodes.
            Element e{0, family, t, Order(o), Elem::build(t), 0};
            e.dim = e.elem->dim();
            try
              {
                e.n_shapes = FEInterface::n_shape_functions(fe_t, e.elem.get());
                for (unsigned int i = 0; i != e.n_shapes; ++i)
                  FEInterface::shape(fe_t, e.elem.get(), i, Point(0.2, 0.15, 0.1));
              }
            catch (const std::exception &)
              {
                // FEInterface rejects this (type, order) pair
                continue;
              }
            result.push_back(std::move(e));
          }
    return result;
  }

  static std::vector<Point> points()
  {
    return {Point(0.2, 0.15, 0.1), Point(1./3, 1./3, 0.25),
            Point(-0.4, 0.3, 0.2), Point(1.4, -0.3, 0.6)};
  }

public:
  void setUp() {}

  void tearDown() {}

  void testRealMatchesFEInterface()
  {
    LOG_UNIT_TEST;

    for (const Element & e : elements())
      for (const Point & p : points())
        for (unsigned int i = 0; i != e.n_shapes; ++i)
          {
            const FEType fe_t(e.order, e.family);
            const Real host = FEInterface::shape(fe_t, e.elem.get(), i, p);
            const Real shared = fe_shape(e.dim, e.family, e.type, e.order, i, p);
            CPPUNIT_ASSERT(!std::memcmp(&host, &shared, sizeof(Real)));

            for (unsigned int j = 0; j != e.dim; ++j)
              {
                const Real host_d = FEInterface::shape_deriv(fe_t, e.elem.get(), i, j, p);
                const Real shared_d = fe_shape_deriv(e.dim, e.family, e.type, e.order, i, j, p);
                CPPUNIT_ASSERT(!std::memcmp(&host_d, &shared_d, sizeof(Real)));
              }

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            for (unsigned int j = 0; j != e.dim*(e.dim+1)/2; ++j)
              {
                const Real host_d2 = FEInterface::shape_second_deriv(fe_t, e.elem.get(), i, j, p);
                const Real shared_d2 = fe_shape_second_deriv(e.dim, e.family, e.type, e.order, i, j, p);
                CPPUNIT_ASSERT(!std::memcmp(&host_d2, &shared_d2, sizeof(Real)));
              }
#endif
          }
  }

#ifdef LIBMESH_HAVE_METAPHYSICL
  void testDualDerivatives()
  {
    LOG_UNIT_TEST;

    typedef MetaPhysicL::DualNumber<Real, MetaPhysicL::NumberArray<3, Real>> ADReal;

    for (const Element & e : elements())
      for (const Point & p : points())
        {
          // AD and the hand-written derivatives order their arithmetic
          // differently, so they only agree to rounding, except on
          // PYRAMID18, whose derivatives are central differences of its
          // shapes (and its second derivatives, differences of those):
          // good to about 1e-10 and, with rounding divided by both
          // steps, 1e-5.
          const bool fdm = (e.type == PYRAMID18);
          const Real tol = TOLERANCE * TOLERANCE;
          const Real d_tol = fdm ? 1e-8 : tol;
          const Real d2_tol = fdm ? std::sqrt(TOLERANCE) : tol;

          // The point, seeded so that derivatives()[d] is d/d(p(d))
          VectorValue<ADReal> p_ad;
          for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
            {
              p_ad(d) = p(d);
              p_ad(d).derivatives()[d] = 1;
            }

          for (unsigned int i = 0; i != e.n_shapes; ++i)
            {
              const ADReal phi = fe_shape(e.dim, e.family, e.type, e.order, i, p_ad);
              const Real value = fe_shape(e.dim, e.family, e.type, e.order, i, p);
              LIBMESH_ASSERT_FP_EQUAL(value, phi.value(), tol * (1 + std::abs(value)));

              for (unsigned int j = 0; j != e.dim; ++j)
                {
                  const Real dphi = fe_shape_deriv(e.dim, e.family, e.type, e.order, i, j, p);
                  LIBMESH_ASSERT_FP_EQUAL(dphi, phi.derivatives()[j], d_tol * (1 + std::abs(dphi)));

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                  // libMesh numbers second derivatives xx, xy, yy, xz, yz, zz
                  const ADReal dphi_ad = fe_shape_deriv(e.dim, e.family, e.type, e.order, i, j, p_ad);
                  for (unsigned int k = j; k != e.dim; ++k)
                    {
                      const Real d2phi = fe_shape_second_deriv(e.dim, e.family, e.type, e.order,
                                                               i, k*(k+1)/2 + j, p);
                      LIBMESH_ASSERT_FP_EQUAL(d2phi, dphi_ad.derivatives()[k], d2_tol * (1 + std::abs(d2phi)));
                    }
#endif
                }
            }
        }
  }
#endif // LIBMESH_HAVE_METAPHYSICL
};

CPPUNIT_TEST_SUITE_REGISTRATION( FEShapeTest );
