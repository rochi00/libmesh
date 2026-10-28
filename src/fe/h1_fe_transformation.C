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

#include "libmesh/fe_interface.h"
#include "libmesh/h1_fe_transformation.h"
#include "libmesh/tensor_value.h"
#include "libmesh/elem.h"
#include "libmesh/int_range.h"
#include "libmesh/fe_map_point.h"

// C++ includes
#include <algorithm> // std::copy

namespace
{
using namespace libMesh;

// Point p of map's inverse map derivatives, as many as the element
// has reference directions, gathered from its per-quantity vectors
// into the form the shared fe_physical_*derivs*() functions take.
FEMapPoint<> map_point(const FEMap & map,
                       const unsigned int dim,
                       const std::size_t p,
                       const bool second_derivs)
{
  const std::vector<Real> * dxyz[] =
    {&map.get_dxidx(), &map.get_dxidy(), &map.get_dxidz(),
     &map.get_detadx(), &map.get_detady(), &map.get_detadz(),
     &map.get_dzetadx(), &map.get_dzetady(), &map.get_dzetadz()};

  FEMapPoint<> m;
  RealVectorValue * grads[] = {&m.dxi, &m.deta, &m.dzeta};
  for (unsigned int j=0; j != dim; ++j)
    *grads[j] = RealVectorValue((*dxyz[3*j])[p], (*dxyz[3*j+1])[p], (*dxyz[3*j+2])[p]);

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  if (second_derivs)
    {
      const std::vector<std::vector<Real>> * d2xyz[] =
        {&map.get_d2xidxyz2(), &map.get_d2etadxyz2(), &map.get_d2zetadxyz2()};
      Real * second[] = {m.d2xi, m.d2eta, m.d2zeta};
      for (unsigned int j=0; j != dim; ++j)
        std::copy((*d2xyz[j])[p].begin(), (*d2xyz[j])[p].end(), second[j]);
    }
#else
  libmesh_ignore(second_derivs);
#endif

  return m;
}
}

namespace libMesh
{
template<typename OutputShape>
void H1FETransformation<OutputShape>::init_map_phi(const FEGenericBase<OutputShape> & /* fe */ ) const
{
  // Nothing needed
}



template<typename OutputShape>
void H1FETransformation<OutputShape>::init_map_dphi(const FEGenericBase<OutputShape> & fe) const
{
  fe.get_fe_map().get_dxidx();
}



template<typename OutputShape>
void H1FETransformation<OutputShape>::init_map_d2phi(const FEGenericBase<OutputShape> & fe) const
{
  fe.get_fe_map().get_dxidx();
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  fe.get_fe_map().get_d2xidxyz2();
#endif
}

template<typename OutputShape>
void H1FETransformation<OutputShape>::map_phi( const unsigned int dim,
                                               const Elem * const elem,
                                               const std::vector<Point> & qp,
                                               const FEGenericBase<OutputShape> & fe,
                                               std::vector<std::vector<OutputShape>> & phi,
                                               const bool add_p_level) const
{
  FEInterface::all_shapes<OutputShape>(dim, fe.get_fe_type(), elem, qp, phi, add_p_level);
}


template<typename OutputShape>
void H1FETransformation<OutputShape>::map_dphi( const unsigned int dim,
                                                const Elem * const,
                                                const std::vector<Point> &,
                                                const FEGenericBase<OutputShape> & fe,
                                                std::vector<std::vector<typename FEGenericBase<OutputShape>::OutputGradient>> & dphi,
                                                std::vector<std::vector<OutputShape>> & dphidx,
                                                std::vector<std::vector<OutputShape>> & dphidy,
                                                std::vector<std::vector<OutputShape>> & dphidz ) const
{
  switch(dim)
    {
    case 0: // No derivatives in 0D
      {
        for (auto i : index_range(dphi))
          for (auto p : index_range(dphi[i]))
            dphi[i][p] = 0.;
        break;
      }

    case 1:
    case 2:
    case 3:
      {
        // Shape function derivatives in reference space
        const std::vector<std::vector<OutputShape>> * dphiref[] =
          {&fe.get_dphidxi(), &fe.get_dphideta(), &fe.get_dphidzeta()};

        for (auto p : make_range(dphi.empty() ? std::size_t(0) : dphi[0].size()))
          {
            const FEMapPoint<> m = map_point(fe.get_fe_map(), dim, p, false);

            for (auto i : index_range(dphi))
              {
                OutputShape dphi_ip[3], d[3];
                for (unsigned int j=0; j != dim; ++j)
                  dphi_ip[j] = (*dphiref[j])[i][p];

                fe_physical_derivs(dim, dphi_ip, m, d);

                dphi[i][p].slice(0) = dphidx[i][p] = d[0];
#if LIBMESH_DIM>1
                dphi[i][p].slice(1) = dphidy[i][p] = d[1];
#endif
#if LIBMESH_DIM>2
                dphi[i][p].slice(2) = dphidz[i][p] = d[2];
#endif
              }
          }
        break;
      }

    default:
      libmesh_error_msg("Invalid dim = " << dim);
    } // switch(dim)
}

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
template<typename OutputShape>
void H1FETransformation<OutputShape>::map_d2phi(const unsigned int dim,
                                                const std::vector<Point> &,
                                                const FEGenericBase<OutputShape> & fe,
                                                std::vector<std::vector<typename FEGenericBase<OutputShape>::OutputTensor>> & d2phi,
                                                std::vector<std::vector<OutputShape>> & d2phidx2,
                                                std::vector<std::vector<OutputShape>> & d2phidxdy,
                                                std::vector<std::vector<OutputShape>> & d2phidxdz,
                                                std::vector<std::vector<OutputShape>> & d2phidy2,
                                                std::vector<std::vector<OutputShape>> & d2phidydz,
                                                std::vector<std::vector<OutputShape>> & d2phidz2) const
{
  switch (dim)
    {
    case 0: // No derivatives in 0D
      {
        for (auto i : index_range(d2phi))
          for (auto p : index_range(d2phi[i]))
            d2phi[i][p] = 0.;

        break;
      }

    case 1:
    case 2:
    case 3:
      {
        // Shape function derivatives in reference space
        const std::vector<std::vector<OutputShape>> * dphiref[] =
          {&fe.get_dphidxi(), &fe.get_dphideta(), &fe.get_dphidzeta()};
        const std::vector<std::vector<OutputShape>> * d2phiref[] =
          {&fe.get_d2phidxi2(), &fe.get_d2phidxideta(), &fe.get_d2phideta2(),
           &fe.get_d2phidxidzeta(), &fe.get_d2phidetadzeta(), &fe.get_d2phidzeta2()};

        for (auto p : make_range(d2phi.empty() ? std::size_t(0) : d2phi[0].size()))
          {
            const FEMapPoint<> m = map_point(fe.get_fe_map(), dim, p, true);

            for (auto i : index_range(d2phi))
              {
                OutputShape dphi[3], d2phiref_ip[6], d2[6] = {};
                for (unsigned int j=0; j != dim; ++j)
                  dphi[j] = (*dphiref[j])[i][p];
                for (unsigned int j=0; j != dim*(dim+1)/2; ++j)
                  d2phiref_ip[j] = (*d2phiref[j])[i][p];

                if (dim == 1)
                  fe_physical_second_derivs_1D(dphi, d2phiref_ip, m, d2);
                else if (dim == 2)
                  fe_physical_second_derivs_2D(dphi, d2phiref_ip, m, d2);
                else
                  fe_physical_second_derivs_3D(dphi, d2phiref_ip, m, d2);

                d2phi[i][p].slice(0).slice(0) = d2phidx2[i][p] = d2[0];
#if LIBMESH_DIM>1
                d2phi[i][p].slice(0).slice(1) = d2phi[i][p].slice(1).slice(0) = d2phidxdy[i][p] = d2[1];
                d2phi[i][p].slice(1).slice(1) = d2phidy2[i][p] = d2[3];
#endif
#if LIBMESH_DIM>2
                d2phi[i][p].slice(0).slice(2) = d2phi[i][p].slice(2).slice(0) = d2phidxdz[i][p] = d2[2];
                d2phi[i][p].slice(1).slice(2) = d2phi[i][p].slice(2).slice(1) = d2phidydz[i][p] = d2[4];
                d2phi[i][p].slice(2).slice(2) = d2phidz2[i][p] = d2[5];
#endif
              }
          }
        break;
      }

    default:
      libmesh_error_msg("Invalid dim = " << dim);
    } // switch(dim)
}
#endif //LIBMESH_ENABLE_SECOND_DERIVATIVES

template<>
void H1FETransformation<Real>::map_curl(const unsigned int,
                                        const Elem * const,
                                        const std::vector<Point> &,
                                        const FEGenericBase<Real> &,
                                        std::vector<std::vector<Real>> &) const
{
  libmesh_error_msg("Computing the curl of a shape function only \nmakes sense for vector-valued elements.");
}

template<>
void H1FETransformation<RealGradient>::map_curl(const unsigned int dim,
                                                const Elem * const,
                                                const std::vector<Point> &,
                                                const FEGenericBase<RealGradient> & fe,
                                                std::vector<std::vector<RealGradient>> & curl_phi ) const
{
  switch (dim)
    {
    case 0:
    case 1:
      libmesh_error_msg("The curl only make sense in 2D and 3D");

    case 2:
      {
        const std::vector<std::vector<RealGradient>> & dphidxi = fe.get_dphidxi();
        const std::vector<std::vector<RealGradient>> & dphideta = fe.get_dphideta();

        const std::vector<Real> & dxidx_map = fe.get_fe_map().get_dxidx();
        const std::vector<Real> & dxidy_map = fe.get_fe_map().get_dxidy();
#if LIBMESH_DIM > 2
        const std::vector<Real> & dxidz_map = fe.get_fe_map().get_dxidz();
#endif

        const std::vector<Real> & detadx_map = fe.get_fe_map().get_detadx();
        const std::vector<Real> & detady_map = fe.get_fe_map().get_detady();
#if LIBMESH_DIM > 2
        const std::vector<Real> & detadz_map = fe.get_fe_map().get_detadz();
#endif

        // For 2D elements in 3D space:
        // curl = ( -dphi_y/dz, dphi_x/dz, dphi_y/dx - dphi_x/dy )
        for (auto i : index_range(curl_phi))
          for (auto p : index_range(curl_phi[i]))
            {

              Real dphiy_dx = (dphidxi[i][p].slice(1))*dxidx_map[p]
                + (dphideta[i][p].slice(1))*detadx_map[p];

              Real dphix_dy = (dphidxi[i][p].slice(0))*dxidy_map[p]
                + (dphideta[i][p].slice(0))*detady_map[p];

              curl_phi[i][p].slice(2) = dphiy_dx - dphix_dy;

#if LIBMESH_DIM > 2
              Real dphiy_dz = (dphidxi[i][p].slice(1))*dxidz_map[p]
                + (dphideta[i][p].slice(1))*detadz_map[p];

              Real dphix_dz = (dphidxi[i][p].slice(0))*dxidz_map[p]
                + (dphideta[i][p].slice(0))*detadz_map[p];

              curl_phi[i][p].slice(0) = -dphiy_dz;
              curl_phi[i][p].slice(1) = dphix_dz;
#endif
            }

        break;
      }
    case 3:
      {
        const std::vector<std::vector<RealGradient>> & dphidxi = fe.get_dphidxi();
        const std::vector<std::vector<RealGradient>> & dphideta = fe.get_dphideta();
        const std::vector<std::vector<RealGradient>> & dphidzeta = fe.get_dphidzeta();

        const std::vector<Real> & dxidx_map = fe.get_fe_map().get_dxidx();
        const std::vector<Real> & dxidy_map = fe.get_fe_map().get_dxidy();
        const std::vector<Real> & dxidz_map = fe.get_fe_map().get_dxidz();

        const std::vector<Real> & detadx_map = fe.get_fe_map().get_detadx();
        const std::vector<Real> & detady_map = fe.get_fe_map().get_detady();
        const std::vector<Real> & detadz_map = fe.get_fe_map().get_detadz();

        const std::vector<Real> & dzetadx_map = fe.get_fe_map().get_dzetadx();
        const std::vector<Real> & dzetady_map = fe.get_fe_map().get_dzetady();
        const std::vector<Real> & dzetadz_map = fe.get_fe_map().get_dzetadz();

        // In 3D: curl = ( dphi_z/dy - dphi_y/dz, dphi_x/dz - dphi_z/dx, dphi_y/dx - dphi_x/dy )
        for (auto i : index_range(curl_phi))
          for (auto p : index_range(curl_phi[i]))
            {
              Real dphiz_dy = (dphidxi[i][p].slice(2))*dxidy_map[p]
                + (dphideta[i][p].slice(2))*detady_map[p]
                + (dphidzeta[i][p].slice(2))*dzetady_map[p];

              Real dphiy_dz = (dphidxi[i][p].slice(1))*dxidz_map[p]
                + (dphideta[i][p].slice(1))*detadz_map[p]
                + (dphidzeta[i][p].slice(1))*dzetadz_map[p];

              Real dphix_dz = (dphidxi[i][p].slice(0))*dxidz_map[p]
                + (dphideta[i][p].slice(0))*detadz_map[p]
                + (dphidzeta[i][p].slice(0))*dzetadz_map[p];

              Real dphiz_dx = (dphidxi[i][p].slice(2))*dxidx_map[p]
                + (dphideta[i][p].slice(2))*detadx_map[p]
                + (dphidzeta[i][p].slice(2))*dzetadx_map[p];

              Real dphiy_dx = (dphidxi[i][p].slice(1))*dxidx_map[p]
                + (dphideta[i][p].slice(1))*detadx_map[p]
                + (dphidzeta[i][p].slice(1))*dzetadx_map[p];

              Real dphix_dy = (dphidxi[i][p].slice(0))*dxidy_map[p]
                + (dphideta[i][p].slice(0))*detady_map[p]
                + (dphidzeta[i][p].slice(0))*dzetady_map[p];

              curl_phi[i][p].slice(0) = dphiz_dy - dphiy_dz;

              curl_phi[i][p].slice(1) = dphix_dz - dphiz_dx;

              curl_phi[i][p].slice(2) = dphiy_dx - dphix_dy;
            }

        break;
      }
    default:
      libmesh_error_msg("Invalid dim = " << dim);
    }
}


template<>
void H1FETransformation<Real>::map_div(const unsigned int,
                                       const Elem * const,
                                       const std::vector<Point> &,
                                       const FEGenericBase<Real> &,
                                       std::vector<std::vector<FEGenericBase<Real>::OutputDivergence>> & ) const
{
  libmesh_error_msg("Computing the divergence of a shape function only \nmakes sense for vector-valued elements.");
}


template<>
void H1FETransformation<RealGradient>::map_div(const unsigned int dim,
                                               const Elem * const,
                                               const std::vector<Point> &,
                                               const FEGenericBase<RealGradient> & fe,
                                               std::vector<std::vector<FEGenericBase<RealGradient>::OutputDivergence>> & div_phi) const
{
  switch (dim)
    {
    case 0:
    case 1:
      libmesh_error_msg("The divergence only make sense in 2D and 3D");

    case 2:
      {
        const std::vector<std::vector<RealGradient>> & dphidxi = fe.get_dphidxi();
        const std::vector<std::vector<RealGradient>> & dphideta = fe.get_dphideta();

        const std::vector<Real> & dxidx_map = fe.get_fe_map().get_dxidx();
        const std::vector<Real> & dxidy_map = fe.get_fe_map().get_dxidy();

        const std::vector<Real> & detadx_map = fe.get_fe_map().get_detadx();
        const std::vector<Real> & detady_map = fe.get_fe_map().get_detady();

        // In 2D: div = dphi_x/dx + dphi_y/dy
        for (auto i : index_range(div_phi))
          for (auto p : index_range(div_phi[i]))
            {
              Real dphix_dx = (dphidxi[i][p].slice(0))*dxidx_map[p]
                + (dphideta[i][p].slice(0))*detadx_map[p];

              Real dphiy_dy = (dphidxi[i][p].slice(1))*dxidy_map[p]
                + (dphideta[i][p].slice(1))*detady_map[p];

              div_phi[i][p] = dphix_dx + dphiy_dy;
            }
        break;
      }
    case 3:
      {
        const std::vector<std::vector<RealGradient>> & dphidxi = fe.get_dphidxi();
        const std::vector<std::vector<RealGradient>> & dphideta = fe.get_dphideta();
        const std::vector<std::vector<RealGradient>> & dphidzeta = fe.get_dphidzeta();

        const std::vector<Real> & dxidx_map = fe.get_fe_map().get_dxidx();
        const std::vector<Real> & dxidy_map = fe.get_fe_map().get_dxidy();
        const std::vector<Real> & dxidz_map = fe.get_fe_map().get_dxidz();

        const std::vector<Real> & detadx_map = fe.get_fe_map().get_detadx();
        const std::vector<Real> & detady_map = fe.get_fe_map().get_detady();
        const std::vector<Real> & detadz_map = fe.get_fe_map().get_detadz();

        const std::vector<Real> & dzetadx_map = fe.get_fe_map().get_dzetadx();
        const std::vector<Real> & dzetady_map = fe.get_fe_map().get_dzetady();
        const std::vector<Real> & dzetadz_map = fe.get_fe_map().get_dzetadz();

        // In 3D: div = dphi_x/dx + dphi_y/dy + dphi_z/dz
        for (auto i : index_range(div_phi))
          for (auto p : index_range(div_phi[i]))
            {
              Real dphix_dx = (dphidxi[i][p].slice(0))*dxidx_map[p]
                + (dphideta[i][p].slice(0))*detadx_map[p]
                + (dphidzeta[i][p].slice(0))*dzetadx_map[p];

              Real dphiy_dy = (dphidxi[i][p].slice(1))*dxidy_map[p]
                + (dphideta[i][p].slice(1))*detady_map[p]
                + (dphidzeta[i][p].slice(1))*dzetady_map[p];

              Real dphiz_dz = (dphidxi[i][p].slice(2))*dxidz_map[p]
                + (dphideta[i][p].slice(2))*detadz_map[p]
                + (dphidzeta[i][p].slice(2))*dzetadz_map[p];

              div_phi[i][p] = dphix_dx + dphiy_dy + dphiz_dz;
            }

        break;
      }

    default:
      libmesh_error_msg("Invalid dim = " << dim);
    } // switch(dim)

  return;
}

// Explicit Instantiations
template class LIBMESH_EXPORT H1FETransformation<Real>;
template class LIBMESH_EXPORT H1FETransformation<RealGradient>;

} //namespace libMesh
