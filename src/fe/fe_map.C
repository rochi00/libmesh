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
#include "libmesh/fe.h"
#include "libmesh/elem.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_macro.h"
#include "libmesh/fe_map.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/fe_xyz_map.h"
#include "libmesh/inf_fe_map.h"
#include "libmesh/mesh_subdivision_support.h"
#include "libmesh/tensor_value.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/int_range.h"

// C++ includes
#include <cstdlib> // *must* precede <cmath> for proper std:abs() on PGI, Sun Studio CC
#include <algorithm> // std::copy
#include <cmath> // for std::sqrt, std::abs
#include <memory>

namespace libMesh
{
Threads::spin_mutex FEMap::_point_inv_err_mutex;

FEFamily
FEMap::map_fe_type(const Elem & elem)
{
  switch (elem.mapping_type())
  {
  case RATIONAL_BERNSTEIN_MAP:
    return RATIONAL_BERNSTEIN;
  case LAGRANGE_MAP:
    return LAGRANGE;
  default:
    libmesh_error_msg("Unknown mapping type " << elem.mapping_type());
  }
  return LAGRANGE;
}



// Constructor
FEMap::FEMap(Real jtol) :
  calculations_started(false),
  calculate_xyz(false),
  calculate_dxyz(false),
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  calculate_d2xyz(false),
#endif
  jacobian_tolerance(jtol)
{}



std::unique_ptr<FEMap> FEMap::build( FEType fe_type )
{
  switch( fe_type.family )
    {
    case XYZ:
      return std::make_unique<FEXYZMap>();

    default:
      return std::make_unique<FEMap>();
    }
}



void FEMap::add_calculations()
{
  this->calculations_started = false;
  this->phi_map.clear();
  this->dphidxi_map.clear();
  this->dphideta_map.clear();
  this->dphidzeta_map.clear();
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  this->d2phidxi2_map.clear();
  this->d2phidxideta_map.clear();
  this->d2phideta2_map.clear();
  this->d2phidxidzeta_map.clear();
  this->d2phidetadzeta_map.clear();
  this->d2phidzeta2_map.clear();
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
}



template<unsigned int Dim>
void FEMap::init_reference_to_physical_map(const std::vector<Point> & qp,
                                           const Elem * elem)
{
  // Start logging the reference->physical map initialization
  LOG_SCOPE("init_reference_to_physical_map()", "FEMap");

  // We're calculating now!
  this->determine_calculations();

  // The number of quadrature points.
  const std::size_t n_qp = qp.size();

  // The element type and order to use in
  // the map
  const FEFamily mapping_family = FEMap::map_fe_type(*elem);
  const FEType map_fe_type(elem->default_order(), mapping_family);

  // Number of shape functions used to construct the map
  // (Lagrange shape functions are used for mapping)
  // Do not consider the Elem::p_level(), if any, when computing the
  // number of shape functions.
  const unsigned int n_mapping_shape_functions =
    FEInterface::n_shape_functions (map_fe_type, /*extra_order=*/0, elem);

  // Maybe we already have correctly-sized data?  Check data sizes,
  // and get ready to break out of a "loop" if all these resize()
  // calls are redundant.
  unsigned int old_n_qp = 0;

  do
    {
      if (calculate_xyz)
        {
          if (this->phi_map.size() == n_mapping_shape_functions)
            {
              old_n_qp = n_mapping_shape_functions ? this->phi_map[0].size() : 0;
              break;
            }
          this->phi_map.resize         (n_mapping_shape_functions);
        }
      if (Dim > 0)
        {
          if (calculate_dxyz)
            {
              if (this->dphidxi_map.size() == n_mapping_shape_functions)
                {
                  old_n_qp = n_mapping_shape_functions ? this->dphidxi_map[0].size() : 0;
                  break;
                }
              this->dphidxi_map.resize     (n_mapping_shape_functions);
            }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          // Don't waste time considering early break here; if we
          // asked for d2xyz then we surely asked for dxyz too and
          // considered early break above.
          if (calculate_d2xyz)
            this->d2phidxi2_map.resize   (n_mapping_shape_functions);
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        }

      if (Dim > 1)
        {
          if (calculate_dxyz)
            this->dphideta_map.resize  (n_mapping_shape_functions);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          if (calculate_d2xyz)
            {
              this->d2phidxideta_map.resize   (n_mapping_shape_functions);
              this->d2phideta2_map.resize     (n_mapping_shape_functions);
            }
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        }

      if (Dim > 2)
        {
          if (calculate_dxyz)
            this->dphidzeta_map.resize (n_mapping_shape_functions);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          if (calculate_d2xyz)
            {
              this->d2phidxidzeta_map.resize  (n_mapping_shape_functions);
              this->d2phidetadzeta_map.resize (n_mapping_shape_functions);
              this->d2phidzeta2_map.resize    (n_mapping_shape_functions);
            }
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        }
    }
  while (false);

  if (old_n_qp != n_qp)
    for (unsigned int i=0; i<n_mapping_shape_functions; i++)
      {
        if (calculate_xyz)
          this->phi_map[i].resize         (n_qp);
        if (Dim > 0)
          {
            if (calculate_dxyz)
              this->dphidxi_map[i].resize     (n_qp);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            if (calculate_d2xyz)
              {
                this->d2phidxi2_map[i].resize     (n_qp);
                if (Dim > 1)
                  {
                    this->d2phidxideta_map[i].resize (n_qp);
                    this->d2phideta2_map[i].resize (n_qp);
                  }
                if (Dim > 2)
                  {
                    this->d2phidxidzeta_map[i].resize  (n_qp);
                    this->d2phidetadzeta_map[i].resize (n_qp);
                    this->d2phidzeta2_map[i].resize    (n_qp);
                  }
              }
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

            if (Dim > 1 && calculate_dxyz)
              this->dphideta_map[i].resize  (n_qp);

            if (Dim > 2 && calculate_dxyz)
                this->dphidzeta_map[i].resize (n_qp);
          }
      }

  // Optimize for the *linear* geometric elements case:
  bool is_linear = elem->is_linear();

  FEInterface::shape_deriv_ptr shape_deriv_ptr =
    FEInterface::shape_deriv_function(map_fe_type, elem);

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  FEInterface::shape_second_deriv_ptr shape_second_deriv_ptr =
    FEInterface::shape_second_deriv_function(map_fe_type, elem);
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

  if (calculate_xyz)
    FEInterface::all_shapes(Dim, map_fe_type, elem, qp, this->phi_map, false);

  switch (Dim)
    {
      //------------------------------------------------------------
      // 0D
    case 0:
      {
        // No mapping derivatives here
        break;
      }

      //------------------------------------------------------------
      // 1D
    case 1:
      {
        // Compute gradients of mapping shape functions i at quadrature points p

        // TODO: optimizations for the RATIONAL_BERNSTEIN+linear case
        if (is_linear)
          {
            for (unsigned int i=0; i<n_mapping_shape_functions; i++)
              {
                if (calculate_dxyz)
                  {
                    this->dphidxi_map[i][0] =
                      shape_deriv_ptr(map_fe_type, elem, i, 0, qp[0], false);
                    for (std::size_t p=1; p<n_qp; p++)
                      this->dphidxi_map[i][p]  = this->dphidxi_map[i][0];
                  }

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                if (calculate_d2xyz)
                  {
                    this->d2phidxi2_map[i][0] =
                      shape_second_deriv_ptr(map_fe_type, elem, i, 0, qp[0], false);
                    for (std::size_t p=1; p<n_qp; p++)
                      this->d2phidxi2_map[i][p] = this->d2phidxi2_map[i][0];
                  }
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
              }
          }
        else
          {
            if (calculate_dxyz)
              {
                std::vector<std::vector<Real>> * comps[3]
                  { &this->dphidxi_map, &this->dphideta_map, &this->dphidzeta_map };
                FEInterface::all_shape_derivs(Dim, map_fe_type, elem, qp, comps, false);
              }

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            if (calculate_d2xyz)
              for (unsigned int i=0; i<n_mapping_shape_functions; i++)
                for (std::size_t p=0; p<n_qp; p++)
                  this->d2phidxi2_map[i][p] = shape_second_deriv_ptr (map_fe_type, elem, i, 0, qp[p], false);
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          }

        break;
      }
      //------------------------------------------------------------
      // 2D
    case 2:
      {
        // Compute gradients of mapping shape functions i at quadrature points p

        // TODO: optimizations for the RATIONAL_BERNSTEIN+linear case
        if (is_linear)
          {
            for (unsigned int i=0; i<n_mapping_shape_functions; i++)
              {
                if (calculate_dxyz)
                  {
                    this->dphidxi_map[i][0]  = shape_deriv_ptr (map_fe_type, elem, i, 0, qp[0], false);
                    this->dphideta_map[i][0] = shape_deriv_ptr (map_fe_type, elem, i, 1, qp[0], false);
                    for (std::size_t p=1; p<n_qp; p++)
                      {
                        this->dphidxi_map[i][p]  = this->dphidxi_map[i][0];
                        this->dphideta_map[i][p] = this->dphideta_map[i][0];
                      }
                  }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                if (calculate_d2xyz)
                  {
                    this->d2phidxi2_map[i][0]    = shape_second_deriv_ptr (map_fe_type, elem, i, 0, qp[0], false);
                    this->d2phidxideta_map[i][0] = shape_second_deriv_ptr (map_fe_type, elem, i, 1, qp[0], false);
                    this->d2phideta2_map[i][0]   = shape_second_deriv_ptr (map_fe_type, elem, i, 2, qp[0], false);
                    for (std::size_t p=1; p<n_qp; p++)
                      {
                        this->d2phidxi2_map[i][p] = this->d2phidxi2_map[i][0];
                        this->d2phidxideta_map[i][p] = this->d2phidxideta_map[i][0];
                        this->d2phideta2_map[i][p] = this->d2phideta2_map[i][0];
                      }
                  }
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
              }
          }
        else
          {
            if (calculate_dxyz)
              {
                std::vector<std::vector<Real>> * comps[3]
                  { &this->dphidxi_map, &this->dphideta_map, &this->dphidzeta_map };
                FEInterface::all_shape_derivs(Dim, map_fe_type, elem, qp, comps, false);
              }

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            if (calculate_d2xyz)
              for (unsigned int i=0; i<n_mapping_shape_functions; i++)
                for (std::size_t p=0; p<n_qp; p++)
                  {
                    this->d2phidxi2_map[i][p] = shape_second_deriv_ptr (map_fe_type, elem, i, 0, qp[p], false);
                    this->d2phidxideta_map[i][p] = shape_second_deriv_ptr (map_fe_type, elem, i, 1, qp[p], false);
                    this->d2phideta2_map[i][p] = shape_second_deriv_ptr (map_fe_type, elem, i, 2, qp[p], false);
                  }
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          }

        break;
      }

      //------------------------------------------------------------
      // 3D
    case 3:
      {
        // Compute gradients of mapping shape functions i at quadrature points p

        // TODO: optimizations for the RATIONAL_BERNSTEIN+linear case
        if (is_linear)
          {
            for (unsigned int i=0; i<n_mapping_shape_functions; i++)
              {
                if (calculate_dxyz)
                  {
                    this->dphidxi_map[i][0]  = shape_deriv_ptr (map_fe_type, elem, i, 0, qp[0], false);
                    this->dphideta_map[i][0] = shape_deriv_ptr (map_fe_type, elem, i, 1, qp[0], false);
                    this->dphidzeta_map[i][0] = shape_deriv_ptr (map_fe_type, elem, i, 2, qp[0], false);

                    for (std::size_t p=1; p<n_qp; p++)
                      {
                        this->dphidxi_map[i][p]  = this->dphidxi_map[i][0];
                        this->dphideta_map[i][p] = this->dphideta_map[i][0];
                        this->dphidzeta_map[i][p] = this->dphidzeta_map[i][0];
                      }
                  }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                if (calculate_d2xyz)
                  {
                    this->d2phidxi2_map[i][0]      = shape_second_deriv_ptr (map_fe_type, elem, i, 0, qp[0], false);
                    this->d2phidxideta_map[i][0]   = shape_second_deriv_ptr (map_fe_type, elem, i, 1, qp[0], false);
                    this->d2phideta2_map[i][0]     = shape_second_deriv_ptr (map_fe_type, elem, i, 2, qp[0], false);
                    this->d2phidxidzeta_map[i][0]  = shape_second_deriv_ptr (map_fe_type, elem, i, 3, qp[0], false);
                    this->d2phidetadzeta_map[i][0] = shape_second_deriv_ptr (map_fe_type, elem, i, 4, qp[0], false);
                    this->d2phidzeta2_map[i][0]    = shape_second_deriv_ptr (map_fe_type, elem, i, 5, qp[0], false);

                    for (std::size_t p=1; p<n_qp; p++)
                      {
                        this->d2phidxi2_map[i][p] = this->d2phidxi2_map[i][0];
                        this->d2phidxideta_map[i][p] = this->d2phidxideta_map[i][0];
                        this->d2phideta2_map[i][p] = this->d2phideta2_map[i][0];
                        this->d2phidxidzeta_map[i][p] = this->d2phidxidzeta_map[i][0];
                        this->d2phidetadzeta_map[i][p] = this->d2phidetadzeta_map[i][0];
                        this->d2phidzeta2_map[i][p] = this->d2phidzeta2_map[i][0];
                      }
                  }
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
              }
          }
        else
          {
            if (calculate_dxyz)
              {
                std::vector<std::vector<Real>> * comps[3]
                  { &this->dphidxi_map, &this->dphideta_map, &this->dphidzeta_map };
                FEInterface::all_shape_derivs(Dim, map_fe_type, elem, qp, comps, false);
              }

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            if (calculate_d2xyz)
              for (unsigned int i=0; i<n_mapping_shape_functions; i++)
                for (std::size_t p=0; p<n_qp; p++)
                  {
                    this->d2phidxi2_map[i][p]      = shape_second_deriv_ptr (map_fe_type, elem, i, 0, qp[p], false);
                    this->d2phidxideta_map[i][p]   = shape_second_deriv_ptr (map_fe_type, elem, i, 1, qp[p], false);
                    this->d2phideta2_map[i][p]     = shape_second_deriv_ptr (map_fe_type, elem, i, 2, qp[p], false);
                    this->d2phidxidzeta_map[i][p]  = shape_second_deriv_ptr (map_fe_type, elem, i, 3, qp[p], false);
                    this->d2phidetadzeta_map[i][p] = shape_second_deriv_ptr (map_fe_type, elem, i, 4, qp[p], false);
                    this->d2phidzeta2_map[i][p]    = shape_second_deriv_ptr (map_fe_type, elem, i, 5, qp[p], false);
                  }
#endif // ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          }

        break;
      }

    default:
      libmesh_error_msg("Invalid Dim = " << Dim);
    }
}



void FEMap::compute_single_point_map(const unsigned int dim,
                                     const std::vector<Real> & qw,
                                     const Elem * elem,
                                     unsigned int p,
                                     const std::vector<const Node *> & elem_nodes,
                                     bool compute_second_derivatives)
{
  libmesh_assert(elem);
  libmesh_assert(calculations_started);
#ifndef LIBMESH_ENABLE_SECOND_DERIVATIVES
  libmesh_assert(!compute_second_derivatives);
#endif

  // this parameter is only accounted for if LIBMESH_DIM==2.
  libmesh_ignore(compute_second_derivatives);

  if (calculate_xyz)
    libmesh_assert_equal_to(phi_map.size(), elem_nodes.size());

  auto check_for_degenerate_map =
    [this, elem, p]
    (Real det_J) {
    if (det_J <= jacobian_tolerance)
      {
        // Don't call get_info() recursively if we're already
        // failing.  get_info() calls Elem::volume() which may
        // call FE::reinit() and trigger the same failure again.
        static bool failing = false;
        if (!failing)
          {
            failing = true;
            std::string elem_info = elem->get_info();
            failing = false;
            if (calculate_xyz)
              {
                libmesh_degenerate_mapping_msg
                  ("Jacobian " << det_J << " at or under tolerance " <<
                   jacobian_tolerance << " at point " << xyz[p] <<
                   " in element:\n" << elem_info);
              }
            else
              {
                // In this case xyz[p] is not defined, so don't
                // try to print it out.
                libmesh_degenerate_mapping_msg
                  ("Jacobian " << det_J << " at or under tolerance " <<
                   jacobian_tolerance << " at point index " << p <<
                   " in element:\n" << elem_info);
              }
          }
        else
          {
            // We were already failing when we called this, so just
            // stop the current computation and return with
            // incomplete results.
            return;
          }
      }
  };

  switch (dim)
    {
      //--------------------------------------------------------------------
      // 0D
    case 0:
      {
        libmesh_assert(elem_nodes[0]);
        if (calculate_xyz)
          xyz[p] = *elem_nodes[0];
        if (calculate_dxyz)
          {
            jac[p] = 1.0;
            JxW[p] = qw[p];
          }
        break;
      }

      //--------------------------------------------------------------------
      // 1D
    case 1:
      {
        // Clear the entities that will be summed
        if (calculate_xyz)
          xyz[p].zero();
        if (calculate_dxyz)
          dxyzdxi_map[p].zero();
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        if (calculate_d2xyz)
          {
            d2xyzdxi2_map[p].zero();
            // Inverse map second derivatives
            d2xidxyz2_map[p].assign(6, 0.);
          }
#endif

        // compute x, dx, d2x at the quadrature point
        for (auto i : index_range(elem_nodes)) // sum over the nodes
          {
            // Reference to the point, helps eliminate
            // excessive temporaries in the inner loop
            libmesh_assert(elem_nodes[i]);
            const Point & elem_point = *elem_nodes[i];

            if (calculate_xyz)
              xyz[p].add_scaled          (elem_point, phi_map[i][p]    );
            if (calculate_dxyz)
              dxyzdxi_map[p].add_scaled  (elem_point, dphidxi_map[i][p]);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            if (calculate_d2xyz)
              d2xyzdxi2_map[p].add_scaled(elem_point, d2phidxi2_map[i][p]);
#endif
          }

        // Compute the jacobian, and from it the inverse map
        FEMapPoint<> m;
        if (calculate_dxyz)
          {
            fe_map_1D(dxyzdxi_map[p], m, check_for_degenerate_map);
            jac[p] = m.jac;

            dxidx_map[p] = m.dxi(0);
#if LIBMESH_DIM > 1
            dxidy_map[p] = m.dxi(1);
#endif
#if LIBMESH_DIM > 2
            dxidz_map[p] = m.dxi(2);
#endif

            JxW[p] = jac[p]*qw[p];
          }

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

        if (calculate_d2xyz)
          {
#if LIBMESH_DIM > 1
            // (x_xi)^2 + (y_xi)^2 + (z_xi)^2 must be > 0.0
            if (dxyzdxi_map[p].norm_sq() <= 0.0)
              {
                // Don't call print_info() recursively if we're already
                // failing.  print_info() calls Elem::volume() which may
                // call FE::reinit() and trigger the same failure again.
                static bool failing = false;
                if (!failing)
                  {
                    failing = true;
                    elem->print_info(libMesh::err);
                    failing = false;
                    libmesh_error_msg("Encountered invalid 1D element!");
                  }
                else
                  {
                    // We were already failing when we called this, so just
                    // stop the current computation and return with
                    // incomplete results.
                    return;
                  }
              }
#endif

            const RealVectorValue d2xyz[] = {d2xyzdxi2_map[p]};
            fe_map_second_derivs_1D(dxyzdxi_map[p], d2xyz, m);
            std::copy(m.d2xi, m.d2xi + 6, d2xidxyz2_map[p].begin());
          } // calculate_d2xyz

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES

        // done computing the map
        break;
      }


      //--------------------------------------------------------------------
      // 2D
    case 2:
      {
        //------------------------------------------------------------------
        // Compute the (x,y) values at the quadrature points,
        // the Jacobian at the quadrature points

        if (calculate_xyz)
          xyz[p].zero();

        if (calculate_dxyz)
          {
            dxyzdxi_map[p].zero();
            dxyzdeta_map[p].zero();
          }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        if (calculate_d2xyz)
          {
            d2xyzdxi2_map[p].zero();
            d2xyzdxideta_map[p].zero();
            d2xyzdeta2_map[p].zero();
            // Inverse map second derivatives
            d2xidxyz2_map[p].assign(6, 0.);
            d2etadxyz2_map[p].assign(6, 0.);
          }
#endif


        // compute (x,y) at the quadrature points, derivatives once
        for (auto i : index_range(elem_nodes)) // sum over the nodes
          {
            // Reference to the point, helps eliminate
            // excessive temporaries in the inner loop
            libmesh_assert(elem_nodes[i]);
            const Point & elem_point = *elem_nodes[i];

            if (calculate_xyz)
              xyz[p].add_scaled          (elem_point, phi_map[i][p]     );

            if (calculate_dxyz)
              {
                dxyzdxi_map[p].add_scaled      (elem_point, dphidxi_map[i][p] );
                dxyzdeta_map[p].add_scaled     (elem_point, dphideta_map[i][p]);
              }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            if (calculate_d2xyz)
              {
                d2xyzdxi2_map[p].add_scaled    (elem_point, d2phidxi2_map[i][p]);
                d2xyzdxideta_map[p].add_scaled (elem_point, d2phidxideta_map[i][p]);
                d2xyzdeta2_map[p].add_scaled   (elem_point, d2phideta2_map[i][p]);
              }
#endif
          }

        if (calculate_dxyz)
          {
            // compute the jacobian once
            FEMapPoint<> m;
            fe_map_2D(dxyzdxi_map[p], dxyzdeta_map[p], m, check_for_degenerate_map);
            jac[p] = m.jac;

            JxW[p] = jac[p]*qw[p];

#if LIBMESH_DIM == 2
            dxidx_map[p]  = m.dxi(0);
            dxidy_map[p]  = m.dxi(1);
            detadx_map[p] = m.deta(0);
            detady_map[p] = m.deta(1);

            dxidz_map[p] = detadz_map[p] = 0.;
#else // LIBMESH_DIM == 3
            dxidx_map[p]  = m.dxi(0);
            dxidy_map[p]  = m.dxi(1);
            dxidz_map[p]  = m.dxi(2);

            detadx_map[p] = m.deta(0);
            detady_map[p] = m.deta(1);
            detadz_map[p] = m.deta(2);
#endif // LIBMESH_DIM == 3

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            // 2D space has always asked compute_second_derivatives
            // here, and 3D space calculate_d2xyz
            if (LIBMESH_DIM == 2 ? compute_second_derivatives : calculate_d2xyz)
              {
                const RealVectorValue d2xyz[] =
                  {d2xyzdxi2_map[p], d2xyzdxideta_map[p], d2xyzdeta2_map[p]};
                fe_map_second_derivs_2D(dxyzdxi_map[p], dxyzdeta_map[p], d2xyz, m);
                std::copy(m.d2xi, m.d2xi + 6, d2xidxyz2_map[p].begin());
                std::copy(m.d2eta, m.d2eta + 6, d2etadxyz2_map[p].begin());
              }
#endif
          }
        // done computing the map
        break;
      }



      //--------------------------------------------------------------------
      // 3D
    case 3:
      {
        //------------------------------------------------------------------
        // Compute the (x,y,z) values at the quadrature points,
        // the Jacobian at the quadrature point

        // Clear the entities that will be summed
        if (calculate_xyz)
          xyz[p].zero           ();
        if (calculate_dxyz)
          {
            dxyzdxi_map[p].zero   ();
            dxyzdeta_map[p].zero  ();
            dxyzdzeta_map[p].zero ();
          }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        if (calculate_d2xyz)
          {
            d2xyzdxi2_map[p].zero();
            d2xyzdxideta_map[p].zero();
            d2xyzdxidzeta_map[p].zero();
            d2xyzdeta2_map[p].zero();
            d2xyzdetadzeta_map[p].zero();
            d2xyzdzeta2_map[p].zero();
            // Inverse map second derivatives
            d2xidxyz2_map[p].assign(6, 0.);
            d2etadxyz2_map[p].assign(6, 0.);
            d2zetadxyz2_map[p].assign(6, 0.);
          }
#endif


        // compute (x,y,z) at the quadrature points,
        // dxdxi,   dydxi,   dzdxi,
        // dxdeta,  dydeta,  dzdeta,
        // dxdzeta, dydzeta, dzdzeta  all once
        for (auto i : index_range(elem_nodes)) // sum over the nodes
          {
            // Reference to the point, helps eliminate
            // excessive temporaries in the inner loop
            libmesh_assert(elem_nodes[i]);
            const Point & elem_point = *elem_nodes[i];

            if (calculate_xyz)
              xyz[p].add_scaled           (elem_point, phi_map[i][p]      );
            if (calculate_dxyz)
              {
                dxyzdxi_map[p].add_scaled   (elem_point, dphidxi_map[i][p]  );
                dxyzdeta_map[p].add_scaled  (elem_point, dphideta_map[i][p] );
                dxyzdzeta_map[p].add_scaled (elem_point, dphidzeta_map[i][p]);
              }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            if (calculate_d2xyz)
              {
                d2xyzdxi2_map[p].add_scaled      (elem_point,
                                                  d2phidxi2_map[i][p]);
                d2xyzdxideta_map[p].add_scaled   (elem_point,
                                                  d2phidxideta_map[i][p]);
                d2xyzdxidzeta_map[p].add_scaled  (elem_point,
                                                  d2phidxidzeta_map[i][p]);
                d2xyzdeta2_map[p].add_scaled     (elem_point,
                                                  d2phideta2_map[i][p]);
                d2xyzdetadzeta_map[p].add_scaled (elem_point,
                                                  d2phidetadzeta_map[i][p]);
                d2xyzdzeta2_map[p].add_scaled    (elem_point,
                                                  d2phidzeta2_map[i][p]);
              }
#endif
          }

        FEMapPoint<> m;
        if (calculate_dxyz)
          {
            // compute the jacobian
            fe_map_3D(dxyzdxi_map[p], dxyzdeta_map[p], dxyzdzeta_map[p], m,
                      check_for_degenerate_map);
            jac[p] = m.jac;

            JxW[p] = jac[p]*qw[p];

            dxidx_map[p]   = m.dxi(0);
            dxidy_map[p]   = m.dxi(1);
            dxidz_map[p]   = m.dxi(2);

            detadx_map[p]  = m.deta(0);
            detady_map[p]  = m.deta(1);
            detadz_map[p]  = m.deta(2);

            dzetadx_map[p] = m.dzeta(0);
            dzetady_map[p] = m.dzeta(1);
            dzetadz_map[p] = m.dzeta(2);
          }

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        if (compute_second_derivatives)
          {
            const RealVectorValue d2xyz[] =
              {d2xyzdxi2_map[p], d2xyzdxideta_map[p], d2xyzdeta2_map[p],
               d2xyzdxidzeta_map[p], d2xyzdetadzeta_map[p], d2xyzdzeta2_map[p]};
            fe_map_second_derivs_square(d2xyz, m);
            std::copy(m.d2xi, m.d2xi + 6, d2xidxyz2_map[p].begin());
            std::copy(m.d2eta, m.d2eta + 6, d2etadxyz2_map[p].begin());
            std::copy(m.d2zeta, m.d2zeta + 6, d2zetadxyz2_map[p].begin());
          }
#endif
        // done computing the map
        break;
      }

    default:
      libmesh_error_msg("Invalid dim = " << dim);
    }
}



void FEMap::resize_quadrature_map_vectors(const unsigned int dim, unsigned int n_qp)
{
  // We're calculating now!
  this->determine_calculations();

  // Resize the vectors to hold data at the quadrature points
  if (calculate_xyz)
    xyz.resize(n_qp);
  if (calculate_dxyz)
    {
      dxyzdxi_map.resize(n_qp);
      dxidx_map.resize(n_qp);
      dxidy_map.resize(n_qp); // 1D element may live in 2D ...
      dxidz_map.resize(n_qp); // ... or 3D
    }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  if (calculate_d2xyz)
    {
      d2xyzdxi2_map.resize(n_qp);

      // Inverse map second derivatives
      d2xidxyz2_map.resize(n_qp);
      for (auto i : index_range(d2xidxyz2_map))
        d2xidxyz2_map[i].assign(6, 0.);
    }
#endif
  if (dim > 1)
    {
      if (calculate_dxyz)
        {
          dxyzdeta_map.resize(n_qp);
          detadx_map.resize(n_qp);
          detady_map.resize(n_qp);
          detadz_map.resize(n_qp);
        }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
      if (calculate_d2xyz)
        {
          d2xyzdxideta_map.resize(n_qp);
          d2xyzdeta2_map.resize(n_qp);

          // Inverse map second derivatives
          d2etadxyz2_map.resize(n_qp);
          for (auto i : index_range(d2etadxyz2_map))
            d2etadxyz2_map[i].assign(6, 0.);
        }
#endif
      if (dim > 2)
        {
          if (calculate_dxyz)
            {
              dxyzdzeta_map.resize (n_qp);
              dzetadx_map.resize   (n_qp);
              dzetady_map.resize   (n_qp);
              dzetadz_map.resize   (n_qp);
            }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          if (calculate_d2xyz)
            {
              d2xyzdxidzeta_map.resize(n_qp);
              d2xyzdetadzeta_map.resize(n_qp);
              d2xyzdzeta2_map.resize(n_qp);

              // Inverse map second derivatives
              d2zetadxyz2_map.resize(n_qp);
              for (auto i : index_range(d2zetadxyz2_map))
                d2zetadxyz2_map[i].assign(6, 0.);
            }
#endif
        }
    }

  if (calculate_dxyz)
    {
      jac.resize(n_qp);
      JxW.resize(n_qp);
    }
}



void FEMap::compute_affine_map(const unsigned int dim,
                               const std::vector<Real> & qw,
                               const Elem * elem)
{
  // Start logging the map computation.
  LOG_SCOPE("compute_affine_map()", "FEMap");

  libmesh_assert(elem);

  const unsigned int n_qp = cast_int<unsigned int>(qw.size());

  // Resize the vectors to hold data at the quadrature points
  this->resize_quadrature_map_vectors(dim, n_qp);

  // Determine the nodes contributing to element elem
  unsigned int n_nodes = elem->n_nodes();
  _elem_nodes.resize(elem->n_nodes());
  for (unsigned int i=0; i<n_nodes; i++)
    _elem_nodes[i] = elem->node_ptr(i);

  // Compute map at quadrature point 0
  this->compute_single_point_map(dim, qw, elem, 0, _elem_nodes, /*compute_second_derivatives=*/false);

  // Compute xyz at all other quadrature points
  if (calculate_xyz)
    for (unsigned int p=1; p<n_qp; p++)
      {
        xyz[p].zero();
        for (auto i : index_range(phi_map)) // sum over the nodes
          xyz[p].add_scaled (*_elem_nodes[i], phi_map[i][p]);
      }

  // Copy other map data from quadrature point 0
  if (calculate_dxyz)
    for (unsigned int p=1; p<n_qp; p++) // for each extra quadrature point
      {
        dxyzdxi_map[p] = dxyzdxi_map[0];
        dxidx_map[p] = dxidx_map[0];
        dxidy_map[p] = dxidy_map[0];
        dxidz_map[p] = dxidz_map[0];
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        // The map should be affine, so second derivatives are zero
        if (calculate_d2xyz)
          d2xyzdxi2_map[p] = 0.;
#endif
        if (dim > 1)
          {
            dxyzdeta_map[p] = dxyzdeta_map[0];
            detadx_map[p] = detadx_map[0];
            detady_map[p] = detady_map[0];
            detadz_map[p] = detadz_map[0];
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            if (calculate_d2xyz)
              {
                d2xyzdxideta_map[p] = 0.;
                d2xyzdeta2_map[p] = 0.;
              }
#endif
            if (dim > 2)
              {
                dxyzdzeta_map[p] = dxyzdzeta_map[0];
                dzetadx_map[p] = dzetadx_map[0];
                dzetady_map[p] = dzetady_map[0];
                dzetadz_map[p] = dzetadz_map[0];
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                if (calculate_d2xyz)
                  {
                    d2xyzdxidzeta_map[p] = 0.;
                    d2xyzdetadzeta_map[p] = 0.;
                    d2xyzdzeta2_map[p] = 0.;
                  }
#endif
              }
          }
        jac[p] = jac[0];
        JxW[p] = JxW[0] / qw[0] * qw[p];
      }
}



void FEMap::compute_null_map(const unsigned int dim,
                             const std::vector<Real> & qw)
{
  // Start logging the map computation.
  LOG_SCOPE("compute_null_map()", "FEMap");

  const unsigned int n_qp = cast_int<unsigned int>(qw.size());

  // Resize the vectors to hold data at the quadrature points
  this->resize_quadrature_map_vectors(dim, n_qp);

  // Compute "fake" xyz
  for (unsigned int p=1; p<n_qp; p++)
    {
      if (calculate_xyz)
        xyz[p].zero();

      if (calculate_dxyz)
        {
          dxyzdxi_map[p] = 0;
          dxidx_map[p] = 0;
          dxidy_map[p] = 0;
          dxidz_map[p] = 0;
        }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
      if (calculate_d2xyz)
        {
          d2xyzdxi2_map[p] = 0;
        }
#endif
      if (dim > 1)
        {
          if (calculate_dxyz)
            {
              dxyzdeta_map[p] = 0;
              detadx_map[p] = 0;
              detady_map[p] = 0;
              detadz_map[p] = 0;
            }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          if (calculate_d2xyz)
            {
              d2xyzdxideta_map[p] = 0.;
              d2xyzdeta2_map[p] = 0.;
            }
#endif
          if (dim > 2)
            {
              if (calculate_dxyz)
                {
                  dxyzdzeta_map[p] = 0;
                  dzetadx_map[p] = 0;
                  dzetady_map[p] = 0;
                  dzetadz_map[p] = 0;
                }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
              if (calculate_d2xyz)
                {
                  d2xyzdxidzeta_map[p] = 0;
                  d2xyzdetadzeta_map[p] = 0;
                  d2xyzdzeta2_map[p] = 0;
                }
#endif
            }
        }
      if (calculate_dxyz)
        {
          jac[p] = 1;
          JxW[p] = qw[p];
        }
    }
}



void FEMap::compute_map(const unsigned int dim,
                        const std::vector<Real> & qw,
                        const Elem * elem,
                        bool calculate_d2phi)
{
  if (!elem)
    {
      compute_null_map(dim, qw);
      return;
    }

  if (elem->has_affine_map())
    {
      compute_affine_map(dim, qw, elem);
      return;
    }
#ifndef LIBMESH_ENABLE_SECOND_DERIVATIVES
    libmesh_assert(!calculate_d2phi);
#endif

  // Start logging the map computation.
  LOG_SCOPE("compute_map()", "FEMap");

  libmesh_assert(elem);

  const unsigned int n_qp = cast_int<unsigned int>(qw.size());

  // Resize the vectors to hold data at the quadrature points
  this->resize_quadrature_map_vectors(dim, n_qp);

  // Determine the nodes contributing to element elem
  if (elem->type() == TRI3SUBDIVISION)
    {
      // Subdivision surface FE require the 1-ring around elem
      libmesh_assert_equal_to (dim, 2);
      const Tri3Subdivision * sd_elem = static_cast<const Tri3Subdivision *>(elem);
      MeshTools::Subdivision::find_one_ring(sd_elem, _elem_nodes);
    }
  else
    {
      // All other FE use only the nodes of elem itself
      _elem_nodes.resize(elem->n_nodes(), nullptr);
      for (auto i : elem->node_index_range())
        _elem_nodes[i] = elem->node_ptr(i);
    }

  // Compute map at all quadrature points
  for (unsigned int p=0; p!=n_qp; p++)
    this->compute_single_point_map(dim, qw, elem, p, _elem_nodes, calculate_d2phi);
}



void FEMap::print_JxW(std::ostream & os) const
{
  for (auto i : index_range(JxW))
    os << " [" << i << "]: " << JxW[i] << std::endl;
}



void FEMap::print_xyz(std::ostream & os) const
{
  for (auto i : index_range(xyz))
    os << " [" << i << "]: " << xyz[i];
}



// The map FEMap::inverse_map() inverts, and what it does when Newton
// fails: throw or warn, as it always has, or return the singular node
// or a far away point.
namespace
{
using namespace libMesh;

struct HostMapEval
{
  const unsigned int dim;
  const Elem * elem;

  Point map(const Point & p) const
  { return FEMap::map(dim, elem, p); }

  Point map_deriv(const unsigned int j, const Point & p) const
  { return FEMap::map_deriv(dim, elem, j, p); }
};

struct HostInverseMapPolicy
{
  const unsigned int dim;
  const Elem * elem;
  const Point & physical_point;
  const Real tolerance;
  const bool secure;

  bool singular(const Point & p) const
  {
    libmesh_degenerate_mapping_msg
      ("inverse_map found a singular Jacobian " <<
       " at master point " << p << " in element " <<
       elem->id());
    return true;
  }

  bool solve(const RealTensorValue & J,
             const Point & delta,
             Point & dp,
             Point & p) const
  {
    libmesh_try
      {
        J.solve(delta, dp);
      }
    libmesh_catch (ConvergenceFailure &)
      {
        // If we encountered a singular Jacobian, but are at
        // a singular node, return said singular node.
        const unsigned int local_singular_node =
          elem->local_singular_node(physical_point, tolerance);
        if (local_singular_node != invalid_uint)
        {
          libmesh_assert_less(local_singular_node, elem->n_nodes());
          p = elem->master_point(local_singular_node);
          return false;
        }

        // We encountered a singular Jacobian.  The value of
        // dp is zero, since it was never changed during the
        // call to RealTensorValue::solve().  We don't want to
        // continue iterating until max_cnt since there is no
        // update to the Newton iterate, and we don't want to
        // print the inverse_map_error value since it will
        // confusingly be 0.  Therefore, in the secure case we
        // need to throw an error message while in the !secure
        // case we can just return a far away point.
        if (secure)
          {
            libmesh_degenerate_mapping_msg(
              "inverse_map found a singular Jacobian" <<
              " at master point " << p << " in element " <<
              elem->id());
          }
        else
          {
            fe_inverse_map_far_point(dim, p);
            return false;
          }
      }

    return true;
  }

  bool not_converged(const unsigned int cnt,
                     const unsigned int max_cnt,
                     const Point & physical_guess,
                     const Point & dp,
                     const Point & p,
                     const Real inverse_map_error) const
  {
    // Print every time in devel/dbg modes
#ifndef NDEBUG
    libmesh_here();
    libMesh::err << "WARNING: Newton scheme has not converged in "
                 << cnt << " iterations:" << std::endl
                 << "   physical_point="
                 << physical_point
                 << "   physical_guess="
                 << physical_guess
                 << "   dp="
                 << dp
                 << "   p="
                 << p
                 << "   error=" << inverse_map_error
                 << "   in element " << elem->id()
                 << std::endl;

    elem->print_info(libMesh::err);
#else
    // In optimized mode, just print once that an inverse_map() call
    // had trouble converging its Newton iteration.
    libmesh_do_once(libMesh::err << "WARNING: At least one element took more than "
                    << max_cnt
                    << " iterations to converge in inverse_map()...\n"
                    << "Rerun in devel/dbg mode for more details."
                    << std::endl;);

#endif // NDEBUG

    if (cnt > 2*max_cnt)
      {
        // Whether or not this is a degenerate map in the
        // "Jacobian becomes singular or negative" sense,
        // it's at least degenerate in the "straightforward
        // Newton is failing to invert it" sense.
        libmesh_degenerate_mapping_msg(
          "inverse_map Newton FAILED to converge in " <<
          cnt << " iterations in element " << elem->id() <<
          " for physical point = " << physical_point <<
          std::endl << elem->get_info());
      }

    return false;
  }
};

}



Point FEMap::inverse_map (const unsigned int dim,
                          const Elem * elem,
                          const Point & physical_point,
                          const Real tolerance,
                          const bool secure,
                          const bool
                          extra_checks
                          )
{
  libmesh_assert(elem);
  libmesh_assert_greater_equal (tolerance, 0.);

  libmesh_ignore(extra_checks);

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS

  // TODO: possibly use the extra_checks parameter in InfFEMap::inverse_map() as well.

  if (elem->infinite())
    return InfFEMap::inverse_map(dim, elem, physical_point, tolerance,
                                 secure);
#endif

  // Start logging the map inversion.
  LOG_SCOPE("inverse_map()", "FEMap");

  const HostMapEval eval{dim, elem};
  const HostInverseMapPolicy policy{dim, elem, physical_point, tolerance, secure};

  bool converged;
  const Point p = fe_inverse_map(dim, physical_point, tolerance, secure,
                                 eval, policy, converged);

  // A point Newton gave up on is returned as is, without the checks
  if (!converged)
    return p;



  //  If we are in debug mode and the user requested it, do two extra sanity checks.
#ifdef DEBUG

  if (extra_checks)
    {
      // Make sure the point \p p on the reference element actually
      // does map to the point \p physical_point within a tolerance.

      const Point check = map (dim, elem, p);
      const Point diff  = physical_point - check;

      if (diff.norm() > tolerance)
        {
          libmesh_here();
          libMesh::err << "WARNING:  diff is "
                       << diff.norm()
                       << std::endl
                       << " point="
                       << physical_point;
          libMesh::err << " local=" << check;
          libMesh::err << " lref= " << p;

          elem->print_info(libMesh::err);
        }

      // Make sure the point \p p on the reference element actually
      // is

      if (!elem->on_reference_element(p, 2*tolerance))
        {
          libmesh_here();
          libMesh::err << "WARNING:  inverse_map of physical point "
                       << physical_point
                       << " is not on element." << '\n';
          elem->print_info(libMesh::err);
        }
    }

#endif

  return p;
}



void FEMap::inverse_map (const unsigned int dim,
                         const Elem * elem,
                         const std::vector<Point> & physical_points,
                         std::vector<Point> &       reference_points,
                         const Real tolerance,
                         const bool secure,
                         const bool extra_checks)
{
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
  if (elem->infinite())
    {
      // TODO: possibly use the extra_checks parameter in InfFEMap::inverse_map() as well.
      libmesh_ignore(extra_checks);

      return InfFEMap::inverse_map(dim, elem, physical_points, reference_points, tolerance, secure);
    }
#endif

  // The number of points to find the
  // inverse map of
  const std::size_t n_points = physical_points.size();

  // Resize the vector to hold the points
  // on the reference element
  reference_points.resize(n_points);

  // Find the coordinates on the reference
  // element of each point in physical space
  for (std::size_t p=0; p<n_points; p++)
    reference_points[p] =
      inverse_map (dim, elem, physical_points[p], tolerance, secure, extra_checks);
}



Point FEMap::map (const unsigned int dim,
                  const Elem * elem,
                  const Point & reference_point)
{
  libmesh_assert(elem);
  libmesh_ignore(dim);

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
  if (elem->infinite())
    return InfFEMap::map(dim, elem, reference_point);
#endif

  Point p;

  const FEFamily mapping_family = FEMap::map_fe_type(*elem);
  const FEType fe_type (elem->default_order(), mapping_family);

  // Do not consider the Elem::p_level(), if any, when computing the
  // number of shape functions.
  const unsigned int n_sf = FEInterface::n_shape_functions(fe_type, /*extra_order=*/0, elem);

  FEInterface::shape_ptr shape_ptr =
    FEInterface::shape_function(fe_type, elem);

  // Lagrange basis functions are used for mapping
  for (unsigned int i=0; i<n_sf; i++)
    p.add_scaled (elem->point(i),
                  shape_ptr(fe_type, elem, i, reference_point, false));

  return p;
}



Point FEMap::map_deriv (const unsigned int dim,
                        const Elem * elem,
                        const unsigned int j,
                        const Point & reference_point)
{
  libmesh_assert(elem);
  libmesh_ignore(dim);

  if (elem->infinite())
    libmesh_not_implemented();

  Point p;

  const FEFamily mapping_family = FEMap::map_fe_type(*elem);
  const FEType fe_type (elem->default_order(), mapping_family);

  // Do not consider the Elem::p_level(), if any, when computing the
  // number of shape functions.
  const unsigned int n_sf =
    FEInterface::n_shape_functions(fe_type, /*total_order=*/0, elem);

  FEInterface::shape_deriv_ptr shape_deriv_ptr =
    FEInterface::shape_deriv_function(fe_type, elem);

  // Lagrange basis functions are used for mapping
  for (unsigned int i=0; i<n_sf; i++)
    p.add_scaled (elem->point(i),
                  shape_deriv_ptr(fe_type, elem, i, j, reference_point,
                                  /*add_p_level=*/false));

  return p;
}



// Explicit instantiation of FEMap member functions
template LIBMESH_EXPORT void FEMap::init_reference_to_physical_map<0>( const std::vector<Point> &, const Elem *);
template LIBMESH_EXPORT void FEMap::init_reference_to_physical_map<1>( const std::vector<Point> &, const Elem *);
template LIBMESH_EXPORT void FEMap::init_reference_to_physical_map<2>( const std::vector<Point> &, const Elem *);
template LIBMESH_EXPORT void FEMap::init_reference_to_physical_map<3>( const std::vector<Point> &, const Elem *);

// subdivision elements are implemented only for 2D meshes & reimplement
// the inverse_maps method separately
INSTANTIATE_SUBDIVISION_MAPS;

} // namespace libMesh
