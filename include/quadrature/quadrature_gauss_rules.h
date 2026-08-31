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

// Gauss quadrature rules shared between the host QGauss classes and
// Kokkos device code.  The device-readable constexpr value functions are
// the single source; the host-side rule arrays are generated from them at
// compile time.

#ifndef LIBMESH_QUADRATURE_GAUSS_RULES_H
#define LIBMESH_QUADRATURE_GAUSS_RULES_H

#include "libmesh/libmesh.h" // for the _R literal suffix
#include "libmesh/libmesh_common.h"
#include "libmesh/libmesh_device.h"

namespace libMesh::Quadrature::Gauss
{

struct PointWeight2D
{
  Real x;
  Real y;
  Real w;
};

struct PointWeight3D
{
  Real x;
  Real y;
  Real z;
  Real w;
};

namespace detail
{

LIBMESH_DEVICE_INLINE constexpr
unsigned char gauss_legendre_rule_id(const unsigned int order)
{
  // Rules 1..7 integrate orders 0..13; an n-point rule id doubles as its
  // point count.
  return order <= 13 ? static_cast<unsigned char>(order / 2 + 1) : 0u;
}

LIBMESH_DEVICE_INLINE constexpr
unsigned char triangle_rule_id(const unsigned int order)
{
  return order <= 7 ? static_cast<unsigned char>(order ? order : 1) : 0u;
}

LIBMESH_DEVICE_INLINE constexpr
unsigned char tetrahedron_rule_id(const unsigned int order,
                                  const bool allow_negative_weights)
{
  if (order > 6 ||
      ((order == 3 || order == 4) && !allow_negative_weights))
    return 0u;
  return static_cast<unsigned char>(order ? order : 1);
}

// Per-rule point counts — the single place the rule sizes are recorded.
LIBMESH_DEVICE_INLINE constexpr
unsigned int triangle_rule_size(const unsigned char rule_id)
{
  switch (rule_id)
  {
    case 1: return 1;
    case 2: return 3;
    case 3: return 4;
    case 4: return 6;
    case 5: return 7;
    case 6:
    case 7: return 12;
    default: return 0;
  }
}

LIBMESH_DEVICE_INLINE constexpr
unsigned int tetrahedron_rule_size(const unsigned char rule_id)
{
  switch (rule_id)
  {
    case 1: return 1;
    case 2: return 4;
    case 3: return 5;
    case 4: return 11;
    case 5: return 14;
    case 6: return 24;
    default: return 0;
  }
}

LIBMESH_DEVICE_INLINE constexpr
Real gauss_legendre_point_value(const unsigned char rule_id,
                                const unsigned int i)
{
  switch (rule_id)
  {
    case 1:
      return 0._R;

    case 2:
      switch (i)
      {
        case 0: return -5.7735026918962576450914878050196e-01_R;
        case 1: return  5.7735026918962576450914878050196e-01_R;
        default: return 0._R;
      }

    case 3:
      switch (i)
      {
        case 0: return -7.7459666924148337703585307995648e-01_R;
        case 1: return  0._R;
        case 2: return  7.7459666924148337703585307995648e-01_R;
        default: return 0._R;
      }

    case 4:
      switch (i)
      {
        case 0: return -8.6113631159405257522394648889281e-01_R;
        case 1: return -3.3998104358485626480266575910324e-01_R;
        case 2: return  3.3998104358485626480266575910324e-01_R;
        case 3: return  8.6113631159405257522394648889281e-01_R;
        default: return 0._R;
      }

    case 5:
      switch (i)
      {
        case 0: return -9.0617984593866399279762687829939e-01_R;
        case 1: return -5.3846931010568309103631442070021e-01_R;
        case 2: return  0._R;
        case 3: return  5.3846931010568309103631442070021e-01_R;
        case 4: return  9.0617984593866399279762687829939e-01_R;
        default: return 0._R;
      }

    case 6:
      switch (i)
      {
        case 0: return -9.3246951420315202781230155449399e-01_R;
        case 1: return -6.6120938646626451366139959501991e-01_R;
        case 2: return -2.3861918608319690863050172168071e-01_R;
        case 3: return  2.3861918608319690863050172168071e-01_R;
        case 4: return  6.6120938646626451366139959501991e-01_R;
        case 5: return  9.3246951420315202781230155449399e-01_R;
        default: return 0._R;
      }

    case 7:
      switch (i)
      {
        case 0: return -9.4910791234275852452618968404785e-01_R;
        case 1: return -7.4153118559939443986386477328079e-01_R;
        case 2: return -4.0584515137739716690660641207696e-01_R;
        case 3: return  0._R;
        case 4: return  4.0584515137739716690660641207696e-01_R;
        case 5: return  7.4153118559939443986386477328079e-01_R;
        case 6: return  9.4910791234275852452618968404785e-01_R;
        default: return 0._R;
      }

    default:
      return 0._R;
  }
}

LIBMESH_DEVICE_INLINE constexpr
Real gauss_legendre_weight_value(const unsigned char rule_id,
                                 const unsigned int i)
{
  switch (rule_id)
  {
    case 1:
      return i == 0 ? 2._R : 0._R;

    case 2:
      switch (i)
      {
        case 0:
        case 1:
          return 1._R;
        default:
          return 0._R;
      }

    case 3:
      switch (i)
      {
        case 0: return 5.5555555555555555555555555555556e-01_R;
        case 1: return 8.8888888888888888888888888888889e-01_R;
        case 2: return 5.5555555555555555555555555555556e-01_R;
        default: return 0._R;
      }

    case 4:
      switch (i)
      {
        case 0: return 3.4785484513745385737306394922200e-01_R;
        case 1: return 6.5214515486254614262693605077800e-01_R;
        case 2: return 6.5214515486254614262693605077800e-01_R;
        case 3: return 3.4785484513745385737306394922200e-01_R;
        default: return 0._R;
      }

    case 5:
      switch (i)
      {
        case 0: return 2.3692688505618908751426404071992e-01_R;
        case 1: return 4.7862867049936646804129151483564e-01_R;
        case 2: return 5.6888888888888888888888888888889e-01_R;
        case 3: return 4.7862867049936646804129151483564e-01_R;
        case 4: return 2.3692688505618908751426404071992e-01_R;
        default: return 0._R;
      }

    case 6:
      switch (i)
      {
        case 0: return 1.7132449237917034504029614217273e-01_R;
        case 1: return 3.6076157304813860756983351383772e-01_R;
        case 2: return 4.6791393457269104738987034398955e-01_R;
        case 3: return 4.6791393457269104738987034398955e-01_R;
        case 4: return 3.6076157304813860756983351383772e-01_R;
        case 5: return 1.7132449237917034504029614217273e-01_R;
        default: return 0._R;
      }

    case 7:
      switch (i)
      {
        case 0: return 1.2948496616886969327061143267908e-01_R;
        case 1: return 2.7970539148927666790146777142378e-01_R;
        case 2: return 3.8183005050511894495036977548898e-01_R;
        case 3: return 4.1795918367346938775510204081633e-01_R;
        case 4: return 3.8183005050511894495036977548898e-01_R;
        case 5: return 2.7970539148927666790146777142378e-01_R;
        case 6: return 1.2948496616886969327061143267908e-01_R;
        default: return 0._R;
      }

    default:
      return 0._R;
  }
}

LIBMESH_DEVICE_INLINE constexpr
PointWeight2D triangle_rule_point_value(const unsigned char rule_id,
                                        const unsigned int i)
{
  switch (rule_id)
  {
    case 1:
      switch (i)
      {
        case 0: return {Real(1) / 3, Real(1) / 3, Real(1) / 2};
        default: return {0._R, 0._R, 0._R};
      }

    case 2:
      switch (i)
      {
        case 0: return {Real(2) / 3, Real(1) / 6, Real(1) / 6};
        case 1: return {Real(1) / 6, Real(2) / 3, Real(1) / 6};
        case 2: return {Real(1) / 6, Real(1) / 6, Real(1) / 6};
        default: return {0._R, 0._R, 0._R};
      }

    case 3:
      switch (i)
      {
        case 0: return {1.5505102572168219018027159252941e-01_R, 1.7855872826361642311703513337422e-01_R, 1.5902069087198858469718450103758e-01_R};
        case 1: return {6.4494897427831780981972840747059e-01_R, 7.5031110222608118177475598324603e-02_R, 9.0979309128011415302815498962418e-02_R};
        case 2: return {1.5505102572168219018027159252941e-01_R, 6.6639024601470138670269327409637e-01_R, 1.5902069087198858469718450103758e-01_R};
        case 3: return {6.4494897427831780981972840747059e-01_R, 2.8001991549907407200279599420481e-01_R, 9.0979309128011415302815498962418e-02_R};
        default: return {0._R, 0._R, 0._R};
      }

    case 4:
      switch (i)
      {
        case 0: return {4.4594849091596488631832925388305199e-01_R, 4.4594849091596488631832925388305199e-01_R, 1.1169079483900573284750350421656140e-01_R};
        case 1: return {4.4594849091596488631832925388305199e-01_R, 1.0810301816807022736334149223389602e-01_R, 1.1169079483900573284750350421656140e-01_R};
        case 2: return {1.0810301816807022736334149223389602e-01_R, 4.4594849091596488631832925388305199e-01_R, 1.1169079483900573284750350421656140e-01_R};
        case 3: return {9.1576213509770743459571463402201508e-02_R, 9.1576213509770743459571463402201508e-02_R, 5.4975871827660933819163162450105264e-02_R};
        case 4: return {9.1576213509770743459571463402201508e-02_R, 8.1684757298045851308085707319559698e-01_R, 5.4975871827660933819163162450105264e-02_R};
        case 5: return {8.1684757298045851308085707319559698e-01_R, 9.1576213509770743459571463402201508e-02_R, 5.4975871827660933819163162450105264e-02_R};
        default: return {0._R, 0._R, 0._R};
      }

    case 5:
      switch (i)
      {
        case 0: return {Real(1) / 3, Real(1) / 3, Real(9) / 80};
        case 1: return {4.7014206410511510905399914607621e-01_R, 4.7014206410511510905399914607621e-01_R, 6.6197076394253091074561155334870e-02_R};
        case 2: return {4.7014206410511510905399914607621e-01_R, 5.9715871789769778189200170784758e-02_R, 6.6197076394253091074561155334870e-02_R};
        case 3: return {5.9715871789769778189200170784758e-02_R, 4.7014206410511510905399914607621e-01_R, 6.6197076394253091074561155334870e-02_R};
        case 4: return {1.0128650732345631956123890497612e-01_R, 1.0128650732345631956123890497612e-01_R, 6.2969590272413576325050340212953e-02_R};
        case 5: return {1.0128650732345631956123890497612e-01_R, 7.9742698535308736087752219004777e-01_R, 6.2969590272413576325050340212953e-02_R};
        case 6: return {7.9742698535308736087752219004777e-01_R, 1.0128650732345631956123890497612e-01_R, 6.2969590272413576325050340212953e-02_R};
        default: return {0._R, 0._R, 0._R};
      }

    case 6:
      {
        constexpr Real tri6_a1 = 2.4928674517091042129163855310701908e-01_R;
        constexpr Real tri6_b1 = 1._R - 2._R * tri6_a1;
        constexpr Real tri6_a2 = 6.3089014491502228340331602870819157e-02_R;
        constexpr Real tri6_b2 = 1._R - 2._R * tri6_a2;
        constexpr Real tri6_a3 = 3.1035245103378440541660773395655215e-01_R;
        constexpr Real tri6_b3 = 6.3650249912139864723014259441204970e-01_R;
        constexpr Real tri6_c3 = 1._R - tri6_a3 - tri6_b3;

        switch (i)
        {
          case 0: return {tri6_a1, tri6_a1, 5.8393137863189683012644805692789721e-02_R};
          case 1: return {tri6_a1, tri6_b1, 5.8393137863189683012644805692789721e-02_R};
          case 2: return {tri6_b1, tri6_a1, 5.8393137863189683012644805692789721e-02_R};
          case 3: return {tri6_a2, tri6_a2, 2.5422453185103408460468404553434492e-02_R};
          case 4: return {tri6_a2, tri6_b2, 2.5422453185103408460468404553434492e-02_R};
          case 5: return {tri6_b2, tri6_a2, 2.5422453185103408460468404553434492e-02_R};
          case 6: return {tri6_a3, tri6_b3, 4.1425537809186787596776728210221227e-02_R};
          case 7: return {tri6_b3, tri6_a3, 4.1425537809186787596776728210221227e-02_R};
          case 8: return {tri6_a3, tri6_c3, 4.1425537809186787596776728210221227e-02_R};
          case 9: return {tri6_c3, tri6_a3, 4.1425537809186787596776728210221227e-02_R};
          case 10: return {tri6_b3, tri6_c3, 4.1425537809186787596776728210221227e-02_R};
          case 11: return {tri6_c3, tri6_b3, 4.1425537809186787596776728210221227e-02_R};
          default: return {0._R, 0._R, 0._R};
        }
      }

    case 7:
      switch (i)
      {
        case 0: return {6.2382265094402118173683000996350e-02_R, 6.7517867073916085442557131050869e-02_R, 2.6517028157436251428754180460739e-02_R};
        case 1: return {8.7009986783168172748385986795285e-01_R, 6.2382265094402118173683000996350e-02_R, 2.6517028157436251428754180460739e-02_R};
        case 2: return {6.7517867073916085442557131050869e-02_R, 8.7009986783168172748385986795285e-01_R, 2.6517028157436251428754180460739e-02_R};
        case 3: return {5.5225456656926611737479190275645e-02_R, 3.2150249385198182266630784919920e-01_R, 4.3881408714446055036769903139288e-02_R};
        case 4: return {6.2327204949109156559621296052516e-01_R, 5.5225456656926611737479190275645e-02_R, 4.3881408714446055036769903139288e-02_R};
        case 5: return {3.2150249385198182266630784919920e-01_R, 6.2327204949109156559621296052516e-01_R, 4.3881408714446055036769903139288e-02_R};
        case 6: return {3.4324302945097146469630642483938e-02_R, 6.6094919618673565761198031019780e-01_R, 2.8775042784981585738445496900219e-02_R};
        case 7: return {3.0472650086816719591838904731826e-01_R, 3.4324302945097146469630642483938e-02_R, 2.8775042784981585738445496900219e-02_R};
        case 8: return {6.6094919618673565761198031019780e-01_R, 3.0472650086816719591838904731826e-01_R, 2.8775042784981585738445496900219e-02_R};
        case 9: return {5.1584233435359177925746338682643e-01_R, 2.7771616697639178256958187139372e-01_R, 6.7493187009802774462697086166421e-02_R};
        case 10: return {2.0644149867001643817295474177985e-01_R, 5.1584233435359177925746338682643e-01_R, 6.7493187009802774462697086166421e-02_R};
        case 11: return {2.7771616697639178256958187139372e-01_R, 2.0644149867001643817295474177985e-01_R, 6.7493187009802774462697086166421e-02_R};
        default: return {0._R, 0._R, 0._R};
      }

    default:
      return {0._R, 0._R, 0._R};
  }
}

LIBMESH_DEVICE_INLINE constexpr
PointWeight3D tetrahedron_rule_point_value(const unsigned char rule_id,
                                           const unsigned int i)
{
  switch (rule_id)
  {
    case 1:
      switch (i)
      {
        case 0: return {0.25_R, 0.25_R, 0.25_R, Real(1) / 6};
        default: return {0._R, 0._R, 0._R, 0._R};
      }

    case 2:
    {
      const Real tet2_b = 0.25_R * (1._R - 1._R / 2.2360679774997896964_R);
      const Real tet2_a = 1._R - 3._R * tet2_b;

      switch (i)
      {
        case 0: return {tet2_a, tet2_b, tet2_b, Real(1) / 24};
        case 1: return {tet2_b, tet2_a, tet2_b, Real(1) / 24};
        case 2: return {tet2_b, tet2_b, tet2_a, Real(1) / 24};
        case 3: return {tet2_b, tet2_b, tet2_b, Real(1) / 24};
        default: return {0._R, 0._R, 0._R, 0._R};
      }
    }

    case 3:
      switch (i)
      {
        case 0: return {0.25_R, 0.25_R, 0.25_R, Real(-2) / 15};
        case 1: return {0.5_R, Real(1) / 6, Real(1) / 6, 0.075_R};
        case 2: return {Real(1) / 6, 0.5_R, Real(1) / 6, 0.075_R};
        case 3: return {Real(1) / 6, Real(1) / 6, 0.5_R, 0.075_R};
        case 4: return {Real(1) / 6, Real(1) / 6, Real(1) / 6, 0.075_R};
        default: return {0._R, 0._R, 0._R, 0._R};
      }

    case 4:
      switch (i)
      {
        case 0: return {2.5e-01_R, 2.5e-01_R, 2.5e-01_R, -1.31555555555555556e-02_R};
        case 1: return {7.85714285714285714e-01_R, 7.14285714285714285e-02_R, 7.14285714285714285e-02_R, 7.62222222222222222e-03_R};
        case 2: return {7.14285714285714285e-02_R, 7.85714285714285714e-01_R, 7.14285714285714285e-02_R, 7.62222222222222222e-03_R};
        case 3: return {7.14285714285714285e-02_R, 7.14285714285714285e-02_R, 7.85714285714285714e-01_R, 7.62222222222222222e-03_R};
        case 4: return {7.14285714285714285e-02_R, 7.14285714285714285e-02_R, 7.14285714285714285e-02_R, 7.62222222222222222e-03_R};
        case 5: return {3.99403576166799219e-01_R, 3.99403576166799219e-01_R, 1.00596423833200785e-01_R, 2.48888888888888889e-02_R};
        case 6: return {3.99403576166799219e-01_R, 1.00596423833200785e-01_R, 1.00596423833200785e-01_R, 2.48888888888888889e-02_R};
        case 7: return {1.00596423833200785e-01_R, 1.00596423833200785e-01_R, 3.99403576166799219e-01_R, 2.48888888888888889e-02_R};
        case 8: return {1.00596423833200785e-01_R, 3.99403576166799219e-01_R, 1.00596423833200785e-01_R, 2.48888888888888889e-02_R};
        case 9: return {1.00596423833200785e-01_R, 3.99403576166799219e-01_R, 3.99403576166799219e-01_R, 2.48888888888888889e-02_R};
        case 10: return {3.99403576166799219e-01_R, 1.00596423833200785e-01_R, 3.99403576166799219e-01_R, 2.48888888888888889e-02_R};
        default: return {0._R, 0._R, 0._R, 0._R};
      }

    case 5:
      switch (i)
      {
        case 0: return {3.1088591926330060980e-01_R, 3.1088591926330060980e-01_R, 3.1088591926330060980e-01_R, 1.8781320953002641800e-02_R};
        case 1: return {3.1088591926330060980e-01_R, 6.7342242201009817060e-02_R, 3.1088591926330060980e-01_R, 1.8781320953002641800e-02_R};
        case 2: return {6.7342242201009817060e-02_R, 3.1088591926330060980e-01_R, 3.1088591926330060980e-01_R, 1.8781320953002641800e-02_R};
        case 3: return {3.1088591926330060980e-01_R, 3.1088591926330060980e-01_R, 6.7342242201009817060e-02_R, 1.8781320953002641800e-02_R};
        case 4: return {9.2735250310891226402e-02_R, 9.2735250310891226402e-02_R, 9.2735250310891226402e-02_R, 1.2248840519393658257e-02_R};
        case 5: return {9.2735250310891226402e-02_R, 7.2179424906732632079e-01_R, 9.2735250310891226402e-02_R, 1.2248840519393658257e-02_R};
        case 6: return {7.2179424906732632079e-01_R, 9.2735250310891226402e-02_R, 9.2735250310891226402e-02_R, 1.2248840519393658257e-02_R};
        case 7: return {9.2735250310891226402e-02_R, 9.2735250310891226402e-02_R, 7.2179424906732632079e-01_R, 1.2248840519393658257e-02_R};
        case 8: return {4.5503704125649649492e-02_R, 4.5449629587435035051e-01_R, 4.5449629587435035051e-01_R, 7.0910034628469110730e-03_R};
        case 9: return {4.5449629587435035051e-01_R, 4.5503704125649649492e-02_R, 4.5503704125649649492e-02_R, 7.0910034628469110730e-03_R};
        case 10: return {4.5503704125649649492e-02_R, 4.5503704125649649492e-02_R, 4.5449629587435035051e-01_R, 7.0910034628469110730e-03_R};
        case 11: return {4.5503704125649649492e-02_R, 4.5449629587435035051e-01_R, 4.5503704125649649492e-02_R, 7.0910034628469110730e-03_R};
        case 12: return {4.5449629587435035051e-01_R, 4.5503704125649649492e-02_R, 4.5449629587435035051e-01_R, 7.0910034628469110730e-03_R};
        case 13: return {4.5449629587435035051e-01_R, 4.5449629587435035051e-01_R, 4.5503704125649649492e-02_R, 7.0910034628469110730e-03_R};
        default: return {0._R, 0._R, 0._R, 0._R};
      }

    case 6:
      switch (i)
      {
        case 0: return {3.56191386222544953e-01_R, 2.14602871259151684e-01_R, 2.14602871259151684e-01_R, 6.65379170969464506e-03_R};
        case 1: return {2.14602871259151684e-01_R, 3.56191386222544953e-01_R, 2.14602871259151684e-01_R, 6.65379170969464506e-03_R};
        case 2: return {2.14602871259151684e-01_R, 2.14602871259151684e-01_R, 3.56191386222544953e-01_R, 6.65379170969464506e-03_R};
        case 3: return {2.14602871259151684e-01_R, 2.14602871259151684e-01_R, 2.14602871259151684e-01_R, 6.65379170969464506e-03_R};
        case 4: return {8.77978124396165982e-01_R, 4.06739585346113397e-02_R, 4.06739585346113397e-02_R, 1.67953517588677620e-03_R};
        case 5: return {4.06739585346113397e-02_R, 8.77978124396165982e-01_R, 4.06739585346113397e-02_R, 1.67953517588677620e-03_R};
        case 6: return {4.06739585346113397e-02_R, 4.06739585346113397e-02_R, 8.77978124396165982e-01_R, 1.67953517588677620e-03_R};
        case 7: return {4.06739585346113397e-02_R, 4.06739585346113397e-02_R, 4.06739585346113397e-02_R, 1.67953517588677620e-03_R};
        case 8: return {3.29863295731730594e-02_R, 3.22337890142275646e-01_R, 3.22337890142275646e-01_R, 9.22619692394239843e-03_R};
        case 9: return {3.22337890142275646e-01_R, 3.29863295731730594e-02_R, 3.22337890142275646e-01_R, 9.22619692394239843e-03_R};
        case 10: return {3.22337890142275646e-01_R, 3.22337890142275646e-01_R, 3.29863295731730594e-02_R, 9.22619692394239843e-03_R};
        case 11: return {3.22337890142275646e-01_R, 3.22337890142275646e-01_R, 3.22337890142275646e-01_R, 9.22619692394239843e-03_R};
        case 12: return {6.36610018750175299e-02_R, 6.36610018750175299e-02_R, 2.69672331458315867e-01_R, 8.03571428571428248e-03_R};
        case 13: return {6.36610018750175299e-02_R, 6.36610018750175299e-02_R, 6.03005664791649076e-01_R, 8.03571428571428248e-03_R};
        case 14: return {2.69672331458315867e-01_R, 6.36610018750175299e-02_R, 6.36610018750175299e-02_R, 8.03571428571428248e-03_R};
        case 15: return {6.03005664791649076e-01_R, 6.36610018750175299e-02_R, 6.36610018750175299e-02_R, 8.03571428571428248e-03_R};
        case 16: return {6.36610018750175299e-02_R, 2.69672331458315867e-01_R, 6.36610018750175299e-02_R, 8.03571428571428248e-03_R};
        case 17: return {6.36610018750175299e-02_R, 6.03005664791649076e-01_R, 6.36610018750175299e-02_R, 8.03571428571428248e-03_R};
        case 18: return {6.36610018750175299e-02_R, 2.69672331458315867e-01_R, 6.03005664791649076e-01_R, 8.03571428571428248e-03_R};
        case 19: return {6.36610018750175299e-02_R, 6.03005664791649076e-01_R, 2.69672331458315867e-01_R, 8.03571428571428248e-03_R};
        case 20: return {2.69672331458315867e-01_R, 6.36610018750175299e-02_R, 6.03005664791649076e-01_R, 8.03571428571428248e-03_R};
        case 21: return {2.69672331458315867e-01_R, 6.03005664791649076e-01_R, 6.36610018750175299e-02_R, 8.03571428571428248e-03_R};
        case 22: return {6.03005664791649076e-01_R, 6.36610018750175299e-02_R, 2.69672331458315867e-01_R, 8.03571428571428248e-03_R};
        case 23: return {6.03005664791649076e-01_R, 2.69672331458315867e-01_R, 6.36610018750175299e-02_R, 8.03571428571428248e-03_R};
        default: return {0._R, 0._R, 0._R, 0._R};
      }

    default:
      return {0._R, 0._R, 0._R, 0._R};
  }
}

} // namespace detail

struct Rule1D
{
  unsigned int count;
  const Real * points;
  const Real * weights;
};

struct Rule2D
{
  unsigned int count;
  const PointWeight2D * points;
};

struct Rule3D
{
  unsigned int count;
  const PointWeight3D * points;
};

// The host-side rule storage is generated at compile time from the same
// detail:: value functions the device path reads, so every point and weight
// exists in exactly one place.
namespace detail
{

template <unsigned int N> struct Rule1DData { Real points[N]; Real weights[N]; };
template <unsigned int N> struct Rule2DData { PointWeight2D points[N]; };
template <unsigned int N> struct Rule3DData { PointWeight3D points[N]; };

template <unsigned char Id>
constexpr Rule1DData<Id> build_gauss_legendre_rule()
{
  Rule1DData<Id> r {};
  for (unsigned int i = 0; i != Id; ++i)
    {
      r.points[i] = gauss_legendre_point_value(Id, i);
      r.weights[i] = gauss_legendre_weight_value(Id, i);
    }
  return r;
}

template <unsigned char Id>
constexpr Rule2DData<triangle_rule_size(Id)> build_triangle_rule()
{
  Rule2DData<triangle_rule_size(Id)> r {};
  for (unsigned int i = 0; i != triangle_rule_size(Id); ++i)
    r.points[i] = triangle_rule_point_value(Id, i);
  return r;
}

template <unsigned char Id>
constexpr Rule3DData<tetrahedron_rule_size(Id)> build_tetrahedron_rule()
{
  Rule3DData<tetrahedron_rule_size(Id)> r {};
  for (unsigned int i = 0; i != tetrahedron_rule_size(Id); ++i)
    r.points[i] = tetrahedron_rule_point_value(Id, i);
  return r;
}

} // namespace detail

inline constexpr auto gauss_legendre_data_1 = detail::build_gauss_legendre_rule<1>();
inline constexpr auto gauss_legendre_data_2 = detail::build_gauss_legendre_rule<2>();
inline constexpr auto gauss_legendre_data_3 = detail::build_gauss_legendre_rule<3>();
inline constexpr auto gauss_legendre_data_4 = detail::build_gauss_legendre_rule<4>();
inline constexpr auto gauss_legendre_data_5 = detail::build_gauss_legendre_rule<5>();
inline constexpr auto gauss_legendre_data_6 = detail::build_gauss_legendre_rule<6>();
inline constexpr auto gauss_legendre_data_7 = detail::build_gauss_legendre_rule<7>();

inline constexpr auto tri_data_1 = detail::build_triangle_rule<1>();
inline constexpr auto tri_data_2 = detail::build_triangle_rule<2>();
inline constexpr auto tri_data_3 = detail::build_triangle_rule<3>();
inline constexpr auto tri_data_4 = detail::build_triangle_rule<4>();
inline constexpr auto tri_data_5 = detail::build_triangle_rule<5>();
inline constexpr auto tri_data_6 = detail::build_triangle_rule<6>();
inline constexpr auto tri_data_7 = detail::build_triangle_rule<7>();

inline constexpr auto tet_data_1 = detail::build_tetrahedron_rule<1>();
inline constexpr auto tet_data_2 = detail::build_tetrahedron_rule<2>();
inline constexpr auto tet_data_3 = detail::build_tetrahedron_rule<3>();
inline constexpr auto tet_data_4 = detail::build_tetrahedron_rule<4>();
inline constexpr auto tet_data_5 = detail::build_tetrahedron_rule<5>();
inline constexpr auto tet_data_6 = detail::build_tetrahedron_rule<6>();

// One Rule per rule id; the public order-based lookups below share the
// detail::*_rule_id() mapping with the device path.
inline constexpr Rule1D gauss_legendre_rules[] = {
  {1u, gauss_legendre_data_1.points, gauss_legendre_data_1.weights},
  {2u, gauss_legendre_data_2.points, gauss_legendre_data_2.weights},
  {3u, gauss_legendre_data_3.points, gauss_legendre_data_3.weights},
  {4u, gauss_legendre_data_4.points, gauss_legendre_data_4.weights},
  {5u, gauss_legendre_data_5.points, gauss_legendre_data_5.weights},
  {6u, gauss_legendre_data_6.points, gauss_legendre_data_6.weights},
  {7u, gauss_legendre_data_7.points, gauss_legendre_data_7.weights}
};

inline constexpr Rule2D triangle_rules[] = {
  {1u, tri_data_1.points},
  {3u, tri_data_2.points},
  {4u, tri_data_3.points},
  {6u, tri_data_4.points},
  {7u, tri_data_5.points},
  {12u, tri_data_6.points},
  {12u, tri_data_7.points}
};

inline constexpr Rule3D tetrahedron_rules[] = {
  {1u, tet_data_1.points},
  {4u, tet_data_2.points},
  {5u, tet_data_3.points},
  {11u, tet_data_4.points},
  {14u, tet_data_5.points},
  {24u, tet_data_6.points}
};

LIBMESH_DEVICE_INLINE
Rule1D gauss_legendre_rule(const unsigned int order)
{
  const unsigned char id = detail::gauss_legendre_rule_id(order);
  return id ? gauss_legendre_rules[id - 1] : Rule1D{0u, nullptr, nullptr};
}

LIBMESH_DEVICE_INLINE
Rule2D triangle_rule(const unsigned int order)
{
  const unsigned char id = detail::triangle_rule_id(order);
  return id ? triangle_rules[id - 1] : Rule2D{0u, nullptr};
}

LIBMESH_DEVICE_INLINE
Rule3D tetrahedron_rule(const unsigned int order,
                        const bool allow_negative_weights = true)
{
  const unsigned char id = detail::tetrahedron_rule_id(order, allow_negative_weights);
  return id ? tetrahedron_rules[id - 1] : Rule3D{0u, nullptr};
}

LIBMESH_DEVICE_INLINE
Real gauss_legendre_point(const unsigned int order,
                          const unsigned int i)
{
#ifdef ACSM_KOKKOS_COMPILATION
  return detail::gauss_legendre_point_value(detail::gauss_legendre_rule_id(order), i);
#else
  const auto rule = gauss_legendre_rule(order);
  return (i < rule.count) ? rule.points[i] : 0._R;
#endif
}

LIBMESH_DEVICE_INLINE
unsigned int gauss_legendre_count(const unsigned int order)
{
#ifdef ACSM_KOKKOS_COMPILATION
  return detail::gauss_legendre_rule_id(order);
#else
  return gauss_legendre_rule(order).count;
#endif
}

LIBMESH_DEVICE_INLINE
Real gauss_legendre_weight(const unsigned int order,
                           const unsigned int i)
{
#ifdef ACSM_KOKKOS_COMPILATION
  return detail::gauss_legendre_weight_value(detail::gauss_legendre_rule_id(order), i);
#else
  const auto rule = gauss_legendre_rule(order);
  return (i < rule.count) ? rule.weights[i] : 0._R;
#endif
}

LIBMESH_DEVICE_INLINE
unsigned int triangle_count(const unsigned int order)
{
#ifdef ACSM_KOKKOS_COMPILATION
  return detail::triangle_rule_size(detail::triangle_rule_id(order));
#else
  return triangle_rule(order).count;
#endif
}

LIBMESH_DEVICE_INLINE
PointWeight2D triangle_point(const unsigned int order,
                             const unsigned int i)
{
#ifdef ACSM_KOKKOS_COMPILATION
  return detail::triangle_rule_point_value(detail::triangle_rule_id(order), i);
#else
  const auto rule = triangle_rule(order);
  return (i < rule.count) ? rule.points[i] : PointWeight2D{0._R, 0._R, 0._R};
#endif
}

LIBMESH_DEVICE_INLINE
Real triangle_weight(const unsigned int order,
                     const unsigned int i)
{
  return triangle_point(order, i).w;
}

LIBMESH_DEVICE_INLINE
unsigned int tetrahedron_count(const unsigned int order,
                               const bool allow_negative_weights = true)
{
#ifdef ACSM_KOKKOS_COMPILATION
  return detail::tetrahedron_rule_size(detail::tetrahedron_rule_id(order, allow_negative_weights));
#else
  return tetrahedron_rule(order, allow_negative_weights).count;
#endif
}

LIBMESH_DEVICE_INLINE
PointWeight3D tetrahedron_point(const unsigned int order,
                                const unsigned int i,
                                const bool allow_negative_weights = true)
{
#ifdef ACSM_KOKKOS_COMPILATION
  return detail::tetrahedron_rule_point_value(detail::tetrahedron_rule_id(order, allow_negative_weights), i);
#else
  const auto rule = tetrahedron_rule(order, allow_negative_weights);
  return (i < rule.count) ? rule.points[i] : PointWeight3D{0._R, 0._R, 0._R, 0._R};
#endif
}

LIBMESH_DEVICE_INLINE
Real tetrahedron_weight(const unsigned int order,
                        const unsigned int i,
                        const bool allow_negative_weights = true)
{
  return tetrahedron_point(order, i, allow_negative_weights).w;
}

} // namespace libMesh::Quadrature::Gauss

#endif // LIBMESH_QUADRATURE_GAUSS_RULES_H
