// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// Kokkos volume assembly for the standalone adaptivity_ex2 Kokkos driver.

#include "libmesh/libmesh_config.h"

#if defined(LIBMESH_HAVE_KOKKOS) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)

#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_shape_traits.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/mesh_base.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/static_condensation.h"
#include "libmesh/transient_system.h"

#include "gpu/kokkos_fe_evaluator.h"
#include "gpu/kokkos_fe_map.h"
#include "gpu/kokkos_quadrature.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <vector>

using namespace libMesh;

Real adaptivity_ex2_kokkos_exact_solution(Real, Real, Real);

#ifdef LIBMESH_ENABLE_AMR
namespace
{

constexpr unsigned int kokkos_max_dofs = 9;
constexpr unsigned int kokkos_max_nodes = 9;

struct ElementRecord
{
  const Elem * elem = nullptr;
  std::vector<dof_id_type> dof_indices;
};

template <typename CoordinateView>
struct ElementNodeAccess
{
  CoordinateView coordinates;
  unsigned int elem = 0;

  KOKKOS_INLINE_FUNCTION Real operator()(const unsigned int node,
                                         const unsigned int component) const
  {
    return coordinates(elem, node, component);
  }
};

bool supported_element(const Elem & elem,
                       const FEType & fe_type,
                       const unsigned int n_dofs,
                       const unsigned int quadrature_order)
{
  const int effective_order =
      fe_type.order.get_order() + cast_int<int>(elem.p_level());
  if (effective_order < 1 || effective_order > 2 ||
      fe_type.family != LAGRANGE || elem.mapping_type() != LAGRANGE_MAP ||
      elem.dim() != 2 || elem.n_nodes() > kokkos_max_nodes ||
      n_dofs > kokkos_max_dofs)
    return false;

  switch (elem.type())
  {
    case TRI3:
    case TRI6:
    case QUAD4:
    case QUAD9:
      break;
    default:
      return false;
  }

  const FEShapeKey key{fe_type.family,
                       elem.type(),
                       static_cast<Order>(effective_order)};
  return supports_shape_with_lagrange_map(key) && supports_grad_shape(key) &&
         libMesh::n_dofs_or_zero(key) == n_dofs &&
         libMesh::Kokkos::GaussQuadrature::n_points(elem.type(), quadrature_order) != 0;
}

} // anonymous namespace

void assemble_cd_kokkos(EquationSystems & es,
                        const std::string & libmesh_dbg_var(system_name))
{
  libmesh_assert_equal_to(system_name, "Convection-Diffusion");

  ::Kokkos::Profiling::ScopedRegion region("adaptivity_ex2 Kokkos assembly");

  const MeshBase & mesh = es.get_mesh();
  auto & system =
      es.get_system<TransientLinearImplicitSystem>("Convection-Diffusion");
  const DofMap & dof_map = system.get_dof_map();
  const FEType fe_type = system.variable_type(0);
  const unsigned int quadrature_order =
      cast_int<unsigned int>(fe_type.default_quadrature_order());

  std::vector<ElementRecord> records;
  records.reserve(mesh.n_active_local_elem());
  for (const auto * elem : mesh.active_local_element_ptr_range())
  {
    ElementRecord record;
    record.elem = elem;
    dof_map.dof_indices(elem, record.dof_indices);
    libmesh_error_msg_if(
        !supported_element(*elem,
                           fe_type,
                           cast_int<unsigned int>(record.dof_indices.size()),
                           quadrature_order),
        "Unsupported element in adaptivity_ex2 Kokkos order-1/2 assembly: type="
            << Utility::enum_to_string(elem->type()) << ", p_level=" << elem->p_level()
            << ", n_dofs=" << record.dof_indices.size());
    records.push_back(std::move(record));
  }

  const unsigned int n_elements = cast_int<unsigned int>(records.size());

  using RealView1D = ::Kokkos::View<Real *>;
  using RealView2D = ::Kokkos::View<Real **>;
  using RealView3D = ::Kokkos::View<Real ***>;
  using UIntView1D = ::Kokkos::View<unsigned int *>;
  using ElemTypeView1D = ::Kokkos::View<ElemType *>;
  using MappingTypeView1D = ::Kokkos::View<ElemMappingType *>;

  RealView3D coordinates("adaptivity_ex2 coordinates",
                         n_elements,
                         kokkos_max_nodes,
                         LIBMESH_DIM);
  RealView2D old_values("adaptivity_ex2 old values", n_elements, kokkos_max_dofs);
  UIntView1D n_nodes("adaptivity_ex2 node counts", n_elements);
  UIntView1D n_dofs("adaptivity_ex2 dof counts", n_elements);
  UIntView1D orders("adaptivity_ex2 FE orders", n_elements);
  ElemTypeView1D elem_types("adaptivity_ex2 element types", n_elements);
  MappingTypeView1D mapping_types("adaptivity_ex2 mapping types", n_elements);
  RealView3D element_matrices("adaptivity_ex2 element matrices",
                              n_elements,
                              kokkos_max_dofs,
                              kokkos_max_dofs);
  RealView2D element_rhs("adaptivity_ex2 element rhs", n_elements, kokkos_max_dofs);

  auto h_coordinates = ::Kokkos::create_mirror_view(coordinates);
  auto h_old_values = ::Kokkos::create_mirror_view(old_values);
  auto h_n_nodes = ::Kokkos::create_mirror_view(n_nodes);
  auto h_n_dofs = ::Kokkos::create_mirror_view(n_dofs);
  auto h_orders = ::Kokkos::create_mirror_view(orders);
  auto h_elem_types = ::Kokkos::create_mirror_view(elem_types);
  auto h_mapping_types = ::Kokkos::create_mirror_view(mapping_types);

  for (unsigned int e = 0; e != n_elements; ++e)
  {
    const Elem & elem = *records[e].elem;
    const unsigned int elem_n_dofs =
        cast_int<unsigned int>(records[e].dof_indices.size());
    h_n_nodes(e) = elem.n_nodes();
    h_n_dofs(e) = elem_n_dofs;
    h_orders(e) = cast_int<unsigned int>(fe_type.order.get_order() +
                                         cast_int<int>(elem.p_level()));
    h_elem_types(e) = elem.type();
    h_mapping_types(e) = elem.mapping_type();

    for (unsigned int n = 0; n != kokkos_max_nodes; ++n)
      for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
        h_coordinates(e, n, d) = n < elem.n_nodes() ? elem.point(n)(d) : 0.;

    for (unsigned int i = 0; i != kokkos_max_dofs; ++i)
      h_old_values(e, i) =
          i < elem_n_dofs ? system.old_solution(records[e].dof_indices[i]) : 0.;
  }

  ::Kokkos::deep_copy(coordinates, h_coordinates);
  ::Kokkos::deep_copy(old_values, h_old_values);
  ::Kokkos::deep_copy(n_nodes, h_n_nodes);
  ::Kokkos::deep_copy(n_dofs, h_n_dofs);
  ::Kokkos::deep_copy(orders, h_orders);
  ::Kokkos::deep_copy(elem_types, h_elem_types);
  ::Kokkos::deep_copy(mapping_types, h_mapping_types);
  ::Kokkos::deep_copy(element_matrices, 0.);
  ::Kokkos::deep_copy(element_rhs, 0.);

  const RealVectorValue velocity =
      es.parameters.get<RealVectorValue>("velocity");
  const Real velocity_x = velocity(0);
  const Real velocity_y = velocity(1);
  const Real diffusivity = es.parameters.get<Real>("diffusivity");
  const Real dt = es.parameters.get<Real>("dt");

  using ExecutionSpace = typename RealView1D::execution_space;
  ::Kokkos::parallel_for(
      "adaptivity_ex2 volume integration",
      ::Kokkos::RangePolicy<ExecutionSpace>(0, cast_int<int>(n_elements)),
      KOKKOS_LAMBDA(const int raw_elem) {
        const unsigned int e = static_cast<unsigned int>(raw_elem);
        const unsigned int elem_n_dofs = n_dofs(e);
        const ElemType elem_type = elem_types(e);
        const FEShapeKey shape_key{LAGRANGE,
                                   elem_type,
                                   static_cast<Order>(orders(e))};
        const ElementNodeAccess<RealView3D> elem_nodes{coordinates, e};
        const unsigned int n_qpoints =
            libMesh::Kokkos::GaussQuadrature::n_points(elem_type, quadrature_order);

        for (unsigned int qp = 0; qp != n_qpoints; ++qp)
        {
          const auto reference_point =
              libMesh::Kokkos::GaussQuadrature::point(elem_type, quadrature_order, qp);
          auto xyz = libMesh::Kokkos::zero_vector();
          auto jacobian = libMesh::Kokkos::zero_tensor();
          libMesh::Kokkos::physical_point_and_jacobian(mapping_types(e),
                                                       elem_type,
                                                       elem_nodes,
                                                       n_nodes(e),
                                                       reference_point(0),
                                                       reference_point(1),
                                                       reference_point(2),
                                                       xyz,
                                                       jacobian);
          const auto inverse_jacobian =
              libMesh::Kokkos::inverse<libMesh::Kokkos::RealTensor>(jacobian, 2);
          const Real jxw = libMesh::Kokkos::volume_jxw(
              jacobian,
              2,
              libMesh::Kokkos::GaussQuadrature::weight(
                  elem_type, quadrature_order, qp));

          Real phi[kokkos_max_dofs];
          libMesh::Kokkos::RealVector dphi[kokkos_max_dofs];
          Real u_old = 0.;
          auto grad_u_old = libMesh::Kokkos::zero_vector();

          for (unsigned int i = 0; i != elem_n_dofs; ++i)
          {
            phi[i] = libMesh::Kokkos::shape(shape_key,
                                            i,
                                            reference_point(0),
                                            reference_point(1),
                                            reference_point(2));
            dphi[i] = inverse_jacobian *
                       libMesh::Kokkos::grad_shape(shape_key,
                                                   i,
                                                   reference_point(0),
                                                   reference_point(1),
                                                   reference_point(2));
            u_old += phi[i] * old_values(e, i);
            grad_u_old.add_scaled(dphi[i], old_values(e, i));
          }

          const Real old_convection =
              velocity_x * grad_u_old(0) + velocity_y * grad_u_old(1);
          for (unsigned int i = 0; i != elem_n_dofs; ++i)
          {
            const Real old_diffusion =
                diffusivity * (grad_u_old(0) * dphi[i](0) +
                               grad_u_old(1) * dphi[i](1));
            element_rhs(e, i) +=
                jxw * (u_old * phi[i] - 0.5 * dt *
                       (old_convection * phi[i] + old_diffusion));

            for (unsigned int j = 0; j != elem_n_dofs; ++j)
            {
              const Real convection =
                  (velocity_x * dphi[j](0) + velocity_y * dphi[j](1)) * phi[i];
              const Real diffusion =
                  diffusivity * (dphi[i](0) * dphi[j](0) +
                                 dphi[i](1) * dphi[j](1));
              element_matrices(e, i, j) +=
                  jxw * (phi[i] * phi[j] + 0.5 * dt * (convection + diffusion));
            }
          }
        }
      });

  auto h_element_matrices = ::Kokkos::create_mirror_view(element_matrices);
  auto h_element_rhs = ::Kokkos::create_mirror_view(element_rhs);
  ::Kokkos::deep_copy(h_element_matrices, element_matrices);
  ::Kokkos::deep_copy(h_element_rhs, element_rhs);

  StaticCondensation * static_condensation =
      system.has_static_condensation() ? &system.get_static_condensation() : nullptr;
  SparseMatrix<Number> & matrix = system.get_system_matrix();

  std::unique_ptr<FEBase> fe_face(FEBase::build(mesh.mesh_dimension(), fe_type));
  QGauss qface(mesh.mesh_dimension() - 1, fe_type.default_quadrature_order());
  fe_face->attach_quadrature_rule(&qface);
  const auto & jxw_face = fe_face->get_JxW();
  const auto & psi = fe_face->get_phi();
  const auto & qface_points = fe_face->get_xyz();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;
  constexpr Real penalty = 1.e10;

  for (unsigned int e = 0; e != n_elements; ++e)
  {
    const Elem * elem = records[e].elem;
    auto & dof_indices = records[e].dof_indices;
    const unsigned int elem_n_dofs =
        cast_int<unsigned int>(dof_indices.size());
    Ke.resize(elem_n_dofs, elem_n_dofs);
    Fe.resize(elem_n_dofs);

    for (unsigned int i = 0; i != elem_n_dofs; ++i)
    {
      Fe(i) = h_element_rhs(e, i);
      for (unsigned int j = 0; j != elem_n_dofs; ++j)
        Ke(i, j) = h_element_matrices(e, i, j);
    }

    for (const auto side : elem->side_index_range())
      if (elem->neighbor_ptr(side) == nullptr)
      {
        fe_face->reinit(elem, side);
        libmesh_assert_equal_to(elem_n_dofs, psi.size());

        for (unsigned int qp = 0; qp != qface.n_points(); ++qp)
        {
          const Number value = adaptivity_ex2_kokkos_exact_solution(
              qface_points[qp](0), qface_points[qp](1), system.time);
          for (unsigned int i = 0; i != elem_n_dofs; ++i)
          {
            Fe(i) += penalty * jxw_face[qp] * value * psi[i][qp];
            for (unsigned int j = 0; j != elem_n_dofs; ++j)
              Ke(i, j) += penalty * jxw_face[qp] * psi[i][qp] * psi[j][qp];
          }
        }
      }

    dof_map.constrain_element_matrix_and_vector(Ke, Fe, dof_indices);
    if (static_condensation)
      static_condensation->set_current_elem(*elem);
    matrix.add_matrix(Ke, dof_indices);
    system.rhs->add_vector(Fe, dof_indices);
  }
}
#endif // LIBMESH_ENABLE_AMR

#endif // LIBMESH_HAVE_KOKKOS && !LIBMESH_USE_COMPLEX_NUMBERS
