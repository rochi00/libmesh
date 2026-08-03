// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// Kokkos volume assembly for the standalone adaptivity_ex2 Kokkos driver.

#include "libmesh/libmesh_config.h"

#if defined(LIBMESH_HAVE_KOKKOS) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)

#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dof_map.h"
#include "libmesh/dof_map_kokkos.h"
#include "libmesh/elem.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe_reference_element_traits.h"
#include "libmesh/fe_shape_traits.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/mesh_base.h"
#include "libmesh/mesh_base_kokkos.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/petsc_matrix_base.h"
#include "libmesh/petsc_vector.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/static_condensation.h"
#include "libmesh/transient_system.h"

#include "gpu/kokkos_fe_evaluator.h"
#include "gpu/kokkos_fe_map.h"
#include "gpu/kokkos_quadrature.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <memory>
#include <vector>

using namespace libMesh;

#ifdef LIBMESH_ENABLE_AMR
namespace
{

constexpr unsigned int kokkos_max_dofs = 9;
constexpr unsigned int kokkos_max_nodes = 9;

using RealView2D = KokkosConstraintAssemblyPlan::element_real_view;

struct AdaptivityKokkosAssemblyCache
{
  const void * geometry_data = nullptr;
  KokkosConstraintAssemblyPlan plan;
  bool matrix_values_valid = false;
  Real matrix_dt = 0.;
  Real matrix_diffusivity = 0.;
  Real matrix_velocity_x = 0.;
  Real matrix_velocity_y = 0.;
};

std::unique_ptr<AdaptivityKokkosAssemblyCache> adaptivity_kokkos_assembly_cache;

template <typename CoordinateView, typename ElementNodeView> struct ElementNodeAccess
{
  CoordinateView coordinates;
  ElementNodeView element_nodes;
  unsigned int elem = 0;

  KOKKOS_INLINE_FUNCTION Real operator()(const unsigned int node,
                                         const unsigned int component) const
  {
    return coordinates(element_nodes(elem, node), component);
  }
};

template <typename CoordinateView, typename ElementNodeView, typename SideLocalNodeView>
struct ElementSideNodeAccess
{
  CoordinateView coordinates;
  ElementNodeView element_nodes;
  SideLocalNodeView side_local_nodes;
  unsigned int elem = 0;
  unsigned int side = 0;

  KOKKOS_INLINE_FUNCTION Real operator()(const unsigned int node,
                                         const unsigned int component) const
  {
    return coordinates(element_nodes(elem, side_local_nodes(elem, side, node)), component);
  }
};

KOKKOS_INLINE_FUNCTION Real kokkos_exact_solution(const Real x, const Real y, const Real time)
{
  constexpr Real x0 = 0.2;
  constexpr Real y0 = 0.2;
  constexpr Real velocity = 0.8;
  const Real scale = 4. * time + 1.;
  const Real dx = x - velocity * time - x0;
  const Real dy = y - velocity * time - y0;
  return ::Kokkos::exp(-(dx * dx + dy * dy) / (0.01 * scale)) / scale;
}

bool supported_element(const ElemType elem_type, const ElemMappingType mapping_type,
                       const unsigned int n_nodes, const unsigned int p_level,
                       const FEType & fe_type, const unsigned int n_dofs,
                       const unsigned int quadrature_order)
{
  const int effective_order = fe_type.order.get_order() + cast_int<int>(p_level);
  if (effective_order < 1 || effective_order > 2 || fe_type.family != LAGRANGE ||
      mapping_type != LAGRANGE_MAP || topology_dim_or_zero(elem_type) != 2 ||
      n_nodes > kokkos_max_nodes ||
      n_dofs > kokkos_max_dofs)
    return false;

  switch (elem_type)
  {
    case TRI3:
    case TRI6:
    case QUAD4:
    case QUAD9:
      break;
    default:
      return false;
  }

  const FEShapeKey key{fe_type.family, elem_type, static_cast<Order>(effective_order)};
  return supports_shape_with_lagrange_map(key) && supports_grad_shape(key) &&
         libMesh::n_dofs_or_zero(key) == n_dofs &&
         libMesh::Kokkos::GaussQuadrature::n_points(elem_type, quadrature_order) != 0;
}

} // anonymous namespace

void clear_adaptivity_ex2_kokkos_assembly_cache()
{
  adaptivity_kokkos_assembly_cache.reset();
}

void assemble_cd_kokkos(EquationSystems & es, const std::string & libmesh_dbg_var(system_name))
{
  libmesh_assert_equal_to(system_name, "Convection-Diffusion");

  ::Kokkos::Profiling::ScopedRegion region("adaptivity_ex2 Kokkos assembly");

  const MeshBase & mesh = es.get_mesh();
  auto & system = es.get_system<TransientLinearImplicitSystem>("Convection-Diffusion");
  const DofMap & dof_map = system.get_dof_map();
  const FEType fe_type = system.variable_type(0);
  const unsigned int quadrature_order = cast_int<unsigned int>(fe_type.default_quadrature_order());

  // These caches persist until mesh or DoF metadata changes.  Reusing them here
  // avoids rebuilding and copying element geometry and DoF tables on every
  // transient assembly.
  const auto & geometry_cache = mesh.get_kokkos_geometry_cache();
  const auto * dof_index_cache = dof_map.get_kokkos_dof_index_cache(0);
  libmesh_error_msg_if(!dof_index_cache, "adaptivity_ex2 requires a Kokkos DoF-index cache");

  libmesh_assert(system.old_local_solution);
  const auto * old_local_index_cache =
      dof_map.require_kokkos_local_index_cache(*system.old_local_solution, 0);
  auto * old_petsc_vector = dynamic_cast<PetscVector<Number> *>(system.old_local_solution);
  libmesh_error_msg_if(!old_local_index_cache || !old_petsc_vector ||
                           !old_petsc_vector->supports_kokkos_access(),
                       "adaptivity_ex2 Kokkos assembly requires a PETSc old-local-solution "
                       "vector accessible from the active Kokkos execution space");

  const unsigned int n_elements = cast_int<unsigned int>(geometry_cache.host_element_ids.size());
  const unsigned int max_element_dofs = dof_index_cache->max_dofs;
  libmesh_assert_less_equal(max_element_dofs, kokkos_max_dofs);
  libmesh_assert_equal_to(n_elements, dof_index_cache->host_element_ids.size());
  libmesh_assert_equal_to(n_elements, dof_index_cache->host_element_n_dofs.size());

  for (unsigned int e = 0; e != n_elements; ++e)
  {
    const ElemType elem_type = geometry_cache.host_element_types[e];
    const ElemMappingType mapping_type = geometry_cache.host_element_mapping_types[e];
    const unsigned int n_nodes = geometry_cache.host_element_n_nodes[e];
    const unsigned int p_level = geometry_cache.host_element_p_levels[e];
    const unsigned int elem_n_dofs = dof_index_cache->host_element_n_dofs[e];
    libmesh_error_msg_if(!supported_element(elem_type, mapping_type, n_nodes, p_level, fe_type,
                                            elem_n_dofs, quadrature_order),
                         "Unsupported element in adaptivity_ex2 Kokkos order-1/2 assembly: type="
                             << Utility::enum_to_string(elem_type) << ", p_level=" << p_level
                             << ", n_dofs=" << elem_n_dofs);
  }

  StaticCondensation * static_condensation =
      system.has_static_condensation() ? &system.get_static_condensation() : nullptr;
  SparseMatrix<Number> & matrix = system.get_system_matrix();
  auto * petsc_matrix = dynamic_cast<PetscMatrixBase<Number> *>(&matrix);
  auto * petsc_rhs = dynamic_cast<PetscVector<Number> *>(system.rhs);
  const bool use_petsc_coo = petsc_matrix && petsc_rhs && !static_condensation;

  AdaptivityKokkosAssemblyCache * assembly_cache = nullptr;
  KokkosConstraintAssemblyPlan * assembly_plan = nullptr;
  if (use_petsc_coo)
  {
    if (!adaptivity_kokkos_assembly_cache)
      adaptivity_kokkos_assembly_cache = std::make_unique<AdaptivityKokkosAssemblyCache>();
    assembly_cache = adaptivity_kokkos_assembly_cache.get();
    if (!kokkos_constraint_assembly_plan_matches(assembly_cache->plan, *dof_index_cache))
    {
      build_kokkos_constraint_assembly_plan(dof_map, *dof_index_cache, assembly_cache->plan);
      assembly_cache->matrix_values_valid = false;
    }
    if (assembly_cache->geometry_data != geometry_cache.node_coordinates.data())
    {
      assembly_cache->geometry_data = geometry_cache.node_coordinates.data();
      assembly_cache->matrix_values_valid = false;
    }
    assembly_plan = &assembly_cache->plan;
    prepare_kokkos_petsc_coo(system.comm(), *assembly_plan, *petsc_matrix, *petsc_rhs);
  }

  // The shared constraint assembly plan owns and reuses raw element storage.
  // The host/static-condensation fallback gets the same layout for this call.
  const unsigned int matrix_stride = max_element_dofs;
  RealView2D element_matrices =
      use_petsc_coo
          ? assembly_plan->element_matrices
          : RealView2D("adaptivity_ex2 element matrices", n_elements,
                       matrix_stride * matrix_stride);
  RealView2D element_rhs =
      use_petsc_coo ? assembly_plan->element_rhs
                    : RealView2D("adaptivity_ex2 element RHS", n_elements, max_element_dofs);

  const auto node_coordinates = geometry_cache.node_coordinates;
  const auto element_node_ids = geometry_cache.element_node_ids;
  const auto elem_types = geometry_cache.element_types;
  const auto mapping_types = geometry_cache.element_mapping_types;
  const auto n_nodes = geometry_cache.element_n_nodes;
  const auto n_sides = geometry_cache.element_n_sides;
  const auto side_types = geometry_cache.element_side_types;
  const auto side_n_nodes = geometry_cache.element_side_n_nodes;
  const auto side_local_nodes = geometry_cache.element_side_local_nodes;
  const auto side_is_boundary = geometry_cache.element_side_is_boundary;
  const auto p_levels = geometry_cache.element_p_levels;
  const auto n_dofs = dof_index_cache->element_n_dofs;
  const auto old_local_indices = old_local_index_cache->element_local_indices;

  const RealVectorValue velocity = es.parameters.get<RealVectorValue>("velocity");
  const Real velocity_x = velocity(0);
  const Real velocity_y = velocity(1);
  const Real diffusivity = es.parameters.get<Real>("diffusivity");
  const Real dt = es.parameters.get<Real>("dt");
  const Real time = system.time;
  const int base_order = fe_type.order.get_order();
  constexpr Real penalty = 1.e10;
  const bool assemble_matrix =
      !use_petsc_coo || !assembly_cache->matrix_values_valid || assembly_cache->matrix_dt != dt ||
      assembly_cache->matrix_diffusivity != diffusivity ||
      assembly_cache->matrix_velocity_x != velocity_x ||
      assembly_cache->matrix_velocity_y != velocity_y;

  using ExecutionSpace = typename RealView2D::execution_space;
  {
    // Keep PETSc's borrowed array alive until all kernels reading it finish.
    auto old_guard = old_petsc_vector->make_kokkos_read_view_guard();
    const auto old_values = old_guard.view();
    ::Kokkos::parallel_for(
        "adaptivity_ex2 volume and boundary integration",
        ::Kokkos::RangePolicy<ExecutionSpace>(0, cast_int<int>(n_elements)),
        KOKKOS_LAMBDA(const int raw_elem) {
          const unsigned int e = static_cast<unsigned int>(raw_elem);
          const unsigned int elem_n_dofs = n_dofs(e);
          const ElemType elem_type = elem_types(e);
          const FEShapeKey shape_key{LAGRANGE, elem_type,
                                     static_cast<Order>(base_order + p_levels(e))};
          const ElemType shape_topology =
              libMesh::Kokkos::lagrange_shape_topology_for_key(shape_key);
          const ElementNodeAccess<decltype(node_coordinates), decltype(element_node_ids)>
              elem_nodes{node_coordinates, element_node_ids, e};
          const unsigned int n_qpoints =
              libMesh::Kokkos::GaussQuadrature::n_points(elem_type, quadrature_order);

          // The transient state is constant throughout this element integration.
          // Gather it once instead of reloading the PETSc view at every quadrature
          // point.  Initialize the output here as well, folding two whole-buffer
          // zeroing kernels into the integration launch.
          Real element_old_values[kokkos_max_dofs];
          Real local_rhs[kokkos_max_dofs];
          for (unsigned int i = 0; i != elem_n_dofs; ++i)
          {
            element_old_values[i] = old_values(old_local_indices(e, i));
            local_rhs[i] = 0.;
            if (assemble_matrix)
              for (unsigned int j = 0; j != elem_n_dofs; ++j)
                element_matrices(e, i * matrix_stride + j) = 0.;
          }

          for (unsigned int qp = 0; qp != n_qpoints; ++qp)
          {
            const auto reference_point =
                libMesh::Kokkos::GaussQuadrature::point(elem_type, quadrature_order, qp);
            auto xyz = libMesh::Kokkos::zero_vector();
            auto jacobian = libMesh::Kokkos::zero_tensor();
            libMesh::Kokkos::physical_point_and_jacobian(
                mapping_types(e), elem_type, elem_nodes, n_nodes(e), reference_point(0),
                reference_point(1), reference_point(2), xyz, jacobian);
            const auto inverse_jacobian =
                libMesh::Kokkos::inverse<libMesh::Kokkos::RealTensor>(jacobian, 2);
            const Real jxw = libMesh::Kokkos::volume_jxw(
                jacobian, 2,
                libMesh::Kokkos::GaussQuadrature::weight(elem_type, quadrature_order, qp));

            Real phi[kokkos_max_dofs];
            libMesh::Kokkos::RealVector dphi[kokkos_max_dofs];
            Real u_old = 0.;
            auto grad_u_old = libMesh::Kokkos::zero_vector();

            for (unsigned int i = 0; i != elem_n_dofs; ++i)
            {
              phi[i] = libMesh::Kokkos::eval_lagrange_shape(
                  shape_topology, i, reference_point(0), reference_point(1), reference_point(2));
              dphi[i] = inverse_jacobian *
                        libMesh::Kokkos::eval_lagrange_grad_shape(
                            shape_topology, i, reference_point(0), reference_point(1),
                            reference_point(2));
              const Real old_value = element_old_values[i];
              u_old += phi[i] * old_value;
              grad_u_old.add_scaled(dphi[i], old_value);
            }

            const Real old_convection = velocity_x * grad_u_old(0) + velocity_y * grad_u_old(1);
            for (unsigned int i = 0; i != elem_n_dofs; ++i)
            {
              const Real old_diffusion =
                  diffusivity * (grad_u_old(0) * dphi[i](0) + grad_u_old(1) * dphi[i](1));
              local_rhs[i] +=
                  jxw * (u_old * phi[i] - 0.5 * dt * (old_convection * phi[i] + old_diffusion));

              if (assemble_matrix)
                for (unsigned int j = 0; j != elem_n_dofs; ++j)
                {
                  const Real convection =
                      (velocity_x * dphi[j](0) + velocity_y * dphi[j](1)) * phi[i];
                  const Real diffusion =
                      diffusivity * (dphi[i](0) * dphi[j](0) + dphi[i](1) * dphi[j](1));
                  element_matrices(e, i * matrix_stride + j) +=
                      jxw * (phi[i] * phi[j] + 0.5 * dt * (convection + diffusion));
                }
            }
          }

          // The penalty boundary condition uses the same device-resident mesh
          // cache, including side topology and the physical-boundary mask.
          for (unsigned int side = 0; side != n_sides(e); ++side)
          {
            if (!side_is_boundary(e, side))
              continue;

            const ElemType side_type = side_types(e, side);
            const unsigned int elem_side_n_nodes = side_n_nodes(e, side);
            const unsigned int n_side_qpoints =
                libMesh::Kokkos::GaussQuadrature::n_points(side_type, quadrature_order);
            const ElementSideNodeAccess<decltype(node_coordinates), decltype(element_node_ids),
                                        decltype(side_local_nodes)>
                face_nodes{node_coordinates, element_node_ids, side_local_nodes, e, side};

            for (unsigned int qp = 0; qp != n_side_qpoints; ++qp)
            {
              const auto face_point =
                  libMesh::Kokkos::GaussQuadrature::point(side_type, quadrature_order, qp);
              auto parent_point = libMesh::Kokkos::zero_vector();
              for (unsigned int k = 0; k != elem_side_n_nodes; ++k)
              {
                Point reference_node;
                if (!libMesh::try_reference_side_node(elem_type, side, k, reference_node))
                  ::Kokkos::abort("adaptivity_ex2 unsupported reference side node");
                const Real side_shape = libMesh::Kokkos::map_shape(
                    mapping_types(e), side_type, k, face_point(0), face_point(1), face_point(2));
                for (unsigned int d = 0; d != LIBMESH_DIM; ++d)
                  parent_point(d) += side_shape * reference_node(d);
              }

              const auto xyz = libMesh::Kokkos::physical_point(
                  mapping_types(e), side_type, face_nodes, elem_side_n_nodes, face_point(0),
                  face_point(1), face_point(2));
              const auto face_jacobian = libMesh::Kokkos::face_jacobian(
                  mapping_types(e), side_type, face_nodes, elem_side_n_nodes, face_point(0),
                  face_point(1), face_point(2));
              const Real face_jxw = libMesh::Kokkos::face_jxw(
                  face_jacobian, 2,
                  libMesh::Kokkos::GaussQuadrature::weight(side_type, quadrature_order, qp));
              const Real boundary_value = kokkos_exact_solution(xyz(0), xyz(1), time);

              Real face_phi[kokkos_max_dofs];
              for (unsigned int i = 0; i != elem_n_dofs; ++i)
                face_phi[i] = libMesh::Kokkos::eval_lagrange_shape(
                    shape_topology, i, parent_point(0), parent_point(1), parent_point(2));

              for (unsigned int i = 0; i != elem_n_dofs; ++i)
              {
                local_rhs[i] += penalty * face_jxw * boundary_value * face_phi[i];
                if (assemble_matrix)
                  for (unsigned int j = 0; j != elem_n_dofs; ++j)
                    element_matrices(e, i * matrix_stride + j) +=
                        penalty * face_jxw * face_phi[i] * face_phi[j];
              }
            }
          }

          for (unsigned int i = 0; i != elem_n_dofs; ++i)
            element_rhs(e, i) = local_rhs[i];
        });
    ::Kokkos::fence();
  }

  if (use_petsc_coo)
  {
    transform_kokkos_element_values(*assembly_plan, *dof_index_cache, assemble_matrix);
    assembly_cache->matrix_values_valid = true;
    assembly_cache->matrix_dt = dt;
    assembly_cache->matrix_diffusivity = diffusivity;
    assembly_cache->matrix_velocity_x = velocity_x;
    assembly_cache->matrix_velocity_y = velocity_y;
    add_kokkos_petsc_coo_values(
        system.comm(), *assembly_plan, *petsc_matrix, *petsc_rhs);
    return;
  }

  auto h_element_matrices = ::Kokkos::create_mirror_view(element_matrices);
  auto h_element_rhs = ::Kokkos::create_mirror_view(element_rhs);
  {
    ::Kokkos::Profiling::ScopedRegion copy_region("adaptivity_ex2 device-to-host element data");
    ::Kokkos::deep_copy(h_element_matrices, element_matrices);
    ::Kokkos::deep_copy(h_element_rhs, element_rhs);
  }

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;
  std::vector<dof_id_type> dof_indices;

  {
    ::Kokkos::Profiling::ScopedRegion host_region(
        "adaptivity_ex2 host constraints and insertion");
    unsigned int e = 0;
    for (const Elem * elem : mesh.active_local_element_ptr_range())
    {
      libmesh_assert(elem);
      libmesh_assert_less(e, n_elements);
      libmesh_assert_equal_to(elem->id(), geometry_cache.host_element_ids[e]);
      const unsigned int elem_n_dofs = dof_index_cache->host_element_n_dofs[e];
      const auto dof_begin =
          dof_index_cache->host_element_dof_indices.begin() + e * dof_index_cache->max_dofs;
      dof_indices.assign(dof_begin, dof_begin + elem_n_dofs);
      Ke.resize(elem_n_dofs, elem_n_dofs);
      Fe.resize(elem_n_dofs);

      for (unsigned int i = 0; i != elem_n_dofs; ++i)
      {
        Fe(i) = h_element_rhs(e, i);
        for (unsigned int j = 0; j != elem_n_dofs; ++j)
          Ke(i, j) = h_element_matrices(e, i * matrix_stride + j);
      }

      dof_map.constrain_element_matrix_and_vector(Ke, Fe, dof_indices);
      if (static_condensation)
        static_condensation->set_current_elem(*elem);
      matrix.add_matrix(Ke, dof_indices);
      system.rhs->add_vector(Fe, dof_indices);
      ++e;
    }
    libmesh_assert_equal_to(e, n_elements);
  }
}
#endif // LIBMESH_ENABLE_AMR

#endif // LIBMESH_HAVE_KOKKOS && !LIBMESH_USE_COMPLEX_NUMBERS
