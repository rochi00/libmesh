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

#include "L2system.h"

#include "../../include/systems/hilbert_assembly.h"

#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fem_context.h"
#include "libmesh/linear_solver.h"
#include "libmesh/libmesh.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/mesh.h"
#include "libmesh/parsed_fem_function.h"
#include "libmesh/parsed_function.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef LIBMESH_HAVE_PETSC
#include "libmesh/petsc_matrix_base.h"
#include "libmesh/petsc_macro.h"
#include "libmesh/petsc_vector.h"
#endif

#if defined(LIBMESH_HAVE_KOKKOS) && defined(LIBMESH_HAVE_PETSC) &&                                 \
    !defined(LIBMESH_USE_COMPLEX_NUMBERS)
#include "../../include/gpu/kokkos_hilbert_system.h"
#include "libmesh/dof_map_kokkos.h"
#include "libmesh/fe_shape_traits.h"
#include "libmesh/mesh_base_kokkos.h"

#define PETSC_SKIP_CXX_COMPLEX_FIX 1
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#ifdef LIBMESH_HAVE_NVTX_API
#include <nvtx3/nvtx3.hpp>
#endif
#undef __CUDACC_VER__
#endif

#if defined(LIBMESH_HAVE_KOKKOS) && defined(LIBMESH_HAVE_PETSC) &&                                 \
    !defined(LIBMESH_USE_COMPLEX_NUMBERS)
namespace libMesh
{
constexpr unsigned int kokkos_hilbert_max_dofs = 27;
constexpr unsigned int kokkos_parsed_fem_max_fields = 16;

template <typename T>
using KokkosScalarView = ::Kokkos::View<T *>;

using KokkosDenseJacobianView = ::Kokkos::View<Number **>;
using KokkosUnsignedIntView = ::Kokkos::View<unsigned int *>;
using KokkosSizeView = ::Kokkos::View<std::size_t *>;
using KokkosShapeKeyRecordView = ::Kokkos::View<FEShapeKey *>;
using KokkosFieldKeyRecordView = ::Kokkos::View<FEShapeKey **>;
using KokkosFieldDofRecordView = ::Kokkos::View<unsigned int **>;
using KokkosFieldKeyStorage =
    libMesh::Kokkos::detail::StaticArrayAccess<FEShapeKey, kokkos_parsed_fem_max_fields>;
using KokkosFieldDofStorage =
    libMesh::Kokkos::detail::StaticArrayAccess<unsigned int, kokkos_parsed_fem_max_fields>;
using KokkosLocalIndexView = DofMap::KokkosLocalIndexCache::elem_local_index_view;

struct HilbertElementAssemblyRecord
{
  dof_id_type elem_id = DofObject::invalid_id;
  std::size_t rhs_offset = 0;
  std::size_t mat_offset = 0;
  unsigned int elem_index = libMesh::invalid_uint;
  unsigned int quadrature_order = 0;
  unsigned int n_dofs = 0;
};

FEShapeKey
make_hilbert_shape_key(const ElemType elem_type,
                       const unsigned int elem_p_level,
                       const FEType & fe_type)
{
  return {fe_type.family,
          elem_type,
          static_cast<Order>(fe_type.order.get_order() + cast_int<int>(elem_p_level))};
}

std::uint32_t
kokkos_region_color(const char * name)
{
  if (!name)
    return 0xff8f8f8f;
  if (std::strstr(name, "PETSc"))
    return 0xffe15759;
  if (std::strstr(name, "solve") || std::strstr(name, "KSPSolve"))
    return 0xffedc948;
  if (std::strstr(name, "COO"))
    return 0xfff28e2b;
  if (std::strstr(name, "analytic"))
    return 0xffb07aa1;
  if (std::strstr(name, "assemble") || std::strstr(name, "scatter") ||
      std::strstr(name, "entry values"))
    return 0xff4e79a7;
  if (std::strstr(name, "plan") || std::strstr(name, "batch"))
    return 0xff59a14f;
  if (std::strstr(name, "zero") || std::strstr(name, "close") || std::strstr(name, "cleanup"))
    return 0xffbab0ac;
  return 0xff9c755f;
}

struct KokkosProfilingRegion
{
  explicit KokkosProfilingRegion(const char * name)
  {
    ::Kokkos::Profiling::pushRegion(name);
#ifdef LIBMESH_HAVE_NVTX_API
    nvtxEventAttributes_t event_attributes = {};
    event_attributes.version = NVTX_VERSION;
    event_attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event_attributes.colorType = NVTX_COLOR_ARGB;
    event_attributes.color = kokkos_region_color(name);
    event_attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event_attributes.message.ascii = name;
    nvtxRangePushEx(&event_attributes);
#endif
  }

  ~KokkosProfilingRegion()
  {
#ifdef LIBMESH_HAVE_NVTX_API
    nvtxRangePop();
#endif
    ::Kokkos::Profiling::popRegion();
  }

  KokkosProfilingRegion(const KokkosProfilingRegion &) = delete;
  KokkosProfilingRegion & operator=(const KokkosProfilingRegion &) = delete;
};

bool
kokkos_l2_env_enabled(const char * name)
{
  const char * value = std::getenv(name);
  return value && std::strcmp(value, "0") && std::strcmp(value, "false") &&
         std::strcmp(value, "FALSE") && std::strcmp(value, "off") &&
         std::strcmp(value, "OFF");
}

template <typename View>
void
debug_kokkos_entry_values(const Parallel::Communicator & comm,
                          const char * name,
                          const View & values)
{
  using ExecutionSpace = typename View::execution_space;
  using RangePolicy = ::Kokkos::RangePolicy<ExecutionSpace, ::Kokkos::IndexType<std::size_t>>;

  unsigned long long bad_count = 0;
  double max_abs = 0.;

  ::Kokkos::parallel_reduce(
      RangePolicy(0, values.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, unsigned long long & local_bad) {
        const double value = static_cast<double>(values(i));
        if (!(value == value) || value > 1.e300 || value < -1.e300)
          ++local_bad;
      },
      bad_count);

  ::Kokkos::parallel_reduce(
      RangePolicy(0, values.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, double & local_max) {
        const double value = static_cast<double>(values(i));
        if (value == value)
        {
          const double abs_value = value < 0. ? -value : value;
          if (abs_value > local_max)
            local_max = abs_value;
        }
      },
      ::Kokkos::Max<double>(max_abs));
  ::Kokkos::fence();

  comm.sum(bad_count);
  comm.max(max_abs);

  if (comm.rank() == 0)
    libMesh::out << "Kokkos L2 debug " << name << ": n=" << values.extent(0)
                 << " bad=" << bad_count << " max_abs=" << max_abs << std::endl;
}

template <typename Scalar>
void
debug_parsed_program(const Parallel::Communicator & comm,
                     const char * name,
                     const ParsedFunctionProgram<Scalar> & program)
{
  if (comm.rank() != 0)
    return;

  libMesh::out << "Kokkos L2 debug parsed " << name
               << ": bytecode_n=" << program.bytecode.size()
               << " immediates_n=" << program.immediates.size()
               << " stack_size=" << program.stack_size
               << " n_variables=" << program.n_variables;

  libMesh::out << " bytecode=[";
  for (std::size_t i = 0; i != program.bytecode.size() && i != 12; ++i)
    libMesh::out << (i ? "," : "") << program.bytecode[i];
  if (program.bytecode.size() > 12)
    libMesh::out << ",...";
  libMesh::out << "]";

  libMesh::out << " immediates=[";
  for (std::size_t i = 0; i != program.immediates.size() && i != 12; ++i)
    libMesh::out << (i ? "," : "") << program.immediates[i];
  if (program.immediates.size() > 12)
    libMesh::out << ",...";
  libMesh::out << "]" << std::endl;
}

#ifdef LIBMESH_HAVE_PETSC
void
debug_petsc_object_types(const Parallel::Communicator & comm, Vec vec, Mat mat)
{
  const char * vec_type = nullptr;
  const char * mat_type = nullptr;
  LibmeshPetscCall2(comm, VecGetType(vec, &vec_type));
  LibmeshPetscCall2(comm, MatGetType(mat, &mat_type));

  if (comm.rank() == 0)
    libMesh::out << "Kokkos L2 debug PETSc types: vec=" << vec_type
                 << " mat=" << mat_type << std::endl;
}

void
debug_petsc_object_norms(const Parallel::Communicator & comm, Vec vec, Mat mat)
{
  PetscReal vec_norm = 0.;
  PetscReal mat_norm = 0.;
  LibmeshPetscCall2(comm, VecNorm(vec, NORM_INFINITY, &vec_norm));
  LibmeshPetscCall2(comm, MatNorm(mat, NORM_INFINITY, &mat_norm));

  if (comm.rank() == 0)
    libMesh::out << "Kokkos L2 debug PETSc norms: rhs_inf=" << vec_norm
                 << " mat_inf=" << mat_norm << std::endl;
}
#endif

void
accumulate_hilbert_dense_outputs(const KokkosScalarView<Number> & d_F,
                                 const KokkosDenseJacobianView & d_K,
                                 const bool request_jacobian,
                                 DenseSubVector<Number> & F,
                                 DenseSubMatrix<Number> & K)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::accumulate dense element outputs");

  auto h_F = ::Kokkos::create_mirror_view(d_F);
  ::Kokkos::deep_copy(h_F, d_F);

  for (unsigned int i = 0; i != F.size(); ++i)
    F(i) += h_F(i);

  if (!request_jacobian)
    return;

  auto h_K = ::Kokkos::create_mirror_view(d_K);
  ::Kokkos::deep_copy(h_K, d_K);

  for (unsigned int i = 0; i != F.size(); ++i)
    for (unsigned int j = 0; j != F.size(); ++j)
      K(i, j) += h_K(i, j);
}

void
build_hilbert_element_records(::HilbertSystem & sys,
                              std::vector<HilbertElementAssemblyRecord> & records,
                              std::size_t & total_rhs_entries,
                              std::size_t & total_mat_entries)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::build element records");

  DofMap & dof_map = sys.get_dof_map();
  const auto * dof_index_cache = dof_map.get_kokkos_dof_index_cache(0);
  libmesh_assert(dof_index_cache);
  total_rhs_entries = 0;
  total_mat_entries = 0;

  for (auto elem_index : index_range(dof_index_cache->host_element_ids))
  {
    if (!sys.subdomains_list().empty() &&
        !sys.subdomains_list().count(dof_index_cache->host_element_subdomains[elem_index]))
      continue;

    const unsigned int n_dofs = dof_index_cache->host_element_n_dofs[elem_index];
    if (!n_dofs)
      continue;

    HilbertElementAssemblyRecord record;
    record.elem_id = dof_index_cache->host_element_ids[elem_index];
    record.rhs_offset = total_rhs_entries;
    record.mat_offset = total_mat_entries;
    record.elem_index = cast_int<unsigned int>(elem_index);
    record.n_dofs = n_dofs;
    total_rhs_entries += n_dofs;
    total_mat_entries += n_dofs * n_dofs;
    records.push_back(std::move(record));
  }
}

bool
build_hilbert_record_quadrature_orders(::HilbertSystem & sys,
                                       std::vector<HilbertElementAssemblyRecord> & records)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::build record quadrature orders");

  const auto & geometry_cache = sys.get_mesh().get_kokkos_geometry_cache();
  const FEType fe_type = sys.variable_type(0);
  const int quadrature_order =
      cast_int<int>(fe_type.default_quadrature_order()) + sys.extra_quadrature_order;

  for (auto & record : records)
  {
    libmesh_error_msg_if(quadrature_order < 0,
                         "Negative quadrature order is not supported for Kokkos Hilbert assembly");
    record.quadrature_order = cast_int<unsigned int>(quadrature_order);

    const FEShapeKey shape_key =
        make_hilbert_shape_key(geometry_cache.host_element_types[record.elem_index],
                               geometry_cache.host_element_p_levels[record.elem_index],
                               fe_type);
    if (!libMesh::Kokkos::detail::supports_hilbert_local_assembly(
            shape_key,
            geometry_cache.host_element_mapping_types[record.elem_index],
            record.quadrature_order) ||
        record.n_dofs > kokkos_hilbert_max_dofs)
      return false;
  }

  return true;
}

template <typename ParsedGoal, typename CachedGoal, typename HostGoalPtr, typename BuildCache>
const CachedGoal *
ensure_kokkos_goal_cache(std::unique_ptr<CachedGoal> & cache,
                         const HostGoalPtr & host_goal,
                         BuildCache && build_cache)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::ensure parsed goal cache");

  if (cache)
    return cache.get();

  if (!host_goal)
    return nullptr;

  const auto * parsed_goal = dynamic_cast<const ParsedGoal *>(host_goal.get());
  if (!parsed_goal)
    return nullptr;

  cache = build_cache(*parsed_goal);
  return cache.get();
}

class HostExactParsedFEMGoalAccess
{
public:
  HostExactParsedFEMGoalAccess(const libMesh::Kokkos::KokkosParsedFEMFunction<Number> & goal,
                               FEMContext & input_context)
    : _goal(goal), _input_context(input_context)
  {
  }

  template <typename QpData>
  Number value(const QpData & qp_data, const Point & xyz) const
  {
    Number vars[LIBMESH_DIM + 1 + kokkos_parsed_fem_max_fields] = {};
    fill_variables(qp_data, xyz, vars);
    return _goal.value(vars);
  }

  template <typename QpData>
  Gradient gradient(const QpData & qp_data, const Point & xyz) const
  {
    Number vars[LIBMESH_DIM + 1 + kokkos_parsed_fem_max_fields] = {};
    Gradient field_gradients[kokkos_parsed_fem_max_fields];
    fill_variables(qp_data, xyz, vars);

    for (unsigned int field = 0; field != _goal.n_field_variables(); ++field)
      field_gradients[field] =
          _input_context.interior_gradient(_goal.field_variable_number(field), qp_data.qp_index());

    return _goal.gradient(vars, field_gradients);
  }

private:
  template <typename QpData>
  void fill_variables(const QpData & qp_data, const Point & xyz, Number * vars) const
  {
    vars[0] = xyz(0);
#if LIBMESH_DIM > 1
    vars[1] = xyz(1);
#endif
#if LIBMESH_DIM > 2
    vars[2] = xyz(2);
#endif
    vars[LIBMESH_DIM] = _goal.time();

    for (unsigned int field = 0; field != _goal.n_field_variables(); ++field)
      vars[LIBMESH_DIM + 1 + field] =
          _input_context.interior_value(_goal.field_variable_number(field), qp_data.qp_index());
  }

  const libMesh::Kokkos::KokkosParsedFEMFunction<Number> & _goal;
  FEMContext & _input_context;
};

struct KokkosElementAssemblyState
{
  const MeshBase::KokkosGeometryCache * geometry_cache = nullptr;
  const DofMap::KokkosLocalIndexCache * solution_local_indices = nullptr;
  PetscVector<Number> * solution_vector = nullptr;
  unsigned int elem_index = libMesh::invalid_uint;
  ElemType elem_type = INVALID_ELEM;
  ElemMappingType mapping_type = LAGRANGE_MAP;
  unsigned int elem_n_nodes = 0;
  unsigned int elem_p_level = 0;
};

struct KokkosFEMGoalData
{
  KokkosFieldKeyStorage field_keys;
  KokkosFieldDofStorage field_dofs;
  std::array<KokkosLocalIndexView, kokkos_parsed_fem_max_fields> field_local_indices;
  PetscVector<Number> * input_vector = nullptr;
};

struct KokkosHilbertBatchData
{
  KokkosUnsignedIntView elem_indices;
  KokkosUnsignedIntView elem_n_dofs;
  KokkosUnsignedIntView quadrature_orders;
  KokkosSizeView rhs_offsets;
  KokkosSizeView mat_offsets;
  KokkosShapeKeyRecordView shape_keys;
};

struct KokkosAnalyticGoalBatchData
{
  KokkosSizeView qp_offsets;
  KokkosScalarView<Number> values;
  KokkosScalarView<Number> gradients;
};

struct KokkosFEMGoalBatchData
{
  KokkosFieldKeyRecordView field_keys;
  KokkosFieldDofRecordView field_dofs;
  libMesh::Kokkos::detail::StaticArrayAccess<KokkosLocalIndexView, kokkos_parsed_fem_max_fields>
      field_local_indices;
  PetscVector<Number> * input_vector = nullptr;
};

struct KokkosHilbertHomogeneousBucket
{
  FEShapeKey key{INVALID_FE, INVALID_ELEM, INVALID_ORDER};
  ElemMappingType mapping_type = INVALID_MAP;
  unsigned int n_nodes = 0;
  unsigned int quadrature_order = 0;
  unsigned int n_dofs = 0;
};

struct KokkosPetscAssemblyPlan
{
  std::vector<HilbertElementAssemblyRecord> records;
  std::size_t total_rhs_entries = 0;
  std::size_t total_mat_entries = 0;
  KokkosHilbertBatchData batch_data;
  std::vector<PetscInt> rhs_coo_rows;
  std::vector<PetscInt> mat_coo_rows;
  std::vector<PetscInt> mat_coo_cols;
  KokkosScalarView<Number> rhs_entry_values;
  KokkosScalarView<Number> mat_entry_values;
  KokkosAnalyticGoalBatchData analytic_goal_batch_data;
  KokkosFEMGoalBatchData fem_goal_batch_data;
  const void * geometry_cache_id = nullptr;
  const void * dof_index_cache_id = nullptr;
  FEType fe_type;
  unsigned int hilbert_order = 0;
  int extra_quadrature_order = 0;
  std::set<subdomain_id_type> subdomains;
  const void * coo_matrix_target = nullptr;
  const void * coo_rhs_target = nullptr;
  const void * analytic_goal_target = nullptr;
  const void * analytic_goal_gradient_target = nullptr;
  Real analytic_goal_time = 0.;
  const void * fem_goal_target = nullptr;
  const void * input_vector_target = nullptr;
  KokkosHilbertHomogeneousBucket single_hilbert_bucket;
  bool has_single_hilbert_bucket = false;
};

bool
get_single_supported_hilbert_bucket(const KokkosPetscAssemblyPlan & plan,
                                    const MeshBase::KokkosGeometryCache & geometry_cache,
                                    KokkosHilbertHomogeneousBucket & bucket)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::detect homogeneous bucket");

  if (plan.records.empty())
    return false;

  const auto & first_record = plan.records.front();
  bucket.key = make_hilbert_shape_key(geometry_cache.host_element_types[first_record.elem_index],
                                      geometry_cache.host_element_p_levels[first_record.elem_index],
                                      plan.fe_type);
  bucket.mapping_type = geometry_cache.host_element_mapping_types[first_record.elem_index];
  bucket.n_nodes = geometry_cache.host_element_n_nodes[first_record.elem_index];
  bucket.quadrature_order = first_record.quadrature_order;
  bucket.n_dofs = first_record.n_dofs;

  if (bucket.n_dofs > kokkos_hilbert_max_dofs ||
      !libMesh::Kokkos::detail::supports_hilbert_local_assembly(
          bucket.key, bucket.mapping_type, bucket.quadrature_order))
    return false;

  for (const auto & record : plan.records)
  {
    const FEShapeKey key =
        make_hilbert_shape_key(geometry_cache.host_element_types[record.elem_index],
                               geometry_cache.host_element_p_levels[record.elem_index],
                               plan.fe_type);
    if (key.family != bucket.key.family || key.elem_type != bucket.key.elem_type ||
        key.order != bucket.key.order ||
        geometry_cache.host_element_mapping_types[record.elem_index] != bucket.mapping_type ||
        geometry_cache.host_element_n_nodes[record.elem_index] != bucket.n_nodes ||
        record.quadrature_order != bucket.quadrature_order || record.n_dofs != bucket.n_dofs)
      return false;
  }

  return true;
}

void
cache_single_supported_hilbert_bucket(KokkosPetscAssemblyPlan & plan,
                                      const MeshBase::KokkosGeometryCache & geometry_cache)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::cache homogeneous bucket");

  KokkosHilbertHomogeneousBucket bucket;
  plan.has_single_hilbert_bucket =
      get_single_supported_hilbert_bucket(plan, geometry_cache, bucket);
  plan.single_hilbert_bucket =
      plan.has_single_hilbert_bucket ? bucket : KokkosHilbertHomogeneousBucket{};
}

bool
get_cached_single_supported_hilbert_bucket(const KokkosPetscAssemblyPlan & plan,
                                           KokkosHilbertHomogeneousBucket & bucket)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::get homogeneous bucket cache");

  if (!plan.has_single_hilbert_bucket)
    return false;

  bucket = plan.single_hilbert_bucket;
  return true;
}

void
ensure_kokkos_entry_value_storage(KokkosPetscAssemblyPlan & plan)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::ensure entry value storage");

  if (plan.rhs_entry_values.extent(0) != plan.total_rhs_entries)
    plan.rhs_entry_values =
        KokkosScalarView<Number>("hilbert_rhs_entry_values", plan.total_rhs_entries);
  if (plan.mat_entry_values.extent(0) != plan.total_mat_entries)
    plan.mat_entry_values =
        KokkosScalarView<Number>("hilbert_mat_entry_values", plan.total_mat_entries);
}

bool
is_petsc_kokkos_vec(const Parallel::Communicator & comm, Vec vec)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::check PETSc Kokkos vector");

  const char * vec_type = nullptr;
  LibmeshPetscCall2(comm, VecGetType(vec, &vec_type));

  PetscBool is_seq_kokkos = PETSC_FALSE;
  PetscBool is_mpi_kokkos = PETSC_FALSE;
  LibmeshPetscCall2(comm, PetscStrcmp(vec_type, "seqkokkos", &is_seq_kokkos));
  LibmeshPetscCall2(comm, PetscStrcmp(vec_type, "mpikokkos", &is_mpi_kokkos));
  return is_seq_kokkos || is_mpi_kokkos;
}

bool
is_petsc_kokkos_mat(const Parallel::Communicator & comm, Mat mat)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::check PETSc Kokkos matrix");

  const char * mat_type = nullptr;
  LibmeshPetscCall2(comm, MatGetType(mat, &mat_type));

  PetscBool is_seq_aijkokkos = PETSC_FALSE;
  PetscBool is_mpi_aijkokkos = PETSC_FALSE;
  LibmeshPetscCall2(comm, PetscStrcmp(mat_type, MATSEQAIJKOKKOS, &is_seq_aijkokkos));
  LibmeshPetscCall2(comm, PetscStrcmp(mat_type, MATMPIAIJKOKKOS, &is_mpi_aijkokkos));
  return is_seq_aijkokkos || is_mpi_aijkokkos;
}

void
build_kokkos_petsc_element_coo(const DofMap::KokkosDofIndexCache & dof_index_cache,
                               KokkosPetscAssemblyPlan & plan)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::build element COO");

  plan.rhs_coo_rows.assign(plan.total_rhs_entries, 0);
  plan.mat_coo_rows.assign(plan.total_mat_entries, 0);
  plan.mat_coo_cols.assign(plan.total_mat_entries, 0);

  for (const auto & record : plan.records)
  {
    const auto elem_dofs =
        &dof_index_cache.host_element_dof_indices[record.elem_index * dof_index_cache.max_dofs];

    for (unsigned int i = 0; i != record.n_dofs; ++i)
    {
      const PetscInt row = cast_int<PetscInt>(elem_dofs[i]);
      plan.rhs_coo_rows[record.rhs_offset + i] = row;

      for (unsigned int j = 0; j != record.n_dofs; ++j)
      {
        const std::size_t mat_entry = record.mat_offset + i * record.n_dofs + j;
        plan.mat_coo_rows[mat_entry] = row;
        plan.mat_coo_cols[mat_entry] = cast_int<PetscInt>(elem_dofs[j]);
      }
    }
  }
}

bool
ensure_kokkos_petsc_coo_storage(::HilbertSystem & sys,
                                KokkosPetscAssemblyPlan & plan,
                                PetscMatrixBase<Number> & system_matrix,
                                PetscVector<Number> & system_rhs)
{
  LOG_SCOPE("L2Kokkos::prepare PETSc COO storage", "HilbertSystem");
  KokkosProfilingRegion kokkos_region("L2Kokkos::prepare PETSc COO storage");

  if (!is_petsc_kokkos_mat(sys.comm(), system_matrix.mat()))
    libmesh_error_msg("HilbertSystem Kokkos PETSc COO assembly requires PETSc AIJKokkos "
                      "matrices. Run with -mat_type aijkokkos.");

  if (plan.coo_matrix_target != system_matrix.mat())
  {
    {
      KokkosProfilingRegion preallocation_region("PETSc::MatSetPreallocationCOO");
      LibmeshPetscCall2(sys.comm(),
                        MatSetPreallocationCOO(system_matrix.mat(),
                                               cast_int<PetscCount>(plan.mat_coo_rows.size()),
                                               plan.mat_coo_rows.data(),
                                               plan.mat_coo_cols.data()));
    }
    LibmeshPetscCall2(
        sys.comm(), MatSetOption(system_matrix.mat(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
    plan.coo_matrix_target = system_matrix.mat();
  }

  if (plan.coo_rhs_target != system_rhs.vec())
  {
    KokkosProfilingRegion preallocation_region("PETSc::VecSetPreallocationCOO");
    LibmeshPetscCall2(sys.comm(),
                      VecSetPreallocationCOO(system_rhs.vec(),
                                             cast_int<PetscCount>(plan.rhs_coo_rows.size()),
                                             plan.rhs_coo_rows.data()));
    plan.coo_rhs_target = system_rhs.vec();
  }

  return true;
}

namespace
{

bool
build_kokkos_element_state(::HilbertSystem & sys,
                           const Elem & elem,
                           KokkosElementAssemblyState & state)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::build element state");

  auto * solution_vector = dynamic_cast<PetscVector<Number> *>(sys.current_local_solution.get());
  if (!solution_vector || !solution_vector->supports_kokkos_access())
    return false;

  const DofMap & dof_map = sys.get_dof_map();
  state.geometry_cache = &sys.get_mesh().get_kokkos_geometry_cache();
  state.elem_index = sys.get_mesh().get_kokkos_elem_index(elem);
  if (state.elem_index == libMesh::invalid_uint)
    return false;

  state.elem_type = state.geometry_cache->host_element_types[state.elem_index];
  state.mapping_type = state.geometry_cache->host_element_mapping_types[state.elem_index];
  state.elem_n_nodes = state.geometry_cache->host_element_n_nodes[state.elem_index];
  state.elem_p_level = state.geometry_cache->host_element_p_levels[state.elem_index];

  state.solution_local_indices =
      dof_map.require_kokkos_local_index_cache(*sys.current_local_solution, 0);
  if (!state.solution_local_indices)
    return false;

  state.solution_vector = solution_vector;
  return true;
}

bool
build_kokkos_fem_goal_data(const ElemType elem_type,
                           const ElemMappingType mapping_type,
                           const unsigned int elem_p_level,
                           const unsigned int hilbert_order,
                           System & input_system,
                           const libMesh::Kokkos::KokkosParsedFEMFunction<Number> & goal_function,
                           KokkosFEMGoalData & goal_data)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::build FEM goal data");

  const unsigned int n_fields = goal_function.n_field_variables();
  if (n_fields > kokkos_parsed_fem_max_fields)
    return false;

  auto * input_vector =
      dynamic_cast<PetscVector<Number> *>(input_system.current_local_solution.get());
  if (!input_vector || !input_vector->supports_kokkos_access())
    return false;

  goal_data.field_keys.size = n_fields;
  goal_data.field_dofs.size = n_fields;
  goal_data.input_vector = input_vector;

  for (unsigned int field = 0; field != n_fields; ++field)
  {
    const unsigned int var_num = goal_function.field_variable_number(field);
    const FEType field_type = input_system.variable_type(var_num);
    const FEShapeKey field_key = make_hilbert_shape_key(elem_type, elem_p_level, field_type);

    if (mapping_type != LAGRANGE_MAP || !supports_shape_with_lagrange_map(field_key) ||
        (hilbert_order > 0 && !supports_grad_shape(field_key)))
      return false;

    const unsigned int field_n_dofs =
        FEInterface::n_dofs(libMesh::Kokkos::dim_from_topology(elem_type), field_type, elem_type);
    if (field_n_dofs > kokkos_hilbert_max_dofs)
      return false;

    const auto * local_index_cache = input_system.get_dof_map().require_kokkos_local_index_cache(
        *input_system.current_local_solution, var_num);
    if (!local_index_cache)
      return false;

    goal_data.field_keys.values[field] = field_key;
    goal_data.field_dofs.values[field] = field_n_dofs;
    goal_data.field_local_indices[field] = local_index_cache->element_local_indices;
  }

  return true;
}

void
build_hilbert_batch_data(const std::vector<HilbertElementAssemblyRecord> & records,
                         const FEType & fe_type,
                         const MeshBase::KokkosGeometryCache & geometry_cache,
                         KokkosHilbertBatchData & batch_data)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::build batch data");

  batch_data.elem_indices = KokkosUnsignedIntView("hilbert_elem_indices", records.size());
  batch_data.elem_n_dofs = KokkosUnsignedIntView("hilbert_elem_n_dofs", records.size());
  batch_data.quadrature_orders = KokkosUnsignedIntView("hilbert_quadrature_orders", records.size());
  batch_data.rhs_offsets = KokkosSizeView("hilbert_rhs_offsets", records.size());
  batch_data.mat_offsets = KokkosSizeView("hilbert_mat_offsets", records.size());
  batch_data.shape_keys = KokkosShapeKeyRecordView("hilbert_shape_keys", records.size());

  auto h_elem_indices = ::Kokkos::create_mirror_view(batch_data.elem_indices);
  auto h_elem_n_dofs = ::Kokkos::create_mirror_view(batch_data.elem_n_dofs);
  auto h_quadrature_orders = ::Kokkos::create_mirror_view(batch_data.quadrature_orders);
  auto h_rhs_offsets = ::Kokkos::create_mirror_view(batch_data.rhs_offsets);
  auto h_mat_offsets = ::Kokkos::create_mirror_view(batch_data.mat_offsets);

  for (auto record_index : index_range(records))
  {
    const auto & record = records[record_index];
    h_elem_indices(record_index) = record.elem_index;
    h_elem_n_dofs(record_index) = record.n_dofs;
    h_quadrature_orders(record_index) = record.quadrature_order;
    h_rhs_offsets(record_index) = record.rhs_offset;
    h_mat_offsets(record_index) = record.mat_offset;
  }

  ::Kokkos::deep_copy(batch_data.elem_indices, h_elem_indices);
  ::Kokkos::deep_copy(batch_data.elem_n_dofs, h_elem_n_dofs);
  ::Kokkos::deep_copy(batch_data.quadrature_orders, h_quadrature_orders);
  ::Kokkos::deep_copy(batch_data.rhs_offsets, h_rhs_offsets);
  ::Kokkos::deep_copy(batch_data.mat_offsets, h_mat_offsets);

  const auto elem_indices = batch_data.elem_indices;
  const auto element_types = geometry_cache.element_types_dual.view_device();
  const auto element_p_levels = geometry_cache.element_p_levels_dual.view_device();
  const auto shape_keys = batch_data.shape_keys;
  const auto family = fe_type.family;
  const auto base_order = fe_type.order.get_order();
  const auto n_records = records.size();
  using GeometryExecutionSpace = typename std::decay_t<decltype(element_types)>::execution_space;

  ::Kokkos::parallel_for(
      "hilbert_build_shape_keys",
      ::Kokkos::RangePolicy<GeometryExecutionSpace>(0, cast_int<int>(n_records)),
      KOKKOS_LAMBDA(const int raw_record_index) {
        const unsigned int record_index = static_cast<unsigned int>(raw_record_index);
        const unsigned int elem_index = elem_indices(record_index);
        shape_keys(record_index) = FEShapeKey{
            family,
            element_types(elem_index),
            static_cast<Order>(base_order + static_cast<int>(element_p_levels(elem_index)))};
      });
  ::Kokkos::fence();
}

bool
kokkos_petsc_plan_matches(const ::HilbertSystem & sys,
                          const MeshBase::KokkosGeometryCache & geometry_cache,
                          const DofMap::KokkosDofIndexCache & dof_index_cache,
                          const KokkosPetscAssemblyPlan & plan)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::check assembly plan cache");

  return plan.geometry_cache_id == &geometry_cache && plan.dof_index_cache_id == &dof_index_cache &&
         plan.fe_type == sys.variable_type(0) && plan.hilbert_order == sys.hilbert_order() &&
         plan.extra_quadrature_order == sys.extra_quadrature_order &&
         plan.subdomains == sys.subdomains_list();
}

bool
build_kokkos_petsc_assembly_plan(::HilbertSystem & sys, KokkosPetscAssemblyPlan & plan)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::build assembly plan");

  const auto & geometry_cache = sys.get_mesh().get_kokkos_geometry_cache();
  const auto * dof_index_cache = sys.get_dof_map().get_kokkos_dof_index_cache(0);
  if (!dof_index_cache)
    return false;

  std::size_t total_rhs_entries = 0;
  std::size_t total_mat_entries = 0;
  std::vector<HilbertElementAssemblyRecord> records;
  build_hilbert_element_records(sys, records, total_rhs_entries, total_mat_entries);
  if (!build_hilbert_record_quadrature_orders(sys, records))
    return false;

  KokkosHilbertBatchData batch_data;
  build_hilbert_batch_data(records, sys.variable_type(0), geometry_cache, batch_data);

  plan.records = std::move(records);
  plan.total_rhs_entries = total_rhs_entries;
  plan.total_mat_entries = total_mat_entries;
  plan.batch_data = batch_data;
  plan.rhs_entry_values = {};
  plan.mat_entry_values = {};
  build_kokkos_petsc_element_coo(*dof_index_cache, plan);
  plan.geometry_cache_id = &geometry_cache;
  plan.dof_index_cache_id = dof_index_cache;
  plan.fe_type = sys.variable_type(0);
  plan.hilbert_order = sys.hilbert_order();
  plan.extra_quadrature_order = sys.extra_quadrature_order;
  plan.subdomains = sys.subdomains_list();
  plan.coo_matrix_target = nullptr;
  plan.coo_rhs_target = nullptr;
  plan.analytic_goal_target = nullptr;
  plan.analytic_goal_gradient_target = nullptr;
  plan.analytic_goal_time = 0.;
  plan.fem_goal_target = nullptr;
  plan.input_vector_target = nullptr;
  cache_single_supported_hilbert_bucket(plan, geometry_cache);
  return true;
}

bool
build_kokkos_analytic_goal_batch_data(
    ::HilbertSystem & sys,
    const KokkosPetscAssemblyPlan & plan,
    FunctionBase<Number> & goal_function,
    FunctionBase<Gradient> * goal_gradient,
    KokkosAnalyticGoalBatchData & batch_data)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::sample analytic goal");

  if (sys.hilbert_order() > 0 && !goal_gradient)
    return false;

  const auto & geometry_cache = sys.get_mesh().get_kokkos_geometry_cache();
  const auto h_node_coordinates = geometry_cache.node_coordinates_dual.view_host();
  const auto h_element_node_ids = geometry_cache.element_node_ids_dual.view_host();

  batch_data.qp_offsets =
      KokkosSizeView("hilbert_analytic_goal_qp_offsets",
                     geometry_cache.host_element_ids.size());
  auto h_qp_offsets = ::Kokkos::create_mirror_view(batch_data.qp_offsets);

  std::size_t total_qpoints = 0;
  for (const auto & record : plan.records)
  {
    h_qp_offsets(record.elem_index) = total_qpoints;
    total_qpoints += libMesh::Kokkos::GaussQuadrature::n_points(
        geometry_cache.host_element_types[record.elem_index],
        record.quadrature_order);
  }

  batch_data.values = KokkosScalarView<Number>("hilbert_analytic_goal_values", total_qpoints);
  batch_data.gradients =
      KokkosScalarView<Number>("hilbert_analytic_goal_gradients", total_qpoints * LIBMESH_DIM);

  auto h_values = ::Kokkos::create_mirror_view(batch_data.values);
  auto h_gradients = ::Kokkos::create_mirror_view(batch_data.gradients);
  std::vector<libMesh::Kokkos::RealVector> elem_nodes(geometry_cache.max_nodes);

  for (const auto & record : plan.records)
  {
    const unsigned int elem_index = record.elem_index;
    const ElemType elem_type = geometry_cache.host_element_types[elem_index];
    const ElemMappingType mapping_type = geometry_cache.host_element_mapping_types[elem_index];
    const unsigned int n_nodes = geometry_cache.host_element_n_nodes[elem_index];
    const unsigned int n_qpoints =
        libMesh::Kokkos::GaussQuadrature::n_points(elem_type, record.quadrature_order);
    const std::size_t qp_offset = h_qp_offsets(elem_index);

    for (unsigned int n = 0; n != n_nodes; ++n)
    {
      const unsigned int node_index = h_element_node_ids(elem_index, n);
      elem_nodes[n] = libMesh::Kokkos::zero_vector();
      elem_nodes[n](0) = h_node_coordinates(node_index, 0);
#if LIBMESH_DIM > 1
      elem_nodes[n](1) = h_node_coordinates(node_index, 1);
#endif
#if LIBMESH_DIM > 2
      elem_nodes[n](2) = h_node_coordinates(node_index, 2);
#endif
    }

    for (unsigned int qp = 0; qp != n_qpoints; ++qp)
    {
      const libMesh::Kokkos::RealVector qp_ref =
          libMesh::Kokkos::GaussQuadrature::point(elem_type, record.quadrature_order, qp);
      libMesh::Kokkos::RealVector xyz_vector = libMesh::Kokkos::zero_vector();
      libMesh::Kokkos::RealTensor J = libMesh::Kokkos::zero_tensor();
      libMesh::Kokkos::physical_point_and_jacobian(mapping_type,
                                                   elem_type,
                                                   elem_nodes.data(),
                                                   n_nodes,
                                                   qp_ref(0),
                                                   qp_ref(1),
                                                   qp_ref(2),
                                                   xyz_vector,
                                                   J);
      libmesh_ignore(J);

      const Point xyz(xyz_vector(0), xyz_vector(1), xyz_vector(2));
      const std::size_t flat_qp = qp_offset + qp;
      h_values(flat_qp) = goal_function(xyz, sys.time);

      Gradient grad;
      grad.zero();
      if (goal_gradient)
        grad = (*goal_gradient)(xyz, sys.time);

      h_gradients(flat_qp * LIBMESH_DIM) = grad(0);
#if LIBMESH_DIM > 1
      h_gradients(flat_qp * LIBMESH_DIM + 1) = grad(1);
#endif
#if LIBMESH_DIM > 2
      h_gradients(flat_qp * LIBMESH_DIM + 2) = grad(2);
#endif
    }
  }

  ::Kokkos::deep_copy(batch_data.qp_offsets, h_qp_offsets);
  ::Kokkos::deep_copy(batch_data.values, h_values);
  ::Kokkos::deep_copy(batch_data.gradients, h_gradients);

  return true;
}

bool
ensure_kokkos_analytic_goal_batch_data(
    ::HilbertSystem & sys,
    FunctionBase<Number> & goal_function,
    FunctionBase<Gradient> * goal_gradient,
    KokkosPetscAssemblyPlan & plan)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::ensure analytic goal samples");

  if (plan.analytic_goal_target == &goal_function &&
      plan.analytic_goal_gradient_target == goal_gradient &&
      plan.analytic_goal_time == sys.time)
    return true;

  KokkosAnalyticGoalBatchData batch_data;
  if (!build_kokkos_analytic_goal_batch_data(
          sys, plan, goal_function, goal_gradient, batch_data))
    return false;

  plan.analytic_goal_batch_data = batch_data;
  plan.analytic_goal_target = &goal_function;
  plan.analytic_goal_gradient_target = goal_gradient;
  plan.analytic_goal_time = sys.time;
  return true;
}

bool
build_kokkos_fem_goal_batch_data(
    ::HilbertSystem & sys,
    const KokkosPetscAssemblyPlan & plan,
    const libMesh::Kokkos::KokkosParsedFEMFunction<Number> & goal_function,
    KokkosFEMGoalBatchData & batch_data)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::build FEM goal batch data");

  libmesh_assert(sys.input_system);
  System & input_system = *sys.input_system;
  const auto & geometry_cache = sys.get_mesh().get_kokkos_geometry_cache();
  const unsigned int n_fields = goal_function.n_field_variables();
  if (n_fields > kokkos_parsed_fem_max_fields)
    return false;

  auto * input_vector =
      dynamic_cast<PetscVector<Number> *>(input_system.current_local_solution.get());
  if (!input_vector || !input_vector->supports_kokkos_access())
    return false;

  batch_data.input_vector = input_vector;
  batch_data.field_local_indices.size = n_fields;
  batch_data.field_keys =
      KokkosFieldKeyRecordView("hilbert_fem_goal_field_keys", n_fields, plan.records.size());
  batch_data.field_dofs =
      KokkosFieldDofRecordView("hilbert_fem_goal_field_dofs", n_fields, plan.records.size());
  auto h_field_keys = ::Kokkos::create_mirror_view(batch_data.field_keys);
  auto h_field_dofs = ::Kokkos::create_mirror_view(batch_data.field_dofs);

  for (unsigned int field = 0; field != n_fields; ++field)
  {
    const unsigned int var_num = goal_function.field_variable_number(field);
    const FEType field_type = input_system.variable_type(var_num);
    const auto * local_index_cache = input_system.get_dof_map().require_kokkos_local_index_cache(
        *input_system.current_local_solution, var_num);
    if (!local_index_cache)
      return false;

    batch_data.field_local_indices.values[field] = local_index_cache->element_local_indices;

    for (auto record_index : index_range(plan.records))
    {
      const auto & record = plan.records[record_index];
      const ElemType elem_type = geometry_cache.host_element_types[record.elem_index];
      const unsigned int elem_p_level = geometry_cache.host_element_p_levels[record.elem_index];
      const ElemMappingType mapping_type =
          geometry_cache.host_element_mapping_types[record.elem_index];
      const FEShapeKey field_key = make_hilbert_shape_key(elem_type, elem_p_level, field_type);
      if (mapping_type != LAGRANGE_MAP || !supports_shape_with_lagrange_map(field_key) ||
          (sys.hilbert_order() > 0 && !supports_grad_shape(field_key)))
        return false;

      const unsigned int field_n_dofs =
          FEInterface::n_dofs(libMesh::Kokkos::dim_from_topology(elem_type), field_type, elem_type);
      if (field_n_dofs > kokkos_hilbert_max_dofs)
        return false;

      h_field_keys(field, record_index) = field_key;
      h_field_dofs(field, record_index) = field_n_dofs;
    }
  }

  ::Kokkos::deep_copy(batch_data.field_keys, h_field_keys);
  ::Kokkos::deep_copy(batch_data.field_dofs, h_field_dofs);

  return true;
}

bool
ensure_kokkos_fem_goal_batch_data(
    ::HilbertSystem & sys,
    const libMesh::Kokkos::KokkosParsedFEMFunction<Number> & goal_function,
    KokkosPetscAssemblyPlan & plan)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::ensure FEM goal batch data");

  libmesh_assert(sys.input_system);
  const void * input_vector_target = sys.input_system->current_local_solution.get();
  if (plan.fem_goal_target == &goal_function && plan.input_vector_target == input_vector_target)
    return true;

  KokkosFEMGoalBatchData batch_data;
  if (!build_kokkos_fem_goal_batch_data(sys, plan, goal_function, batch_data))
    return false;

  plan.fem_goal_batch_data = batch_data;
  plan.fem_goal_target = &goal_function;
  plan.input_vector_target = input_vector_target;
  return true;
}

bool
assemble_kokkos_hilbert_element(const FEShapeKey key,
                                const unsigned int quadrature_order,
                                const unsigned int hilbert_order,
                                const Number solution_derivative,
                                const libMesh::Kokkos::KokkosParsedFunction<Number> & goal_function,
                                const KokkosElementAssemblyState & state,
                                const bool request_jacobian,
                                DenseSubVector<Number> & F,
                                DenseSubMatrix<Number> & K)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::assemble analytic element");

  const unsigned int n_dofs = F.size();
  KokkosScalarView<Number> d_F("hilbert_residual", n_dofs);
  KokkosDenseJacobianView d_K("hilbert_jacobian", n_dofs, n_dofs);
  auto goal_copy = goal_function;
  auto goal_grad = goal_copy.gradient_function();
  const auto goal_access = libMesh::Kokkos::detail::make_hilbert_analytic_goal_access(
      std::move(goal_copy), std::move(goal_grad));
  const libMesh::Kokkos::detail::DenseElementOutputSink<decltype(d_F), decltype(d_K)> sink{
      d_F, d_K, n_dofs, request_jacobian};

  auto coeff_guard = state.solution_vector->make_kokkos_read_view_guard();
  const auto coeff = libMesh::Kokkos::detail::make_gathered_coeff_access(
      coeff_guard.view(), state.solution_local_indices->element_local_indices, state.elem_index);

  const bool success =
      libMesh::Kokkos::detail::run_hilbert_system_assembly<kokkos_hilbert_max_dofs>(
          key,
          state.mapping_type,
          state.geometry_cache->node_coordinates,
          state.geometry_cache->element_node_ids,
          state.elem_index,
          state.elem_n_nodes,
          quadrature_order,
          hilbert_order,
          coeff,
          solution_derivative,
          goal_access,
          request_jacobian,
          sink,
          "hilbert_local_assembly");

  if (!success)
    return false;

  accumulate_hilbert_dense_outputs(d_F, d_K, request_jacobian, F, K);
  return true;
}

bool
assemble_kokkos_hilbert_fem_goal_element(
    const FEShapeKey output_key,
    const unsigned int quadrature_order,
    const unsigned int hilbert_order,
    const Number solution_derivative,
    System & input_system,
    const libMesh::Kokkos::KokkosParsedFEMFunction<Number> & goal_function,
    const KokkosElementAssemblyState & state,
    const bool request_jacobian,
    DenseSubVector<Number> & F,
    DenseSubMatrix<Number> & K)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::assemble FEM goal element");

  KokkosFEMGoalData goal_data;
  if (!build_kokkos_fem_goal_data(state.elem_type,
                                  state.mapping_type,
                                  state.elem_p_level,
                                  hilbert_order,
                                  input_system,
                                  goal_function,
                                  goal_data))
    return false;

  const unsigned int n_dofs = F.size();
  KokkosScalarView<Number> d_F("hilbert_residual", n_dofs);
  KokkosDenseJacobianView d_K("hilbert_jacobian", n_dofs, n_dofs);
  const libMesh::Kokkos::detail::DenseElementOutputSink<decltype(d_F), decltype(d_K)> sink{
      d_F, d_K, n_dofs, request_jacobian};

  const auto assemble_with_input_coeffs =
      [&](const auto & coeff_values, const auto & input_coeff_values)
  {
    const auto coeff = libMesh::Kokkos::detail::make_gathered_coeff_access(
        coeff_values, state.solution_local_indices->element_local_indices, state.elem_index);

    const auto goal_access = libMesh::Kokkos::detail::GatheredParsedFEMGoalAccess<
        KokkosFieldKeyStorage,
        KokkosFieldDofStorage,
        std::decay_t<decltype(input_coeff_values)>,
        KokkosLocalIndexView,
        libMesh::Kokkos::KokkosParsedFEMFunction<Number>,
        kokkos_parsed_fem_max_fields>(goal_data.field_keys,
                                      goal_data.field_dofs,
                                      input_coeff_values,
                                      goal_data.field_local_indices.data(),
                                      goal_function);

    return libMesh::Kokkos::detail::run_hilbert_system_assembly<kokkos_hilbert_max_dofs>(
        output_key,
        state.mapping_type,
        state.geometry_cache->node_coordinates,
        state.geometry_cache->element_node_ids,
        state.elem_index,
        state.elem_n_nodes,
        quadrature_order,
        hilbert_order,
        coeff,
        solution_derivative,
        goal_access,
        request_jacobian,
        sink,
        "hilbert_local_fem_goal_assembly");
  };

  auto coeff_guard = state.solution_vector->make_kokkos_read_view_guard();
  const bool success = (goal_data.input_vector == state.solution_vector)
                           ? assemble_with_input_coeffs(coeff_guard.view(), coeff_guard.view())
                           : [&]()
  {
    auto input_guard = goal_data.input_vector->make_kokkos_read_view_guard();
    return assemble_with_input_coeffs(coeff_guard.view(), input_guard.view());
  }();

  if (!success)
    return false;

  accumulate_hilbert_dense_outputs(d_F, d_K, request_jacobian, F, K);
  return true;
}

#if defined(LIBMESH_HAVE_PETSC)
bool
assemble_host_exact_parsed_fem_goal_element(
    ::HilbertSystem & sys,
    FEMContext & c,
    const bool request_jacobian,
    DenseSubVector<Number> & F,
    DenseSubMatrix<Number> & K,
    const libMesh::Kokkos::KokkosParsedFEMFunction<Number> & goal_function)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::assemble host exact FEM goal element");

  if (!sys.input_system)
    return false;

  FEMContext * goal_context_ptr = sys.get_input_context(c);
  if (!goal_context_ptr)
    return false;

  FEMContext & goal_context = *goal_context_ptr;
  goal_context.pre_fe_reinit(*sys.input_system, &c.get_elem());
  goal_context.elem_fe_reinit();

  detail::HostHilbertFEAccess fe(c, 0, sys.hilbert_order());
  detail::HostHilbertAccumulator accum(F, K);
  auto solution = detail::make_hilbert_solution_access(
      fe, c.get_elem_solution(0), c.get_elem_solution_derivative());
  HostExactParsedFEMGoalAccess goal_access(goal_function, goal_context);
  detail::assemble_hilbert_element(
      fe, solution, goal_access, request_jacobian, sys.hilbert_order(), accum);
  return true;
}
#endif

template <unsigned int MaxDofs,
          typename GoalAccess,
          typename NodeCoordinateStorage,
          typename ElemNodeIdStorage>
void
run_kokkos_hilbert_bucket_value_batch(const KokkosHilbertHomogeneousBucket & bucket,
                                      const NodeCoordinateStorage & node_coordinates,
                                      const ElemNodeIdStorage & element_node_ids,
                                      KokkosPetscAssemblyPlan & plan,
                                      const unsigned int hilbert_order,
                                      GoalAccess goal_access,
                                      const char * const kernel_name)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::launch bucket value batch");

  if (libMesh::Kokkos::detail::run_hilbert_system_bucket_host_reference_value_batch<MaxDofs>(
          bucket.key,
          bucket.mapping_type,
          bucket.n_nodes,
          bucket.quadrature_order,
          node_coordinates,
          element_node_ids,
          plan.batch_data.elem_indices,
          plan.batch_data.elem_n_dofs,
          plan.batch_data.rhs_offsets,
          plan.batch_data.mat_offsets,
          hilbert_order,
          goal_access,
          plan.rhs_entry_values,
          plan.mat_entry_values,
          "hilbert_bucket_host_reference_record_values"))
    return;

  libMesh::Kokkos::detail::run_hilbert_system_bucket_team_value_batch<MaxDofs>(
      bucket.key,
      bucket.mapping_type,
      bucket.n_nodes,
      bucket.n_dofs,
      bucket.quadrature_order,
      node_coordinates,
      element_node_ids,
      plan.batch_data.elem_indices,
      plan.batch_data.elem_n_dofs,
      plan.batch_data.rhs_offsets,
      plan.batch_data.mat_offsets,
      hilbert_order,
      goal_access,
      plan.rhs_entry_values,
      plan.mat_entry_values,
      kernel_name);
}

template <typename GoalAccess, typename NodeCoordinateStorage, typename ElemNodeIdStorage>
bool
kokkos_hilbert_bucket_value_batch(const KokkosPetscAssemblyPlan & const_plan,
                                      const NodeCoordinateStorage & node_coordinates,
                                      const ElemNodeIdStorage & element_node_ids,
                                      KokkosPetscAssemblyPlan & plan,
                                      const unsigned int hilbert_order,
                                      GoalAccess goal_access)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::try bucket value batch");

  using ExecutionSpace = typename std::decay_t<NodeCoordinateStorage>::execution_space;
  if constexpr (!libMesh::Kokkos::detail::is_host_thread_execution_space<ExecutionSpace>())
    if (kokkos_l2_env_enabled("LIBMESH_KOKKOS_L2_DISABLE_BUCKET"))
      return false;

  KokkosHilbertHomogeneousBucket bucket;
  if (!get_cached_single_supported_hilbert_bucket(const_plan, bucket))
    return false;

  run_kokkos_hilbert_bucket_value_batch<kokkos_hilbert_max_dofs>(
      bucket,
      node_coordinates,
      element_node_ids,
      plan,
      hilbert_order,
      goal_access,
      "hilbert_bucket_team_record_values");
  return true;
}

bool
assemble_kokkos_petsc_global_system(
    ::HilbertSystem & sys,
    KokkosPetscAssemblyPlan & plan,
    const bool analytic_goal,
    FunctionBase<Number> * analytic_goal_func,
    FunctionBase<Gradient> * analytic_goal_grad,
    const libMesh::Kokkos::KokkosParsedFunction<Number> * kokkos_analytic_goal,
    const libMesh::Kokkos::KokkosParsedFEMFunction<Number> * fem_goal,
    PetscMatrixBase<Number> & system_matrix,
    PetscVector<Number> & system_rhs,
    ::HilbertSystem::KokkosAssemblyPath & assembly_path,
    ::HilbertSystem::KokkosTimingInfo * timing)
{
  LOG_SCOPE("L2Kokkos::assemble PETSc global system", "HilbertSystem");
  KokkosProfilingRegion kokkos_region("L2Kokkos::assemble PETSc global system");
  using clock = std::chrono::steady_clock;
  const auto seconds_between = [](const auto start, const auto stop)
  { return std::chrono::duration_cast<std::chrono::duration<Real>>(stop - start).count(); };

  libmesh_error_msg_if(sys.has_static_condensation(),
                       "HilbertSystem Kokkos PETSc COO assembly does not support static "
                       "condensation.");
  libmesh_error_msg_if(sys.get_dof_map().n_constrained_dofs(),
                       "HilbertSystem Kokkos PETSc COO assembly does not yet support "
                       "constrained dofs.");
  libmesh_error_msg_if(!analytic_goal && !fem_goal,
                       "HilbertSystem Kokkos PETSc COO assembly requires an analytic "
                       "goal or parsed FEM goal.");

  {
    const auto coo_setup_start = clock::now();
    if (!ensure_kokkos_petsc_coo_storage(sys, plan, system_matrix, system_rhs))
      libmesh_error_msg("Failed to prepare PETSc COO storage for HilbertSystem Kokkos assembly.");
    if (timing)
      timing->assembly_coo_setup_seconds += seconds_between(coo_setup_start, clock::now());
  }

  assembly_path = ::HilbertSystem::KokkosAssemblyPath::petsc_coo;

  {
    const auto records_start = clock::now();
    LOG_SCOPE("L2Kokkos::assemble all records", "HilbertSystem");
    KokkosProfilingRegion records_region("L2Kokkos::assemble all records");

    const auto & geometry_cache = sys.get_mesh().get_kokkos_geometry_cache();
    if (analytic_goal)
    {
      libmesh_error_msg_if(!analytic_goal_func,
                           "HilbertSystem Kokkos PETSc COO assembly requires an analytic goal.");
      libmesh_error_msg_if(sys.hilbert_order() > 0 && !analytic_goal_grad,
                           "HilbertSystem Kokkos PETSc COO assembly requires an analytic goal "
                           "gradient for Hilbert orders greater than zero.");

      // Prefer evaluating the parsed goal (and its symbolic gradient) on device at the
      // quadrature points the batch kernels already compute; the host pre-sampling sweep
      // (serial fparser evaluation per element x qp, FDM gradients) is kept as a fallback
      // for goals without device support.
      const bool use_device_goal =
          kokkos_analytic_goal && !kokkos_l2_env_enabled("LIBMESH_KOKKOS_L2_HOST_GOAL_SAMPLING");

      ensure_kokkos_entry_value_storage(plan);
      if (!use_device_goal)
        if (!ensure_kokkos_analytic_goal_batch_data(
                sys, *analytic_goal_func, analytic_goal_grad, plan))
          return false;

      const auto node_coordinates = geometry_cache.node_coordinates_dual.view_device();
      const auto element_node_ids = geometry_cache.element_node_ids_dual.view_device();
      const auto element_mapping_types = geometry_cache.element_mapping_types_dual.view_device();
      const auto element_n_nodes = geometry_cache.element_n_nodes_dual.view_device();

      const auto assemble_with_goal = [&](auto goal_access, const char * bucket_name)
      {
        if (!kokkos_hilbert_bucket_value_batch(
                plan, node_coordinates, element_node_ids, plan, sys.hilbert_order(), goal_access))
          libMesh::Kokkos::detail::run_hilbert_system_value_batch<kokkos_hilbert_max_dofs>(
              node_coordinates,
              element_node_ids,
              element_mapping_types,
              element_n_nodes,
              plan.batch_data.shape_keys,
              plan.batch_data.elem_indices,
              plan.batch_data.elem_n_dofs,
              plan.batch_data.quadrature_orders,
              plan.batch_data.rhs_offsets,
              plan.batch_data.mat_offsets,
              sys.hilbert_order(),
              goal_access,
              plan.rhs_entry_values,
              plan.mat_entry_values,
              bucket_name);
      };

      if (use_device_goal)
      {
        // Move the functors in so the goal access stores them by value; storing
        // references to these host stack objects would crash the device kernels.
        auto timed_goal = kokkos_analytic_goal->with_time(sys.time);
        auto timed_goal_grad = timed_goal.gradient_function();
        auto goal_access = libMesh::Kokkos::detail::make_hilbert_analytic_goal_access(
            std::move(timed_goal), std::move(timed_goal_grad));
        assemble_with_goal(goal_access, "hilbert_device_analytic_all_record_values");
      }
      else
      {
        auto goal_access = libMesh::Kokkos::detail::SampledAnalyticGoalAccess<
            KokkosSizeView,
            KokkosScalarView<Number>,
            KokkosScalarView<Number>>(plan.analytic_goal_batch_data.qp_offsets,
                                      plan.analytic_goal_batch_data.values,
                                      plan.analytic_goal_batch_data.gradients);
        assemble_with_goal(goal_access, "hilbert_device_analytic_all_record_values");
      }
    }
    else
    {
      if (!ensure_kokkos_fem_goal_batch_data(sys, *fem_goal, plan))
        return false;

      auto input_guard = plan.fem_goal_batch_data.input_vector->make_kokkos_read_view_guard();
      const auto timed_goal = fem_goal->with_time(sys.time);
      ensure_kokkos_entry_value_storage(plan);
      libMesh::Kokkos::detail::run_hilbert_system_fem_value_batch<kokkos_hilbert_max_dofs,
                                                                  kokkos_parsed_fem_max_fields>(
          geometry_cache.node_coordinates_dual.view_device(),
          geometry_cache.element_node_ids_dual.view_device(),
          geometry_cache.element_mapping_types_dual.view_device(),
          geometry_cache.element_n_nodes_dual.view_device(),
          plan.batch_data.shape_keys,
          plan.batch_data.elem_indices,
          plan.batch_data.elem_n_dofs,
          plan.batch_data.quadrature_orders,
          plan.batch_data.rhs_offsets,
          plan.batch_data.mat_offsets,
          plan.fem_goal_batch_data.field_keys,
          plan.fem_goal_batch_data.field_dofs,
          plan.fem_goal_batch_data.field_local_indices,
          input_guard.view(),
          timed_goal,
          sys.hilbert_order(),
          plan.rhs_entry_values,
          plan.mat_entry_values,
          "hilbert_fem_all_record_values");
    }
    ::Kokkos::fence();
    if (kokkos_l2_env_enabled("LIBMESH_KOKKOS_L2_DEBUG"))
    {
      debug_kokkos_entry_values(sys.comm(), "rhs_entry_values", plan.rhs_entry_values);
      debug_kokkos_entry_values(sys.comm(), "mat_entry_values", plan.mat_entry_values);
      debug_petsc_object_types(sys.comm(), system_rhs.vec(), system_matrix.mat());
    }
    if (timing)
      timing->assembly_record_seconds += seconds_between(records_start, clock::now());
  }

  {
    const auto vecset_start = clock::now();
    LOG_SCOPE("L2Kokkos::VecSetValuesCOO", "HilbertSystem");
    KokkosProfilingRegion vec_values_region("PETSc::VecSetValuesCOO");
    if (kokkos_l2_env_enabled("LIBMESH_KOKKOS_L2_HOST_COO_VALUES"))
    {
      auto rhs_host =
          ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), plan.rhs_entry_values);
      LibmeshPetscCall2(sys.comm(), VecSetValuesCOO(system_rhs.vec(), rhs_host.data(), ADD_VALUES));
    }
    else
      LibmeshPetscCall2(
          sys.comm(), VecSetValuesCOO(system_rhs.vec(), plan.rhs_entry_values.data(), ADD_VALUES));
    if (timing)
      timing->assembly_vecset_values_seconds += seconds_between(vecset_start, clock::now());
  }

  {
    const auto matset_start = clock::now();
    LOG_SCOPE("L2Kokkos::MatSetValuesCOO", "HilbertSystem");
    KokkosProfilingRegion mat_values_region("PETSc::MatSetValuesCOO");
    if (kokkos_l2_env_enabled("LIBMESH_KOKKOS_L2_HOST_COO_VALUES"))
    {
      auto mat_host =
          ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), plan.mat_entry_values);
      LibmeshPetscCall2(
          sys.comm(), MatSetValuesCOO(system_matrix.mat(), mat_host.data(), ADD_VALUES));
    }
    else
      LibmeshPetscCall2(
          sys.comm(),
          MatSetValuesCOO(system_matrix.mat(), plan.mat_entry_values.data(), ADD_VALUES));
    if (timing)
      timing->assembly_matset_values_seconds += seconds_between(matset_start, clock::now());
  }

  if (kokkos_l2_env_enabled("LIBMESH_KOKKOS_L2_DEBUG"))
    debug_petsc_object_norms(sys.comm(), system_rhs.vec(), system_matrix.mat());

  return true;
}
} // anonymous namespace

} // namespace libMesh

using libMesh::assemble_host_exact_parsed_fem_goal_element;
using libMesh::assemble_kokkos_hilbert_element;
using libMesh::assemble_kokkos_hilbert_fem_goal_element;
using libMesh::assemble_kokkos_petsc_global_system;
using libMesh::build_kokkos_element_state;
using libMesh::build_kokkos_petsc_assembly_plan;
using libMesh::cast_int;
using libMesh::DenseSubMatrix;
using libMesh::DenseSubVector;
using libMesh::Elem;
using libMesh::ensure_kokkos_goal_cache;
using libMesh::FEMContext;
using libMesh::FEShapeKey;
using libMesh::is_petsc_kokkos_vec;
using libMesh::kokkos_hilbert_max_dofs;
using libMesh::kokkos_parsed_fem_max_fields;
using libMesh::kokkos_petsc_plan_matches;
using libMesh::KokkosElementAssemblyState;
using libMesh::KokkosPetscAssemblyPlan;
using libMesh::KokkosProfilingRegion;
using libMesh::KokkosScalarView;
using libMesh::libmesh_ignore;
using libMesh::LinearSolver;
using libMesh::make_hilbert_shape_key;
using libMesh::Number;
using libMesh::ParsedFEMFunction;
using libMesh::ParsedFunction;
using libMesh::PetscMatrixBase;
using libMesh::PetscVector;
using libMesh::Real;
namespace detail = libMesh::detail;

struct HilbertSystemKokkosState
{
  std::unique_ptr<libMesh::Kokkos::KokkosParsedFunction<libMesh::Number, 64>> goal_func;
  std::unique_ptr<libMesh::Kokkos::KokkosParsedFEMFunction<libMesh::Number, 64, 16>> fem_goal_func;
#if defined(LIBMESH_HAVE_PETSC)
  std::unique_ptr<libMesh::KokkosPetscAssemblyPlan> petsc_plan;
#endif
};

void
destroy_hilbert_system_kokkos_state(HilbertSystemKokkosState * state)
{
  delete state;
}

HilbertSystemKokkosState &
HilbertSystem::kokkos_state()
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::get Kokkos state");

  if (!this->_kokkos_state)
    this->_kokkos_state.reset(new HilbertSystemKokkosState);

  return *this->_kokkos_state;
}

HilbertSystemKokkosState *
HilbertSystem::maybe_kokkos_state()
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::maybe Kokkos state");

  return this->_kokkos_state.get();
}

const HilbertSystemKokkosState *
HilbertSystem::maybe_kokkos_state() const
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::maybe const Kokkos state");

  return this->_kokkos_state.get();
}

const libMesh::Kokkos::KokkosParsedFunction<libMesh::Number> *
HilbertSystem::ensure_kokkos_goal_func()
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::ensure analytic goal function");

  auto & state = this->kokkos_state();
  return ensure_kokkos_goal_cache<libMesh::ParsedFunction<libMesh::Number>>(
      state.goal_func,
      _analytic_goal_func,
      [this](const auto & parsed_goal)
          -> std::unique_ptr<libMesh::Kokkos::KokkosParsedFunction<libMesh::Number>>
      {
#ifdef LIBMESH_HAVE_FPARSER
        const auto program_bundle = parsed_goal.build_program_bundle();
        if (libMesh::kokkos_l2_env_enabled("LIBMESH_KOKKOS_L2_DEBUG"))
        {
          libMesh::debug_parsed_program(this->comm(), "analytic value", program_bundle.value);
          libMesh::debug_parsed_program(this->comm(), "analytic dx", program_bundle.dx);
#if LIBMESH_DIM > 1
          libMesh::debug_parsed_program(this->comm(), "analytic dy", program_bundle.dy);
#endif
#if LIBMESH_DIM > 2
          libMesh::debug_parsed_program(this->comm(), "analytic dz", program_bundle.dz);
#endif
          libMesh::debug_parsed_program(this->comm(), "analytic dt", program_bundle.dt);
        }
        return std::make_unique<libMesh::Kokkos::KokkosParsedFunction<libMesh::Number>>(
            program_bundle);
#else
        libmesh_ignore(parsed_goal);
        return nullptr;
#endif
      });
}

bool
HilbertSystem::needs_exact_kokkos_fem_goal_context()
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::check exact FEM goal context");

  return input_system && dynamic_cast<ParsedFEMFunction<Number> *>(_goal_func.get()) &&
         this->ensure_kokkos_fem_goal_func();
}

bool
HilbertSystem::kokkos_exact_analytic_goal_host_assembly(FEMContext & c,
                                                            const bool request_jacobian,
                                                            DenseSubVector<Number> & F,
                                                            DenseSubMatrix<Number> & K)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::try exact analytic host assembly");

  if (const auto * kokkos_goal = this->ensure_kokkos_goal_func())
  {
    detail::HostHilbertFEAccess fe(c, 0, _hilbert_order);
    auto solution = detail::make_hilbert_solution_access(
        fe, c.get_elem_solution(0), c.get_elem_solution_derivative());
    detail::HostHilbertAccumulator accum(F, K);
    const auto parsed_goal = kokkos_goal->with_time(this->time);
    auto goal =
        detail::make_hilbert_analytic_goal_access(parsed_goal, parsed_goal.gradient_function());
    detail::assemble_hilbert_element(fe, solution, goal, request_jacobian, _hilbert_order, accum);
    return true;
  }

  return false;
}

const libMesh::Kokkos::KokkosParsedFEMFunction<libMesh::Number> *
HilbertSystem::ensure_kokkos_fem_goal_func()
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::ensure FEM goal function");

  auto & state = this->kokkos_state();
  return ensure_kokkos_goal_cache<libMesh::ParsedFEMFunction<libMesh::Number>>(
      state.fem_goal_func,
      _goal_func,
      [](const auto & parsed_goal)
          -> std::unique_ptr<libMesh::Kokkos::KokkosParsedFEMFunction<libMesh::Number>>
      {
#ifdef LIBMESH_HAVE_FPARSER
        const auto program_bundle = parsed_goal.build_program_bundle();
        if (!program_bundle.supports_kokkos_value_goal() ||
            program_bundle.value_variable_numbers.size() > kokkos_parsed_fem_max_fields)
          return nullptr;

        return std::make_unique<libMesh::Kokkos::KokkosParsedFEMFunction<libMesh::Number>>(
            program_bundle);
#else
        libmesh_ignore(parsed_goal);
        return nullptr;
#endif
      });
}

bool
HilbertSystem::kokkos_exact_fem_goal_host_assembly(FEMContext & c,
                                                       const bool request_jacobian,
                                                       DenseSubVector<Number> & F,
                                                       DenseSubMatrix<Number> & K)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::try exact FEM host assembly");

#if defined(LIBMESH_HAVE_PETSC)
  if (!input_system)
    return false;

  if (const auto * kokkos_fem_goal = this->ensure_kokkos_fem_goal_func())
    return assemble_host_exact_parsed_fem_goal_element(
        *this, c, request_jacobian, F, K, kokkos_fem_goal->with_time(this->time));
#else
  libmesh_ignore(c, request_jacobian, F, K);
#endif

  return false;
}

void
HilbertSystem::reset_kokkos_goal_cache()
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::reset goal cache");

  if (auto * state = this->maybe_kokkos_state())
  {
    state->goal_func.reset();
    state->fem_goal_func.reset();
  }
}

KokkosPetscAssemblyPlan *
HilbertSystem::ensure_kokkos_petsc_plan(bool * rebuilt)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::ensure PETSc assembly plan");

  auto & state = this->kokkos_state();
  const auto & geometry_cache = this->get_mesh().get_kokkos_geometry_cache();
  const auto * dof_index_cache = this->get_dof_map().get_kokkos_dof_index_cache(0);
  if (!dof_index_cache)
    return nullptr;

  if (state.petsc_plan &&
      kokkos_petsc_plan_matches(*this, geometry_cache, *dof_index_cache, *state.petsc_plan))
  {
    if (rebuilt)
      *rebuilt = false;
    return state.petsc_plan.get();
  }

  auto plan = std::make_unique<KokkosPetscAssemblyPlan>();
  if (!build_kokkos_petsc_assembly_plan(*this, *plan))
    return nullptr;

  state.petsc_plan = std::move(plan);
  if (rebuilt)
    *rebuilt = true;
  return state.petsc_plan.get();
}

bool
HilbertSystem::kokkos_element_assembly(FEMContext & c,
                                           const bool request_jacobian,
                                           DenseSubVector<Number> & F,
                                           DenseSubMatrix<Number> & K)
{
  KokkosProfilingRegion kokkos_region("L2Kokkos::try element assembly");

#if defined(LIBMESH_HAVE_PETSC)
  const Elem & elem = c.get_elem();
  const unsigned int quadrature_order = cast_int<unsigned int>(c.get_element_qrule().get_order());
  KokkosElementAssemblyState state;
  if (!build_kokkos_element_state(*this, elem, state))
    return false;
  const FEShapeKey shape_key =
      make_hilbert_shape_key(state.elem_type, state.elem_p_level, this->variable_type(0));

  if (const auto * kokkos_goal = this->ensure_kokkos_goal_func();
      kokkos_goal && assemble_kokkos_hilbert_element(shape_key,
                                                     quadrature_order,
                                                     _hilbert_order,
                                                     c.get_elem_solution_derivative(),
                                                     kokkos_goal->with_time(this->time),
                                                     state,
                                                     request_jacobian,
                                                     F,
                                                     K))
    return true;

  if (!input_system)
    return false;

  if (const auto * kokkos_goal = this->ensure_kokkos_fem_goal_func();
      kokkos_goal && assemble_kokkos_hilbert_fem_goal_element(shape_key,
                                                              quadrature_order,
                                                              _hilbert_order,
                                                              c.get_elem_solution_derivative(),
                                                              *input_system,
                                                              kokkos_goal->with_time(this->time),
                                                              state,
                                                              request_jacobian,
                                                              F,
                                                              K))
    return true;

  return false;
#else
  libmesh_ignore(c, request_jacobian, F, K);
  return false;
#endif
}

#if defined(LIBMESH_HAVE_PETSC)
bool
HilbertSystem::kokkos_petsc_solve()
{
  LOG_SCOPE("L2Kokkos::solve", "HilbertSystem");
  KokkosProfilingRegion kokkos_region("L2Kokkos::solve");

  using clock = std::chrono::steady_clock;
  auto * petsc_matrix = dynamic_cast<PetscMatrixBase<Number> *>(this->matrix);
  auto * petsc_rhs = dynamic_cast<PetscVector<Number> *>(this->rhs);
  auto * petsc_solution = dynamic_cast<PetscVector<Number> *>(this->solution.get());

  const bool analytic_goal = static_cast<bool>(_analytic_goal_func);
  const auto * fem_goal = this->ensure_kokkos_fem_goal_func();

  libmesh_error_msg_if(!petsc_matrix || !petsc_rhs || !petsc_solution,
                       "HilbertSystem Kokkos PETSc COO assembly requires PETSc-backed matrix, "
                       "RHS, and solution objects.");
  libmesh_error_msg_if(!(analytic_goal || fem_goal),
                       "HilbertSystem Kokkos PETSc COO assembly requires an analytic goal "
                       "or a parsed FEM goal with Kokkos support.");
  libmesh_error_msg_if(!is_petsc_kokkos_vec(this->comm(), petsc_rhs->vec()) ||
                           !is_petsc_kokkos_vec(this->comm(), petsc_solution->vec()),
                       "HilbertSystem Kokkos PETSc COO assembly requires PETSc Kokkos vectors. "
                       "Run with -vec_type kokkos.");

  this->_last_kokkos_timing = {};
  const auto total_start = clock::now();
  {
    LOG_SCOPE("L2Kokkos::zero PETSc objects", "HilbertSystem");
    KokkosProfilingRegion zero_region("L2Kokkos::zero PETSc objects");
    petsc_matrix->zero();
    petsc_rhs->zero();
    petsc_solution->zero();
  }

  bool rebuilt_plan = false;
  const auto plan_start = clock::now();
  auto * plan = [&]()
  {
    LOG_SCOPE("L2Kokkos::ensure assembly plan", "HilbertSystem");
    KokkosProfilingRegion plan_region("L2Kokkos::ensure assembly plan");
    return this->ensure_kokkos_petsc_plan(&rebuilt_plan);
  }();
  libmesh_error_msg_if(!plan,
                       "HilbertSystem Kokkos PETSc COO assembly could not build a supported "
                       "assembly plan for the current FE/mapping/quadrature configuration.");
  libmesh_error_msg_if(analytic_goal && _hilbert_order > 0 && !_analytic_goal_grad,
                       "HilbertSystem Kokkos PETSc COO assembly requires an analytic goal "
                       "gradient for Hilbert orders greater than zero.");
  const auto plan_stop = clock::now();
  this->_last_kokkos_timing.plan_seconds =
      rebuilt_plan
          ? std::chrono::duration_cast<std::chrono::duration<Real>>(plan_stop - plan_start).count()
          : 0.;

  const auto assembly_start = clock::now();
  auto assembly_path = HilbertSystem::KokkosAssemblyPath::none;
  {
    LOG_SCOPE("L2Kokkos::assemble global system", "HilbertSystem");
    KokkosProfilingRegion assembly_region("L2Kokkos::assemble global system");
    if (!assemble_kokkos_petsc_global_system(*this,
                                             *plan,
                                             analytic_goal,
                                             _analytic_goal_func.get(),
                                             _analytic_goal_grad.get(),
                                             analytic_goal ? this->ensure_kokkos_goal_func()
                                                           : nullptr,
                                             fem_goal,
                                             *petsc_matrix,
                                             *petsc_rhs,
                                             assembly_path,
                                             &this->_last_kokkos_timing))
      libmesh_error_msg("HilbertSystem Kokkos PETSc COO assembly failed.");
  }
  const auto assembly_stop = clock::now();
  this->_last_kokkos_timing.assembly_seconds =
      std::chrono::duration_cast<std::chrono::duration<Real>>(assembly_stop - assembly_start)
          .count();
  this->_last_kokkos_timing.assembly_path = assembly_path;

  {
    LOG_SCOPE("L2Kokkos::close PETSc objects", "HilbertSystem");
    KokkosProfilingRegion close_region("L2Kokkos::close PETSc objects");
    petsc_matrix->close();
    petsc_rhs->close();
    petsc_solution->close();
  }

  LinearSolver<Number> * solver = this->get_linear_solver();
  {
    LOG_SCOPE("L2Kokkos::init PETSc solver", "HilbertSystem");
    KokkosProfilingRegion init_region("L2Kokkos::init PETSc solver");
    if (this->prefix_with_name())
      solver->init(this->prefix().c_str());
    else
      solver->init();
  }

  const auto [maxlinearits, linear_tol] = this->get_linear_solve_parameters();
  const auto solve_start = clock::now();
  {
    LOG_SCOPE("L2Kokkos::PETSc solve", "HilbertSystem");
    KokkosProfilingRegion solve_region("PETSc::KSPSolve");
    solver->solve(*this->matrix, *this->solution, *this->rhs, linear_tol, maxlinearits);
  }
  const auto solve_stop = clock::now();
  this->_last_kokkos_timing.solve_seconds =
      std::chrono::duration_cast<std::chrono::duration<Real>>(solve_stop - solve_start).count();

  this->update();
  this->mesh_position_set();
  const auto total_stop = clock::now();
  this->_last_kokkos_timing.total_seconds =
      std::chrono::duration_cast<std::chrono::duration<Real>>(total_stop - total_start).count();
  return true;
}
#endif

#endif // LIBMESH_HAVE_KOKKOS && LIBMESH_HAVE_PETSC && !LIBMESH_USE_COMPLEX_NUMBERS
