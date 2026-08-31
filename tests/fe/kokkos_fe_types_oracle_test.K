// GPU kernel oracle tests for libMesh::Kokkos FE type helpers.
// The test suite covers:
//   A. get_side_topology() against libMesh side topology, with the 1D
//      NODEELEM -> EDGE2 surrogate used by the Kokkos path.
//   B. class_from_topology() against a class inferred from libMesh
//      first-order LAGRANGE FE spaces.
//   C. n_dofs(FEShapeKey) for Kokkos-supported exact LAGRANGE keys against
//      libMesh::FEInterface::n_dofs().
//   D. n_dofs(FEShapeKey) for Kokkos-supported MONOMIAL keys against
//      libMesh::FEInterface::n_dofs().
//   E. n_dofs(FEShapeKey) for Kokkos-supported SIDE_HIERARCHIC trace keys
//      against libMesh::FEInterface::n_dofs().
//   F. supports_* predicates agree on the current Kokkos evaluator support
//      boundary, including trace-only SIDE_HIERARCHIC keys.
//   G. MeshBase::KokkosGeometryCache side/edge/face metadata against Elem
//      oracles.
//
// Returns 0 on success, non-zero on failure.

#include "gpu/kokkos_fe_types.h"

#include "libmesh/elem.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_type.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_base.h"
#include "libmesh/mesh_base_kokkos.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/utility.h"

#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "timpi/communicator.h"

// Avoid conflicting complex operators between CUDA and PETSc
#define PETSC_SKIP_CXX_COMPLEX_FIX 1
#include <Kokkos_Core.hpp>
#undef __CUDACC_VER__

#include "kokkos_fe_oracle_test_utils.h"

#include <cstdio>
#include <memory>
#include <string>

namespace libMeshTest
{
namespace KokkosFETypesOracle
{

using kokkos_test_utils::build_reference_elem;

namespace
{

struct side_topology_case
{
  libMesh::ElemType parent;
  libMesh::ElemType expected;
};

struct class_from_topology_case
{
  libMesh::ElemType    topo;
  libMesh::FEElemClass expected;
};

struct n_dof_case
{
  libMesh::Kokkos::FEShapeKey key;
  unsigned int                expected;
};

struct support_case
{
  libMesh::Kokkos::FEShapeKey key;
  bool                        expected;
};

static libMesh::ElemType
host_side_topology_oracle(libMesh::ElemType parent_type)
{
  return libMesh::side_topology_or_invalid(parent_type);
}

static libMesh::FEElemClass
host_class_from_topology_oracle(libMesh::ElemType topo)
{
  return libMesh::class_from_topology_or_invalid(topo);
}

static unsigned int
host_n_dofs_oracle(libMesh::Kokkos::FEShapeKey key)
{
  auto elem = build_reference_elem(key.elem_type);
  return libMesh::FEInterface::n_dofs(libMesh::FEType(key.order, key.family), 0, elem.get());
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test 1: get_side_topology() against libMesh side topology.
// ---------------------------------------------------------------------------
static int
test_side_topology()
{
  static const libMesh::ElemType parents[] = {
    libMesh::EDGE2,
    libMesh::EDGE3,
    libMesh::EDGE4,
    libMesh::TRI3,
    libMesh::TRI7,
    libMesh::QUAD4,
    libMesh::TRI6,
    libMesh::QUAD8,
    libMesh::QUAD9,
    libMesh::TET4,
    libMesh::HEX8,
    libMesh::TET10,
    libMesh::TET14,
    libMesh::HEX20,
    libMesh::HEX27
  };
  constexpr int n_cases = sizeof(parents) / sizeof(parents[0]);

  Kokkos::View<side_topology_case *> d_cases(std::string("side_cases"), n_cases);
  {
    auto h = Kokkos::create_mirror_view(d_cases);
    for (int i = 0; i < n_cases; ++i)
    {
      h(i).parent = parents[i];
      h(i).expected = host_side_topology_oracle(parents[i]);
    }
    Kokkos::deep_copy(d_cases, h);
  }

  Kokkos::View<int> d_fail(std::string("side_fail"));
  Kokkos::deep_copy(d_fail, 0);

  Kokkos::parallel_for(
    n_cases,
    KOKKOS_LAMBDA(int i) {
      using namespace libMesh::Kokkos;
      if (get_side_topology(d_cases(i).parent) != d_cases(i).expected)
        Kokkos::atomic_add(&d_fail(), 1);
    });
  Kokkos::fence();

  int fail = 0;
  Kokkos::deep_copy(fail, d_fail);
  return fail;
}

// ---------------------------------------------------------------------------
// Test 2: class_from_topology() against libMesh FE oracle.
// ---------------------------------------------------------------------------
static int
test_class_from_topology()
{
  static const libMesh::ElemType topos[] = {
    libMesh::EDGE2,
    libMesh::EDGE3,
    libMesh::EDGE4,
    libMesh::TRI3,
    libMesh::TRI6,
    libMesh::TRI7,
    libMesh::QUAD4,
    libMesh::QUAD8,
    libMesh::QUAD9,
    libMesh::TET4,
    libMesh::TET10,
    libMesh::TET14,
    libMesh::HEX8,
    libMesh::HEX20,
    libMesh::HEX27,
    libMesh::PRISM6,
    libMesh::PRISM15,
    libMesh::PRISM18,
    libMesh::PRISM20,
    libMesh::PRISM21,
    libMesh::PYRAMID5,
    libMesh::PYRAMID13,
    libMesh::PYRAMID14,
    libMesh::PYRAMID18
  };
  constexpr int n_cases = sizeof(topos) / sizeof(topos[0]);

  Kokkos::View<class_from_topology_case *> d_cases(std::string("class_cases"), n_cases);
  {
    auto h = Kokkos::create_mirror_view(d_cases);
    for (int i = 0; i < n_cases; ++i)
    {
      h(i).topo = topos[i];
      h(i).expected = host_class_from_topology_oracle(topos[i]);
    }
    Kokkos::deep_copy(d_cases, h);
  }

  Kokkos::View<int> d_fail(std::string("class_fail"));
  Kokkos::deep_copy(d_fail, 0);

  Kokkos::parallel_for(
    n_cases,
    KOKKOS_LAMBDA(int i) {
      using namespace libMesh::Kokkos;
      if (class_from_topology(d_cases(i).topo) != d_cases(i).expected)
        Kokkos::atomic_add(&d_fail(), 1);
    });
  Kokkos::fence();

  int fail = 0;
  Kokkos::deep_copy(fail, d_fail);
  return fail;
}

// ---------------------------------------------------------------------------
// Test 3: n_dofs() for Kokkos-supported exact LAGRANGE keys against
// libMesh FEInterface.
// ---------------------------------------------------------------------------
static int
test_n_dofs_lagrange()
{
  using libMesh::Kokkos::FEShapeKey;

  static const FEShapeKey keys[] = {
    { libMesh::LAGRANGE, libMesh::EDGE2,     libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::EDGE3,     libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::EDGE3,     libMesh::SECOND   },
    { libMesh::LAGRANGE, libMesh::EDGE4,     libMesh::FIRST    },

    { libMesh::LAGRANGE, libMesh::TRI3,      libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::TRI6,      libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::TRI6,      libMesh::SECOND   },
    { libMesh::LAGRANGE, libMesh::TRI7,      libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::TRI7,      libMesh::SECOND   },
    { libMesh::LAGRANGE, libMesh::QUAD4,     libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::QUAD8,     libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::QUAD8,     libMesh::SECOND   },
    { libMesh::LAGRANGE, libMesh::QUAD9,     libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::QUAD9,     libMesh::SECOND   },

    { libMesh::LAGRANGE, libMesh::TET4,      libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::TET10,     libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::TET10,     libMesh::SECOND   },
    { libMesh::LAGRANGE, libMesh::TET14,     libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::TET14,     libMesh::SECOND   },
    { libMesh::LAGRANGE, libMesh::HEX8,      libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::HEX20,     libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::HEX20,     libMesh::SECOND   },
    { libMesh::LAGRANGE, libMesh::HEX27,     libMesh::FIRST    },
    { libMesh::LAGRANGE, libMesh::HEX27,     libMesh::SECOND   }
  };
  constexpr int n_cases = sizeof(keys) / sizeof(keys[0]);

  Kokkos::View<n_dof_case *> d_cases(std::string("lagrange_cases"), n_cases);
  {
    auto h = Kokkos::create_mirror_view(d_cases);
    for (int i = 0; i < n_cases; ++i)
    {
      h(i).key = keys[i];
      h(i).expected = host_n_dofs_oracle(keys[i]);
    }
    Kokkos::deep_copy(d_cases, h);
  }

  Kokkos::View<int> d_fail(std::string("lagrange_fail"));
  Kokkos::deep_copy(d_fail, 0);

  Kokkos::parallel_for(
    n_cases,
    KOKKOS_LAMBDA(int i) {
      using namespace libMesh::Kokkos;
      if (n_dofs(d_cases(i).key) != d_cases(i).expected)
        Kokkos::atomic_add(&d_fail(), 1);
    });
  Kokkos::fence();

  int fail = 0;
  Kokkos::deep_copy(fail, d_fail);
  return fail;
}

// ---------------------------------------------------------------------------
// Test 4: n_dofs() for Kokkos-supported MONOMIAL keys against libMesh
// FEInterface.
// ---------------------------------------------------------------------------
static int
test_n_dofs_monomial()
{
  using libMesh::Kokkos::FEShapeKey;

  static const libMesh::Order orders[] = {
    libMesh::CONSTANT,
    libMesh::FIRST,
    libMesh::SECOND,
    libMesh::THIRD,
    libMesh::FOURTH,
    libMesh::FIFTH
  };
  static const libMesh::ElemType higher_dim_topos[] = {
    libMesh::TRI7,
    libMesh::QUAD9,
    libMesh::TET14,
    libMesh::HEX27,
    libMesh::PRISM21
  };

  constexpr int n_cases = sizeof(orders) / sizeof(orders[0]) *
                          (2 + sizeof(higher_dim_topos) / sizeof(higher_dim_topos[0]));

  Kokkos::View<n_dof_case *> d_cases(std::string("monomial_cases"), n_cases);
  {
    auto h = Kokkos::create_mirror_view(d_cases);
    int i = 0;
    for (const auto order : orders)
    {
      const libMesh::ElemType edge_topo =
        (order <= libMesh::THIRD) ? libMesh::EDGE4 : libMesh::EDGE3;
      const libMesh::ElemType pyramid_topo =
        (order <= libMesh::THIRD) ? libMesh::PYRAMID18 : libMesh::PYRAMID14;

      h(i).key = { libMesh::MONOMIAL, edge_topo, order };
      h(i).expected = host_n_dofs_oracle(h(i).key);
      ++i;

      h(i).key = { libMesh::MONOMIAL, pyramid_topo, order };
      h(i).expected = host_n_dofs_oracle(h(i).key);
      ++i;

      for (const auto topo : higher_dim_topos)
      {
        h(i).key = { libMesh::MONOMIAL, topo, order };
        h(i).expected = host_n_dofs_oracle(h(i).key);
        ++i;
      }
    }
    Kokkos::deep_copy(d_cases, h);
  }

  Kokkos::View<int> d_fail(std::string("monomial_fail"));
  Kokkos::deep_copy(d_fail, 0);

  Kokkos::parallel_for(
    n_cases,
    KOKKOS_LAMBDA(int i) {
      using namespace libMesh::Kokkos;
      if (n_dofs(d_cases(i).key) != d_cases(i).expected)
        Kokkos::atomic_add(&d_fail(), 1);
    });
  Kokkos::fence();

  int fail = 0;
  Kokkos::deep_copy(fail, d_fail);
  return fail;
}

// ---------------------------------------------------------------------------
// Test 5: n_dofs() for Kokkos-supported SIDE_HIERARCHIC trace keys against
// FEInterface.
// ---------------------------------------------------------------------------
static int
test_n_dofs_side_hierarchic()
{
  using libMesh::Kokkos::FEShapeKey;

  static const FEShapeKey keys[] = {
    { libMesh::SIDE_HIERARCHIC, libMesh::TRI6,  libMesh::FIRST },
    { libMesh::SIDE_HIERARCHIC, libMesh::TRI7,  libMesh::FIRST },
    { libMesh::SIDE_HIERARCHIC, libMesh::QUAD8, libMesh::FIRST },
    { libMesh::SIDE_HIERARCHIC, libMesh::QUAD9, libMesh::FIRST }
  };
  constexpr int n_cases = sizeof(keys) / sizeof(keys[0]);

  Kokkos::View<n_dof_case *> d_cases(std::string("side_hierarchic_cases"), n_cases);
  {
    auto h = Kokkos::create_mirror_view(d_cases);
    for (int i = 0; i < n_cases; ++i)
    {
      h(i).key = keys[i];
      h(i).expected = host_n_dofs_oracle(keys[i]);
    }
    Kokkos::deep_copy(d_cases, h);
  }

  Kokkos::View<int> d_fail(std::string("side_hierarchic_fail"));
  Kokkos::deep_copy(d_fail, 0);

  Kokkos::parallel_for(
    n_cases,
    KOKKOS_LAMBDA(int i) {
      using namespace libMesh::Kokkos;
      if (n_dofs(d_cases(i).key) != d_cases(i).expected)
        Kokkos::atomic_add(&d_fail(), 1);
    });
  Kokkos::fence();

  int fail = 0;
  Kokkos::deep_copy(fail, d_fail);
  return fail;
}

// ---------------------------------------------------------------------------
// Test 6: support predicates agree on the Kokkos evaluator boundary.
// ---------------------------------------------------------------------------
static int
test_support_contract()
{
  using libMesh::Kokkos::FEShapeKey;

  static const support_case cases[] = {
    { { libMesh::LAGRANGE, libMesh::EDGE2,     libMesh::FIRST    }, true  },
    { { libMesh::LAGRANGE, libMesh::EDGE3,     libMesh::SECOND   }, true  },
    { { libMesh::LAGRANGE, libMesh::TRI7,      libMesh::SECOND   }, true  },
    { { libMesh::LAGRANGE, libMesh::QUAD9,     libMesh::SECOND   }, true  },
    { { libMesh::LAGRANGE, libMesh::TET14,     libMesh::SECOND   }, true  },
    { { libMesh::LAGRANGE, libMesh::HEX27,     libMesh::SECOND   }, true  },
    { { libMesh::LAGRANGE, libMesh::NODEELEM,  libMesh::CONSTANT }, false },
    { { libMesh::LAGRANGE, libMesh::NODEELEM,  libMesh::FIRST    }, false },
    { { libMesh::LAGRANGE, libMesh::EDGE4,     libMesh::THIRD    }, true  },
    { { libMesh::LAGRANGE, libMesh::TRI7,      libMesh::THIRD    }, false },
    { { libMesh::LAGRANGE, libMesh::TET14,     libMesh::THIRD    }, false },
    { { libMesh::LAGRANGE, libMesh::PRISM6,    libMesh::FIRST    }, false },
    { { libMesh::LAGRANGE, libMesh::PRISM15,   libMesh::SECOND   }, false },
    { { libMesh::LAGRANGE, libMesh::PYRAMID5,  libMesh::FIRST    }, false },
    { { libMesh::LAGRANGE, libMesh::PYRAMID14, libMesh::SECOND   }, false },
    { { libMesh::LAGRANGE, libMesh::PYRAMID18, libMesh::THIRD    }, false },
    { { libMesh::LAGRANGE, libMesh::EDGE2,     libMesh::INVALID_ORDER }, false },
    { { libMesh::MONOMIAL, libMesh::EDGE4,     libMesh::THIRD    }, true  },
    { { libMesh::MONOMIAL, libMesh::TRI7,      libMesh::FIFTH    }, true  },
    { { libMesh::MONOMIAL, libMesh::QUAD9,     libMesh::FIFTH    }, true  },
    { { libMesh::MONOMIAL, libMesh::TET14,     libMesh::FIFTH    }, true  },
    { { libMesh::MONOMIAL, libMesh::PRISM21,   libMesh::FIFTH    }, true  },
    { { libMesh::MONOMIAL, libMesh::PYRAMID14, libMesh::FIFTH    }, true  },
    { { libMesh::MONOMIAL, libMesh::NODEELEM,  libMesh::CONSTANT }, false },
    { { libMesh::MONOMIAL, libMesh::EDGE4,     libMesh::FOURTH   }, false },
    { { libMesh::MONOMIAL, libMesh::PYRAMID18, libMesh::FOURTH   }, false },
    { { libMesh::MONOMIAL, libMesh::TRI7,      libMesh::SIXTH    }, false },
    { { libMesh::MONOMIAL, libMesh::HEX27,     libMesh::SIXTH    }, false },
    { { libMesh::MONOMIAL, libMesh::EDGE2,     libMesh::INVALID_ORDER }, false }
  };
  constexpr int n_cases = sizeof(cases) / sizeof(cases[0]);

  Kokkos::View<support_case *> d_cases(std::string("support_cases"), n_cases);
  {
    auto h = Kokkos::create_mirror_view(d_cases);
    for (int i = 0; i < n_cases; ++i)
      h(i) = cases[i];
    Kokkos::deep_copy(d_cases, h);
  }

  Kokkos::View<int> d_fail(std::string("support_fail"));
  Kokkos::deep_copy(d_fail, 0);

  Kokkos::parallel_for(
    n_cases,
    KOKKOS_LAMBDA(int i) {
      using namespace libMesh::Kokkos;

      const bool shape_supported = supports_shape(d_cases(i).key);
      const bool grad_supported = supports_grad_shape(d_cases(i).key);
      const bool ndofs_supported = supports_n_dofs(d_cases(i).key);

      if (shape_supported != d_cases(i).expected ||
          grad_supported != d_cases(i).expected ||
          ndofs_supported != d_cases(i).expected ||
          shape_supported != grad_supported ||
          shape_supported != ndofs_supported)
        Kokkos::atomic_add(&d_fail(), 1);
    });
  Kokkos::fence();

  int fail = 0;
  Kokkos::deep_copy(fail, d_fail);

  static const support_case trace_cases[] = {
    { { libMesh::SIDE_HIERARCHIC, libMesh::TRI6,  libMesh::FIRST  }, true  },
    { { libMesh::SIDE_HIERARCHIC, libMesh::TRI7,  libMesh::FIRST  }, true  },
    { { libMesh::SIDE_HIERARCHIC, libMesh::QUAD8, libMesh::FIRST  }, true  },
    { { libMesh::SIDE_HIERARCHIC, libMesh::QUAD9, libMesh::FIRST  }, true  },
    { { libMesh::SIDE_HIERARCHIC, libMesh::TRI6,  libMesh::SECOND }, false },
    { { libMesh::SIDE_HIERARCHIC, libMesh::TRI3,  libMesh::FIRST  }, false }
  };
  constexpr int n_trace_cases = sizeof(trace_cases) / sizeof(trace_cases[0]);

  Kokkos::View<support_case *> d_trace_cases(std::string("trace_support_cases"),
                                             n_trace_cases);
  {
    auto h = Kokkos::create_mirror_view(d_trace_cases);
    for (int i = 0; i < n_trace_cases; ++i)
      h(i) = trace_cases[i];
    Kokkos::deep_copy(d_trace_cases, h);
  }

  Kokkos::deep_copy(d_fail, 0);
  Kokkos::parallel_for(
    n_trace_cases,
    KOKKOS_LAMBDA(int i) {
      using namespace libMesh::Kokkos;

      const bool trace_supported = supports_side_trace_shape(d_trace_cases(i).key);
      const bool ndofs_supported = supports_n_dofs(d_trace_cases(i).key);

      if (trace_supported != d_trace_cases(i).expected ||
          ndofs_supported != d_trace_cases(i).expected ||
          supports_shape(d_trace_cases(i).key) ||
          supports_grad_shape(d_trace_cases(i).key))
        Kokkos::atomic_add(&d_fail(), 1);
    });
  Kokkos::fence();

  int trace_fail = 0;
  Kokkos::deep_copy(trace_fail, d_fail);
  fail += trace_fail;

  return fail;
}

// ---------------------------------------------------------------------------
// Test 7: Kokkos geometry cache side metadata against libMesh Elem.
// ---------------------------------------------------------------------------
static int
check_geometry_cache_side_metadata(const libMesh::MeshBase & mesh)
{
  const auto & cache = mesh.get_kokkos_geometry_cache();

  auto h_n_sides =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cache.element_n_sides);
  auto h_n_edges =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cache.element_n_edges);
  auto h_n_faces =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cache.element_n_faces);
  auto h_side_types =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cache.element_side_types);
  auto h_side_n_nodes =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cache.element_side_n_nodes);
  auto h_side_local_nodes =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cache.element_side_local_nodes);
  auto h_edge_orientations = Kokkos::create_mirror_view_and_copy(
    Kokkos::HostSpace(), cache.element_edge_positive_orientations);
  auto h_face_orientations = Kokkos::create_mirror_view_and_copy(
    Kokkos::HostSpace(), cache.element_face_positive_orientations);

  int fail = 0;

  for (std::size_t elem_index = 0; elem_index != cache.host_element_ids.size(); ++elem_index)
    {
      const libMesh::Elem & elem = *mesh.query_elem_ptr(cache.host_element_ids[elem_index]);

      if (h_n_sides(elem_index) != elem.n_sides() ||
          h_n_edges(elem_index) != elem.n_edges() ||
          h_n_faces(elem_index) != elem.n_faces())
        ++fail;

      for (unsigned int side = 0; side != elem.n_sides(); ++side)
        {
          const auto side_nodes = elem.nodes_on_side(side);

          if (h_side_types(elem_index, side) != elem.side_type(side) ||
              h_side_n_nodes(elem_index, side) != side_nodes.size())
            ++fail;

          for (std::size_t n = 0; n != side_nodes.size(); ++n)
            if (h_side_local_nodes(elem_index, side, n) != side_nodes[n])
              ++fail;

          for (unsigned int n = libMesh::cast_int<unsigned int>(side_nodes.size());
               n != cache.max_side_nodes;
               ++n)
            if (h_side_local_nodes(elem_index, side, n) != libMesh::invalid_uint)
              ++fail;
        }

      for (unsigned int side = elem.n_sides(); side != cache.max_sides; ++side)
        {
          if (h_side_types(elem_index, side) != libMesh::INVALID_ELEM ||
              h_side_n_nodes(elem_index, side) != 0)
            ++fail;

          for (unsigned int n = 0; n != cache.max_side_nodes; ++n)
            if (h_side_local_nodes(elem_index, side, n) != libMesh::invalid_uint)
              ++fail;
        }

      for (unsigned int edge = 0; edge != elem.n_edges(); ++edge)
        if (static_cast<bool>(h_edge_orientations(elem_index, edge)) !=
            elem.positive_edge_orientation(edge))
          ++fail;

      for (unsigned int edge = elem.n_edges(); edge != cache.max_edges; ++edge)
        if (h_edge_orientations(elem_index, edge) != 0)
          ++fail;

      for (unsigned int face = 0; face != elem.n_faces(); ++face)
        if (static_cast<bool>(h_face_orientations(elem_index, face)) !=
            elem.positive_face_orientation(face))
          ++fail;

      for (unsigned int face = elem.n_faces(); face != cache.max_faces; ++face)
        if (h_face_orientations(elem_index, face) != 0)
          ++fail;
    }

  return fail;
}

static int
test_geometry_cache_side_metadata()
{
  libMesh::Parallel::Communicator comm(libMesh::GLOBAL_COMM_WORLD);
  int fail = 0;

#if LIBMESH_DIM >= 2
  {
    libMesh::Mesh mesh(comm, 2);
    libMesh::MeshTools::Generation::build_square(
      mesh, 1, 1, 0., 1., 0., 1., libMesh::TRI6);
    mesh.prepare_for_use();
    fail += check_geometry_cache_side_metadata(mesh);
  }
#endif

#if LIBMESH_DIM >= 3
  {
    libMesh::Mesh mesh(comm, 3);
    libMesh::MeshTools::Generation::build_cube(
      mesh, 1, 1, 1, 0., 1., 0., 1., 0., 1., libMesh::HEX27);
    mesh.prepare_for_use();
    fail += check_geometry_cache_side_metadata(mesh);
  }

  {
    libMesh::Mesh mesh(comm, 3);
    libMesh::MeshTools::Generation::build_cube(
      mesh, 1, 1, 1, 0., 1., 0., 1., 0., 1., libMesh::TET10);
    mesh.prepare_for_use();
    fail += check_geometry_cache_side_metadata(mesh);
  }
#endif

  return fail;
}

inline int
run_all_oracles()
{
  int total_fail = 0;

  {
    const int f = test_side_topology();
    std::printf("[side_topology_oracle]     %s  (%d failures)\n", f ? "FAIL" : "PASS", f);
    total_fail += f;
  }
  {
    const int f = test_class_from_topology();
    std::printf("[class_topology_oracle]    %s  (%d failures)\n", f ? "FAIL" : "PASS", f);
    total_fail += f;
  }
  {
    const int f = test_n_dofs_lagrange();
    std::printf("[lagrange_ndofs_oracle]    %s  (%d failures)\n", f ? "FAIL" : "PASS", f);
    total_fail += f;
  }
  {
    const int f = test_n_dofs_monomial();
    std::printf("[monomial_ndofs_oracle]    %s  (%d failures)\n", f ? "FAIL" : "PASS", f);
    total_fail += f;
  }
  {
    const int f = test_n_dofs_side_hierarchic();
    std::printf("[trace_ndofs_oracle]       %s  (%d failures)\n", f ? "FAIL" : "PASS", f);
    total_fail += f;
  }
  {
    const int f = test_support_contract();
    std::printf("[support_contract_oracle]  %s  (%d failures)\n", f ? "FAIL" : "PASS", f);
    total_fail += f;
  }
  {
    const int f = test_geometry_cache_side_metadata();
    std::printf("[geometry_cache_oracle]   %s  (%d failures)\n", f ? "FAIL" : "PASS", f);
    total_fail += f;
  }

  return total_fail ? 1 : 0;
}

} // namespace KokkosFETypesOracle
} // namespace libMeshTest
