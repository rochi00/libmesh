// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// Standalone Kokkos port of adaptivity_ex2.  The original CPU example is kept
// unchanged so that the two implementations can be built and compared.

#include "libmesh/libmesh.h"
#include "libmesh/replicated_mesh.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/transient_system.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/error_vector.h"
#include "libmesh/kelly_error_estimator.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/enum_norm_type.h"
#include "libmesh/enum_solver_package.h"
#include "libmesh/enum_xdr_mode.h"

#include <cmath>
#include <iomanip>
#include <sstream>

using namespace libMesh;

#ifdef LIBMESH_ENABLE_AMR
void assemble_cd_kokkos(EquationSystems &, const std::string &);
void clear_adaptivity_ex2_kokkos_assembly_cache();
#endif

Real adaptivity_ex2_kokkos_exact_solution(Real x, Real y, Real t)
{
  constexpr Real x0 = 0.2;
  constexpr Real y0 = 0.2;
  constexpr Real vx = 0.8;
  constexpr Real vy = 0.8;

  const Real denominator = 0.01 * (4. * t + 1.);
  const Real dx = x - vx * t - x0;
  const Real dy = y - vy * t - y0;
  return std::exp(-(dx * dx + dy * dy) / denominator) / (4. * t + 1.);
}

namespace
{

Number exact_value(const Point & p,
                   const Parameters & parameters,
                   const std::string &,
                   const std::string &)
{
  return adaptivity_ex2_kokkos_exact_solution(
      p(0), p(1), parameters.get<Real>("time"));
}

void init_cd_kokkos(EquationSystems & es,
                    const std::string & libmesh_dbg_var(system_name))
{
  libmesh_assert_equal_to(system_name, "Convection-Diffusion");

  auto & system =
      es.get_system<TransientLinearImplicitSystem>("Convection-Diffusion");
  es.parameters.set<Real>("time") = system.time = 0.;
  system.project_solution(exact_value, nullptr, es.parameters);
}

unsigned int requested_order()
{
  const int order = libMesh::command_line_next("-order", 1);
  libmesh_error_msg_if(order < 1 || order > 2,
                       "adaptivity_ex2 Kokkos currently supports only -order 1 or -order 2");
  return cast_int<unsigned int>(order);
}

} // anonymous namespace

int main(int argc, char ** argv)
{
  LibMeshInit init(argc, argv);

  libmesh_example_requires(libMesh::default_solver_package() != INVALID_SOLVER_PACKAGE,
                           "--enable-petsc, --enable-trilinos, or --enable-eigen");
  libmesh_example_requires(2 <= LIBMESH_DIM, "2D support");

#ifndef LIBMESH_HAVE_KOKKOS
  libmesh_example_requires(false, "--enable-kokkos");
#elif defined(LIBMESH_USE_COMPLEX_NUMBERS)
  libmesh_example_requires(false, "a real-valued libMesh build");
#elif !defined(LIBMESH_ENABLE_AMR)
  libmesh_example_requires(false, "--enable-amr");
#else
  libmesh_example_requires(libMesh::default_solver_package() != TRILINOS_SOLVERS,
                           "--enable-petsc");

  libMesh::out << "Usage:\n"
               << "\t " << argv[0]
               << " -init_timestep 0 -n_timesteps 25 [-n_refinements 5] [-order 1|2]\n"
               << "OR\n"
               << "\t " << argv[0]
               << " -read_solution -init_timestep 26 -n_timesteps 25\n\n"
               << "Running: " << argv[0];
  for (int i = 1; i < argc; ++i)
    libMesh::out << ' ' << argv[i];
  libMesh::out << "\n\n";

  const bool read_solution = libMesh::on_command_line("-read_solution");
  const unsigned int init_timestep =
      libMesh::command_line_next("-init_timestep", libMesh::invalid_uint);
  const unsigned int n_timesteps =
      libMesh::command_line_next("-n_timesteps", libMesh::invalid_uint);
  const unsigned int max_h_level =
      libMesh::command_line_next("-max_h_level", 5);

  libmesh_error_msg_if(init_timestep == libMesh::invalid_uint,
                       "Initial timestep not specified");
  libmesh_error_msg_if(n_timesteps == libMesh::invalid_uint,
                       "Number of timesteps not specified");

  ReplicatedMesh mesh(init.comm());
  EquationSystems equation_systems(mesh);
  MeshRefinement mesh_refinement(mesh);

  if (!read_solution)
  {
    const unsigned int order = requested_order();
    mesh.read("mesh.xda");
    if (order == 2)
      mesh.all_second_order();

    const unsigned int n_refinements =
        libMesh::command_line_next("-n_refinements", 5);
    mesh_refinement.uniformly_refine(n_refinements);
    mesh.print_info();

    auto & system = equation_systems.add_system<TransientLinearImplicitSystem>(
        "Convection-Diffusion");
    system.add_variable("u", static_cast<Order>(order), LAGRANGE);
    system.attach_assemble_function(assemble_cd_kokkos);
    system.attach_init_function(init_cd_kokkos);
    equation_systems.init();
  }
  else
  {
    mesh.read("saved_mesh.xda");
    mesh.print_info();
    equation_systems.read("saved_solution.xda", READ);

    auto & system = equation_systems.get_system<TransientLinearImplicitSystem>(
        "Convection-Diffusion");
    const unsigned int order =
        cast_int<unsigned int>(system.variable_type(0).order.get_order());
    libmesh_error_msg_if(order < 1 || order > 2 ||
                             system.variable_type(0).family != LAGRANGE,
                         "The Kokkos restart path supports only first- and second-order "
                         "LAGRANGE solutions");
    system.update();
    system.attach_assemble_function(assemble_cd_kokkos);

    const Real h1_norm = system.calculate_norm(*system.solution, SystemNorm(H1));
    libMesh::out << "Initial H1 norm = " << h1_norm << "\n\n";
  }

  equation_systems.print_info();
  equation_systems.parameters.set<unsigned int>("linear solver maximum iterations") = 250;
  equation_systems.parameters.set<Real>("linear solver tolerance") = TOLERANCE;
  equation_systems.parameters.set<RealVectorValue>("velocity") = RealVectorValue(0.8, 0.8);
  equation_systems.parameters.set<Real>("diffusivity") = 0.01;

  auto & system = equation_systems.get_system<TransientLinearImplicitSystem>(
      "Convection-Diffusion");

#ifdef LIBMESH_HAVE_EXODUS_API
  if (!read_solution)
    ExodusII_IO(mesh).write_equation_systems("out-kokkos.e.000", equation_systems);
  else
    ExodusII_IO(mesh).write_equation_systems("solution_read_in-kokkos.e",
                                             equation_systems);
#endif

  constexpr Real dt = 0.025;
  system.time = init_timestep * dt;
  const Real refine_fraction =
      libMesh::command_line_next("-refine_fraction", 0.80);
  const Real coarsen_fraction =
      libMesh::command_line_next("-coarsen_fraction", 0.07);

  for (unsigned int t_step = init_timestep;
       t_step < init_timestep + n_timesteps;
       ++t_step)
  {
    system.time += dt;
    equation_systems.parameters.set<Real>("time") = system.time;
    equation_systems.parameters.set<Real>("dt") = dt;

    std::ostringstream progress;
    progress << " Solving Kokkos time step " << std::setw(2) << std::right << t_step
             << ", time=" << std::fixed << std::setw(6) << std::setprecision(3)
             << std::setfill('0') << std::left << system.time << "...";
    libMesh::out << progress.str() << '\n';

    *system.old_local_solution = *system.current_local_solution;

    constexpr unsigned int max_r_steps = 2;
    for (unsigned int r_step = 0; r_step != max_r_steps; ++r_step)
    {
      system.solve();
      const Real h1_norm = system.calculate_norm(*system.solution, SystemNorm(H1));
      libMesh::out << "H1 norm = " << h1_norm << '\n';

      if (r_step + 1 != max_r_steps)
      {
        libMesh::out << "  Refining the mesh...\n";
        ErrorVector error;
        KellyErrorEstimator error_estimator;
        error_estimator.use_unweighted_quadrature_rules = true;
        error_estimator.estimate_error(system, error);

        mesh_refinement.refine_fraction() = refine_fraction;
        mesh_refinement.coarsen_fraction() = coarsen_fraction;
        mesh_refinement.max_h_level() = max_h_level;
        mesh_refinement.flag_elements_by_error_fraction(error);
        mesh_refinement.refine_and_coarsen_elements();
        equation_systems.reinit();
      }
    }

    const unsigned int output_freq =
        libMesh::command_line_next("-output_freq", 10);
    if ((t_step + 1) % output_freq == 0)
    {
      equation_systems.print_info();
#ifdef LIBMESH_HAVE_EXODUS_API
      std::ostringstream filename;
      filename << "out-kokkos.e." << std::setw(3) << std::setfill('0')
               << std::right << t_step + 1;
      ExodusII_IO(mesh).write_equation_systems(filename.str(), equation_systems);
#endif
    }
  }

  if (!read_solution)
  {
    const Real h1_norm = system.calculate_norm(*system.solution, SystemNorm(H1));
    libMesh::out << "Final H1 norm = " << h1_norm << "\n\n";
    mesh.write("saved_mesh.xda");
    equation_systems.write("saved_solution.xda", WRITE);
#ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO(mesh).write_equation_systems("saved_solution-kokkos.e",
                                             equation_systems);
#endif
  }
#endif

#ifdef LIBMESH_ENABLE_AMR
  // Release persistent device views before LibMeshInit finalizes Kokkos.
  clear_adaptivity_ex2_kokkos_assembly_cache();
#endif
  return 0;
}
