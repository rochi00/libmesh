BUILT_SOURCES = exact_solution.C
exact_solution.C: $(top_srcdir)/examples/transient/transient_ex1/exact_solution.C
	$(AM_V_GEN)$(LN_S) -f $< $@

DISTCLEANFILES = $(BUILT_SOURCES)

example_name  = adaptivity_ex2
check_SCRIPTS = run.sh
install_dir   = $(examples_install_path)/adaptivity/ex2
data          = adaptivity_ex2.C adaptivity_ex2_kokkos.C adaptivity_ex2_kokkos.K exact_solution.C mesh.xda run.sh run_kokkos.sh
sources       = adaptivity_ex2.C exact_solution.C mesh.xda run.sh

CLEANFILES = solution_read_in.gmv saved_solution.xda saved_solution.gmv saved_mesh.xda out.gmv.* \
  out-kokkos.e.* saved_solution-kokkos.e solution_read_in-kokkos.e \
  adaptivity_ex2_kokkos_dbg.o adaptivity_ex2_kokkos_devel.o \
  adaptivity_ex2_kokkos_opt.o adaptivity_ex2_kokkos_prof.o \
  adaptivity_ex2_kokkos_oprof.o

# also need to link mesh.xda for VPATH builds
if LIBMESH_VPATH_BUILD
  BUILT_SOURCES += .linkstamp
.linkstamp:
	-rm -f mesh.xda && $(LN_S) -f $(srcdir)/mesh.xda .
	$(AM_V_GEN)touch .linkstamp

  CLEANFILES += mesh.xda .linkstamp
endif

##############################################
# include common example environment
include $(top_srcdir)/examples/Make.common

# Keep the CPU example and the Kokkos port as separate executables.  In
# particular, adaptivity_ex2.C is never compiled into an example-kokkos-*
# target, and the .K translation unit is always compiled with KOKKOS_CXX.
if LIBMESH_ENABLE_KOKKOS
if LIBMESH_ENABLE_COMPLEX
else

check_SCRIPTS += run_kokkos.sh

KOKKOS_ADAPTIVITY_EX2_DEPS = adaptivity_ex2_kokkos.K \
  $(top_srcdir)/include/gpu/kokkos_fe_evaluator.h \
  $(top_srcdir)/include/gpu/kokkos_fe_map.h \
  $(top_srcdir)/include/gpu/kokkos_quadrature.h

adaptivity_ex2_kokkos_dbg.o: KOKKOS_ADAPTIVITY_EX2_METHOD_CPPFLAGS = $(CPPFLAGS_DBG)
adaptivity_ex2_kokkos_devel.o: KOKKOS_ADAPTIVITY_EX2_METHOD_CPPFLAGS = $(CPPFLAGS_DEVEL)
adaptivity_ex2_kokkos_opt.o: KOKKOS_ADAPTIVITY_EX2_METHOD_CPPFLAGS = $(CPPFLAGS_OPT)
adaptivity_ex2_kokkos_prof.o: KOKKOS_ADAPTIVITY_EX2_METHOD_CPPFLAGS = $(CPPFLAGS_PROF)
adaptivity_ex2_kokkos_oprof.o: KOKKOS_ADAPTIVITY_EX2_METHOD_CPPFLAGS = $(CPPFLAGS_OPROF)
adaptivity_ex2_kokkos_%.o: $(KOKKOS_ADAPTIVITY_EX2_DEPS)
	$(KOKKOS_CXX) $(KOKKOS_CXXFLAGS) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	  $(KOKKOS_ADAPTIVITY_EX2_METHOD_CPPFLAGS) $(AM_CPPFLAGS) -I$(top_srcdir)/include $(CPPFLAGS) $(KOKKOS_MPI_CPPFLAGS) $(KOKKOS_CPPFLAGS) \
	  -c -o $@ `test -f '$<' || echo '$(srcdir)/'`$<

if LIBMESH_DBG_MODE
check_PROGRAMS += example-kokkos-dbg
example_kokkos_dbg_SOURCES = adaptivity_ex2_kokkos.C
example_kokkos_dbg_CPPFLAGS = $(CPPFLAGS_DBG) $(AM_CPPFLAGS)
example_kokkos_dbg_CXXFLAGS = $(CXXFLAGS_DBG)
example_kokkos_dbg_LDFLAGS = $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)
example_kokkos_dbg_LDADD = adaptivity_ex2_kokkos_dbg.o $(top_builddir)/libmesh_dbg.la $(KOKKOS_LIBS)
example_kokkos_dbg_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_kokkos_dbg_LDFLAGS) $(LDFLAGS) -o $@
endif

if LIBMESH_DEVEL_MODE
check_PROGRAMS += example-kokkos-devel
example_kokkos_devel_SOURCES = adaptivity_ex2_kokkos.C
example_kokkos_devel_CPPFLAGS = $(CPPFLAGS_DEVEL) $(AM_CPPFLAGS)
example_kokkos_devel_CXXFLAGS = $(CXXFLAGS_DEVEL)
example_kokkos_devel_LDFLAGS = $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)
example_kokkos_devel_LDADD = adaptivity_ex2_kokkos_devel.o $(top_builddir)/libmesh_devel.la $(KOKKOS_LIBS)
example_kokkos_devel_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_kokkos_devel_LDFLAGS) $(LDFLAGS) -o $@
endif

if LIBMESH_OPT_MODE
check_PROGRAMS += example-kokkos-opt
example_kokkos_opt_SOURCES = adaptivity_ex2_kokkos.C
example_kokkos_opt_CPPFLAGS = $(CPPFLAGS_OPT) $(AM_CPPFLAGS)
example_kokkos_opt_CXXFLAGS = $(CXXFLAGS_OPT)
example_kokkos_opt_LDFLAGS = $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)
example_kokkos_opt_LDADD = adaptivity_ex2_kokkos_opt.o $(top_builddir)/libmesh_opt.la $(KOKKOS_LIBS)
example_kokkos_opt_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_kokkos_opt_LDFLAGS) $(LDFLAGS) -o $@
endif

if LIBMESH_PROF_MODE
check_PROGRAMS += example-kokkos-prof
example_kokkos_prof_SOURCES = adaptivity_ex2_kokkos.C
example_kokkos_prof_CPPFLAGS = $(CPPFLAGS_PROF) $(AM_CPPFLAGS)
example_kokkos_prof_CXXFLAGS = $(CXXFLAGS_PROF)
example_kokkos_prof_LDFLAGS = $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)
example_kokkos_prof_LDADD = adaptivity_ex2_kokkos_prof.o $(top_builddir)/libmesh_prof.la $(KOKKOS_LIBS)
example_kokkos_prof_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_kokkos_prof_LDFLAGS) $(LDFLAGS) -o $@
endif

if LIBMESH_OPROF_MODE
check_PROGRAMS += example-kokkos-oprof
example_kokkos_oprof_SOURCES = adaptivity_ex2_kokkos.C
example_kokkos_oprof_CPPFLAGS = $(CPPFLAGS_OPROF) $(AM_CPPFLAGS)
example_kokkos_oprof_CXXFLAGS = $(CXXFLAGS_OPROF)
example_kokkos_oprof_LDFLAGS = $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)
example_kokkos_oprof_LDADD = adaptivity_ex2_kokkos_oprof.o $(top_builddir)/libmesh_oprof.la $(KOKKOS_LIBS)
example_kokkos_oprof_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_kokkos_oprof_LDFLAGS) $(LDFLAGS) -o $@
endif

endif
endif
