example_name  = vector_fe_ex9
check_SCRIPTS = run.sh
install_dir   = $(examples_install_path)/vector_fe/ex9
data          = vector_fe_ex9.C vector_fe_ex9.in vector_fe_ex9_kokkos.K hdg_problem.h hdg_problem.C exact_soln.h run.sh
sources       = vector_fe_ex9.C hdg_problem.C hdg_problem.h exact_soln.h

CLEANFILES = out.e

# also need to link files for VPATH builds
if LIBMESH_VPATH_BUILD
  BUILT_SOURCES = .linkstamp
.linkstamp:
	-rm -f vector_fe_ex9.in && $(LN_S) -f $(srcdir)/vector_fe_ex9.in .
	$(AM_V_GEN)touch .linkstamp

  CLEANFILES += vector_fe_ex9.in .linkstamp
endif

##############################################
# include common example environment
include $(top_srcdir)/examples/Make.common

if LIBMESH_ENABLE_KOKKOS
if LIBMESH_ENABLE_PETSC
if LIBMESH_ENABLE_COMPLEX
KOKKOS_VECTOR_FE_EX9_DBG_OBJECTS =
KOKKOS_VECTOR_FE_EX9_DEVEL_OBJECTS =
KOKKOS_VECTOR_FE_EX9_OPT_OBJECTS =
KOKKOS_VECTOR_FE_EX9_PROF_OBJECTS =
KOKKOS_VECTOR_FE_EX9_OPROF_OBJECTS =
else
KOKKOS_VECTOR_FE_EX9_DBG_OBJECTS = vector_fe_ex9_kokkos_dbg.o
KOKKOS_VECTOR_FE_EX9_DEVEL_OBJECTS = vector_fe_ex9_kokkos_devel.o
KOKKOS_VECTOR_FE_EX9_OPT_OBJECTS = vector_fe_ex9_kokkos_opt.o
KOKKOS_VECTOR_FE_EX9_PROF_OBJECTS = vector_fe_ex9_kokkos_prof.o
KOKKOS_VECTOR_FE_EX9_OPROF_OBJECTS = vector_fe_ex9_kokkos_oprof.o
KOKKOS_VECTOR_FE_EX9_DEPS = vector_fe_ex9_kokkos.K hdg_problem.h $(top_srcdir)/include/gpu/kokkos_hdg_assembly.h

vector_fe_ex9_kokkos_dbg.o: KOKKOS_VECTOR_FE_EX9_METHOD_CPPFLAGS = $(CPPFLAGS_DBG)
vector_fe_ex9_kokkos_devel.o: KOKKOS_VECTOR_FE_EX9_METHOD_CPPFLAGS = $(CPPFLAGS_DEVEL)
vector_fe_ex9_kokkos_opt.o: KOKKOS_VECTOR_FE_EX9_METHOD_CPPFLAGS = $(CPPFLAGS_OPT)
vector_fe_ex9_kokkos_prof.o: KOKKOS_VECTOR_FE_EX9_METHOD_CPPFLAGS = $(CPPFLAGS_PROF)
vector_fe_ex9_kokkos_oprof.o: KOKKOS_VECTOR_FE_EX9_METHOD_CPPFLAGS = $(CPPFLAGS_OPROF)
vector_fe_ex9_kokkos_%.o: $(KOKKOS_VECTOR_FE_EX9_DEPS)
	$(KOKKOS_CXX) $(KOKKOS_CXXFLAGS) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	  $(KOKKOS_VECTOR_FE_EX9_METHOD_CPPFLAGS) $(AM_CPPFLAGS) -I$(top_srcdir)/include $(CPPFLAGS) $(KOKKOS_MPI_CPPFLAGS) $(KOKKOS_CPPFLAGS) \
	  -c -o $@ `test -f '$<' || echo '$(srcdir)/'`$<
endif
else
KOKKOS_VECTOR_FE_EX9_DBG_OBJECTS =
KOKKOS_VECTOR_FE_EX9_DEVEL_OBJECTS =
KOKKOS_VECTOR_FE_EX9_OPT_OBJECTS =
KOKKOS_VECTOR_FE_EX9_PROF_OBJECTS =
KOKKOS_VECTOR_FE_EX9_OPROF_OBJECTS =
endif

example_dbg_LDFLAGS += $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)
example_devel_LDFLAGS += $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)
example_opt_LDFLAGS += $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)
example_prof_LDFLAGS += $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)
example_oprof_LDFLAGS += $(AM_LDFLAGS) $(KOKKOS_LDFLAGS)

example_dbg_LDADD += $(KOKKOS_VECTOR_FE_EX9_DBG_OBJECTS) $(KOKKOS_LIBS)
example_devel_LDADD += $(KOKKOS_VECTOR_FE_EX9_DEVEL_OBJECTS) $(KOKKOS_LIBS)
example_opt_LDADD += $(KOKKOS_VECTOR_FE_EX9_OPT_OBJECTS) $(KOKKOS_LIBS)
example_prof_LDADD += $(KOKKOS_VECTOR_FE_EX9_PROF_OBJECTS) $(KOKKOS_LIBS)
example_oprof_LDADD += $(KOKKOS_VECTOR_FE_EX9_OPROF_OBJECTS) $(KOKKOS_LIBS)

example_dbg_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_dbg_LDFLAGS) $(LDFLAGS) -o $@
example_devel_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_devel_LDFLAGS) $(LDFLAGS) -o $@
example_opt_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_opt_LDFLAGS) $(LDFLAGS) -o $@
example_prof_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_prof_LDFLAGS) $(LDFLAGS) -o $@
example_oprof_LINK = \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(KOKKOS_CXX) \
	$(example_oprof_LDFLAGS) $(LDFLAGS) -o $@
else
KOKKOS_VECTOR_FE_EX9_DBG_OBJECTS =
KOKKOS_VECTOR_FE_EX9_DEVEL_OBJECTS =
KOKKOS_VECTOR_FE_EX9_OPT_OBJECTS =
KOKKOS_VECTOR_FE_EX9_PROF_OBJECTS =
KOKKOS_VECTOR_FE_EX9_OPROF_OBJECTS =
endif
