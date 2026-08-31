#!/bin/sh

. "$LIBMESH_DIR"/examples/run_common.sh

example_name=adaptivity_ex2_kokkos

run_kokkos_example()
{
  if test "${LIBMESH_BENCHMARK}" != ""; then
    return
  fi

  common_options=$*

  if test "${METHODS}" = ""; then
    if test "${METHOD}" = ""; then
      METHODS=opt
    else
      METHODS="${METHOD}"
    fi
  fi

  ordered_methods="dbg debug devel profiling pro prof oprofile oprof optimized opt"
  selected_methods=""
  for method in ${ordered_methods}; do
    for selected_method in ${METHODS}; do
      if test "${selected_method}" = "${method}"; then
        selected_methods="${selected_methods} ${selected_method}"
      fi
    done
  done

  for method in ${selected_methods}; do
    case "${method}" in
      optimized|opt)      executable=example-kokkos-opt ;;
      debug|dbg)          executable=example-kokkos-dbg ;;
      devel)              executable=example-kokkos-devel ;;
      profiling|pro|prof) executable=example-kokkos-prof ;;
      oprofile|oprof)     executable=example-kokkos-oprof ;;
      *) echo "ERROR: unknown method: ${method}!"; exit 1 ;;
    esac

    if test ! -x "${executable}"; then
      echo "ERROR: cannot find ${executable}!"
      exit 1
    fi

    message_running "${example_name}" "${executable}" "${common_options}"
    ${LIBMESH_RUN} ./"${executable}" ${common_options} ${LIBMESH_OPTIONS}
    retval=$?
    if test ${retval} -ne 0 && test ${retval} -ne 77; then
      exit ${retval}
    fi
    message_done_running "${example_name}" "${executable}" "${common_options}"
  done
}

for order in 1 2; do
  options="-n_timesteps 1 -n_refinements 0 -max_h_level 1 -output_freq 100 -init_timestep 0 -order ${order}"
  run_kokkos_example ${options}

  options="-read_solution -n_timesteps 1 -max_h_level 1 -output_freq 100 -init_timestep 1"
  run_kokkos_example ${options}
done
