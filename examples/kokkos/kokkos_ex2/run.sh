#!/bin/sh

#set -x

. "$LIBMESH_DIR"/examples/run_common.sh

example_name=kokkos_ex2

run_example "$example_name" "-n 8"
