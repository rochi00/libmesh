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

// A flat, device-interpretable copy of fparser bytecode:
// ParsedFunctionProgram holds a parsed expression's bytecode and
// constants as plain arrays that a Kokkos kernel can evaluate without
// host calls.

#ifndef LIBMESH_PARSED_FUNCTION_PROGRAM_H
#define LIBMESH_PARSED_FUNCTION_PROGRAM_H

#include "libmesh/libmesh_common.h"

#include <vector>

template <typename Value_t>
class FunctionParserBase;
template <typename Value_t>
class FunctionParserADBase;

namespace libMesh
{

// fparser's opcodes, in fparser's order, so that a program's bytecode is
// fparser's own and the device interpreter switches on these names.
// fparser's list (extrasrc/fptypes.hh) is not an installed header, so
// this copy exists; parsed_function_program.C checks each entry against
// it.  The complex-number and function-call opcodes are listed only to
// keep the numbering; build_parsed_function_program() rejects them.
#define LIBMESH_PARSED_FUNCTION_OPCODES(X)                              \
  X(cAbs) X(cAcos) X(cAcosh) X(cArg) X(cAsin) X(cAsinh) X(cAtan)        \
  X(cAtan2) X(cAtanh) X(cCbrt) X(cCeil) X(cConj) X(cCos) X(cCosh)       \
  X(cCot) X(cCsc) X(cExp) X(cExp2) X(cFloor) X(cHypot) X(cIf) X(cImag)  \
  X(cInt) X(cLog) X(cLog10) X(cLog2) X(cMax) X(cMin) X(cPolar) X(cPow)  \
  X(cReal) X(cSec) X(cSin) X(cSinh) X(cSqrt) X(cTan) X(cTanh) X(cTrunc) \
  X(cImmed) X(cJump) X(cNeg) X(cAdd) X(cSub) X(cMul) X(cDiv) X(cMod)    \
  X(cEqual) X(cNEqual) X(cLess) X(cLessOrEq) X(cGreater)                \
  X(cGreaterOrEq) X(cNot) X(cAnd) X(cOr) X(cNotNot) X(cDeg) X(cRad)     \
  X(cFCall) X(cPCall) X(cPopNMov) X(cLog2by) X(cNop) X(cSinCos)         \
  X(cSinhCosh) X(cAbsAnd) X(cAbsOr) X(cAbsNot) X(cAbsNotNot) X(cAbsIf)  \
  X(cDup) X(cFetch) X(cInv) X(cSqr) X(cRDiv) X(cRSub) X(cRSqrt)         \
  X(VarBegin)

namespace ParsedFunctionOpcode
{
#define LIBMESH_PARSED_FUNCTION_OPCODE(name) name,
enum : unsigned int { LIBMESH_PARSED_FUNCTION_OPCODES(LIBMESH_PARSED_FUNCTION_OPCODE) };
#undef LIBMESH_PARSED_FUNCTION_OPCODE
}

template <typename Scalar>
struct ParsedFunctionProgram
{
  std::vector<unsigned int> bytecode;
  std::vector<Scalar> immediates;
  unsigned int stack_size = 0;
  unsigned int n_variables = 0;
  Scalar epsilon = 0;

  bool empty() const { return bytecode.empty(); }
};

template <typename Scalar>
struct ParsedFunctionProgramBundle
{
  ParsedFunctionProgram<Scalar> value;
  ParsedFunctionProgram<Scalar> dx;
#if LIBMESH_DIM > 1
  ParsedFunctionProgram<Scalar> dy;
#endif
#if LIBMESH_DIM > 2
  ParsedFunctionProgram<Scalar> dz;
#endif
  ParsedFunctionProgram<Scalar> dt;
};

template <typename Scalar>
struct ParsedFEMFunctionProgramBundle : ParsedFunctionProgramBundle<Scalar>
{
  std::vector<unsigned int> value_variable_numbers;
  std::vector<ParsedFunctionProgram<Scalar>> value_variable_derivatives;
  bool uses_field_gradients = false;
  bool uses_field_hessians = false;
  bool uses_normals = false;
  bool uses_additional_variables = false;

  bool supports_kokkos_value_goal() const
  {
    return !uses_field_gradients &&
           !uses_field_hessians &&
           !uses_normals &&
           !uses_additional_variables &&
           value_variable_numbers.size() == value_variable_derivatives.size();
  }
};

template <typename Scalar>
ParsedFunctionProgram<Scalar>
build_parsed_function_program(const FunctionParserADBase<Scalar> & parser);

} // namespace libMesh

#endif // LIBMESH_PARSED_FUNCTION_PROGRAM_H
