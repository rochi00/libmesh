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

#ifndef LIBMESH_PARSED_FUNCTION_PROGRAM_H
#define LIBMESH_PARSED_FUNCTION_PROGRAM_H

#include "libmesh/libmesh_common.h"
#include "libmesh/libmesh_device.h"

#include <vector>

template <typename Value_t>
class FunctionParserBase;
template <typename Value_t>
class FunctionParserADBase;

namespace libMesh
{

namespace ParsedFunctionOpcode
{
static constexpr unsigned int cAbs = 0;
static constexpr unsigned int cAcos = 1;
static constexpr unsigned int cAcosh = 2;
static constexpr unsigned int cArg = 3;
static constexpr unsigned int cAsin = 4;
static constexpr unsigned int cAsinh = 5;
static constexpr unsigned int cAtan = 6;
static constexpr unsigned int cAtan2 = 7;
static constexpr unsigned int cAtanh = 8;
static constexpr unsigned int cCbrt = 9;
static constexpr unsigned int cCeil = 10;
static constexpr unsigned int cConj = 11;
static constexpr unsigned int cCos = 12;
static constexpr unsigned int cCosh = 13;
static constexpr unsigned int cCot = 14;
static constexpr unsigned int cCsc = 15;
static constexpr unsigned int cExp = 16;
static constexpr unsigned int cExp2 = 17;
static constexpr unsigned int cFloor = 18;
static constexpr unsigned int cHypot = 19;
static constexpr unsigned int cIf = 20;
static constexpr unsigned int cImag = 21;
static constexpr unsigned int cInt = 22;
static constexpr unsigned int cLog = 23;
static constexpr unsigned int cLog10 = 24;
static constexpr unsigned int cLog2 = 25;
static constexpr unsigned int cMax = 26;
static constexpr unsigned int cMin = 27;
static constexpr unsigned int cPolar = 28;
static constexpr unsigned int cPow = 29;
static constexpr unsigned int cReal = 30;
static constexpr unsigned int cSec = 31;
static constexpr unsigned int cSin = 32;
static constexpr unsigned int cSinh = 33;
static constexpr unsigned int cSqrt = 34;
static constexpr unsigned int cTan = 35;
static constexpr unsigned int cTanh = 36;
static constexpr unsigned int cTrunc = 37;
static constexpr unsigned int cImmed = 38;
static constexpr unsigned int cJump = 39;
static constexpr unsigned int cNeg = 40;
static constexpr unsigned int cAdd = 41;
static constexpr unsigned int cSub = 42;
static constexpr unsigned int cMul = 43;
static constexpr unsigned int cDiv = 44;
static constexpr unsigned int cMod = 45;
static constexpr unsigned int cEqual = 46;
static constexpr unsigned int cNEqual = 47;
static constexpr unsigned int cLess = 48;
static constexpr unsigned int cLessOrEq = 49;
static constexpr unsigned int cGreater = 50;
static constexpr unsigned int cGreaterOrEq = 51;
static constexpr unsigned int cNot = 52;
static constexpr unsigned int cAnd = 53;
static constexpr unsigned int cOr = 54;
static constexpr unsigned int cNotNot = 55;
static constexpr unsigned int cDeg = 56;
static constexpr unsigned int cRad = 57;
static constexpr unsigned int cFCall = 58;
static constexpr unsigned int cPCall = 59;
static constexpr unsigned int cPopNMov = 60;
static constexpr unsigned int cLog2by = 61;
static constexpr unsigned int cNop = 62;
static constexpr unsigned int cSinCos = 63;
static constexpr unsigned int cSinhCosh = 64;
static constexpr unsigned int cAbsAnd = 65;
static constexpr unsigned int cAbsOr = 66;
static constexpr unsigned int cAbsNot = 67;
static constexpr unsigned int cAbsNotNot = 68;
static constexpr unsigned int cAbsIf = 69;
static constexpr unsigned int cDup = 70;
static constexpr unsigned int cFetch = 71;
static constexpr unsigned int cInv = 72;
static constexpr unsigned int cSqr = 73;
static constexpr unsigned int cRDiv = 74;
static constexpr unsigned int cRSub = 75;
static constexpr unsigned int cRSqrt = 76;
static constexpr unsigned int VarBegin = 77;
}

LIBMESH_DEVICE_INLINE constexpr unsigned int
parsed_function_var_begin()
{
  return ParsedFunctionOpcode::VarBegin;
}

LIBMESH_DEVICE_INLINE constexpr bool
parsed_function_is_var_opcode(const unsigned int opcode)
{
  return opcode >= parsed_function_var_begin();
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
struct ParsedFEMFunctionProgramBundle
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
