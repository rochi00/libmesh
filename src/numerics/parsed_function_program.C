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

#include "libmesh/parsed_function_program.h"

#include "fparser_ad.hh"
#include "extrasrc/fptypes.hh"

namespace
{
namespace FParser = FUNCTIONPARSERTYPES;
namespace Opcode = libMesh::ParsedFunctionOpcode;

// Programs keep fparser's bytecode as is, so every opcode the device
// interpreter names must have fparser's value.  fparser drops cPopNMov,
// cLog2by and cNop without its optimizer, renumbering everything after
// them; libMesh never builds it that way.
#ifndef FP_SUPPORT_OPTIMIZER
#error "ParsedFunctionProgram requires fparser's optimizer opcodes"
#endif

#define LIBMESH_CHECK_OPCODE(name)                                      \
  static_assert(Opcode::name == static_cast<unsigned int>(FParser::name), \
                "ParsedFunctionOpcode::" #name " does not match fparser");
LIBMESH_PARSED_FUNCTION_OPCODES(LIBMESH_CHECK_OPCODE)
#undef LIBMESH_CHECK_OPCODE

} // anonymous namespace

namespace libMesh
{

template <typename Scalar>
ParsedFunctionProgram<Scalar>
build_parsed_function_program(const FunctionParserADBase<Scalar> & parser)
{
  const auto * data = parser.parser_data();
  libmesh_assert(data);
  const auto & bytecode = data->mByteCode;

  // The device interpreter has no complex arithmetic and no way to
  // call another parser.  Opcodes with operands are followed by them in
  // the bytecode; step over those so they aren't read as opcodes.
  for (std::size_t ip = 0; ip < bytecode.size(); ++ip)
    switch (bytecode[ip])
      {
      case Opcode::cArg:
      case Opcode::cConj:
      case Opcode::cImag:
      case Opcode::cPolar:
      case Opcode::cReal:
        libmesh_error_msg("Kokkos parsed-function export does not support complex-valued fparser opcodes");

      case Opcode::cFCall:
      case Opcode::cPCall:
        libmesh_error_msg("Kokkos parsed-function export does not support user-defined or nested parser calls");

      case Opcode::cIf:
      case Opcode::cJump:
      case Opcode::cPopNMov:
        ip += 2;
        libmesh_assert_less(ip, bytecode.size());
        break;

      case Opcode::cFetch:
        ip += 1;
        libmesh_assert_less(ip, bytecode.size());
        break;

      default:
        break;
      }

  ParsedFunctionProgram<Scalar> program;
  program.bytecode.assign(bytecode.begin(), bytecode.end());
  program.immediates.assign(data->mImmed.begin(), data->mImmed.end());
  program.stack_size = data->mStackSize;
  program.n_variables = data->mVariablesAmount;
  program.epsilon = FunctionParserBase<Scalar>::epsilon();

  return program;
}

template ParsedFunctionProgram<Real>
build_parsed_function_program(const FunctionParserADBase<Real> & parser);

} // namespace libMesh
