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

template <typename Scalar>
void
throw_unsupported_complex_opcode()
{
  libmesh_error_msg("Kokkos parsed-function export does not support complex-valued fparser "
                    "opcodes");
}

template <typename Scalar>
void
throw_unsupported_call_opcode()
{
  libmesh_error_msg("Kokkos parsed-function export does not support user-defined or nested "
                    "parser calls");
}

template <typename Scalar>
unsigned int
translate_fparser_opcode(const unsigned int opcode)
{
  if (opcode >= FParser::VarBegin)
    return Opcode::VarBegin + (opcode - FParser::VarBegin);

  switch (static_cast<FParser::OPCODE>(opcode))
    {
    case FParser::cAbs: return Opcode::cAbs;
    case FParser::cAcos: return Opcode::cAcos;
    case FParser::cAcosh: return Opcode::cAcosh;
    case FParser::cAsin: return Opcode::cAsin;
    case FParser::cAsinh: return Opcode::cAsinh;
    case FParser::cAtan: return Opcode::cAtan;
    case FParser::cAtan2: return Opcode::cAtan2;
    case FParser::cAtanh: return Opcode::cAtanh;
    case FParser::cCbrt: return Opcode::cCbrt;
    case FParser::cCeil: return Opcode::cCeil;
    case FParser::cCos: return Opcode::cCos;
    case FParser::cCosh: return Opcode::cCosh;
    case FParser::cCot: return Opcode::cCot;
    case FParser::cCsc: return Opcode::cCsc;
    case FParser::cExp: return Opcode::cExp;
    case FParser::cExp2: return Opcode::cExp2;
    case FParser::cFloor: return Opcode::cFloor;
    case FParser::cHypot: return Opcode::cHypot;
    case FParser::cIf: return Opcode::cIf;
    case FParser::cInt: return Opcode::cInt;
    case FParser::cLog: return Opcode::cLog;
    case FParser::cLog10: return Opcode::cLog10;
    case FParser::cLog2: return Opcode::cLog2;
    case FParser::cMax: return Opcode::cMax;
    case FParser::cMin: return Opcode::cMin;
    case FParser::cPow: return Opcode::cPow;
    case FParser::cSec: return Opcode::cSec;
    case FParser::cSin: return Opcode::cSin;
    case FParser::cSinh: return Opcode::cSinh;
    case FParser::cSqrt: return Opcode::cSqrt;
    case FParser::cTan: return Opcode::cTan;
    case FParser::cTanh: return Opcode::cTanh;
    case FParser::cTrunc: return Opcode::cTrunc;
    case FParser::cImmed: return Opcode::cImmed;
    case FParser::cJump: return Opcode::cJump;
    case FParser::cNeg: return Opcode::cNeg;
    case FParser::cAdd: return Opcode::cAdd;
    case FParser::cSub: return Opcode::cSub;
    case FParser::cMul: return Opcode::cMul;
    case FParser::cDiv: return Opcode::cDiv;
    case FParser::cMod: return Opcode::cMod;
    case FParser::cEqual: return Opcode::cEqual;
    case FParser::cNEqual: return Opcode::cNEqual;
    case FParser::cLess: return Opcode::cLess;
    case FParser::cLessOrEq: return Opcode::cLessOrEq;
    case FParser::cGreater: return Opcode::cGreater;
    case FParser::cGreaterOrEq: return Opcode::cGreaterOrEq;
    case FParser::cNot: return Opcode::cNot;
    case FParser::cAnd: return Opcode::cAnd;
    case FParser::cOr: return Opcode::cOr;
    case FParser::cNotNot: return Opcode::cNotNot;
    case FParser::cDeg: return Opcode::cDeg;
    case FParser::cRad: return Opcode::cRad;
#ifdef FP_SUPPORT_OPTIMIZER
    case FParser::cPopNMov: return Opcode::cPopNMov;
    case FParser::cLog2by: return Opcode::cLog2by;
    case FParser::cNop: return Opcode::cNop;
#endif
    case FParser::cSinCos: return Opcode::cSinCos;
    case FParser::cSinhCosh: return Opcode::cSinhCosh;
    case FParser::cAbsAnd: return Opcode::cAbsAnd;
    case FParser::cAbsOr: return Opcode::cAbsOr;
    case FParser::cAbsNot: return Opcode::cAbsNot;
    case FParser::cAbsNotNot: return Opcode::cAbsNotNot;
    case FParser::cAbsIf: return Opcode::cAbsIf;
    case FParser::cDup: return Opcode::cDup;
    case FParser::cFetch: return Opcode::cFetch;
    case FParser::cInv: return Opcode::cInv;
    case FParser::cSqr: return Opcode::cSqr;
    case FParser::cRDiv: return Opcode::cRDiv;
    case FParser::cRSub: return Opcode::cRSub;
    case FParser::cRSqrt: return Opcode::cRSqrt;

    case FParser::cArg:
    case FParser::cConj:
    case FParser::cImag:
    case FParser::cPolar:
    case FParser::cReal:
      throw_unsupported_complex_opcode<Scalar>();

    case FParser::cFCall:
    case FParser::cPCall:
      throw_unsupported_call_opcode<Scalar>();

    case FParser::VarBegin:
      return Opcode::VarBegin;
    }

  libmesh_error_msg("Kokkos parsed-function export encountered an unknown opcode " << opcode);
  return Opcode::cNop;
}

template <typename Scalar>
void
append_translated_instruction(const std::vector<unsigned int> & fparser_bytecode,
                              std::size_t & ip,
                              std::vector<unsigned int> & translated_bytecode)
{
  const unsigned int opcode = translate_fparser_opcode<Scalar>(fparser_bytecode[ip]);
  translated_bytecode.push_back(opcode);

  switch (opcode)
    {
    case Opcode::cIf:
    case Opcode::cJump:
    case Opcode::cPopNMov:
      libmesh_error_msg_if(ip + 2 >= fparser_bytecode.size(),
                           "Kokkos parsed-function export found a truncated bytecode operand");
      translated_bytecode.push_back(fparser_bytecode[++ip]);
      translated_bytecode.push_back(fparser_bytecode[++ip]);
      break;

    case Opcode::cFetch:
      libmesh_error_msg_if(ip + 1 >= fparser_bytecode.size(),
                           "Kokkos parsed-function export found a truncated bytecode operand");
      translated_bytecode.push_back(fparser_bytecode[++ip]);
      break;

    default:
      break;
    }
}

} // anonymous namespace

namespace libMesh
{

template <typename Scalar>
ParsedFunctionProgram<Scalar>
build_parsed_function_program(const FunctionParserADBase<Scalar> & parser)
{
  ParsedFunctionProgram<Scalar> program;
  const auto * data = parser.parser_data();
  libmesh_assert(data);

  program.bytecode.reserve(data->mByteCode.size());
  for (std::size_t ip = 0; ip != data->mByteCode.size(); ++ip)
    append_translated_instruction<Scalar>(data->mByteCode, ip, program.bytecode);

  program.immediates.assign(data->mImmed.begin(), data->mImmed.end());
  program.stack_size = data->mStackSize;
  program.n_variables = data->mVariablesAmount;
  program.epsilon = FunctionParserBase<Scalar>::epsilon();

  return program;
}

template ParsedFunctionProgram<Real>
build_parsed_function_program(const FunctionParserADBase<Real> & parser);

} // namespace libMesh
