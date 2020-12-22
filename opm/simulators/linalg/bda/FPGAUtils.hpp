/*
  Copyright 2020 Equinor ASA

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FPGA_UTILS_HEADER_INCLUDED
#define FPGA_UTILS_HEADER_INCLUDED

// define BDA_DEBUG_LEVEL to a value greater than 0 to activate debug printouts for some functions
#if !defined (BDA_DEBUG_LEVEL)
#define BDA_DEBUG_LEVEL 0
#endif

#define BDA_DEBUG(y,x) { if (y <= BDA_DEBUG_LEVEL) { x; fflush(NULL); } }

namespace bda
{

union double2int
{
    unsigned long int int_val;
    double double_val;
};

double second(void);
bool even(int n);
int roundUpTo(int i, int n);
bool fileExists(const char *filename);

} // end namespace bda

#endif // FPGA_UTILS_HEADER_INCLUDED
