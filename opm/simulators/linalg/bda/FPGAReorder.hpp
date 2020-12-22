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

#ifndef FPGA_REORDER_HEADER_INCLUDED
#define FPGA_REORDER_HEADER_INCLUDED

#include <opm/simulators/linalg/bda/FPGABlockedMatrix.hpp>
#include <opm/simulators/linalg/bda/FPGAMatrix.hpp>

namespace bda
{

#define MAX_TRIES 100
#define MAX_PARALLELISM 100000
#define MAX_COLORS 256

int findPreviousReordering(BlockedMatrixFpga *mat, int **rowsPerColor, int maxRowsPerColor, int maxColsPerColor, int maxColors);
int findPreviousReordering(Matrix *mat, int **rowsPerColor, int maxRowsPerColor, int maxColsPerColor, int maxColors);

int colorNodes(int rows, const int *rowPointers, const int *colIndices, int *colors);
void reorder_matrix_by_pattern(Matrix *mat, int *toOrder, int *fromOrder, Matrix *rMat);
void reorder_blocked_matrix_by_pattern(BlockedMatrixFpga *mat, int *toOrder, int *fromOrder, BlockedMatrixFpga *rMat);
void reorder_vector_by_pattern(int size, double *vector, int *fromOrder, double *rVector);
void reorder_vector_by_blocked_pattern(int size, double *vector, int *fromOrder, double *rVector);

int *findLevelScheduling(int *CSRColIndices, int *CSRRowPointers, int *CSCColIndices, int *CSCRowPointers, int rowSize, int *iters, int *toOrder, int* fromOrder);
int* findGraphColoring(int *colIndices, int *rowPointers, int rowSize, int maxRowsPerColor, int maxColsPerColor, int *numColors, int *toOrder, int* fromOrder);

void bcsrToBcsc(Block *Avals, int *Acols, int *Arows, Block *Bvals, int *Bcols, int *Brows, int numRows);

}

#endif // FPGA_REORDER_HEADER_INCLUDED
