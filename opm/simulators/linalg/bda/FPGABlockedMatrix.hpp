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

#ifndef FPGA_BLOCKED_MATRIX_HEADER_INCLUDED
#define FPGA_BLOCKED_MATRIX_HEADER_INCLUDED

#include <opm/simulators/linalg/bda/FPGAMatrix.hpp>

#define BLOCK_SIZE 3

typedef double Block[BLOCK_SIZE*BLOCK_SIZE];

namespace bda
{

typedef struct {
    Block *nnzValues = NULL;
    int *colIndices = NULL;
    int *rowPointers = NULL;
    int rowSize;
    int valSize;
} BlockedMatrixFpga;

BlockedMatrixFpga *allocateBlockedMatrixFpga(int rowSize, int valSize);
void freeBlockedMatrixFpga(BlockedMatrixFpga **mat);
BlockedMatrixFpga *soft_copyBlockedMatrixFpga(BlockedMatrixFpga *mat);

int findPartitionColumns(BlockedMatrixFpga *mat, int numColors, int *nodesPerColor,
    int rowsPerColorLimit, int columnsPerColorLimit,
    int **colIndicesInColor, int *PIndicesAddr, int *colorSizes,
    int **LColIndicesInColor, int *LPIndicesAddr, int *LColorSizes,
    int **UColIndicesInColor, int *UPIndicesAddr, int *UColorSizes);

void blockedDiagtoRDF(Block *blockedDiagVals, int rowSize, int numColors, int *rowsPerColor, double *RDFDiag);

int BlockedMatrixFpgaToRDF(BlockedMatrixFpga *mat, int numColors, int *nodesPerColor, bool isUMatrix,
    int **colIndicesInColor, int nnzsPerRowLimit,
    int numFieldSplits, bool readInterleaved, int readBatchSize, int *nnzValsSizes,
    double **nnzValues, short int *colIndices, unsigned char *NROffsets, int *colorSizes,  int *valSize);

void sortBlockedRow(int *colIndices, Block *data, int left, int right);

void blockSub(Block mat1, Block mat2, Block resMat);
void blockMult(Block mat1, Block mat2, Block resMat);
int blockInvert3x3(Block mat, Block res);
void blockVectMult(Block mat, double *vect, double scale, double *resVect, bool resetRes);

} // end namespace bda

#endif // FPGA_BLOCKED_MATRIX_HEADER_INCLUDED
