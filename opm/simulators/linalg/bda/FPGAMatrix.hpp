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

#ifndef FPGA_MATRIX_HEADER_INCLUDED
#define FPGA_MATRIX_HEADER_INCLUDED

#include <opm/simulators/linalg/bda/FPGAUtils.hpp>

namespace bda
{

typedef struct {
	double *nnzValues;
	int *colIndices;
	int *rowPointers;
	int rowSize;
	int colSize;
        int valSize;
} Matrix;

Matrix *allocateMatrix(int rowSize, int valSize);
void freeMatrix(Matrix **mat);

int matrixToRDF(Matrix *mat, int numColors, int *nodesPerColor, 
    int **colIndicesInColor, int nnzsPerRowLimit, 
    double **nnzValues, short int *colIndices, int *nnzValsSizes, unsigned char *NROffsets, int *colorSizes,
    int numFieldSplits, bool readInterleaved,  int readBatchSize);

void sortRow(int *colIndices, double *data, int left, int right);

} // end namespace bda

#endif // FPGA_MATRIX_HEADER_INCLUDED
