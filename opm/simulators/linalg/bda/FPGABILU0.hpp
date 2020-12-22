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

#ifndef FPGA_BILU0_HEADER_INCLUDED
#define FPGA_BILU0_HEADER_INCLUDED

#include <opm/simulators/linalg/bda/ILUReorder.hpp>
#include <opm/simulators/linalg/bda/FPGABlockedMatrix.hpp>

namespace bda
{

/*
 * This class implements a Blocked ILU0 preconditioner, with output data
 * specifically formatted for the FPGA.
 * The decomposition and reorders of the rows of the matrix are done on CPU.
 */

class FPGABILU0
{

private:
    int N;       // number of rows of the matrix
    int Nb;      // number of blockrows of the matrix
    int nnz;     // number of nonzeroes of the matrix (scalar)
    int nnzbs;   // number of blocks of the matrix
    bool initialized = false;
    BlockedMatrixFpga *LMat = NULL, *UMat = NULL, *LUMat = NULL;
    BlockedMatrixFpga *rMat = NULL; // reordered mat
    Block *invDiagVals = nullptr;
    int *diagIndex = nullptr;
    int *toOrder = nullptr, *fromOrder = nullptr;
    int *rowsPerColor = NULL;
    int numColors;
    int nnzSplit;

    // sizes and arrays used during RDF generation
    double **nnzValues = NULL, **LnnzValues = NULL, **UnnzValues = NULL;
    short int *colIndices = NULL, *LColIndices = NULL, *UColIndices = NULL;
    unsigned char *NROffsets = NULL, *LNROffsets = NULL, *UNROffsets = NULL;
    int *PIndicesAddr = NULL, *LPIndicesAddr = NULL, *UPIndicesAddr = NULL;
    int *colorSizes = NULL, *LColorSizes = NULL, *UColorSizes = NULL;
    int *nnzValsSizes = NULL, *LnnzValsSizes = NULL, *UnnzValsSizes = NULL;
    int **colIndicesInColor = NULL, **LColIndicesInColor = NULL, **UColIndicesInColor = NULL;

    int rowSize, valSize;
    int LRowSize, LValSize, LNumColors;
    int URowSize, UValSize, UNumColors;
    double *blockDiag = NULL;
    ILUReorder opencl_ilu_reorder;
    bool level_scheduling = false, graph_coloring = false;
    void **resultPointers = nullptr;
    int *resultSizes = nullptr;
    int maxRowsPerColor, maxColsPerColor, maxNNZsPerRow, maxNumColors;
    int readBatchSize = 2048; // currently used only when readInterleaved==true

public:

    FPGABILU0(ILUReorder opencl_ilu_reorder, int maxRowsPerColor, int maxColsPerColor, int maxNNZsPerRow, int maxNumColors);

    ~FPGABILU0();

    // analysis (optional)
    bool init(BlockedMatrixFpga *mat);

    // ilu_decomposition
    bool create_preconditioner(BlockedMatrixFpga *mat);

    // apply preconditioner, y = prec(x)
    void apply(double *x, double *y);

    int* getToOrder()
    {
        return toOrder;
    }

    int* getFromOrder()
    {
        return fromOrder;
    }

    BlockedMatrixFpga* getRMat()
    {
        return rMat;
    }

    void **getResultPointers()
    {
        return resultPointers;
    }

    int *getResultSizes()
    {
        return resultSizes;
    }

};

} //namespace bda

#endif // FPGA_BILU0_HEADER_INCLUDED
