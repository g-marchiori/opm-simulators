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

#include <stdio.h>
#include <cstdlib>
#include <cstring>

#include <opm/simulators/linalg/bda/FPGABILU0.hpp>
#include <opm/simulators/linalg/bda/FPGAReorder.hpp>
#include <opm/simulators/linalg/bda/FPGAUtils.hpp>

#define PRINT_TIMERS 0

namespace bda
{

    FPGABILU0::FPGABILU0(ILUReorder opencl_ilu_reorder_, int maxRowsPerColor_, int maxColsPerColor_, int maxNNZsPerRow_, int maxNumColors_) :
        opencl_ilu_reorder(opencl_ilu_reorder_), maxRowsPerColor(maxRowsPerColor_), maxColsPerColor(maxColsPerColor_), maxNNZsPerRow(maxNNZsPerRow_), maxNumColors(maxNumColors_)
    {
        if (opencl_ilu_reorder == ILUReorder::LEVEL_SCHEDULING) {
            level_scheduling = true;
        } else if (opencl_ilu_reorder == ILUReorder::GRAPH_COLORING) {
            graph_coloring = true;
        } else {
            printf("ERROR: FPGABILU0: ilu reordering strategy not set correctly\n");
            exit(1);
        }
    }


    FPGABILU0::~FPGABILU0()
    {
        if (!initialized) {
            return;
        }
        delete[] toOrder;
        delete[] fromOrder;
        delete[] invDiagVals;
        delete[] diagIndex;
        if (rowsPerColor != NULL) { free(rowsPerColor); }
        if (LUMat != NULL) { if (LUMat->nnzValues != NULL) { free(LUMat->nnzValues); } }
        if (LUMat != NULL) { free(LUMat); }
        freeBlockedMatrixFpga(&LMat);
        freeBlockedMatrixFpga(&UMat);
        freeBlockedMatrixFpga(&rMat);
        if (colorSizes != NULL) { free(colorSizes); }
        if (PIndicesAddr != NULL) { free(PIndicesAddr); }
        for(int i = 0; i < nnzSplit; i++){
            if (nnzValues[i] != NULL) { free(nnzValues[i]); }
            if (LnnzValues[i] != NULL) { free(LnnzValues[i]); }
            if (UnnzValues[i] != NULL) { free(UnnzValues[i]); }
        }
        if (nnzValues != NULL) { free(nnzValues); }
        if (nnzValsSizes != NULL) { free(nnzValsSizes); }
        if (colIndices != NULL) { free(colIndices); }
        if (NROffsets != NULL) { free(NROffsets); }
        if (LColorSizes != NULL) { free(LColorSizes); }
        if (LPIndicesAddr != NULL) { free(LPIndicesAddr); }
        if (LnnzValues != NULL) { free(LnnzValues); }
        if (LnnzValsSizes != NULL) { free(LnnzValsSizes); }
        if (LColIndices != NULL) { free(LColIndices); }
        if (LNROffsets != NULL) { free(LNROffsets); }
        if (UColorSizes != NULL) { free(UColorSizes); }
        if (UPIndicesAddr != NULL) { free(UPIndicesAddr); }
        if (UnnzValues != NULL) { free(UnnzValues); }
        if (UnnzValsSizes != NULL) { free(UnnzValsSizes); }
        if (UColIndices != NULL) { free(UColIndices); }
        if (UNROffsets != NULL) { free(UNROffsets); }
        if (blockDiag != NULL) { free(blockDiag); }
        for(int c = 0; c < numColors; c++){
            if (colIndicesInColor[c] != NULL) { free(colIndicesInColor[c]); }
            if (LColIndicesInColor[c] != NULL) { free(LColIndicesInColor[c]); }
            if (UColIndicesInColor[c] != NULL) { free(UColIndicesInColor[c]); }
        }
        if (colIndicesInColor != NULL) { free(colIndicesInColor); }
        if (LColIndicesInColor != NULL) { free(LColIndicesInColor); }
        if (UColIndicesInColor != NULL) { free(UColIndicesInColor); }
    }


    bool FPGABILU0::init(BlockedMatrixFpga *mat)
    {
        int i;

        resultPointers = new void*[21];
        resultSizes = new int[18];
        for (i = 0; i < 21; i++) { resultPointers[i] = nullptr; }

        // Set nnzSplit as hardcoded constant until support for more than one nnzVals read array is added.
        nnzSplit = 1;

#if PRINT_TIMERS
        double t1, t2;
#endif
        BlockedMatrixFpga *CSCmat = nullptr;

        this->N = mat->rowSize * BLOCK_SIZE;
        this->Nb = mat->rowSize;
        this->nnz = mat->valSize * BLOCK_SIZE * BLOCK_SIZE;
        this->nnzbs = mat->valSize;

        toOrder = new int[N];
        fromOrder = new int[N];
        if (level_scheduling) {
            CSCmat = new BlockedMatrixFpga;
            CSCmat->rowSize = Nb;
            CSCmat->valSize = nnzbs;
            CSCmat->nnzValues = new Block[nnzbs];
            CSCmat->colIndices = new int[nnzbs];
            CSCmat->rowPointers = new int[Nb + 1];
            CSCmat->colIndices[0] = 0;
            CSCmat->colIndices[nnzbs - 1] = 0;
            CSCmat->rowPointers[0] = 0;
            CSCmat->rowPointers[Nb] = 0;
#if PRINT_TIMERS
            t1 = second();
#endif
            bcsrToBcsc(mat->nnzValues, mat->colIndices, mat->rowPointers, CSCmat->nnzValues, CSCmat->colIndices, CSCmat->rowPointers, mat->rowSize);
#if PRINT_TIMERS
            t2 = second();
            printf("Convert CSR to CSC: %.2f ms\n", 1e3 * (t2 - t1));
#endif
        }
#if PRINT_TIMERS
        t1 = second();
#endif
        rMat = allocateBlockedMatrixFpga(mat->rowSize, mat->valSize);
        LUMat = soft_copyBlockedMatrixFpga(rMat);
        if (level_scheduling) {
            rowsPerColor = findLevelScheduling(mat->colIndices, mat->rowPointers, CSCmat->colIndices, CSCmat->rowPointers, mat->rowSize, &numColors, toOrder, fromOrder);
        } else if (graph_coloring) {
            rowsPerColor = findGraphColoring(mat->colIndices, mat->rowPointers, mat->rowSize, maxRowsPerColor, maxColsPerColor, &numColors, toOrder, fromOrder);
        }
        if (rowsPerColor == NULL) {
            printf("ERROR: cannot find a matrix reordering.\n");
            if (level_scheduling) {
                delete[] CSCmat->nnzValues;
                delete[] CSCmat->colIndices;
                delete[] CSCmat->rowPointers;
                delete CSCmat;
            }
            exit(1);
        }
        if (numColors > maxNumColors) {
            printf("ERROR: the matrix was reordered into too many colors. Created %d colors, while hardware only supports up to %d.\n", numColors, maxNumColors);
            if (level_scheduling) {
                delete[] CSCmat->nnzValues;
                delete[] CSCmat->colIndices;
                delete[] CSCmat->rowPointers;
                delete CSCmat;
            }
            exit(1);
        }
#if PRINT_TIMERS
        t2 = second();
        printf("Analysis: %.2f ms\n", 1e3 * (t2 - t1));
#endif
        int colorRoundedValSize = 0, LColorRoundedValSize = 0, UColorRoundedValSize = 0;
        int NROffsetSize = 0, LNROffsetSize = 0, UNROffsetSize = 0;
        int worstCaseColumnAccessNum;
        int colorSizesNum;
        int blockDiagSize = 0;
        // This reordering is needed here only to te result can be used to calculate worst-case scenario array sizes
        reorder_blocked_matrix_by_pattern(mat, toOrder, fromOrder, rMat);
        int doneRows = 0;
        int columnIndex;
        for (int c = 0; c < numColors; c++) {
            for (i = doneRows; i < doneRows + rowsPerColor[c]; i++) {
                for (int j = rMat->rowPointers[i]; j < rMat->rowPointers[i + 1]; j++) {
                    columnIndex = rMat->colIndices[j];
                    if (columnIndex < i) {
                        LColorRoundedValSize += 9;
                        LNROffsetSize += 9;
                    }
                    if (columnIndex > i) {
                        UColorRoundedValSize += 9;
                        UNROffsetSize += 9;
                    }
                    colorRoundedValSize += 9;
                    NROffsetSize += 9;
                }
                blockDiagSize += 12;
            }
            // End of color: round all sizes to nearest cacheline
            colorRoundedValSize = roundUpTo(colorRoundedValSize, 32);
            LColorRoundedValSize = roundUpTo(LColorRoundedValSize, 32);
            UColorRoundedValSize = roundUpTo(UColorRoundedValSize, 32);
            NROffsetSize = roundUpTo(NROffsetSize, 64);
            LNROffsetSize = roundUpTo(LNROffsetSize, 64);
            UNROffsetSize = roundUpTo(UNROffsetSize, 64);
            blockDiagSize = roundUpTo(blockDiagSize, 8);

            doneRows += rowsPerColor[c];
        }
        colorSizesNum = 8 + roundUpTo(4 * numColors, 16);
        worstCaseColumnAccessNum = numColors * maxColsPerColor;

        nnzValues = (double **) malloc(sizeof(double *) * nnzSplit);
        LnnzValues = (double **) malloc(sizeof(double *) * nnzSplit);
        UnnzValues = (double **) malloc(sizeof(double *) * nnzSplit);
        nnzValsSizes = (int *) malloc(sizeof(int) * nnzSplit);
        LnnzValsSizes = (int *) malloc(sizeof(int) * nnzSplit);
        UnnzValsSizes = (int *) malloc(sizeof(int) * nnzSplit);
        for(i = 0; i < nnzSplit; i++) {
            nnzValues[i] = (double *) malloc(sizeof(double) * colorRoundedValSize);
            LnnzValues[i] = (double *) malloc(sizeof(double) * LColorRoundedValSize);
            UnnzValues[i] = (double *) malloc(sizeof(double) * UColorRoundedValSize);
            // initial number of nnz, used to allocate
            nnzValsSizes[i] = colorRoundedValSize;
            LnnzValsSizes[i] = LColorRoundedValSize;
            UnnzValsSizes[i] = UColorRoundedValSize;
        }
        colIndices = (short int *) malloc(sizeof(short int) * colorRoundedValSize);
        LColIndices = (short int *) malloc(sizeof(short int) * LColorRoundedValSize);
        UColIndices = (short int *) malloc(sizeof(short int) * UColorRoundedValSize);
        NROffsets = (unsigned char *) malloc(sizeof(char) * NROffsetSize);
        LNROffsets = (unsigned char *) malloc(sizeof(char) * LNROffsetSize);
        UNROffsets = (unsigned char *) malloc(sizeof(char) * UNROffsetSize);
        PIndicesAddr = (int *) malloc(sizeof(int) * worstCaseColumnAccessNum);
        LPIndicesAddr = (int *) malloc(sizeof(int) * worstCaseColumnAccessNum);
        UPIndicesAddr = (int *) malloc(sizeof(int) * worstCaseColumnAccessNum);
        colorSizes = (int *) malloc(sizeof(int) * colorSizesNum);
        LColorSizes = (int *) malloc(sizeof(int) * colorSizesNum);
        UColorSizes = (int *) malloc(sizeof(int) * colorSizesNum);
        blockDiag = (double *)malloc(sizeof(double) * blockDiagSize);
        colIndicesInColor = (int **) malloc(sizeof(int *) * numColors);
        LColIndicesInColor = (int **) malloc(sizeof(int *) * numColors);
        UColIndicesInColor = (int **) malloc(sizeof(int *) * numColors);
        for(int c = 0; c < numColors; c++){
            colIndicesInColor[c] = (int *)malloc(sizeof(int) * BLOCK_SIZE * rMat->rowSize);
            LColIndicesInColor[c] = (int *)malloc(sizeof(int) * BLOCK_SIZE * rMat->rowSize);
            UColIndicesInColor[c] = (int *)malloc(sizeof(int) * BLOCK_SIZE * rMat->rowSize);
        }

        int err = findPartitionColumns(rMat, numColors, rowsPerColor,
                maxRowsPerColor, maxColsPerColor,
                colIndicesInColor, PIndicesAddr, colorSizes,
                LColIndicesInColor, LPIndicesAddr, LColorSizes,
                UColIndicesInColor, UPIndicesAddr, UColorSizes);
        if(err != 0){
            printf("ERROR: findPartitionColumns failed (%d).\n",err);
            exit(1);
        }

        diagIndex = new int[mat->rowSize];
        invDiagVals = new Block[mat->rowSize];
        LMat = allocateBlockedMatrixFpga(mat->rowSize, (mat->valSize - mat->rowSize) / 2);
        UMat = allocateBlockedMatrixFpga(mat->rowSize, (mat->valSize - mat->rowSize) / 2);
        LUMat->nnzValues = (Block*)malloc(mat->valSize * sizeof(Block));
        resultPointers[0] = (void *) colorSizes;
        resultPointers[1] = (void *) PIndicesAddr;
        resultPointers[2] = (void *) nnzValues;
        resultPointers[3] = (void *) colIndices;
        resultPointers[4] = (void *) NROffsets;
        resultPointers[5] = (void *) nnzValsSizes;
        resultPointers[6] = (void *) LColorSizes;
        resultPointers[7] = (void *) LPIndicesAddr;
        resultPointers[8] = (void *) LnnzValues;
        resultPointers[9] = (void *) LColIndices;
        resultPointers[10] = (void *) LNROffsets;
        resultPointers[11] = (void *) LnnzValsSizes;
        resultPointers[12] = (void *) UColorSizes;
        resultPointers[13] = (void *) UPIndicesAddr;
        resultPointers[14] = (void *) UnnzValues;
        resultPointers[15] = (void *) UColIndices;
        resultPointers[16] = (void *) UNROffsets;
        resultPointers[17] = (void *) UnnzValsSizes;
        resultPointers[18] = (void *) blockDiag;
        //resultPointers[19] and [20] are set by the caller
        resultSizes[0] = mat->rowSize * BLOCK_SIZE;
        resultSizes[1] = colorRoundedValSize; // zeropadded valSize;
        resultSizes[2] = numColors;
        resultSizes[3] = worstCaseColumnAccessNum; //totalCols
        resultSizes[4] = NROffsetSize; //NRFlagSize
        resultSizes[5] = blockDiagSize; //diagValsSize
        resultSizes[6] = mat->rowSize * BLOCK_SIZE;
        resultSizes[7] = LColorRoundedValSize; // zeropadded LValSize;
        resultSizes[8] = numColors;
        resultSizes[9] = worstCaseColumnAccessNum; //LTotalCols
        resultSizes[10] = LNROffsetSize; //LNRFlagSize
        resultSizes[11] = blockDiagSize; //LDiagValsSize
        resultSizes[12] = mat->rowSize * BLOCK_SIZE;
        resultSizes[13] = UColorRoundedValSize; // zeropadded UValSize;
        resultSizes[14] = numColors;
        resultSizes[15] = worstCaseColumnAccessNum; //UTotalCols
        resultSizes[16] = UNROffsetSize; //UNRFlagSize
        resultSizes[17] = blockDiagSize; //UDiagValsSize
        if (level_scheduling) {
            delete[] CSCmat->nnzValues;
            delete[] CSCmat->colIndices;
            delete[] CSCmat->rowPointers;
            delete CSCmat;
        }
        initialized = true;
        return true;
    } // end init()


    bool FPGABILU0::create_preconditioner(BlockedMatrixFpga *mat)
    {
#if PRINT_TIMERS
        double t1, t2;
        t1 = second();
#endif
        reorder_blocked_matrix_by_pattern(mat, toOrder, fromOrder, rMat);
#if PRINT_TIMERS
        t2 = second();
        printf("Reorder matrix: %f s\n", t2 - t1);
#endif
        // TODO: remove this copy by replacing inplace ilu decomp by out-of-place ilu decomp
#if PRINT_TIMERS
        t1 = second();
#endif
        memcpy(LUMat->nnzValues, rMat->nnzValues, sizeof(Block) * rMat->valSize);
#if PRINT_TIMERS
        t2 = second();
        printf("memcpy: %f s\n", t2 - t1);
#endif

        int i, j, k, ij, ik, jk;
        int iRowStart, iRowEnd, jRowEnd;
        Block pivot, tempBlock;
        int LSize = 0;
        const int blockSquare = BLOCK_SIZE * BLOCK_SIZE;

#if PRINT_TIMERS
        t1 = second();
#endif
        // go through all rows
        for (i = 0; i < LUMat->rowSize; i++) {
            iRowStart = LUMat->rowPointers[i];
            iRowEnd = LUMat->rowPointers[i + 1];

            // go through all elements of the row
            for (ij = iRowStart; ij < iRowEnd; ij++) {
                j = LUMat->colIndices[ij];
                // if the element is the diagonal, store the index and go to next row
                if (j == i) {
                    diagIndex[i] = ij;
                    break;
                }
                // if an element beyond the diagonal is reach, no diagonal was found
                // throw an error now. TODO: perform reordering earlier to prevent this
                if (j > i) {
                    printf("ERROR: could not find diagonal value in row %d.\n", i);
                    return false;
                }

                LSize++;
                // calculate the pivot of this row
                blockMult(LUMat->nnzValues[ij], invDiagVals[j], pivot);

                for (k = 0; k < blockSquare; k++)
                { LUMat->nnzValues[ij][k] = pivot[k]; }

                jRowEnd = LUMat->rowPointers[j + 1];
                jk = diagIndex[j] + 1;
                ik = ij + 1;
                // substract that row scaled by the pivot from this row.
                while (ik < iRowEnd && jk < jRowEnd) {
                    if (LUMat->colIndices[ik] == LUMat->colIndices[jk]) {
                        blockMult(pivot, LUMat->nnzValues[jk], tempBlock);
                        blockSub(LUMat->nnzValues[ik], tempBlock, LUMat->nnzValues[ik]);
                        ik++;
                        jk++;
                    } else {
                        if (LUMat->colIndices[ik] < LUMat->colIndices[jk])
                        { ik++; }
                        else
                        { jk++; }
                    }
                }
            }
            // store the inverse in the diagonal!
            blockInvert3x3(LUMat->nnzValues[ij], invDiagVals[i]);
            for (k = 0; k < blockSquare; k++) {
                LUMat->nnzValues[ij][k] = invDiagVals[i][k];
            }
        }

        LMat->rowPointers[0] = 0;
        UMat->rowPointers[0] = 0;

        // Split the LU matrix into two by comparing column indices to diagonal indices
        for (i = 0; i < LUMat->rowSize; i++) {
            LMat->rowPointers[i + 1] = LMat->rowPointers[i];
            for (j = LUMat->rowPointers[i]; j < LUMat->rowPointers[i + 1]; j++) {
                if (j < diagIndex[i]) {
                    for (k = 0; k < blockSquare; k++) {
                        LMat->nnzValues[LMat->rowPointers[i + 1]][k] = LUMat->nnzValues[j][k];
                    }
                    LMat->colIndices[LMat->rowPointers[i + 1]] = LUMat->colIndices[j];
                    LMat->rowPointers[i + 1] = LMat->rowPointers[i + 1] + 1;
                }
            }
        }
        // Reverse the order or the (blocked) rows for the U matrix,
        // because the rows are accessed in reverse order when applying the ILU0
        int URowIndex = 0;
        for (i = LUMat->rowSize - 1; i >= 0; i--) {
            UMat->rowPointers[URowIndex + 1] = UMat->rowPointers[URowIndex];
            for (j = LUMat->rowPointers[i]; j < LUMat->rowPointers[i + 1]; j++) {
                if (j > diagIndex[i]) {
                    for (k = 0; k < blockSquare; k++) {
                        UMat->nnzValues[UMat->rowPointers[URowIndex + 1]][k] = LUMat->nnzValues[j][k];
                    }
                    UMat->colIndices[UMat->rowPointers[URowIndex + 1]] = LUMat->colIndices[j];
                    UMat->rowPointers[URowIndex + 1] = UMat->rowPointers[URowIndex + 1] + 1;
                }
            }
            URowIndex++;
        }

#if PRINT_TIMERS
        t2 = second();
        printf("ilu decomposition: %f s\n", t2 - t1);
#endif

        int *URowsPerColor = nullptr;
        rowSize = BLOCK_SIZE * rMat->rowSize;
        LRowSize = BLOCK_SIZE * LMat->rowSize;
        URowSize = BLOCK_SIZE * UMat->rowSize;
        LNumColors = numColors;
        UNumColors = numColors;
        URowsPerColor = new int[numColors];
        for (int c = 0; c < numColors; c++) {
            URowsPerColor[numColors - c - 1] = rowsPerColor[c];
        }
        int err;
        err = BlockedMatrixFpgaToRDF(rMat, numColors, rowsPerColor, /*isUMatrix:*/ false, 
                                 colIndicesInColor, maxNNZsPerRow,
                                 nnzSplit, /*readInterleaved:*/ false, readBatchSize, nnzValsSizes,
                                 nnzValues, colIndices, NROffsets, colorSizes, &valSize);
        if (err != 0) {
            delete[] URowsPerColor;
            return false;
        }
        err = BlockedMatrixFpgaToRDF(LMat, LNumColors, rowsPerColor, /*isUMatrix:*/ false, 
                                    LColIndicesInColor, maxNNZsPerRow,
                                    nnzSplit, /*readInterleaved:*/ false, readBatchSize, LnnzValsSizes,
                                    LnnzValues, LColIndices, LNROffsets, LColorSizes, &LValSize);
        if (err != 0) {
            delete[] URowsPerColor;
            return false;
        }
        err = BlockedMatrixFpgaToRDF(UMat, UNumColors, URowsPerColor, /*isUMatrix:*/ true, 
                                 UColIndicesInColor, maxNNZsPerRow,
                                 nnzSplit, /*readInterleaved:*/ false, readBatchSize, UnnzValsSizes,
                                 UnnzValues, UColIndices, UNROffsets, UColorSizes, &UValSize);
        if (err != 0) {
            delete[] URowsPerColor;
            return false;
        }
        blockedDiagtoRDF(invDiagVals, rMat->rowSize, numColors, URowsPerColor, blockDiag);
        delete[] URowsPerColor;
        // resultPointers are set in the init method
        resultSizes[0] = rowSize;
        resultSizes[1] = colorSizes[3]; // zeropadded valSize;
        resultSizes[2] = numColors;
        resultSizes[3] = colorSizes[2]; //totalCols
        resultSizes[4] = colorSizes[5]; //NRFlagSize
        resultSizes[5] = colorSizes[6]; //diagValsSize
        resultSizes[6] = LRowSize;
        resultSizes[7] = LColorSizes[3]; // zeropadded LValSize;
        resultSizes[8] = LNumColors;
        resultSizes[9] = LColorSizes[2]; //LTotalCols
        resultSizes[10] = LColorSizes[5]; //LNRFlagSize
        resultSizes[11] = LColorSizes[6]; //LDiagValsSize
        resultSizes[12] = URowSize;
        resultSizes[13] = UColorSizes[3]; // zeropadded UValSize;
        resultSizes[14] = UNumColors;
        resultSizes[15] = UColorSizes[2]; //UTotalCols
        resultSizes[16] = UColorSizes[5]; //UNRFlagSize
        resultSizes[17] = UColorSizes[6]; //UDiagValsSize
        return true;
    } // end create_preconditioner()

// memcpy might be faster
#define COPY_LOOP 1

    void FPGABILU0::apply(double *x, double *y)
    {
        int i, j, k;
        double temp[BLOCK_SIZE];

        // Do forward substitution:
        for (i = 0; i < LMat->rowSize; i++) {
#if COPY_LOOP
            for (j = 0; j < BLOCK_SIZE; j++) {
                temp[j] = x[BLOCK_SIZE * i + j];
            }
#else
            memcpy(temp, x + BLOCK_SIZE * i, BLOCK_SIZE * sizeof(double));
#endif
            for (k = LMat->rowPointers[i]; k < LMat->rowPointers[i + 1]; k++) {
                blockVectMult(LMat->nnzValues[k], &(y[LMat->colIndices[k] * BLOCK_SIZE]), -1.0, temp, false);
            }
#if COPY_LOOP
            for (j = 0; j < BLOCK_SIZE; j++) {
                y[BLOCK_SIZE * i + j] = temp[j];
            }
#else
            memcpy(y + BLOCK_SIZE * i, temp, BLOCK_SIZE * sizeof(double));
#endif
        }
        // Do backward substitution
        for (i = 0; i < UMat->rowSize; i++) {
            int index = UMat->rowSize - i - 1;
#if COPY_LOOP
            for (j = 0; j < BLOCK_SIZE; j++) {
                temp[j] = y[BLOCK_SIZE * index + j];
            }
#else
            memcpy(temp, y + BLOCK_SIZE * i, BLOCK_SIZE * sizeof(double));
#endif
            // order should not matter here, but could introduce LSB rounding errors
            for (k = UMat->rowPointers[i + 1] - 1; k >= UMat->rowPointers[i]; k--) {
                blockVectMult(UMat->nnzValues[k], &(y[UMat->colIndices[k] * BLOCK_SIZE]), -1.0, temp, false);
            }
            blockVectMult(invDiagVals[index], &(temp[0]), 1.0, &(y[index * BLOCK_SIZE]), true);
        }
    } // end apply()

} //namespace bda
