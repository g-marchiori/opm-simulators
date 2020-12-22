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

#include <cstdio>
#include <cstring>
#include <sys/time.h>
#include <cmath>
#include <iostream>

#include <opm/simulators/linalg/bda/FPGABlockedMatrix.hpp>
#include <opm/simulators/linalg/bda/FPGAMatrix.hpp>
#include <opm/simulators/linalg/bda/FPGAUtils.hpp>

using bda::Matrix;
using bda::BlockedMatrixFpga;

namespace bda
{

BlockedMatrixFpga *allocateBlockedMatrixFpga(int rowSize, int valSize){
    BlockedMatrixFpga *mat = (BlockedMatrixFpga *)malloc(sizeof(BlockedMatrixFpga));
    mat->nnzValues  = (Block *)malloc(sizeof(Block) * valSize);
    mat->colIndices = (int *)malloc(sizeof(int) * valSize);
    mat->rowPointers = (int *)malloc(sizeof(int) * (rowSize+1));
    mat->rowSize = rowSize;
    mat->valSize = valSize;
    return mat;
}

void freeBlockedMatrixFpga(BlockedMatrixFpga **mat){
    if (*mat) {
        if ((*mat)->nnzValues) free((*mat)->nnzValues);
        if ((*mat)->colIndices) free((*mat)->colIndices);
        if ((*mat)->rowPointers) free((*mat)->rowPointers);
        free(*mat);
        *mat = NULL;
    }
}

BlockedMatrixFpga *soft_copyBlockedMatrixFpga(BlockedMatrixFpga *mat){
    BlockedMatrixFpga *res = (BlockedMatrixFpga *)malloc(sizeof(BlockedMatrixFpga));
    res->nnzValues = mat->nnzValues;
    res->colIndices = mat->colIndices;
    res->rowPointers = mat->rowPointers;
    res->rowSize = mat->rowSize;
    res->valSize = mat->valSize;
    return res;
}

int findPartitionColumns(BlockedMatrixFpga *mat, int numColors, int *nodesPerColor,
        int rowsPerColorLimit, int columnsPerColorLimit,
        int **colIndicesInColor, int *PIndicesAddr, int *colorSizes,
        int **LColIndicesInColor, int *LPIndicesAddr, int *LColorSizes,
        int **UColIndicesInColor, int *UPIndicesAddr, int *UColorSizes){
    int i, row, col, c;
    // Data related to column indices per partition
    int doneRows = 0;
    int numColAccesses, LNumColAccesses, UNumColAccesses;
    char *isColAccessed = (char*) malloc(sizeof(char) * mat->rowSize);
    char *isLColAccessed = (char*) malloc(sizeof(char) * mat->rowSize);
    int totalCols = 0, LTotalCols = 0, UTotalCols = 0, maxCols = 0;
    int numRows, maxRowsPerColor = 0;
    int colsPerRow, maxColsPerRow = 0;
    int *colsPerColor = (int *) malloc(sizeof(int) * numColors);
    int **colsInColor = (int **) malloc(sizeof(int *) * numColors);
    int *LColsPerColor = (int *) malloc(sizeof(int) * numColors);
    int **LColsInColor = (int **) malloc(sizeof(int *) * numColors);
    int *UColsPerColor = (int *) malloc(sizeof(int) * numColors);
    int **UColsInColor = (int **) malloc(sizeof(int *) * numColors);
    // Find which columns are accessed in each color, as well as how many non-zeroes there are per color.
    for(c = 0; c < numColors; c++){
        numRows = 0;
        colsInColor[c] = (int *) malloc(sizeof(int) * roundUpTo(mat->rowSize, 16)); // 8192 for level scheduling
        LColsInColor[c] = (int *) malloc(sizeof(int) * roundUpTo(mat->rowSize, 16)); // 8192 for level scheduling
        UColsInColor[numColors - c - 1] = (int *) malloc(sizeof(int) * roundUpTo(mat->rowSize, 16)); // 8192 for level scheduling
        for(row = 0; row < mat->rowSize; row++){
            isColAccessed[row] = (char) 0;
            isLColAccessed[row] = (char) 0;
            for(i = 0; i <  BLOCK_SIZE; i++){
                colIndicesInColor[c][row * BLOCK_SIZE + i] = 0;
                LColIndicesInColor[c][row * BLOCK_SIZE + i] = 0;
                UColIndicesInColor[numColors - c - 1][row * BLOCK_SIZE + i] = 0;
            }
        }
        if(c > 0){
            for(i = doneRows - nodesPerColor[c - 1]; i < doneRows; i++){
                isLColAccessed[i] = (char) 1;
            }
        }
        numColAccesses = 0;
        LNumColAccesses = 0;
        UNumColAccesses = 0;
        // Go over every row in this color:
        for(row = doneRows; row < doneRows + nodesPerColor[c]; row++){
            colsPerRow = 0;
            char isRowEmpty = 1;
            for(int val = mat->rowPointers[row]; val < mat->rowPointers[row + 1]; val++){
                // For every column in the current row, check if that column was accessed before this color:
                if(isColAccessed[mat->colIndices[val]] == (char) 0 ){
                    colsInColor[c][numColAccesses] = mat->colIndices[val];
                    isColAccessed[mat->colIndices[val]] = (char) 1;
                    numColAccesses ++;
                    if(mat->colIndices[val] > row){
                        UColsInColor[numColors - c - 1][UNumColAccesses] = mat->colIndices[val];
                        UNumColAccesses++;
                    }
                }
                if(isLColAccessed[mat->colIndices[val]] == (char) 0 ){
                    if(mat->colIndices[val] < row){
                        LColsInColor[c][LNumColAccesses] = mat->colIndices[val];
                        LNumColAccesses++;
                        isLColAccessed[mat->colIndices[val]] = (char) 1;
                    }
                }
                colsPerRow++;
                isRowEmpty = (char) 0;
            }
            if(isRowEmpty != (char) 1)
                numRows++;
            if(colsPerRow > maxColsPerRow)
                maxColsPerRow = colsPerRow;
        }
        // Add columns from previous color into L partition to simplify data forwarding
        if(c > 0){
            for(i = doneRows - nodesPerColor[c - 1]; i < doneRows; i++){
                LColsInColor[c][LNumColAccesses] = i;
                LNumColAccesses ++;
            }
        }
        // Zeropad the colsInColor number to the nearest multiple of 16, because there are 16 32-bit color_col_index values per cacheline
        colorSizes[c * 4 + 10] = numColAccesses*BLOCK_SIZE;
        LColorSizes[c * 4 + 10] = LNumColAccesses*BLOCK_SIZE;
        UColorSizes[(numColors - c - 1) * 4 + 10] = UNumColAccesses*BLOCK_SIZE;
        for(col = 0; col < numColAccesses; col++){
            if(colsInColor[c][col] >= 0){
                for(i = 0; i < BLOCK_SIZE; i++){
                    colIndicesInColor[c][colsInColor[c][col]*BLOCK_SIZE + i] = col*BLOCK_SIZE + i;
                }
            }
        }
        for(col = 0; col < LNumColAccesses; col++){
            if(LColsInColor[c][col] >= 0){
                for(i = 0; i < BLOCK_SIZE; i++){
                    LColIndicesInColor[c][LColsInColor[c][col]*BLOCK_SIZE + i] = col*BLOCK_SIZE + i;
                }
            }
        }
        for(col = 0; col < UNumColAccesses; col++){
            if(UColsInColor[numColors - c - 1][col] >= 0){
                for(i = 0; i < BLOCK_SIZE; i++){
                    UColIndicesInColor[numColors - c - 1][UColsInColor[numColors - c - 1][col]*BLOCK_SIZE + i] = col*BLOCK_SIZE + i;
                }
            }
        }
        while(numColAccesses%16 != 0){
            colsInColor[c][numColAccesses] = colsInColor[c][numColAccesses - 1]; 
            numColAccesses++;
        }
        while(LNumColAccesses%16 != 0){
            LColsInColor[c][LNumColAccesses] = LColsInColor[c][LNumColAccesses - 1]; 
            LNumColAccesses++;
        }
        while(UNumColAccesses%16 != 0){
            UColsInColor[numColors - c - 1][UNumColAccesses] = UColsInColor[numColors - c - 1][UNumColAccesses - 1]; 
            UNumColAccesses++;
        }
        colsPerColor[c] = numColAccesses;
        LColsPerColor[c] = LNumColAccesses;
        UColsPerColor[numColors - c - 1] = UNumColAccesses;
        if(numColAccesses > maxCols)
            maxCols = numColAccesses;
        totalCols += numColAccesses;
        LTotalCols += LNumColAccesses;
        UTotalCols += UNumColAccesses;
        doneRows = row;
        if(numRows > maxRowsPerColor)
            maxRowsPerColor = numRows;
    }
    if(maxCols * BLOCK_SIZE > columnsPerColorLimit){
        printf("ERROR: Current reordering exceeds maximum number of columns per color limit: %d/%d.\n", maxCols * BLOCK_SIZE, columnsPerColorLimit);
        free(isColAccessed);
        free(colsPerColor);
        free(LColsPerColor);
        free(UColsPerColor);
        for(c = 0; c < numColors; c++){
            free(colsInColor[c]);
            free(LColsInColor[c]);
            free(UColsInColor[c]);
        }
        free(colsInColor);
        free(LColsInColor);
        free(UColsInColor);
        return -1;
    }
    doneRows = 0;
    int diagValsSize = 0;
    int maxRows = 0;
    // Go through all rows to determine the row offset of each row (amount of zero rows between it and the previous non-zero row)
    for(c = 0; c < numColors; c++){
        // Calculate sizes that include zeropadding
        diagValsSize += roundUpTo(nodesPerColor[c]*BLOCK_SIZE * 4, 8);
        doneRows += nodesPerColor[c];
        if(nodesPerColor[c] * BLOCK_SIZE > maxRows)
            maxRows = nodesPerColor[c];
        colorSizes[c * 4 + 9] = nodesPerColor[c]*BLOCK_SIZE;
        LColorSizes[c * 4 + 9] = nodesPerColor[c]*BLOCK_SIZE;
        UColorSizes[c * 4 + 9] = nodesPerColor[numColors - c - 1]*BLOCK_SIZE;
    }
    if(maxRows * BLOCK_SIZE > rowsPerColorLimit){
        printf("ERROR: Current reordering exceeds maximum number of rows per color limit: %d/%d.\n", maxRows * BLOCK_SIZE, rowsPerColorLimit);
        free(isColAccessed);
        free(colsPerColor);
        free(LColsPerColor);
        free(UColsPerColor);
        for(c = 0; c < numColors; c++){
            free(colsInColor[c]);
            free(LColsInColor[c]);
            free(UColsInColor[c]);
        }
        free(colsInColor);
        free(LColsInColor);
        free(UColsInColor);
        return -1;
    }
    // create and fill sizes array as far as already possible
    colorSizes[0] = mat->rowSize*BLOCK_SIZE;
    LColorSizes[0] = mat->rowSize*BLOCK_SIZE;
    UColorSizes[0] = mat->rowSize*BLOCK_SIZE;
    // col_sizes (but the matrix is square)
    colorSizes[1] = mat->rowSize*BLOCK_SIZE;
    LColorSizes[1] = mat->rowSize*BLOCK_SIZE;
    UColorSizes[1] = mat->rowSize*BLOCK_SIZE;
    colorSizes[2] = totalCols*BLOCK_SIZE;
    LColorSizes[2] = LTotalCols*BLOCK_SIZE;
    UColorSizes[2] = UTotalCols*BLOCK_SIZE;
    //missing val_size
    colorSizes[4] = numColors;
    LColorSizes[4] = numColors;
    UColorSizes[4] = numColors;
    //missing: NRFlagsSize
    colorSizes[6] = diagValsSize;
    LColorSizes[6] = diagValsSize;
    UColorSizes[6] = diagValsSize;
    while(c % 4 != 0){
        for(i = 0; i < 4; i ++){
            colorSizes[c*4 + 8 + i] = 0;
            LColorSizes[c*4 + 8 + i] = 0;
            UColorSizes[c*4 + 8 + i] = 0;
        }
        c++;
    }
    int index = 0, Lindex = 0, Uindex = 0;
    col = 0;
    for(c = 0; c < numColors; c++){
        for(col = 0; col < colorSizes[c * 4 + 10] / BLOCK_SIZE ; col++){
            for(i = 0; i < BLOCK_SIZE; i++){
                PIndicesAddr[index] = colsInColor[c][col] * BLOCK_SIZE + i;
                index++;
            }
        }
        while(index % 16 != 0){
            PIndicesAddr[index] = PIndicesAddr[index - 1];
            index++;
        }
        for(col = 0; col < LColorSizes[c * 4 + 10] / BLOCK_SIZE ; col++){
            for(i = 0; i < BLOCK_SIZE; i++){
                LPIndicesAddr[Lindex] = LColsInColor[c][col] * BLOCK_SIZE + i;
                Lindex++;
            }
        }
        while(Lindex % 16 != 0){
            LPIndicesAddr[Lindex] = LPIndicesAddr[Lindex - 1];
            Lindex++;
        }
        for(col = 0; col < UColorSizes[c * 4 + 10] / BLOCK_SIZE ; col++){
            for(i = 0; i < BLOCK_SIZE; i++){
                UPIndicesAddr[Uindex] = UColsInColor[c][col] * BLOCK_SIZE + i;
                Uindex++;
            }
        }
        while(Uindex % 16 != 0){
            UPIndicesAddr[Uindex] = UPIndicesAddr[Uindex - 1];
            Uindex++;
        }
    }
    free(isColAccessed);
    free(colsPerColor);
    free(LColsPerColor);
    free(UColsPerColor);
    for(c = 0; c < numColors; c++){
        free(colsInColor[c]);
        free(LColsInColor[c]);
        free(UColsInColor[c]);
    }
    free(colsInColor);
    free(LColsInColor);
    free(UColsInColor);
    return 0;
}

/*
 * Unblock the blocked matrix. Input the blocked matrix and output a CSR matrix without blocks.
 * If unblocking the U matrix, the rows in all blocks need to written to the new matrix in reverse order.
*/
static void unblock(BlockedMatrixFpga *bMat, Matrix *mat, bool isUMatrix){
    int valIndex = 0, nnzsPerRow;
    int maxRowSize = 0, minRowSize = bMat->rowSize * BLOCK_SIZE;

    for(int row = 0; row < bMat->rowSize; row++){
        for(int col = bMat->rowPointers[row]; col < bMat->rowPointers[row+1]; col++){
            for(int bRow = 0; bRow < BLOCK_SIZE; bRow++){
                for(int bCol = 0; bCol < BLOCK_SIZE; bCol++){
                    if(fabs(bMat->nnzValues[col][bRow * BLOCK_SIZE + bCol]) > 1e-80)
                        valIndex++;
                }
            }
        }
    }
    // Initialize the non-blocked matrix with the obtained size.
    mat->rowSize = bMat->rowSize * BLOCK_SIZE;
    mat->valSize = valIndex;
    mat->nnzValues = (double *) malloc(sizeof(double) * valIndex);
    mat->colIndices = (int *) malloc(sizeof(int) * valIndex);
    mat->rowPointers = (int *) malloc(sizeof(int) * (mat->rowSize + 1));
    valIndex = 0;
    mat->rowPointers[0] = 0;
    // go through the blocked matrix row-by row of blocks, and then row-by-row inside the block, and
    // write all non-zero values and corresponding column indices that belong to the same row into the new matrix.
    for(int row = 0; row < bMat->rowSize; row++){
        for(int bRow = 0; bRow < BLOCK_SIZE; bRow++){
            nnzsPerRow = 0;
            for(int col = bMat->rowPointers[row]; col < bMat->rowPointers[row+1]; col++){
                for(int bCol = 0; bCol < BLOCK_SIZE; bCol++){
                    if(isUMatrix){
                        // If the matrix is the U matrix, store the rows inside a block in reverse order.
                        if(fabs(bMat->nnzValues[col][(BLOCK_SIZE - bRow - 1) * BLOCK_SIZE + bCol]) > 1e-80){
                            mat->nnzValues[valIndex] = bMat->nnzValues[col][(BLOCK_SIZE - bRow - 1) * BLOCK_SIZE + bCol];
                            mat->colIndices[valIndex] = bMat->colIndices[col] * BLOCK_SIZE + bCol;
                            valIndex++;
                            nnzsPerRow++;
                        }
                    }else{
                        if(fabs(bMat->nnzValues[col][bRow*BLOCK_SIZE + bCol]) > 1e-80){
                            mat->nnzValues[valIndex] = bMat->nnzValues[col][bRow * BLOCK_SIZE + bCol];
                            mat->colIndices[valIndex] = bMat->colIndices[col] * BLOCK_SIZE + bCol;
                            valIndex++;
                            nnzsPerRow++;
                        }
                    }
                }
            }
            // Update the rowpointers of the new matrix
            mat->rowPointers[row*BLOCK_SIZE + bRow + 1] = mat->rowPointers[row*BLOCK_SIZE + bRow] + nnzsPerRow;
            if(nnzsPerRow > maxRowSize)
                maxRowSize = nnzsPerRow;
            if(nnzsPerRow < minRowSize)
                minRowSize = nnzsPerRow;
        }
    }
}

void blockedDiagtoRDF(Block *blockedDiagVals, int rowSize, int numColors, int *rowsPerColor, double *RDFDiag){
    int doneRows = rowSize - 1, RDFIndex = 0;
    for(int c = 0; c < numColors; c++){
        for(int r = 0; r < rowsPerColor[c]; r++){
            for(int i = BLOCK_SIZE - 1; i >= 0; i--){
                for(int j = 0; j < BLOCK_SIZE; j++){
                    RDFDiag[RDFIndex] = blockedDiagVals[doneRows - r][i*BLOCK_SIZE + j];
                    RDFIndex++;
                }
                RDFDiag[RDFIndex] = 0.0;
                RDFIndex++;
            }
        }
        doneRows -= rowsPerColor[c];
        if(RDFIndex%8 != 0){
            for(int j = 0; j < 4; j++){
                RDFDiag[RDFIndex] = 0.0;
                RDFIndex++;
            }
        }
    }
}

/*Optimized version*/
int BlockedMatrixFpgaToRDF(BlockedMatrixFpga *mat, int numColors, int *nodesPerColor, bool isUMatrix,
        int **colIndicesInColor, int nnzsPerRowLimit,
        int numFieldSplits, bool readInterleaved, int readBatchSize, int *nnzValsSizes,
        double **nnzValues, short int *colIndices, unsigned char *NROffsets, int *colorSizes,  int *valSize){
    int res;
    Matrix *ubMat = (Matrix *)malloc(sizeof(Matrix));
    unblock(mat, ubMat, isUMatrix);
    int *ubNodesPerColor = (int *) malloc(sizeof(int) * numColors);
    for(int i = 0; i < numColors; i++)
        ubNodesPerColor[i] = nodesPerColor[i] * BLOCK_SIZE;
    *valSize = ubMat->valSize;
    res = matrixToRDF(ubMat, numColors, ubNodesPerColor, 
        colIndicesInColor, nnzsPerRowLimit,
        nnzValues, colIndices, nnzValsSizes,
        NROffsets, colorSizes,
        numFieldSplits, readInterleaved, readBatchSize);
    freeMatrix(&ubMat);
    free(ubNodesPerColor);
    return res;
}

/*Sort a row of matrix elements from a blocked CSR-format.*/
void sortBlockedRow(int *colIndices, Block *data, int left, int right){
    int l = left;
    int r = right;
    int middle = colIndices[(l+r) >> 1];
    double lDatum[9];

    do{
        while(colIndices[l] < middle)
            l++;
        while(colIndices[r] > middle)
            r--;
        if(l <= r){
            int lColIndex = colIndices[l];
            colIndices[l] = colIndices[r];
            colIndices[r] = lColIndex;
            memcpy(lDatum, data+l, sizeof(double)*BLOCK_SIZE * BLOCK_SIZE);
            memcpy(data+l, data+r, sizeof(double)*BLOCK_SIZE * BLOCK_SIZE);
            memcpy(data+r, lDatum, sizeof(double)*BLOCK_SIZE * BLOCK_SIZE);
            l++;
            r--;
        }
    }while (l < r);
    if(left < r)
        sortBlockedRow(colIndices, data, left, r);
    if(right > l)
        sortBlockedRow(colIndices, data, l, right);
}

/*Subtract two blocks from one another element by element*/
void blockSub(Block mat1, Block mat2, Block resMat){
    for(int row = 0; row < BLOCK_SIZE; row++){
        for(int col = 0; col < BLOCK_SIZE; col++){
            resMat[row*BLOCK_SIZE+col] = mat1[row*BLOCK_SIZE + col] - mat2[row*BLOCK_SIZE + col];
        }
    }
}

/*Perform a 3x3 matrix-matrix multiplicationj on two blocks*/
void blockMult(Block mat1, Block mat2, Block resMat){
    for(int row = 0; row < BLOCK_SIZE; row++){
        for(int col = 0; col < BLOCK_SIZE; col++){
            double temp = 0;
            for(int k = 0; k < BLOCK_SIZE; k++){
                temp += mat1[BLOCK_SIZE*row + k] * mat2[BLOCK_SIZE*k + col];
            }
            resMat[BLOCK_SIZE*row+col] = temp;
        }
    }
}

/* Calculate the inverse of a block. This function is specific for only 3x3 block size.*/
int blockInvert3x3(Block mat, Block res){
    int i;
    // code generated by maple, copied from DUNE
    double t4  = mat[0] * mat[4];
    double t6  = mat[0] * mat[5];
    double t8  = mat[1] * mat[3];
    double t10 = mat[2] * mat[3];
    double t12 = mat[1] * mat[6];
    double t14 = mat[2] * mat[6];
    double det = (t4*mat[8]-t6*mat[7]-t8*mat[8]+
                  t10*mat[7]+t12*mat[5]-t14*mat[4]);
    if(det == 0){
        for(i = 0; i < 9; i++)
            printf("%le, ", mat[i]);
        printf("\n");
        printf("Error block not invertible\n");
        exit(1);
    }
    double t17 = 1.0/det;
    res[0] =  (mat[4] * mat[8] - mat[5] * mat[7])*t17;
    res[1] = -(mat[1] * mat[8] - mat[2] * mat[7])*t17;
    res[2] =  (mat[1] * mat[5] - mat[2] * mat[4])*t17;
    res[3] = -(mat[3] * mat[8] - mat[5] * mat[6])*t17;
    res[4] =  (mat[0] * mat[8] - t14) * t17;
    res[5] = -(t6-t10) * t17;
    res[6] =  (mat[3] * mat[7] - mat[4] * mat[6]) * t17;
    res[7] = -(mat[0] * mat[7] - t12) * t17;
    res[8] =  (t4-t8) * t17;
    return 0;
}

/*Multiply a block with a vector block, and add the result, scaled by a constant, to the result vector*/
void blockVectMult(Block mat, double *vect, double scale, double *resVect, bool resetRes){
    for(int row = 0; row < BLOCK_SIZE; row++){
        if(resetRes){
            resVect[row] = 0.0;
        }
        for(int col = 0; col < BLOCK_SIZE; col++){
            resVect[row] += scale * mat[row*BLOCK_SIZE + col] * vect[col];
        }
    }
}

} // end namespace bda
