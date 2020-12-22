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
#include <algorithm> // for fill()

#include <opm/simulators/linalg/bda/FPGAReorder.hpp>
#include <opm/simulators/linalg/bda/FPGABlockedMatrix.hpp>

#define DETERMINISTIC_RANDOM 1

namespace bda
{

#if DETERMINISTIC_RANDOM
int rand_state = 0;

int det_rand(){
    int x = rand_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rand_state = x;
}
#endif

int findPreviousReordering(int *colIndices, int *rowPointers, int rowSize, int **rowsPerColor, int maxRowsPerColor, int maxColsPerColor, int maxColors) {
    int rowIndex, valIndex, colIndex;
    int currentColor = 0, startingRow = 0, endColumn = rowSize;
    int *tempRowsPerColor = (int *) malloc(sizeof(int) * rowSize);

    for(rowIndex = 0; rowIndex < rowSize; rowIndex++) {
        // If a row is reached that a previous row in the current color was dependent on, 
        // start a new color with that row as first row.
        if(rowIndex >= endColumn) {
            tempRowsPerColor[currentColor] = rowIndex - startingRow;
            currentColor++;
            endColumn = rowSize;
            startingRow = rowIndex;
        }
        for(valIndex = rowPointers[rowIndex]; valIndex < rowPointers[rowIndex + 1]; valIndex++) {
            colIndex = colIndices[valIndex];
            if(colIndex > rowIndex) {
                if(colIndex < endColumn) {
                    endColumn = colIndex;
                }
                break;
            }
            // If the current row is dependent on a previous row inside the current color,
            // start a new color with that row as first row.
            if(colIndex != rowIndex && colIndex >= startingRow) {
                tempRowsPerColor[currentColor] = rowIndex - startingRow;
                currentColor++;
                endColumn = rowSize;
                startingRow = rowIndex;
            }
        }
    }
    tempRowsPerColor[currentColor] = rowIndex - startingRow;
    currentColor++;
    *rowsPerColor = (int *)malloc(sizeof(int) * currentColor);
    for(int i = 0; i < currentColor; i++) {
        rowsPerColor[0][i] = tempRowsPerColor[i];
    }
    free(tempRowsPerColor);
    return currentColor;
}

int findPreviousReordering(BlockedMatrixFpga *mat, int **rowsPerColor, int maxRowsPerColor, int maxColsPerColor, int maxColors) {
    return findPreviousReordering(mat->colIndices, mat->rowPointers, mat->rowSize, rowsPerColor, maxRowsPerColor, maxColsPerColor, maxColors);
}

int findPreviousReordering(Matrix *mat, int **rowsPerColor, int maxRowsPerColor, int maxColsPerColor, int maxColors) {
    return findPreviousReordering(mat->colIndices, mat->rowPointers, mat->rowSize, rowsPerColor, maxRowsPerColor, maxColsPerColor, maxColors);
}

/* Give every node in the matrix (of which only the sparsity pattern in the 
 * form of row pointers and column indices arrasy are in the input), a color 
 * in the colors array. Also return the amount of colors in the return integer. */

int colorBlockedNodes(int rows, const int *rowPointers, const int *colIndices, int *colors, int maxRowsPerColor, int maxColsPerColor) {
    int left, c, t, i, j, k;
    int *randoms; // allocate and init random array 
    randoms = (int*) malloc(sizeof(int)*rows);

    bool *visitedColumn = (bool *)malloc(sizeof(bool) * rows);
    int colsInColor;
    int additionalColsInRow;

    for(t = 0; t < MAX_TRIES; t++) {
        #if DETERMINISTIC_RANDOM
            rand_state = 0x64a62455;
            for(i = 0; i < rows; i++) {
                randoms[i] = det_rand();
                colors[i] = -1;
            }
        #else
            struct timeval tm;
            gettimeofday(&tm, NULL);
            srand(tm.tv_sec + tm.tv_usec * 1000000ul);
            for(i = 0; i  < rows; i++) {
                randoms[i] = rand();
                colors[i] = -1;
            }
        #endif
        for(c = 0; c < MAX_COLORS; c++) {
            int rowsInColor = 0;
            colsInColor = 0;
            memset(visitedColumn, 0, sizeof(bool) * rows);
            // for (int index = 0; index < rows; index++)
            for (i = 0; i < rows; i++) {
                char f = 1; // true iff you have max random
                // ignore nodes colored earlier
                if ((colors[i] != -1)) {
                    continue;
                }
                int ir = randoms[i];
                // look at neighbors to check their random number
                for (k = rowPointers[i]; k < rowPointers[i+1]; k++) {
                    // ignore nodes colored earlier (and yourself)
                    j = colIndices[k]; 
                    int jc = colors[j];
                    if (((jc != -1) && (jc != c))|| (i == j)) {
                        continue;
                    }
                    // The if statement below makes it both true graph coloring and no longer guaranteed to converge
                    if(jc == c) {
                        f = 0;
                        break;
                    }
                    int jr = randoms[j];
                    if (ir <= jr) {
                        f = 0;
                    }
                }
                // assign color if you have the maximum random number
                if (f == 1)
                {
                    additionalColsInRow = 0;
                    for(k = rowPointers[i]; k < rowPointers[i+1]; k++) {
                        j = colIndices[k];
                        if(!visitedColumn[j]) {
                            visitedColumn[j] = true;
                            additionalColsInRow+=3;
                        }
                    }
                    // Breaking off color because it already has enough columns
                    if((colsInColor + additionalColsInRow) > maxColsPerColor) {
                        break;
                    }
                    colsInColor += additionalColsInRow;
                    colors[i] = c;
                    rowsInColor+=3;
                    // Breaking off color because it already has enough rows
                    if((rowsInColor+2) >= maxRowsPerColor) {
                        break;
                    }
                }
            }
            // Check if graph coloring is done.
            left = 0;
            for(k = 0; k < rows; k++) {
                if(colors[k] == -1) {
                    left++;
                }
            }
            if (left == 0) {
                free(visitedColumn);
                free(randoms);
                return c+1;
            }
        }
    }
    free(visitedColumn);
    free(randoms);
    //FIXME
    //<< "Could not find a graph coloring with " << c " colors after " << t << " tries.\n";
    //<< "Amount of colorless nodes: " << left << ".\n";
    return -1;
}

/* Reorder a matrix by a specified input order. 
 * Both a to order array, which contains for every node from the old matrix where it will move in the new matrix, 
 * and the from order, which contains for every node in the new matrix where it came from in the old matrix.*/

void reorder_matrix_by_pattern(Matrix *mat, int *toOrder, int *fromOrder, Matrix *rMat){
    int rIndex = 0;
    int i, k;

    rMat->rowPointers[0] = 0;
    for(i = 0; i < mat->rowSize; i++) {
        int thisRow = fromOrder[i];
        // put thisRow from the old matrix into row i of the new matrix
        rMat->rowPointers[i+1] = rMat->rowPointers[i] + mat->rowPointers[thisRow+1] - mat->rowPointers[thisRow];
        for(k = mat->rowPointers[thisRow]; k < mat->rowPointers[thisRow+1]; k++) {
            rMat->nnzValues[rIndex] = mat->nnzValues[k];
            rMat->colIndices[rIndex] = mat->colIndices[k];
            rIndex++;
        }
    }
    // re-assign column indices according to the new positions of the nodes referenced by the column indices
    for(i = 0; i < mat->valSize; i++) {
        rMat->colIndices[i] = toOrder[rMat->colIndices[i]];
    }
    // re-sort the column indices of every row.
    for(i = 0; i < mat->rowSize; i++) {
        sortRow(rMat->colIndices, rMat->nnzValues, rMat->rowPointers[i], rMat->rowPointers[i+1]-1);
    }
}

/* Reorder a matrix by a specified input order. 
 * Both a to order array, which contains for every node from the old matrix where it will move in the new matrix, 
 * and the from order, which contains for every node in the new matrix where it came from in the old matrix.*/

void reorder_blocked_matrix_by_pattern(BlockedMatrixFpga *mat, int *toOrder, int *fromOrder, BlockedMatrixFpga *rMat){
    int rIndex = 0;
    int i, j, k;

    rMat->rowPointers[0] = 0;
    for(i = 0; i < mat->rowSize; i++) {
        int thisRow = fromOrder[i];
        // put thisRow from the old matrix into row i of the new matrix
        rMat->rowPointers[i+1] = rMat->rowPointers[i] + mat->rowPointers[thisRow+1] - mat->rowPointers[thisRow];
        for(k = mat->rowPointers[thisRow]; k < mat->rowPointers[thisRow+1]; k++) {
            for(j = 0; j < BLOCK_SIZE * BLOCK_SIZE; j++) {
                rMat->nnzValues[rIndex][j] = mat->nnzValues[k][j];
            }
            rMat->colIndices[rIndex] = mat->colIndices[k];
            rIndex++;
        }
    }
    // re-assign column indices according to the new positions of the nodes referenced by the column indices
    for(i = 0; i < mat->valSize; i++) {
        rMat->colIndices[i] = toOrder[rMat->colIndices[i]];
    }
    // re-sort the column indices of every row.
    for(i = 0; i < mat->rowSize; i++) {
        sortBlockedRow(rMat->colIndices, rMat->nnzValues, rMat->rowPointers[i], rMat->rowPointers[i+1]-1);
    }
}

/* Reorder a matrix according to the colors that every node of the matrix has received */

void colorsToReordering(int rowSize, int *colors, int *toOrder, int *fromOrder, int *iters){
    int reordered = 0;
    int i, c;

    for(i = 0; i <  MAX_COLORS; i++) {
        iters[i] = 0;
    }
    // Find reordering patterns
    for(c = 0; c <  MAX_COLORS; c++) {
        for(i = 0; i < rowSize; i++) {
            if(colors[i] == c) {
                iters[c]++;
                toOrder[i] = reordered;
                fromOrder[reordered] = i;
                reordered++;
            }
        }
    }
}

/* Reorder a matrix according to a reordering pattern */

void reorder_vector_by_blocked_pattern(int size, double *vector, int *fromOrder, double *rVector) {
    int i, j;

    for(i = 0; i < size; i++) {
        for(j = 0; j < BLOCK_SIZE; j++) {
            rVector[BLOCK_SIZE * i + j] = vector[BLOCK_SIZE * fromOrder[i] + j];
        }
    }
}

void reorder_vector_by_pattern(int size, double *vector, int *fromOrder, double *rVector) {
    int i;

    for(i = 0; i < size; i++) {
        rVector[i] = vector[fromOrder[i]]; 
    }
}

/* Check is operations on a node in the matrix can be started 
 * A node can only be started if all nodes that it depends on during sequential execution have already completed. */

char canBeStarted(int rowIndex, int *rowPointers, int *colIndices, char *doneRows) {
    char canStart = !doneRows[rowIndex];
    int i, thisDependency;
    if(canStart) {
        for(i = rowPointers[rowIndex]; i < rowPointers[rowIndex + 1]; i++) {
            thisDependency = colIndices[i];
            // Only dependencies on rows that should execute before the current one are relevant
            if(thisDependency >= rowIndex) {
                break;
            }
            // Check if dependency has been resolved
            if(!doneRows[thisDependency]) {
                canStart = 0;
                break;
            }
        }
    }
    return canStart;
}

/*
 * The level scheduling of a non-symmetric, blocked matrix requires access to a CSC encoding and a CSR encoding of the same matrix.
*/

int *findLevelScheduling(int *CSRColIndices, int *CSRRowPointers, int *CSCColIndices, int *CSCRowPointers, int rowSize, int *iters, int *toOrder, int* fromOrder){
    int activeRowIndex = 0, iterEnd, nextActiveRowIndex = 0;
    int  i, thisRow, thatRow;
    char *doneRows;
    int *rowsPerIter, *resRowsPerIter;
    int *rowsToStart;
    int rowsFound;

    // initialize searching arrays
    doneRows = (char *)malloc(sizeof(char) * rowSize);
    rowsPerIter = (int *)malloc(sizeof(int) * rowSize);
    rowsToStart = (int *)malloc(sizeof(int) * MAX_PARALLELISM);
    memset(doneRows, 0, sizeof(char) * rowSize);
    // find starting rows: rows that are independent from all rows that come before them.
    for(thisRow = 0; thisRow < rowSize; thisRow++) {
        if(canBeStarted(thisRow, CSRRowPointers, CSCColIndices, doneRows)) {
            fromOrder[nextActiveRowIndex] = thisRow;
            toOrder[thisRow] = nextActiveRowIndex;
            nextActiveRowIndex++;
        }
    }
    // 'do' compute on all active rows
    for(iterEnd = 0;iterEnd < nextActiveRowIndex; iterEnd++) {
        doneRows[fromOrder[iterEnd]] = 1;
    }
    rowsPerIter[0] = nextActiveRowIndex - activeRowIndex;
    *iters = 1;
    while(iterEnd < rowSize) {
        rowsFound = 0;
        // Go over all rows active from the last iteration, and check which of their neighbours can be activated this iteration
        for(; activeRowIndex < iterEnd; activeRowIndex++) {
            thisRow = fromOrder[activeRowIndex];
            for(i = CSCRowPointers[thisRow]; i < CSCRowPointers[thisRow + 1]; i++) {
                thatRow = CSCColIndices[i];
                if(canBeStarted(thatRow, CSRRowPointers, CSRColIndices, doneRows)) {
                    rowsToStart[rowsFound] = thatRow;
                    rowsFound++;
                }
            }
        }
        // 'do' compute on all active rows
        for(i = 0; i < rowsFound; i++) {
            thisRow = rowsToStart[i];
            if(!doneRows[thisRow]) {
                doneRows[thisRow] = 1;
                fromOrder[nextActiveRowIndex] = thisRow;
                toOrder[thisRow] = nextActiveRowIndex;
                nextActiveRowIndex++;
            }
        }
        iterEnd = nextActiveRowIndex;
        rowsPerIter[*iters] = nextActiveRowIndex - activeRowIndex;
        *iters = *iters + 1;
    }
    // Crop the rowsPerIter array to it minimum size.
    resRowsPerIter = (int *)malloc(sizeof(int) * *iters);
    for(i = 0; i  < *iters; i++) {
        resRowsPerIter[i] = rowsPerIter[i];
    }
    // free searching arrays
    free(rowsPerIter);
    free(doneRows);
    free(rowsToStart);
    return resRowsPerIter;
}

/* Perform the complete graph coloring algorithm on a matrix. Return an array with the amount of nodes per color. */

int* findGraphColoring(int *colIndices, int *rowPointers, int rowSize, int maxRowsPerColor, int maxColsPerColor, int *numColors, int *toOrder, int* fromOrder){
    int *rowColor = (int *)malloc(sizeof(int) * rowSize);
    int *rowsPerColor = (int *)malloc(sizeof(int) * MAX_COLORS);

    if(colorBlockedNodes(rowSize, rowPointers, colIndices, rowColor, maxRowsPerColor, maxColsPerColor)== -1) {
        return NULL;
    }
    colorsToReordering(rowSize, rowColor, toOrder, fromOrder, rowsPerColor);
    *numColors = MAX_COLORS;
    while(rowsPerColor[*numColors - 1] == 0) {
        *numColors = *numColors - 1;
    }
    free(rowColor);
    return rowsPerColor;
}

// based on the scipy package from python, scipy/sparse/sparsetools/csr.h on github
// input : matrix A via Avals, Acols, Arows, numRows
// output: matrix B via Bvals, Bcols, Brows
// arrays for B must be preallocated
void bcsrToBcsc(Block *Avals, int *Acols, int *Arows, Block *Bvals, int *Bcols, int *Brows, int numRows){
    int nnz = Arows[numRows];

    // compute number of nnzs per column
    std::fill(Brows, Brows + numRows, 0);
    for(int n = 0; n < nnz; ++n) {
        Brows[Acols[n]]++;
    }
    // cumsum the nnz per col to get Brows
    for(int col = 0, cumsum = 0; col < numRows; ++col) {
        int temp = Brows[col];
        Brows[col] = cumsum;
        cumsum += temp;
    }
    Brows[numRows] = nnz;
    for(int row = 0; row < numRows; ++row) {
        for(int j = Arows[row]; j < Arows[row+1]; ++j) {
            int col = Acols[j];
            int dest = Brows[col];
            Bcols[dest] = row;
            memcpy(Bvals+dest, Avals+dest, sizeof(Block));
            Brows[col]++;
        }
    }
    for(int col = 0, last = 0; col <= numRows; ++col) {
        int temp = Brows[col];
        Brows[col] = last;
        last = temp;
    }
}

} //namespace bda
