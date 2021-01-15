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

#include <opm/simulators/linalg/bda/FPGAMatrix.hpp>

namespace bda
{

/* Reserve space for a matrix of a certain size */
Matrix *allocateMatrix(int rowSize, int valSize){
	Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
	mat->nnzValues = (double *)malloc(sizeof(double) * valSize);
	mat->colIndices = (int *)malloc(sizeof(int) * valSize);
	mat->rowPointers = (int *)malloc(sizeof(int) * (rowSize+1));
	mat->rowSize = rowSize;
    // Assumption: all matrices this solver works on are square
	mat->colSize = rowSize;
	mat->valSize = valSize;
	return mat;
}

void freeMatrix(Matrix **mat){
    if (*mat) {
        if ((*mat)->nnzValues) free((*mat)->nnzValues);
        if ((*mat)->colIndices) free((*mat)->colIndices);
        if ((*mat)->rowPointers) free((*mat)->rowPointers);
        free(*mat);
        *mat = NULL;
    }
}

/*Sort a row of matrix elements from a CSR-format.*/
void sortRow(int *colIndices, double *data, int left, int right){
	int l = left;
	int r = right;
	int middle = colIndices[(l+r) >> 1];
	do{
		while(colIndices[l] < middle)
			l++;
		while(colIndices[r] > middle)
			r--;
		if(l <= r){
			int lColIndex = colIndices[l];
			colIndices[l] = colIndices[r];
			colIndices[r] = lColIndex;
			double lDatum = data[l];
			data[l] = data[r];
			data[r] = lDatum;

			l++;
			r--;
		}
	}while (l < r);
	if(left < r)
		sortRow(colIndices, data, left, r);
	if(right > l)
		sortRow(colIndices, data, l, right);

}

/* 
 * Write all data used by the VHDL testbenches to raw data arrays. The arrays are as follows:
 * - The "colorSizes" array, which first contains the number of rows, columns, non-zero values 
 *   and colors, and the size, in elements, of the NROffsets array, followed by: 
 *   the number of rows (rounded to the nearest 32), the number of rows (not rounded), 
 *   the number of columns (not rounded) and the number of non-zero values 
 *   (rounded to the nearest 32) for every partition.
 *   This array is zero padded up to the nearest 64-byte cacheline.
 * - The "PindicesAddr" array, which contains for every partition, from which elements 
 *   in the global X vector the elements of that X vector partition came. 
 *   For example, if a matrix partition only has non-zero values in columns 1, 3 and 6, then 
 *   that X vector partition will only have three elements, and the color_col_indices array 
 *   will contain 1, 3 and 6 for that partition.
 *   This array is zero padded up to the nearest 64-byte cacheline for evey partition.
 * - The "nnzValues" array contains all non-zero values of each partition of the matrix. 
 *   This array is zero-padded so that each color has a multiple of 32 elements (to have the 
 *   same number of elements per partition as the column indices array).
 * - The "colIndices" array contains all column indices of each partition of the matrix. 
 *   These column indices are the local indices for that partition, so to be used, first a 
 *   local X vector partition needs to be loaded into some local memory (this is done using 
 *   data from the _color_col_indices array), before these column indices can be used as 
 *   addresses to that local memory to read the desired X vector values. 
 *   This array is zero-padded so that data for every partition fills up a number of complete 
 *   cachelines (this means every color has a multiple of 32 elements).   
 * - "NROffsets" is the name of the array that contains the new row offsets for 
 *   all elements of every partition of the matrix. New row offsets are 8-bit values which 
 *   are 0 if that element is not the first element in a row, or which, if that element IS 
 *   the first elemetn of a row) is equal to the amount of empty rows between that new row 
 *   and the row before it plus 1. This array is zero-padded so that data for every partition 
 *   fills up a number of complete cachelines (this means every color has a multiple of 64 elements).   
 */
int matrixToRDF(Matrix *mat, int numColors, int *nodesPerColor, 
        int **colIndicesInColor, int nnzsPerRowLimit, 
        double **nnzValues, short int *colIndices, int *nnzValsSizes, unsigned char *NROffsets, int *colorSizes,
        int numFieldSplits, bool readInterleaved,  int readBatchSize){
    int i, row, c;

    // Data related to column indices per partition

    int doneRows = 0;
    int totalRowNum = 0, numRows = 0;
    int nnzsPerColor = 0, maxNNZsPerColor = 0;

    totalRowNum = 0;
    int totalValSize = 0;

    int *nnzRowsPerColor = (int *) malloc(sizeof(int) * numColors);

    // Find which columns are accessed in each color, as well as how many non-zeroes there are per color.

    // printf("Finding accessed columns in colors.\n");
    for(c = 0; c < numColors; c++){
        numRows = 0;
        nnzRowsPerColor[c] = 0;
        nnzsPerColor = roundUpTo(mat->rowPointers[doneRows + nodesPerColor[c]] - mat->rowPointers[doneRows], 32); // round up to nearest 16 for short ints of column indices
        totalValSize += nnzsPerColor;
        if(nnzsPerColor > maxNNZsPerColor)
            maxNNZsPerColor = nnzsPerColor;
        for(row = doneRows; row < doneRows + nodesPerColor[c]; row++){
            if( mat->rowPointers[row] != mat->rowPointers[row + 1]){
                numRows++;
                nnzRowsPerColor[c] = nnzRowsPerColor[c] + 1;
            }
        }

        doneRows = row;

        totalRowNum += numRows;
    }

    // Data relating to zero rows and row offsets:

    int conseqZeroRows = 0;
    int maxConseqZeroRows = 0;
    int numEmptyRows = 0;
    int *rowOffsets = (int *)malloc(sizeof(int) * totalRowNum);
    int *nnzRowPointers = (int *)malloc(sizeof(int) * (totalRowNum + 1));
    int *colorValPointers = (int *) malloc(sizeof(int) * (numColors + 1));
    int *colorValZeroPointers = (int *) malloc(sizeof(int) * numColors);

    int thatRow = 0;
    int totalOffsets = 0;
    doneRows = 0;
    nnzRowPointers[0] = 0;
    colorValPointers[0] = 0;

    int addedValSize = 0;
    int NRFlagsSize = 0;
    int maxRows = 0;
    int nnzsPerRow = 0, maxNNZsPerRow = 0;
    //int largestRow = 0;

    // printf("Done initializing for sizes: totalRowNum:  %d, numColros: %d.\n", totalRowNum, numColors);

    int fieldsPerLine = 64/sizeof(double);

    // Go through all rows to determine the row offset of each row (amount of zero rows between it and the previous non-zero row)
    for(c = 0; c < numColors; c++){
        totalOffsets += conseqZeroRows;
        conseqZeroRows = 0; 
        // printf("Rows in color %d: %d.\n", c, nodesPerColor[c]);
        for(row = doneRows; row < doneRows + nodesPerColor[c]; row++){
            nnzsPerRow = mat->rowPointers[row + 1] - mat->rowPointers[row];
            if(nnzsPerRow == 0){
                conseqZeroRows++;
                numEmptyRows++;
            }else{
                if(nnzsPerRow > maxNNZsPerRow){
                    //largestRow = row;
                    maxNNZsPerRow = nnzsPerRow;
                }
                nnzRowPointers[thatRow + 1] = mat->rowPointers[row + 1];
                rowOffsets[thatRow] = conseqZeroRows;
                totalOffsets += rowOffsets[thatRow];
                if(maxConseqZeroRows < conseqZeroRows)
                    maxConseqZeroRows = conseqZeroRows;
                conseqZeroRows = 0; 
                thatRow++;
            }
        }
        // Calculate sizes that include zeropadding
        colorValZeroPointers[c] = nnzRowPointers[thatRow] + addedValSize;
        colorValPointers[c + 1] = roundUpTo(colorValZeroPointers[c], 32);
        // printf("In row %d: colorValZeroPointer = %d, colorValPointer = %d.\n", thatRow, colorValZeroPointers[c], colorValPointers[c + 1]);
        addedValSize += colorValPointers[c + 1] - colorValZeroPointers[c];
        NRFlagsSize += roundUpTo(colorValPointers[c + 1] - colorValPointers[c], 64);

        doneRows += nodesPerColor[c];
        if(nodesPerColor[c] > maxRows)
            maxRows = nodesPerColor[c];
    }
    // rowOffsets[thatRow] = conseqZeroRows;
    totalOffsets += rowOffsets[thatRow];

    if(maxNNZsPerRow > nnzsPerRowLimit){
    printf("ERROR: %s: Current reordering exceeds maximum number of non-zero values per row limit: %d/%d.\n",
            __func__, maxNNZsPerRow, nnzsPerRowLimit);
        free(nnzRowsPerColor);
        // free(actualColsPerColor);
        free(rowOffsets);
        free(nnzRowPointers);
        free(colorValPointers);
        free(colorValZeroPointers);
        return -1;
    }

    // printf("Done with preprocessing in %d/%d rows.\n", thatRow, doneRows);

    // create and fill RDF arrays

    // printf("Setting color sizes for %d colors.\n", numColors);

    // colorSizes[0] = (int *)malloc(sizeof(int) * (6 + 4 * roundUpTo(numColors, 4)));
    // printf("ColorSizes pointer: %llx.\n", (unsigned long long int)colorSizes[0]);
    colorSizes[3] = colorValPointers[numColors];
    // printf("Total valSize: %d.\n", colorValPointers[numColors]);
    colorSizes[5] = NRFlagsSize;

    for(c = 0; c < numColors; c++){
        // printf("Color sizes[%d] = %d.\n", c*4+6,  nodesPerColor[c]);
        colorSizes[c * 4 + 8] = nnzRowsPerColor[c];
        colorSizes[c * 4 + 11] = colorValPointers[c + 1] - colorValPointers[c];
    }

    // printf("Done filling sizes array.\n");

    int rowIndex = 0;
    int valIndex = 0;
    int NRIndex = 0;
    int halfwayPoint = 0;
    bool read_sel = false;
    if(readInterleaved && numFieldSplits > 1){
        for(c = 0; c < numColors; c++){
            int thisColorSize = colorValPointers[c+ 1] - colorValPointers[c];
            if((thisColorSize/readBatchSize)%2 == 0){
                halfwayPoint = halfwayPoint + (thisColorSize/readBatchSize/2) * readBatchSize + thisColorSize%readBatchSize;
            }else{
                halfwayPoint = halfwayPoint +(thisColorSize/readBatchSize/2 + 1) * readBatchSize;
            }
        }
        for(i = 0; i < numFieldSplits/2; i++){
            nnzValsSizes[i] = halfwayPoint / (numFieldSplits/2);
            nnzValsSizes[numFieldSplits/2 + i] = (colorValPointers[numColors] - halfwayPoint) /  (numFieldSplits/2);
        }
    }else{
        halfwayPoint = colorValPointers[numColors] / 2;
        for(i = 0; i < numFieldSplits; i++){
            nnzValsSizes[i] = colorValPointers[numColors] / numFieldSplits;
        }
        halfwayPoint = colorValPointers[numColors] / 2;
    }

    colorSizes[7] = halfwayPoint;

    int *vb_array;
    int *v_array ;
    int splitNum;

    vb_array = (int *) malloc(sizeof(int) * numFieldSplits);
    v_array = (int *) malloc(sizeof(int) * numFieldSplits);
    memset(v_array, 0, sizeof(int) * numFieldSplits);

    for(c = 0; c < numColors; c++){
        read_sel = false;
        memset(vb_array, 0, sizeof(int) * numFieldSplits);

        for(int v = colorValPointers[c]; v < colorValPointers[c + 1]; v+=32){
            //int NRs = 0;
            for(int vb = 0; vb < 32; vb++){
                //printf("Debug: color %d, row %d, valIndex %d.\n", c, rowIndex, v+vb);
                if(v + vb < colorValZeroPointers[c]){
                    if(numFieldSplits > 1){
                        if(readInterleaved){
                            if(!read_sel){
                                splitNum = (vb % fieldsPerLine) / (fieldsPerLine / (numFieldSplits/2));
                            }else{
                                splitNum = (numFieldSplits/2) + (vb % fieldsPerLine) / (fieldsPerLine / (numFieldSplits/2));
                            }   
                        }else{
                            splitNum = (vb % fieldsPerLine) / (fieldsPerLine / numFieldSplits);
                        }
                        
                        nnzValues[splitNum][v_array[splitNum]+vb_array[splitNum]] = mat->nnzValues[valIndex];
                        vb_array[splitNum] = vb_array[splitNum] + 1;
                        if (readInterleaved) {
                            if((v-colorValPointers[c]+vb+1)%readBatchSize == 0){
                                read_sel = !read_sel;
                                //printf("INFO: %s: Flipping readSel in color %d after %d nnzValues.\n",__func__, c, v-colorValPointers[c]+vb+1);
                            }
                        }
                    }else{
                        nnzValues[0][v+vb] = mat->nnzValues[valIndex];
                    }
                    colIndices[v+vb] = (unsigned short int)colIndicesInColor[c][mat->colIndices[valIndex]];
                    
                    //printf("colIndex %d (global index %d) in color %d: %d\n", v+vb, mat->colIndices[valIndex], c, colIndices[0][v+vb]);
                    //printf("nnzRowPointers[rowIndex] = %d, valIndex = %d.\n", nnzRowPointers[rowIndex], valIndex);
                    //if this row is done
                    if(nnzRowPointers[rowIndex] == valIndex){
                        // printf("Nonzeroes in row %d: %d.\n", rowIndex, nnzsPerRow);
                        
                        // Skip all empty rows
                        if(rowOffsets[rowIndex] + 1 >= 255){
                            printf("ERROR: row offset size exceeded in row %d with an offset of %d.\n", rowIndex, rowOffsets[rowIndex] + 1 );
                            free(v_array);
                            free(vb_array);
                            free(rowOffsets);
                            free(nnzRowsPerColor);
                            free(nnzRowPointers);
                            free(colorValPointers);
                            free(colorValZeroPointers);
                            return -1;
                        }
                        NROffsets[NRIndex] =  (unsigned char)(rowOffsets[rowIndex] + 1);
                        while(rowIndex < mat->rowSize && nnzRowPointers[rowIndex] == valIndex){
                            //printf("Nonzeroes in row %d: %d.\n", rowIndex, nnzsPerRow);
                            rowIndex++;
                            nnzsPerRow = 0;
                        }
                        nnzsPerRow++;
                    }
                    else{
                        nnzsPerRow++;
                        NROffsets[NRIndex] = (unsigned char) 0;
                    }
                    valIndex++;
                    
                }else{
                    // Zero pad the vals and column indices files
                    //printf("Zero-padding. v+vb = %d/%d, NRIndex = %d/%d.\n", v+vb, colorValPointers[numColors], NRIndex, NRFlagsSize);
                    //nnzsPerRow++;
                    if(numFieldSplits > 1){
                        if(readInterleaved){
                            if(!read_sel){
                                splitNum = (vb % fieldsPerLine) / (fieldsPerLine / (numFieldSplits/2));
                            }else{
                                splitNum = (numFieldSplits/2) + (vb % fieldsPerLine) / (fieldsPerLine / (numFieldSplits/2));
                            }   
                        }else{
                            splitNum = (vb % fieldsPerLine) / (fieldsPerLine / numFieldSplits);
                        }
                        nnzValues[splitNum][v_array[splitNum]+vb_array[splitNum]] = 0.0;
                        vb_array[splitNum] = vb_array[splitNum] + 1;
                        if (readInterleaved) {
                            if((v-colorValPointers[c]+vb+1)%readBatchSize == 0){
                                read_sel = !read_sel;
                                //printf("INFO: %s: Flipping readSel in color %d after %d nnzValues.\n",__func__, c, v-colorValPointers[c]+vb+1);
                            }
                        }
                    }else{
                        nnzValues[0][v+vb] = 0.0;
                    }
                    colIndices[v+vb] = (unsigned short int) colIndicesInColor[c][mat->colIndices[valIndex - 1]];
                    NROffsets[NRIndex] = (unsigned char) 0;
                    //printf("Done zeropadding.\n");
                }
                NRIndex++;
                
            }
            //printIntToRDF(NRFile, NRs);
        }
        if(numFieldSplits > 1){
            for(i = 0; i < numFieldSplits; i++){
                v_array[i] = v_array[i] + vb_array[i];
            }
        }

        // zeropad the NR file
        while(NRIndex % 64 != 0){
            //printf("Zero-padding NR\n");
            NROffsets[NRIndex] = (unsigned char) 0;
            NRIndex++;
        }
    }
    if(numFieldSplits > 1){
        for(i = 0; i < numFieldSplits; i++){
            if(v_array[i] != nnzValsSizes[i]){
                printf("WARNING: %s: did not fill up array numbed %d of nnz_vals: %d instead of %d.\n",__func__, i,  v_array[i], nnzValsSizes[i]);
            }
        }
    }

    free(v_array);
    free(vb_array);

    //printf("The maximum nnzs per row is: %d.\n", maxNNZsPerRow);
    //printf("Done filling matrix arrays.\n");

    free(nnzRowsPerColor);
    free(rowOffsets);
    free(nnzRowPointers);
    free(colorValPointers);
    free(colorValZeroPointers);

    return 0;
}

} // end namespace bda
