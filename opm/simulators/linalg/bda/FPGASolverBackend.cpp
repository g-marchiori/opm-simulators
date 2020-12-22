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
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <math.h>

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/ErrorMacros.hpp>

#include <opm/simulators/linalg/bda/FPGASolverBackend.hpp>
#include <opm/simulators/linalg/bda/BdaResult.hpp>
#include <opm/simulators/linalg/bda/FPGAReorder.hpp>
#include <opm/simulators/linalg/bda/FPGABlockedMatrix.hpp>
#include <opm/simulators/linalg/bda/FPGAUtils.hpp>
#include <opm/simulators/linalg/bda/FPGABILU0.hpp>

// if defined, any HW kernel failure will terminate flow; otherwise, the FPGA
// kernel will be disabled and execution will continue using DUNE
#define FPGA_EXIT_WITH_HW_FAILURE

namespace bda
{

using Opm::OpmLog;

template <unsigned int block_size>
FpgaSolverBackend<block_size>::FpgaSolverBackend(std::string fpga_bitstream, int verbosity_, int maxit_, double tolerance_, ILUReorder opencl_ilu_reorder) : BdaSolver<block_size>(fpga_bitstream, verbosity_, maxit_, tolerance_)
{
    int err;
    std::ostringstream oss;
    double start = second();

    // currently, only block size == 3 is supported by the FPGA backend
    assert(block_size==3);

    // setup performance records per call
    perf_call = (perf_call_metrics_t*)malloc((PERF_RECORDS+1) * sizeof(perf_call_metrics_t)); // PERF_RECORDS+1 because index 0 is not used
    if (perf_call == NULL) {
        printf("WARNING: cannot allocate memory for performance counters, disabling performance data collection for FPGA.\n");
        perf_call_disabled = true;
    }
    if (!perf_call_disabled) { memset(perf_call, 0, (PERF_RECORDS+1) * sizeof(perf_call_metrics_t)); }
    // setup performance records for overall results
    if (!perf_call_disabled) {
        memset(&perf_total, 0, sizeof(perf_total_metrics_t));
        perf_total.s_preconditioner_create_min = 1.0e+16;
        perf_total.s_analysis_min = 1.0e+16;
        perf_total.s_reorder_min = 1.0e+16;
        perf_total.s_mem_setup_min = 1.0e+16;
        perf_total.s_mem_h2d_min = 1.0e+16;
        perf_total.s_kernel_exec_min = 1.0e+16;
        perf_total.n_kernel_exec_cycles_min = (unsigned int)(((unsigned long)1 << 32) - 1);
        perf_total.n_kernel_exec_iters_min = 1.0e+10;
        perf_total.s_mem_d2h_min = 1.0e+16;
        perf_total.s_solve_min = 1.0e+16;
        perf_total.s_postprocess_min = 1.0e+16;
    }
    // setup bitstream name and other parameters
    if (fpga_bitstream.compare("") == 0) {
        OPM_THROW(std::logic_error, "Error fpgaSolver called but bitstream file has not been specified");
    }
    if (!fileExists(fpga_bitstream.c_str())) {
        printf("Error fpgaSolver called but bitstream file specified does not exists (%s)",fpga_bitstream.c_str());
        OPM_THROW(std::logic_error, "");
    }
    printf("INFO: FPGA solver binary file=%s\n", fpga_bitstream.c_str());
    // -----------------------------
    // FPGA: setup the OpenCL platform
    // -----------------------------
    main_xcl_binary = new char[1024];
    main_kernel_name = new char[1024];
    strcpy(main_kernel_name, KERNEL_NAME);
    strcpy(main_xcl_binary, fpga_bitstream.c_str());
    // auto-select the proper FPGA device and create context and other CL objects
    err = setup_opencl(NULL, &device_id, &context, &commands, &program, &kernel, main_kernel_name, main_xcl_binary, &platform_awsf1);
    if (err != 0) {
        oss << "Failed to setup the OpenCL device (" << err << ").\n";
        OPM_THROW(std::logic_error, oss.str());
    }
    delete [] main_xcl_binary;
    delete [] main_kernel_name;
    oss << "Detected FPGA platform type is ";
    if (platform_awsf1) { oss << "AWS-F1\n"; } else { oss << "Xilinx Alveo\n"; }
    OpmLog::info(oss.str());
    // -----------------------------
    // FPGA: setup the debug buffer
    // -----------------------------
    // set kernel debug lines depending on an environment variable
    const char *xem = getenv("XCL_EMULATION_MODE");
    if ((xem != NULL) && (strcmp(xem, "sw_emu") == 0 || strcmp(xem, "hw_emu") == 0)) {
        debug_outbuf_words = DEBUG_OUTBUF_WORDS_MAX_EMU;
        oss << "Detected co-simulation mode, debug_outbuf_words set to " << debug_outbuf_words << ".\n";
        OpmLog::info(oss.str());
    } else {
        debug_outbuf_words = 2; // set to 2 to reduce overhead in reading back and interpreting the debug lines
    }
    // host debug buffer setup
    err = fpga_setup_host_debugbuf(debug_outbuf_words, &debugBuffer, &debugbufferSize);
    if (err != 0) {
        oss << "Failed to call fpga_setup_host_debug_buffer (" << err << ").\n";
        OPM_THROW(std::logic_error, oss.str());
    }
    // device debug buffer setup
    err = fpga_setup_device_debugbuf(context, debugBuffer, &cldebug, debugbufferSize);
    if (err != 0) {
        oss << "Failed to call fpga_setup_device_debug_buffer (" << err << ").\n";
        OPM_THROW(std::logic_error, oss.str());
    }
    // copy debug buffer to device
    err = fpga_copy_to_device_debugbuf(commands, cldebug, debugBuffer, debugbufferSize, debug_outbuf_words);
    if (err != 0) {
        oss << "Failed to call fpga_copy_to_device_debugbuf (" << err << ").\n";
        OPM_THROW(std::logic_error, oss.str());
    }
    // ------------------------------------------------
    // FPGA: query the kernel for limits/configuration
    // ------------------------------------------------
    err = fpga_kernel_query(context, commands, kernel, cldebug,
        debugBuffer, debug_outbuf_words,
        rst_assert_cycles, rst_settle_cycles,
        &hw_x_vector_elem, &hw_max_row_size,
        &hw_max_column_size, &hw_max_colors_size,
        &hw_max_nnzs_per_row, &hw_max_matrix_size,
        &hw_use_uram, &hw_write_ilu0_results,
        &hw_dma_data_width, &hw_mult_num,
        &hw_x_vector_latency, &hw_add_latency, &hw_mult_latency,
        &hw_num_read_ports, &hw_num_write_ports,
        &hw_reset_cycles, &hw_reset_settle);
    if (err != 0) {
        oss << "Failed to call fpga_kernel_query (" << err << ").\n";
        OPM_THROW(std::logic_error, oss.str());
    }
    BDA_DEBUG(1,
        printf("INFO: kernel limits/configuration:\n");
        printf("INFO:  x_vector_elem=%u, max_row_size=%u, max_column_size=%u\n"
            "INFO:  max_colors_size=%u, max_nnzs_per_row=%u, max_matrix_size=%u\n"
            "INFO:  use_uram=%d, write_ilu0_results=%d\n"
            "INFO:  dma_data_width=%u, mult_num=%u\n"
            "INFO:  x_vector_latency=%u\n"
            "INFO:  add_latency=%u, mult_latency=%u\n"
            "INFO:  num_read_ports=%u, num_write_ports=%u\n"
            "INFO:  reset_cycles=%u, reset_settle=%u\n",
            hw_x_vector_elem,hw_max_row_size,hw_max_column_size,
            hw_max_colors_size,hw_max_nnzs_per_row,hw_max_matrix_size,
            (int)hw_use_uram,(int)hw_write_ilu0_results,
            hw_dma_data_width,hw_mult_num,
            hw_x_vector_latency,
            hw_add_latency, hw_mult_latency,
            hw_num_read_ports, hw_num_write_ports,
            hw_reset_cycles, hw_reset_settle);
    )
    // check that LU results are generated by the kernel
    if (use_LU_res && !hw_write_ilu0_results) {
        printf("WARNING: kernel reports that LU results are not written to memory, but use_LU_res is set.\n");
        printf("WARNING: disabling LU results usage.\n");
        use_LU_res = false;
    }

    // setup preconditioner
    double start_prec = second();
    prec = new Preconditioner(opencl_ilu_reorder, hw_max_row_size, hw_max_column_size, hw_max_nnzs_per_row, hw_max_colors_size);
    perf_total.s_preconditioner_setup = second() - start_prec;

    if (opencl_ilu_reorder == ILUReorder::LEVEL_SCHEDULING) { level_scheduling = true; }

    perf_total.s_initialization = second() - start;
} // end fpgaSolverBackend


template <unsigned int block_size>
FpgaSolverBackend<block_size>::~FpgaSolverBackend()
{
    generate_statistics();
    if (perf_call != NULL) { free(perf_call); }
    if (mat != NULL) { free(mat); }
    delete[] x;
    delete[] rx;
    delete[] rb;
    delete prec;
    delete[] processedPointers;
    delete[] processedSizes;
    if (nnzValArrays != NULL) { free(nnzValArrays); }
    if (nnzValArrays_sizes != NULL) { free(nnzValArrays_sizes); }
    if (L_nnzValArrays != NULL) { free(L_nnzValArrays); }
    if (L_nnzValArrays_sizes != NULL) { free(L_nnzValArrays_sizes); }
    if (U_nnzValArrays != NULL) { free(U_nnzValArrays); }
    if (U_nnzValArrays_sizes != NULL) { free(U_nnzValArrays_sizes); }
    // FPGA: buffers
    free(debugBuffer);
    for (int b = 0; b < RW_BUF; b++) {
        free(dataBuffer[b]);
    }
    free(databufferSize);
    // FPGA: OpenCL objects
    if (cldebug != nullptr) { clReleaseMemObject(cldebug); }
    for (int b = 0; b < RW_BUF; b++) { if (cldata[b] != nullptr) { clReleaseMemObject(cldata[b]); } }
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseDevice(device_id);
} // end ~fpgaSolverBackend()


// copy result to host memory
// caller must be sure that x is a valid array
template <unsigned int block_size>
void FpgaSolverBackend<block_size>::get_result(double *x_)
{
    std::ostringstream oss;
    double start = 0;

    // (partial) dump of results buffers BEFORE reordering
    BDA_DEBUG(2,
        int cl_max = resultsBufferSize[0] / CACHELINE_BYTES;
        // limit the number of cachelines displayed
        if (cl_max > 8) { cl_max = 8; }
        printf("INFO: results buffer rx dump (first %d cachelines):\n", cl_max);
        for (int c = 0; c < cl_max; c++) {
            printf(" cl %5d: 0x", c);
            for (int i = CACHELINE_DBL_WORDS - 1; i >= 0; i--) {
                union double2int conv;
                conv.double_val = rx[c * CACHELINE_DBL_WORDS + i];
                printf("%016lx", conv.int_val);
                printf(" ");
            }
            printf("\n");
        }
    )

    if (!perf_call_disabled) { start = second(); }
    // apply to results the reordering (stored in toOrder)
    reorder_vector_by_blocked_pattern(mat->rowSize, rx, toOrder, x_);
    // TODO: check if it is more efficient to avoid copying resultsBuffer[0] to rx in solve_system (private)
    //reorder_vector_by_blocked_pattern(mat->rowSize, resultsBuffer[0], toOrder, x_);
    if (!perf_call_disabled) { perf_call[fpga_calls].s_postprocess = second() - start; }
} // end get_result()


template <unsigned int block_size>
SolverStatus FpgaSolverBackend<block_size>::solve_system(int N_, int nnz_, int dim, double *vals, int *rows, int *cols, double *b, WellContributions& wellContribs, BdaResult &res)
{
    if (initialized == false) {
        initialize(N_, nnz_,  dim, vals, rows, cols);
        if (!analyse_matrix()) {
            return SolverStatus::BDA_SOLVER_ANALYSIS_FAILED;
        }
    }
    update_system(vals, b);
    if (!create_preconditioner()) {
        return SolverStatus::BDA_SOLVER_CREATE_PRECONDITIONER_FAILED;
    }
    solve_system(res);

    BDA_DEBUG(1,
        printf("fpgaSolverBackend::%s (public) - converged=%d, iterations=%d, reduction=%.3f, conv_rate=%.3f, elapsed=%.3f\n",
            __func__,res.converged,res.iterations,res.reduction,res.conv_rate,res.elapsed);
    )
    return SolverStatus::BDA_SOLVER_SUCCESS;
}


template <unsigned int block_size>
void FpgaSolverBackend<block_size>::initialize(int N_, int nnz_, int dim, double *vals, int *rows, int *cols)
{
    std::ostringstream oss;

    double start = second();
    this->N = N_;
    this->nnz = nnz_;
    this->nnzb = nnz_ / block_size / block_size;
    Nb = (N + dim - 1) / dim;

    // allocate host memory for matrices and vectors
    // actual data for mat points to std::vector.data() in ISTLSolverEbos, so no alloc/free here
    // CSCmat does need its own allocs
    mat = (BlockedMatrixFpga *)malloc(sizeof(BlockedMatrixFpga));
    mat->rowSize = N_ / BLOCK_SIZE;
    mat->valSize = nnz_ / BLOCK_SIZE / BLOCK_SIZE;
    mat->nnzValues = (Block*)vals;
    mat->colIndices = cols;
    mat->rowPointers = rows;

    printf("Initializing FPGA data, matrix size: %d blocks, nnz: %d blocks, block size: %d, total nnz: %d\n", this->N, this->nnzb, dim, this->nnz);
    printf("Maxit: %d, tolerance: %.1e\n", maxit, tolerance);

    x  = new double[roundUpTo(N_, CACHELINE_BYTES / sizeof(double))];
    rx = new double[roundUpTo(N_, CACHELINE_BYTES / sizeof(double))];
    rb = new double[roundUpTo(N_, CACHELINE_BYTES / sizeof(double))];
    // allocate the vectors holding the nnz arrays sizes
    nnzValArrays_sizes = (int*)malloc(sizeof(int));
    memset(nnzValArrays_sizes,0,sizeof(int));
    L_nnzValArrays_sizes = (int*)malloc(sizeof(int));
    memset(L_nnzValArrays_sizes,0,sizeof(int));
    U_nnzValArrays_sizes = (int*)malloc(sizeof(int));
    memset(U_nnzValArrays_sizes,0,sizeof(int));

    perf_total.s_initialization += second() - start;
    initialized = true;
} // end initialize()


template <unsigned int block_size>
bool FpgaSolverBackend<block_size>::analyse_matrix()
{
    std::ostringstream oss;
    int err;

    double start = second();
    bool success = prec->init(mat);

    if (!success) {
       printf("ERROR: preconditioner for FPGA solver failed to initialize.\n");
       return success;
    }

    toOrder = prec->getToOrder();
    fromOrder = prec->getFromOrder();
    rMat = prec->getRMat();
    processedPointers = prec->getResultPointers();
    processedSizes = prec->getResultSizes();
    processedPointers[19] = rb;
    processedPointers[20] = rx;
    // fill the nnzValArrays_sizes array
    memcpy(nnzValArrays_sizes, (int *)processedPointers[5], sizeof(int));
    memcpy(L_nnzValArrays_sizes, (int *)processedPointers[11], sizeof(int));
    memcpy(U_nnzValArrays_sizes, (int *)processedPointers[17], sizeof(int));
    // -------------------------------------
    // FPGA: setup host/device data buffers
    // -------------------------------------
    BDA_DEBUG(1,
        printf("INFO: processedSizes array:\n");
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 6; i++) {
                printf("[%2d]=%7d ", i + j, processedSizes[i + (6 * j)]);
            }
            printf("\n");
        }
        printf("INFO: nnzValArrays_sizes[0]=%d, L_nnzValArrays_sizes[0]=%d, U_nnzValArrays_sizes[0]=%d\n",
            nnzValArrays_sizes[0],L_nnzValArrays_sizes[0],U_nnzValArrays_sizes[0]);
    )
    // allocate memory and setup data layout
    err = fpga_setup_host_datamem(level_scheduling, fpga_config_bits,
        processedSizes,
        &setupArray,
        &nnzValArrays,   nnzValArrays_sizes,   &columnIndexArray,   &newRowOffsetArray,   &PIndexArray,   &colorSizesArray,
        &L_nnzValArrays, L_nnzValArrays_sizes, &L_columnIndexArray, &L_newRowOffsetArray, &L_PIndexArray, &L_colorSizesArray,
        &U_nnzValArrays, U_nnzValArrays_sizes, &U_columnIndexArray, &U_newRowOffsetArray, &U_PIndexArray, &U_colorSizesArray,
        &BLKDArray, &X1Array, &R1Array,
        &X2Array, &R2Array,
        &LresArray, &UresArray,
        &databufferSize, dataBuffer,
        result_offsets, 1 /*num_splitfields*/,
        true /*reset_data_buffers*/,  /* WARNING: leave reset_data_buffers always ENABLED to avoid data corruption! */
        debugbufferSize);
    if (err) {
        printf("ERROR: failed to call fpga_setup_host_datamem (%d)\n", err);
        OPM_THROW(std::logic_error, "");
    }
    // results buffers setup
    if (use_LU_res) { resultsBufferNum = 4; } else { resultsBufferNum = 2; }
    if (resultsBufferNum > RES_BUF_MAX) {
        printf("ERROR: number of results buffer (%d) is out of range (max %d).\n", resultsBufferNum, RES_BUF_MAX);
        OPM_THROW(std::logic_error, "");
    }
    resultsNum = processedSizes[0]; // rowSize, invariant between system solves
    for (int i = 0; i < resultsBufferNum; i++) {
        resultsBufferSize[i] = roundUpTo(resultsNum, CACHELINE_BYTES / sizeof(double)) * sizeof(double);
    }
    // device data memory setup
    err = fpga_setup_device_datamem(context, databufferSize, dataBuffer, cldata);
    if (err != 0) {
        printf("ERROR: failed to call fpga_setup_device_datamem (%d)\n", err);
        OPM_THROW(std::logic_error, "");
    }
    // ------------------------------------
    // FPGA: setup the kernel's parameters
    // ------------------------------------
    err = fpga_set_kernel_parameters(kernel, abort_cycles, debug_outbuf_words - 1, maxit,
        debug_sample_rate, tolerance, cldata, cldebug);
    if (err != 0) {
        printf("ERROR: failed to call fpga_set_kernel_parameters (%d)\n", err);
        OPM_THROW(std::logic_error, "");
    }

    perf_total.s_analysis = second() - start;
    analysis_done = true;

    return success;
} // end analyse_matrix()


template <unsigned int block_size>
bool FpgaSolverBackend<block_size>::create_preconditioner()
{
    std::ostringstream oss;
    double start = 0;
    bool result;

    if (!perf_call_disabled) { start = second(); }
    memset(rx, 0, sizeof(double) * N);
    result = prec->create_preconditioner(mat);
    if (!result) { printf("WARNING: create_preconditioner failed.\n"); }
    if (!perf_call_disabled) { perf_call[fpga_calls].s_preconditioner_create = second() - start; }
    return result;
} // end create_preconditioner()


template <unsigned int block_size>
void FpgaSolverBackend<block_size>::solve_system(BdaResult &res)
{
    std::ostringstream oss;
    int err;
    double start = 0, start_total = 0;

    // ------------------------------------
    // FPGA: return immediately if FPGA is disabled
    // ------------------------------------
    if (fpga_disabled) {
        res.converged = false;
        printf("WARNING: FPGA is disabled, fallback to SW execution.\n");
        return;
    }

    fpga_calls++;
    if (!perf_call_disabled && fpga_calls > PERF_RECORDS) {
        perf_call_disabled = true;
        printf("WARNING: too many samples, disabling performance counters for FPGA.\n");
    }

    if (!perf_call_disabled) {
        start = second();
        start_total = start;
    }

    // check if any buffer is larger than the size set in preconditioner->init
    BDA_DEBUG(1,
        printf("INFO: nnzValArrays_sizes[0]=%d, L_nnzValArrays_sizes[0]=%d, U_nnzValArrays_sizes[0]=%d\n",
            nnzValArrays_sizes[0],L_nnzValArrays_sizes[0],U_nnzValArrays_sizes[0]); fflush(NULL);
        printf("INFO: processedPointers[5][0]=%d, processedPointers[11][0]=%d, processedPointers[17][0]=%d\n",
            ((int *)processedPointers[5])[0],((int *)processedPointers[11])[0],((int *)processedPointers[17])[0]); fflush(NULL);
    )
    // TODO: add check for all other buffer sizes that may overflow?
    err = 0;
    if ( ((int *)processedPointers[5])[0]  > nnzValArrays_sizes[0] ||
         ((int *)processedPointers[11])[0] > L_nnzValArrays_sizes[0] ||
         ((int *)processedPointers[17])[0] > U_nnzValArrays_sizes[0] ) {
        err = 1;
    }
    if (err != 0) {
        OPM_THROW(std::logic_error, "A buffer size is larger than the initial allocation in solve_system (check preconditioner init).");
    }

    // ------------------------------------
    // FPGA: copy input data to host data buffers
    // ------------------------------------
    if (!perf_call_disabled) { start = second(); }
    err = fpga_copy_host_datamem(
        processedPointers, processedSizes, setupArray,
        nnzValArrays,   nnzValArrays_sizes,   columnIndexArray,   newRowOffsetArray,   PIndexArray,   colorSizesArray,
        L_nnzValArrays, L_nnzValArrays_sizes, L_columnIndexArray, L_newRowOffsetArray, L_PIndexArray, L_colorSizesArray,
        U_nnzValArrays, U_nnzValArrays_sizes, U_columnIndexArray, U_newRowOffsetArray, U_PIndexArray, U_colorSizesArray,
        BLKDArray, X1Array, R1Array, X2Array, R2Array,
        use_LU_res, LresArray, UresArray,
        databufferSize, dataBuffer,
        1 /* nnzValArrays_num */,
        reset_data_buffers, fill_results_buffers,
        dump_data_buffers, fpga_calls);
    if (!perf_call_disabled) { perf_call[fpga_calls].s_mem_setup = second() - start; }
    if (err != 0) {
        printf("ERROR: failed to call fpga_copy_to_device_debugbuf (%d)\n", err);
        OPM_THROW(std::logic_error, "");
    }
    // ------------------------------------
    // FPGA: copy buffers to device
    // ------------------------------------
    // copy debug buffer to device
    if (!perf_call_disabled) { start = second(); }
    err = fpga_copy_to_device_debugbuf(commands,
        cldebug, debugBuffer, debugbufferSize,
        debug_outbuf_words);
    if (err != 0) {
        printf("ERROR: failed to call fpga_copy_to_device_debugbuf (%d)\n", err);
        OPM_THROW(std::logic_error, "");
    }
    // copy data buffers to device
    err = fpga_copy_to_device_datamem(commands, RW_BUF, cldata);
    if (err != 0) {
        printf("ERROR: failed to call fpga_copy_to_device_datamem (%d)\n", err);
        OPM_THROW(std::logic_error, "");
    }
    if (!perf_call_disabled) { perf_call[fpga_calls].s_mem_h2d = second() - start; }
    // ------------------------------------
    // FPGA: execute the kernel
    // ------------------------------------
    double time_elapsed_ms;
    if (!perf_call_disabled) { start = second(); }
    err = fpga_kernel_run(commands, kernel, &time_elapsed_ms);
    if (!perf_call_disabled) { perf_call[fpga_calls].s_kernel_exec = second() - start; }
    if (err != 0) {
        printf("ERROR: failed to call fpga_kernel_run (%d)\n", err);
        OPM_THROW(std::logic_error, "");
    }
    // ----------------------------------------
    // FPGA: read back debug buffer from device
    // ----------------------------------------
    if (!perf_call_disabled) { start = second(); }
    err = fpga_copy_from_device_debugbuf((bool)(BDA_DEBUG_LEVEL <= 1),
        commands,
        debug_outbuf_words, debugbufferSize,
        cldebug, debugBuffer,
        abort_cycles,
        &kernel_cycles, &kernel_iter_run,
        norms, &last_norm_idx,
        &kernel_aborted, &kernel_signature, &kernel_overflow, &kernel_noresults,
        &kernel_wrafterend, &kernel_dbgfifofull);
    if (err != 0) {
        printf("ERROR: failed to call fpga_copy_from_device_debugbuf (%d)\n", err);
        OPM_THROW(std::logic_error, "");
    }
    if (kernel_wrafterend) {
        printf("WARNING: detected recoverable FPGA error: kernel write after end\n");
    }
    if (kernel_dbgfifofull) {
       printf("WARNING: detected recoverable FPGA error: debug FIFO full\n");
    }
    if (kernel_aborted || kernel_signature || kernel_overflow) {
        printf("ERROR: detected unrecoverable FPGA error (ABRT=%d,SIG=%d,OVF=%d).\n",
            kernel_aborted, kernel_signature, kernel_overflow);
#if defined(FPGA_EXIT_WITH_HW_FAILURE)
        OPM_THROW(std::logic_error, "");
#else
        printf("WARNING: disabling FPGA kernel: execution will continue with SW kernel.\n");
        fpga_disabled = true;
#endif
    }
    if (!perf_call_disabled) { perf_call[fpga_calls].n_kernel_exec_cycles = kernel_cycles; }
    // copy (back) results only if FPGA is not disabled
    if (!fpga_disabled) {
        if (kernel_noresults) {
            BDA_DEBUG(1, printf("WARNING: kernel did not return results because the required precision is already reached.\n");)
            // copy input buffers to output buffers
            memcpy(rx, x, resultsNum * sizeof(double));
        } else {
            // ------------------------------------
            // FPGA: read back results from device
            // ------------------------------------
            // DEBUG: enable to dump data buffers to file
            char *data_dir = nullptr;
            char *basename = nullptr;
            BDA_DEBUG(99,
                dump_results = true;
                data_dir = new char[1024];
                strcpy(data_dir,"rdfout");
                basename = new char[1024];
                strcpy(basename,"spe1case1");
                sequence = fpga_calls;
            )
            err = fpga_map_results(even(kernel_iter_run),
                use_residuals, use_LU_res, commands,
                resultsNum, resultsBufferNum, resultsBufferSize,
                debugbufferSize,
                cldata, resultsBuffer,
                result_offsets,
                dump_results, data_dir, basename, sequence);
            if (err != 0) {
                printf("ERROR: failed to call fpga_map_results (%d)\n", err);
                OPM_THROW(std::logic_error, "");
            }
            if (dump_results) {
                delete data_dir;
                delete basename;
            }
            // TODO: copy results buffers to reordering output buffers
            memcpy(rx, resultsBuffer[0], resultsNum * sizeof(double));
            err = fpga_unmap_results(even(kernel_iter_run),
                use_residuals, use_LU_res,
                commands, cldata, resultsBuffer);
            if (err != 0) {
                 printf("ERROR: failed to call fpga_unmap_results (%d)\n", err);
                 OPM_THROW(std::logic_error, "");
            }
        }
    } // fpga_disabled
    // set results and update statistics (if enabled)
    if (!perf_call_disabled) { perf_call[fpga_calls].s_mem_d2h = second() - start; }
    float iter = ((float)kernel_iter_run / 2.0) + 0.5; // convert from half iteration int to actual iterationns
    res.iterations = (int)iter;
    res.reduction = norms[0] / norms[last_norm_idx]; // norms[0] is the initial norm
    res.conv_rate = pow(res.reduction, 1.0 / iter);
    res.elapsed = second() - start_total;
    if (!perf_call_disabled) {
        perf_call[fpga_calls].s_solve = res.elapsed;
        perf_call[fpga_calls].n_kernel_exec_iters = iter;
    }
    // convergence depends on number of iterations reached and hw execution errors
    res.converged = true;
    if (fpga_disabled || kernel_aborted || kernel_signature || kernel_overflow || iter >= (float)maxit) {
        res.converged = false;
        BDA_DEBUG(1,
            printf("WARNING: kernel did not converge, reason: ");
            printf("fpga_disabled=%d, kernel_aborted=%d, kernel_signature=%d, kernel_overflow=%d, iter>=%d=%d\n",
                fpga_disabled, kernel_aborted, kernel_signature, kernel_overflow, maxit, (iter >= (float)maxit));
        )
    }
    if (!perf_call_disabled) {
        perf_call[fpga_calls].converged = res.converged;
        perf_call[fpga_calls].converged_flags = ((unsigned int)fpga_disabled) +
            ((unsigned int)kernel_aborted << 1) + ((unsigned int)kernel_signature << 2) +
            ((unsigned int)kernel_overflow << 3) + ((unsigned int)(iter >= (float)maxit) << 4);
    }
} // end solve_system()


template <unsigned int block_size>
void FpgaSolverBackend<block_size>::update_system(double *vals, double *b)
{
    double start = 0;

    mat->nnzValues = (Block*)vals;

    // reorder inputs using previously found ordering (stored in fromOrder)
    if (!perf_call_disabled) { start = second(); }
    reorder_vector_by_blocked_pattern(mat->rowSize, b, fromOrder, rb);
    if (!perf_call_disabled) { perf_call[fpga_calls].s_reorder = second() - start; }
} // end update_system()


template <unsigned int block_size>
void FpgaSolverBackend<block_size>::generate_statistics()
{
    unsigned int conv_iter = 0, conv_ovf = 0;
    FILE *fout = NULL;

    if (perf_call == NULL || fpga_calls == 0) {
        printf("WARNING: FPGA statistics were not collected.\n");
        return;
    }
    unsigned int data_points = fpga_calls > PERF_RECORDS ? PERF_RECORDS : fpga_calls;
    if (data_points != fpga_calls) {
        printf("WARNING: FPGA statistics are incomplete because there were too many data points.\n");
    }
    printf("--- FPGA statistics ---\n");
    printf("total calls: %u\n", fpga_calls);
    printf("time initialization.........: %8.6f s\n", perf_total.s_initialization);
    printf("time preconditioner setup...: %8.6f s\n", perf_total.s_preconditioner_setup);
    fout = fopen("fpga_statistics_details.csv", "w");
    if (fout != NULL) {
        fprintf(fout, "call,preconditioner_create,analysis,reorder,mem_setup,mem_h2d,kernel_exec,kernel_cycles,kernel_iters,mem_d2h,solve,postprocess,converged\n");
    }
    for (int i = 1; i <= (int)data_points; i++) {
        perf_total.s_preconditioner_create += perf_call[i].s_preconditioner_create;
        if (perf_call[i].s_preconditioner_create > perf_total.s_preconditioner_create_max) { perf_total.s_preconditioner_create_max = perf_call[i].s_preconditioner_create; }
        if (perf_call[i].s_preconditioner_create < perf_total.s_preconditioner_create_min) { perf_total.s_preconditioner_create_min = perf_call[i].s_preconditioner_create; }
        perf_total.s_analysis += perf_call[i].s_analysis;
        if (perf_call[i].s_analysis > perf_total.s_analysis_max) { perf_total.s_analysis_max = perf_call[i].s_analysis; }
        if (perf_call[i].s_analysis < perf_total.s_analysis_min) { perf_total.s_analysis_min = perf_call[i].s_analysis; }
        perf_total.s_reorder += perf_call[i].s_reorder;
        if (perf_call[i].s_reorder > perf_total.s_reorder_max) { perf_total.s_reorder_max = perf_call[i].s_reorder; }
        if (perf_call[i].s_reorder < perf_total.s_reorder_min) { perf_total.s_reorder_min = perf_call[i].s_reorder; }
        perf_total.s_mem_setup += perf_call[i].s_mem_setup;
        if (perf_call[i].s_mem_setup > perf_total.s_mem_setup_max) { perf_total.s_mem_setup_max = perf_call[i].s_mem_setup; }
        if (perf_call[i].s_mem_setup < perf_total.s_mem_setup_min) { perf_total.s_mem_setup_min = perf_call[i].s_mem_setup; }
        perf_total.s_mem_h2d += perf_call[i].s_mem_h2d;
        if (perf_call[i].s_mem_h2d > perf_total.s_mem_h2d_max) { perf_total.s_mem_h2d_max = perf_call[i].s_mem_h2d; }
        if (perf_call[i].s_mem_h2d < perf_total.s_mem_h2d_min) { perf_total.s_mem_h2d_min = perf_call[i].s_mem_h2d; }
        perf_total.s_kernel_exec += perf_call[i].s_kernel_exec;
        if (perf_call[i].s_kernel_exec > perf_total.s_kernel_exec_max) { perf_total.s_kernel_exec_max = perf_call[i].s_kernel_exec; }
        if (perf_call[i].s_kernel_exec < perf_total.s_kernel_exec_min) { perf_total.s_kernel_exec_min = perf_call[i].s_kernel_exec; }
        perf_total.n_kernel_exec_cycles += (unsigned long)perf_call[i].n_kernel_exec_cycles;
        if (perf_call[i].n_kernel_exec_cycles > perf_total.n_kernel_exec_cycles_max) { perf_total.n_kernel_exec_cycles_max = perf_call[i].n_kernel_exec_cycles; }
        if (perf_call[i].n_kernel_exec_cycles < perf_total.n_kernel_exec_cycles_min) { perf_total.n_kernel_exec_cycles_min = perf_call[i].n_kernel_exec_cycles; }
        perf_total.n_kernel_exec_iters += perf_call[i].n_kernel_exec_iters;
        if (perf_call[i].n_kernel_exec_iters > perf_total.n_kernel_exec_iters_max) { perf_total.n_kernel_exec_iters_max = perf_call[i].n_kernel_exec_iters; }
        if (perf_call[i].n_kernel_exec_iters < perf_total.n_kernel_exec_iters_min) { perf_total.n_kernel_exec_iters_min = perf_call[i].n_kernel_exec_iters; }
        perf_total.s_mem_d2h += perf_call[i].s_mem_d2h;
        if (perf_call[i].s_mem_d2h > perf_total.s_mem_d2h_max) { perf_total.s_mem_d2h_max = perf_call[i].s_mem_d2h; }
        if (perf_call[i].s_mem_d2h < perf_total.s_mem_d2h_min) { perf_total.s_mem_d2h_min = perf_call[i].s_mem_d2h; }
        perf_total.s_solve += perf_call[i].s_solve;
        if (perf_call[i].s_solve > perf_total.s_solve_max) { perf_total.s_solve_max = perf_call[i].s_solve; }
        if (perf_call[i].s_solve < perf_total.s_solve_min) { perf_total.s_solve_min = perf_call[i].s_solve; }
        perf_total.s_postprocess += perf_call[i].s_postprocess;
        if (perf_call[i].s_postprocess > perf_total.s_postprocess_max) { perf_total.s_postprocess_max = perf_call[i].s_postprocess; }
        if (perf_call[i].s_postprocess < perf_total.s_postprocess_min) { perf_total.s_postprocess_min = perf_call[i].s_postprocess; }
        perf_total.n_converged += (unsigned int)perf_call[i].converged;
        if (perf_call[i].converged_flags & 1 << 4) { conv_iter += 1; }
        if (perf_call[i].converged_flags & 1 << 3) { conv_ovf += 1; }
        if (fout != NULL) {
            fprintf(fout, "%d,%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,%8.6f,%u,%.1f,%8.6f,%8.6f,%8.6f,%u\n",
                i, perf_call[i].s_preconditioner_create, perf_call[i].s_analysis, perf_call[i].s_reorder,
                perf_call[i].s_mem_setup, perf_call[i].s_mem_h2d, perf_call[i].s_kernel_exec, perf_call[i].n_kernel_exec_cycles,
                perf_call[i].n_kernel_exec_iters, perf_call[i].s_mem_d2h, perf_call[i].s_solve, perf_call[i].s_postprocess,
                (unsigned int)perf_call[i].converged);
        }
    }
    if (fout != NULL) { fclose(fout); }
    perf_total.s_preconditioner_create_avg = perf_total.s_preconditioner_create / (double)data_points;
    perf_total.s_analysis_avg = perf_total.s_analysis / (double)data_points;
    perf_total.s_reorder_avg = perf_total.s_reorder / (double)data_points;
    perf_total.s_mem_setup_avg = perf_total.s_mem_setup / (double)data_points;
    perf_total.s_mem_h2d_avg = perf_total.s_mem_h2d / (double)data_points;
    perf_total.s_kernel_exec_avg = perf_total.s_kernel_exec / (double)data_points;
    perf_total.n_kernel_exec_cycles_avg = perf_total.n_kernel_exec_cycles / data_points;
    perf_total.n_kernel_exec_iters_avg = perf_total.n_kernel_exec_iters / (float)data_points;
    perf_total.s_mem_d2h_avg = perf_total.s_mem_d2h / (double)data_points;
    perf_total.s_solve_avg = perf_total.s_solve / (double)data_points;
    perf_total.s_postprocess_avg = perf_total.s_postprocess / (double)data_points;
    printf("time preconditioner creation: total %8.6f s, avg %8.6f s, min %8.6f s, max %8.6f s\n",
        perf_total.s_preconditioner_create, perf_total.s_preconditioner_create_avg, perf_total.s_preconditioner_create_min, perf_total.s_preconditioner_create_max);
    printf("time analysis...............: total %8.6f s, avg %8.6f s, min %8.6f s, max %8.6f s\n",
        perf_total.s_analysis, perf_total.s_analysis_avg, perf_total.s_analysis_min, perf_total.s_analysis_max);
    printf("time reorder................: total %8.6f s, avg %8.6f s, min %8.6f s, max %8.6f s\n",
        perf_total.s_reorder, perf_total.s_reorder_avg, perf_total.s_reorder_min, perf_total.s_reorder_max);
    printf("time memory setup...........: total %8.6f s, avg %8.6f s, min %8.6f s, max %8.6f s\n",
        perf_total.s_mem_setup, perf_total.s_mem_setup_avg, perf_total.s_mem_setup_min, perf_total.s_mem_setup_max);
    printf("time memory host2dev........: total %8.6f s, avg %8.6f s, min %8.6f s, max %8.6f s\n",
        perf_total.s_mem_h2d, perf_total.s_mem_h2d_avg, perf_total.s_mem_h2d_min, perf_total.s_mem_h2d_max);
    printf("time kernel execution.......: total %8.6f s, avg %8.6f s, min %8.6f s, max %8.6f s\n",
        perf_total.s_kernel_exec, perf_total.s_kernel_exec_avg, perf_total.s_kernel_exec_min, perf_total.s_kernel_exec_max);
    printf("cycles kernel execution.....: total %lu, avg %lu, min %lu, max %lu\n",
        perf_total.n_kernel_exec_cycles, perf_total.n_kernel_exec_cycles_avg, perf_total.n_kernel_exec_cycles_min, perf_total.n_kernel_exec_cycles_max);
    printf("iterations kernel execution.: total %.1f, avg %.1f, min %.1f, max %.1f\n",
        perf_total.n_kernel_exec_iters, perf_total.n_kernel_exec_iters_avg, perf_total.n_kernel_exec_iters_min, perf_total.n_kernel_exec_iters_max);
    printf("time memory dev2host........: total %8.6f s, avg %8.6f s, min %8.6f s, max %8.6f s\n",
        perf_total.s_mem_d2h, perf_total.s_mem_d2h_avg, perf_total.s_mem_d2h_min, perf_total.s_mem_d2h_max);
    printf("time solve..................: total %8.6f s, avg %8.6f s, min %8.6f s, max %8.6f s\n",
        perf_total.s_solve, perf_total.s_solve_avg, perf_total.s_solve_min, perf_total.s_solve_max);
    printf("time postprocess............: total %8.6f s, avg %8.6f s, min %8.6f s, max %8.6f s\n",
        perf_total.s_postprocess, perf_total.s_postprocess_avg, perf_total.s_postprocess_min, perf_total.s_postprocess_max);
    printf("converged...................: %u/%u, with iter>%d=%u, overflow=%u\n",
        perf_total.n_converged, data_points, maxit, conv_iter, conv_ovf);
    printf("-----------------------\n");
} //end generate_statistics()


#define INSTANTIATE_BDA_FUNCTIONS(n)                                                          \
template FpgaSolverBackend<n>::FpgaSolverBackend(std::string, int, int, double, ILUReorder);  \

INSTANTIATE_BDA_FUNCTIONS(1);
INSTANTIATE_BDA_FUNCTIONS(2);
INSTANTIATE_BDA_FUNCTIONS(3);
INSTANTIATE_BDA_FUNCTIONS(4);

#undef INSTANTIATE_BDA_FUNCTIONS

} //namespace bda

