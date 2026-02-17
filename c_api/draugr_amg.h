/*
 * draugr_amg.h — C API for the Draugr AMG library
 *
 * Compiled from Draugr.jl via PackageCompiler.jl.
 * Link against libdraugr.so (Linux), libdraugr.dylib (macOS),
 * or draugr.dll (Windows).
 *
 * Lifecycle:
 *   1. Call init_julia(argc, argv) once at startup.
 *   2. Create configs and hierarchies, solve, cycle, resetup as needed.
 *   3. Free handles when done.
 *   4. Call shutdown_julia(0) once at shutdown.
 */

#ifndef DRAUGR_AMG_H
#define DRAUGR_AMG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Julia runtime lifecycle (provided by PackageCompiler) ─────────────── */

/**
 * Initialize the Julia runtime.  Must be called once before any draugr_amg_*
 * function.  Pass (0, NULL) if you have no special arguments.
 */
void init_julia(int argc, char **argv);

/**
 * Shut down the Julia runtime.  Call once when the application exits.
 * Pass 0 for a normal exit.
 */
void shutdown_julia(int retcode);

/* ── Enum values ───────────────────────────────────────────────────────── */

/* Coarsening algorithms */
#define DRAUGR_AMG_COARSENING_AGGREGATION           0
#define DRAUGR_AMG_COARSENING_PMIS                  1
#define DRAUGR_AMG_COARSENING_HMIS                  2
#define DRAUGR_AMG_COARSENING_RS                    3
#define DRAUGR_AMG_COARSENING_AGGRESSIVE_PMIS       4
#define DRAUGR_AMG_COARSENING_AGGRESSIVE_HMIS       5
#define DRAUGR_AMG_COARSENING_SMOOTHED_AGGREGATION  6

/* Smoother types */
#define DRAUGR_AMG_SMOOTHER_JACOBI          0
#define DRAUGR_AMG_SMOOTHER_COLORED_GS      1
#define DRAUGR_AMG_SMOOTHER_SERIAL_GS       2
#define DRAUGR_AMG_SMOOTHER_SPAI0           3
#define DRAUGR_AMG_SMOOTHER_SPAI1           4
#define DRAUGR_AMG_SMOOTHER_L1_JACOBI       5
#define DRAUGR_AMG_SMOOTHER_CHEBYSHEV       6
#define DRAUGR_AMG_SMOOTHER_ILU0            7
#define DRAUGR_AMG_SMOOTHER_L1_COLORED_GS   8

/* Interpolation types */
#define DRAUGR_AMG_INTERPOLATION_DIRECT      0
#define DRAUGR_AMG_INTERPOLATION_STANDARD    1
#define DRAUGR_AMG_INTERPOLATION_EXTENDED_I  2

/* Cycle types */
#define DRAUGR_AMG_CYCLE_V  0
#define DRAUGR_AMG_CYCLE_W  1

/* Strength of connection */
#define DRAUGR_AMG_STRENGTH_ABSOLUTE  0
#define DRAUGR_AMG_STRENGTH_SIGNED    1

/* ── Config ────────────────────────────────────────────────────────────── */

/**
 * Create an AMG configuration.
 *
 * @param coarsening           CoarseningEnum value (DRAUGR_AMG_COARSENING_*)
 * @param smoother             SmootherEnum value   (DRAUGR_AMG_SMOOTHER_*)
 * @param interpolation        InterpolationEnum    (DRAUGR_AMG_INTERPOLATION_*)
 * @param strength             StrengthEnum         (DRAUGR_AMG_STRENGTH_*)
 * @param cycle                CycleEnum            (DRAUGR_AMG_CYCLE_*)
 * @param theta                Strength threshold (e.g. 0.25)
 * @param trunc_factor         Interpolation truncation factor (0 = none)
 * @param jacobi_omega         Jacobi relaxation weight (e.g. 2/3)
 * @param max_levels           Maximum AMG levels (e.g. 20)
 * @param max_coarse_size      Coarsest-level size for direct solve (e.g. 50)
 * @param pre_smoothing_steps  Pre-smoothing sweeps per level (e.g. 1)
 * @param post_smoothing_steps Post-smoothing sweeps per level (e.g. 1)
 * @param verbose              Verbosity: 0=silent, 1=summary, 2=per-iter
 * @param initial_coarsening   CoarseningEnum value for initial levels
 * @param initial_coarsening_levels Number of levels that use initial_coarsening
 * @param max_row_sum          hypre-like max_row_sum threshold (1.0 disables)
 * @param coarse_solve_on_cpu  0=false, 1=true
 *
 * @return  Config handle (> 0) on success, -1 on error.
 */
int32_t draugr_amg_config_create(int32_t coarsening, int32_t smoother,
                                 int32_t interpolation, int32_t strength,
                                 int32_t cycle, double theta,
                                 double trunc_factor, double jacobi_omega,
                                 int32_t max_levels, int32_t max_coarse_size,
                                 int32_t pre_smoothing_steps,
                                 int32_t post_smoothing_steps,
                                 int32_t verbose,
                                 int32_t initial_coarsening,
                                 int32_t initial_coarsening_levels,
                                 double max_row_sum,
                                 int32_t coarse_solve_on_cpu);

/* ── Setup / resetup ───────────────────────────────────────────────────── */

/**
 * Build an AMG hierarchy from a CSR matrix.
 *
 * @param n              Number of rows (square matrix, n×n)
 * @param nnz            Number of nonzeros
 * @param rowptr         Row-pointer array, length n+1
 * @param colval         Column-index array, length nnz
 * @param nzval          Nonzero-value array, length nnz
 * @param config_handle  Config handle from draugr_amg_config_create()
 * @param index_base     0 for C-style zero-based indexing,
 *                       1 for Fortran/Julia-style one-based indexing
 *
 * @return  Hierarchy handle (> 0) on success, -1 on error.
 */
int32_t draugr_amg_setup(int32_t n, int32_t nnz,
                         const int32_t *rowptr, const int32_t *colval,
                         const double *nzval,
                         int32_t config_handle,
                         int32_t index_base);

/**
 * Update an existing hierarchy with new matrix coefficients.
 * The sparsity pattern (rowptr, colval) must be identical to the
 * original call to draugr_amg_setup().
 *
 * @param index_base     0 for zero-based, 1 for one-based indexing
 *
 * @return  0 on success, -1 on error.
 */
int32_t draugr_amg_resetup(int32_t handle, int32_t n, int32_t nnz,
                           const int32_t *rowptr, const int32_t *colval,
                           const double *nzval,
                           int32_t config_handle,
                           int32_t index_base);

/* ── Solve / cycle ─────────────────────────────────────────────────────── */

/**
 * Solve Ax = b using AMG as a standalone solver.
 * The solution is written in-place into x.
 *
 * @param handle         Hierarchy handle
 * @param n              Vector length
 * @param x              Solution vector (in/out), length n
 * @param b              Right-hand side vector, length n
 * @param config_handle  Config handle
 * @param tol            Convergence tolerance (relative residual)
 * @param maxiter        Maximum iterations
 *
 * @return  Number of iterations on success, -1 on error.
 */
int32_t draugr_amg_solve(int32_t handle, int32_t n,
                         double *x, const double *b,
                         int32_t config_handle,
                         double tol, int32_t maxiter);

/**
 * Apply a single AMG cycle (preconditioner mode).
 * Improves x for the system Ax = b.
 *
 * @return  0 on success, -1 on error.
 */
int32_t draugr_amg_cycle(int32_t handle, int32_t n,
                         double *x, const double *b,
                         int32_t config_handle);

/* ── Cleanup ───────────────────────────────────────────────────────────── */

/**
 * Free an AMG hierarchy.
 * @return  0 on success, -1 if handle not found.
 */
int32_t draugr_amg_free(int32_t handle);

/**
 * Free an AMG configuration.
 * @return  0 on success, -1 if handle not found.
 */
int32_t draugr_amg_config_free(int32_t handle);

#ifdef __cplusplus
}
#endif

#endif /* DRAUGR_AMG_H */
