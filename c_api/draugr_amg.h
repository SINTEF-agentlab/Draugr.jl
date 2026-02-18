/*
 * draugr_amg.h — C API for the Draugr AMG library
 *
 * Compiled from Draugr.jl via PackageCompiler.jl.
 * Link against libdraugr.so (Linux), libdraugr.dylib (macOS),
 * or draugr.dll (Windows).
 *
 * Lifecycle:
 *   1. Call init_julia(argc, argv) once at startup.
 *   2. Create configs, build hierarchies, solve/cycle as needed.
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

/* ── Error reporting ───────────────────────────────────────────────────── */

/**
 * Retrieve the last error message from a failed draugr_amg_* call.
 *
 * @return  Null-terminated error string.  Empty string if no error.
 *          The pointer is valid until the next draugr_amg_* call.
 */
const char *draugr_amg_last_error(void);

/* ── Config (JSON-based) ───────────────────────────────────────────────── */

/**
 * Create an AMG configuration from a JSON string.
 *
 * All keys are optional.  Pass "{}" or NULL for defaults (see AMGConfig()
 * in types.jl for the default values).
 *
 * Unknown keys are silently ignored (allows pass-through from host apps).
 *
 * Supported JSON keys:
 *
 *   coarsening             (string)  "aggregation", "pmis", "hmis", "rs",
 *                                    "aggressive_pmis", "aggressive_hmis",
 *                                    "smoothed_aggregation"
 *
 *   interpolation          (string or object)
 *                          String:   "direct", "standard", "extended_i"
 *                          Object:   {"type": "...", "trunc_factor": ...,
 *                                     "max_elements": ..., "norm_p": ...,
 *                                     "rescale": ...}
 *
 *   smoother               (string or object)
 *                          String:   "jacobi", "colored_gs", "serial_gs",
 *                                    "spai0", "spai1", "l1_jacobi",
 *                                    "l1_colored_gs", "chebyshev", "ilu0"
 *                          Object:   {"type": "...", "omega": ...}
 *
 *   strength               (string)  "absolute", "signed"
 *   cycle                  (string)  "v", "w"
 *   theta                  (double)  Strength-of-connection threshold
 *   max_levels             (int)
 *   max_coarse_size        (int)
 *   pre_smoothing_steps    (int)
 *   post_smoothing_steps   (int)
 *   verbose                (int)     0=silent, 1=summary, 2=per-iter
 *   jacobi_omega           (double)  Also settable via smoother "omega"
 *   max_row_sum            (double)  Dependency weakening threshold
 *   coarse_solve_on_cpu    (bool)
 *   initial_coarsening     (string)  Coarsening for first N levels
 *   initial_coarsening_levels (int)
 *
 * @param json_str  Null-terminated JSON string. Pass "{}" or NULL for defaults.
 * @return  Config handle (> 0) on success, -1 on error.
 *          Call draugr_amg_last_error() for the error message.
 */
int32_t draugr_amg_config_from_json(const char *json_str);

/* ── Setup / resetup ───────────────────────────────────────────────────── */

/**
 * Build an AMG hierarchy from a CSR matrix.
 *
 * @param n              Number of rows (square matrix, n×n)
 * @param nnz            Number of nonzeros
 * @param rowptr         Row-pointer array, length n+1
 * @param colval         Column-index array, length nnz
 * @param nzval          Nonzero-value array, length nnz
 * @param config_handle  Config handle from draugr_amg_config_from_json()
 * @param index_base     0 for C-style zero-based indexing,
 *                       1 for Fortran/Julia-style one-based indexing
 * @param allow_partial_resetup  1 to build restriction maps for fast partial
 *                       resetup (default use case), 0 to skip them for a
 *                       faster initial setup (only full resetup available)
 *
 * @return  Hierarchy handle (> 0) on success, -1 on error.
 */
int32_t draugr_amg_setup(int32_t n, int32_t nnz,
                         const int32_t *rowptr, const int32_t *colval,
                         const double *nzval,
                         int32_t config_handle,
                         int32_t index_base,
                         int32_t allow_partial_resetup);

/**
 * Update an existing hierarchy with new matrix data.
 *
 * When partial=1: only matrix values, Galerkin products, smoothers and the
 * coarse solver are recomputed — the coarsening structure is kept.  This is
 * the fast path for coefficient-only updates (same sparsity pattern).
 * Requires the hierarchy to have restriction maps (setup/resetup with
 * allow_partial_resetup=1).
 *
 * When partial=0: the hierarchy is fully rebuilt (new coarsening, prolongation,
 * smoothers) while reusing workspace arrays from the existing hierarchy.
 * More efficient than freeing + re-calling draugr_amg_setup() because
 * workspace vectors whose sizes haven't changed are reused.
 *
 * @param index_base     0 for zero-based, 1 for one-based indexing
 * @param partial        1 for coefficient-only update, 0 for full rebuild
 * @param allow_partial_resetup  1 to build restriction maps for future
 *                       partial resetups (only relevant when partial=0)
 *
 * @return  0 on success, -1 on error.
 */
int32_t draugr_amg_resetup(int32_t handle, int32_t n, int32_t nnz,
                           const int32_t *rowptr, const int32_t *colval,
                           const double *nzval,
                           int32_t config_handle,
                           int32_t index_base,
                           int32_t partial,
                           int32_t allow_partial_resetup);

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
