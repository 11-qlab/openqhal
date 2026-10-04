/*
 * qhal.h — Quantum Hardware Abstraction Layer
 *
 * Frozen C ABI. This file is the contract between:
 *   - High-level code (Python, Rust, anything with a C FFI)
 *   - The pulse compiler core (C++)
 *   - Hardware backends (Zurich, Spectrum, QICK, simulators)
 *
 * ABI version: 1. Breaking changes require a version bump.
 */

#ifndef QHAL_H
#define QHAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Version                                                            */
/* ------------------------------------------------------------------ */

#define QHAL_ABI_VERSION 1
#define QHAL_VERSION_MAJOR 0
#define QHAL_VERSION_MINOR 1
#define QHAL_VERSION_PATCH 0

/* ------------------------------------------------------------------ */
/* Limits                                                             */
/* ------------------------------------------------------------------ */

#define QHAL_MAX_QUBITS_PER_GATE 4
#define QHAL_MAX_GATE_PARAMS     4
#define QHAL_MAX_CHANNELS        16
#define QHAL_MAX_ERROR_MSG       512

/* ------------------------------------------------------------------ */
/* Enumerations                                                       */
/* ------------------------------------------------------------------ */

typedef enum {
    QHAL_OK              =  0,
    QHAL_ERR_GENERIC     = -1,
    QHAL_ERR_INVALID_ARG = -2,
    QHAL_ERR_NO_MEMORY   = -3,
    QHAL_ERR_UNKNOWN_GATE = -4,
    QHAL_ERR_CALIBRATION = -5,
    QHAL_ERR_SCHEDULING  = -6,
    QHAL_ERR_BACKEND     = -7,
    QHAL_ERR_IO          = -8,
    QHAL_ERR_VERSION     = -9
} qhal_status_t;

typedef enum {
    QHAL_ENV_SQUARE   = 0,   /* Ω(t) = A                              */
    QHAL_ENV_GAUSSIAN = 1,   /* Ω(t) = A·exp(-(t-t₀)²/(2σ²))         */
    QHAL_ENV_DRAG     = 2,   /* Gaussian + iβ·(dΩ/dt)/Δ              */
    QHAL_ENV_SECH     = 3,   /* Ω(t) = A·sech((t-t₀)/σ)              */
    QHAL_ENV_HERMITE  = 4,   /* Hermite-Gaussian                     */
    QHAL_ENV_CUSTOM   = 5    /* User-supplied samples                */
} qhal_envelope_t;

typedef enum {
    QHAL_CH_I       = 0,
    QHAL_CH_Q       = 1,
    QHAL_CH_LASER   = 2,
    QHAL_CH_READOUT = 3,
    QHAL_CH_FLUX    = 4
} qhal_channel_t;

/* ------------------------------------------------------------------ */
/* Core types                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    const char* name;                              /* "H", "CNOT", ... */
    int32_t     qubit_count;
    int32_t     qubits[QHAL_MAX_QUBITS_PER_GATE];
    double      params[QHAL_MAX_GATE_PARAMS];
} qhal_gate_t;

typedef struct {
    int32_t         channel;
    double          start_ns;
    double          duration_ns;
    double          frequency_hz;
    double          amplitude;
    double          phase_rad;
    qhal_envelope_t envelope;
    double          drag_beta;
    double          envelope_param;
} qhal_pulse_t;

typedef struct {
    qhal_pulse_t* pulses;
    int32_t       count;
    double        total_duration_ns;
    int32_t       shot_count;
} qhal_program_t;

typedef struct {
    const char* device_name;
    int32_t     qubit_count;
    const char* calibration_json;
} qhal_device_t;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                          */
/* ------------------------------------------------------------------ */

int  qhal_init(void);
void qhal_shutdown(void);

const char* qhal_version(void);
int         qhal_abi_version(void);

/* ------------------------------------------------------------------ */
/* Compile: gates → pulse program                                     */
/* ------------------------------------------------------------------ */

qhal_status_t qhal_compile(const qhal_gate_t*    gates,
                           int32_t               gate_count,
                           const qhal_device_t*  device,
                           qhal_program_t*       out);

void qhal_free_program(qhal_program_t* prog);

/* ------------------------------------------------------------------ */
/* Serialize / deserialize                                            */
/* ------------------------------------------------------------------ */

qhal_status_t qhal_save_program  (const qhal_program_t* prog,
                                  const char*           path);

qhal_status_t qhal_load_program  (const char*     path,
                                  qhal_program_t* out);

/* ------------------------------------------------------------------ */
/* Execute                                                            */
/* ------------------------------------------------------------------ */

qhal_status_t qhal_execute(const qhal_device_t*  device,
                           const qhal_program_t* prog,
                           int32_t*              results,
                           int32_t               shot_count);

/* ------------------------------------------------------------------ */
/* Diagnostics                                                        */
/* ------------------------------------------------------------------ */

const char* qhal_last_error(void);
void        qhal_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* QHAL_H */