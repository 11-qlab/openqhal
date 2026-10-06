# Cover Letter

**To:** The Editors, *Quantum* (or *Physical Review Applied*)

**Re:** Submission of "OpenQHAL: A Portable Quantum Pulse Compiler
with Compiler–Decoder Co-Design for qLDPC Codes"

**Date:** [DATE]

---

Dear Editors,

I am pleased to submit the enclosed manuscript for consideration as a
Research Article in *[JOURNAL]*.

## Summary

Quantum computing is at an inflection point: the field has converged on
qLDPC codes as the path to fault tolerance, but no portable software
infrastructure exists to compile circuits for them. Qiskit Pulse was
deprecated in 2024 without a replacement. Every lab builds its own
pulse-level stack, badly.

This paper presents **OpenQHAL**, an open-source hardware abstraction
layer that:

1. **Compiles logical circuits to physical I/Q waveforms** through a
   frozen C ABI, with four backends (statevector simulator, IBM Quantum,
   QICK, Zurich HDAWG).
2. **Bridges the compiler–decoder interface** for qLDPC codes by
   annotating every compiled pulse with its check structure and
   producing a per-check detector error model. This closes the circular
   dependency named as unsolved in the 2026 FTQC compilation survey.
3. **Achieves depth-optimal syndrome extraction** for the
   $[[98,6,12]]$ Bivariate Bicycle code, matching the published lower
   bound that depth 6 is provably unattainable.
4. **Validates on real hardware** on IBM's 156-qubit Heron r2 processor,
   with Bell-state fidelity 98.2% and GHZ-4 fidelity 95.6%.

## Novelty

The compiler–decoder co-design bridge is, to our knowledge, the first
open-source implementation that carries qLDPC check structure through
the entire compilation pipeline to the decoder. The 2026 survey
*Quantum Compiler Design for Fault-Tolerant Quantum Computing* names
this as an open problem. No prior work has closed this loop in a
portable, hardware-agnostic way.

A secondary contribution is the discovery of a concrete IBM hardware
constraint: mid-circuit reset instructions require an explicit
`ConvertToMidCircuitResetAndMeasure` transpiler pass, or the backend
rejects the job with error 6056. This is documented and reproducible.

## Suitability for *[JOURNAL]*

The work is empirical, reproducible, and open-source. All code,
calibration files, and raw measurement data are available at
https://github.com/11-qlab/openqhal under Apache 2.0. Every figure and
table in the manuscript can be regenerated from a single `git clone`
followed by the commands in Appendix A.

## Competing interests

The author declares no competing financial interests.

## Suggested reviewers

We suggest the following reviewers with expertise in quantum compiler
infrastructure, qLDPC codes, and hardware abstraction:

- [Name], [Institution] — qLDPC codes, decoding
- [Name], [Institution] — quantum compilers, transpilation
- [Name], [Institution] — quantum networking, HAL design

(Reviewer suggestions are optional; we defer to the editors.)

---

Thank you for your consideration. I look forward to your response.

Sincerely,

**11-qlab**
https://github.com/11-qlab
