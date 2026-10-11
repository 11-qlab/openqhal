"""Optimized 6x6 Heisenberg AFM RBM-NQS VMC with SR.

v3 changes vs. v2:
    - ALPHA 2 -> 4 (main expressivity fix; previous 2.26% error was the
      alpha=2 ceiling, not an optimization problem)
    - Cholesky solve for the sample-space SR system (G is SPD)
    - SR_SUBSAMPLE: build the Gram matrix on a subset of chains while
      still using all chains for the energy estimate
    - per-stage timing printed every 50 iterations
    - SR_DAMPING 1e-3 -> 3e-3 (raw |d| ~ 11 was being clipped hard;
      slightly stronger damping avoids relying on the clip)

Basis convention:
    Marshall-rotated basis. Ground state is nodeless-positive there,
    so a real non-negative RBM can represent |psi|, and the off-diagonal
    (S+ S- + S- S+) contribution carries a minus sign. Do NOT flip the
    sign of e_off unless you also switch to a complex RBM.
"""

import os
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.flatten_util import ravel_pytree


# ============================================================
# CONFIG
# ============================================================

L = 6
N = L * L
J = 1.0

# --- expressivity ---
ALPHA = 4                # was 2
M = ALPHA * N

# --- sampling ---
N_CHAINS = 2048
SR_SUBSAMPLE = 1024      # chains used to build the SR Gram matrix

N_SWEEPS_WARMUP = 20
N_SWEEPS_STEADY = 8
N_WARMUP_ITERS  = 30

# --- optimization ---
N_ITER     = 600
SR_DAMPING = 3e-3        # was 1e-3
LR         = 0.02
LR_MIN     = 0.005

MAX_UPDATE_NORM = 1.0
CLIP_ELOC       = 5.0

SEED = 0

REF_ENERGIES = {4: -0.7017, 6: -0.6789, 8: -0.6735}
E_REF = REF_ENERGIES.get(L, None)

os.makedirs("results", exist_ok=True)


# ============================================================
# JAX banner
# ============================================================

print("JAX devices:", jax.devices())
try:
    print("JAX version:", jax.__version__)
except Exception:
    pass
print("x64 enabled:", jax.config.x64_enabled)
print()


# ============================================================
# LATTICE
# ============================================================

bond_list = []
for r in range(L):
    for c in range(L):
        i = r * L + c
        bond_list.append((i, r * L + ((c + 1) % L)))
        bond_list.append((i, ((r + 1) % L) * L + c))

bonds     = jnp.asarray(bond_list, dtype=jnp.int32)
BOND_I    = bonds[:, 0]
BOND_J    = bonds[:, 1]
ARANGE_B  = jnp.arange(len(bond_list), dtype=jnp.int32)
N_BONDS   = len(bond_list)

print(f"Lattice       = {L}x{L}")
print(f"N spins       = {N}")
print(f"Bonds         = {N_BONDS}")
print(f"RBM alpha     = {ALPHA}")
print(f"Hidden units  = {M}")
print(f"Parameters    = {N + M + N*M}")
print(f"Chains        = {N_CHAINS}  (SR on {SR_SUBSAMPLE})")
print(f"Sweeps        = {N_SWEEPS_WARMUP} -> {N_SWEEPS_STEADY}")
print(f"Iterations    = {N_ITER}")
print(f"SR damping    = {SR_DAMPING}")
print(f"LR            = {LR} -> {LR_MIN} (cosine)")
print()


# ============================================================
# RBM
# ============================================================

key = jax.random.PRNGKey(SEED)
k1, k2, k3 = jax.random.split(key, 3)

params = {
    "a": 0.01 * jax.random.normal(k1, (N,)),
    "b": 0.01 * jax.random.normal(k2, (M,)),
    "W": 0.01 * jax.random.normal(k3, (N, M)),
}

flat_params, unflatten_fn = ravel_pytree(params)
N_PARAMS = flat_params.shape[0]

print(f"Total RBM parameters = {N_PARAMS}")
print()


def logpsi(p, sigma):
    theta = sigma @ p["W"] + p["b"]
    logcosh = jnp.logaddexp(theta, -theta) - jnp.log(2.0)
    return sigma @ p["a"] + jnp.sum(logcosh, axis=-1)


def logpsi_flat(fp, sigma):
    return logpsi(unflatten_fn(fp), sigma)


# ============================================================
# LOCAL ENERGY (batched over bond flips)
# ============================================================

def local_energy_single(p, sigma):
    si = sigma[BOND_I]
    sj = sigma[BOND_J]

    e_diag = J * 0.25 * jnp.sum(si * sj)

    flipped = jnp.tile(sigma[None, :], (N_BONDS, 1))
    flipped = flipped.at[ARANGE_B, BOND_I].set(sigma[BOND_J])
    flipped = flipped.at[ARANGE_B, BOND_J].set(sigma[BOND_I])

    lp_sigma   = logpsi(p, sigma)
    lp_flipped = logpsi(p, flipped)

    anti = (si * sj) < 0
    contrib = jnp.where(anti, jnp.exp(lp_flipped - lp_sigma), 0.0)
    e_off = -J * 0.5 * jnp.sum(contrib)     # Marshall-rotated sign

    return e_diag + e_off


local_energy_batch = jax.jit(
    jax.vmap(local_energy_single, in_axes=(None, 0))
)


# ============================================================
# MCMC
# ============================================================

def step_chains(p, chains, s1, s2, u):
    def swap_single(chain, i, j):
        xi = chain[i]
        xj = chain[j]
        return chain.at[i].set(xj).at[j].set(xi)

    proposed = jax.vmap(swap_single)(chains, s1, s2)

    lp_c = jnp.real(logpsi(p, chains))
    lp_p = jnp.real(logpsi(p, proposed))
    dlog = 2.0 * (lp_p - lp_c)

    accept = jnp.log(u) < dlog
    return jnp.where(accept[:, None], proposed, chains)


@partial(jax.jit, static_argnums=(2,))
def mcmc_sweep(key, p, n_sweeps, chains):
    C, n_sites = chains.shape
    num_steps = n_sweeps * n_sites

    k1, k2, k3 = jax.random.split(key, 3)
    sites1   = jax.random.randint(k1, (num_steps, C), 0, n_sites)
    sites2   = jax.random.randint(k2, (num_steps, C), 0, n_sites)
    uniforms = jax.random.uniform (k3, (num_steps, C))

    def body(chains, data):
        s1, s2, u = data
        return step_chains(p, chains, s1, s2, u), None

    chains, _ = jax.lax.scan(body, chains, (sites1, sites2, uniforms))
    return chains


# ============================================================
# SR
# ============================================================

def logpsi_single_real(fp, sigma):
    return jnp.real(logpsi_flat(fp, sigma))


grad_single = jax.grad(logpsi_single_real, argnums=0)


@jax.jit
def log_derivatives(fp, chains):
    return jax.vmap(grad_single, in_axes=(None, 0))(fp, chains)


@jax.jit
def sr_step_from_O(fp, chains_all, chains_sr):
    """Energy from all chains, SR update from the subsample.

    Sample-space SR:
        delta = O_c^T (O_c O_c^T / S + lambda I)^-1 (E_c / S)
    solved with Cholesky since the matrix is SPD.
    """
    p = unflatten_fn(fp)
    S = chains_sr.shape[0]

    energies_all = jnp.real(local_energy_batch(p, chains_all))
    E_mean = jnp.mean(energies_all)

    # energies for the SR subset (recompute — cheap relative to O)
    energies_sr = jnp.real(local_energy_batch(p, chains_sr))
    e_std = jnp.std(energies_sr) + 1e-12
    energies_sr = jnp.clip(
        energies_sr,
        E_mean - CLIP_ELOC * e_std,
        E_mean + CLIP_ELOC * e_std,
    )
    E_mean_sr = jnp.mean(energies_sr)

    O = log_derivatives(fp, chains_sr)        # (S, Np)
    O_c = O - jnp.mean(O, axis=0, keepdims=True)
    E_c = energies_sr - E_mean_sr

    G = (O_c @ O_c.T) / S
    G = G + SR_DAMPING * jnp.eye(S, dtype=G.dtype)

    # Cholesky factor, then solve G x = rhs
    L_chol = jsp_linalg.cholesky(G, lower=True)
    rhs = E_c / S
    lam = jsp_linalg.cho_solve((L_chol, True), rhs)

    delta = O_c.T @ lam
    norm  = jnp.linalg.norm(delta)
    scale = jnp.minimum(1.0, MAX_UPDATE_NORM / (norm + 1e-12))
    delta = delta * scale

    return E_mean, delta, norm


# ============================================================
# INIT
# ============================================================

key, sub = jax.random.split(key)

s0 = jnp.concatenate([jnp.ones(N // 2), -jnp.ones(N // 2)])
chain_keys = jax.random.split(sub, N_CHAINS)
chains = jax.vmap(lambda k: jax.random.permutation(k, s0))(chain_keys)

print("Initial chains:", chains.shape)
print()


# ============================================================
# TRAINING
# ============================================================

hist        = []
best_energy = np.inf
best_params = flat_params

# ---- warm compile (avoid timing the first iteration as if it were typical)
_ = mcmc_sweep(jax.random.PRNGKey(0), unflatten_fn(flat_params), 1, chains[:64])
_ = sr_step_from_O(flat_params, chains[:64], chains[:32])
jax.block_until_ready(_[0])

t0 = time.time()
print("Starting optimized SR training (alpha={})...".format(ALPHA))
print()

for it in range(N_ITER):

    frac = it / max(N_ITER - 1, 1)
    current_lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + np.cos(np.pi * frac))

    n_sweeps = N_SWEEPS_WARMUP if it < N_WARMUP_ITERS else N_SWEEPS_STEADY

    key, sub = jax.random.split(key)
    current_params = unflatten_fn(flat_params)

    t_mcmc0 = time.time()
    chains = mcmc_sweep(sub, current_params, n_sweeps, chains)
    jax.block_until_ready(chains)
    t_mcmc = time.time() - t_mcmc0

    chains_sr = chains[:SR_SUBSAMPLE]

    t_sr0 = time.time()
    E_mean, delta, delta_norm = sr_step_from_O(flat_params, chains, chains_sr)
    jax.block_until_ready(delta)
    t_sr = time.time() - t_sr0

    energy = float(jnp.real(E_mean))

    flat_params = flat_params - current_lr * delta
    hist.append(energy)

    if energy < best_energy:
        best_energy = energy
        best_params = flat_params

    if it % 10 == 0 or it == N_ITER - 1:
        elapsed = time.time() - t0
        e_site  = energy / N

        if E_REF is not None:
            rel_error = abs(e_site - E_REF) / abs(E_REF) * 100.0
            err_text  = f"error={rel_error:.4f}%"
        else:
            err_text = ""

        print(
            f"iter {it:4d}  E/N={e_site:.8f}  {err_text}  "
            f"lr={current_lr:.5f}  |d|={float(delta_norm):.4e}  "
            f"mcmc={t_mcmc:.2f}s sr={t_sr:.2f}s  time={elapsed:.1f}s"
        )

    if E_REF is not None:
        e_site    = energy / N
        rel_error = abs(e_site - E_REF) / abs(E_REF)
        if rel_error <= 0.001:
            print()
            print("TARGET REACHED: relative error <= 0.1%")
            break


# ============================================================
# FINAL
# ============================================================

best_E_per_site = best_energy / N

print()
print("==============================================")
print(f"Best E/N = {best_E_per_site:.8f}")

if E_REF is not None:
    best_error = abs(best_E_per_site - E_REF) / abs(E_REF) * 100.0
    print(f"Reference = {E_REF:.8f}")
    print(f"Best relative error = {best_error:.6f}%")

print(f"Total iterations = {len(hist)}")
print(f"Runtime = {time.time() - t0:.1f}s")
print("==============================================")


# ============================================================
# SAVE
# ============================================================

output_file = f"results/08_sr_{L}x{L}_alpha{ALPHA}.npz"

np.savez(
    output_file,
    hist=np.asarray(hist),
    E_per_site=best_E_per_site,
    best_energy=best_energy,
    L=L, N=N,
    ALPHA=ALPHA,
    N_CHAINS=N_CHAINS,
    SR_SUBSAMPLE=SR_SUBSAMPLE,
    N_SWEEPS_WARMUP=N_SWEEPS_WARMUP,
    N_SWEEPS_STEADY=N_SWEEPS_STEADY,
    N_ITER=len(hist),
    SR_DAMPING=SR_DAMPING,
    LR=LR,
    seed=SEED,
)

print()
print(f"Saved: {output_file}")
