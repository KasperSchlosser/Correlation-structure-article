# code taken from NABQR https://github.com/bast0320/nabqr
#
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from pathlib import Path

PATH = Path("C:/Users/kpfs/Projects/Correlation-structure-article")

#%%
def generate_ou_ensembles(
        X: np.ndarray,
        kappa: float,
        sigma: float,
        chunk_size: int = 24,
        n_ensembles: int = 50
) -> np.ndarray:
    """
    Generate continuous Ornstein-Uhlenbeck (OU) ensemble paths that revert
    to the given reference series X[t], in chunk_size increments, but also
    simulate 'extra' future steps to account for OU lag and shift
    them back so the paths better align with X in real-time.

    The ensemble is clipped to remain within [0,1].

    Parameters
    ----------
    X : np.ndarray
        Reference series of length T that serves as the time-varying mean
        for each OU path.
    kappa : float
        Mean-reversion speed for the OU process.
        The characteristic lag ~ 1/kappa.
    sigma : float
        Diffusion (volatility) parameter.
    chunk_size : int, optional
        Size of each chunk in timesteps. Defaults to 24.
    n_ensembles : int, optional
        Number of ensemble paths to generate. Defaults to 50.

    Returns
    -------
    Y_corrected : np.ndarray, shape (T, n_ensembles)
        The lag-corrected OU ensemble paths, each of length T.

    Notes
    -----
    - We break the timeline [0..T-1] into blocks of `chunk_size` steps.
      At chunk boundaries, each ensemble path is continuous
      (meaning; the new chunk starts where the old chunk ended).
    - We simulate extra steps (about 1/kappa) at the end, then shift the entire
      simulation backward by ~1/kappa to reduce the effective lag in real time.
    - For fractional lag, a simple linear interpolation is applied.
    - This "lag correction" is heuristic but often aligns the OU paths with
      X(t) more tightly when the reversion is slow.
    """

    T = len(X)

    # ------------------------------
    # 1. Calculate the lag
    # ------------------------------
    # For an OU with mean reversion kappa, characteristic timescale is 1/kappa
    lag = 1.0 / kappa
    lag_int = int(np.floor(lag))
    lag_frac = lag - lag_int

    # We'll simulate T + lag_int + 1 steps. The "+1" helps with
    # fractional shift interpolation.
    sim_len = T + lag_int + 1

    # ------------------------------
    # 2. Allocate array for ensembles
    # ------------------------------
    # We'll simulate a bigger array, shape (sim_len, n_ensembles).
    Y_big = np.zeros((sim_len, n_ensembles))

    # For the very first step, set them all to X[0]
    Y_big[0, :] = X[0]

    # Precompute chunk boundaries for the extended length
    chunk_starts = list(range(0, sim_len, chunk_size))

    # ------------------------------
    # 3. Simulate with chunking
    # ------------------------------
    for idx_chunk in range(len(chunk_starts)):
        start = chunk_starts[idx_chunk]
        end = min(start + chunk_size, sim_len)

        if start > 0:
            # Continue from previous chunk's last step
            Y_big[start, :] = Y_big[start - 1, :]

        # Step through the chunk
        for i in range(start, end - 1):
            # We must clamp i to the length of X when indexing X:
            # Because beyond T-1, we can assume X stays at X[-1]
            iX = min(i, T - 1)

            # OU update
            drift = kappa * (X[iX] - Y_big[i, :])
            diffusion = sigma * np.random.randn(n_ensembles)
            Y_big[i + 1, :] = Y_big[i, :] + drift + diffusion

            # Enforce [0, 1] clipping
            Y_big[i + 1, :] = np.clip(Y_big[i + 1, :], 0.0, 1.0)

    # Now we have Y_big of length sim_len = T + lag_int + 1

    # ------------------------------
    # 4. Shift the result backward
    #    by "lag = lag_int + lag_frac"
    # ------------------------------
    # We'll produce a final array Y_corrected of shape (T, n_ensembles).
    Y_corrected = np.zeros((T, n_ensembles))

    # If there's no fractional part, we can just slice
    # If lag_frac == 0, then just do:
    # Y_corrected[t] = Y_big[t + lag_int]
    # for t = 0..T-1
    if np.isclose(lag_frac, 0.0, atol=1e-14):
        for t in range(T):
            Y_corrected[t, :] = Y_big[t + lag_int, :]
    else:
        # We have a fractional shift, so do a linear interpolation
        # Y_corrected[t] = (1 - alpha)* Y_big[t + lag_int] + alpha * Y_big[t + lag_int + 1]
        # where alpha = lag_frac
        alpha = lag_frac
        for t in range(T):
            i1 = t + lag_int
            i2 = i1 + 1
            # clamp i2 to avoid out-of-bounds (should not occur with sim_len = T + lag_int + 1)
            i2 = min(i2, sim_len - 1)
            Y_corrected[t, :] = (1 - alpha) * Y_big[i1, :] + alpha * Y_big[i2, :]

    # We have a T x n_ensembles array that’s effectively “time-shifted” back by lag steps
    return Y_corrected

def simulate_wind_power_sde(params: Dict[str, float], T: float = 500, dt: float = 1.0, n_ensembles = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate wind power production using an Ornstein-Uhlenbeck process with GARCH volatility
    and jumps of normally distributed sizes driven by a Poisson process. The mean reversion
    is state-dependent with a repelling mechanism near 1.0 (upper boundary), and the diffusion
    term vanishes at the boundaries to avoid unphysical values outside [0, 1].

    A few additional tweaks include:
    - GARCH volatility that captures 'vol_shock' from recent values.
    - Repellent forces that strengthen near 1.0, reducing both the drift and diffusion.
    - Jumps that can persist over multiple steps, and become more negative if values are near 1.0.

    Parameters
    ----------
    params : Dict[str, float]
        A dictionary containing all model parameters:

        - X0 : float
            Initial wind power production level in [0, 1].
        - theta : float
            Long-term mean level; typically in [0, 1].
        - kappa : float
            Mean reversion speed (absolute value is used).
        - sigma_base : float
            Base volatility level (absolute value is used).
        - alpha : float
            ARCH parameter (absolute value is used).
        - beta : float
            GARCH parameter; must be in [0, 1].
        - lambda_jump : float
            Intensity of jump arrivals in the Poisson process (absolute value is used).
        - jump_mu : float
            Mean jump size (can be positive or negative).
        - jump_sigma : float
            Standard deviation of jump sizes (absolute value is used).
    T : float, optional
        The end time of the simulation (total number of steps is T/dt). Default is 500.
    dt : float, optional
        The size of each time step. Default is 1.0.

    Returns
    -------
    t : np.ndarray
        Array of time points of length N = int(T/dt).
    X : np.ndarray
        Simulated wind power production values of length N, clipped to the interval [0, 1].

    Notes
    -----
    - The drift term implements a state-dependent mean reversion that weakens
      near 1.0 and introduces a strong downward force very close to 1.0.
    - The diffusion term is modified as
      (X_t * (1 - X_t)) * (X_t / (X_t + 0.5)) dB_t,
      ensuring it decreases to zero when X_t is near 0 or 1.
    - GARCH effects are included to model changing volatility based on recent
      shocks in the process.
    - Jumps arrive according to a Poisson process with random normal magnitudes,
      and can persist over multiple time steps with some decay.

    Examples
    --------
    """
    # Unpack parameters and ensure they're valid
    X0 = np.clip(params['X0'], 0, 1)
    theta = np.clip(params['theta'], 0, 1)  # Mean level
    kappa = abs(params['kappa'])  # Mean reversion speed
    sigma_base = abs(params['sigma_base'])  # Base volatility

    # GARCH parameters
    alpha = abs(params['alpha'])  # ARCH parameter
    beta = np.clip(params['beta'], 0, 1)  # GARCH parameter

    # Jump parameters
    lambda_jump = abs(params['lambda_jump'])  # Jump intensity
    jump_mu = params['jump_mu']  # Jump mean
    jump_sigma = abs(params['jump_sigma'])  # Jump size volatility

    # Time grid
    t = np.linspace(0, T, int(T / dt))
    N = len(t)

    # Initialize arrays
    X = np.zeros(N)
    X[0] = X0

    # Initialize GARCH volatility
    sigma = np.zeros(N)
    sigma[0] = sigma_base

    # Initialize jump state variables
    jump_state = 0
    previous_jump = 0

    # Generate jumps
    jump_times = np.random.poisson(max(1e-10, lambda_jump * dt), N)
    jump_sizes = np.random.normal(jump_mu, max(1e-10, jump_sigma), N)

    # Simulation
    for i in range(1, N):
        # Update GARCH volatility
        vol_shock = np.abs(X[i - 1] - X[max(0, i - 2)]) / dt
        sigma[i] = np.sqrt(alpha * vol_shock ** 2 + beta * sigma[i - 1] ** 2)

        # Get previous value and recent average
        X_prev = X[i - 1]
        recent_avg = np.mean(X[max(0, i - 5):i]) if i > 5 else X_prev

        # State-dependent mean reversion with repellent near 1.0
        if X_prev > 0.99:
            # Strong repellent force very close to 1.0
            drift = -kappa * 2.0 * dt
        elif X_prev > 0.7:
            drift = kappa * (theta - X_prev) * dt * 0.5
        else:
            drift = kappa * (theta - X_prev) * dt

        # Modified diffusion term with memory
        diff_term = (X_prev * (1 - X_prev)) * 0.4 * (X_prev / (X_prev + 0.1))
        if X_prev > 0.75:
            # Reduce volatility if we've been stable at high values
            if abs(X_prev - recent_avg) < 0.1:
                diff_term *= 0.5

        # Additional repellent in diffusion term near 1.0
        if X_prev > 0.95:
            diff_term *= 0.3

        diffusion = sigma[i] * diff_term * np.sqrt(dt) * np.random.normal()

        # Persistent jumps with stronger downward bias near 1.0
        if jump_state > 0:
            # Continue the previous jump with decay
            jump = previous_jump * 0.9
            jump_state -= 1
        elif jump_times[i] > 0:
            # Start new jump
            jump = jump_sizes[i]
            if X_prev > 0.98:
                jump = -abs(jump) * 1.5
            jump_state = np.random.geometric(0.2)
            previous_jump = jump
            # Make downward jumps more persistent at high values
            if X_prev > 0.8 and jump < 0:
                jump_state = int(jump_state * 1.5)
        else:
            jump = 0

        # Update process
        X[i] = X[i - 1] + drift + diffusion + jump

        # Ensure bounds
        X[i] = np.clip(X[i], 0, 1)

    # Apply moving average filter
    X = np.convolve(X, np.ones((4,)) / 4, mode='valid')

    # The following line is referencing an undefined 'initial_params'.
    # It seems intended to reuse X0. We'll just assume it was meant as X0 in params.
    X = np.concatenate(([X0] * 3, X))  # Append initial value to keep the same length

    # Final adjustment and bounds
    X = np.clip(X * 1.05, 0, 1)

    # Now we will generate ensembles from X.
    # We will use an OU process to generate ensembles

    ensembles = generate_ou_ensembles(
        X,
        kappa=0.3,
        sigma=0.05,
        chunk_size=24,  # 24 hours
        n_ensembles=n_ensembles
    )
    return t, X, ensembles
#%%
if __name__ == "__main__":

    zones = ["DK1-onshore", "DK2-onshore", "DK1-offshore", "DK2-offshore"]
    zone_max = {
        "DK1-onshore": 3560,
        "DK2-onshore": 690,
        "DK1-offshore": 1370,
        "DK2-offshore": 990,
    }

    params = {'X0': 0.6,
              'theta': 0.77,
              'kappa': 0.12,
              'sigma_base': 1.05,
              'alpha': 0.57,
              'beta': 1.2,
              'lambda_jump': 0.045,
              'jump_mu': 0.0,
              'jump_sigma': 0.1
    }
    # Your existing code
    ensembles = pd.read_pickle(PATH / "Data" / "Real" / "ensembles.pkl")
    observations = pd.read_pickle(PATH / "Data" / "Real" / "observations.pkl")


    for zone in ensembles.index.unique(0):
        t, X, ens = simulate_wind_power_sde(params, T=len(ensembles.loc[zone]), dt=1.0, n_ensembles = 52)


        ensembles.loc[zone] = ens * zone_max[zone]
        observations.loc[zone] = X * zone_max[zone]*0.99 + 0.01


    ensembles.to_pickle("Data/Simulated/ensembles.pkl")
    observations.to_pickle("Data/Simulated/observations.pkl")



