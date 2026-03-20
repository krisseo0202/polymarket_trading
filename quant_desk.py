import numpy as np

def simulate_binary_contract(S0, K, mu, sig, T, N_paths=100_000):
    """
    Monte Carlo Simulation for a binary contract.

    S0: Current asset price
    K:  Strike/ threshold
    mu: Annual drift
    sig:    Anuual Voltality
    T:  Time to expiry in years
    N_paths: Number of simulated paths
    """

    Z = np.random.standard_normal(N_paths)
    S_T = S0 * np.exp((mu - 0.5 * sig**2) * T + sig * np.sqrt(T) * Z)

    payoffs = (S_T > K).astype(float)

    p_hat = payoffs.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / N_paths)
    CI_lower, CI_upper = (p_hat - 1.96 * se, p_hat + 1.96 * se)

    return {
        'probability': p_hat,
        'std_error': se,
        'ci_95': (CI_lower, CI_upper),
        'N_paths': N_paths,
    }


def simulate_up_prob(
    start_price: float,
    current_price: float,
    time_left_sec: float,
    vol: float,
    mu: float = 0.0,
    n_paths: int = 1000,
) -> float:
    """
    Estimate the probability that price ends >= start_price via GBM Monte Carlo.

    start_price:   reference level — paths finishing at or above this count as "up"
    current_price: S_0 for the simulation
    time_left_sec: time horizon in seconds (converted to years internally)
    vol:           annualised volatility (σ)
    mu:            annualised drift (µ), default 0
    n_paths:       number of simulated paths (>= 1000 recommended)

    Uses S_T = S_0 * exp((µ - 0.5 σ²) t + σ √t Z), Z ~ N(0,1).
    Returns fraction of paths where S_T >= start_price.
    """
    T = time_left_sec / (365.25 * 24 * 3600)  # seconds → years
    Z = np.random.standard_normal(n_paths)
    S_T = current_price * np.exp((mu - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
    return float((S_T >= start_price).mean())