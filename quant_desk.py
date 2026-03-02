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