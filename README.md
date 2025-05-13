# Basic-Monte-Carlo-Simulation
# Trying to use monte carlo simulation with brownian motion stock prices to construct a 3 stock portfolio of META NVDA and the Index SOXL with a caveat of also having long option calls on the VIX to interact with tail risk (IE market blowing up during portfolio construction) did not use mean variance optimization either, very simple and basic.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def monte_carlo_simulation(
    ticker: str,
    last_price: float,
    mu: float,
    sigma: float,
    sim_days: int = 15,
    n_simulations: int = 10000
):
    """
    Monte Carlo simulation for projected stock price.

    Args:
        ticker (str): Stock or ETF symbol (for labeling only)
        last_price (float): Most recent closing price
        mu (float): Daily mean return (e.g. 0.0008)
        sigma (float): Daily volatility (e.g. 0.025)
        sim_days (int): Trading days to simulate
        n_simulations (int): Number of simulations

    Returns:
        dict: Summary statistics (expected price, std dev, CI)
    """

    # Geometric Monte Carlo, using vix to simulate minor tail risk. Like the VIX hitting near 20 nothing over
    simulated_prices = np.zeros(n_simulations)
    for i in range(n_simulations):
        price = last_price
        for _ in range(sim_days):
            rand = np.random.normal()
            price *= np.exp((mu - 0.5 * sigma ** 2) + sigma * rand)
        simulated_prices[i] = price

    # Summary statistics
    expected_price = np.mean(simulated_prices)
    std_dev = np.std(simulated_prices)
    ci_95 = (np.percentile(simulated_prices, 2.5), np.percentile(simulated_prices, 97.5))

    # Plot results
    plt.hist(simulated_prices, bins=50, alpha=0.7)
    plt.title(f'{ticker} Price Simulation - {sim_days} Trading Days')
    plt.xlabel('Simulated Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Return results as dictionary
    return {
        'Ticker': ticker,
        'Last Price': last_price,
        'Expected Price': expected_price,
        'Standard Deviation': std_dev,
        '95% CI Low': ci_95[0],
        '95% CI High': ci_95[1]
    }

# ---- Tickers I used ----

# META
meta_results = monte_carlo_simulation(
    ticker="META",
    last_price=656.03,
    mu=0.0008,
    sigma=0.025,
    sim_days=15
)

# NVDA
nvda_results = monte_carlo_simulation(
    ticker="NVDA",
    last_price=129.93,
    mu=0.0008,
    sigma=0.025,
    sim_days=15
)

# VIX
vix_results = monte_carlo_simulation(
    ticker="VIX",
    last_price=18.22,
    mu=0.0008,
    sigma=0.025,
    sim_days=15
)

# SOXL (3x leveraged ETF)
soxl_results = monte_carlo_simulation(
    ticker="SOXL",
    last_price=18.60,
    mu=0.0015,
    sigma=0.045,
    sim_days=15
)

df = pd.DataFrame([meta_results, nvda_results, vix_results, soxl_results])
print(df)
