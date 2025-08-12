
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_financial_ratios(df, tax_rate=0.28):
    """
    Calculates weighted average financial ratios from a DataFrame of firm data for a single year.
    """
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    K = df['CAt'] + df['MAt'] + df['BUt'] + df['OFAt']
    total_K = K.sum()

    if total_K == 0:
        return {ratio: 0 for ratio in [
            'Current Ratio', 'Debt Ratio', 'Debt/Equity Ratio',
            'Interest Coverage Ratio', 'Return on Assets', 'Return on Equity'
        ]}

    current_ratio = df['CAt'].sum() / df['CLt'].sum() if df['CLt'].sum() != 0 else 0

    total_debt = (df['CLt'] + df['LLt'] + tax_rate * (df['ASDt'] + df['PFt_t'] + df['OURt'])).sum()
    debt_ratio = total_debt / total_K

    debt_equity_ratio = debt_ratio / (1 - debt_ratio) if (1 - debt_ratio) != 0 else 0

    op_earnings = (df['OIBDt'] - df['EDEPMAt'] - df['EDEPBUt'] + df['FIt']).sum()
    interest_expense = df['FEt'].sum()
    interest_coverage_ratio = op_earnings / interest_expense if interest_expense != 0 else 0

    earnings_for_roa = (df['OIBDt'] - df['EDEPMAt'] - df['EDEPBUt'] + df['FIt'] + df['TLt']).sum()
    return_on_assets = earnings_for_roa / total_K

    avg_debt_interest = interest_expense / total_debt if total_debt != 0 else 0
    return_on_equity = return_on_assets + (return_on_assets - avg_debt_interest) * debt_equity_ratio

    return {
        'Current Ratio': current_ratio,
        'Debt Ratio': debt_ratio,
        'Debt/Equity Ratio': debt_equity_ratio,
        'Interest Coverage Ratio': interest_coverage_ratio,
        'Return on Assets': return_on_assets,
        'Return on Equity': return_on_equity,
    }

def plot_firm_state_scatter(simulation_folder):
    """
    Generates and saves scatter plots for key firm state variables.
    """
    sim_path = Path(simulation_folder)
    all_data = []
    csv_files = sorted(sim_path.glob('*.csv'))

    if not csv_files:
        print("Scatter plot generation skipped: No CSV files found.")
        return

    for csv_file in csv_files:
        year = int(csv_file.stem)
        df = pd.read_csv(csv_file)
        df['year'] = year
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    key_variables = ['CAt', 'MAt', 'BUt', 'OFAt', 'CLt', 'LLt', 'UREt']
    
    nrows = 4
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 16), sharex=True)
    fig.suptitle('Firm State Variables Scatter Plots (All Firms)', fontsize=16)

    axes_flat = axes.flatten()

    for i, var in enumerate(key_variables):
        ax = axes_flat[i]
        ax.scatter(full_df['year'], full_df[var], alpha=0.3, s=1, edgecolor='none')
        ax.set_title(f'{var} Over Time')
        ax.set_ylabel('Value')
        ax.grid(True)

    for i in range(len(key_variables), len(axes_flat)):
        axes_flat[i].axis('off')

    fig.text(0.5, 0.04, 'Year', ha='center', va='center')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    output_path = sim_path / 'firm_state_scatter_plots.png'
    plt.savefig(output_path, dpi=300)
    print(f"Scatter plots saved to {output_path}")
    plt.close(fig)

def main(simulation_folder):
    """
    Generates and saves plots for financial ratios from simulation CSV files.
    """
    sim_path = Path(simulation_folder)
    if not sim_path.is_dir():
        print(f"Error: Directory not found at '{sim_path}'")
        return

    csv_files = sorted(sim_path.glob('*.csv'))
    if not csv_files:
        print(f"Ratio plot generation skipped: No CSV files found in '{sim_path}'")
        return

    yearly_ratios = []
    years = []

    for csv_file in csv_files:
        year = int(csv_file.stem)
        years.append(year)
        
        print(f"Processing ratios for {year}...")
        df = pd.read_csv(csv_file)
        ratios = calculate_financial_ratios(df)
        yearly_ratios.append(ratios)

    ratios_df = pd.DataFrame(yearly_ratios, index=years)

    # Plotting
    nrows = 3
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10), sharex=True)
    fig.suptitle('Financial Ratios Over Time', fontsize=16)

    axes_flat = axes.flatten()

    for i, ratio_name in enumerate(ratios_df.columns):
        ax = axes_flat[i]
        ax.plot(ratios_df.index, ratios_df[ratio_name], marker='o', linestyle='-')
        ax.set_title(ratio_name)
        ax.set_ylabel('Ratio Value')
        ax.grid(True)

    for i in range(len(ratios_df.columns), len(axes_flat)):
        axes_flat[i].axis('off')

    fig.text(0.5, 0.04, 'Year', ha='center', va='center')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    output_path = sim_path / 'financial_ratios_plot.png'
    plt.savefig(output_path, dpi=150)
    print(f"Ratio plot saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_simulation.py <path_to_simulation_folder>")
    else:
        folder = sys.argv[1]
        main(folder)
        plot_firm_state_scatter(folder)
