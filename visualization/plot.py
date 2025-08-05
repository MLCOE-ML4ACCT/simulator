
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from visualization.financial_ratios import FinancialRatioCalculator

def plot_simulation_results(timestamp, realr):
    """Plots the financial ratios and balance sheet composition over time."""
    output_dir = Path(__file__).parent.parent / "data" / "simulation_outputs" / timestamp
    csv_files = sorted(output_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {output_dir}")
        return

    yearly_mean_ratios = {
        "CR": [], "DR": [], "DER": [], "ECR": [], "FQ": [], "ICR": [],
        "ROE": [], "ROI": [], "EFFTAX": [], "RROI": [], "ER": []
    }
    yearly_composition = {
        "Asset_CA": [], "Asset_MA": [], "Asset_BU": [], "Asset_OFA": [],
        "Liability_CL": [], "Liability_LL": [], "Equity_EC": [], "Untaxed_Reserves_UR": []
    }
    years = []
    all_data = []

    for csv_file in csv_files:
        year = int(csv_file.stem)
        years.append(year)

        df = pd.read_csv(csv_file)
        df['year'] = year
        all_data.append(df)

        result_tensors = {col: tf.constant(df[col].values, dtype=tf.float32) for col in df.columns}

        ratio_calculator = FinancialRatioCalculator(result_tensors)
        ratios = ratio_calculator.calculate_all_ratios(realr)
        composition = ratio_calculator.calculate_composition()

        for ratio_name, ratio_values in ratios.items():
            mean_ratio = tf.reduce_mean(ratio_values).numpy()
            yearly_mean_ratios[ratio_name].append(mean_ratio)

        for comp_name, comp_values in composition.items():
            mean_comp = tf.reduce_mean(comp_values).numpy()
            yearly_composition[comp_name].append(mean_comp)

    # Plot Financial Ratios
    num_ratios = len(yearly_mean_ratios)
    fig1, axes1 = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
    fig1.suptitle(f'Mean Financial Ratios Over Time (Simulation: {timestamp})', fontsize=16)
    axes1 = axes1.flatten()

    ratio_names = list(yearly_mean_ratios.keys())

    for i, ratio_name in enumerate(ratio_names):
        ax = axes1[i]
        ax.plot(years, yearly_mean_ratios[ratio_name], marker='o')
        ax.set_title(ratio_name)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Ratio")
        ax.grid(True)

    for i in range(num_ratios, len(axes1)):
        axes1[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename_ratios = output_dir / "financial_ratios.png"
    plt.savefig(plot_filename_ratios)
    print(f"Saved financial ratio plots to {plot_filename_ratios}")
    plt.close(fig1)

    # Plot Balance Sheet Composition
    asset_composition = {k: v for k, v in yearly_composition.items() if k.startswith("Asset")}
    liability_composition = {k: v for k, v in yearly_composition.items() if not k.startswith("Asset")}

    fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    # Asset Composition
    ax2.stackplot(years, asset_composition.values(), labels=asset_composition.keys(), alpha=0.8)
    ax2.set_title('Asset Composition Over Time')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Proportion of Total Assets')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True)

    # Liability & Equity Composition
    ax3.stackplot(years, liability_composition.values(), labels=liability_composition.keys(), alpha=0.8)
    ax3.set_title('Liability & Equity Composition Over Time')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Proportion of Total Assets')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename_composition = output_dir / "balance_sheet_composition.png"
    plt.savefig(plot_filename_composition)
    print(f"Saved balance sheet composition plots to {plot_filename_composition}")
    plt.close(fig2)

    # Histograms and Summary Statistics for the final year
    final_year_df = pd.read_csv(csv_files[-1])
    final_year_tensors = {col: tf.constant(final_year_df[col].values, dtype=tf.float32) for col in final_year_df.columns}
    final_year_ratio_calculator = FinancialRatioCalculator(final_year_tensors)
    final_year_ratios = final_year_ratio_calculator.calculate_all_ratios(realr)

    key_ratios_for_dist = ["ROE", "DER", "CR"]
    fig3, axes3 = plt.subplots(nrows=1, ncols=len(key_ratios_for_dist), figsize=(15, 5))
    fig3.suptitle(f'Distribution of Key Financial Ratios (Year: {years[-1]})', fontsize=16)

    for i, ratio_name in enumerate(key_ratios_for_dist):
        ax = axes3[i]
        ax.hist(final_year_ratios[ratio_name].numpy(), bins=50, alpha=0.7)
        ax.set_title(f'Distribution of {ratio_name}')
        ax.set_xlabel(ratio_name)
        ax.set_ylabel('Frequency')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename_dist = output_dir / "key_ratio_distributions.png"
    plt.savefig(plot_filename_dist)
    print(f"Saved key ratio distribution plots to {plot_filename_dist}")
    plt.close(fig3)

    # Summary Statistics Table
    summary_stats = {}
    for ratio_name in key_ratios_for_dist:
        values = final_year_ratios[ratio_name].numpy()
        summary_stats[ratio_name] = {
            'mean': values.mean(),
            'median': pd.Series(values).median(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            '25%': pd.Series(values).quantile(0.25),
            '75%': pd.Series(values).quantile(0.75)
        }

    summary_df = pd.DataFrame(summary_stats).T
    print("\nSummary Statistics for Key Financial Ratios (Final Year):")
    print(summary_df)
    summary_df.to_csv(output_dir / "summary_statistics.csv")
    print(f"Saved summary statistics to {output_dir / 'summary_statistics.csv'}")

    # Time-series scatter plots for key variables
    all_data_df = pd.concat(all_data, ignore_index=True)
    key_variables_for_scatter = ["CAt", "MAt", "BUt", "OFAt", "CLt", "LLt", "UREt", "OIBDt", "NBIt", "IMAt", "IBUt", "DIVt"]
    num_key_vars = len(key_variables_for_scatter)
    fig4, axes4 = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
    fig4.suptitle(f'Time-Series Scatter Plots of Key Variables (Simulation: {timestamp})', fontsize=16)
    axes4 = axes4.flatten()

    for i, var_name in enumerate(key_variables_for_scatter):
        ax = axes4[i]
        ax.scatter(all_data_df['year'], all_data_df[var_name], alpha=0.1)
        ax.set_title(f'Time-Series Scatter of {var_name}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.grid(True)

    for i in range(num_key_vars, len(axes4)):
        axes4[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename_scatter = output_dir / "key_variable_time_series_scatters.png"
    plt.savefig(plot_filename_scatter)
    print(f"Saved key variable time-series scatter plots to {plot_filename_scatter}")
    plt.close(fig4)
