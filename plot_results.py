#!/usr/bin/env python3
"""
Generate comparison plots for benchmark results.

Usage:
    python plot_results.py                          # Uses my_results_updated.csv
    python plot_results.py --csv my_results.csv     # Specify CSV file
    python plot_results.py --output my_plot.png     # Specify output file
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_filter_data(csv_path):
    """Load CSV and filter for complete data."""
    df = pd.read_csv(csv_path)

    # Check which strategy columns exist
    new_strategies = ['Dense Grid (3x3)_pred', 'Dense Grid (12x12)_pred', 'VLM Think-and-Zoom_pred']
    existing = [col for col in new_strategies if col in df.columns]

    if existing:
        df_complete = df.dropna(subset=existing)
    else:
        df_complete = df

    return df_complete


def calc_mape(df, pred_col, true_col='true_count'):
    """Calculate Mean Absolute Percentage Error."""
    valid = df[true_col] > 0
    mape = np.abs(df.loc[valid, pred_col] - df.loc[valid, true_col]) / df.loc[valid, true_col] * 100
    return mape.mean()


def calc_signed_pe(df, pred_col, true_col='true_count'):
    """Calculate Signed Percentage Error (bias)."""
    valid = df[true_col] > 0
    signed_pe = (df.loc[valid, pred_col] - df.loc[valid, true_col]) / df.loc[valid, true_col] * 100
    return signed_pe.mean()


def get_strategies(df):
    """Get available strategies from dataframe columns."""
    all_strategies = {
        'Direct VLM': 'Direct VLM_pred',
        'Dense Grid (3x3)': 'Dense Grid (3x3)_pred',
        'Dense Grid (6x6)': 'Dense Grid (6x6)_pred',
        'Dense Grid (12x12)': 'Dense Grid (12x12)_pred',
        'VLM Think-and-Zoom': 'VLM Think-and-Zoom_pred',
        'SAM3 Full Image': 'SAM3 Full Image_pred'
    }

    return {name: col for name, col in all_strategies.items() if col in df.columns}


def print_statistics(df, strategies):
    """Print MAPE and bias statistics."""
    print(f"\nTotal images: {len(df)}")

    print("\n=== MAPE (Mean Absolute Percentage Error) ===")
    for name, col in strategies.items():
        mape = calc_mape(df, col)
        print(f"{name:25s}: {mape:.1f}%")

    print("\n=== Signed Percentage Error (Bias) ===")
    for name, col in strategies.items():
        bias = calc_signed_pe(df, col)
        sign = "+" if bias > 0 else ""
        direction = "overestimates" if bias > 0 else "underestimates"
        print(f"{name:25s}: {sign}{bias:.1f}% ({direction})")


def create_plot(df, strategies, output_path):
    """Create the comparison plot."""
    # Create density bins
    df = df.copy()
    df['density_bin'] = pd.cut(
        df['true_count'],
        bins=[0, 10, 25, 50, 100, 200, 500, 1000, float('inf')],
        labels=['1-10', '11-25', '26-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
    )

    # Print bin distribution
    print("\nImages per density bin:")
    print(df['density_bin'].value_counts().sort_index())

    # Colors and markers for each strategy
    colors = {
        'Direct VLM': '#1f77b4',
        'Dense Grid (3x3)': '#2ca02c',
        'Dense Grid (6x6)': '#ff7f0e',
        'Dense Grid (12x12)': '#d62728',
        'VLM Think-and-Zoom': '#9467bd',
        'SAM3 Full Image': '#8c564b'
    }

    markers = {
        'Direct VLM': 'o',
        'Dense Grid (3x3)': 's',
        'Dense Grid (6x6)': '^',
        'Dense Grid (12x12)': 'D',
        'VLM Think-and-Zoom': 'v',
        'SAM3 Full Image': 'p'
    }

    bin_labels = ['1-10', '11-25', '26-50', '51-100', '101-200', '201-500', '501-1000', '1000+']

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: MAPE vs Object Density
    ax1 = axes[0]
    for name, col in strategies.items():
        mapes = []
        valid_bins = []
        for bin_label in bin_labels:
            bin_data = df[df['density_bin'] == bin_label]
            if len(bin_data) > 0:
                valid = bin_data['true_count'] > 0
                mape = (np.abs(bin_data.loc[valid, col] - bin_data.loc[valid, 'true_count']) /
                       bin_data.loc[valid, 'true_count'] * 100).mean()
                mapes.append(mape)
                valid_bins.append(bin_label)

        if mapes:
            ax1.plot(valid_bins, mapes, marker=markers.get(name, 'o'), label=name,
                    color=colors.get(name, '#333333'), linewidth=2, markersize=8)

    ax1.set_xlabel('Object Count (Density)', fontsize=12)
    ax1.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax1.set_title(f'MAPE vs Object Density (n={len(df)})\n(Lower is Better)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Bias (Signed Percentage Error)
    ax2 = axes[1]
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axhspan(-100, 0, alpha=0.1, color='blue')
    ax2.axhspan(0, 1000, alpha=0.1, color='red')

    for name, col in strategies.items():
        biases = []
        valid_bins = []
        for bin_label in bin_labels:
            bin_data = df[df['density_bin'] == bin_label]
            if len(bin_data) > 0:
                valid = bin_data['true_count'] > 0
                bias = ((bin_data.loc[valid, col] - bin_data.loc[valid, 'true_count']) /
                       bin_data.loc[valid, 'true_count'] * 100).mean()
                biases.append(bias)
                valid_bins.append(bin_label)

        if biases:
            ax2.plot(valid_bins, biases, marker=markers.get(name, 'o'), label=name,
                    color=colors.get(name, '#333333'), linewidth=2, markersize=8)

    ax2.set_xlabel('Object Count (Density)', fontsize=12)
    ax2.set_ylabel('Mean Signed Percentage Error (%)', fontsize=12)
    ax2.set_title(f'Bias: Overestimate (+) vs Underestimate (-) (n={len(df)})\n(Closer to 0 is Better)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark comparison plots')
    parser.add_argument('--csv', type=str, default='my_results_updated.csv',
                        help='Path to CSV file with benchmark results')
    parser.add_argument('--output', type=str, default='updated_comparison.png',
                        help='Output path for the plot')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.csv}...")
    df = load_and_filter_data(args.csv)

    # Get available strategies
    strategies = get_strategies(df)
    print(f"Found {len(strategies)} strategies: {', '.join(strategies.keys())}")

    # Print statistics
    print_statistics(df, strategies)

    # Create plot
    create_plot(df, strategies, args.output)


if __name__ == "__main__":
    main()
