#!/usr/bin/env python3
"""
Demo script for producing unfocused radargrams using the reason_pds_review library.

This script loads REASON partially processed data and generates radargrams
without pulse compression (unfocused/raw radargrams).
"""

import sys

import demo_radargrams

sys.path.insert(0, 'src')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from reason_pds_review import load_ppdp, apply_stacking, calculate_amplitude_db


# Configuration
data_dir = demo_radargrams.data_dir
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Stacking configuration (matches plot_all_channels.py)
STACKING = {
    'HF': 10,
    'VHF_POSX': 20,
    'VHF_FULL': 20,
    'VHF_NEGX': 20
}

# Channel ordering for subplot layout (left to right, top to bottom)
CHANNEL_ORDER = ['HF', 'VHF_POSX', 'VHF_FULL', 'VHF_NEGX']


def main():
    print("="*60)
    print("REASON Unfocused Radargram Demo")
    print("="*60)

    # Load data
    print(f"\nLoading data from {data_dir}...")
    tree = load_ppdp(data_dir)

    # Create 2x2 subplot figure with shared y-axis
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    axes = axes.flatten()

    # Calculate maximum fast time for shared y-axis
    max_fast_time_us = 0
    channel_data = {}

    for channel_name in CHANNEL_ORDER:
        if channel_name not in tree.children:
            print(f"Warning: {channel_name} not found in data")
            continue

        # Get science data
        science = tree[f'{channel_name}/science'].ds
        channel_data[channel_name] = science

        # Calculate fast time range for this channel
        fast_time_max = float(science.coords['fast_time'].max())
        max_fast_time_us = max(max_fast_time_us, fast_time_max)

    print(f"\nShared y-axis range: 0 to {max_fast_time_us:.2f} μs")

    # Plot each channel
    for idx, channel_name in enumerate(CHANNEL_ORDER):
        if channel_name not in channel_data:
            continue

        print(f"\nProcessing {channel_name}...")
        science = channel_data[channel_name]
        ax = axes[idx]

        # Get complex data
        complex_data = science['complex'].values

        # Apply stacking
        stack_factor = STACKING.get(channel_name, 1)
        if stack_factor > 1:
            print(f"  Applying {stack_factor}x stacking...")
            complex_data = apply_stacking(complex_data, stack_factor)
            n_slow_display = complex_data.shape[0]
        else:
            n_slow_display = complex_data.shape[0]

        # Calculate amplitude in dB
        amplitude_db = calculate_amplitude_db(complex_data)

        # Filter valid data for percentile calculation
        valid_data = amplitude_db[amplitude_db > -100]

        # Plot
        im = ax.imshow(amplitude_db.T,
                       aspect='auto',
                       cmap='gray',
                       vmin=np.percentile(valid_data, 1),
                       vmax=np.percentile(valid_data, 99),
                       extent=[0, n_slow_display,
                               science.coords['fast_time'].max(),
                               science.coords['fast_time'].min()])

        # Labels and title
        ax.set_xlabel('Slow Time (Pulse Number)', fontsize=10)
        ax.set_ylabel('Fast Time (μs)', fontsize=10)
        ax.set_ylim(max_fast_time_us, 0)

        description = science.attrs.get('description', '')
        frequency = science.attrs.get('frequency', '')
        if stack_factor > 1:
            description += f', {stack_factor}x stacked'

        ax.set_title(f"{channel_name} ({frequency})\n{description}",
                     fontsize=11, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Amplitude (dB)', pad=0.02)

        # Add statistics text
        stats_text = f"Range: [{valid_data.min():.1f}, {valid_data.max():.1f}] dB\n"
        stats_text += f"Pulses: {n_slow_display:,}, Samples: {science.sizes['fast_time']:,}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        print(f"  Amplitude range: [{valid_data.min():.1f}, {valid_data.max():.1f}] dB")

    # Overall title
    fig.suptitle('REASON Uncompressed Radargrams - Mars Gravity Assist\n' +
                 'Partially Processed Data (No Pulse Compression)',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "uncompressed_radargrams.png"
    plt.savefig(output_file, dpi=75, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"✓ Uncompressed radargrams saved to: {output_file.absolute()}")
    print(f"{'='*60}")

    print("\nDemo completed successfully!")


if __name__ == '__main__':
    main()
