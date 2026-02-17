#!/usr/bin/env python3
"""
Demo script for producing pulse-compressed (focused) radargrams.

This script loads REASON partially processed data and generates radargrams
with full processing: pulse compression, delay alignment, and geometric correction.

Processing steps:
1. Generate reference chirp from engineering data
2. Apply pulse compression via matched filter
3. Stack pulses within dwells for noise reduction
4. Align records by varying delays between dwells
5. Apply geometric correction using spacecraft altitude
6. Visualize as amplitude in dB
"""

import sys
sys.path.insert(0, 'src')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from reason_pds_review import (
    load_ppdp,
    generate_chirp,
    pulse_compress,
    apply_stacking,
    align_by_delay,
    geometric_correction,
    calculate_amplitude_db
)

# Configuration
data_dir = "../urn-nasa-pds-clipper.rea.partiallyprocessed/DATA/000MGA/2025060T1736"
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Stacking configuration (coherent stacking after pulse compression)
STACKING = {
    'HF': 10,
    'VHF_POSX': 20,
    'VHF_FULL': 20,
    'VHF_NEGX': 20
}

# Channel ordering for subplot layout (left to right, top to bottom)
CHANNEL_ORDER = ['HF', 'VHF_POSX', 'VHF_FULL', 'VHF_NEGX']


def process_channel(channel_name, science_ds, eng_ds, med_ds, sample_rate, stack_factor):
    """
    Process a single channel through all compression and alignment steps.

    Processing is done separately for each dwell as chirp parameters may vary.

    Parameters
    ----------
    channel_name : str
        Channel identifier (e.g., 'HF')
    science_ds : xr.Dataset
        Science dataset containing complex I/Q data
    eng_ds : xr.Dataset
        Engineering dataset with chirp and timing parameters
    med_ds : xr.Dataset
        Mission engineering dataset with altitude and position data
    sample_rate : float
        Sample rate in Hz
    stack_factor : int
        Number of pulses to stack (coherent averaging)

    Returns
    -------
    np.ndarray
        Processed radargram data (amplitude in dB)
    """
    print(f"\nProcessing {channel_name}...")

    # Get complex data
    complex_data = science_ds['complex'].values
    print(f"  Input shape: {complex_data.shape}")

    # Get dwell IDs to process each dwell separately
    dwell_ids = eng_ds['Dwell_ID'].values
    unique_dwells = np.unique(dwell_ids)
    print(f"  Found {len(unique_dwells)} dwells")

    # Pass 1: Pulse compress and stack each dwell, collect delay values
    dwell_compressed = []
    dwell_eng_stacked = []

    for dwell_id in unique_dwells:
        # Get indices for this dwell
        dwell_mask = dwell_ids == dwell_id
        dwell_indices = np.where(dwell_mask)[0]
        n_pulses = len(dwell_indices)

        # Extract dwell data
        dwell_data = complex_data[dwell_indices, :]

        # Step 1: Generate reference chirp for this dwell
        chirp_start = eng_ds['Chirp_start_frequency'].values[dwell_indices][0]
        chirp_end = eng_ds['Chirp_end_frequency'].values[dwell_indices][0]
        chirp_length = int(eng_ds['Chirp_length_ticks'].values[dwell_indices][0])

        # Confirm chirp parameters are constant within dwell
        assert np.all(eng_ds['Chirp_start_frequency'].values[dwell_indices] == chirp_start), \
            "Chirp start frequency varies within dwell"
        assert np.all(eng_ds['Chirp_end_frequency'].values[dwell_indices] == chirp_end), \
            "Chirp end frequency varies within dwell"
        assert np.all(eng_ds['Chirp_length_ticks'].values[dwell_indices] == chirp_length), \
            "Chirp length varies within dwell"

        chirp = generate_chirp(
            chirp_start_freq=chirp_start,
            chirp_end_freq=chirp_end,
            chirp_length_ticks=chirp_length,
            sample_rate=sample_rate,
            window='hann'
        )

        # Step 2: Pulse compression
        compressed = pulse_compress(dwell_data, chirp, axis=1)

        # Step 3: Coherent stacking within dwell
        if stack_factor > 1 and n_pulses >= stack_factor:
            stacked = apply_stacking(compressed, stack_factor=stack_factor, axis=0)

            # Also stack engineering parameters for alignment
            eng_stacked = {}
            for key in ['HW_RX_opening_ticks', 'TX_start_ticks', 'Chirp_length_ticks',
                        'RX_window_length_ticks', 'Raw_active_mode_length']:
                if key in eng_ds.data_vars:
                    # Take first value of each stack window
                    dwell_eng = eng_ds[key].values[dwell_indices]
                    eng_stacked[key] = dwell_eng[::stack_factor][:stacked.shape[0]]
        else:
            stacked = compressed
            eng_stacked = {}
            for key in ['HW_RX_opening_ticks', 'TX_start_ticks', 'Chirp_length_ticks',
                        'RX_window_length_ticks', 'Raw_active_mode_length']:
                if key in eng_ds.data_vars:
                    eng_stacked[key] = eng_ds[key].values[dwell_indices]

        dwell_compressed.append(stacked)
        dwell_eng_stacked.append(eng_stacked)

    # Compute global reference delay across all dwells
    all_delay_samples = []
    for eng_stacked in dwell_eng_stacked:
        hw_rx = eng_stacked['HW_RX_opening_ticks']
        tx_start = eng_stacked['TX_start_ticks']
        rx_win = eng_stacked['RX_window_length_ticks']
        raw_active = eng_stacked['Raw_active_mode_length']
        sr = raw_active / rx_win
        delay = (hw_rx - tx_start) * sr
        all_delay_samples.append(delay)
    global_ref_delay = np.concatenate(all_delay_samples)[0]

    # Pass 2: Apply delay alignment with global reference
    dwell_results = []
    for stacked, eng_stacked in zip(dwell_compressed, dwell_eng_stacked):
        aligned = align_by_delay(
            data=stacked,
            hw_rx_opening_ticks=eng_stacked['HW_RX_opening_ticks'],
            tx_start_ticks=eng_stacked['TX_start_ticks'],
            chirp_length_ticks=eng_stacked['Chirp_length_ticks'],
            rx_window_length_ticks=eng_stacked['RX_window_length_ticks'],
            raw_active_mode_length=eng_stacked['Raw_active_mode_length'],
            axis=0,
            reference_delay_samples=global_ref_delay,
        )

        dwell_results.append(aligned)

    # Concatenate all dwells
    print(f"  Concatenating {len(dwell_results)} dwells...")
    all_data = np.concatenate(dwell_results, axis=0)
    print(f"  Final shape after concatenation: {all_data.shape}")

    # Step 5: Geometric correction (align to reference altitude)
    # Shift all records to a common reference altitude using MED data
    if med_ds is not None and 'SC_altitude_above_target_ellipsoid' in med_ds.data_vars:
        print(f"  Applying geometric correction...")

        # Get altitude data (in meters)
        altitude_m = med_ds['SC_altitude_above_target_ellipsoid'].values

        # Stack altitude to match stacked data if stacking was applied
        if stack_factor > 1:
            # Take first value of each stack window to match stacked data
            altitude_stacked = altitude_m[::stack_factor][:all_data.shape[0]]
        else:
            altitude_stacked = altitude_m[:all_data.shape[0]]

        # Convert to km
        altitude_km = altitude_stacked / 1000.0

        print(f"    Altitude range: {altitude_km.min():.1f} to {altitude_km.max():.1f} km")
        

        # Apply geometric correction using the library function
        geometrically_corrected = geometric_correction(
            data=all_data,
            altitude_km=altitude_km,
            sample_rate=sample_rate,
            axis=0
        )

        # # Reference altitude for alignment
        # # This aligns the top of the radargram to 3398 km altitude
        # reference_altitude_km = 3398.0
        # print(f"    Reference altitude: {reference_altitude_km:.1f} km")
        #
        # # Now shift to reference altitude
        # # Calculate additional shift needed to align to reference altitude
        # mean_altitude = altitude_km.mean()
        # altitude_offset_km = reference_altitude_km - mean_altitude

        # # Convert altitude offset to samples (two-way distance)
        # c = 299792458  # Speed of light in m/s
        # altitude_offset_m = altitude_offset_km * 1000
        # range_offset_samples = (2 * altitude_offset_m) / (c / sample_rate)
        # roll_amount = int(np.round(range_offset_samples))

        # print(f"    Shifting to reference altitude: {roll_amount} samples")

        # # Apply reference altitude shift to all records
        # if roll_amount != 0:
        #     geometrically_corrected = np.roll(geometrically_corrected, -roll_amount, axis=1)

        print(f"    Geometric correction complete")
    else:
        print(f"  Geometric correction skipped")
        geometrically_corrected = all_data

    # Step 6: Convert to amplitude in dB for visualization
    print(f"  Converting to amplitude (dB)...")
    amplitude_db = calculate_amplitude_db(geometrically_corrected)

    return amplitude_db


def main():
    print("="*60)
    print("REASON Pulse-Compressed Radargram Demo")
    print("="*60)
    print("\nProcessing steps:")
    print("  1. Process each dwell separately (chirp params vary by dwell)")
    print("  2. Pulse compression (matched filter per dwell)")
    print("  3. Coherent stacking within each dwell")
    print("  4. Delay alignment within each dwell")
    print("  5. Concatenate all dwells")
    print("  6. Geometric correction (optional)")
    print("  7. Amplitude visualization")

    # Load data
    print(f"\nLoading data from {data_dir}...")
    tree = load_ppdp(data_dir)

    # Sample rates for each channel type
    SAMPLE_RATES = {
        'HF': 1.2e6,      # 1.2 MHz
        'VHF_POSX': 12e6, # 12 MHz
        'VHF_FULL': 12e6, # 12 MHz
        'VHF_NEGX': 12e6  # 12 MHz
    }

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

        # Get science and engineering data
        science = tree[f'{channel_name}/science'].ds

        # Check if engineering data exists
        channel_node = tree[channel_name]
        if 'engineering' in channel_node.children:
            engineering = tree[f'{channel_name}/engineering'].ds
        else:
            print(f"Warning: No engineering data for {channel_name}, skipping")
            continue

        # Get MED data if available
        med = None
        if 'med' in channel_node.children:
            med = tree[f'{channel_name}/med'].ds

        # Process channel
        sample_rate = SAMPLE_RATES[channel_name]
        stack_factor = STACKING.get(channel_name, 1)

        try:
            amplitude_db = process_channel(
                channel_name, science, engineering, med, sample_rate, stack_factor
            )
            channel_data[channel_name] = {
                'amplitude_db': amplitude_db,
                'science': science,
                'stack_factor': stack_factor
            }

            # Calculate fast time range for this channel
            fast_time_max = float(science.coords['fast_time'].max())
            max_fast_time_us = max(max_fast_time_us, fast_time_max)

        except Exception as e:
            print(f"Error processing {channel_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nShared y-axis range: 0 to {max_fast_time_us:.2f} μs")

    # Plot each channel
    for idx, channel_name in enumerate(CHANNEL_ORDER):
        if channel_name not in channel_data:
            continue

        data = channel_data[channel_name]
        amplitude_db = data['amplitude_db']
        science = data['science']
        stack_factor = data['stack_factor']
        ax = axes[idx]

        # Filter valid data for percentile calculation
        valid_data = amplitude_db[amplitude_db > -100]

        if len(valid_data) == 0:
            print(f"Warning: No valid data for {channel_name}")
            continue

        # Plot
        im = ax.imshow(amplitude_db.T,
                       aspect='auto',
                       cmap='gray',
                       vmin=np.percentile(valid_data, 1),
                       vmax=np.percentile(valid_data, 99),
                       extent=[0, amplitude_db.shape[0],
                               science.coords['fast_time'].max(),
                               science.coords['fast_time'].min()])

        # Set y-axis limits for all plots
        ax.set_ylim(max_fast_time_us, 0)

        # Labels and title
        ax.set_xlabel('Slow Time (Pulse Number)', fontsize=10)
        ax.set_ylabel('Fast Time (μs)', fontsize=10)

        description = science.attrs.get('description', '')
        frequency = science.attrs.get('frequency', '')
        title_parts = [f"{channel_name} ({frequency})"]
        if description:
            title_parts.append(description)
        if stack_factor > 1:
            title_parts.append(f'{stack_factor}x stacked')
        title_parts.append('Pulse Compressed')

        ax.set_title('\n'.join([title_parts[0], ', '.join(title_parts[1:])]),
                     fontsize=11, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Amplitude (dB)', pad=0.02)

        # Add statistics text
        stats_text = f"Range: [{valid_data.min():.1f}, {valid_data.max():.1f}] dB\n"
        stats_text += f"Pulses: {amplitude_db.shape[0]:,}, Samples: {amplitude_db.shape[1]:,}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        print(f"\n{channel_name} processed successfully:")
        print(f"  Final shape: {amplitude_db.shape}")
        print(f"  Amplitude range: [{valid_data.min():.1f}, {valid_data.max():.1f}] dB")

    # Overall title
    fig.suptitle('REASON Pulse-Compressed Radargrams - Mars Gravity Assist\n' +
                 'Full Processing: Matched Filter + Alignment',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "radargrams.png"
    plt.savefig(output_file, dpi=75, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"✓ Pulse-compressed radargrams saved to: {output_file.absolute()}")
    print(f"{'='*60}")

    print("\nDemo completed successfully!")


if __name__ == '__main__':
    main()
