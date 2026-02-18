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
from pathlib import Path
import sys
from bdb import BdbQuit

#import matplotlib
# Set the backend to QtAgg (must be before importing pyplot)
#matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt

import numpy as np

sys.path.insert(0, 'src')

from reason_pds_review import (
    load_ppdp,
    generate_chirp,
    pulse_compress, pulse_compress_match_fft,
    apply_stacking,
    align_by_delay,
    geometric_correction,
    calculate_amplitude_db,
    roll_radargram,
    roll_radargram2,
)

# Configuration
#data_dir = "../urn-nasa-pds-clipper.rea.partiallyprocessed/DATA/000MGA/2025060T1736"
#data_dir = '/disk/kea/SDS/targ/xtra/REASON/2025_PDS4_Review/20251003_draft5/PDS/bundle_pp/DATA/000MGA/2025060T1736'
#data_dir = '/disk/kea/SDS/targ/xtra/REASON/2025_PDS4_Review/20251001_draft4/PDS/bundle_pp/DATA/000MGA/2025060T1736'
#data_dir = '/disk/kea/SDS/targ/xtra/REASON/2025_PDS4_Review/20250910_draft3/PDS/bundle_pp/DATA/000MGA/2025060T1736'
#data_dir = '/disk/kea/SDS/targ/xtra/REASON/2025_PDS4_Review/20250724_draft1/PDS/PARTIALLYPROCESSED/000M01'
#data_dir = '/disk/kea/SDS/targ/xtra/REASON/2025_PDS4_Review/20250805_draft2/PDS/PARTIALLYPROCESSED/000XXX'
data_dir = '/disk/kea/SDS/code/work/ngg/202507_sds2pds4/SDS2-PDS4/tests/out1_tt/PDS/bundle_pp/DATA/000MGA/2025060T1736'
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
    unique_dwells = sorted(set(dwell_ids))
    #assert min(unique_dwells) > 600
    #assert max(unique_dwells) < 630
    print(f"  Found {len(unique_dwells)} dwells")

    # Process each dwell separately
    dwell_results = []
    dly_roll_amounts = []
    dwell_eng_vals = []
    eng_keys = []
    gc_roll_amounts = []

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

        chirp, chirp_win = generate_chirp(
            chirp_start_freq=chirp_start,
            chirp_end_freq=chirp_end,
            chirp_length_ticks=chirp_length,
            sample_rate=sample_rate,
            window='hamming' # changed from hann to hamming to match day code
        )

        # Step 2: Pulse compression
        compressed = pulse_compress(dwell_data, chirp, axis=1)
        # Step 3: Coherent stacking within dwell
        eng_keys = ['HW_RX_opening_ticks', 'TX_start_ticks', 'Chirp_length_ticks',
                    'RX_window_length_ticks', 'Raw_active_mode_length', 'RX_delay_tracking_offset']
        if stack_factor > 1: #   and n_pulses >= stack_factor:
            stacked = apply_stacking(compressed, stack_factor=stack_factor, axis=0)

            # Also stack engineering parameters for alignment

            eng_stacked = {}
            for key in eng_keys:
                if key in eng_ds.data_vars:
                    # Take first value of each stack window
                    dwell_eng = eng_ds[key].values[dwell_indices]
                    eng_stacked[key] = dwell_eng[::stack_factor][:stacked.shape[0]]
                    assert len(eng_stacked[key]) == len(stacked), "mismatch in eng dataset %s dwell %r" % (key, dwell_id)
            # Stack med data
            if med_ds is not None:
                med_stacked = {}
                for key in med_ds.data_vars:
                    dwell_med = med_ds[key].values[dwell_indices]
                    med_stacked[key] = dwell_med[::stack_factor][:stacked.shape[0]]
                    assert len(med_stacked[key]) == len(stacked), "mismatch in med dataset %s dwell %r" % (key, dwell_id)
        else:
            stacked = compressed
            eng_stacked = {}
            for key in eng_keys:
                if key in eng_ds.data_vars:
                    eng_stacked[key] = eng_ds[key].values[dwell_indices]
                    assert len(eng_stacked[key]) == len(stacked), "mismatch in eng dataset %s dwell %r" % (key, dwell_id)
            med_stacked = {}
            if med_ds is not None:
                for key in med_ds.data_vars:
                    med_stacked[key] = med_ds[key].values[dwell_indices]
                    assert len(med_stacked[key]) == len(stacked), "mismatch in med dataset %s dwell %r" % (key, dwell_id)

        # Step 4: Align records within dwell by delay
        # TODO: I'm still missing some correction factor. See notes in align_by_delay
        
        aligned, dly_roll_amount = align_by_delay(
            data=stacked,
            hw_rx_opening_ticks=eng_stacked['HW_RX_opening_ticks'],
            tx_start_ticks=eng_stacked['TX_start_ticks'],
            chirp_length_ticks=eng_stacked['Chirp_length_ticks'],
            rx_window_length_ticks=eng_stacked['RX_window_length_ticks'],
            raw_active_mode_length=eng_stacked['Raw_active_mode_length'],
            rx_delay_tracking_offset=eng_stacked['RX_delay_tracking_offset'],
            axis=0
        )

        # Step 5: Geometric delay correction

        # Get altitude data (in meters)
        #altitude_m = med_stacked['SC_altitude_above_target_ellipsoid']

        # Leopold's calculation using fixed target body radius instead of ellipsoid
        ref_altitude = 3398000.
        altitude_m = med_stacked['SC_distance_from_target_center'] - ref_altitude

        # Convert to km
        altitude_km = altitude_m / 1000.0

        gc_roll_amount = geometric_correction(
            data=stacked,
            altitude_km=altitude_km,
            sample_rate=sample_rate,
            axis=0
        )


        #dwell_results.append(aligned)
        dwell_results.append(stacked)
        dly_roll_amounts.append(dly_roll_amount)
        #dly_roll_amounts[-1] += 50 # for testing. shift by 50 samples 2026-02-11 meeting
        dwell_eng_vals.append(eng_stacked)
        gc_roll_amounts.append(gc_roll_amount)


    # Concatenate all dwells
    print(f"  Concatenating {len(dwell_results)} dwells...")
    all_data = np.concatenate(dwell_results, axis=0)
    arr_dly_roll_amounts = np.concatenate(dly_roll_amounts, axis=0)
    assert len(all_data) == len(arr_dly_roll_amounts)
    print(f"  Final shape after concatenation: {all_data.shape}")

    # Step 5: Geometric correction (align to reference altitude)
    arr_gc_roll_amounts = np.concatenate(gc_roll_amounts, axis=0)

    # Roll correction for delay and geometric together
    roll_amounts = (-arr_dly_roll_amounts + arr_gc_roll_amounts) * -1.0
    # just do a relative shift
    roll_amounts1 = roll_amounts - np.mean(roll_amounts)
    #geometrically_corrected = roll_radargram(data=all_data, roll_amounts=roll_amounts1, axis=0)
    geometrically_corrected = roll_radargram2(data=all_data, roll_amounts=roll_amounts1)


    # Step 6: Convert to amplitude in dB for visualization
    print(f"  Converting to amplitude (dB)...")
    amplitude_db = calculate_amplitude_db(geometrically_corrected)

    # Step 7: calculate roll amounts for additional visualization
    eng = {k: np.concatenate([d[k] for d in dwell_eng_vals])
    for k in eng_keys}
    
    return amplitude_db, (arr_dly_roll_amounts, arr_gc_roll_amounts, roll_amounts, eng)


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
            amplitude_db, (dly_roll_amounts, gc_roll_amounts, roll_amounts, eng) = process_channel(
                channel_name, science, engineering, med, sample_rate, stack_factor
            )
            channel_data[channel_name] = {
                'amplitude_db': amplitude_db,
                'science': science,
                'stack_factor': stack_factor,
                'dly_roll_amounts': dly_roll_amounts,
                'gc_roll_amounts': gc_roll_amounts,
                'total_roll_amounts': roll_amounts,
                'eng': eng,
            }

            # Calculate fast time range for this channel
            fast_time_max = float(science.coords['fast_time'].max())
            max_fast_time_us = max(max_fast_time_us, fast_time_max)
        except (BdbQuit, KeyboardInterrupt):
            raise
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

    # Save figures
    output_file = output_dir / "radargrams.png"
    plt.savefig(output_file, dpi=75, bbox_inches='tight')

    print(f"\n{'='*60}")
    print(f"✓ Pulse-compressed radargrams saved to: {output_file.absolute()}")
    print(f"{'='*60}")
    plt.close(fig)


    # Create 2x2 subplot figure with shared y-axis for roll
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    axes2 = axes2.flatten()

    # Plot each channel's roll components
    for idx, channel_name in enumerate(CHANNEL_ORDER):
        if channel_name not in channel_data:
            continue

        data = channel_data[channel_name]
        #amplitude_db = data['amplitude_db']
        science = data['science']
        stack_factor = data['stack_factor']
        ax = axes2[idx]

        # Filter valid data for percentile calculation
        #valid_data = amplitude_db[amplitude_db > -100]
        sample_rate = SAMPLE_RATES[channel_name] / 1e6

        # Plot
        ax.plot(data['dly_roll_amounts'] / sample_rate, label='Delay')
        ax.plot(data['gc_roll_amounts'] / sample_rate, label='Geom')
        #ax.plot(data['total_roll_amounts'] / sample_rate, label='Total')
        ax.legend()
        

        ## Set y-axis limits for all plots
        #ax.set_ylim(max_fast_time_us, 0)

        # Labels and title
        ax.set_xlabel('Slow Time (Pulse Number)', fontsize=10)
        ax.set_ylabel('Fast Time (μs)', fontsize=10)
        ax.grid(True)

        description = science.attrs.get('description', '')
        frequency = science.attrs.get('frequency', '')
        title_parts = [f"{channel_name} ({frequency})"]
        if description:
            title_parts.append(description)
        if stack_factor > 1:
            title_parts.append(f'{stack_factor}x stacked')
        title_parts.append('Roll Amounts')

        ax.set_title('\n'.join([title_parts[0], ', '.join(title_parts[1:])]),
                     fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / "radargrams_roll.png"
    plt.savefig(output_file, dpi=75, bbox_inches='tight')
    plt.close(fig2)


    # Create 2x2 subplot figure with shared y-axis for roll
    fig3, axes3 = plt.subplots(2, 2, figsize=(18, 12), sharey=False)
    axes3 = axes3.flatten()
    ax_lims = {
        'HF': {'x': (10915 // 2 - 500, 10915 // 2 + 500), 'y': (0, 16), 'y2': (0, 140)}
        #'VHF_POSX': 20,
        #'VHF_FULL': 20,
        #'VHF_NEGX': 20
    }
    # Plot each channel's roll total
    for idx, channel_name in enumerate(CHANNEL_ORDER):
        if channel_name not in channel_data:
            continue

        data = channel_data[channel_name]
        #amplitude_db = data['amplitude_db']
        science = data['science']
        stack_factor = data['stack_factor']
        ax = axes3[idx]

        # Filter valid data for percentile calculation
        #valid_data = amplitude_db[amplitude_db > -100]
        sample_rate = SAMPLE_RATES[channel_name] / 1e6
        tick_rate = 48e6 / 1e6
        #breakpoint()
        # Plot
        ax2 = ax.twinx()
        total_roll = data['total_roll_amounts'] / sample_rate
        total_roll = total_roll - np.min(total_roll)
        ax.plot(total_roll, color='purple', label='Total (zeroed)')
        rxtx = (data['eng']['HW_RX_opening_ticks'] - data['eng']['TX_start_ticks']) / tick_rate
        rxtx = rxtx - np.min(rxtx)
        ax2.plot(rxtx, color='red', label='rxtx (zeroed)')
        geom = data['gc_roll_amounts'] / sample_rate
        geom = geom - np.min(geom)
        ax2.plot(geom, label='Geom (zeroed)')
        plot_min_line(ax2, geom)
        #ax.plot(data['eng']['Chirp_length_ticks'] / tick_rate, label='Chirp_length (not used)')
        ax.plot(data['eng']['RX_delay_tracking_offset'] / 32. / tick_rate, label='RX_delay_tracking')
        ax.grid(True)
        ax.legend()
        ax2.legend()
        

        ## Set y-axis limits for all plots
        #ax.set_ylim(max_fast_time_us, 0)

        # Labels and title
        ax.set_xlabel('Slow Time (Pulse Number)', fontsize=10)
        ax.set_ylabel('Fast Time (μs)', fontsize=10)
        ax2.set_ylabel('rxtx/geom - Fast Time (μs)', fontsize=10)

        ax.grid(True)

        description = science.attrs.get('description', '')
        frequency = science.attrs.get('frequency', '')
        title_parts = [f"{channel_name} ({frequency})"]
        if description:
            title_parts.append(description)
        if stack_factor > 1:
            title_parts.append(f'{stack_factor}x stacked')
        title_parts.append('Roll Amounts')

        ax.set_title('\n'.join([title_parts[0], ', '.join(title_parts[1:])]),
                     fontsize=11, fontweight='bold')

        ''' only needed for full flight
        ax.set_xlim(ax_lims[channel_name]['x'])
        ax2.set_xlim(ax_lims[channel_name]['x'])
        ax.set_ylim(ax_lims[channel_name]['y'])
        ax2.set_ylim(ax_lims[channel_name]['y2'])
        '''

    plt.tight_layout()
    output_file = output_dir / "radargrams_roll_total.png"
    plt.savefig(output_file, dpi=75, bbox_inches='tight')
    plt.close(fig3)


    print("\nDemo completed successfully!")


def plot_min_line(ax2, geom):
    gmin = geom.min()
    gmax = geom.max()
    idx_min = np.argmin(geom)
    ax2.plot([idx_min, idx_min], [gmin, gmax],
        color='black', alpha=0.5,
        label=f'closest approach (idx={idx_min:d})')


if __name__ == '__main__':
    main()
