#!/usr/bin/env python3
"""Plot surface peak tracking vs delay and geometric predictions — one figure per channel."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from pathlib import Path
from reason_pds_review import load_ppdp, generate_chirp, pulse_compress

data_dir = "../urn-nasa-pds-clipper.rea.partiallyprocessed/DATA/000MGA/2025060T1736"
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

CHANNELS = {
    'HF':       {'fs': 1.2e6, 'stack': 10},
    'VHF_POSX': {'fs': 12e6,  'stack': 20},
    'VHF_FULL': {'fs': 12e6,  'stack': 20},
    'VHF_NEGX': {'fs': 12e6,  'stack': 20},
}
c = 299792458.0

tree = load_ppdp(data_dir)

for ch, cfg in CHANNELS.items():
    if ch not in tree.children or 'engineering' not in tree[ch].children or 'med' not in tree[ch].children:
        print(f'{ch}: data missing, skipping')
        continue

    science = tree[f'{ch}/science'].ds
    eng = tree[f'{ch}/engineering'].ds
    med = tree[f'{ch}/med'].ds
    fs = cfg['fs']
    sf = cfg['stack']

    complex_data = science['complex'].values
    dwell_ids = eng['Dwell_ID'].values
    unique_dwells = np.unique(dwell_ids)

    # Pulse compress, stack, and collect per-stacked-pulse delay/range
    chunks, all_delay, all_range, all_dwell = [], [], [], []
    all_hw_rx, all_tx_st, all_raw_active, all_rx_win = [], [], [], []

    for did in unique_dwells:
        idx = np.where(dwell_ids == did)[0]
        n_st = len(idx) // sf
        if n_st == 0:
            continue

        chirp = generate_chirp(
            eng['Chirp_start_frequency'].values[idx[0]],
            eng['Chirp_end_frequency'].values[idx[0]],
            int(eng['Chirp_length_ticks'].values[idx[0]]),
            fs, window='hann')
        compressed = pulse_compress(complex_data[idx], chirp, axis=1)
        stacked = np.mean(compressed[:n_st * sf].reshape(n_st, sf, -1), axis=1)
        chunks.append(stacked)

        si = idx[::sf][:n_st]
        hw_rx = eng['HW_RX_opening_ticks'].values[si]
        tx_st = eng['TX_start_ticks'].values[si]
        sr_t = eng['Raw_active_mode_length'].values[si] / eng['RX_window_length_ticks'].values[si]
        all_delay.append((hw_rx - tx_st) * sr_t)
        all_range.append((2 * med['SC_altitude_above_target_ellipsoid'].values[si]) / (c / fs))
        all_hw_rx.append(hw_rx)
        all_tx_st.append(tx_st)
        all_raw_active.append(eng['Raw_active_mode_length'].values[si])
        all_rx_win.append(eng['RX_window_length_ticks'].values[si])
        all_dwell.extend([did] * n_st)

    all_data = np.concatenate(chunks)
    delays = np.concatenate(all_delay)
    ranges = np.concatenate(all_range)
    hw_rx_arr = np.concatenate(all_hw_rx)
    tx_st_arr = np.concatenate(all_tx_st)
    raw_active_arr = np.concatenate(all_raw_active)
    rx_win_arr = np.concatenate(all_rx_win)
    dwell_arr = np.array(all_dwell)

    # Surface peak positions
    raw_peaks = np.argmax(np.abs(all_data), axis=1)
    peaks_sm = median_filter(raw_peaks, size=15)
    ref = int(np.median(peaks_sm))

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                                         gridspec_kw={'height_ratios': [3, 2, 2]})
    x = np.arange(len(delays))

    # Top panel: peak tracking
    ax1.plot(x, raw_peaks, '.', markersize=0.8, alpha=0.4, color='gray', label='raw peaks')
    ax1.plot(x, peaks_sm, '-', linewidth=1.5, color='red', label='median filtered')
    ax1.plot(x, ranges - ranges[0] + ref, '-', linewidth=1, color='orange', alpha=0.8, label='2*SC_altitude_above_target_ellipsoid / (c/fs)')
    ax1.plot(x, delays - delays[0] + ref, '-', linewidth=1, color='blue', alpha=0.5, label='(HW_RX_opening_ticks - TX_start_ticks) * (Raw_active_mode_length / RX_window_length_ticks)')

    for d in unique_dwells[1:]:
        matches = np.where(dwell_arr == d)[0]
        if len(matches):
            ax1.axvline(matches[0], color='gray', alpha=0.2, linewidth=0.5)
            ax2.axvline(matches[0], color='gray', alpha=0.2, linewidth=0.5)
            ax3.axvline(matches[0], color='gray', alpha=0.2, linewidth=0.5)

    ax1.set_ylabel('Fast Time Sample')
    ax1.set_title(f'{ch}  (fs={fs/1e6:.1f} MHz, {sf}x stacked) — Peak Tracking vs Predictions', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # Middle panel: raw engineering ticks
    #ax2.plot(x, hw_rx_arr, '-', linewidth=1, color='green', label='HW_RX_opening_ticks')
    #ax2.plot(x, tx_st_arr, '-', linewidth=1, color='purple', label='TX_start_ticks')
    ax2.plot(x, hw_rx_arr - tx_st_arr, '-', linewidth=1, color='brown', label='HW_RX_opening_ticks - TX_start_ticks')
    ax2.set_ylabel('Ticks')
    ax2.legend(loc='upper right', fontsize=9)

    # Bottom panel: window sizes
    #ax3.plot(x, raw_active_arr, '-', linewidth=1, color='teal', label='Raw_active_mode_length')
    #ax3.plot(x, rx_win_arr, '-', linewidth=1, color='brown', label='RX_window_length_ticks')
    ax3.plot(x, raw_active_arr / rx_win_arr, '-', linewidth=1, color='magenta', label='Raw_active_mode_length / RX_window_length_ticks')
    ax3.axhline(fs / 48e6, color='gray', alpha=0.5, linestyle='--', label='fs / 48e6 (nominal ticks/second)')
    ax3.set_xlabel('Stacked Pulse Index')
    ax3.set_ylabel('samples / tick')
    ax3.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    out = output_dir / f'debug_peak_tracking_{ch}.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
