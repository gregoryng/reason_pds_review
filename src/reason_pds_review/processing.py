"""Processing functions for REASON data."""

import numpy as np
import xarray as xr
from typing import Union, Optional
from scipy import signal
from scipy.constants import speed_of_light

def apply_stacking(data: Union[np.ndarray, xr.DataArray],
                   stack_factor: int,
                   axis: int = 0) -> Union[np.ndarray, xr.DataArray]:
    """
    Apply incoherent averaging (stacking) to reduce noise.

    Stacking averages multiple consecutive pulses together to improve
    signal-to-noise ratio at the cost of reduced along-track resolution.
    This is also known as incoherent integration or multi-look averaging.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Input data to stack. For radar data, typically has shape
        (slow_time, fast_time) with complex values.
    stack_factor : int
        Number of consecutive samples to average together along the
        stacking axis. Must be >= 1. If 1, returns data unchanged.
    axis : int, optional
        Axis along which to perform stacking. Default is 0 (slow_time).

    Returns
    -------
    np.ndarray or xr.DataArray
        Stacked data with reduced dimension along stacking axis.
        Output shape along axis is input_size // stack_factor.
        If input is xr.DataArray, coordinates are updated accordingly.

    Examples
    --------
    >>> # Stack complex radar data 10x along slow time
    >>> complex_data = hf_science['complex'].values  # shape: (9698, 964)
    >>> stacked = apply_stacking(complex_data, stack_factor=10)
    >>> stacked.shape  # (969, 964)

    >>> # Stack xarray DataArray preserving coordinates
    >>> stacked_da = apply_stacking(hf_science['complex'], stack_factor=10)
    >>> stacked_da.coords['slow_time']  # Updated to [0, 1, 2, ..., 968]

    Notes
    -----
    - Partial stacks at the end are discarded (uses floor division)
    - For complex data, averaging is done on complex values (coherent averaging),
      but typically applied to already-pulse-compressed data (incoherent)
    - Common stacking factors: HF 10x, VHF 20x
    """
    if stack_factor <= 1:
        return data

    is_dataarray = isinstance(data, xr.DataArray)

    # Get numpy array
    arr = data.values if is_dataarray else data

    # Calculate output size
    input_size = arr.shape[axis]
    output_size = input_size // stack_factor

    # Create index slices for reshaping
    # Move stacking axis to front, reshape to (output_size, stack_factor, ...)
    arr = np.moveaxis(arr, axis, 0)
    original_shape = arr.shape

    # Reshape: (input_size, ...) -> (output_size, stack_factor, ...)
    new_shape = (output_size, stack_factor) + original_shape[1:]
    arr_reshaped = arr[:output_size * stack_factor].reshape(new_shape)

    # Average along stack_factor dimension (axis 1)
    arr_stacked = np.mean(arr_reshaped, axis=1)

    # Move axis back to original position
    arr_stacked = np.moveaxis(arr_stacked, 0, axis)

    # If input was DataArray, return DataArray with updated coordinates
    if is_dataarray:
        # Get dimension name for the stacking axis
        dim_name = data.dims[axis]

        # Create new coordinates
        new_coords = {}
        for coord_name, coord_data in data.coords.items():
            if dim_name in coord_data.dims:
                # Update coordinate for stacked dimension
                if coord_name == dim_name:
                    # For the main coordinate, just use indices
                    new_coords[coord_name] = np.arange(output_size)
                else:
                    # For other coordinates, take first value of each stack
                    coord_axis = coord_data.dims.index(dim_name)
                    new_coords[coord_name] = np.take(coord_data.values,
                                                     np.arange(0, output_size * stack_factor, stack_factor),
                                                     axis=coord_axis)
            else:
                # Coordinate doesn't depend on stacked dimension
                new_coords[coord_name] = coord_data

        # Create new DataArray
        result = xr.DataArray(
            arr_stacked,
            coords=new_coords,
            dims=data.dims,
            attrs=data.attrs.copy()
        )

        # Add stacking info to attributes
        result.attrs['stacking_factor'] = stack_factor
        result.attrs['stacking_axis'] = dim_name

        return result
    else:
        return arr_stacked


def calculate_amplitude_db(complex_data: Union[np.ndarray, xr.DataArray],
                          floor_value: float = 1e-10) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate amplitude in decibels from complex radar data.

    Parameters
    ----------
    complex_data : np.ndarray or xr.DataArray
        Complex-valued radar data (I + jQ)
    floor_value : float, optional
        Minimum value to avoid log(0). Default is 1e-10.

    Returns
    -------
    np.ndarray or xr.DataArray
        Amplitude in dB scale: 20*log10(|complex_data| + floor_value)

    Examples
    --------
    >>> complex_iq = hf_science['complex']
    >>> amp_db = calculate_amplitude_db(complex_iq)
    >>> amp_db.plot()  # Radargram visualization

    Notes
    -----
    Uses 20*log10 for power-in-amplitude (voltage) conversion.
    For power values, use 10*log10 instead.
    """
    is_dataarray = isinstance(complex_data, xr.DataArray)

    # Get numpy array
    arr = complex_data.values if is_dataarray else complex_data

    # Calculate amplitude in dB
    amplitude = np.abs(arr)
    amplitude_db = 20 * np.log10(amplitude + floor_value)

    if is_dataarray:
        result = xr.DataArray(
            amplitude_db,
            coords=complex_data.coords,
            dims=complex_data.dims,
            attrs=complex_data.attrs.copy()
        )
        result.attrs['description'] = 'Amplitude in dB scale'
        result.attrs['units'] = 'dB'
        return result
    else:
        return amplitude_db

def generate_chirp(chirp_start_freq: float,
                   chirp_end_freq: float,
                   chirp_length_ticks: int,
                   sample_rate: float,
                   window: str = None) -> np.ndarray:
    """
    Generate complex baseband chirp equivalent to original make_chirp().
    Window argument is ignored to preserve original behavior.
    """
    fc = (chirp_start_freq + chirp_end_freq) / 2
    # Subtracting to set fc=0 makes HF come out correct.
    chirp_start_freq -= fc
    chirp_end_freq -= fc

    # Convert ticks (48 MHz clock) to seconds
    duration_sec = chirp_length_ticks / 48e6

    # Number of samples at requested sample rate
    duration_samples = int(duration_sec * sample_rate)

    # Time vector equivalent to np.arange(0, T, 1/fs)
    t = np.arange(duration_samples) / sample_rate

    # Linear chirp phase (identical to original implementation)
    phase = 2 * np.pi * (
        chirp_start_freq * t +
        (chirp_end_freq - chirp_start_freq) * t**2 / (2 * duration_sec)
    )

    chirp_win = None
    return np.exp(1j * phase), chirp_win


def generate_chirp_old(chirp_start_freq: float,
                   chirp_end_freq: float,
                   chirp_length_ticks: int,
                   sample_rate: float,
                   window: str = 'hann') -> np.ndarray:
    """
    Generate a complex reference chirp for pulse compression.

    The chirp is centered at zero frequency (complex baseband) and spans
    the specified bandwidth with optional windowing to suppress range sidelobes.

    Parameters
    ----------
    chirp_start_freq : float
        Starting frequency of the chirp in Hz
    chirp_end_freq : float
        Ending frequency of the chirp in Hz
    chirp_length_ticks : int
        Duration of the chirp in sample ticks
    sample_rate : float
        Sample rate in Hz
    window : str, optional
        Window function to apply ('hann', 'hamming', 'blackman', or None).
        Default is 'hann' to suppress range sidelobes.

    Returns
    -------
    np.ndarray
        Complex reference chirp signal

    Examples
    --------
    >>> # Generate HF chirp from engineering data
    >>> chirp = generate_chirp(
    ...     chirp_start_freq=chirp_start[0],
    ...     chirp_end_freq=chirp_end[0],
    ...     chirp_length_ticks=chirp_length[0],
    ...     sample_rate=1.2e6
    ... )

    Notes
    -----
    The chirp is complex baseband, spanning the bandwidth centered on zero.
    Windowing (e.g., Hann) suppresses range sidelobes at the cost of slight
    range resolution degradation.
    """
    # Calculate bandwidth and center frequency (baseband, so center = 0)
    bandwidth = chirp_end_freq - chirp_start_freq
    f0 = -bandwidth / 2  # Start at -BW/2 for baseband centered at 0
    f1 = bandwidth / 2   # End at +BW/2

    # Convert chirp length from ticks to seconds
    duration_sec = chirp_length_ticks / 48e6  # Ticks are from a 48 MHz clock, as inferred from Table B1
    duration_samples = int(duration_sec * sample_rate)

    # Time vector for the chirp
    t = np.linspace(0, duration_sec, duration_samples, endpoint=False)

    # Generate complex linear chirp
    chirp_complex = signal.chirp(t, f0=f0, f1=f1, t1=t[-1], method='linear', phi=0, complex=True)

    # Apply window to suppress sidelobes
    win = np.ones(len(t))
    if window is not None and window.lower() != 'none':
        if window.lower() == 'hann':
            win = np.hanning(len(t))
        elif window.lower() == 'hamming':
            win = np.hamming(len(t))
        elif window.lower() == 'blackman':
            win = np.blackman(len(t))
        else:
            raise ValueError(f"Unknown window type: {window}")

        chirp_complex = chirp_complex * win

    return chirp_complex, win


def pulse_compress(data: np.ndarray,
                   chirp: np.ndarray,
                   axis: int = -1) -> np.ndarray:
    """
    Pulse compression using FFT-based circular correlation,
    matching former implementation.
    """

    # Ensure chirp length matches data length along axis
    N = data.shape[axis]

    if chirp.shape[0] != N:
        #raise ValueError("Chirp length must match data length for circular FFT method.")
        # zero pad to data
        #padded = np.zeros(N, dtype=chirp.dtype)
        #padded[:chirp.shape[0]] = chirp
        padded = np.pad(chirp, (N - len(chirp), 0), 'constant', constant_values=0)
        chirp = padded

    # FFT along specified axis
    data_fft = np.fft.fft(np.conjugate(data), axis=axis)
    chirp_fft = np.fft.fft(chirp, n=N)

    # Multiply in frequency domain (broadcast chirp over slow-time dimension)
    result_fft = data_fft * np.expand_dims(chirp_fft, tuple(
        i for i in range(data.ndim) if i != axis
    ))

    # Apply frequency-domain Hamming window (with fftshift like original)
    window = np.fft.fftshift(np.hamming(N))
    window_shape = [1] * data.ndim
    window_shape[axis] = N
    window = window.reshape(window_shape)

    result_fft *= window

    # Inverse FFT to obtain compressed signal
    compressed = np.fft.ifft(result_fft, axis=axis)

    return compressed

def pulse_compress_old(data: np.ndarray,
                   chirp: np.ndarray,
                   axis: int = -1) -> np.ndarray:
    """
    Apply pulse compression via matched filter

    Parameters
    ----------
    data : np.ndarray
        Complex radar data to compress. Typically shape (slow_time, fast_time).
    chirp : np.ndarray
        Complex reference chirp signal from generate_chirp()
    axis : int, optional
        Axis along which to apply compression (default: -1, fast time axis)

    Returns
    -------
    np.ndarray
        Pulse-compressed complex data with same shape as input

    Examples
    --------
    >>> # Compress a single record
    >>> chirp = generate_chirp(chirp_start, chirp_end, chirp_len, sample_rate)
    >>> compressed = pulse_compress(raw_data, chirp, axis=1)
    """

    # Apply cross-correlation along specified axis
    # Use 'same' mode to maintain input length
    compressed = np.apply_along_axis(
        lambda m: signal.correlate(m, chirp, mode='same'),
        axis=axis,
        arr=data
    )

    return compressed

def pulse_compress_match_fft(data: np.ndarray,
                             chirp: np.ndarray,
                             axis: int = -1) -> np.ndarray:
    """
    Reproduce:
        ifft( fft(chirp) * fft(conj(data)) )

    using scipy.signal.correlate, compare with DAY
    """

    N = data.shape[axis]

    def _process(m):
        # Circular correlation equivalent
        # correlate(chirp, conj(m)) gives linear correlation,
        # so we take 'full' and wrap manually.
        full = signal.correlate(chirp, np.conjugate(m), mode='full')

        # Extract first N samples to mimic circular result
        return full[:N]

    return np.apply_along_axis(_process, axis, data)



def align_by_delay(data: np.ndarray,
                   hw_rx_opening_ticks: np.ndarray,
                   tx_start_ticks: np.ndarray,
                   chirp_length_ticks: np.ndarray,
                   rx_window_length_ticks: np.ndarray,
                   raw_active_mode_length: np.ndarray,
                   rx_delay_tracking_offset: np.ndarray,
                   axis: int = 0) -> np.ndarray:
    """
    Align fast time records by rolling to account for varying delays.

    In the Partially Processed Data Product, dwell delays are varied to maintain
    tracking of the surface. This function aligns records by computing the delay
    offset and rolling each fast time record accordingly.

    Parameters
    ----------
    data : np.ndarray
        Complex radar data, typically shape (slow_time, fast_time)
    hw_rx_opening_ticks : np.ndarray
        Hardware RX opening time in ticks for each pulse (slow_time,)
    tx_start_ticks : np.ndarray
        TX start time in ticks for each pulse (slow_time,)
    chirp_length_ticks : np.ndarray
        Chirp length in ticks for each pulse (slow_time,)
    rx_window_length_ticks : np.ndarray
        RX window length in ticks for each pulse (slow_time,)
    raw_active_mode_length : np.ndarray
        Raw active mode length (number of fast time samples) for each pulse
    rx_delay_tracking_offset: np.ndarray
        RX delay tracking offset in 1/32 ticks (slow_time,)
    axis : int, optional
        Axis corresponding to slow time (default: 0)

    Returns
    -------
    np.ndarray
        Aligned radar data with same shape as input

    Notes
    -----
    The delay between transmit pulse and receive window start is:
        delay_ticks = (hw_rx_opening_ticks - tx_start_ticks)

    This is converted to samples using the sample rate derived from:
        sample_rate = raw_active_mode_length / rx_window_length_ticks

    Each fast time record is then rolled by the computed sample offset.
    """
    aligned_data = None #data.copy()

    # Calculate delay in ticks for each pulse
    delay_ticks = (hw_rx_opening_ticks - tx_start_ticks - rx_delay_tracking_offset/32.)# + chirp_length_ticks

    # TODO: Documentation says:
    # The delay between the transmit pulse and the start of the receive window is
    # (ENG:HW_RX_opening_ticks - ENG:TX_start_ticks), and offsets resulting from the
    # variability of the chirp length is ENG:chirp_length_ticks; when converted to a
    # number of fast time samples using the sample rate
    # (ENG:raw_active_mode_length/ENG:RX_window_length_ticks), you can roll each fast
    # time record array to align.

    # I believe I'm still missing some correction factor that varies by dwell here, but
    # I haven't been able to figure out what.
    # sample rate in samples per tick
    sample_rate = raw_active_mode_length / rx_window_length_ticks
    delay_samples = delay_ticks * sample_rate

    # Compute roll amounts relative to first pulse
    #reference_delay = 0 #delay_samples[0]
    #roll_amounts = delay_samples - reference_delay
    # zero out for testing
    #roll_amounts[:] = 0

    # Roll each fast time record
    ''' disabled
    for i in range(aligned_data.shape[axis]):
        if axis == 0:
            aligned_data[i, :] = np.roll(aligned_data[i, :], -roll_amounts[i])
        elif axis == 1:
            aligned_data[:, i] = np.roll(aligned_data[:, i], -roll_amounts[i])
        else:
            raise ValueError("axis must be 0 or 1 for 2D data")
    '''
    return aligned_data, delay_samples


def geometric_correction(data: np.ndarray,
                        altitude_km: np.ndarray,
                        sample_rate: float,
                        axis: int = 0) -> np.ndarray:
    """
    Apply geometric correction by aligning records to reference ellipsoid range.

    Uses spacecraft altitude to compute range to reference ellipsoid and rolls
    each fast time record to align surface returns geometrically.

    Parameters
    ----------
    data : np.ndarray
        Complex radar data, typically shape (slow_time, fast_time)
    altitude_km : np.ndarray
        Spacecraft altitude above target ellipsoid in km (slow_time,)
    sample_rate : float
        Sample rate in Hz
    axis : int, optional
        Axis corresponding to slow time (default: 0)

    Returns
    -------
    np.ndarray
        Geometrically corrected radar data with same shape as input

    Notes
    -----
    Range to ellipsoid is computed using two-way speed of light:
        range_samples = (2 * altitude_km * 1000) / (c / sample_rate)
    where c = 299792458 m/s (speed of light in vacuum)

    For subsurface analysis, use speed of light in ice (~169 m/µs).
    Each record is rolled to align the surface return.
    """
    c = speed_of_light  # Speed of light in m/s

    corrected_data = None #data.copy()

    # Calculate two-way range in samples
    # altitude_km * 1000 = altitude in meters
    # Two-way distance = 2 * altitude
    # Sample distance = c / sample_rate
    range_samples = (2 * altitude_km * 1000) / (c / sample_rate)

    return range_samples

    ''' no
    # Compute roll amounts relative to first pulse

    reference_range = range_samples[0]

    roll_amounts = range_samples - reference_range
    roll_amounts = np.round(roll_amounts).astype(int)
    # zero out for testing
    #roll_amounts[:] = 0

    # Roll each fast time record
    # Positive roll_amounts means higher altitude -> later delay -> shift right (positive roll)
    for i in range(corrected_data.shape[axis]):
        if axis == 0:
            corrected_data[i, :] = np.roll(corrected_data[i, :], roll_amounts[i])
        elif axis == 1:
            corrected_data[:, i] = np.roll(corrected_data[:, i], roll_amounts[i])
        else:
            raise ValueError("axis must be 0 or 1 for 2D data")

    return corrected_data, roll_amounts
    '''


def roll_radargram2(data:np.ndarray, roll_amounts:np.ndarray):
    axis = 0
    corrected_data = np.empty_like(data)
    reference_roll = 0 #roll_amounts[0]
    roll_amounts1 = np.round(roll_amounts - reference_roll).astype(int)
    assert len(roll_amounts1) == corrected_data.shape[axis]
    
    # Roll each fast time record
    # Positive roll_amounts means higher altitude -> later delay -> shift right (positive roll)
    for i, nsamples in enumerate(roll_amounts1):
        corrected_data[i, :] = np.roll(data[i, :], nsamples)# % data.shape[1])

    return corrected_data


def roll_radargram(data:np.ndarray, roll_amounts:np.ndarray, axis:int=0):

    corrected_data = np.empty_like(data)
    reference_roll = 0 #roll_amounts[0]
    roll_amounts1 = np.round(roll_amounts - reference_roll).astype(int)
    assert len(roll_amounts1) == corrected_data.shape[axis]
    
    # Roll each fast time record
    # Positive roll_amounts means higher altitude -> later delay -> shift right (positive roll)
    for i, nsamples in enumerate(roll_amounts1):
        if axis == 0:
            corrected_data[i, :] = np.roll(data[i, :], nsamples)# % data.shape[1])
        elif axis == 1:
            corrected_data[:, i] = np.roll(data[:, i], nsamples) #% data.shape[0])
        else:
            raise ValueError("axis must be 0 or 1 for 2D data")
    return corrected_data
