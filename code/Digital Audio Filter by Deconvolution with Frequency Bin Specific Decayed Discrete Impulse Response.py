# -*- coding: utf-8 -*-
"""
Started on Wed Jul 10 20:01:40 2024
Finished on Sat Aug  9 13:41:53 2025

@author: Stefan Ciba, M.Sc.
"""
import numpy as np
import soundfile as sf
import scipy.signal as sp_signal
import os
import datetime
import time
import sounddevice as sd
import librosa

def generate_periodic_sine_sweep(start_freq, end_freq, duration_sec, sample_rate, periods):
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    sweep = sp_signal.chirp(t, f0=start_freq, f1=end_freq, t1=duration_sec, method='linear')
    periodic_sweep = np.tile(sweep, periods)
    return periodic_sweep

def calculate_rms(signal):
    return np.sqrt(np.mean(signal**2))

# the Sine Sweep and the recorded Sine Sweep can be off phase, so it appears, that especially in the lowest frequency range periodic signal appear instead of decayed response.
def calculate_impulse_response(input_sig, output_sig, n_fft, hop_length):
    # Perform cross-correlation to align input and output signals
    corr = sp_signal.correlate(output_sig, input_sig, mode='full')
    delay = np.argmax(corr) - len(input_sig) + 1

    # Adjust output signal length to match input signal, crop if necessary
    if delay > 0:
        output_sig = output_sig[delay:]
    elif delay < 0:
        input_sig = input_sig[-delay:]

    # Calculate number of frames
    n_frames = (len(input_sig) + hop_length - 1) // hop_length

    # Initialize impulse response
    impulse_response = np.zeros(n_fft)

    # Process each frame
    for i in range(n_frames):
        # Define frame indices
        start_idx = i * hop_length
        end_idx = min(start_idx + n_fft, len(input_sig))

        # Extract frames from input and output signals
        input_frame = input_sig[start_idx:end_idx]
        output_frame = output_sig[start_idx:end_idx]

        # Compute FFT of frames
        input_spec = np.fft.fft(input_frame, n_fft)
        output_spec = np.fft.fft(output_frame, n_fft)

        # Regularization in the frequency domain
        input_spec[np.abs(input_spec) < np.finfo(float).eps] = np.finfo(float).eps
        output_spec[np.abs(output_spec) < np.finfo(float).eps] = np.finfo(float).eps
        
        # Compute log magnitude spectrum (cepstrum) of input and output signals
        cepstrum_input = np.fft.ifft(np.log(np.abs(input_spec) + np.finfo(float).eps)).real
        cepstrum_output = np.fft.ifft(np.log(np.abs(output_spec) + np.finfo(float).eps)).real

        # Perform deconvolution in cepstral domain
        cepstrum_ir = cepstrum_output - cepstrum_input

        # Inverse FFT to get time-domain impulse response frame
        impulse_response_frame = (np.fft.ifft(np.exp(np.fft.fft(cepstrum_ir).real), n_fft).real)
        
        # Accumulate impulse response frame (overlap-add)
        impulse_response[:len(impulse_response_frame)] += impulse_response_frame

    # Average impulse response
    impulse_response /= n_frames
    
    # max impulse response
    impulse_response_max = np.max(np.abs(impulse_response))
    
    # Normalize impulse response
    impulse_response /= impulse_response_max
    
    #Select valid Impulse Response
    ir=impulse_response[:int(hop_length)-1]
    
    #Filter out artifacts that appear in the rear end of the impulse response, which is caused by transient in the time domain from Sine Sweep jump high to low frequency.
    taper_start = int(len(ir) * 0.75)  # start of last 25%
    taper_length = len(ir) - taper_start

    # Create soft taper from 1 to 0 to fade out last 25%
    taper = np.exp(-0.00025*np.arange(taper_length))

    # Apply taper
    ir_tapered = np.copy(ir)
    ir_tapered[taper_start:] *= taper
    
    ir=ir_tapered
    return ir
"""
def calculate_impulse_response(input_sig, output_sig, n_fft, hop_length): # uses complex cepstrum for correct back transformation
    corr = sp_signal.correlate(output_sig, input_sig, mode='full')
    delay = np.argmax(corr) - len(input_sig) + 1
    if delay > 0:
        output_sig = output_sig[delay:]
    elif delay < 0:
        input_sig = input_sig[-delay:]
    n_frames = (len(input_sig) + hop_length - 1) // hop_length
    impulse_response = np.zeros(n_fft, dtype=np.complex128)  # Use complex for phase

    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = min(start_idx + n_fft, len(input_sig))
        input_frame = input_sig[start_idx:end_idx]
        output_frame = output_sig[start_idx:end_idx]
        # Zero-pad for full FFT size
        if input_frame.size < n_fft:
            input_frame = np.pad(input_frame, (0, n_fft - input_frame.size))
        if output_frame.size < n_fft:
            output_frame = np.pad(output_frame, (0, n_fft - output_frame.size))
        input_spec = np.fft.fft(input_frame, n_fft)
        output_spec = np.fft.fft(output_frame, n_fft)
        # Regularization to avoid log(0)
        input_spec[np.abs(input_spec) < np.finfo(float).eps] = np.finfo(float).eps
        output_spec[np.abs(output_spec) < np.finfo(float).eps] = np.finfo(float).eps

        # Complex cepstrum
        cepstrum_input = np.fft.ifft(np.log(input_spec))
        cepstrum_output = np.fft.ifft(np.log(output_spec))
        # Cepstral deconvolution
        cepstrum_ir = cepstrum_output - cepstrum_input

        # Inverse complex cepstrum: exp(FFT(cepstrum_ir)), then ifft
        log_spectrum_ir = np.fft.fft(cepstrum_ir)
        spectrum_ir = np.exp(log_spectrum_ir)
        impulse_response_frame = np.fft.ifft(spectrum_ir).real  # restore to time domain

        impulse_response[:len(impulse_response_frame)] += impulse_response_frame

    impulse_response = impulse_response.real
    impulse_response /= n_frames
    impulse_response_max = np.max(np.abs(impulse_response))
    impulse_response /= impulse_response_max

    return impulse_response[:int(hop_length)-1]
"""

def calculate_t60(signal, sample_rate, n_fft=2048, hop_length=512):
    overlap_fac = n_fft/2 * 1/hop_length
    stft = np.abs(stft_custom(signal, n_fft, hop_length)) ** 2
    cumulative_energy = np.cumsum(stft[::-1], axis=0)[::-1]
    cumulative_energy /= (cumulative_energy[0, :]+np.finfo(float).eps)
    threshold_index = np.argmax(cumulative_energy < 0.001, axis=0)
    t60 = ((overlap_fac)* threshold_index / sample_rate) # overlap_fac originally: threshold_index * hp_length/ sample_rate
    return t60

def apply_decay_function(stft_in, sample_rate, hop_length, t60_diff):
    num_frames = stft_in.shape[1]
    times = np.linspace(0, num_frames * hop_length / sample_rate, num_frames)
    decay_functions = np.zeros((stft_in.shape[0], num_frames))
    stft_out=stft_in
    # Apply decay to each frequency bin
    for freq_bin in range(stft_in.shape[0]):
        decay_rate = t60_diff[freq_bin]
        if decay_rate != 0:
            decay_function = np.exp(-times/(decay_rate))
            decay_function = decay_function#/np.max(np.abs(decay_function))
            stft_out[freq_bin, :] *= decay_function  
           
            decay_functions[freq_bin, :] = decay_function
        else:
            decay_functions[freq_bin, :] = np.ones_like(times)

    
    #plt.figure(figsize=(10, 6))
    #plt.imshow(decay_functions, aspect='auto', origin='lower', cmap='viridis',
    #           extent=[0, decay_functions.shape[1] * hop_length / sample_rate, 0, decay_functions.shape[0]])
    #plt.colorbar(label='Decay Factor')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Frequency Bin')
    #plt.title('Decay Functions for Each Frequency Bin')
    #plt.show()
    
    test=stft_out-stft_in
    
    return stft_out


def stft_custom(signal, n_fft=2048, hop_length=512, win_length=None):
    if win_length is None:
        win_length = n_fft
    stft_data = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return stft_data

def istft_custom(stft_data, hop_length=512, length=None, win_length=None):
    if win_length is None:
        win_length = 2 * (stft_data.shape[0] - 1)  # Default window length based on STFT data shape
    istft_data = librosa.istft(stft_data, hop_length=hop_length, win_length=win_length, length=length)
    return istft_data

def adjust_impulse_response(ir, test_signal, recorded_output, sample_rate, n_fft=2048, hop_length=512):
    stft_ir = stft_custom(ir, n_fft, hop_length)
    stft_test_signal = stft_custom(test_signal, n_fft, hop_length)
    stft_recorded_output = stft_custom(recorded_output, n_fft, hop_length)
    
    t60_test_signal = calculate_t60(test_signal, sample_rate, n_fft=n_fft, hop_length=hop_length)
    t60_recorded_output = calculate_t60(recorded_output, sample_rate, n_fft=n_fft, hop_length=hop_length)
    
    num_bins = min(stft_ir.shape[0], stft_test_signal.shape[0], stft_recorded_output.shape[0])
    t60_diff = 1 * t60_recorded_output[:num_bins] / (t60_test_signal[:num_bins]+np.finfo(float).eps) 

    stft_ir_adjusted = apply_decay_function(stft_ir, sample_rate, hop_length, t60_diff)
    
    ir_adjusted = istft_custom(stft_ir_adjusted, hop_length, len(ir))

    #ir_adjusted = ir_adjusted / np.max(ir_adjusted)
    return ir_adjusted

def add_decay_to_impulse_response(impulse_response, decay_factor):
    """
    Apply exponential decay to the impulse response to achieve smoother decay in the time domain.

    Parameters:
    - impulse_response: The original impulse response.
    - decay_factor: The decay factor to control the rate of decay (default is 0.01).

    Returns:
    - decayed_impulse_response: The impulse response with added decay.
    """
    length = len(impulse_response)
    decay_window = np.exp(-decay_factor * np.arange(length))
    decayed_impulse_response = impulse_response * decay_window
    return decayed_impulse_response

"""def filter_signal_with_impulse_response(test_sig, impulse_response, gain_factor):
    n_fft = len(impulse_response)
    filtered_sig = np.zeros(len(test_sig), dtype=np.float32)  # Initialize filtered signal array

    # Apply FFT to impulse response
    impulse_spec = np.fft.fft(impulse_response, n_fft)

    # Regularization in the frequency domain
    impulse_spec[np.abs(impulse_spec) < np.finfo(float).eps] = np.finfo(float).eps

    # Calculate number of frames
    n_frames = (len(test_sig) + n_fft - 1) // n_fft

    for i in range(n_frames):
        # Define frame indices
        start_idx = i * n_fft
        end_idx = min(start_idx + n_fft, len(test_sig))

        # Extract frame from test signal
        test_frame = test_sig[start_idx:end_idx]

        # Apply FFT to test frame
        test_spec = np.fft.fft(test_frame, n_fft)

        # deconvolve in frequency domain
        conv_spec = test_spec / impulse_spec

        # Inverse FFT to get time-domain frame
        filtered_frame = np.fft.ifft(conv_spec).real

        # Add to filtered signal (overlap-add method)
        filtered_sig[start_idx:end_idx] += filtered_frame[:len(test_frame)]

    # Apply gain factor to the entire filtered signal
    filtered_sig *= gain_factor

    return filtered_sig
"""

def filter_signal_with_impulse_response(test_sig, impulse_response, gain_factor, hop_factor=0.5):
    n_fft = len(impulse_response)
    hop_size = int(n_fft * hop_factor)
    window = np.hanning(n_fft)
    filtered_sig = np.zeros(len(test_sig) + n_fft, dtype=np.float32)  # Zero-pad tail for overlap-add
    window_correction = np.zeros_like(filtered_sig)

    # FFT of impulse response (frequency response)
    impulse_spec = np.fft.fft(impulse_response, n_fft)
    # Regularize small values to avoid division by zero
    impulse_spec[np.abs(impulse_spec) < np.finfo(float).eps] = np.finfo(float).eps

    # Calculate number of frames with overlap
    n_frames = (len(test_sig) - n_fft + hop_size) // hop_size + 1

    for i in range(n_frames):
        start_idx = i * hop_size
        # Extract current frame, zero-pad if needed 
        frame = test_sig[start_idx:start_idx + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))

        # Apply analysis window
        frame_win = frame * window

        # FFT of the windowed frame
        frame_spec = np.fft.fft(frame_win, n_fft)

        # Frequency domain deconvolution (division)
        conv_spec = frame_spec / impulse_spec

        # Back to time domain (real part)
        filtered_frame = np.fft.ifft(conv_spec).real

        # Apply synthesis window
        filtered_win = filtered_frame * window

        # Overlap-add to output buffer
        filtered_sig[start_idx:start_idx + n_fft] += filtered_win
        # Accumulate squared window for normalization later
        window_correction[start_idx:start_idx + n_fft] += window**2

    # Normalize to compensate for overlap and windowing
    nonzero = window_correction > 1e-10
    filtered_sig[nonzero] /= window_correction[nonzero]

    # Truncate to original signal length
    filtered_sig = filtered_sig[:len(test_sig)]

    # Apply gain factor
    filtered_sig *= gain_factor

    return filtered_sig


# Objective Measurement:

def sr_db(original_signal, filtered_signal):   
    Ps1 = np.mean(np.square((original_signal-np.mean(original_signal)))) # normazized without DC component
    Ps2 = np.mean(np.square((filtered_signal-np.mean(filtered_signal)))) # normalized without DC component
    Pn = Ps1 - Ps2 # implicates that noise = original_signal - filtered_signal
    
    # calculate the signal ratios of the filtered and the original Signal Power:
    snr_db = 10 * (np.log10(Ps2) - np.log10(Pn))
    lsd_db = 10 * (np.log10(Ps2) - np.log10(Ps1))
    return snr_db, lsd_db


def calculate_room_volume(echo_time):
    """
    Estimate the room volume based on the most relevant echo time.

    Parameters:
    - echo_time: The most relevant echo time in seconds.

    Returns:
    - The estimated room volume in cubic meters.
    """
    # Speed of sound in air at room temperature (m/s)
    speed_of_sound = 343.0
    
    # Calculate the distance sound has traveled (round trip)
    distance = (echo_time * speed_of_sound) / 2
    
    # Assume a cubic room for simplicity
    room_dimension = distance  # Since distance corresponds to a single trip
    room_volume = room_dimension ** 3
    
    return room_volume

def schroeder_integration(impulse_response):
    """Compute the backward energy decay curve using Schroeder integration."""
    energy = impulse_response ** 2
    cumulative_energy = np.cumsum(energy[::-1])[::-1]
    normalized_energy = cumulative_energy / (np.max(cumulative_energy) + np.finfo(float).eps)
    return normalized_energy

def early_decay_time(impulse_response, fs):
    """Calculate Early Decay Time (EDT) in seconds.
    EDT is the time taken for the energy to drop by 10 dB."""
    energy_decay = schroeder_integration(impulse_response)
    # Convert to dB
    decay_db = 10 * np.log10(energy_decay + np.finfo(float).eps)

    # Find indices where decay crosses 0 dB and -10 dB
    try:
        idx_0 = np.where(decay_db <= 0)[0][0]
        idx_10 = np.where(decay_db <= -10)[0][0]
    except IndexError:
        # If the signal does not decay enough, fallback
        return np.nan

    # Linear regression on decay between 0 and -10 dB, find slope in dB/s
    x = np.array([idx_0, idx_10]) / fs
    y = decay_db[[idx_0, idx_10]]

    slope = (y[1] - y[0]) / (x[1] - x[0])

    edt = -60 / slope  # Extrapolate time to 60 dB decay
    return edt

def clarity_index(impulse_response, fs, time_ms):
    """
    Calculate clarity index (C50 or C80).
    time_ms: 50 for C50, 80 for C80
    """
    time_samples = int(time_ms * 1e-3 * fs)
    energy_early = np.sum(impulse_response[:time_samples] ** 2)
    energy_late = np.sum(impulse_response[time_samples:] ** 2)
    if energy_late == 0:
        return np.nan
    clarity = 10 * np.log10(energy_early / energy_late)
    return clarity

def definition_index(impulse_response, fs):
    """Calculate Definition (D50) index as ratio of early to total energy within 50 ms."""
    time_samples = int(0.05 * fs)  # 50 ms
    energy_early = np.sum(impulse_response[:time_samples] ** 2)
    energy_total = np.sum(impulse_response ** 2)
    if energy_total == 0:
        return np.nan
    definition = energy_early / energy_total
    return definition

def direct_to_reverberant_ratio(impulse_response, fs, direct_time_ms=5):
    """
    Calculate Direct-to-Reverberant Ratio (DRR).
    direct_time_ms: Integration window for direct sound energy (default 5 ms).
    Remaining energy assumed reverberant.
    """
    direct_samples = int(direct_time_ms * 1e-3 * fs)
    energy_direct = np.sum(impulse_response[:direct_samples] ** 2)
    energy_reverb = np.sum(impulse_response[direct_samples:] ** 2)
    if energy_reverb == 0:
        return np.nan
    drr = 10 * np.log10(energy_direct / energy_reverb)
    return drr


def process_files(input_file1, input_file2, test_file, output_folder, sample_rate):
    start_time = time.time()

    input_sig, _ = sf.read(input_file1, dtype='float32')
    output_sig, _ = sf.read(input_file2, dtype='float32')
    test_sig, _ = sf.read(test_file, dtype='float32')

    # Ensure input_sig and output_sig are the same length
    min_len = min(len(input_sig), len(output_sig))
    input_sig = input_sig[:min_len]
    output_sig = output_sig[:min_len]
    test_sig=test_sig[:]

    #plot_wavelet_transform(input_sig, sample_rate)
    #plot_wavelet_transform(output_sig, sample_rate)

    # Determine FFT parameters
    n_fft = int(5*(sample_rate))  # Example value, replace with actual n_fft
    hop_length = n_fft // 2  # 50% overlap

    # Calculate gain factor   # Optimize impulse response using NLMS adaptive filter
    # Smooth the optimized impulse response
    impulse_response = calculate_impulse_response(input_sig, output_sig, n_fft, hop_length)
    #plot_impulse_response(impulse_response, sample_rate)
    #impulse_response = impulse_response / np.max(impulse_response)

    # Calculate gain factor 
    #impulse_response_rms = calculate_rms(impulse_response)
    #output_sig_rms = calculate_rms(output_sig)
    #impulse_response=adjust_impulse_response(impulse_response, test_sig, output_sig, sample_rate)
    
    # additional decay as needed
    #impulse_response = add_decay_to_impulse_response(impulse_response, decay_factor=0.01)
    #plot_impulse_response(impulse_response, sample_rate)   

    # Filter test signal with decayed impulse response
    filtered_sig = filter_signal_with_impulse_response(test_sig, impulse_response[:int(1*len(impulse_response))], 1)
    filtered_sig = filtered_sig[10000:len(test_sig)]

    # Filter Sine Sweep for benchmarking and comparability
    filtered_sweep = filter_signal_with_impulse_response(output_sig, impulse_response[:int(1*len(impulse_response))], 1)
    filtered_sweep = filtered_sweep[:len(output_sig)]
    impulse_response_system = calculate_impulse_response(input_sig, output_sig, n_fft, hop_length)

    # calculate gain factor test signal
    filtered_sig_rms = calculate_rms(filtered_sig)
    test_sig_rms = calculate_rms(test_sig)
    gain_factor_sig = test_sig_rms / filtered_sig_rms # Aim for same RMS
    filtered_sig *= gain_factor_sig
    if np.max(np.abs(filtered_sig)) > 1: #strictly preventing clipping
        filtered_sig /= np.max(np.abs(filtered_sig))

    # calculate gain factor sweep signal
    filtered_sweep_rms = calculate_rms(filtered_sweep)
    sweep_rms = calculate_rms(input_sig)
    gain_factor_sweep = sweep_rms / filtered_sweep_rms # Aim for same RMS
    filtered_sweep *= gain_factor_sweep
    if np.max(np.abs(filtered_sweep)) > 1: #strictly preventing clipping
        filtered_sweep /= np.max(np.abs(filtered_sweep))

    # Save files
    impulse_response_system_file = os.path.join(output_folder, "average_impulse_response_system.wav")
    impulse_response_file = os.path.join(output_folder, "average_impulse_response_filter.wav")
    filtered_signal_file = os.path.join(output_folder, "filtered_signal.wav")
    filtered_sweep_file = os.path.join(output_folder, "filtered_sweep.wav")

    sf.write(impulse_response_system_file, impulse_response_system.astype('float32'), sample_rate, subtype='FLOAT')
    sf.write(impulse_response_file, impulse_response.astype('float32'), sample_rate, subtype='FLOAT')
    sf.write(filtered_signal_file, filtered_sig.astype('float32'), sample_rate, subtype='FLOAT')
    sf.write(filtered_sweep_file, filtered_sweep.astype('float32'), sample_rate, subtype='FLOAT')
    
    end_time = time.time()
    print(f"Calculation time: {end_time - start_time:.2f} seconds")

    # The SNR is relevant if the filter only filtered undesired early reflections and maybe other noise.
    print("the signal to noise ratio (snr) and the signal to signal ratio (lsd) in db are:") 
    print( sr_db(test_sig,filtered_sig) )
    
    # Find the most relevant echo time (example: 60 dB decay)
    t60=calculate_t60(impulse_response_system , sample_rate, n_fft=2048, hop_length=512)
    most_relevant_echo_time = np.mean(t60)
    room_volume=calculate_room_volume(most_relevant_echo_time)
    print("Experimental but not valid..Initial Room volume in square-meter estimate (oversimplified room shape):")
    print(room_volume)
    print("Most relevant initial echo time in seconds:")
    print(most_relevant_echo_time)
    
    # Find the most relevant echo time (example: 60 dB decay)
    t60=calculate_t60(impulse_response, sample_rate, n_fft=2048, hop_length=512)
    most_relevant_echo_time = np.mean(t60)
    room_volume=calculate_room_volume(most_relevant_echo_time)
    print("Experimental but not valid..Filtered Room volume in square-meter estimate (oversimplified room shape):")
    print(room_volume)
    print("Most relevant filtered echo time in seconds:")
    print(most_relevant_echo_time)
    
    
    normalized_energy_system = schroeder_integration(impulse_response_system)
    print("normalized_energy of the system")
    print(normalized_energy_system)
    edt_system = early_decay_time(impulse_response_system, sample_rate)
    print("EDT is the time taken for the energy to drop by 10 dB.")
    print(edt_system)
    clarity_80_system = clarity_index(impulse_response_system, sample_rate, time_ms=80)
    print("Clarity index 80 ms:")
    print(clarity_80_system)
    clarity_50_system = clarity_index(impulse_response_system, sample_rate, time_ms=50)
    print("Clarity index 50 ms:")
    print(clarity_50_system)
    def_index_system = definition_index(impulse_response_system, sample_rate)
    print("Ratio of early to total energy within 50 ms:")
    print(def_index_system)
    drr_system = direct_to_reverberant_ratio(impulse_response_system, sample_rate, direct_time_ms=5)
    print("Energy assumed reverberant:")
    print(drr_system)
    
    normalized_energy_filtered = schroeder_integration(impulse_response)
    print("normalized_energy of the filtered impulse response")
    print(normalized_energy_filtered)
    edt_filtered = early_decay_time(impulse_response, sample_rate)
    print("Filtered EDT is the time taken for the energy to drop by 10 dB of the filtered impulse response.")
    print(edt_filtered)
    clarity_80_filtered = clarity_index(impulse_response, sample_rate, time_ms=80)
    print("Filtered clarity index 80 ms:")
    print(clarity_80_filtered)
    clarity_50_filtered = clarity_index(impulse_response, sample_rate, time_ms=50)
    print("Filtered clarity index 50 ms:")
    print(clarity_50_filtered)
    def_index_filtered = definition_index(impulse_response, sample_rate)
    print("Filtered ratio of early to total energy within 50 ms:")
    print(def_index_filtered)
    drr_filtered = direct_to_reverberant_ratio(impulse_response, sample_rate, direct_time_ms=5)
    print("Remaining energy assumed reverberant for the filtered impulse response:")
    print(drr_filtered)

# Example usage with impulse response signal `ir` and sampling rate `fs`:
# edt = early_decay_time(ir, fs)
# c50 = clarity_index(ir, fs, 50)
# c80 = clarity_index(ir, fs, 80)
# d50 = definition_index(ir, fs)
# drr = direct_to_reverberant_ratio(ir, fs)
    
    
    sd.play(filtered_sig, samplerate=sample_rate)
    sd.wait()

    return impulse_response_file, filtered_signal_file
	
	

def main():
    # Parameters to adjust
    sample_rate = 48000
    duration_sec = 5  # Duration of one sweep in seconds
    start_freq = 20
    end_freq = 20000
    periods = 10
    
    # Define directory
    input_dir = 'C:/Users/Stefan/Documents/GitHub/Digital-Deconvolution-Audio-Filter-repo/data'

    # Generate periodic sine sweep
    periodic_sweep = generate_periodic_sine_sweep(start_freq, end_freq, duration_sec, sample_rate, periods)
    sweep_file = os.path.join(input_dir, 'x/periodic_sine_sweep.wav')
    sf.write(sweep_file, periodic_sweep, sample_rate)

    # Input and output files
    #input_file1 = os.path.join(input_dir, 'x/periodic_sine_sweep.wav')
    #input_file2 = os.path.join(input_dir, 'y/periodic_sine_sweep_beside_the_drums.wav')
    #test_file = os.path.join(input_dir, 'z/Record.wav')
    
    input_file1 = os.path.join(input_dir, 'x/periodic_sine_sweep.wav')
    input_file2 = os.path.join(input_dir, 'y/periodic_sine_sweep_Schreibtisch hinten im Eck 1.wav')
    #input_file2 = os.path.join(input_dir, 'y/periodic_sine_sweep_Schreibtisch hinten im Eck 2.wav')
    test_file = os.path.join(input_dir, 'z/speech.wav')
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(input_dir, f"output_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    impulse_response_file, filtered_signal_file = process_files(input_file1, input_file2, test_file, output_folder, sample_rate)

    print(f"Average impulse response saved to {impulse_response_file}")
    print(f"Filtered signal saved to {filtered_signal_file}")


if __name__ == "__main__":
    main()