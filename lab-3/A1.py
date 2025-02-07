import numpy as np
import matplotlib.pyplot as plt

def create_signal():
    """
    Creates the input signal for convolution.
    Returns:
        list: Signal values representing a periodic pattern
    """
    return [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]

def define_filters():
    """
    Defines the low-pass and high-pass filters.
    Returns:
        tuple: (low_pass_filter, high_pass_filter)
    """
    # Low-pass filter: Smooths the signal by weighted averaging of neighboring samples
    h_l = [0.05, 0.2, 0.5, 0.2, 0.05]
    
    # High-pass filter: Detects rapid changes/edges in the signal
    h_h = [-1, 2, -1]
    
    return h_l, h_h

def apply_filters(signal, low_pass, high_pass):
    """
    Applies convolution with both filters to the input signal.
    Args:
        signal (list): Input signal
        low_pass (list): Low-pass filter coefficients
        high_pass (list): High-pass filter coefficients
    Returns:
        tuple: (low_pass_output, high_pass_output)
    """
    y_low = np.convolve(signal, low_pass)
    y_high = np.convolve(signal, high_pass)
    return y_low, y_high

def plot_results(signal, y_low, y_high):
    """
    Creates and saves a plot comparing original and filtered signals.
    Args:
        signal (list): Original input signal
        y_low (array): Low-pass filtered signal
        y_high (array): High-pass filtered signal
    """
    plt.figure(figsize=(12, 6))
    
    # Plot all signals
    plt.plot(signal, label='Original Signal', marker='o')
    plt.plot(y_low, label='Low-pass Filtered', marker='.')
    plt.plot(y_high, label='High-pass Filtered', marker='.')
    
    # Add plot decorations
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Signal Convolution with Low-pass and High-pass Filters')
    plt.legend()
    plt.grid(True)
    
    # Save plot to file instead of showing it
    plt.savefig('convolution_results.png')
    plt.close()

def main():
    """
    Main function to run the convolution demonstration.
    """
    # Get signal and filter definitions
    signal = create_signal()
    low_pass, high_pass = define_filters()
    
    # Apply filters
    y_low, y_high = apply_filters(signal, low_pass, high_pass)
    
    # Display results
    plot_results(signal, y_low, y_high)

if __name__ == "__main__":
    main()
