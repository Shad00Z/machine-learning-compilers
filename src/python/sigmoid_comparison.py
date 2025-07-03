"""
Sigmoid Function Comparison Plot

This script plots the fast sigmoid function, sigmoid Taylor approximation, 
and sigmoid linear interpolation function against the real sigmoid function.

The linear interpolation is defined with a lookup table over the range [-8, 8] with 0.5 steps.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def real_sigmoid(x):
    """Real sigmoid function: σ(x) = 1 / (1 + e^(-x))"""
    return 1.0 / (1.0 + np.exp(-x))

def fast_sigmoid(x):
    """Fast sigmoid approximation: f(x) = 0.5 * (x / (1 + abs(x)) + 1)"""
    return 0.5 * (x / (1 + np.abs(x)) + 1)

def taylor_sigmoid(x):
    """Taylor series approximation (5th order): f(x) = 0.5 + 0.25*x - 0.020833*x^3 + 0.002083*x^5"""
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    return 0.5 + 0.25 * x - 0.020833333 * x3 + 0.002083333 * x5

def interpolation_sigmoid(x):
    """Linear interpolation using lookup table with 33 values"""
    sig_table = np.array([
        0.000335, 0.000553, 0.000911, 0.001503, 0.002473, 0.004070, 0.006693,
        0.011109, 0.017986, 0.029312, 0.047426, 0.075858, 0.119203, 0.182426,
        0.268941, 0.377541, 0.500000, 0.622459, 0.731059, 0.817574, 0.880797,
        0.924142, 0.952574, 0.970688, 0.982014, 0.988891, 0.993307, 0.995930,
        0.997527, 0.998497, 0.999089, 0.999447, 0.999665
    ])
    
    # Create x values for the lookup table (32 intervals from -8 to 8)
    x_table = np.linspace(-8, 8, 33)
    
    # Create interpolation function
    interp_func = interp1d(x_table, 
                           sig_table, 
                           kind='linear', 
                           bounds_error=False, 
                           fill_value=(sig_table[0], sig_table[-1]))
    
    return interp_func(x)

def main():
    # Generate x values from -8 to 8 with 0.5 steps
    x = np.arange(-8, 8.5, 0.5)
    
    # Calculate all sigmoid functions
    y_real = real_sigmoid(x)
    y_fast = fast_sigmoid(x)
    y_taylor = taylor_sigmoid(x)
    y_interp = interpolation_sigmoid(x)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot functions
    plt.plot(x, y_real,   color='red',   linestyle='-',  linewidth=3, label='Real Sigmoid',         alpha=0.8)
    plt.plot(x, y_fast,   color='blue',  linestyle='--', linewidth=2, label='Fast Sigmoid',         alpha=0.8)
    plt.plot(x, y_taylor, color='green', linestyle=':',  linewidth=2, label='Taylor Approximation', alpha=0.8)
    plt.plot(x, y_interp, color='black', linestyle='-.', linewidth=2, label='Linear Interpolation', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('x',    fontsize=14)
    plt.ylabel('σ(x)', fontsize=14)
    plt.title('Sigmoid Function Comparison\nRange: [-8, 8]', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Axis limits
    plt.xlim(-8, 8)
    plt.ylim(-0.3, 1.3)
    
    # Styling
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('docs/_static/sigmoid_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'sigmoid_comparison.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main() 
