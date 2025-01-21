import math

"""
Calculate the IQ metric based on the provided formula.
IQ = e^w * sqrt(r) * ln(s)

Parameters:
- w (float): Total average Player 2 win rate [0.0, 1.0]
- r (float): Average reward in the interval [1, 50]
- s (int): Current episode number (>= 2)

Returns:
- iq (float): Calculated IQ metric
"""

try:
    # Define the input lists
    w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    r = [-55, -25, 15, -5, 1, 5, 15, 25, 30, 40, 50]  # Includes invalid values for square root
    s = [100, 1000, 10000, 20000, 50000, 60000, 70000, 80000, 100000]
    
    # Compute the result for each combination of w, r, and s
    for wi in w:
        for ri in r:
            for si in s:
                if ri > 0:  # Ensure r is positive for sqrt
                    iq = math.exp(wi) * math.sqrt(ri) * math.log(si)
                    print(f"IQ for w={wi}, r={ri}, s={si}: {iq}")
                else:
                    print(f"Skipped invalid r={ri} for w={wi}, s={si}")
except (ValueError, OverflowError, ZeroDivisionError) as e:
    print(f"Error calculating IQ: {e}")
