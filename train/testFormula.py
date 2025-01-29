import math

def calculate_iq(w, r, s, q):
    """
    Simple local IQ formula:
       IQ = e^(w) * sqrt(r/2) * s^q
    If s <= pre_episode, we clamp s=pre_episode.
    """
    try:
        if r <= 0:
            print(f"Average Reward <= 0: {r}, using 0.0001 instead")
            r = 0.0001
        base = math.sqrt(r) * math.log(s)
        factor_q = math.pow(w, q)

        return base * factor_q
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        print(f"Error in calculate_iq: {e}")
        return None

# Define input ranges
w_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
r_range = list(range(-50, 51))  # Rewards from -50 to 50
s_range = range(0, 100001)  # Episodes from 0 to 100,000

# Initialize counters for iteration control
w_idx, q_idx, r_idx = 0, 0, 0
w, q, r = w_list[w_idx], q_list[q_idx], r_range[0]
iq_total = 0

# Perform calculations
for s in s_range:
    # Update `r` every 1,000 iterations
    if s % 1000 == 0 and r_idx < len(r_range) - 1:
        r_idx += 1
        r = r_range[r_idx]
        if r < 0:
            r = 0.0001  # Ensure valid positive value for sqrt

    # Update `w` and `q` every 10,000 iterations
    if s % 10000 == 0 and s != 0:
        if w_idx < len(w_list) - 1:
            w_idx += 1
        if q_idx < len(q_list) - 1:
            q_idx += 1
        w, q = w_list[w_idx], q_list[q_idx]

    # Calculate IQ
    iq = calculate_iq(w, r, s, q)
    if iq is not None:
        # Normalize IQ for better scaling
        iq_normalized = iq / 1000
        iq_total += iq_normalized
        print(f"IQ for w={w}, r={r}, s={s}, q={q}: {iq_total:.5f}")
    else:
        print(f"Skipped invalid calculation for w={w}, r={r}, s={s}, q={q}")
