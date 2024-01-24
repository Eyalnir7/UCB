import numpy as np
import pandas as pd
from SMPyBandits.Environment.MAB import MAB
from SMPyBandits.Policies import SWUCB

# Number of arms
nb_arms = 2

# True means for each arm (deterministic in this case)
means = [0.5, 0.0]

# Create the multi-armed bandit environment with Deterministic arms
arms = [MAB({'arm_type': 'Deterministic', 'parameters': [mean]}) for mean in means]

# Set up parameters for SWUCB
horizon = 10000
radius_function = lambda t, n: np.sqrt((2 * np.log(t)) / n)

# Run simulations for each window size from 1 to 10000
results = {'Window_Size': [], 'Rounds_Arm2_Picked': []}

for window_size in range(1, 10001):
    swucb = SWUCB(nb_arms, window_size, radius_function)
    rounds_arm2_picked = []

    for t in range(1, horizon + 1):
        chosen_arm = swucb.select_arm()
        reward = arms[chosen_arm].sample(1)[0]
        swucb.update(chosen_arm, reward)

        if chosen_arm == 1:  # Arm 2 picked
            rounds_arm2_picked.append(t)

    # Save results for this window size
    results['Window_Size'].append(window_size)
    results['Rounds_Arm2_Picked'].append(rounds_arm2_picked)

# Convert results to a Pandas DataFrame
df_results = pd.DataFrame(results)

# Save results to a CSV file
df_results.to_csv('swucb_simulations_results.csv', index=False)