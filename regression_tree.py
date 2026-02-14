import numpy as np
import pandas as pd

Dosage = [2, 4, 6, 8, 11, 13, 16, 19, 22, 25, 26, 27, 28, 29, 32, 34, 36, 38, 40]
Effectiveness = [0, 0, 0, 0, 5, 17, 100, 100, 100, 100, 65, 60, 55, 50, 45, 13, 0, 0, 0]

df = pd.DataFrame({'Dosage': Dosage, 'Effectiveness': Effectiveness})
df = df.sort_values(by='Dosage').reset_index(drop=True)

# Get potential split points from the feature
unique_dosages = df['Dosage'].unique()
potential_splits = (unique_dosages[:-1] + unique_dosages[1:]) / 2

ssrs = []

print("Calculating Sum of Squared Residuals (SSR) for each potential split:")
# Calculate SSR for each potential split
for split in potential_splits:
    left_group = df[df['Dosage'] <= split]
    right_group = df[df['Dosage'] > split]
    
    # This check is good practice but unlikely to fail here.
    if left_group.empty or right_group.empty:
        continue

    # For a regression tree, the prediction for a region is the mean of the target values.
    mean_left = left_group['Effectiveness'].mean()
    mean_right = right_group['Effectiveness'].mean()
    
    # Calculate Sum of Squared Residuals (SSR) for each group
    # SSR = sum of (actual - predicted)^2
    ssr_left = np.sum((left_group['Effectiveness'] - mean_left)**2)
    ssr_right = np.sum((right_group['Effectiveness'] - mean_right)**2)
    
    # Total SSR for this split is the sum of SSRs of the two resulting groups.
    total_ssr = ssr_left + ssr_right
    ssrs.append(total_ssr)
    
    print(f"Split at Dosage = {split:<5.2f}, Left Mean = {mean_left:<5.2f}, Right Mean = {mean_right:<5.2f}, Total SSR = {total_ssr:.2f}")

# Find the best split (the one with the minimum SSR)
if ssrs:
    best_ssr = min(ssrs)
    best_split_index = np.argmin(ssrs)
    best_split = potential_splits[best_split_index]

    print(f"\nBest split is at Dosage = {best_split:.2f} with a minimum SSR of {best_ssr:.2f}")

#TO-DO; define a function called "Root Node" where it takes user input data and find the root node for the datasets.