import json
import os
from flask import Flask, request, jsonify
import mysql.connector
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

sns.set(style="white", palette="muted")

# Function to plot the frequency distribution
def plot_frequency_distribution(tokens_list, a):
    unique_values, counts = np.unique(tokens_list, return_counts=True)

    plt.figure(figsize=(10, 6))  # Adjust the figure size
    multiples_of_64 = [x for x in unique_values if x % 64 == 0]
    non_multiples_of_64 = [x for x in unique_values if x % 64 != 0]

    # Plot multiples of 64
    sns.scatterplot(x=[x for x in multiples_of_64], y=[counts[unique_values.tolist().index(x)] for x in multiples_of_64],
                    s=100, marker='o', color='red', label='Multiples of 64')

    # Plot non-multiples of 64
    sns.scatterplot(x=[x for x in non_multiples_of_64], y=[counts[unique_values.tolist().index(x)] for x in non_multiples_of_64],
                    s=100, marker='s', color='blue', label='Non-multiples of 64')


    # Add dashed vertical lines at multiples of 64
    for x in range(0, max(unique_values)+1, 64):
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)

    # Add labels to each point
    for i in range(len(unique_values)):
        plt.text(unique_values[i] + 6, counts[i] + 60, str(unique_values[i]), fontsize=10, ha='left')

    # Add title, labels, and legend
    plt.title("Distribution of Reasoning Tokens for o1-preview (Dashed Lines at Multiples of 64)", fontsize=16)
    plt.xlabel("Reasoning Token Count", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    plt.tight_layout()  # Adjust plot to fit the figure area nicely
    plt.savefig(f'counts_o1preview_wildchat.png')

def main():
    reasoning_tokens_all = torch.load('data/reasoning_tokens_o1preview.pt')
    reasoning_tokens_all = [item for item in reasoning_tokens_all if item < 1024]
    print (len(reasoning_tokens_all))
    plot_frequency_distribution(reasoning_tokens_all, len(reasoning_tokens_all))

if __name__ == "__main__":
    main()
