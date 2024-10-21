import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import time
import torch

# Initialize OpenAI client
client = OpenAI()

# Store data for analysis
tokens_no_limit = []
tokens_max_193 = []

# Fixed test prompt
prompt = "Calculate the product of 100 and 57."

# Function to query the OpenAI API and measure reasoning tokens
def query_openai(prompt, max_completion_tokens=None):
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            model="o1-mini",
            max_completion_tokens=max_completion_tokens
        )
        latency = time.time() - start_time
        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
        output_text = response.choices[0].message.content
        return reasoning_tokens, latency, output_text
    except Exception as e:
        print(f"Error with prompt: {prompt}\nError: {e}")
        return None, None, None

# Number of test cases to run for each configuration
num_tests = 300

d = torch.load('tokens_limit.pt')
tokens_no_limit = d['tokens_no_limit']
tokens_max_193 = d['tokens_max_193']

GENERATE = True
GENERATE = False

if GENERATE:
    # Run the tests for each configuration
    for i in range(num_tests):
        # Test without max_completion_tokens
        reasoning_tokens, latency, output = query_openai(prompt)
        if reasoning_tokens is not None:
            tokens_no_limit.append(reasoning_tokens)
    
        # Test with max_completion_tokens=193
        reasoning_tokens, latency, output = query_openai(prompt, max_completion_tokens=193)
        if reasoning_tokens is not None:
            tokens_max_193.append(reasoning_tokens)
    
    torch.save({'tokens_no_limit': tokens_no_limit, 'tokens_max_193': tokens_max_193}, 'tokens_limit.pt')
else:
    d = torch.load('tokens_limit.pt')
    tokens_no_limit = d['tokens_no_limit']
    tokens_max_193 = d['tokens_max_193']

# Function to plot the frequency distribution for different configurations
def plot_frequency_distribution(tokens_list_1, tokens_list_3):
    tokens_list_1 = [item for item in tokens_list_1 if item <= 256]
    tokens_list_3 = [item for item in tokens_list_3 if item <= 256]
    plt.figure(figsize=(10, 6))

    # Get unique token values and their frequencies
    unique_values_1, counts_1 = np.unique(tokens_list_1, return_counts=True)
    unique_values_3, counts_3 = np.unique(tokens_list_3, return_counts=True)
    #import pdb; pdb.set_trace()

    # Plot for no max_completion_tokens
    sns.scatterplot(x=unique_values_1, y=counts_1, s=200, marker='s', color='blue', label='No Limit')

    # Plot for max_completion_tokens=193
    sns.scatterplot(x=unique_values_3, y=counts_3, s=200, marker='o', color='red', label='Max Completion Tokens = 193')

    # Add dashed vertical lines at multiples of 64
    for x in range(0, max(max(unique_values_1), max(unique_values_3)) + 1, 64):
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)

    # Add text labels to each point
    for i in range(len(unique_values_1)):
        plt.text(unique_values_1[i] + 3, counts_1[i] + 0.5, str(unique_values_1[i]), fontsize=10, ha='left')
    for i in range(len(unique_values_3)):
        plt.text(unique_values_3[i] + 3, counts_3[i] + 0.5, str(unique_values_3[i]), fontsize=10, ha='left')

    # Add title, labels, and legend
    plt.title("Effect of max_completion_tokens on Reasoning Token Distribution", fontsize=16)
    #plt.xlim(plt.xlim()[0], 251)
    plt.xlabel("Reasoning Token Count", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig('reasoning_token_distribution.png')

# Plot the combined frequency distributions for the two configurations
plot_frequency_distribution(tokens_no_limit, tokens_max_193)
