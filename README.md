# o1-reasoning-tokens

## Overview

This repository contains code for analyzing the reasoning token counts returned by OpenAI's o1-mini model and experiments that test the effect of setting `max_completion_tokens` on reasoning tokens.

## Usage

### 1. Running the `max_completion_tokens` Experiment

To run the experiment that tests how `max_completion_tokens` affects reasoning token counts, use the following command:

```bash
python src/main.py
```

### 2. Analyzing o1-mini Reasoning Tokens from WildChat

To analyze reasoning token counts for OpenAI's o1-mini model from conversations collected by the WildChat chatbot, use the following command:

```bash
python src/analyze_wildchat_reasoning_tokens_o1mini.py
```

### 3. Analyzing o1-preview Reasoning Tokens from WildChat

To analyze reasoning token counts for OpenAI's o1-preview model from conversations collected by the WildChat chatbot, use the following command:

```bash
python src/analyze_wildchat_reasoning_tokens_o1preview.py
```
