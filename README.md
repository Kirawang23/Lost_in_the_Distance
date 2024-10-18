# Lost in the Distance

This repository contains the code and data for the paper:  
**Lost in the Distance: Large Language Models Struggle to Capture Long-Distance Relational Knowledge**

## Overview

We expose the "Lost in the Distance" phenomenon, where the performance of large language models in capturing relational knowledge degrades significantly when the relational information is separated by noise, i.e., unrelated sentences that interfere with solving the task.

## How to Run

### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.10.11 or above
- Required Python packages (can be installed via `requirements.txt`).

### Running the Experiments

To reproduce the main experiments, you can run the following scripts:


1. **Main Experiment:**
   ```bash
   bash lost_in_the_distance.sh
   ```
   ```bash
   bash lost_in_the_distance_sonnet.sh
   ```
   ```bash
   bash no_distance_no_degradation.sh
   ```

2. **Ablation Study:**
    ```bash
   bash lablation_study_noise.sh
   ```
   ```bash
   bash ablation_study_task.sh
   ```
