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

### Reproducing the Figures
All the `log` and `predictions` are included in this repository. You can run the following script to reproduce all the `figure` in the paper:

```bash
cd script
python3 plot.py
```

### Running the Experiments
To reproduce the main experiments, you can run the following scripts:

- Main Experiment:
```bash
bash lost_in_the_distance.sh
bash lost_in_the_distance_sonnet.sh
bash no_distance_no_degradation.sh
```

- Ablation Study:
```bash
bash ablation_study_noise.sh
bash ablation_study_task.sh
```

### Note for the args corresponding to the ones in our paper:


- `revname`: Name2Description
- `revcause`: Cause2Effect
- `revparent`: Parent2Child
- `qa`: AB
- `qna`: ANB
- `qnna`: ANNB
- `qnnna`: ANNNB