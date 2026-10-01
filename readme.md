```markdown
# Self-Driving Lab Simulation

A simple software-only simulation of the closed-loop optimization workflow described in the paper **"What is a minimal working example for a self-driving laboratory?"**

## Goal

Find RGB values that produce a spectrum as close as possible to a target spectrum using Bayesian optimization.

## Workflow

```text
Initial Experiments
        ↓
Gaussian Process
        ↓
Acquisition Function
        ↓
Choose Next RGB
        ↓
Run Simulated Experiment
        ↓
Calculate Error
        ↓
Feed Result Back
        ↓
Repeat
```

## Tech

- Python
- NumPy
- scikit-learn
- Gaussian Process Regression
- Lower Confidence Bound acquisition function

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy scikit-learn
```

Run:

```bash
python simulator.py
```

## Current Simulation

The target spectrum is represented as:

```python
target = [0.7, 0.5, 0.3]
```

The simulator tries different RGB values and minimizes the error between the generated and target spectrum.

## Purpose

This project is mainly for understanding the software side of a self-driving laboratory before connecting the optimization loop to real hardware.
```