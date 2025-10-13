#!/bin/bash

# Activate Conda environment
source activate redesign-aging-env

# Run simulations
cd SimulationCode
python simulation.py scenarios.json

# process results and save to CSV
cd ..
mkdir -p Results
cd SimulationCode
python process_results.py "simulation_results" scenarios.json "../Results/redesigning_aging_results.csv"