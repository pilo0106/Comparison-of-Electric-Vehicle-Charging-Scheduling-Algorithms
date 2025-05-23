# Comparison of Electric Vehicle Charging Scheduling Algorithms

This repository contains the implementation and analysis of four electric vehicle (EV) charging scheduling algorithms developed for our graduation project. The goal is to evaluate and compare their performance in optimizing EV charging station scheduling, reducing average wait times, and increasing station utilization efficiency.

## Project Overview

With the growing adoption of electric vehicles, effective charging station scheduling has become a critical challenge. This project explores four different scheduling strategies:

- **EFT-Based (Earliest Finish Time)**
- **EST-Based (Earliest Start Time)**
- **PSO-Based (Particle Swarm Optimization)**
- **DQN-Based (Deep Q-Network)**

Each algorithm is tested and analyzed using simulated vehicle and station data to identify strengths and weaknesses in different traffic conditions.

## Repository Structure
```bash
Comparison-of-Electric-Vehicle-Charging-Scheduling-Algorithms
├── EFT.py # Earliest Finish Time scheduling
├── EST.py # Earliest Start Time scheduling
├── PSO.py # Particle Swarm Optimization-based scheduling
├── DQN.py # Deep Q-Network-based scheduling
└── README.md
```

## Simulation Settings

Each EV is randomly assigned:

- **Battery Capacity**: 50 to 95 kWh  
- **Expected Charging Duration**: 30 to 60 minutes  
- **Start Time**: Between 0 and 1200 minutes

### Charging Stations

- Charging rates: `200, 350, 150, 360, 100 kW`
- Piles per station: `[5, 2, 4, 3, 4]`

### Charging Time Calculation

charging_time = battery_needed / charging_rate

- Time unit: **minutes**

## Evaluation Metrics

- Average vehicle wait time  
- Station pile utilization  
- Number of successfully scheduled vehicles  
- Total scheduling time  

Final results and visualizations are displayed at the end of each script execution.

## Results Summary
DQN achieves the lowest wait time after training (~100,000 episodes), but requires extensive training.

PSO balances performance and execution time.

EFT and EST are simpler and faster but may result in more waiting under high load conditions.
