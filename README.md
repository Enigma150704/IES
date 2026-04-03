# KKC Signal Simulation & Dashboard

This repository contains an industrial signal simulator built with SimPy and a modern visualization dashboard using FastAPI.

## Setup Instructions

### 1. Prerequisite
- Python 3.9+
- pip

### 2. Install Dependencies
Run the following command to install all necessary packages for the simulation engine and the dashboard:
```bash
pip install -r requirements.txt
```

## Running the Simulation
To generate data for the dashboard:
```bash
# Run all 21 scenarios (Standard deterministic)
python run_simulation.py --mode all --db simulation_data.db

# Run RL-driven data generation (e.g. 200k samples)
python run_simulation.py --mode rl --samples 200000 --db rl_generated_200k.db
```

## Running the Dashboard
The dashboard allows you to visualize and analyze simulation results:
```bash
# Start the dashboard on default port (8000)
python dashboard/app.py

# Or specify a custom port
python dashboard/app.py --port 8001
```
Open your browser at: `http://127.0.0.1:8000` (or the port you specified).

## Project Structure
- `module0_sqm/`: Signal Quality Monitor detector logic.
- `module1_simpy/`: SimPy engine for 21 standard scenarios.
- `rl_env/`: RL Gymnasium environment for diverse data generation.
- `dashboard/`: FastAPI/JS web interface for visualizing results.
- `*.db`: SQLite data stores.

---
**Note**: The dashboard includes a "Fallback Mode" to visualize databases that may not have full metadata recorded.
