# Volleyball Trajectory Simulator

A Python application to simulate and analyze volleyball trajectories using physics and video analysis.

## Features

*   **Physics Simulation**: Calculates ball trajectory using Euler's method, considering gravity and drag.
*   **Video Analysis**: Analyze slow-motion videos to track ball movement and extract real-world coordinates.
*   **Interactive GUI**: Built with NiceGUI for easy parameter adjustment and visualization.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Harelzx/Physics.git
    cd Physics
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main application:
```bash
python3 main.py
```

### Simulation
Adjust sliders for Serve Height, Distance, Speed, Angle, and Drag Coefficient to see the predicted trajectory.

### Video Analysis
1.  Go to the "Video Analysis" tab.
2.  Upload your video.
3.  Calibrate the scale by clicking two points and entering the real-world distance.
4.  Set the origin (0,0).
5.  Select the ball to start auto-tracking.

## Requirements
*   Python 3.x
*   nicegui
*   matplotlib
*   numpy
*   opencv-python
