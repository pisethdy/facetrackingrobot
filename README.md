# Intelligent Multi-Face Tracking Robot (AUPPBot)

**Student:** Piseth DY
**Course:** ICT 361 002 Introduction to Robotics

## Project Overview
This project is an autonomous mobile robot capable of detecting multiple faces in real-time and locking onto a specific target using a Raspberry Pi 4. It utilizes a **YOLOv8n-face** model for detection and a **Proportional Control Mixer** for smooth, organic movement.

## Features
* **Real-time Face Detection:** Runs at 8-12 FPS on Raspberry Pi 4 CPU.
* **Target Locking:** Ignores background distractors once a target is selected.
* **Smart Reacquisition:** Remembers the last known position (Left/Right) and spins to find the target if lost.
* **Smooth Control:** Uses a proportional mixer to blend turning and forward movement.
* **Cyberpunk HUD:** Web-based interface for real-time diagnostics.

## Hardware Requirements
* Raspberry Pi 4 (4GB or 8GB recommended)
* USB Webcam
* AUPPBot Chassis with DC Motors
* Motor Driver HAT

## Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/](https://github.com/)[YourUsername]/AUPPBot-Face-Tracker.git
   cd AUPPBot-Face-Tracker
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script:
   ```bash
   python main.py
   ```
2. Open your browser to:
   `http://localhost:5050` (or your Pi's IP address)

## License
Educational Project - 2025
