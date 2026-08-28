<!--
  PROJECT REPOSITORY README
  Save this file as README.md inside your project repo.

  This repo should contain ONLY the polebot_telemetry ROS 2 package
  (polebot_ws/src/polebot_telemetry/ in your workspace) — not the base
  platform packages, and not tongyi_canopen_driver.

  Before pushing actual code/data to a public repo, confirm with your
  supervisor(s) which parts are safe to publish.

  Remaining placeholder to fill: {{PAK_NURJAMIL_GITHUB_LINK}} in the
  Acknowledgments section, once you have it.
-->

# Data Historian Polebot AMR

Predictive analytics using ARIMA-XGBoost for an Autonomous Mobile Robot, using a condition-based switching mechanism validated on real hardware.

[![Paper](https://img.shields.io/badge/Paper-Jurnal%20Sinkron-blue?style=flat-square)](https://jurnal.polgan.ac.id/index.php/sinkron/article/view/16484)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

This repository contains only the **`polebot_telemetry`** ROS 2 package: the predictive analytics pipeline covering data collection, the ARIMA/XGBoost models, and the Condition-Based Temporal Switching (CBTS) mechanism.

It does **not** include the base Polebot AMR platform packages (`polebot_amr`, `polebot_amr_bringup`, `polebot_amr_controller`, `polebot_amr_description`, `polebot_amr_docking`, `polebot_amr_simulation`, `polebot_amr_system_tests`, `ros2_lsc`, `ros2_orbbec`, `ros2_roboteq`, `ros2_serial`) or the `tongyi_canopen_driver` motor driver package — these are prerequisites developed separately, required to run this package on the physical robot, but not part of this project's own contribution.

---

## Results (validated on real hardware)

| Variable | Model | sMAPE |
|---|---|---|
| Battery SOC (%) | ARIMA(2,1,3) | 0.35% |
| Motor Power (W) | XGBoost | 1.32% |

Motor power prediction achieved up to **136x lower error** compared to the time-series baseline. The CBTS mechanism was validated across 7 operational scenarios (111,115 clean data rows collected at 10 Hz).

---

## Tech Stack

![ROS2](https://img.shields.io/badge/ROS2-22314E?style=for-the-badge&logo=ros&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=for-the-badge)
![statsmodels](https://img.shields.io/badge/ARIMA-statsmodels-4C9A2A?style=for-the-badge)
![InfluxDB](https://img.shields.io/badge/InfluxDB-22ADF6?style=for-the-badge&logo=influxdb&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## 0. Quick Start

**Prerequisites:** ROS 2 Jazzy (Python 3.12) for live data collection. A running InfluxDB instance and the PZEM-017 power meter wired in are only needed on the physical robot.

There are two ways to use this repository:

- **Analysis only (no ROS 2 / hardware needed):**
```bash
git clone https://github.com/<username>/polebot_telemetry.git
cd polebot_telemetry
pip install -r requirements.txt
python3 polebot_telemetry/arima_predictor_hw.py   # example
```
- **Live data collection (requires the physical robot):** this package must sit inside a ROS 2 Jazzy workspace's `src/` folder, alongside the `tongyi_canopen_driver` package (see Acknowledgments):
```bash
cd ~/your_ros2_ws/src
git clone https://github.com/<username>/polebot_telemetry.git
cd ..
source /opt/ros/jazzy/setup.bash
colcon build --packages-select polebot_telemetry
source install/setup.bash
```

---

## 1. Directory Structure

```
polebot_telemetry/                          <-- repo root = ROS 2 ament_python package
├── polebot_telemetry/                      <-- Python module (package source)
│   ├── telemetry_logger.py                 <-- ETL: ROS 2 topics -> InfluxDB [ros2 run]
│   ├── influx_bridge.py                    <-- InfluxDB read/write bridge [ros2 run]
│   ├── system_recorder.py                  [ros2 run]
│   ├── scenario_runner.py                  <-- C1-C7 scenario sequencer [ros2 run]
│   ├── scenario_runner2.py                 <-- newer variant (run via python3)
│   ├── run_all_scenarios.sh / run_all_scenarios_hw.sh
│   ├── pzem_publisher.py, raw_pzem017.py, scan_pzem017.py,
│   │   test_pzem017.py, set_shunt50a.py    <-- PZEM-017 power meter interface
│   ├── arima_predictor.py / arima_predictor_hw.py
│   ├── xgboost_predictor.py / xgboost_predictor_hw.py
│   ├── hybrid_switching_predictor.py / hybrid_switching_predictor_hw.py   <-- CBTS core logic
│   ├── realtime_predictor_node.py
│   ├── comparison_plot.py / comparison_plot_hw.py
│   ├── hybrid_chart_split.py / hybrid_chart_split_hw.py
│   ├── hybrid_cycle_annotation.py / hybrid_cycle_annotation_hw.py
│   ├── export_dataset.py / export_dataset_hw.py
│   ├── battery_sim.py [ros2 run] / battery_physics_sim.py [ros2 run: battery_physics]
│   ├── energy_path_predictor.py [ros2 run]
│   ├── path_planning_node.py [ros2 run]
│   └── __init__.py
├── launch/
│   └── polebot_sim.launch.py
├── urdf/
│   └── polebot.urdf.xacro
├── worlds/
│   └── depot.sdf, polman_workshop.sdf, warehouse.sdf
├── meshes/                                 <-- robot 3D model (.stl, CAD source)
├── resource/
├── test/
│   └── test_copyright.py, test_flake8.py, test_pep257.py
├── package.xml
├── setup.py
├── setup.cfg
├── requirements.txt
└── README.md
```

<!-- results/ folder with exported CSVs, metrics JSON, and plots referenced above
     is not yet part of this tree — add it once you decide what to publish -->

---

## 2. Execution Chain (live hardware run)

```
Sensors (motor encoders via tongyi_canopen_driver, PZEM-017 current/voltage)
        |
        |-- ROS 2 topics: /odom, /joint_states, /polebot/battery_status  (10 Hz)
        v
telemetry_logger.py   (ETL: ROS 2 -> InfluxDB)
        |
        v
InfluxDB (bucket: polebot_data)
        |
        +--> arima_predictor_hw.py, xgboost_predictor_hw.py     (offline training / backtesting)
        v
hybrid_switching_predictor_hw.py    (CBTS: condition classification + model routing)
        |
        v
Grafana  (dashboard visualization)
```

---

## 3. Build

This is a ROS 2 **ament_python** package and depends on `rclpy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `std_msgs`, and `message_filters` — all part of a standard ROS 2 Jazzy desktop install.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/your_ros2_ws
colcon build --packages-select polebot_telemetry
source install/setup.bash
```

Python dependencies (not managed by colcon):
```bash
pip install -r requirements.txt
```

---

## 4. Running

Scripts registered as ROS 2 executables (run with `ros2 run` after building):

```bash
# Terminal 1 - start telemetry logging (requires InfluxDB running)
ros2 run polebot_telemetry telemetry_logger

# Terminal 2 - run a scenario
ros2 run polebot_telemetry scenario_runner --ros-args -p scenario:=1
```

Available `ros2 run polebot_telemetry <name>` executables: `telemetry_logger`, `influx_bridge`, `system_recorder`, `scenario_runner`, `battery_sim`, `battery_physics`, `path_planning_node`, `energy_path_predictor`.

All other scripts (models, analysis, PZEM-017 utilities, `scenario_runner2.py`, `realtime_predictor_node.py`) are **not** registered as ROS 2 executables — run them directly from the repo root:

```bash
python3 polebot_telemetry/hybrid_switching_predictor_hw.py
```

CSV export after collection:
```bash
python3 polebot_telemetry/export_dataset_hw.py
```

---

## 5. Calibrated System Parameters (do not change without recalibration)

| Parameter | Value | Notes |
|---|---|---|
| `WHEEL_RADIUS` | 0.078 m | |
| `WHEEL_MASS` | 3.610 kg | |
| `ROBOT_MASS` | 100.845 kg | |
| `I_WHEEL` | 0.01098 kg·m² | wheel moment of inertia |
| `MU_KINETIC` | 0.02 | kinetic friction coefficient |
| `P_RATED` | 2000 W | rated motor power |
| Battery | 48V / 32Ah / 1536 Wh | |
| Peukert `k` | 1.1 | battery discharge correction |
| PZEM current threshold | 2.0 A | static/dynamic condition classification |
| Best ARIMA order | (2,1,3) | AIC = -33,655.47 |

---

## 6. Data Analysis

```bash
python3 polebot_telemetry/comparison_plot_hw.py
python3 polebot_telemetry/hybrid_chart_split_hw.py
python3 polebot_telemetry/hybrid_cycle_annotation_hw.py
```

Key CSV columns analyzed: `linear_velocity_ms`, `motor_power_total_W`, `battery_SOC_percent`, `motor_load_ratio`, `battery_current_A`, `condition_class`.

---

## 7. Simulation vs Hardware Scripts

Scripts without an `_hw` suffix (e.g. `arima_predictor.py`, `export_dataset.py`, `battery_sim.py`, `battery_physics_sim.py`, `comparison_plot.py`) were used during the earlier Gazebo simulation phase. Scripts with the `_hw` suffix are the hardware-validated versions used for the final results reported in the paper — use the `_hw` versions unless you're specifically working with simulated data.

---

## Publication

This work is documented in a peer-reviewed paper:

> Anggraeni, P., Candra, W. A., Jatnika, S. D., Lilansa, N., Sunarya, A. S., Ramadhan, N. J., & Wiyono, A. (2026). *Predictive Analytics for Energy Consumption of Autonomous Mobile Robot Using Hybrid ARIMA-XGBoost*. Jurnal Sinkron (SINTA 3), 10(3).
> [Read the paper](https://jurnal.polgan.ac.id/index.php/sinkron/article/view/16484)

## Author & Acknowledgments

**Surya Dharma Jatnika** — Department of Automation Engineering, Politeknik Manufaktur Bandung

Supervised by Dr. Eng. Pipit Anggraeni and Wahyu Adhie Candra, S.T., M.Sc.

This project runs on top of the Polebot AMR motion control platform (including the `tongyi_canopen_driver` motor driver package) developed by Pak Nurjamil ([nj-ramadhan/polman-mbd-ros2-polebot-amr](https://github.com/nj-ramadhan/polman-mbd-ros2-polebot-amr)). This repository does not include or modify that codebase — it only consumes the ROS 2 topics it publishes.

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and distribute as long as the copyright notice is retained.
