# 🏎️ F1TENTH Autonomous Racing Controllers using ROS 2 Humble

Implementation and benchmarking of multiple autonomous racing controllers in the **F1TENTH Gym Simulator** using **ROS 2 Humble Hawksbill**.

This project implements and evaluates four autonomous racing controllers with performance benchmarking based on lap time, steering stability, collision count, and path tracking accuracy.

---

## 📖 Project Overview

The following controllers were implemented and evaluated:

- 🚗 Wall Following Controller
- 🛣️ Pure Pursuit Controller
- 📍 Stanley Controller
- 🤖 Hybrid Controller (Pure Pursuit + Reactive Gap Following)

Each controller was benchmarked under identical simulation conditions to compare navigation performance and control behaviour.

---

# 🖥️ Development Environment

| Component | Version |
|------------|---------|
| Operating System | Ubuntu 22.04 LTS |
| ROS Distribution | **ROS 2 Humble Hawksbill** |
| Programming Language | Python 3 |
| Simulator | F1TENTH Gym ROS |
| Build System | Colcon |
| Container Platform | Docker |
| Visualization | RViz2 |

> **Note:** This project is developed and tested entirely on **ROS 2 Humble**.

---

# ✨ Features

- ROS 2 Humble based implementation
- Wall Following Controller
- Pure Pursuit Controller
- Stanley Controller
- Hybrid Controller
- LiDAR-based navigation
- Waypoint recording
- Speed profile generation
- Lap timing
- Collision detection
- Performance benchmarking

---

# 🚘 Implemented Controllers

## Wall Following Controller

A reactive LiDAR-based controller that maintains a desired distance from the wall using a PD controller.

**Highlights**

- Reactive navigation
- PD steering control
- Collision detection
- High-speed benchmarking

---

## Pure Pursuit Controller

A waypoint tracking controller that follows prerecorded racing waypoints using a geometric lookahead.

**Highlights**

- Waypoint tracking
- Smooth trajectory following
- Dynamic speed profile support

---

## Stanley Controller

A front-axle geometric controller that minimizes heading error and cross-track error.

**Highlights**

- Speed-adaptive steering
- Smooth path tracking
- Stable cornering

---

## Hybrid Controller

A combination of Pure Pursuit and Reactive Gap Following that switches to reactive navigation whenever obstacles are detected.

**Highlights**

- Path tracking
- Reactive obstacle avoidance
- Automatic controller switching

---

# 📊 Performance Comparison

| Controller | Best Average Lap Time |
|------------|----------------------:|
| Wall Follower | **44.57 s** |
| Stanley Controller | **59.82 s** |
| Hybrid Controller | **63.36 s** |
| Pure Pursuit | **70.00 s** |

---

# 🏆 Controller Ranking

🥇 Wall Follower — **44.57 s**

🥈 Stanley Controller — **59.82 s**

🥉 Hybrid Controller — **63.36 s**

4️⃣ Pure Pursuit — **70.00 s**

---

# ⚙️ Installation

Clone the simulator repository.

```bash
git clone https://github.com/f1tenth/f1tenth_gym_ros.git
```

Build the workspace.

```bash
cd /sim_ws

colcon build
```

Source ROS 2 Humble.

```bash
source /opt/ros/humble/setup.bash

source install/local_setup.bash
```

---

# 🚀 Launch the Simulator

```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

---

# ▶️ Run Controllers

### Wall Following

```bash
ros2 run wall_follower wall_follower_node
```

### Pure Pursuit

```bash
ros2 run pure_pursuit pure_pursuit_node
```

### Hybrid Controller

```bash
ros2 run pure_pursuit hybrid_controller
```

### Stanley Controller

```bash
ros2 run pure_pursuit stanley_controller
```

---

# 📍 Waypoint Recording

```bash
ros2 run pure_pursuit waypoint_recorder
```

---

# ⚡ Generate Speed Profile

```bash
python3 pure_pursuit/generate_speed_profile.py
```

---

# 🐳 Docker Commands

Launch the saved simulator container.

```bash
rocker --nvidia --x11 \
--volume .:/sim_ws/src/f1tenth_gym_ros \
-- f1tenth_saved_state
```

Copy packages from the container.

```bash
docker cp <container_id>:/sim_ws/src/pure_pursuit .

docker cp <container_id>:/sim_ws/src/wall_follower .
```

Save the Docker image.

```bash
docker commit <container_id> f1tenth_saved_state
```

---

# 📈 Results

- Successfully implemented four autonomous racing controllers.
- Benchmarked controller performance using lap time and stability metrics.
- Compared reactive and waypoint-based navigation approaches.
- Evaluated steering behaviour and oscillation characteristics at different speeds.

---

# ⚠️ Challenges

Advanced controllers including **LQR** and **Model Predictive Control (MPC)** were also explored. Despite extensive parameter tuning, both controllers exhibited significant oscillations and frequent wall collisions, and were therefore excluded from the final benchmark comparison.

---

# 🚀 Future Improvements

- Dynamic obstacle avoidance
- Improved speed optimization
- Stable LQR implementation
- MPC tuning
- Real-world F1TENTH deployment
- Adaptive controller switching

---
