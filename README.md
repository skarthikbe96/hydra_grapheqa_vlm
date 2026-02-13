# GraphEQA-ROS2

## Graph-based Embodied Question Answering with Hydra + ROS2 Jazzy + Nav2

This repository adapts **GraphEQA (Graph-based Embodied Question
Answering)** to a real ROS2 robotic system using:

-   ROS2 Jazzy\
-   Gazebo Harmonic\
-   Hydra (Dynamic Scene Graph backend)\
-   Nav2\
-   Vision-Language Models (GPT / Gemini / Claude / LLaMA)

It replaces the original Habitat/Stretch simulation environment with a
ROS2-native pipeline and integrates Hydra's Dynamic Scene Graph directly
from ROS topics.

------------------------------------------------------------------------

# 🚀 Overview

GraphEQA originally operates inside Habitat using a Hydra Python
pipeline.\
This project ports the **VLM planning component** into a real robotic
ROS2 stack.

It connects to:

-   `/hydra/backend/dsg` → Hydra scene graph\
-   `/camera_pan_tilt/image` → Robot RGB stream\
-   `/map` → OccupancyGrid for frontier detection\
-   `navigate_to_pose` → Nav2 exploration



---

# 📦 Core Components

## 1️⃣ dsg_bridge_node
- Subscribes to `/hydra/backend/dsg`
- Converts Hydra Dynamic Scene Graph into GraphEQA-compatible JSON
- Injects agent pose from TF (`map → base_footprint`)

## 2️⃣ SceneGraphRosAdapter
- Mimics GraphEQA’s `SceneGraphSim` interface
- Stores DSG as a NetworkX graph
- Adds frontier nodes
- Connects frontiers to nearby objects
- Provides:
  - `scene_graph_str`
  - `get_position_from_id()`
  - `get_current_semantic_state_str()`

## 3️⃣ frontier_node
- Subscribes to `/map`
- Detects frontier clusters (free cells adjacent to unknown)
- Publishes frontier centroids in `map` frame

## 4️⃣ visual_memory_node
- Subscribes to `/camera_pan_tilt/image`
- Saves periodic frames to disk
- Provides latest image for VLM context

## 5️⃣ vlm_planner_node
- Wraps original GraphEQA VLM planners
- Inputs:
  - Scene graph JSON
  - Latest camera image
  - Frontier nodes
- Outputs:
  - Answer token
  - Confidence score
  - Action decision

## 6️⃣ executor_node
- Subscribes to planner decision
- Converts action into Nav2 `NavigateToPose`
- Sends goals in `map` frame

---

# ✨ Features

- ✔ Real-time Hydra integration (no Habitat required)
- ✔ Frontier-based semantic exploration
- ✔ ROS2-native implementation
- ✔ Nav2 integration
- ✔ Compatible with multiple VLM backends
- ✔ Minimal modification to original GraphEQA planner logic
- ✔ TF-based pose alignment
- ✔ Gazebo Harmonic compatible

---

# 🔧 Dependencies

### ROS
- ROS2 Jazzy
- Nav2
- Hydra ROS backend
- SLAM (publishing `/map`)

### Python
- networkx
- numpy
- opencv-python
- pydantic
- openai (or other VLM SDK)
- cv_bridge

Install Python dependencies:

```bash
pip install networkx numpy opencv-python pydantic openai
'''
Install cv_bridge:

sudo apt install ros-jazzy-cv-bridge

📥 Installation
cd ~/ros2_ws/src
git clone <your_repo_url> grapheqa_ros2
cd ..
colcon build --symlink-install
source install/setup.bash

▶ Running

Make sure the following are already running:

Hydra backend

SLAM publishing /map

Nav2 stack

Camera publishing /camera_pan_tilt/image

Launch:

ros2 launch grapheqa_ros2 grapheqa.launch.py


Monitor planner output:

ros2 topic echo /grapheqa/decision


Monitor scene graph snapshots:

ros2 topic echo /grapheqa/dsg_snapshot

🔄 Planning Loop

Hydra updates the scene graph

DSG bridge converts it to JSON

Frontier detector computes exploration candidates

SceneGraphRosAdapter enriches the graph

VLM planner reasons over:

scene graph

agent state

recent image

Planner outputs:

confident answer → stop

goto object → Nav2

goto frontier → Nav2

Loop continues until confidence threshold met

The robot is capable of:

1.  Building a semantic 3D scene graph (Hydra)\
2.  Detecting semantic exploration frontiers\
3.  Selecting task-relevant visual memory\
4.  Querying a Vision-Language Model\
5.  Deciding:
    -   `Goto_Object`
    -   `Goto_Frontier`
    -   or terminate with a confident answer\
6.  Executing navigation using Nav2

------------------------------------------------------------------------

# 🧠 System Architecture

Hydra DSG → dsg_bridge_node → SceneGraphRosAdapter\
OccupancyGrid → frontier_node\
Camera RGB → visual_memory_node\
Scene Graph + Image → vlm_planner_node → executor_node → Nav2

------------------------------------------------------------------------

# ✨ Features

-   Real-time Hydra integration (no Habitat required)\
-   Frontier-based semantic exploration\
-   ROS2-native implementation\
-   Nav2 integration\
-   Compatible with multiple VLM backends\
-   Minimal modification to original GraphEQA planner logic\
-   TF-based pose alignment\
-   Gazebo Harmonic compatible

------------------------------------------------------------------------

# 📚 Research References

This project builds upon the following works:

## 1. GraphEQA

Saumya Saxena, Blake Buchanan, Chris Paxton, Bingqing Chen, Narunas
Vaskevicius, Luigi Palmieri, Jonathan Francis, and Oliver Kroemer.

**GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied
Question Answering**\
arXiv:2412.14480, 2024.\
https://arxiv.org/abs/2412.14480

``` bibtex
@misc{saxena2024grapheqausing3dsemantic,
      title={GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering}, 
      author={Saumya Saxena and Blake Buchanan and Chris Paxton and Bingqing Chen and Narunas Vaskevicius and Luigi Palmieri and Jonathan Francis and Oliver Kroemer},
      year={2024},
      eprint={2412.14480},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.14480}, 
}
```

------------------------------------------------------------------------

## 2. Hydra / Hierarchical Scene Graphs

Nathan Hughes, Yun Chang, Siyi Hu, Rajat Talak, Rumaia Abdulhai, Jared
Strader, and Luca Carlone.

**Foundations of Spatial Perception for Robotics: Hierarchical
Representations and Real-Time Systems**\
The International Journal of Robotics Research, 43(10), 1457--1505,
2024.

``` bibtex
@article{hughes2024foundations,
  title={Foundations of spatial perception for robotics: Hierarchical representations and real-time systems},
  author={Hughes, Nathan and Chang, Yun and Hu, Siyi and Talak, Rajat and Abdulhai, Rumaia and Strader, Jared and Carlone, Luca},
  journal={The International Journal of Robotics Research},
  volume={43},
  number={10},
  pages={1457--1505},
  year={2024},
  publisher={SAGE Publications Sage UK: London, England}
}
```

------------------------------------------------------------------------

# 📜 License

Original GraphEQA components follow their respective license.\
ROS2 adaptation code released under Apache 2.0.