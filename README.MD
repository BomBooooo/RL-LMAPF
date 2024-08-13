<div align="center">

**Lifelong Pathfinding for Thousands of Agents in One Second via Fast GPU-based Distance Costmap Algorithm and Reinforcement Learning**

</div> 

## Abstract
This paper proposes Real-time Large-scale Method for LMAPF (RL-LMAPF), a scalable and adaptive method. RL-LMAPF utilizes a fast distance cost map algorithm, leveraging GPU to compute the cost map (denoted as $h_{map}$) of all positions relative to the target point. This feature provides critical guidance for agent actions and dense rewards within the reinforcement learning environment, thereby accelerating the learning process. Agents share observations and communicate through a unified model, which streamlines decision-making and reduces computational load. Experiments demonstrate that RL-LMAPF can efficiently manage up to 4,096 agents simultaneously, with an average decision time of approximately 500 ms per step, significantly surpassing existing methods. 

## Installation:

```bash
pip3 install -r requirements.txt
```

## Inference Example:

To execute the **RL-LMAPF** algorithm and save the .svg render files using pre-trained weights, use the following command:

```bash
python3 run.py
```

To change the number of agents, the target position, and other parameters, modify the `paraser.py` file.

If you want to train **RL-LMAPF** on your own map, modify the `read_map()` function in the `env/GridMap.py` file.

All maps in the map folder `mapf-map-porcess` in this repository are from [movingai](https://movingai.com/benchmarks/mapf.html). Please indicate the reference if you need to use them.

## Citation:

```bibtex

```