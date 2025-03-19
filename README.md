## WAREHOUSE ROBOT PATH OPTIMIZATION USING REINFORCEMENT LEARNING - A Q-LEARNING APPROACH

# Project Overview
This project applies **Reinforcement Learning (RL)**, specifically the **Q-Learning algorithm**, to optimize robot navigation in a warehouse environment. The system is designed to navigate a **15x15 grid-based warehouse** while avoiding obstacles and optimizing the path between pick-up and drop-off points. A **Graphical User Interface (GUI)** is implemented to allow users to define locations and visualize the robot’s movement.

##  Features
- **Q-Learning-based Path Optimization**: The robot learns the optimal route using reinforcement learning.
- **Obstacle Avoidance**: Predefined static obstacles simulate real-world constraints.
- **Graphical Interface (GUI)**: Built using **Tkinter**, allowing interactive input of start and drop-off points.
- **Visualization & Analysis**:
  - **Animated Robot Movements** on a 15x15 grid.
  - **Performance Metrics**: Plots for total rewards and Q-value convergence over episodes.
  - **Hyperparameter Tuning**: Optimized learning rate, discount factor, and episode count.

##  Methodology
### Q-Learning Algorithm:
- **States**: Represented as cells in a **15x15 grid**.
- **Actions**:
  - Move **Up**, **Down**, **Left**, or **Right**.
- **Reward Function**:
  - `-1` for moving to a non-goal state.
  - `-10` for colliding with an obstacle.
  - **Positive reward** for reaching the goal.
- **Learning Process**:
  - The agent explores different paths and updates its **Q-table** using the **Bellman Equation**.
  - Over time, the Q-values converge, resulting in an optimal navigation strategy.

### Hyperparameter Tuning:
The model was fine-tuned with:
| Parameter        | Tested Values|
|-----------------|--------------|
| **Learning Rate (α)** | 0.1, 0.2, 0.3 |
| **Discount Factor (γ)** | 0.9, 0.95, 0.99 |
| **Episodes** | 3000, 4000, 5000 |


##  Installation & Setup
### Prerequisites
Ensure you have Python and the required libraries installed.



### Running the Project
1. Clone the repository or download this repository in your machine
2. Open the code as folder in any code compiler. I have used VS code.
 
2. Run the script in the terminal: python main.py
3. Use the GUI to set start and drop-off points and observe the robot's navigation.

##  Future Enhancements
- **Scalability**: Extend to larger warehouse grids.
- **Dynamic Obstacles**: Implement real-time obstacle updates.
- **Deep Reinforcement Learning**: Upgrade to **Deep Q-Networks (DQN)** for complex environments.
- **Multi-Agent Systems**: Coordination between multiple robots for optimized warehouse management.

