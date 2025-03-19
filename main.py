import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time
import matplotlib.pyplot as plt

# Grid configuration
GRID_SIZE = 15
OBSTACLES = [(0, 0), (0, 3), (0, 4), (0, 8), (1, 8), (1, 13), (2, 4), (2, 8),
             (3, 0), (3, 1), (3, 2), (3, 4), (4, 4), (4, 5), (4, 6), (4, 7),
             (4, 8), (4, 9), (5, 14), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5),
             (6, 6), (6, 10), (6, 11), (6, 12), (6, 14), (7, 1), (7, 6), 
             (7, 9), (7, 10), (7, 12), (7, 14), (8, 1), (8, 6), (8, 9), 
             (8, 12), (8, 14), (9, 1), (9, 6), (9, 11), (9, 14), (10, 1), 
             (10, 6), (11, 1), (11, 6), (12, 0), (12, 1), (12, 4), (12, 5), 
             (12, 6), (12, 13), (14, 0), (14, 1), (14, 2), (14, 8), (14, 9), 
             (14, 10), (14, 11)]

# Actions: up, down, left, right
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

PICK_UP_POINT = (0, 6)  # currently set pick-up point


class RobotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Warehouse Robot with Hyperparameter Tuning")

        # Initialize the grid and rewards
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for obs in OBSTACLES:
            self.grid[obs[0]][obs[1]] = -1  # Mark obstacles
        
        self.rewards = np.full((GRID_SIZE, GRID_SIZE), -1)  # Negative reward by default
        for obs in OBSTACLES:
            self.rewards[obs] = -10  # Large penalty for obstacles
        
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

        self.start = None
        self.drop = None
        
        self.rewards_history = []
        self.convergence_history = []

        # Perform hyperparameter tuning before GUI launch
        print("Performing Hyperparameter Tuning...")
        self.best_params = self.hyperparameter_tuning((0, 0), PICK_UP_POINT)  # Example start position
        print(f"Best Hyperparameters: Alpha={self.best_params[0]}, "
            f"Gamma={self.best_params[1]}, Episodes={self.best_params[2]}")
        
        # Load and resize shelf image for obstacles
        self.shelf_image = Image.open("assets/shelf.png")
        self.shelf_image = self.shelf_image.resize((80, 80), Image.Resampling.LANCZOS)
        self.shelf_image_tk = ImageTk.PhotoImage(self.shelf_image)

        # Load and resize location image for pick-up point
        self.location_image = Image.open("assets/location.png")
        self.location_image = self.location_image.resize((80, 80), Image.Resampling.LANCZOS)
        self.location_image_tk = ImageTk.PhotoImage(self.location_image)

        # Load and resize robot images for movement animation
        self.robot_image_1 = Image.open("assets/new_pickup_robot.jpg") 
        self.robot_image_1 = self.robot_image_1.resize((80, 80), Image.Resampling.LANCZOS)
        self.robot_image_1_tk = ImageTk.PhotoImage(self.robot_image_1)

        self.robot_image_2 = Image.open("assets/new_drop_robot.jpg") 
        self.robot_image_2 = self.robot_image_2.resize((80, 80), Image.Resampling.LANCZOS)
        self.robot_image_2_tk = ImageTk.PhotoImage(self.robot_image_2)

        # GUI components
        self.create_widgets()

    
    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.grid(row=0, column=1, padx=20, pady=20, sticky="n")

        self.canvas = tk.Canvas(self.root, width=750, height=750)
        self.canvas.grid(row=0, column=0, padx=20, pady=20)

        self.draw_grid()

        tk.Label(frame, text="Start (a, b):").grid(row=0, column=0)
        self.start_entry = tk.Entry(frame)
        self.start_entry.grid(row=0, column=1)

        tk.Label(frame, text="Drop-Off (a, b):").grid(row=1, column=0)
        self.drop_entry = tk.Entry(frame)
        self.drop_entry.grid(row=1, column=1)

        self.run_button = tk.Button(frame, text="Run", command=self.run_with_tuning)
        self.run_button.grid(row=2, column=0, columnspan=2)

        # Clear button to reset inputs and canvas
        self.clear_button = tk.Button(frame, text="Clear", command=self.clear_input)
        self.clear_button.grid(row=3, column=0, columnspan=2)

    def draw_grid(self):
        cell_size = 750 // GRID_SIZE
        # Resize images based on cell size
        self.shelf_image_tk = ImageTk.PhotoImage(self.shelf_image.resize((cell_size, cell_size), Image.Resampling.LANCZOS))
        self.location_image_tk = ImageTk.PhotoImage(self.location_image.resize((cell_size, cell_size), Image.Resampling.LANCZOS))
        self.robot_image_1_tk = ImageTk.PhotoImage(self.robot_image_1.resize((cell_size, cell_size), Image.Resampling.LANCZOS))
        self.robot_image_2_tk = ImageTk.PhotoImage(self.robot_image_2.resize((cell_size, cell_size), Image.Resampling.LANCZOS))

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                if self.grid[i][j] == -1:
                    self.canvas.create_image(x1 + cell_size / 2, y1 + cell_size / 2, image=self.shelf_image_tk)
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")

                if (i, j) == PICK_UP_POINT:
                    self.canvas.create_image(x1 + cell_size / 2, y1 + cell_size / 2, image=self.location_image_tk)

                self.canvas.create_text(x1 + cell_size / 2, y1 + cell_size / 2, text=f"({i},{j})", font=("Arial", 8))
    
    def animate_path(self, path, robot_image):
        """Animate the robot's path with the specified robot image, moving one grid cell at a time."""
        cell_size = 750 // GRID_SIZE  

        for pos in path:
            x1, y1 = pos[1] * cell_size, pos[0] * cell_size  # Convert grid coordinates to canvas position
            x2, y2 = x1 + cell_size, y1 + cell_size

            # Clear the canvas before moving the robot
            self.canvas.delete("robot")  # Delete the previous robot image

            # Place the robot image at the current position
            self.canvas.create_image(x1 + cell_size / 2, y1 + cell_size / 2, image=robot_image, tags="robot")
            self.root.update()
            time.sleep(0.5)

    def train_q_learning(self, start, goal, alpha, gamma, episodes):
        q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        total_rewards = []
        convergence_diffs = []

        for episode in range(episodes):
            state = start
            episode_reward = 0
            old_q_table = q_table.copy()
            while state != goal:
                action_idx = self.choose_action(state, q_table)
                next_state = self.get_next_state(state, action_idx)
                reward = self.rewards[next_state]

                max_future_q = np.max(q_table[next_state[0], next_state[1]])
                current_q = q_table[state[0], state[1], action_idx]

                q_table[state[0], state[1], action_idx] = current_q + alpha * (
                    reward + gamma * max_future_q - current_q
                )

                episode_reward += reward
                if self.grid[next_state[0]][next_state[1]] == -1:  # Stop if hitting an obstacle
                    break

                state = next_state
            # Track metrics after each episode
            total_rewards.append(episode_reward)
            max_diff = np.max(np.abs(q_table - old_q_table))  # Convergence metric
            convergence_diffs.append(max_diff)
        # Save metrics for visualization
        self.rewards_history = total_rewards
        self.convergence_history = convergence_diffs
        return q_table, sum(total_rewards)

    def plot_evaluation_metrics(self):
        """Plot evaluation metrics: rewards and convergence over episodes."""
        episodes = range(1, len(self.rewards_history) + 1)

        # Subplots for rewards and convergence
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot total rewards
        axs[0].plot(episodes, self.rewards_history, label="Total Rewards", color="blue")
        axs[0].set_title("Total Rewards Over Episodes")
        axs[0].set_xlabel("Episodes")
        axs[0].set_ylabel("Total Rewards")
        axs[0].legend()

        # Plot convergence (max Q-value differences)
        axs[1].plot(episodes, self.convergence_history, label="Max Q-value Difference", color="red")
        axs[1].set_title("Convergence Over Episodes")
        axs[1].set_xlabel("Episodes")
        axs[1].set_ylabel("Max Q-value Difference")
        axs[1].legend()

        # Show plots
        plt.tight_layout()
        plt.show()

    def choose_action(self, state, q_table):
        """Choose an action using an epsilon-greedy policy."""
        if np.random.uniform(0, 1) < 0.1:  # Exploration
            return np.random.randint(0, len(ACTIONS))
        else:  # Exploitation
            return np.argmax(q_table[state[0], state[1]])

    def get_next_state(self, state, action_idx):
        """Get the next state based on the action."""
        action = ACTIONS[action_idx]
        next_state = (state[0] + action[0], state[1] + action[1])

        if 0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE and self.grid[next_state[0]][next_state[1]] != -1:
            return next_state
        return state  # Invalid move, stay in the same state
    
    def hyperparameter_tuning(self, start, goal):
        alpha_values = [0.1, 0.2, 0.3]
        gamma_values = [0.9, 0.95, 0.99]
        episode_values = [3000, 4000, 5000]

        best_params = None
        best_reward = -float('inf')
        best_convergence = float('inf')  # Track the fastest convergence 
        for alpha in alpha_values:
            for gamma in gamma_values:
                for episodes in episode_values:
                    # Evaluate convergence
                    convergence_episodes = self.evaluate_convergence_rate(start, goal, alpha, gamma, episodes)
                    
                    # Train Q-learning and calculate total reward
                    q_table, total_reward = self.train_q_learning(start, goal, alpha, gamma, episodes)
                    
                    # Select the best parameters based on reward and convergence
                    if total_reward > best_reward or (total_reward == best_reward and convergence_episodes < best_convergence):
                        best_reward = total_reward
                        best_params = (alpha, gamma, episodes)
                        best_convergence = convergence_episodes

                    print(f"Alpha={alpha}, Gamma={gamma}, Episodes={episodes}")
        return best_params

    def evaluate_convergence_rate(self, start, goal, alpha, gamma, max_episodes=5000, threshold=1e-3):
        q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        for episode in range(1, max_episodes + 1):
            old_q_table = q_table.copy()  # Store Q-table before the current episode
            q_table, _ = self.train_q_learning(start, goal, alpha, gamma, 1) 

            max_difference = np.max(np.abs(q_table - old_q_table))
            if max_difference < threshold:
                return episode  # Return the episode where convergence occurred

        print(f"Did not converge within {max_episodes} episodes. Last max difference: {max_difference:.6f}")
        return max_episodes

    def find_path(self, start, goal):
        path = [start]
        state = start
        visited = set()

        while state != goal:
            visited.add(state)
            action_idx = np.argmax(self.q_table[state[0], state[1]])
            next_state = self.get_next_state(state, action_idx)

            # Avoid loops and obstacles
            if next_state == state or next_state in path or self.grid[next_state[0]][next_state[1]] == -1:
                break
            path.append(next_state)
            state = next_state
        return path

    def clear_input(self):
        """Clear input fields and canvas."""
    
        self.start_entry.delete(0, tk.END)
        self.drop_entry.delete(0, tk.END)

        # Redraw the grid and remove the previous paths
        self.canvas.delete("all")
        self.draw_grid()

    def run_with_tuning(self):
        try:
            # Parse inputs
            self.start = tuple(map(int, self.start_entry.get().split(',')))
            self.drop = tuple(map(int, self.drop_entry.get().split(',')))

            # Validate inputs
            for pos in [self.start, self.drop]:
                if not (0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE):
                    messagebox.showerror("Error", f"Position out of bounds: {pos}")
                    return
                if self.grid[pos[0]][pos[1]] == -1:
                    messagebox.showerror("Error", f"Position is an obstacle: {pos}")
                    return
            
            # Extract best parameters from tuning
            alpha, gamma, episodes = self.best_params

            # Train Q-table with best parameters for Start → Pick-Up Point
            self.q_table, _ = self.train_q_learning(self.start, PICK_UP_POINT, alpha, gamma, episodes)
            path_to_pickup = self.find_path(self.start, PICK_UP_POINT)
            self.animate_path(path_to_pickup, self.robot_image_1_tk)

            # Train Q-table with best parameters for Pick-Up Point → Drop-Off
            self.q_table, _ = self.train_q_learning(PICK_UP_POINT, self.drop, alpha, gamma, episodes)
            path_to_drop = self.find_path(PICK_UP_POINT, self.drop)
            self.animate_path(path_to_drop, self.robot_image_2_tk)

            # Plot evaluation metrics after training
            self.plot_evaluation_metrics()
            
        except Exception as e:
            print(f"Error during execution: {e}")
            messagebox.showerror("Error", "Invalid input! Please enter coordinates as 'x,y'.")

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = RobotApp(root)
    root.mainloop()
