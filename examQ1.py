from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set
import numpy as np


class Node:
    def __init__(
        self, pos: Tuple[int, int], g: int = 0, h: int = 0, parent=None, time: int = 0
    ):
        self.pos = pos
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic (estimated cost to goal)
        self.f = g + h  # Total cost
        self.parent = parent
        self.time = time  # Time step when agent reaches this node

    def __lt__(self, other):
        return self.f < other.f


class MultiAgentPathfinding:
    def __init__(self, grid_size: Tuple[int, int], obstacles: List[Tuple[int, int]]):
        self.grid_size = grid_size
        self.obstacles = set(obstacles)
        self.time_space_obstacles = (
            {}
        )  # Dictionary to store occupied positions at each time step

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, pos: Tuple[int, int], time: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]  # Including wait action
        neighbors = []

        for dx, dy in directions:
            new_pos = (pos[0] + dx, pos[1] + dy)

            # Check grid boundaries and obstacles
            if (
                0 <= new_pos[0] < self.grid_size[0]
                and 0 <= new_pos[1] < self.grid_size[1]
                and new_pos not in self.obstacles
            ):

                # Check time-space conflicts
                if (
                    time + 1 not in self.time_space_obstacles
                    or new_pos not in self.time_space_obstacles[time + 1]
                ):
                    neighbors.append(new_pos)

        return neighbors

    def find_path(
        self, start: Tuple[int, int], goal: Tuple[int, int], agent_id: int
    ) -> List[Tuple[int, int]]:
        """Find path for a single agent using A* algorithm."""
        open_list = []
        closed_set = set()
        start_node = Node(start, h=self.manhattan_distance(start, goal))
        heappush(open_list, start_node)

        while open_list:
            current = heappop(open_list)

            if current.pos == goal:
                # Reconstruct path
                path = []
                while current:
                    path.append(current.pos)
                    current = current.parent
                path.reverse()
                return path

            if (current.pos, current.time) in closed_set:
                continue

            closed_set.add((current.pos, current.time))

            for next_pos in self.get_neighbors(current.pos, current.time):
                new_g = current.g + 1
                new_node = Node(
                    next_pos,
                    g=new_g,
                    h=self.manhattan_distance(next_pos, goal),
                    parent=current,
                    time=current.time + 1,
                )

                if (next_pos, new_node.time) not in closed_set:
                    heappush(open_list, new_node)

        return []  # No path found

    def solve(
        self, starts: List[Tuple[int, int]], goals: List[Tuple[int, int]]
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Find paths for all agents."""
        paths = {}

        # Sort agents by distance to their goals (prioritize longer paths)
        agents = [
            (i, start, goal) for i, (start, goal) in enumerate(zip(starts, goals))
        ]
        agents.sort(key=lambda x: self.manhattan_distance(x[1], x[2]), reverse=True)

        for agent_id, start, goal in agents:
            # Find path for current agent
            path = self.find_path(start, goal, agent_id)

            if not path:
                raise ValueError(f"No path found for agent {agent_id}")

            # Update time-space obstacles
            for t, pos in enumerate(path):
                if t not in self.time_space_obstacles:
                    self.time_space_obstacles[t] = set()
                self.time_space_obstacles[t].add(pos)

            paths[agent_id] = path

        return paths


def visualize_paths(
    grid_size: Tuple[int, int],
    # obstacles: List[Tuple[int, int]],
    obstacles: 3,
    paths: Dict[int, List[Tuple[int, int]]],
) -> None:
    """Visualize the paths of all agents."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot grid
    ax.grid(True)
    ax.set_xlim(-0.5, grid_size[0] - 0.5)
    ax.set_ylim(-0.5, grid_size[1] - 0.5)

    # Plot obstacles
    for obs in obstacles:
        ax.add_patch(Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color="gray"))

    # Plot paths with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(paths)))
    for (agent_id, path), color in zip(paths.items(), colors):
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        ax.plot(path_x, path_y, "-o", color=color, label=f"Agent {agent_id}")

        # Mark start and goal
        ax.plot(
            path_x[0],
            path_y[0],
            "o",
            color=color,
            markersize=15,
            label=f"Start {agent_id}",
        )
        ax.plot(
            path_x[-1],
            path_y[-1],
            "s",
            color=color,
            markersize=15,
            label=f"Goal {agent_id}",
        )

    ax.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Define grid size and obstacles
    grid_size = (10, 10)
    obstacles = [(2, 2), (2, 3), (3, 2), (7, 7), (7, 6), (6, 7)]

    # Define start and goal positions for agents
    starts = [(0, 0), (0, 9), (9, 0)]
    goals = [(9, 9), (9, 0), (0, 9)]

    # Create solver
    mapf = MultiAgentPathfinding(grid_size, obstacles)

    try:
        # Find paths for all agents
        paths = mapf.solve(starts, goals)

        # Print results
        print("\nPaths found for all agents:")
        for agent_id, path in paths.items():
            print(f"Agent {agent_id}:")
            print(f"Path length: {len(path)}")
            print(f"Path: {path}\n")

        # Visualize paths
        visualize_paths(grid_size, obstacles, paths)

    except ValueError as e:
        print(f"Error: {e}")
