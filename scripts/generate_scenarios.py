#!/usr/bin/env python3
"""
Goal Scenario Generator for Warehouse Benchmark

Creates different difficulty levels and patterns for waypoint navigation tasks.
This helps test algorithm performance across various scenarios.
"""

import argparse
import json
import numpy as np
from typing import List, Dict, Tuple
import random

class GoalScenarioGenerator:
    """Generates different goal scenarios for testing algorithms."""

    def __init__(self):
        # Warehouse boundaries
        self.bounds = {
            'x': (-15, 15),
            'y': (-15, 15)
        }

        # Obstacle positions (approximate)
        self.obstacles = [
            (0, 0),    # Center area
            (8, 6),    # Various warehouse locations
            (-8, 6),
            (8, -6),
            (-8, -6)
        ]

    def generate_scenario(self, difficulty: str, pattern: str, num_goals: int = 4) -> Dict:
        """
        Generate a goal scenario.

        Args:
            difficulty: 'easy', 'medium', 'hard'
            pattern: 'square', 'circle', 'random', 'spiral', 'maze'
            num_goals: Number of goals in the sequence

        Returns:
            Scenario dictionary with waypoints and metadata
        """
        if pattern == 'square':
            waypoints = self._generate_square_pattern(num_goals, difficulty)
        elif pattern == 'circle':
            waypoints = self._generate_circle_pattern(num_goals, difficulty)
        elif pattern == 'random':
            waypoints = self._generate_random_pattern(num_goals, difficulty)
        elif pattern == 'spiral':
            waypoints = self._generate_spiral_pattern(num_goals, difficulty)
        elif pattern == 'maze':
            waypoints = self._generate_maze_pattern(num_goals, difficulty)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        scenario = {
            'name': f'{difficulty}_{pattern}_{num_goals}goals',
            'difficulty': difficulty,
            'pattern': pattern,
            'num_goals': num_goals,
            'waypoints': waypoints,
            'metadata': {
                'total_distance': self._calculate_total_distance(waypoints),
                'complexity_score': self._calculate_complexity(waypoints, difficulty),
                'obstacle_density': self._calculate_obstacle_density(waypoints)
            }
        }

        return scenario

    def _generate_square_pattern(self, num_goals: int, difficulty: str) -> List[Tuple[float, float]]:
        """Generate waypoints in a square pattern."""
        size = {'easy': 8, 'medium': 12, 'hard': 14}[difficulty]

        # Create square corners
        waypoints = [
            (size, size),
            (-size, size),
            (-size, -size),
            (size, -size)
        ]

        # If more goals needed, interpolate between corners
        while len(waypoints) < num_goals:
            for i in range(len(waypoints)):
                if len(waypoints) >= num_goals:
                    break
                # Insert midpoint between current and next waypoint
                next_i = (i + 1) % len(waypoints)
                mid_x = (waypoints[i][0] + waypoints[next_i][0]) / 2
                mid_y = (waypoints[i][1] + waypoints[next_i][1]) / 2
                waypoints.insert(i + 1, (float(mid_x), float(mid_y)))

        return waypoints[:num_goals]

    def _generate_circle_pattern(self, num_goals: int, difficulty: str) -> List[Tuple[float, float]]:
        """Generate waypoints in a circular pattern."""
        radius = {'easy': 6, 'medium': 10, 'hard': 12}[difficulty]

        waypoints = []
        for i in range(num_goals):
            angle = 2 * np.pi * i / num_goals
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            waypoints.append((float(x), float(y)))

        return waypoints

    def _generate_random_pattern(self, num_goals: int, difficulty: str) -> List[Tuple[float, float]]:
        """Generate random waypoints with minimum distance constraints."""
        min_distance = {'easy': 4, 'medium': 3, 'hard': 2}[difficulty]
        max_attempts = 100

        waypoints = [(0, 0)]  # Start at center

        for _ in range(num_goals - 1):
            for attempt in range(max_attempts):
                x = random.uniform(self.bounds['x'][0], self.bounds['x'][1])
                y = random.uniform(self.bounds['y'][0], self.bounds['y'][1])

                # Check minimum distance from existing waypoints
                valid = True
                for wx, wy in waypoints:
                    if np.sqrt((x - wx)**2 + (y - wy)**2) < min_distance:
                        valid = False
                        break

                if valid:
                    waypoints.append((x, y))
                    break
            else:
                # If no valid position found, place randomly
                x = random.uniform(self.bounds['x'][0], self.bounds['x'][1])
                y = random.uniform(self.bounds['y'][0], self.bounds['y'][1])
                waypoints.append((x, y))

        return waypoints

    def _generate_spiral_pattern(self, num_goals: int, difficulty: str) -> List[Tuple[float, float]]:
        """Generate waypoints in a spiral pattern."""
        max_radius = {'easy': 8, 'medium': 12, 'hard': 14}[difficulty]

        waypoints = []
        for i in range(num_goals):
            # Spiral equation: r = a + bθ
            theta = i * 0.8  # Angular step
            radius = 2 + 1.5 * theta  # Spiral out from center
            radius = min(radius, max_radius)

            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            waypoints.append((float(x), float(y)))

        return waypoints

    def _generate_maze_pattern(self, num_goals: int, difficulty: str) -> List[Tuple[float, float]]:
        """Generate waypoints that navigate around obstacles like a maze."""
        # Create a path that weaves around obstacles
        waypoints = []

        # Start at one corner
        waypoints.append((-12, -12))

        # Navigate around obstacles
        maze_points = [
            (-8, -12),  # Navigate around first obstacle area
            (-4, -8),
            (0, -12),   # Around center
            (4, -8),
            (8, -12),   # Around next area
            (12, -8),
            (12, 0),    # Up the side
            (8, 4),
            (4, 8),     # Around top area
            (0, 12),
            (-4, 8),
            (-8, 12),   # Around final area
            (-12, 8),
            (-12, 0),   # Back down
            (-8, -4),
            (-12, -8)   # End
        ]

        # Select subset based on difficulty
        subset_size = {'easy': 6, 'medium': 10, 'hard': 15}[difficulty]
        step = max(1, len(maze_points) // subset_size)

        for i in range(0, len(maze_points), step):
            if len(waypoints) >= num_goals:
                break
            waypoints.append(maze_points[i])

        return waypoints[:num_goals]

    def _calculate_total_distance(self, waypoints: List[Tuple[float, float]]) -> float:
        """Calculate total path distance."""
        if len(waypoints) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(waypoints) - 1):
            dx = waypoints[i+1][0] - waypoints[i][0]
            dy = waypoints[i+1][1] - waypoints[i][1]
            total_distance += np.sqrt(dx**2 + dy**2)

        return total_distance

    def _calculate_complexity(self, waypoints: List[Tuple[float, float]], difficulty: str) -> float:
        """Calculate scenario complexity score."""
        base_complexity = {'easy': 1.0, 'medium': 2.0, 'hard': 3.0}[difficulty]
        distance_factor = self._calculate_total_distance(waypoints) / 50.0  # Normalize
        turns = self._count_direction_changes(waypoints)

        return base_complexity * (1 + distance_factor) * (1 + turns * 0.1)

    def _count_direction_changes(self, waypoints: List[Tuple[float, float]]) -> int:
        """Count significant direction changes in the path."""
        if len(waypoints) < 3:
            return 0

        changes = 0
        for i in range(1, len(waypoints) - 1):
            # Calculate vectors
            v1 = np.array([waypoints[i][0] - waypoints[i-1][0],
                          waypoints[i][1] - waypoints[i-1][1]])
            v2 = np.array([waypoints[i+1][0] - waypoints[i][0],
                          waypoints[i+1][1] - waypoints[i][1]])

            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm

                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)

                # Count significant turns (>30 degrees)
                if angle > np.pi / 6:
                    changes += 1

        return changes

    def _calculate_obstacle_density(self, waypoints: List[Tuple[float, float]]) -> float:
        """Calculate obstacle density around waypoints."""
        total_density = 0.0
        for wx, wy in waypoints:
            min_distance = float('inf')
            for ox, oy in self.obstacles:
                distance = np.sqrt((wx - ox)**2 + (wy - oy)**2)
                min_distance = min(min_distance, distance)
            # Convert distance to density (closer = higher density)
            density = max(0, 1 - min_distance / 10)
            total_density += density

        return total_density / len(waypoints) if waypoints else 0.0

def generate_scenario_suite() -> List[Dict]:
    """Generate a comprehensive suite of test scenarios."""
    generator = GoalScenarioGenerator()
    scenarios = []

    # Generate various scenarios
    difficulties = ['easy', 'medium', 'hard']
    patterns = ['square', 'circle', 'random', 'spiral', 'maze']
    goal_counts = [4, 6, 8]

    for difficulty in difficulties:
        for pattern in patterns:
            for num_goals in goal_counts:
                try:
                    scenario = generator.generate_scenario(difficulty, pattern, num_goals)
                    scenarios.append(scenario)
                except Exception as e:
                    print(f"Failed to generate {difficulty}_{pattern}_{num_goals}: {e}")

    return scenarios

def save_scenarios(scenarios: List[Dict], output_file: str):
    """Save scenarios to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(scenarios, f, indent=2)

    print(f"Saved {len(scenarios)} scenarios to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate goal scenarios for warehouse benchmark")
    parser.add_argument("--output", type=str, default="./scenarios.json",
                       help="Output JSON file for scenarios")
    parser.add_argument("--difficulty", choices=['easy', 'medium', 'hard'],
                       help="Generate only scenarios of this difficulty")
    parser.add_argument("--pattern", choices=['square', 'circle', 'random', 'spiral', 'maze'],
                       help="Generate only scenarios of this pattern")
    parser.add_argument("--num_goals", type=int, choices=[4, 6, 8],
                       help="Generate only scenarios with this many goals")

    args = parser.parse_args()

    print("🎯 Generating goal scenarios...")

    if any([args.difficulty, args.pattern, args.num_goals]):
        # Generate specific scenario
        generator = GoalScenarioGenerator()
        scenario = generator.generate_scenario(
            args.difficulty or 'medium',
            args.pattern or 'square',
            args.num_goals or 4
        )
        scenarios = [scenario]
        print(f"Generated 1 scenario: {scenario['name']}")
    else:
        # Generate full suite
        scenarios = generate_scenario_suite()
        print(f"Generated {len(scenarios)} scenarios")

    # Save scenarios
    save_scenarios(scenarios, args.output)

    # Print summary
    print("\n📊 Scenario Summary:")
    difficulties = {}
    patterns = {}

    for scenario in scenarios:
        diff = scenario['difficulty']
        pattern = scenario['pattern']

        difficulties[diff] = difficulties.get(diff, 0) + 1
        patterns[pattern] = patterns.get(pattern, 0) + 1

    print(f"Difficulties: {difficulties}")
    print(f"Patterns: {patterns}")

    print(f"\n✅ Scenarios saved to {args.output}")
    print("Use these scenarios to test algorithm performance across different conditions!")

if __name__ == "__main__":
    main()