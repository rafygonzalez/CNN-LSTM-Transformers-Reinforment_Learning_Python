from collections import defaultdict
from heapq import heappop, heappush
import math


WALL_COST = 5

# The first param of find_shortest_path "maze" is represent: maze[x][y].x and maze[x][y].y


def find_shortest_path(maze, start, end):
    # Use a binary heap as the priority queue
    queue = []

    route = {}

    costs = {start: 0}

    # Insert the start position into the queue with a priority of 0
    heappush(queue, (0, start))
    route[start] = None

    while queue:
        # Get the position with the lowest estimated cost from the queue
        _, current_pos = heappop(queue)
        if current_pos == end:
            return build_route(route, end)
        for neighbor, cost in get_neighbors(maze, current_pos):
            total_cost = costs[current_pos] + cost
            if neighbor not in costs or total_cost < costs[neighbor]:
                costs[neighbor] = total_cost
                priority = total_cost + estimate_cost(maze, neighbor, end)
                # Insert the neighbor into the queue with the updated priority
                heappush(queue, (priority, neighbor))
                route[neighbor] = current_pos
    return None


def get_cost(maze, pos, neighbor):
    x1, y1 = pos
    x2, y2 = neighbor
    # Initialize the cost to the Euclidean distance between the positions
    cost = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # Check if there is a wall between the current position and the neighbor
    if x1 == x2:
        # If the positions are on the same row, check for walls to the north or south
        if y1 < y2:
            # If the neighbor is to the south of the current position, check for walls to the south
            for y in range(y1 + 1, y2):
                if maze[x1][y].walls['S']:
                    # If there is a wall to the south, add the cost of traversing it
                    cost += WALL_COST
        elif y1 > y2:
            # If the neighbor is to the north of the current position, check for walls to the north
            for y in range(y2 + 1, y1):
                if maze[x1][y].walls['N']:
                    # If there is a wall to the north, add the cost of traversing it
                    cost += WALL_COST
    elif y1 == y2:
        # If the positions are on the same column, check for walls to the east or west
        if x1 < x2:
            # If the neighbor is to the east of the current position, check for walls to the east
            for x in range(x1 + 1, x2):
                if maze[x][y1].walls['E']:
                    # If there is a wall to the east, add the cost of traversing it
                    cost += WALL_COST
        elif x1 > x2:
            # If the neighbor is to the west of the current position, check for walls to the west
            for x in range(x2 + 1, x1):
                if maze[x][y1].walls['W']:
                    # If there is a wall to the west, add the cost of traversing it
                    cost += WALL_COST
    return cost


def estimate_cost(maze, pos, end):
    x1, y1 = pos
    x2, y2 = end
    # Initialize the estimated cost to the Euclidean distance between the positions
    cost = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # Check if there is a wall between the current position and the end position
    if x1 == x2:
        # If the positions are on the same row, check for walls to the north or south
        if y1 < y2:
            # If the end position is to the south of the current position, check for walls to the south
            for y in range(y1 + 1, y2):
                if maze[x1][y].walls['S']:
                    # If there is a wall to the south, add the cost of traversing it
                    cost += WALL_COST
        elif y1 > y2:
            # If the end position is to the north of the current position, check for walls to the north
            for y in range(y2 + 1, y1):
                if maze[x1][y].walls['N']:
                    # If there is a wall to the north, add the cost of traversing it
                    cost += WALL_COST
    elif y1 == y2:
        # If the positions are on the same column, check for walls to the east or west
        if x1 < x2:
            # If the end position is to the east of the current position, check for walls to the east
            for x in range(x1 + 1, x2):
                if maze[x][y1].walls['E']:
                    # If there is a wall to the east, add the cost of traversing it
                    cost += WALL_COST

    elif x1 > x2:
        # If the end position is to the west of the current position, check for walls to the west
        for x in range(x2 + 1, x1):
            if maze[x][y1].walls['W']:
                # If there is a wall to the west, add the cost of traversing it
                cost += WALL_COST

    return cost


def build_route(route, end):
    path = []

    pos = end
    while pos is not None:
        path.append(pos)
        pos = route[pos]

    path = path[::-1]

    return path


def get_neighbors(maze, pos):
    x, y = pos
    neighbors = []
    # Check for neighbors to the north
    if y > 0 and not maze[x][y].walls['N']:
        neighbors.append(((x, y - 1), get_cost(maze, pos, (x, y - 1))))
    # Check for neighbors to the south
    if y < len(maze[0]) - 1 and not maze[x][y].walls['S']:
        neighbors.append(((x, y + 1), get_cost(maze, pos, (x, y + 1))))
    # Check for neighbors to the east
    if x < len(maze) - 1 and not maze[x][y].walls['E']:
        neighbors.append(((x + 1, y), get_cost(maze, pos, (x + 1, y))))
    # Check for neighbors to the west
    if x > 0 and not maze[x][y].walls['W']:
        neighbors.append(((x - 1, y), get_cost(maze, pos, (x - 1, y))))
    return neighbors
