"""
@author Rohan Walia (rwalia@wpi.edu)
@version 2 (4/6/2022)

This script takes an image and calculates a path between given start and goal 
locations on the image using the A* algorithm.
"""

import cv2
import math
import numpy
import time
import sys

# Colors codes to mark path and start/goal points given by computed solution
NEON_GREEN = (0, 255, 0)  # for start/goal points
PURPLE = (139, 26, 85)  # for path
LIGHT_GRAY = (255, 0, 0)  # frontier nodes - currently not used
DARK_GRAY = (0, 0, 255)  # expanded nodes - currently not used

def visualize_search(image, path, start, goal):
    """
    This function shows the calculated path on a greyscale version of the 
    original image and start/end points overlayed on the original image. It 
    saves a copy of the displayed images.

    Args:
        map (numpy.ndarray): grayscale image used to plot computed start/end points
                        and path
        path (List[(int, int)]): list of coordinates to be overlayed on the 
                                  original image
    """

    # draw path pixels
    for waypoint in path:
        image[waypoint[0], waypoint[1]] = PURPLE

    # draw start and end pixels
    image[start[0], start[1]] = NEON_GREEN
    image[goal[0], goal[1]] = NEON_GREEN

    # resize the image for visualization purpose
    if (max(image.shape[0], image.shape[1]) < 500):
        scale_percent = 1000
        new_w = int(image.shape[1] * scale_percent / 100)
        new_h = int(image.shape[0] * scale_percent / 100)
        image= cv2.resize(image, (new_w, new_h))

    cv2.imshow("Result", image)

    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


def grayscale_info(map_original):
    """
    This function creates a binary occupancy grid from an image. The image is
    first converted to greyscale. If the color value of a cell is > 255/2 
    (white), it is marked as free (as 1) in the binary occupancy grid.
    Otherwise, it is marked as occupied (0).

    Args:
        map_original (numpy.ndarray): Image

    Returns:
        2x2 numpy.ndarray: grayscale version of the image
    """
    map = cv2.cvtColor(map_original, cv2.COLOR_BGR2GRAY)

    rows, columns = map.shape[0], map.shape[1]
    graph = numpy.zeros((rows, columns), dtype=int)

    for i in range(0, len(graph) - 1):
        for j in range(0, len(graph[0]) - 1):
            if map[i, j] > 255 / 2:
                graph[i][j] = 1
            else:
                graph[i][j] = 0
    return graph


def h(goal, cell):
    """
    Calculates heuristic distance from (row, col) to goal location. Assumes
    4-connected cells.

    Args:
        goal (int, int): row-column information of goal location on a grid
        cell (int, int): cell of interest on grid (located using (row, col))

    Returns:
        int: Manhattan distance between start location and cell
    """

    heuristic = abs(goal[0] - cell[0]) + abs(goal[1] - cell[1])  # manhattan distance
    return heuristic

# NOTE: g and h both use Manhattan distance for now. This might change.

def g(start, cell):
    """
    Calculates the cost of cell at (row, col) from start location. Assumes
    4-connected cells.

    Args:
        start (int, int): row-column information of start location on a grid
        cell (int, int): cell of interest on grid (located using (row, col))

    Returns:
        int: Manhattan distance between start location and cell
    """
    g_value = abs(start[0] - cell[0]) + abs(start[1] - cell[1])  # manhattan distance
    return g_value


def f(start, goal, cell):
    """

    Function that returns the sum of cost of cell from start (g) and heuristic
    value of that cell to the goal. The cell is located by (row, column) in a 
    grid.

    Args:
        start (int, int): Start location on a grid
        goal (int, int): Goal location on a grid
        row (int): Row of cell in a grid
        column (int): Column of cell in a grid

    Returns:
        int: Total cost associated with the cell
    """
    return g(start, cell) + h(goal, cell)

def return_neighbors(row, col, grid_shape):
    """
    This function returns the 4-connected neighbors of a cell in a grid. The
    cell is located using (row, col) in the grid.

    Args:
        row (int): row of cell in grid
        col (int): col of cell in grid
        grid_shape (numpy.ndarray.shape): Dimensions of occupancy grid

    Returns:
        List[(int, int)]: List of tuples of neighbor locations in grid 
    """
    max_rows = grid_shape[0] - 1 # total rows in the grid
    max_cols = grid_shape[1] - 1 # total cols in the grid

    list = [] # list to hold indexes of all neighbors

    if row > 0:
        list.append((row - 1, col))
    if row < max_rows:
        list.append((row + 1, col))
    if col > 0:
        list.append((row, col - 1))
    if col < max_cols:
        list.append((row, col + 1))
    return list


def a_star(occupancy_grid, start, goal):
    visited = []
    unvisited = [start]

    gList = {}
    hList = {}
    fList = {}

    for i in range(0, occupancy_grid.shape[0]):
        for j in range(0, occupancy_grid.shape[1]):
            gList[(i, j)] = math.inf
            hList[(i, j)] = math.inf
            fList[(i, j)] = math.inf

    gList[start] = g(start, start)
    hList[start] = h(goal, start)
    fList[start] = g(start, start) + h(goal, start)
    prev = {start: None}

    current = start
    counter = 0

    while (current != goal and open.__sizeof__() > 0) and counter <= occupancy_grid.size * 2:
        # print(counter)
        neighbor_list = return_neighbors(
            current[0], current[1], occupancy_grid.shape)
        for neighbor in neighbor_list:
            if occupancy_grid[neighbor[0], neighbor[1]] != 0 and visited.count(neighbor) == 0:
                if unvisited.count(neighbor) == 0:
                    unvisited.append(neighbor)
                if gList[neighbor] > g(start, neighbor):
                    gList[neighbor] = g(start, neighbor)
                if hList[neighbor] > h(start, neighbor):
                    hList[neighbor] = h(start, neighbor)
                if fList[neighbor] > gList[neighbor] + hList[neighbor]:
                    fList[neighbor] = f(start, goal, neighbor)
                    prev[neighbor] = current
        if visited.count(current) == 0:
            visited.append(current)
        if len(unvisited) != 0:
            unvisited.remove(current)
            if len(unvisited) != 0:
                unvisited.sort(key=lambda cell: f(start, goal, cell))
                current = unvisited[0]
        counter = counter + 1

    '''
    'current' cell should ideally be goal cell at this point. Back-track from
    this node to start node to generate a path.
    '''
    path = []
    node = current

    '''
    Only construct path if current node is the goal node, other wise return
    empty path.
    '''
    while node != start:
        node = prev[node]
        path.append(node)

    return path


def main(argv):
    """
    Driver function that does the following- 

    1. Parse command line arguents to obtain image, and start and goal locations
    2. Calculate path using Dijkstra's algorithm
    3. Draw path, and start and goal locations on the image
    4. Report time taken to calculate path and optionally print path in terms
       of grid coordianates

    Args:
        argv (list[string]): 'image_path', 'start location', 'goal location',
                             'show_path'
    """
    
    if len(argv) != 4:
        print("Error: Not enough arguments. Need 4 arguments: relative image" + 
        " path, start location (int, int), goal location(int, int)" + 
        " and show_path=true/false")
        print("Use as: python3 dijkstra.py 'relative_image_path'" +
        " '(start_x, start_y) '(goal_x, goal_y)' 'true/false'")
        sys.exit(1)

    image_path = argv[0]
    image = cv2.imread(image_path)

    occupancy_grid = grayscale_info(image)

    start = eval(argv[1])
    goal = eval(argv[2])
    show_path = argv[3]

    start_time = time.time()    
    path = a_star(occupancy_grid, start, goal)
    duration = time.time() - start_time
    print('Planning Time: ' + str(duration)  + ' seconds')
    if show_path=='true':
        print('Path: ' + str(path))
    visualize_search(image, path, start, goal)

if __name__ == "__main__":
    main(sys.argv[1:])