"""
@author Rohan Walia (rwalia@wpi.edu)
@version 1 (3/22/2021)

This script takes a jpg image of grid puzzles with black and white cells and calculates a path between specified
start and end points using Dijkstra's algorithm and A* algorithm.
"""

import cv2
import math
import numpy
import time

# Colors codes to mark path and start/end points given by computed solution
NEON_GREEN = (0, 255, 0)  # for start/end points
PURPLE = (139, 26, 85)  # for path
LIGHT_GRAY = (255, 0, 0)  # frontier nodes - currently not used
DARK_GRAY = (0, 0, 255)  # expanded nodes - currently not used

def visualize_search(map, path):
    """
    This function shows the calculated path on a greyscale version of the original image and start/end points overlayed
    on the original image. It saves a copy of the displayed images.

    Args:
        map (int, int): grayscale image used to plot computed start/end points and path
        path (List[(int, int)]): list of coordinates to be overlayed on the original image
    """
    
    pixel_access = map.copy()

    # draw start and end pixels
    map[start[0], start[1]] = NEON_GREEN
    map[end[0], end[1]] = NEON_GREEN

    # draw path pixels
    for pixel in path:
        pixel_access[pixel[0], pixel[1]] = PURPLE

    '''
    # draw frontier pixels
    for pixel in frontier.keys():
        pixel_access[pixel[0], pixel[1]] = LIGHT_GRAY

    # draw expanded pixels
    for pixel in expanded.keys():
        pixel_access[pixel[0], pixel[1]] = DARK_GRAY
    '''

    # resize the image for visualization purpose
    if (max(pixel_access.shape[0], pixel_access.shape[1]) < 500):
        scale_percent = 1000
        new_w = int(pixel_access.shape[1] * scale_percent / 100)
        new_h = int(pixel_access.shape[0] * scale_percent / 100)
        pixel_access = cv2.resize(pixel_access, (new_w, new_h))
        map = cv2.resize(map, (new_w, new_h))

    cv2.imshow("Solve attempt", pixel_access)
    cv2.imshow("Original", map)

    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

    cv2.imwrite("test_original.png", map)
    cv2.imwrite("test_output.png", pixel_access)


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


def h(end, row, column):
    """
    Calculates heuristic distance from  (row, col) from goal location. Assumes
    4-connected cells.

    Args:
        start (int, int): row-column information of start location on a grid
        row (int): row of cell in a grid
        col (int): column of cell in a grid

    Returns:
        int: Manhattan distance between start location and cell at (row, col)
    """

    heuristic = abs(end[0] - row) + abs(end[1] - column)  # manhattan distance
    return heuristic

# NOTE: g and h both use Manhattan distance for now. This might change.

def g(start, row, col):
    """
    Calculates the cost of cell at (row, col) from start location. Assumes
    4-connected cells.

    Args:
        start (int, int): row-column information of start location on a grid
        row (int): row of cell in a grid
        col (int): column of cell in a grid

    Returns:
        int: Manhattan distance between start location and cell at (row, col)
    """
    g_value = abs(start[0] - row) + abs(start[1] - col)  # manhattan distance
    return g_value


def f(start, goal, row, column):
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
    return g(start, row, column) + h(goal, row, column)

def get_g(value):
    """
    Wrapper around g() function to help sort 'unvisited' queue by 'g values'

    Args:
        value (int, int): cell denoted by (row, column) in a grid

    Returns:
        int: g value of that cell with respect to 'start' global variable
    """
    return g(start, value[0], value[1])


def get_f(value):
    """
    Wrapper around f() function to help sort 'unvisited' queue by 'f values'

    Args:
        value (int, int): cell denoted by (row, column) in a grid

    Returns:
        int: f value of that cell wrt 'start' and 'end' global variable
    """
    return f(start, end, value[0], value[1])

'''
TODO: Make 'start' and 'end' variables local, and find a way to get rid of 
wrapper functions get_f and get_g.
'''

def return_neighbors(row, col, grid_shape) -> List[(int, int)]:
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


def dijkstra(occupancy_grid, start, end) -> List[(int, int)]:
    """
    This function uses Dijkstra's algorithm to find a path from start location
    to a goal location on the grid.

    Args:
        occupancy_grid (numpy.array): Binary occupancy grid (1:free, 0:occupied)
        start (int, int): start location on grid
        end (int, int): end location on grid

    Returns:
        List[(int, int)]: Path from start to goal location
    """

    # list of visited and unvisited cells
    visited = []
    unvisited = [start]

    # dictionary of 'g' values for each cell in the occupancy grid
    gList = {}

    # set all g values to infinity at start
    for i in range(0, occupancy_grid.shape[0]):
        for j in range(0, occupancy_grid.shape[1]):
            gList[(i, j)] = math.inf

    gList[start] = g(start, start[0], start[1])
    prev = {start: None} # dictionary for keeping track of parent cell for each cell

    current = start
    counter = 0 # counter for stopping search for a path when all cells have been scanned

    # TODO: Find why I added open.__sizeof__() 
    while (current != end and open.__sizeof__() > 0) and counter <= occupancy_grid.size * 2:
        neighbors = return_neighbors(
            current[0], current[1], occupancy_grid.shape)
        for value in neighbors:
            if occupancy_grid[value[0], value[1]] != 0 and visited.count(value) == 0:
                if unvisited.count(value) == 0:
                    unvisited.append(value)
                if gList[value] > g(start, value[0], value[1]):
                    gList[value] = g(start, value[0], value[1])
                    prev[value] = current

        if visited.count(current) == 0:
            visited.append(current)
        if len(unvisited) != 0:
            unvisited.remove(current)
            if len(unvisited) != 0:
                unvisited.sort(key=get_g)
                current = unvisited[0]
        counter = counter + 1

    """
    'current' cell should ideally be goal cell at this point. Back-track from
    this node to start node to generate a path.
    """
    path = []
    node = current

    # TODO: Add end condition for what happens if node never reaches start
    while node != start:
        node = prev[node]
        path.append(node)

    return path


def a_star(occupancy_grid, start, end):
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

    gList[start] = g(start, start[0], start[1])
    hList[start] = h(end, start[0], start[1])
    fList[start] = g(start, start[0], start[1]) + h(end, start[0], start[1])
    prev = {start: None}

    current = start
    counter = 0

    while (current != end and open.__sizeof__() > 0) and counter <= occupancy_grid.size * 2:
        # print(counter)
        neighbors = return_neighbors(
            current[0], current[1], occupancy_grid.shape)
        for value in neighbors:
            if occupancy_grid[value[0], value[1]] != 0 and visited.count(value) == 0:
                if unvisited.count(value) == 0:
                    unvisited.append(value)
                if gList[value] > g(start, value[0], value[1]):
                    gList[value] = g(start, value[0], value[1])
                if hList[value] > h(start, value[0], value[1]):
                    hList[value] = h(start, value[0], value[1])
                if fList[value] > gList[value] + hList[value]:
                    fList[value] = f(start, end, value[0], value[1])
                    prev[value] = current
        if visited.count(current) == 0:
            visited.append(current)
        if len(unvisited) != 0:
            unvisited.remove(current)
            if len(unvisited) != 0:
                unvisited.sort(key=get_f)
                current = unvisited[0]
        counter = counter + 1

    path = []
    node = current
    while node != start:
        node = prev[node]
        path.append(node)

    return path


'''
Main Code
'''

difficulty = "medium.jpg"

image = cv2.imread(difficulty)

if difficulty == "trivial.jpg":
    start = (1, 8)
    end = (1, 20)
elif difficulty == "medium.jpg":
    start = (8, 201)
    end = (110, 1)
elif difficulty == "hard.jpg":
    start = (10, 2)
    end = (400, 220)
elif difficulty == "very_hard.jpg":
    start = (1, 324)
    end = (580, 1)
else:
    assert False, "Incorrect difficulty level provided"

pixel_matrix = grayscale_info(image)


start_time = time.time()
# an ordered list of (x,y) tuples, representing the path to traverse from start-->goal
path = dijkstra(pixel_matrix, start, end)
duration = time.time() - start_time
print(duration)
print(path)
visualize_search(image, path)


start_time = time.time()
# an ordered list of (x,y) tuples, representing the path to traverse from start-->goal
path = a_star(pixel_matrix, start, end)
duration = time.time() - start_time
print(duration)
print(path)
visualize_search(image, path)
