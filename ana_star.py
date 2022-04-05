"""
@author Rohan Walia (rwalia@wpi.edu)
@version 1 (3/22/2021)

This script takes a jpg image of grid puzzles with black and white cells and calculates a path between specified
start and end points using ANA* algorithm.
"""

import cv2
import numpy
import time
import sys

# Colors codes to mark path and start/end points given by computed solution
NEON_GREEN = (0, 255, 0) # for start/end points
PURPLE = (139, 26, 85) # for path
LIGHT_GRAY = (255, 0, 0) # frontier nodes - currently not used
DARK_GRAY = (0, 0, 255) # expanded nodes - currently not used


def visualize_search(map, path):
    """
    This function shows the calculated path on a greyscale version of the original image and start/end points overlayed
    on the original image. It saves a copy of the displayed images.

    @param map: grayscale image used to plot computed start/end points and path
    @type map:
    @param path: list of coordinates (tuples) to be overlayed on the original image
    @type path: list of tuples
    @return: None
    @rtype: type(None)
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

    # show images as final result
    cv2.imshow("Solve attempt", pixel_access)
    cv2.imshow("Original", map)

    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

    # save images
    cv2.imwrite("test_original.png", map)
    cv2.imwrite("test_output.png", pixel_access)


def grayscale_info(map_original):
    """
    This function calculates a grid map (2D Mmtrix) containing information about which cells in the given picture are
    traversable. Traversable cells are stored as 1 and non-traversable cells are stored as 0.

    @param map_original: cv2 image containing
    @type map_original: cv2 image
    @return: grid map containing information about which cells are traversable and which ones are not
    @rtype: 2D int numpy array
    """

    # create 2D numpy array from grayscale version of image
    map = cv2.cvtColor(map_original, cv2.COLOR_BGR2GRAY)
    rows, columns = map.shape[0], map.shape[1]
    graph = numpy.zeros((rows, columns), dtype=int)

    for i in range(0, len(graph) - 1):
        for j in range(0, len(graph[0]) - 1):
            '''
             Since 255 is complete white and 0 is complete black, mark pixels that are closer to 255 as 1 and mark those
             closer to black as 0.
            '''
            if map[i, j] > 255 / 2:
                graph[i][j] = 1
            else:
                graph[i][j] = 0
    return graph


def h(end, row, column):
    """
    This function calculates the heuristic distance (Manhattan distance) of the given node (row, column) with respect to
    the required end point.

    @param end: end node
    @type end: int tuple
    @param row: horizontal position of cell (starting from left)
    @type row: int
    @param column: vertical position of cell (starting from top)
    @type column: int
    @return: heuristic value
    @rtype: float
    """
    heuristic = abs(end[0] - row) + abs(end[1] - column)  # manhattan distance
    return heuristic


def g(start, row, column):
    """
    This function calculates the cost of traversal (Manhattan distsance) of the given node (row, columns) with respect
    to the given start point.

    @param start: start node
    @type start: int tuple
    @param row: horizontal position of cell (starting from left)
    @type row: int
    @param column: vertical position of cell (starting from top)
    @type column: int
    @return: cost of path from starting node
    @rtype: float
    """
    g_value = abs(start[0] - row) + abs(start[1] - column)  # manhattan distance
    return g_value


def f(start, end, row, column):
    return g(start, row, column) + h(end, row, column)


def c(value1, value2):
    c_value = abs(value1[0] - value2[0]) + abs(value1[1] - value2[1])
    return c_value


def get_g(value):
    return g(start, value[0], value[1])


def get_f(value):
    return f(start, end, value[0], value[1])


def return_neighbors(x, y, shape):
    """
    @param x: int
    @type x: horizontal position of cell (starting from left)
    @param y: vertical position of cell (starting from top)
    @type y: int
    @param shape:  dimensions of grid-map
    @type shape: numpy shape object
    @return: list of neighbors
    @rtype: list of int tuples
    """
    max_x = shape[0] - 1
    max_y = shape[1] - 1

    list = []

    if x > 0:
        list.append((x - 1, y))
    if x < max_x:
        list.append((x + 1, y))
    if y > 0:
        list.append((x, y - 1))
    if y < max_y:
        list.append((x, y + 1))
    return list


def get_max_e():
    maxValue = -1
    max_key = None
    for item in unvisited:
        if item != end:
            if G - g(start, item[0], item[1])/h(end, item[0], item[1]) > maxValue:
                maxValue = G - g(start, item[0], item[1])/h(end, item[0], item[1])
                max_key = item
        else:
            return end
    return max_key


def improve_ana():

    mini_counter = 0
    global current

    global E
    global G

    while len(unvisited) > 0 and mini_counter <= grid_map.size * 2:
        key = get_max_e()
        if key is not None:
            current = key
        if len(unvisited) != 0:
            if unvisited.count(current) != 0:
                unvisited.remove(current)
        if eList[current] < E:
            E = eList[current]
        if current == end:
            return
        neighbors = return_neighbors(current[0], current[1], grid_map.shape)
        for value in neighbors:
            if grid_map[value[0], value[1]] != 0 and visited.count(value) == 0: #                       might need to delete visited.count(value) == 0
                if g(start, current[0], current[1]) + c(current, value) < gList[value]:
                    gList[value] = g(start, value[0], value[1]) + c(current, value)
                    prev[value] = current
                    if g(start, value[0], value[1]) + h(end, value[0], value[1]) < G:
                        unvisited.append(value)
                        if (value != end):
                            eList[value] = G - g(start, value[0], value[1])/h(end, value[0], value[1])
        # print(counter, ' | ', current)
        if visited.count(current) == 0:
            visited.append(current)
        mini_counter = mini_counter + 1
    return

def remove_sub_optimal_nodes():
    global unvisited
    for item in unvisited:
        if gList[item] + h(end, item[0], item[1]) < G:
            unvisited.remove(item)


'''
Main Code
'''

difficulty = "hard.jpg"

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


'''
ANA * Begins
'''
start_time = time.time()

visited = []
unvisited = [start]

gList = {}
eList = {}

G = sys.maxsize
E = sys.maxsize

grid_map = pixel_matrix

for i in range(0, grid_map.shape[0]):
    for j in range(0, grid_map.shape[1]):
        gList[(i, j)] = sys.maxsize
        eList[(i, j)] = sys.maxsize  # (G - gList[(i, j)]) / h(end, i, j)

eList[start] = 0

counter = 0
prev = {start: None}


while len(unvisited) > 0 and counter <= grid_map.size * 2:
    print("counter:", counter)
    improve_ana()
    remove_sub_optimal_nodes()

path = []
node = current
while node != start:
    node = prev[node]
    path.append(node)

end_time = time.time()
duration = end_time - start_time
print(duration)
print(path)
#visualize_search(image, path)