'''
@author Rohan Walia
@version 4/15/2021

This file contains code that implements bi-directional RRT*.
'''

import numpy
import random
import time
import math
import matplotlib.pyplot as plt


def get_c_space(size, P):
    """
    Returns the c_space as a 2D numpy array

    :param size: size of the grid
    :type size: int tuple
    :param P: Obstacle spawning probability (range: 0 - 1)
    :type P: float
    :return: grid filled with obstacles
    :rtype: 2D numpy array
    """

    rows = size[0]
    columns = size[1]
    c_space = numpy.zeros((rows, columns), dtype=int)

    for i in range(0, rows):
        for j in range(0, columns):
            if random.random() < P:
                if (i, j) != start and (i, j) != end:
                    c_space[i][j] = -1
    return c_space


def return_random_point(c_space):
    """
    Returns random point in the c_space

    :param c_space: Grid filled with obstacles
    :type c_space: 2D numpy array
    :return: indices of random point
    :rtype: int tuple
    """
    i = random.randint(0, c_space.shape[0] - 1)
    j = random.randint(0, c_space.shape[1] - 1)

    while c_space[i][j] != 0:
        i = random.randint(0, c_space.shape[0] - 1)
        j = random.randint(0, c_space.shape[1] - 1)

    return i, j


def euclidean_distance(point1, point2):
    """
    Returns the euclidean distance between two points

    :param point1: First point on the grid
    :type point1: int tuple
    :param point2: Second point on the grid
    :type point2: int tuple
    :return: euclidean distance between the two points
    :rtype: float
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def find_nearest_neighbor(vertices, x_rand):
    """
    Returns the node in the tree that is nearest to x_rand

    :param vertices: Tree
    :type vertices: python dictionary, with keys as vertices and values as list of connected nodes
    :param x_rand: Random point on the grid
    :type x_rand: int tuple
    :return: point in the tree that is nearest to x_rand
    :rtype: int tuple
    """
    min_distance = math.inf
    key = (0, 0)
    for vertex in vertices:
        cur_distance = euclidean_distance(vertex, x_rand)
        if cur_distance < min_distance:
            min_distance = cur_distance
            key = vertex
    return key


def find_nearest_star_neighbor(vertices, x_near, x_new, cost):
    """
    Returns the neighbor of x_near that has the least traversal cost from the root to x_new


    :param vertices: Tree
    :type vertices: python dictionary, with keys as vertices and values as list of connected nodes
    :param x_near: The point in the tree closes to x_new
    :type x_near: int tuple
    :param x_new:
    :type x_new: int tuple
    :param cost: list of costs of traversal from the root to each node
    :type cost: python dictionary, with keys as nodes in the tree and values as cost of traversal from the tree root
    :return: node in the tree that has the shortest cost of traversal from the root to x_new, neighbors analyzed
    :rtype: int tuple, list of int tuples
    """
    radius = 4 # neighborhood of scanning for nearest neighbors
    neighbors = []
    for vertex in vertices:
        if euclidean_distance(vertex, x_near) <= radius:
            neighbors.append(vertex)

    x_nearest = x_near
    min_cost = cost[x_near] + euclidean_distance(x_near, x_new)
    for vertex in neighbors:
        current_cost = cost[vertex] + euclidean_distance(vertex, x_new)
        if current_cost < min_cost:
            min_cost = current_cost
            x_nearest = vertex
    return x_nearest, neighbors


def rewire(vertices, neighbors, cost, x_new):
    """
    Recalculates the cost of a list of specified neighbors by checking the cost of traversal of each neighbor from
    the root of the tree. Updates the 'vertices' and 'cost' dictionaries accordingly.

    :param vertices: Tree
    :type vertices: python dictionary with keys as nodes (int tuples) and value as list of connected nodes (int tuples)
    :param neighbors: List of neighbors that need to be analyzed
    :type neighbors: python list of int tuples
    :param cost: Costs of the nodes in the tree
    :type cost: python dictionary with keys as nodes (int tuples) and values as costs (float)
    :param x_new: The node to be added to the tree
    :type x_new: int tuple
    :return: Return the updated vertices and cost dictionaries
    :rtype: python dictionaries
    """

    for vertex in neighbors:
        new_cost = cost[x_new] + euclidean_distance(x_new, vertex)
        if new_cost < cost[vertex]:
            cost[vertex] = new_cost # update cost
            vertices[x_new].append(vertex) # replace parent of new cost is shorter
            for key, val in vertices.items(): # find old parent key
                if vertex in val:
                    parent = key
            vertex_list = vertices[parent] # get list of current children
            vertex_list.remove(vertex) # remove current node
            vertices[parent] = vertex_list # reassign list of children

    return vertices, cost


def slope(point1, point2):
    """
    Returns the slope of the two points. If the two points lie in the same column in the grid, it returns
    math.inf as the slope.

    :param point1: The first point to analyze
    :type point1: int tuple
    :param point2: The second point to analyze
    :type point2: int tuple
    :return: Slope between the two points
    :rtype: float
    """
    if point1[1] == point2[1]:
        return math.inf
    return (point2[0] - point1[0])/(point2[1] - point1[1])


def return_new_node(x_near, x_rand, delta):
    """
    New node (x_new) is generated on the line connecting x_rand and x_near based on a configurable step size (delta).

    :param x_near: The node in the tree nearest to the point x_rand
    :type x_near: int tuple
    :param x_rand: Random point on the grid
    :type x_rand: int tuple
    :param delta: step size
    :type delta: int
    :return: The new point
    :rtype: int tuple
    """
    m = slope(x_rand, x_near)  # slope
    if m != math.inf:
        x = x_near[0] + delta * math.sqrt(1/(1 + m ** 2))
        y = x_near[1] + m * delta * math.sqrt(1 / (1 + m ** 2))
    else: # nodes lie in same column. Just return the column
        x = x_near[0] + delta
        y = x_near[1]
    x_new = (round(x), round(y))
    return x_new

def return_new_reverse_node(x_near, x_rand, delta):
    """
    Finds a new node in the opposite direction (from end node to start node). The new node is generated on the line
    connecting x_rand and x_near based on a configurable step size (delta).

    :param x_near: The node in the tree nearest to the point x_rand
    :type x_near: int tuple
    :param x_rand: Random point on the grid
    :type x_rand: int tuple
    :param delta: step size
    :type delta: int
    :return: The new point
    :rtype: int tuple
    """
    m = slope(x_rand, x_near)  # slope
    if m != math.inf:
        x = x_near[0] - delta * math.sqrt(1/(1 + m ** 2))
        y = x_near[1] - m * delta * math.sqrt(1 / (1 + m ** 2))
    else: # nodes lie in same column. Just return the column
        x = x_near[0] - delta
        y = x_near[1]
    x_new = (round(x), round(y))
    return x_new


def init_vertices(type, start, end):
    """
    Returns initialized vertices dictionary based on the type specified. Type 1 corresponds to the first RRT* tree, so
    the dictionary is initialized with the start node. Type 2 corresponds to the second RRT* tree, so the dictionary
    is intialized with the end node.

    :param type: Type of tree
    :type type: int
    :param start: Start node
    :type start: int tuple
    :param end: end node
    :type end: int tuple
    :return: tree dictionary containing the first node
    :rtype: python dictionary with keys as nodes and values as list of connected nodes
    """
    vertices = {}
    if type == 1:
        x_init = start
    elif type == 2:
        x_init = end
    vertices[x_init] = []
    return vertices


def check_avoidance(c_space, point1, point2):
    """
    Checks if there are any obstacles in the c_space between the two specified points. If a point lies on the line
    connecting point1 and point2 and it is an obstacle, the function returns false.

    :param c_space: The occupancy grid
    :type c_space: 2D numpy array
    :param point1: The first point (further towards the top left)
    :type point1:  int tuple
    :param point2: The second point (further towards the bottom right)
    :type point2: int tuple
    :return: return true if there are obstacles in the line of sight between the two points, false otherwise
    :rtype: boolean
    """
    m = slope(point2, point1)
    for i in range(point1[0], point2[0] + 1):
        for j in range(point1[1], point2[1] + 1):
            if slope((i, j), point1) == m and (c_space[i][j] != 0): # if point lies in line of site and is obstacle
                return False
    return True


def is_valid(x_new, size):
    """
    Checks if the node x_new is bound by the specified grid size. Returns true if node is bounded by the size, false
    otherwise.

    :param x_new: The node to check
    :type x_new: int tuple
    :param size: size of the grid
    :type size: int tuple
    :return: true if the point is bounded by the size, false otherwise
    :rtype: boolean
    """
    return x_new[0] < size[0] and x_new[1] < size[1]


def RRT_star(start, end):
    """
    Creates the first RRT* tree that starts form the start node and expands to towards the end node

    :param start: The start node
    :type start: int tuple
    :param end: The end node
    :type end: int tuple
    :return: The first RRT* tree
    :rtype: python dictionary with keys as nodes and values as list of connected nodes
    """
    global c_space

    vertices = init_vertices(1, start, end)
    max_iter = c_space.shape[0]*c_space.shape[1]
    delta = 1
    cost = {start: 0}

    i = 0
    while i < max_iter/2:
        x_rand = return_random_point(c_space)
        x_near = find_nearest_neighbor(vertices, x_rand)
        x_new = return_new_node(x_near, x_rand, delta)
        x_nearest, neighbors = find_nearest_star_neighbor(vertices, x_near, x_new, cost)

        if x_new == end:
            cost[x_new] = cost[x_nearest] + euclidean_distance(x_nearest, x_new)
            vertices[x_nearest].append(x_new)
            c_space[x_new[0]][x_new[1]] = 1
            return vertices

        if (is_valid(x_new, c_space.shape) and check_avoidance(c_space, x_near, x_new)):
            vertices[x_nearest].append(x_new)
            vertices[x_new] = []
            cost[x_new] = cost[x_nearest] + euclidean_distance(x_nearest, x_new)
            c_space[x_new[0]][x_new[1]] = 1

            if x_near != x_nearest:
                vertices, cost = rewire(vertices, neighbors, cost, x_new)
        i = i + 1

    return vertices


def extend_RRT_direction(start, end, straight_vertices):
    """
    Creates the second RRT* tree starting from the end node and expanding towards the start node

    :param start: The start node
    :type start: int tuple
    :param end: The end node
    :type end: int tuple
    :param straight_vertices: the first RRT* tree
    :type straight_vertices: python dictionary with keys as nodes and values as list of connected nodes
    :return: The second RRT* tree
    :rtype: python dictionary with keys as nodes and values as list of connected nodes
    """
    global c_space

    vertices = init_vertices(2, start, end)
    max_iter = c_space.shape[0] * c_space.shape[1]
    delta = 1
    cost = {end: 0}

    i = 0
    while i < max_iter*4:
        x_rand = return_random_point(c_space)
        x_near = find_nearest_neighbor(vertices, x_rand)
        x_new = return_new_reverse_node(x_near, x_rand, delta)
        x_nearest, neighbors = find_nearest_star_neighbor(vertices, x_near, x_new, cost)
        #print(x_nearest, x_new)
        if x_new == start or x_new in straight_vertices:
            vertices[x_nearest].append(x_new)
            vertices[x_new] = []
            c_space[x_new[0]][x_new[1]] = 2
            cost[x_new] = cost[x_nearest] + euclidean_distance(x_nearest, x_new)
            return vertices

        if (is_valid(x_new, c_space.shape) and check_avoidance(c_space, x_new, x_nearest)):
            vertices[x_nearest].append(x_new)
            vertices[x_new] = []
            c_space[x_new[0]][x_new[1]] = 2
            cost[x_new] = cost[x_nearest] + euclidean_distance(x_nearest, x_new)

            if x_near != x_nearest:
                vertices, cost = rewire(vertices, neighbors, cost, x_new)
        i = i + 1

    return vertices


def create_plot(c_space):
    """
    Creates a figure object from the c_space

    :param c_space: The occupancy grid to create the figure from
    :type c_space: 2D numpy array
    :return: figure object with empty c_space
    :rtype: matplotlib.figure()
    """
    fig = plt.figure()

    ax = fig.gca()
    ax.set_xticks(numpy.arange(0, c_space.shape[0] + 1, 1))
    plt.grid()

    ax = plt.gca()  # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    ax.xaxis.tick_top()  # and move the X-Axis
    ax.yaxis.set_ticks(numpy.arange(0, c_space.shape[1] + 1, 1))  # set y-ticks
    ax.yaxis.tick_left()  # remove right y-Ticks

    plt.title('Iteration: 0')

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)

    return fig


def add_lines(straight_vertices, reverse_vertices, fig):
    """
    Adds tree vertices to the given figure

    :param straight_vertices: First RRT* tree
    :type straight_vertices: python dictionary with keys as nodes and values as list of connected nodes
    :param reverse_vertices: Second RRT* tree
    :type reverse_vertices: python dictionary with keys as nodes and values as list of connected nodes
    :param fig: The figure to which the trees need to be added
    :type fig: matplotlib.figure()
    :return: updated figure
    :rtype: matplotlib.figure()
    """
    for vertex in straight_vertices:
        fig.gca().plot(vertex[0], vertex[1], "ob")

    for vertex in reverse_vertices:
        fig.gca().plot(vertex[0], vertex[1], "oy")

    return fig


def update_plot(c_space, fig, start, end):
    """
    Plots the obstacles and end point on the given figure

    :param c_space: Occupancy grid
    :type c_space: 2D numpy array
    :param fig: figure on which the obstacles and end point need to be plotted
    :type fig: matlplotlib.figure()
    :param start: The start node
    :type start: int tuple
    :param end: The end node
    :type end: int tuple
    :return: updated figure
    :rtype: matplotlib.figure()
    """
    for i in range (0, c_space.shape[0]):
        for j in range (0, c_space.shape[1]):
            if c_space[i][j] == -1:
                fig.gca().plot(i, j, 'ok')
            elif c_space[i][j] == 0:
                rectangle = plt.Rectangle((i, j), 1, 1, fc='white')
                fig.gca().add_patch(rectangle)

    fig.gca().plot(start[0], start[1], '*g')
    fig.gca().plot(end[0], end[1], '*r')

    plt.title('Result')
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig


# User defined values
P = 0.005
size = (20, 20)
start = (0, 0)
end = (size[0] - 1, size[1] - 1) # end node can be set manually, currently set as bottom right corner of the grid
end = (9, 19)

# main code
c_space = get_c_space(size, P)
fig = create_plot(c_space)
begin = time.time()
straight_vertices = RRT_star(start, end)
reverse_vertices = extend_RRT_direction(start, end, straight_vertices)
duration = time.time() - begin
fig = add_lines(straight_vertices, reverse_vertices, fig)
update_plot(c_space, fig, start, end)
plt.show(block=True)