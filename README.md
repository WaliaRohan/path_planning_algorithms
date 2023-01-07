# path_planning_algorithms
This repo containts path planning algorithms I wrote as part of my motion my planning course.

Each planner is implemented as a standalone python script. Dijkstra, A* and ANA* take an image, start location and goal location, and plot a path from start to goal on that image. Note that bi-directional rrt* (bi_rrt.py) doesn't take any image input. It shows the planner in action on a grid plot.

Use dijkstra.py and a_star.py as follows - 

python3 <script_name> 'relative_image_path' '(start_x, start_y)' '(goal_x, goal_y)' 'true/false'

The last argument, if true, prints the path as a list of (int, int) on command line. Each (int, int) value locates a waypoint along the path on the binary occupancy grid of the image. Start and goal locations for provided images are listed below.

ANA* and Bi-directional RRT don't require any inputs at the moment. To provide a different image, start or goal location, just change those parameters at the bottom of the script you want to use (ana_star.py or bi_rrt.py).

## Converting images to binary occupancy grid

1. OpenCV is used to load an image as a numpy array.
2. OpenCV is used again on the numpy array to obtain a greyscale version of the image.
3. To convert the grayscale numpy array to a binary occupancy grid, the pixel value of each cell in the greyscale numpy array is checked. If a cell value is greater that 255/2, it is considered to be 'free' (with a value of 1). Otherwise, it is marked as occupied (value of 0).

## Start-end locations for images in this repository

You can use the following start and end locations for the images in this repository -

- Very Trivial
    start = (40, 0) 
    goal = (180, 224)
- Trivial
    start = (1, 8)
    goal = (1, 20)
- Medium
    start = (8, 201)
    goal = (110, 1)
- Hard
    start = (10, 2)
    goal = (400, 220)
- Very Hard
    start = (1, 324)
    goal = (580, 1)

## Dependencies

- Dijkstra, A* and ANA*: CV2, Numpy

- Bi-directional RRT*: Numpy, Matlplotlib

## Known issues

- Dijkstra, A* and ANA*: These planners perform slower than usual. I am looking into whether this is due to plotting or a bug.

- Bi-directional rrt*: The RRT trees don't seem to find the goal in lower left portion of the grid. This is a bug that I am trying to fix.