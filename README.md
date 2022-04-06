# path_planning_algorithms
This repo containts path planning algorithms I wrote as part of my motion my planning course.

Each planner is implemented as a standalone python script. Dijkstra, A* and ANA* take an image, start location and goal location, and plot a path from start to goal on that image. Note that bi-directional rrt* (bi_rrt.py) doesn't take any image input. It shows the planner in action on a grid plot.


All scripts can be run without using any arguments. To provide a different image, start or goal location, just change those parameters at the bottom of the script you want to use.

## Converting images to binary occupancy grid

1. OpenCV is used to load an image as a numpy array.
2. OpenCV is used again on the numpy array to obtain a greyscale version of the image.
3. To convert the grayscale numpy array to a binary occupancy grid, the pixel value of each cell in the greyscale numpy array is checked. If a cell value is greater that 255/2, it is considered to be 'free' (with a value of 1). Otherwise, it is marked as occupied (value of 0).
## Dependencies

- Dijkstra, A* and ANA*: CV2, Numpy

- Bi-directional RRT*: Numpy, Matlplotlib

## Known issues

- Dijkstra, A* and ANA*: These planners perform slower than usual. I am looking into whether this is due to plotting or a bug.

- Bi-directional rrt*: The RRT trees don't seem to find the goal in lower left portion of the grid. This is a bug that I am trying to fix.