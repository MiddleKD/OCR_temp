import cv2
import numpy as np
import heapq

def calculate_cost(image, start):
    h, w = image.shape
    cost = np.inf * np.ones((h, w))
    cost[start] = 0
    return cost

def neighbors(point):
    x, y = point
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def dijkstra(image, start, end):
    h, w = image.shape
    cost = calculate_cost(image, start)
    visited = np.zeros((h, w), dtype=np.bool_)
    pq = [(0, start)]

    while pq:
        current_cost, current_point = heapq.heappop(pq)
        if visited[current_point]:
            continue
        visited[current_point] = True

        if current_point == end:
            break

        for neighbor in neighbors(current_point):
            x, y = neighbor
            if x < 0 or x >= h or y < 0 or y >= w:
                continue

            new_cost = current_cost + abs(int(image[current_point]) - int(image[neighbor]))
            if new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor))

    return cost

def extract_path(cost, start, end):
    path = []
    current = end
    max_iterations = 1000
    iterations = 0

    while current != start:
        path.append(current)
        if iterations > max_iterations:
            print("Max iterations reached. Exiting.")
            break
        x, y = current
        current_cost = cost[current]
        for neighbor in neighbors(current):
            if cost[neighbor] < current_cost:
                current = neighbor
                current_cost = cost[current]

        iterations += 1

    return path

# Read image and convert to grayscale
image = cv2.imread('./data/arrow_rain.jpg', 0)

# Define start and end points
start = (10, 10)
end = (200, 200)

# Run Dijkstra's algorithm
cost = dijkstra(image, start, end)

# Extract the path
path = extract_path(cost, start, end)

# Draw the path on the image
for point in path:
    x, y = point
    image[x, y] = 255

# Show the image
cv2.imshow('Intelligent Scissors', image)
cv2.waitKey(0)
cv2.destroyAllWindows()