import numpy as np
import matplotlib.pyplot as plt

def get_neighborhood(cell, occ_map_shape):
  '''
  Arguments:
  cell -- cell coordinates as [y, x]
  occ_map_shape -- shape of the occupancy map (ny, nx)

  Output:
  neighbors -- list of up to eight neighbor coordinate tuples [(y1, x1), (y2, x2), ...]
  '''

  neighbors = []
  
  y, x = cell
  ny, nx = occ_map_shape

  offsets = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
  ]

  for dy, dx in offsets:
    ny_ = y + dy
    nx_ = x + dx

    # Casos bordes
    if 0 <= ny_ < ny and 0 <= nx_ < nx:
      neighbors.append((ny_, nx_))




  return neighbors

def get_edge_cost(parent, child, occ_map):
  '''
  Calculate cost for moving from parent to child.

  Arguments:
  parent, child -- cell coordinates as [y, x]
  occ_map -- occupancy probability map

  Output:
  edge_cost -- calculated cost
  '''
  
  edge_cost = 0
  
  threshold = 0.5  # Umbral para considerar si está ocupada la celda o no
  k = 5.0 # Peso que le doy a la probabilidad de que esté ocupada cuando calculo el costo

  py, px = parent
  cy, cx = child

  if occ_map[py, px] > threshold or occ_map[cy, cx] > threshold:
    return float('inf') # Costo infinito por moverse a través de un obstáculo

  dy = cy - py
  dx = cx - px
  geom_cost = (dy**2 + dx**2)**0.5 # Defino el costo como la distancia euclídea
  occ_cost = k * occ_map[cy, cx]


  return geom_cost * occ_cost

def get_heuristic(cell, goal, h2_factor=1):
  '''
  Estimate cost for moving from cell to goal based on heuristic.

  Arguments:
  cell, goal -- cell coordinates as [y, x]
  h2_factor -- factor por el cual multiplico la heurística calculada

  Output:
  cost -- estimated cost
  '''
  
  heuristic = 0
 
  # Para que A* sea óptimo, la heurística debe:
  # 1. Ser admisible, osea que nunca debe sobreestimar el costo real para llegar al objetivo.
  # 2. Ser consistente, lo cual es que para cada nodo parent n y su/s hijo/s n', la estimación heurística 
  #    de n debe ser menor o igual al costo del arco de n a n' más la estimación heurística de n'.
  #    Igual, si una heurística es consistente, entonces es admisible también.

  # Multiplicar la heurística, en este caso por el h2_factor, cambia el comportamiento del A*:
  # - Si h2_factor = 1, A* es óptimo y usa la heurística admisible, expandiendo más nodos en general y garantizando el camino de menor costo.
  # - Si h2_factor > 1, como vi con los valores probados 2, 5 y 10, A* pierde optimalidad y expande menos nodos, priorizando la velocidad de
  #   búsqueda del goal sobre la calidad del camino encontrado. Puede ser que encuentre caminos más cortos, pero el costo total de estos es mayor.

  y, x = cell
  gy, gx = goal
  heuristic = np.sqrt((y - gy)**2 + (x - gx)**2)

  return h2_factor * heuristic

def plot_map(occ_map, start, goal):
  plt.imshow(occ_map.T, cmap=plt.cm.gray, interpolation='none', origin='upper')
  plt.plot([start[0]], [start[1]], 'ro')
  plt.plot([goal[0]], [goal[1]], 'go')
  plt.axis([0, occ_map.shape[0]-1, 0, occ_map.shape[1]-1])
  plt.xlabel('x')
  plt.ylabel('y')

def plot_expanded(expanded, start, goal):
  if np.array_equal(expanded, start) or np.array_equal(expanded, goal):
    return
  plt.plot([expanded[0]], [expanded[1]], 'yo')
  plt.pause(1e-6)

def plot_path(path, goal):
  if np.array_equal(path, goal):
    return
  plt.plot([path[0]], [path[1]], 'bo')
  plt.pause(1e-6)

def plot_costs(cost):
  plt.figure()
  plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='upper')
  plt.axis([0, cost.shape[0]-1, 0, cost.shape[1]-1])
  plt.xlabel('x')
  plt.ylabel('y')

def run_path_planning(occ_map, start, goal):
  '''
  This implements the
  - Dikstra algorithm (in case heuristic is 0)
  - A* algorithm (in case heuristic is not 0)
  '''
 
  plot_map(occ_map, start, goal)

  # cost values for each cell, filled incrementally. 
  # Initialize with infinity
  costs = np.ones(occ_map.shape) * np.inf
  
  # cells that have already been visited
  closed_flags = np.zeros(occ_map.shape)
  
  # store predecessors for each visited cell 
  predecessors = -np.ones(occ_map.shape + (2,), dtype=int)

  # heuristic for A*
  heuristic = np.zeros(occ_map.shape)
  for x in range(occ_map.shape[0]):
    for y in range(occ_map.shape[1]):
      heuristic[x, y] = get_heuristic([x, y], goal, h2_factor=1)

  # start search
  parent = start
  costs[start[0], start[1]] = 0

  # loop until goal is found
  while not np.array_equal(parent, goal):
    
    # costs of candidate cells for expansion (i.e. not in the closed list)
    open_costs = np.where(closed_flags==1, np.inf, costs) + heuristic

    # find cell with minimum cost in the open list
    x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)
    
    # break loop if minimal costs are infinite (no open cells anymore)
    if open_costs[x, y] == np.inf:
      break
    
    # set as parent and put it in closed list
    parent = np.array([x, y])
    closed_flags[x, y] = 1;
    
    # update costs and predecessor for neighbors

    neighbors = get_neighborhood(parent, occ_map.shape)

    for child in neighbors:
      cy, cx = child
        
      # No actualizo si el nodo hijo ya está cerrado
      if closed_flags[cy, cx] == 1:
        continue

      # Calculo el costo de arco
      edge_cost = get_edge_cost(parent, child, occ_map)

      # Si la celda es intransitable, saltar
      if edge_cost == float('inf'):
        continue

      # Costo acumulado desde el inicio hasta child
      new_cost = costs[parent[0], parent[1]] + edge_cost

      # Actualizar si el nuevo costo es menor
      if new_cost < costs[cy, cx]:
        costs[cy, cx] = new_cost
        predecessors[cy, cx] = parent

    #visualize grid cells that have been expanded
    plot_expanded(parent, start, goal)
  
  # rewind the path from goal to start (at start predecessor is [-1,-1])
  if np.array_equal(parent, goal):
    path_length = 0
    while predecessors[parent[0], parent[1]][0] >= 0:
      plot_path(parent, goal)
      predecessor = predecessors[parent[0], parent[1]]
      path_length += np.linalg.norm(parent - predecessor)
      parent = predecessor

    print("found goal     : " + str(parent) )
    print("cells expanded : " + str(np.count_nonzero(closed_flags)) )
    print("path cost      : " + str(costs[goal[0], goal[1]]) )
    print("path length    : " + str(path_length) )
  else:
    print("no valid path found")

  #plot the costs 
  plot_costs(costs)
  plt.waitforbuttonpress()

def main():
  # load the occupancy map
  occ_map = np.loadtxt('map.txt')
  
  # start and goal position [x, y]
  start = np.array([22, 33])
  goal = np.array([40, 15])

  run_path_planning(occ_map, start, goal)

if __name__ == "__main__":
  main()
