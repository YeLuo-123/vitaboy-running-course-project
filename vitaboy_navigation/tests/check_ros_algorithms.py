"""Run under the ROS system Python after sourcing Humble."""
from types import SimpleNamespace
from vitaboy_navigation.navigation import Planner

free = SimpleNamespace(is_occ=lambda x,y: not (0 <= x < 5 and 0 <= y < 5))
assert Planner.grid_line_clear(free,(0,0),(3,3))
corner = SimpleNamespace(is_occ=lambda x,y: (x,y)==(1,0) or not (0 <= x < 5 and 0 <= y < 5))
assert not Planner.grid_line_clear(corner,(0,0),(2,2))
wall = SimpleNamespace(is_occ=lambda x,y: x==2 or not (0 <= x < 5 and 0 <= y < 5))
assert not Planner.grid_line_clear(wall,(0,1),(4,1))
assert Planner.grid_line_clear(free,(0,0),(0,0))
print('ROS planner geometry: 4 checks passed')

# Compare accelerated circular inflation with the original implementation.
import numpy as np
from vitaboy_navigation._g1.dijkstra_planner import DijkstraPlanner
for seed in range(4):
    rng=np.random.default_rng(seed)
    occ=rng.choice([-1,0,100],size=23*19,p=[.15,.8,.05]).tolist()
    for radius in [0,1,3,7]:
        assert Planner.inflate_occupancy(None,occ,23,19,radius,50)==DijkstraPlanner.inflate_occupancy(None,occ,23,19,radius,50)
print('Circular inflation equivalence: 16 randomized grid checks passed')
