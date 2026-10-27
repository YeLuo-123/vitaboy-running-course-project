"""A* on an inflated known-free grid, with heading state and corner checks."""
import heapq
import math


def astar_grid(blocked,width,height,start,goal,turn_gain=2.):
    sx,sy,heading=start;gx,gy=goal
    def free(x,y):return 0<=x<width and 0<=y<height and not blocked[y*width+x]
    if not free(sx,sy) or not free(gx,gy):return None
    moves=[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    angles=[math.atan2(dy,dx) for dx,dy in moves]+[heading]
    costs=[[math.hypot(dx,dy)*(1+turn_gain*abs(math.atan2(math.sin(a-previous),math.cos(a-previous))))
            for (dx,dy),a in zip(moves,angles)] for previous in angles]
    initial=(sx,sy,8);dist={initial:0.};parent={}
    queue=[(math.hypot(gx-sx,gy-sy),0.,initial)]
    while queue:
        _,g,state=heapq.heappop(queue)
        if g>dist.get(state,math.inf):continue
        x,y,direction=state
        if (x,y)==(gx,gy):
            path=[]
            while True:
                path.append(state[:2])
                if state==initial:break
                state=parent[state]
            return path[::-1]
        for i,(dx,dy) in enumerate(moves):
            nx,ny=x+dx,y+dy
            if not free(nx,ny) or (dx and dy and (not free(x+dx,y) or not free(x,y+dy))):continue
            target=(nx,ny,i);ng=g+costs[direction][i]
            if ng<dist.get(target,math.inf):
                dist[target]=ng;parent[target]=state
                heapq.heappush(queue,(ng+math.hypot(gx-nx,gy-ny),ng,target))
    return None
