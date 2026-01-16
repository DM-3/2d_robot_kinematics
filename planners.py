from collections.abc import Callable
from abc import ABC, abstractmethod
import numpy as np



class Planner(ABC):
    def __init__(self):
        self.finished = False

    @abstractmethod
    def difference(self, v1, v2) -> np.ndarray: pass

    def distance(self, v1, v2) -> float:
        return np.linalg.norm(self.difference(v1, v2))
    
    @abstractmethod
    def iterate(self) -> None: pass

    @abstractmethod
    def path(self) -> list[np.ndarray]: pass


class RRT(Planner):
    def __init__(self, start, target, shape, delta, step, tdist, sampler: Callable[[np.ndarray], bool]):
        self.vertices = [np.float64(start)]
        self.target = np.float64(target)
        self.edges = []
        self.shape = shape
        self.delta = delta
        self.step = step
        self.tdist = tdist
        self.sampler = sampler
        super().__init__()
    
    def difference(self, v1, v2):
        d = v1 - v2
        d = np.where(d > self.shape / 2, d - self.shape, d)
        d = np.where(d < -self.shape / 2, d + self.shape, d)
        return d

    def iterate(self):
        if self.finished:
            return
        
        # get new random sample point
        v_rand = (np.random.rand(len(self.shape)) - .5) * self.shape

        # sample target with a 10% chance
        if np.random.randint(0, 10) == 0:
            v_rand = self.target

        if self.sampler(v_rand):
            return

        # find known vertex closest to new sample
        v_near = min(self.vertices, key=lambda v: self.distance(v, v_rand))
        finish_candidate = False
        if self.distance(self.target, v_near) < self.tdist:
            finish_candidate = True
            v_rand = self.target

        # sample along direction towards new sample
        d = self.difference(v_rand, v_near)
        dist = np.linalg.norm(d)
        d = d / dist
        
        v_new = v_near + d * self.delta

        v_step = v_near.copy()
        total = 0.0
        add = False
        while True:

            v_step += d * self.step
            v_step = np.mod(v_step + self.shape, self.shape)
            total += self.step
            
            if self.sampler(v_step):
                break

            if total > self.delta:
                add = True
                break

        if add:
            self.vertices.append(v_new)
            self.edges.append((v_near, v_new))
        
        if finish_candidate:
            self.finished = True
            print("planner finished")

    def path(self):
        if len(self.edges) == 0:
            return []
        
        vs = self.edges[0][0]

        vertex_set = {}
        for v1, v2 in self.edges:
            vertex_set[tuple(v1)] = v1
            vertex_set[tuple(v2)] = v2

        prev = {}

        for v1, v2 in self.edges:
            prev[tuple(v2)] = v1

        vt = min(vertex_set.items(), key=lambda kvp: self.distance(kvp[1], self.target))[1]

        path = [vt]
        while True:
            last = tuple(path[-1])
            if not (last in prev.keys()):
                break
            path.append(prev[last])
            if np.allclose(path[-1], vs):
                break

        return path


class PRM(Planner):
    def __init__(self, start, target, shape, step, sampler: Callable[[np.ndarray], bool]):
        start = np.float64(start)
        target = np.float64(target)
        self.start = start
        self.target = target
        self.vertices = [start, target]
        self.edges = []
        self.shape = shape
        self.step = step
        self.sampler = sampler
        super().__init__()

    def difference(self, v1, v2):
        d = v1 - v2
        d = np.where(d > self.shape / 2, d - self.shape, d)
        d = np.where(d < -self.shape / 2, d + self.shape, d)
        return d

    def iterate(self):
        if self.finished:
            return
        
        # get a new random sample point
        v_rand = (np.random.rand(len(self.shape)) - .5) * self.shape
        if self.sampler(v_rand):
            return
        
        # find 3 closesst vertices
        closest = sorted(self.vertices, 
                         key=lambda v: self.distance(v, v_rand)
                         )[:min(3, len(self.vertices))]

        self.vertices.append(v_rand)
        
        # try connecting to the closest vertices
        for v_near in closest:
            d = self.difference(v_rand, v_near)
            dist = np.linalg.norm(d)
            d = d / dist
            total = 0.0
            v_step = v_near.copy()
            while total < dist:
                v_step += d * self.step
                v_step = np.mod(v_step + self.shape, self.shape)
                total += self.step
                if self.sampler(v_step):
                    break
            if total > dist:
                self.edges.append((v_rand, v_near))

        if len(self.path()) > 0:
            self.finished = True
            print("planner finished")

    def path(self):
        if len(self.edges) == 0:
            return []

        unvisited = [tuple(v) for v in self.vertices]

        neighbors = {}
        for v in unvisited:
            neighbors[v] = []
        for v1, v2 in self.edges:
            neighbors[tuple(v1)].append(tuple(v2))
            neighbors[tuple(v2)].append(tuple(v1))

        distances = {}
        for v in unvisited:
            distances[v] = np.inf
        distances[tuple(self.start)] = 0

        prev = {}
        for v in unvisited:
            prev[v] = None

        while len(unvisited) > 0:
            v = min(unvisited, key=lambda v: distances[v])
            unvisited.remove(v)
            d = distances[v]
            if d == np.inf:     # disconnected
                return []
            for n in neighbors[v]:
                if d + 1 < distances[n]:
                    distances[n] = d + 1
                    prev[n] = v
                if tuple(n) == tuple(self.target):
                    unvisited = []

        path = [tuple(self.target)]
        while True:
            p = prev[path[-1]]
            if p is not None:
                path.append(p)
            else:
                break
        
        return [np.array(v) for v in path]
