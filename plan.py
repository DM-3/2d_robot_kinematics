from collections.abc import Callable
import numpy as np



class RRT:

    def __init__(self, start, target, shape, delta, step, tdist, sampler: Callable[[np.ndarray], bool]):
        self.vertices = [np.float64(start)]
        self.target = np.float64(target)
        self.edges = []
        self.shape = shape
        self.delta = delta
        self.step = step
        self.tdist = tdist
        self.sampler = sampler
        self.finished = False

    
    def __wrapDiff(self, v1, v2):
        d = v1 - v2
        d = np.where(d > self.shape / 2, d - self.shape, d)
        d = np.where(d < -self.shape / 2, d + self.shape, d)
        return d


    def __distance(self, v1, v2):
        return np.linalg.norm(self.__wrapDiff(v1, v2))


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
        v_near = min(self.vertices, key=lambda v: self.__distance(v, v_rand))
        finish_candidate = False
        if self.__distance(self.target, v_near) < self.tdist:
            finish_candidate = True
            v_rand = self.target

        # sample along direction towards new sample
        d = self.__wrapDiff(v_rand, v_near)
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

        vt = min(vertex_set.items(), key=lambda kvp: self.__distance(kvp[1], self.target))[1]

        path = [vt]
        while True:
            last = tuple(path[-1])
            if not (last in prev.keys()):
                break
            path.append(prev[last])
            if np.allclose(path[-1], vs):
                break

        return path

