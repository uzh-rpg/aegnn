"""Implementation based on QuadTree shown in https://scipython.com/blog/quadtrees-2-implementation-in-python/
For speedup, the OctTree has been implemented in Cython. Therefore, when altering the code, it has to be
re-compiled, as described in the Readme. The code is available from the module `aegnn.octree`.
"""
from typing import Tuple, Union


cdef class Box:

    cdef double cx, cy, cz
    cdef double w, h, d
    cdef double x_min, x_max, y_min, y_max, z_min, z_max

    def __init__(self, cx: float, cy: float, cz: float, width: float, height: float, depth: float):
        self.cx, self.cy, self.cz = cx, cy, cz
        self.w, self.h, self.d = width, height, depth

        self.x_min, self.x_max = cx - width / 2, cx + width / 2
        self.y_min, self.y_max = cy - height / 2, cy + height / 2
        self.z_min, self.z_max = cz - depth / 2, cz + depth / 2

    def contains(self, point: Tuple[float, float, float]):
        px, py, pz = point
        in_x = self.x_min <= px < self.x_max
        in_y = self.y_min <= py < self.y_max
        in_z = self.z_min <= pz < self.z_max
        return in_x and in_y and in_z

cdef class OctTree:

    cdef int tree_id
    cdef public Box boundary
    cdef int num_points, max_points

    cdef list divisions

    def __init__(self, boundary: Box, max_points: int = 4):
        self.tree_id = hash((boundary.cx, boundary.cy, boundary.cz))  # boundary center is unique for each tree (!)
        self.boundary = boundary

        self.num_points = 0
        self.max_points = max_points
        self.divisions = []

    def divide(self):
        cx, cy, cz = self.boundary.cx, self.boundary.cy, self.boundary.cz
        w, h, d = self.boundary.w / 2, self.boundary.h / 2, self.boundary.d / 2

        self.divisions = [
            OctTree(Box(cx - w / 2, cy - h / 2, cz - d / 2, w, h, d), max_points=self.max_points),
            OctTree(Box(cx + w / 2, cy - h / 2, cz - d / 2, w, h, d), max_points=self.max_points),
            OctTree(Box(cx - w / 2, cy + h / 2, cz - d / 2, w, h, d), max_points=self.max_points),
            OctTree(Box(cx + w / 2, cy + h / 2, cz - d / 2, w, h, d), max_points=self.max_points),
            OctTree(Box(cx - w / 2, cy - h / 2, cz + d / 2, w, h, d), max_points=self.max_points),
            OctTree(Box(cx + w / 2, cy - h / 2, cz + d / 2, w, h, d), max_points=self.max_points),
            OctTree(Box(cx - w / 2, cy + h / 2, cz + d / 2, w, h, d), max_points=self.max_points),
            OctTree(Box(cx + w / 2, cy + h / 2, cz + d / 2, w, h, d), max_points=self.max_points),
        ]

    def insert(self, point: Tuple[float, float, float]) -> bool:
        if not self.boundary.contains(point):
            return False
        if self.num_points < self.max_points:
            self.num_points += 1
            return True

        if len(self.divisions) == 0:
            self.divide()
        return any([div.insert(point) for div in self.divisions])

    def assign(self, point: Tuple[float, float, float]) -> Union[int, None]:
        if len(self.divisions) == 0 and self.boundary.contains(point):
            return self.tree_id

        for div in self.divisions:
            if div.boundary.contains(point):
                return div.assign(point)

        return None

    def __len__(self) -> int:
        return len(self.divisions) + sum([len(div) for div in self.divisions], 0)
