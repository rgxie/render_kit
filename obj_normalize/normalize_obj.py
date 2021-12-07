# encoding:utf-8
import os
import trimesh
import numpy as np
import glob2

class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class DirTransfer(object):
    folders = []
    outFolders = []
    count = 0

    def get_obj(self, dir_path):
        # print("dir_path:", dir_path)
        if dir_path[-1] == '/':
            dir_path = dir_path[:-1]
        sub_name = dir_path.split('/')[-1]
        # print(sub_name)
        obj_path = os.path.join(dir_path, sub_name + '.obj')

        return obj_path


class Normalize(object):
    # def __init__(self, x):
    #     self.x =
    minP = Point(1000000, 1000000, 1000000)
    maxP = Point(-1000000, -1000000, -1000000)

    def get_bounding_box(self, p):
        """
        获取物体的最小x,y,z和最大的x,y,z
        :param p:
        :return:
        """
        self.minP.x = p.x if p.x < self.minP.x else self.minP.x
        self.minP.y = p.y if p.y < self.minP.y else self.minP.y
        self.minP.z = p.z if p.z < self.minP.z else self.minP.z
        self.maxP.x = p.x if p.x > self.maxP.x else self.maxP.x
        self.maxP.y = p.y if p.y > self.maxP.y else self.maxP.y
        self.maxP.z = p.z if p.z > self.maxP.z else self.maxP.z

    def get_bounding_box_length(self):
        """
        获取包围盒的最大长度
        :return:
        """
        box_len = self.maxP.x - self.minP.x
        if box_len < (self.maxP.y - self.minP.y):
            box_len = self.maxP.y - self.minP.y
        if box_len < (self.maxP.z - self.minP.z):
            box_len = self.maxP.z - self.minP.z
        return box_len

    def do_normalize(self, box_len, points):
        """
        归一化处理
        :param center_p: 物体的中心点
        :param box_len: 包围盒的一半
        :param points:要进行归一化处理的点
        :return:
        """
        new_points = []
        for point in points:
            x = (point.x - self.minP.x) * 2 / box_len - 1
            y = (point.y - self.minP.y) * 2 / box_len - 1
            z = (point.z - self.minP.z) * 2 / box_len - 1
            # new_points.append(Point(x, y, z))
            new_points.append([x,y,z])
        return new_points

    def read_points(self, mesh):
        """
        读取一个obj文件里的点
        :param filename:
        :return:
        """
        points = []
        vertice = mesh.vertices
        for v in vertice:
            points.append(Point(float(v[0]), float(v[1]), float(v[2])))
        return points


def normalize_mesh(obj_path):
    normalize = Normalize()
    normalize.minP = Point(1000000, 1000000, 1000000)
    normalize.maxP = Point(-1000000, -1000000, -1000000)
    
    print("obj_path:", obj_path)
    print("count:", count)
    mesh = trimesh.load(obj_path, process=False)
    print("mesh:", mesh.vertices)
    points = normalize.read_points(mesh)

    for point in points:
        normalize.get_bounding_box(point)
    boxLength = normalize.get_bounding_box_length()
    points = normalize.do_normalize(boxLength, points)
    
    mesh.vertices = points
    mesh.export(obj_path)


def normlize_dir(basePath):
    input_dirs = [dirnames for _, dirnames, _ in os.walk(basePath)]
    print(len(input_dirs[0]))
    trans=DirTransfer()
    count = 0
    for subject_name in input_dirs[0]:
        count += 1
        obj_path = trans.get_obj(os.path.join(basePath, subject_name))
        normalize_mesh(obj_path)
        print("obj_path:{0}|count:{1}|".format(obj_path, count))


if __name__ == "__main__":

    basePath = "/Users/rgxie/Projects/stanford-shapenet-renderer/objtest/xiandai"
    
    path_vec = glob2.glob(r"{0}/*.obj".format(basePath))
    count = 0
    for obj_path in path_vec:
        count += 1
        normalize_mesh(obj_path)
        print("obj_path:{0}|count:{1}|".format(obj_path, count))




