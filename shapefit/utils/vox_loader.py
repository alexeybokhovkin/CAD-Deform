import os
import struct
import numpy as np


class Vox:
    def __init__(self, dims=[0, 0, 0], res=0, grid2world=None, sdf=None, pdf=None):
        self.filename = ""
        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]
        self.res = res
        self.grid2world = grid2world
        self.sdf = sdf
        self.pdf = pdf


def load_sample(filename):
    assert os.path.isfile(filename), "file not found: %s" % filename
    if filename.endswith(".df"):
        f_or_c = "C"
    else:
        f_or_c = "F"

    fin = open(filename, 'rb')

    s = Vox()
    s.filename = filename
    s.dimx = struct.unpack('I', fin.read(4))[0]
    s.dimy = struct.unpack('I', fin.read(4))[0]
    s.dimz = struct.unpack('I', fin.read(4))[0]
    s.res = struct.unpack('f', fin.read(4))[0]
    n_elems = s.dimx * s.dimy * s.dimz

    s.grid2world = struct.unpack('f' * 16, fin.read(16 * 4))
    sdf_bytes = fin.read(n_elems * 4)
    try:
        s.sdf = struct.unpack('f' * n_elems, sdf_bytes)
    except struct.error:
        print("Cannot load", filename)
        s.sdf = np.ones((1, s.dimz, s.dimy, s.dimx), dtype=np.float32) * -0.15

    pdf_bytes = fin.read(n_elems * 4)
    if pdf_bytes:
        s.pdf = struct.unpack('f' * n_elems, pdf_bytes)
    fin.close()
    s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order=f_or_c)
    s.sdf = np.asarray(s.sdf, dtype=np.float32).reshape([1, s.dimz, s.dimy, s.dimx])
    if pdf_bytes:
        s.pdf = np.asarray(s.pdf, dtype=np.float32).reshape([1, s.dimz, s.dimy, s.dimx])
    else:
        s.pdf = np.zeros((1, s.dimz, s.dimy, s.dimx), dtype=np.float32)

    return s

def vox2obj(vox, res=None):
    if res is None:
        res = vox.res
    obj = []
    for i, x in enumerate(np.linspace(0, 1, vox.dimx)):
        for j, y in enumerate(np.linspace(0, 1, vox.dimy)):
            for k, z in enumerate(np.linspace(0, 1, vox.dimz)):
                if abs(vox.sdf[0, k, j, i]) < res:
                    obj.append([i, j, k])
    obj = np.array(obj)
    if len(obj) == 0:
        return vox2obj(vox, 2*res)
    obj = obj @ vox.grid2world[:3,:3] + vox.grid2world[:3,-1]
    obj = np.hstack((obj, np.ones((len(obj), 1))))
    return obj