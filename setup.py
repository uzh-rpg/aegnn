from distutils.core import Extension, setup
from Cython.Build import cythonize

ext = Extension(name="aegnn.octree", sources=["aegnn/utils/oct_tree.pyx"])
setup(ext_modules=cythonize(ext))
