from setuptools import setup
from Cython.Build import cythonize
import numpy

# python3 setup.py build_ext --inplace

setup(
    name='cy_fun',
    ext_modules=cythonize('cython_stuff.pyx', language_level=3),
    include_dirs=[numpy.get_include()],
    setup_requires=[
        'Cython',
        'NumPy',
    ],
    install_requires=[
        'NumPy',
    ],
)

