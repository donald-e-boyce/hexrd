"""Transforms module.

Contains different implementations based on Python+Numpy, numba and
a supporting C module. All three should adhere to the same interface,
but performance will vary.

Use the functions under this module scope to use the preferred versions,
import the specific submodule if you want to use a specific version.
"""

# TODO: make public the default definitions
