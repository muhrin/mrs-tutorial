author_info = (('Martin Uhrin', 'martin.uhrin.10@ucl.ac.uk'),
               ('Thomas J. Hardin', 'tjhardi@sandia.gov'),
               ('Tess E. Smidt', 'tsmidt@mit.edu'))
version_info = (0, 1, 0)

__author__ = ", ".join("{} <{}>".format(*info) for info in author_info)
__version__ = ".".join(map(str, version_info))

__all__ = ('__version__',)
