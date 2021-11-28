from typing import Iterator

import ase
from ase import visualize


def view(atoms: ase.Atom, centre=True):
    viewer = visualize.view(atoms, viewer='x3d')
    # if not centre:
    # nglview.view.center(selection='0')
    return viewer


def inclusive(*args) -> Iterator[int]:
    """Like range() but inclusive of upper bound and automatically does iteration of ranges with a
    negative step e.g. 0, -4 will produce a range containing 0, -1, -2, -3, -4"""
    if len(args) not in (1, 2, 3):
        raise ValueError('Takes one or two args, got: {}'.format(args))

    if len(args) == 3:
        # Assume form is start, stop, step
        start, stop, step = args
    else:
        if len(args) == 1:
            start = 0
            stop = args[0]
        else:
            start = args[0]
            stop = args[1]

        step = 1 if start <= stop else -1

    sign = 1 if step > 0 else -1
    idx = start
    while sign * (stop - idx) >= 0:
        yield idx
        idx += step
