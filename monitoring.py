# -*- coding: utf-8 -*-
"""Outils de monitoring : temps (chrono propre) et memoire (echantillonneur).

- ``chrono`` / ``Stopwatch`` : mesure de temps propre, sans le motif bruyant
  ``t = perf_counter(); ...; dt = perf_counter() - t`` repete partout.
"""
import threading
import time
try:
    import resource                      # POSIX uniquement (absent sous Windows)
except ImportError:
    resource = None
from contextlib import contextmanager
from time import perf_counter        


_KEYS = ("VmRSS", "VmData", "VmStk", "VmSize", "VmPeak", "VmHWM")


def _read_status():
    out = {}
    try:
        f = open("/proc/self/status")
        for line in f:
            k = line.split(":")[0]
            if k in _KEYS:
                out[k] = int(line.split()[1])
        f.close()
    except IOError:
        pass
    return out


# --------------------------------------------------------------------------- #
#  MESURE DE TEMPS — propre.
# --------------------------------------------------------------------------- #
class Stopwatch(object):
    """Petit chronometre : ``sw = Stopwatch(); ...; sw.stop()`` -> ``sw.elapsed``."""

    def __init__(self):
        self.t0 = perf_counter()
        self.elapsed = None

    def stop(self):
        self.elapsed = perf_counter() - self.t0
        return self.elapsed


@contextmanager
def chrono(store=None, key=None):
    """Context manager de chronometrage.

    Usage simple ::

        with chrono() as c:
            ... travail ...
        print(c.elapsed)

    Avec collecte dans un dict ::

        times = {}
        with chrono(times, "build"):
            ... travail ...
        # times["build"] contient la duree
    """
    sw = Stopwatch()
    try:
        yield sw
    finally:
        sw.stop()
        if store is not None and key is not None:
            store[key] = sw.elapsed


# --------------------------------------------------------------------------- #
#  MESURE DE MEMOIRE — echantillonneur en arriere-plan.
# --------------------------------------------------------------------------- #
