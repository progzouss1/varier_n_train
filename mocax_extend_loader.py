# -*- coding: utf-8 -*-
"""Loader minimaliste : rend ``import mocaxextendpy.mocax_extend`` fonctionnel.

    from mocax_extend_loader import load_mocax_extend
    me = load_mocax_extend()  # cherche sous ~/mocax par defaut
"""
import os
import sys
import ctypes

_MODULE = None
_NATIVES = ("libtensorvals.so", "libhommat.so")


def load_mocax_extend(search_roots=None):
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    roots = list(search_roots or [os.path.join(os.path.expanduser("~"), "mocax")])

    # 1) package importable
    parent = _find(roots, lambda d, files: os.path.basename(d) == "mocaxextendpy"
                   and "__init__.py" in files, as_dir=True)
    if parent is None:
        raise RuntimeError("Package 'mocaxextendpy' introuvable sous %s" % roots)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # 2) natifs (chemins absolus)
    natives = {}
    for name in _NATIVES:
        p = _find(roots, lambda d, files, n=name: n in files, join=name)
        if p is None:
            raise RuntimeError("Natif '%s' introuvable sous %s" % (name, roots))
        natives[name] = p

    # 3) precharger en RTLD_GLOBAL tous les .so des dossiers natifs
    #    (passes multiples : deps inter-.so, dont l'OpenBLAS versionne)
    lib_dirs = sorted({os.path.dirname(p) for p in natives.values()})
    pending = [os.path.join(d, f) for d in lib_dirs
               for f in sorted(os.listdir(d)) if ".so" in f]
    progress, last_err = True, {}
    while pending and progress:
        progress, still = False, []
        for p in pending:
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                progress = True
            except OSError as exc:
                last_err[p] = exc
                still.append(p)
        pending = still
    blocking = [p for p in pending if p in natives.values()]
    if blocking:
        raise last_err[blocking[0]]

    # 4) import : rediriger les noms nus vers les chemins absolus
    _orig = ctypes.cdll.LoadLibrary
    ctypes.cdll.LoadLibrary = lambda n: _orig(natives.get(os.path.basename(n), n))
    try:
        import mocaxextendpy.mocax_extend as me
    finally:
        ctypes.cdll.LoadLibrary = _orig

    _MODULE = me
    return me


def _find(roots, match, as_dir=False, join=None):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if match(dirpath, files):
                if as_dir:
                    return os.path.dirname(dirpath)
                return os.path.join(dirpath, join)
    return None