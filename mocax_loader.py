# -*- coding: utf-8 -*-
"""Detail technique : rendre ``import mocaxpy`` fonctionnel dans le processus.

Aucune logique applicative ici. On localise le package Python ``mocaxpy`` et on
pre-charge la lib native ``libmocaxc.so``, une seule fois (cache module).

    from mocax_loader import load_mocax
    mocaxpy = load_mocax()                      # racines par defaut : ~/mocax
    mocaxpy = load_mocax(search_roots=["/opt/mocax"])
"""
from __future__ import print_function, division

import os
import sys
import ctypes

_MODULE = None   # cache : evite de re-parcourir le disque a chaque appel


def load_mocax(search_roots=None, native_lib="libmocaxc.so", verbose=True):
    """Localise, pre-charge et importe ``mocaxpy`` ; renvoie le module."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    roots = list(search_roots) if search_roots else \
        [os.path.join(os.path.expanduser("~"), "mocax")]

    # 1) Rendre le package importable (ajout au sys.path).
    parent = _find_package(roots)
    if parent is None:
        raise RuntimeError(
            "Module 'mocaxpy' introuvable.\n"
            " -> Decompresse le binding Python (mocaxpy-4.3.1.linux-x86_64.zip)\n"
            " -> ou passe search_roots=[...] a load_mocax().")
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if verbose:
        print("mocaxpy trouve dans :", parent)

    # 2) Pre-charger la lib native en RTLD_GLOBAL pour le wrapper ctypes.
    so_path = _find_file(native_lib, roots)
    if so_path:
        try:
            ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            if verbose:
                print(native_lib, "pre-chargee :", so_path)
        except OSError as exc:
            if verbose:
                print("Avertissement: pre-chargement echoue :", exc)
    elif verbose:
        print("Avertissement:", native_lib, "non trouvee (l'import peut echouer).")

    # 3) Import effectif.
    import mocaxpy
    _MODULE = mocaxpy
    if verbose:
        print("mocaxpy importe. Version :", getattr(mocaxpy, "__version__", "n/a"))
    return mocaxpy


def _find_package(roots):
    """Repertoire a ajouter au sys.path pour reussir 'import mocaxpy'."""
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if os.path.basename(dirpath) == "mocaxpy" and "__init__.py" in files:
                return os.path.dirname(dirpath)   # package -> son parent
            if "mocaxpy.py" in files:
                return dirpath                     # module simple
    return None


def _find_file(name, roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    return None
