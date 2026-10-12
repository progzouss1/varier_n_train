# -*- coding: utf-8 -*-
"""Cache disque : ecritures atomiques, JSON tolerant a numpy, empreintes.

Arborescence (racine = cfg.run.cache_dir) :

  scenarios/<fp>/            G.npy, S_<h>.npy, S0.npy, manifest.json
  full_repricing/<fp>/       manifest.json (produits, M, seed, horizons...)
      s0/  p001.npy ...      prix unitaire en s0     + p001_times.npy + p001.json
      1j/  p001.npy ...      N prix unitaires        + temps de CHAQUE pricing
      10j/ ...
  proxies/d05_full_cheb_<fp>/ nodes.npy, node_values.npy, node_times.npy,
                              build.json, eval_<h>.npy, eval.json, obj.pkl
  convergence/<fp>.json
  results/<fp>/              tableaux CSV/JSON de synthese, figures

Toute ecriture passe par un fichier temporaire + os.replace : un arret brutal
(Ctrl-C, coupure) ne laisse jamais de fichier a moitie ecrit.
"""

import datetime as _dt
import json
import os
import socket

import numpy as np


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_npy(path, arr):
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp.npy"
    np.save(tmp, np.asarray(arr))
    os.replace(tmp, path)


def load_npy(path, mmap=False):
    return np.load(path, mmap_mode="r" if mmap else None)


def save_json(path, obj):
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=_NpEncoder, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_npz(path, **arrays):
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def stamp():
    """Horodatage + machine : a joindre a chaque mesure de temps."""
    return dict(timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
                host=socket.gethostname())


def save_df(df, path_no_ext):
    """DataFrame -> CSV + JSON (lisible hors Python)."""
    ensure_dir(os.path.dirname(path_no_ext) or ".")
    df.to_csv(path_no_ext + ".csv")
    tmp = path_no_ext + ".json.tmp"
    df.reset_index().to_json(tmp, orient="records", indent=1, force_ascii=False)
    os.replace(tmp, path_no_ext + ".json")
