# -*- coding: utf-8 -*-

__all__ = [
    "get_compile_args",
    "get_cache_version",
    "get_header_dirs",
    "get_librebound_path",
    "get_librebound_name",
]

import os
import sys
import sysconfig
import rebound

from .rebound_pymc_version import __version__


def get_compile_args(compiler):
    opts = ["-std=c++11", "-O2", "-DNDEBUG"]
    if sys.platform == "darwin":
        opts += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
    else:
        libpath = get_librebound_path()
        opts.append("-Wl,-rpath={0}".format(libpath))
    return opts


def get_cache_version():
    if "dev" in __version__:
        return ()
    return tuple(map(int, __version__.split(".")))


def get_header_dirs():
    this_path = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(os.path.abspath(rebound.__file__))
    site = os.path.dirname(pkg)
    # REBOUND 4.x ships rebound.h inside the package; 5.x moved it to a
    # sibling src/ dir. Return whichever directory actually holds the header.
    hdr = [d for d in (pkg, os.path.join(site, "src"), site)
           if os.path.exists(os.path.join(d, "rebound.h"))]
    return [this_path] + (hdr or [pkg])


def get_librebound_path():
    return os.path.dirname(os.path.dirname(rebound.__file__))


def get_librebound_name():
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix is None:
        suffix = ".so"
    path = os.path.dirname(os.path.dirname(rebound.__file__))
    path = os.path.join(path, "librebound" + suffix)
    if not os.path.exists(path):
        raise RuntimeError("can't find librebound")
    return "rebound" + os.path.splitext(suffix)[0]
