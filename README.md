# rebound-pymc

A [PyMC](https://www.pymc.io) (PyTensor) interface to the
[REBOUND](https://rebound.readthedocs.io) / [REBOUNDx](https://reboundx.readthedocs.io)
N-body integrator, providing differentiable custom `PyTensor` ops for evaluating
and differentiating N-body orbits inside a PyMC model.

This is a port of [`rebound-pymc3`](https://github.com/exoplanet-dev/rebound-pymc3)
(Daniel Foreman-Mackey) to the modern stack:

- **Theano/aesara → PyTensor** (PyMC v5)
- **REBOUND 3.x C API → REBOUND 4.x** (`reb_simulation_*`)
- **exoplanet >= 0.6** (`ReboundOrbit` subclasses the PyTensor `KeplerianOrbit`)

Tested against `pytensor` 2.26, `rebound` 4.4, `exoplanet` 0.6, `pymc` 5.18
(Python 3.10). The C `IntegrateOp` reproduces a direct REBOUND integration to
machine precision and supplies analytic gradients via REBOUND variational particles.

## Install

```bash
pip install -e .
```

Requires a C++ compiler (the `IntegrateOp` is compiled at runtime by PyTensor and
links against `librebound`).

## Usage

```python
import numpy as np, pytensor, pytensor.tensor as pt
from rebound_pymc.orbit import ReboundOrbit

orbit = ReboundOrbit(period=8.8, t0=1.0, m_planet=1e-3, m_star=1.0,
                     r_star=1.0, ecc=0.1, omega=0.3, incl=1.5, Omega=0.0)
x, y, z = orbit.get_relative_position(pt.as_tensor_variable(np.linspace(0, 20, 60)))
```

See `tests/` for the low-level `IntegrateOp` / `ReboundOp` API.

## Credit

Original: Daniel Foreman-Mackey (`exoplanet-dev/rebound-pymc3`, GPLv3).
