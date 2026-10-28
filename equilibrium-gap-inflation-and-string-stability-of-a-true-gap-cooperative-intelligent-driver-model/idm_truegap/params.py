"""Parameter set used throughout the study.

Values of ``a``, ``b``, ``s0``, ``delta`` and ``v0`` lie in the highway
range used with the enhanced IDM (Kesting, Treiber and Helbing, 2010).
They are a modelling choice, not a new statistical fit to OpenACC.
The mechanical braking bound ``b_max`` is distinct from the comfortable
deceleration ``b`` that enters the IDM desired gap.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    v0: float = 120.0 / 3.6       # desired free speed [m/s]
    a: float = 1.4                # comfortable acceleration [m/s^2]
    b: float = 2.0                # comfortable deceleration in s* [m/s^2]
    s0: float = 2.0               # jam gap [m]
    delta: float = 4.0            # free-acceleration exponent
    length: float = 5.0           # vehicle length [m]
    T: float = 0.6                # time-gap policy [s]
    b_max: float = 5.0            # mechanical braking bound [m/s^2]
    a_max: float = 2.0            # mechanical acceleration bound [m/s^2]
    tau_act: float = 0.10         # first-order driveline lag [s]
    ka: float = 1.0               # acceleration feedforward gain
    lam: float = 1.6              # beta transition location, in units of s_cf
    gamma: float = 0.35           # beta slope [1/m]
    v_cap: float = 45.0           # supervisory cap on virtual desired speed [m/s]
    c_eidm: float = 0.99          # eIDM coolness
    # Class-K gain. Equilibrium slack is about v(T - tau_r); kappa must
    # exceed v/h_eq, otherwise the filter brakes during steady cruising.
    kappa_cbf: float = 3.0
    # PATH-style linear CACC gains (Ploeg et al., ITSC 2011 experimental set)
    kp: float = 0.2
    kd: float = 0.7


DEFAULT = Params()


def with_updates(p: Params = DEFAULT, **kwargs) -> Params:
    return replace(p, **kwargs)
