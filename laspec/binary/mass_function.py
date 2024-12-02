import numpy as np
import sympy
import astropy.constants as const
import astropy.units as u


def calculate_m2(
    Pday: float = 14.5,
    ideg: float = 90,
    m1: float = 2.16,
    Kkms: float = 50,
):
    """Calculate the mass of the secondary

    Parameters
    ----------
    Pday: float
        Period in days.
    ideg:
        Inclination in degrees.
    m1:
        Mass of the primary in Msun.
    Kkms
        Semi-amplitude K of the primary in km/s.

    Returns
    -------
    M2:
        Mass of the secondary in Msun.
    """
    Kms = Kkms * 1000.0
    Psec = Pday * 86400.0
    irad = np.deg2rad(ideg)

    m1kg = m1 * u.solMass.to(u.kg)
    m2kg = sympy.symbols("m2")
    a = sympy.solve(
        np.power(m2kg * np.sin(irad), 3) / np.power((m1kg + m2kg), 2)
        - Psec * np.power(Kms, 3) / (2 * np.pi * const.G.value),
        m2kg,
    )
    mass = a[0]
    return mass / u.solMass.to(u.kg)


if __name__ == "__main__":
    print(calculate_m2())
