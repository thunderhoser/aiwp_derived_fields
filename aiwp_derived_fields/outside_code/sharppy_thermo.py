"""Thermodynamic calculations from SHARPpy.

Copy-pasta from thermo.py in version 1.4.0 of SHARPpy:
https://github.com/sharppy/SHARPpy/releases/tag/v1.4.0
"""

import numpy
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv

HPA_TO_PASCALS = 100.
KAPPA = 2. / 7
ZEROCNK = 273.15


def wobf(t):
    '''
    Implementation of the Wobus Function for computing the moist adiabats.

    .. caution::
        The Wobus function has been found to have a slight
        pressure dependency (Davies-Jones 2008).  This dependency
        is not included in this implementation.  

    Parameters
    ----------
    t : number, numpy array
        Temperature (C)

    Returns
    -------
    Correction to theta (C) for calculation of saturated potential temperature.

    '''
    t = t - 20
    try:
        # If t is a scalar
        if t is numpy.ma.masked:
            return t
        if t <= 0:
            npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
            npol = 15.13 / (numpy.power(npol,4))
            return npol
        else:
            ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
            ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
            ppol = (29.93 / numpy.power(ppol,4)) + (0.96 * t) - 14.8
            return ppol
    except ValueError:
        # If t is an array
        npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
        npol = 15.13 / (numpy.power(npol,4))
        ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
        ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
        ppol = (29.93 / numpy.power(ppol,4)) + (0.96 * t) - 14.8
        correction = numpy.zeros(t.shape, dtype=numpy.float64)
        correction[t <= 0] = npol[t <= 0]
        correction[t > 0] = ppol[t > 0]
        return correction


def satlift(p, thetam, conv=0.1):
    '''
    Returns the temperature (C) of a saturated parcel (thm) when lifted to a
    new pressure level (hPa)

    .. caution::
        Testing of the SHARPpy parcel lifting routines has revealed that the
        convergence criteria used the SHARP version (and implemented here) may cause 
        drifting the pseudoadiabat to occasionally "drift" when high-resolution 
        radiosonde data is used.  While a stricter convergence criteria (e.g. 0.01) has shown
        to resolve this problem, it creates a noticable departure from the SPC CAPE values and therefore
        may decalibrate the other SHARPpy functions (e.g. SARS).

    Parameters
    ----------
    p : number
        Pressure to which parcel is raised (hPa)
    thetam : number
        Saturated Potential Temperature of parcel (C)
    conv : number
        Convergence criteria for satlift() (C)

    Returns
    -------
    Temperature (C) of saturated parcel at new level

    '''
    try:
        # If p and thetam are scalars
        if numpy.fabs(p - 1000.) - 0.001 <= 0:
            return thetam
        eor = 999
        while numpy.fabs(eor) - conv > 0:
            if eor == 999:                  # First Pass
                pwrp = numpy.power((p / 1000.),KAPPA)
                t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK
                e1 = wobf(t1) - wobf(thetam)
                rate = 1
            else:                           # Successive Passes
                rate = (t2 - t1) / (e2 - e1)
                t1 = t2
                e1 = e2
            t2 = t1 - (e1 * rate)
            e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK
            e2 += wobf(t2) - wobf(e2) - thetam
            eor = e2 * rate
        return t2 - eor
    except ValueError:
        # If p and thetam are arrays
        short = numpy.fabs(p - 1000.) - 0.001 <= 0
        lft = numpy.where(short, thetam, 0)
        if numpy.all(short):
            return lft

        eor = 999
        first_pass = True
        while numpy.fabs(numpy.min(eor)) - conv > 0:
            if first_pass:                  # First Pass
                pwrp = numpy.power((p[~short] / 1000.),KAPPA)
                t1 = (thetam[~short] + ZEROCNK) * pwrp - ZEROCNK
                e1 = wobf(t1) - wobf(thetam[~short])
                rate = 1
                first_pass = False
            else:                           # Successive Passes
                rate = (t2 - t1) / (e2 - e1)
                t1 = t2
                e1 = e2
            t2 = t1 - (e1 * rate)
            e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK
            e2 += wobf(t2) - wobf(e2) - thetam[~short]
            eor = e2 * rate
        lft[~short] = t2 - eor
        return lft


def wetlift(p, t, p2):
    '''
    Lifts a parcel moist adiabatically to its new level.

    Parameters
    -----------
    p : number
        Pressure of initial parcel (hPa)
    t : number
        Temperature of initial parcel (C)
    p2 : number
        Pressure of final level (hPa)

    Returns
    -------
    Temperature (C)

    '''
    #if p == p2:
    #    return t

    thta = temperature_conv.temperatures_to_potential_temperatures(
        temperatures_kelvins=temperature_conv.celsius_to_kelvins(t),
        total_pressures_pascals=HPA_TO_PASCALS * p
    )
    thta = temperature_conv.kelvins_to_celsius(thta)

    if thta is numpy.ma.masked or p2 is numpy.ma.masked:
        return numpy.ma.masked

    thetam = thta - wobf(thta) + wobf(t)
    return satlift(p2, thetam)
