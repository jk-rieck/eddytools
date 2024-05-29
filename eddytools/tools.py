""" tools

Collection of tools to be used for in- and output of the
detection and tracking

"""

import numpy as np
from cftime import date2num as d2n
from cftime import num2date as n2d
import h5py

def dict2hdf5(eddies, outfile, time_units, calendar, has_year_zero, t_dim=False):
    eddies2hdf5 = h5py.File(outfile, 'w')
    if t_dim:
        for t in np.arange(0, len(eddies)):
            tt = str(eddies[t][0]['time'])[0:10]
            eddit = eddies2hdf5.create_group(tt)
            for e in np.arange(0, len(eddies[t])):
                ee = str(e)
                eddi = eddit.create_group(ee)
                if isinstance(eddies[t][e]['time'], np.datetime64):
                    e_time = int(eddies[t][e]['time'])
                else:
                    e_time = d2n(eddies[t][e]['time'], time_units,
                                 calendar=calendar, has_year_zero=has_year_zero)
                eddi["time"] = e_time
                eddi["time"].attrs["calendar"] = calendar
                eddi["time"].attrs["has_year_zero"] = has_year_zero
                eddi["time"].attrs["units"] = time_units
                eddi['lon'] = eddies[t][e]['lon']
                eddi['lat'] = eddies[t][e]['lat']
                eddi['amp'] = eddies[t][e]['amp']
                eddi['eddy_j'] = eddies[t][e]['eddy_j']
                eddi['eddy_i'] = eddies[t][e]['eddy_i']
                eddi['area'] = eddies[t][e]['area']
                eddi['scale'] = eddies[t][e]['scale']
                eddi['type'] = eddies[t][e]['type']
    else:
        tt = str(eddies[0]['time'])[0:10]
        eddit = eddies2hdf5.create_group(tt)
        for e in np.arange(0, len(eddies)):
            ee = str(e)
            eddi = eddit.create_group(ee)
            if isinstance(eddies[e]['time'], np.datetime64):
                e_time = int(eddies[e]['time'])
            else:
                e_time = d2n(eddies[e]['time'], time_units,
                             calendar=calendar, has_year_zero=has_year_zero)
            eddi["time"] = e_time
            eddi["time"].attrs["calendar"] = calendar
            eddi["time"].attrs["has_year_zero"] = has_year_zero
            eddi["time"].attrs["units"] = time_units
            eddi['lon'] = eddies[e]['lon']
            eddi['lat'] = eddies[e]['lat']
            eddi['amp'] = eddies[e]['amp']
            eddi['eddy_j'] = eddies[e]['eddy_j']
            eddi['eddy_i'] = eddies[e]['eddy_i']
            eddi['area'] = eddies[e]['area']
            eddi['scale'] = eddies[e]['scale']
            eddi['type'] = eddies[e]['type']
    return eddies2hdf5


def hdf5time(time):
    if isinstance(time, np.datetime64):
        timeout = np.datetime64(int(time), 'ns')
    else:
        timeout = n2d(time[()], time.attrs["units"],
                      calendar=time.attrs["calendar"], has_year_zero=time.attrs["has_year_zero"])
    return timeout
