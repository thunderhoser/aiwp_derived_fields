"""Input/output methods for time-constant fields in ERA5.

Well, probably just one field.  Currently I'm using only surface geopotential,
which is related to orographic height.
"""

import os
import numpy
import xarray
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from aiwp_derived_fields.utils import model_utils

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))

TOLERANCE = 1e-6


def read_surface_geopotential(forecast_table_xarray, netcdf_file_name=None):
    """Reads surface geopotential from NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param forecast_table_xarray: xarray table with forecast data from another
        NWP model.  This method will make sure that the grids match between
        `forecast_table_xarray` and surface geopotentials from ERA5.
    :param netcdf_file_name: Path to NetCDF file.
    :return: surface_geopotential_matrix_m2_s02: M-by-N numpy array of
        surface geopotentials (units of m^2 s^-2).
    """

    if netcdf_file_name is None:
        netcdf_file_name = '{0:s}/era5_constants_global.nc'.format(
            THIS_DIRECTORY_NAME
        )

    error_checking.assert_file_exists(netcdf_file_name)
    era5_table_xarray = xarray.open_dataset(netcdf_file_name)

    era5_latitudes_deg_n = (
        era5_table_xarray.coords['latitude_deg_n'].values[::-1]
    )
    model_latitudes_deg_n = (
        forecast_table_xarray.coords[model_utils.LATITUDE_DEG_NORTH_DIM].values
    )

    good_indices = numpy.array([
        numpy.where(numpy.absolute(era5_latitudes_deg_n - m) <= TOLERANCE)[0][0]
        for m in model_latitudes_deg_n
    ], dtype=int)
    assert numpy.allclose(
        era5_latitudes_deg_n[good_indices], model_latitudes_deg_n,
        atol=TOLERANCE
    )
    era5_table_xarray = era5_table_xarray.isel({'latitude_deg_n': good_indices})

    era5_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        era5_table_xarray.coords['longitude_deg_e'].values
    )
    model_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        forecast_table_xarray.coords[model_utils.LONGITUDE_DEG_EAST_DIM].values
    )

    good_indices = numpy.array([
        numpy.where(numpy.absolute(era5_longitudes_deg_e - m) <= TOLERANCE)[0][0]
        for m in model_longitudes_deg_e
    ], dtype=int)
    assert numpy.allclose(
        era5_longitudes_deg_e[good_indices], model_longitudes_deg_e,
        atol=TOLERANCE
    )
    era5_table_xarray = era5_table_xarray.isel({'longitude_deg_e': good_indices})

    k = numpy.where(
        era5_table_xarray.coords['field'].values == 'geopotential_m2_s02'
    )[0][0]

    surface_geopotential_matrix_m2_s02 = numpy.flip(
        era5_table_xarray['data'].values[..., k], axis=0
    )
    print(surface_geopotential_matrix_m2_s02.shape)

    return surface_geopotential_matrix_m2_s02
