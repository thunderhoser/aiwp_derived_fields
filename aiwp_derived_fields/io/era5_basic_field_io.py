"""Input/output methods for basic model fields.

This module is the same as basic_field_io.py, except for the ERA5 model instead
of an AIWP model.
"""

import os
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import error_checking
from aiwp_derived_fields.utils import model_utils

TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'
INIT_TIME_INTERVAL_SEC = 6 * 3600


def _subset_data_1init_time(forecast_table_xarray, init_time_unix_sec):
    """Subsets data to one model run (initialization time).

    :param forecast_table_xarray: xarray table.
    :param init_time_unix_sec: Desired init time.
    :return: forecast_table_xarray: Same as input but with only the desired init
        time.
    """

    all_init_time_strings = [
        str(t) for t in forecast_table_xarray.coords['time'].values
    ]
    all_init_time_strings = [s.split('.')[0] for s in all_init_time_strings]
    all_init_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in all_init_time_strings
    ], dtype=int)

    time_indices = numpy.where(all_init_times_unix_sec == init_time_unix_sec)[0]
    if len(time_indices) == 0:
        error_string = (
            'Cannot find desired init time ({0:s}) table.  Found the following '
            'init times in table:\n{1:s}'
        ).format(
            time_conversion.unix_sec_to_string(
                init_time_unix_sec, '%Y-%m-%d-%H'
            ),
            str(all_init_time_strings)
        )

        raise ValueError(error_string)

    forecast_table_xarray = forecast_table_xarray.isel({'time': time_indices})
    forecast_table_xarray = forecast_table_xarray.rename_dims({
        'time': model_utils.VALID_TIME_DIM
    })
    forecast_table_xarray = forecast_table_xarray.drop_vars('time')
    forecast_table_xarray = forecast_table_xarray.assign_coords({
        model_utils.VALID_TIME_DIM: numpy.array([init_time_unix_sec], dtype=int)
    })

    return forecast_table_xarray


def find_files_1run(directory_name, init_time_unix_sec,
                    raise_error_if_missing=True):
    """Finds the two NetCDF files with basic fields for one model run.

    For every init time (actually every init day), there are two NetCDF files:
    one with surface-level variables (2-D) and one with isobaric variables
    (3-D).

    :param directory_name: Directory path (string).
    :param init_time_unix_sec: Model-initialization time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: netcdf_3d_file_name: Path to file with 3-D variables.
    :return: netcdf_2d_file_name: Path to file with 2-D variables.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    # Check input args.
    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    error_checking.assert_is_integer(init_time_unix_sec)
    init_time_unix_sec = int(number_rounding.round_to_nearest(
        init_time_unix_sec, INIT_TIME_INTERVAL_SEC
    ))

    # Determine where the file should be.
    init_date_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, '%Y%m%d'
    )
    netcdf_3d_file_name = '{0:s}/iso_vars_{1:s}.nc'.format(
        directory_name, init_date_string
    )
    netcdf_2d_file_name = '{0:s}/sl_vars_{1:s}.nc'.format(
        directory_name, init_date_string
    )

    # Determine whether the file is actually there.
    both_files_exist = (
        os.path.isfile(netcdf_3d_file_name) and
        os.path.isfile(netcdf_2d_file_name)
    )
    if both_files_exist or not raise_error_if_missing:
        return netcdf_3d_file_name, netcdf_2d_file_name

    if not os.path.isfile(netcdf_3d_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            netcdf_3d_file_name
        )
        raise ValueError(error_string)

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        netcdf_2d_file_name
    )
    raise ValueError(error_string)


def read_files_1run(netcdf_3d_file_name, netcdf_2d_file_name,
                    init_time_unix_sec):
    """Reads basic fields from NetCDF for one model run.

    :param netcdf_3d_file_name: Path to file with 3-D variables.
    :param netcdf_2d_file_name: Path to file with 2-D variables.
    :param init_time_unix_sec: Model-initialization time.
    :return: forecast_table_xarray: xarray table.  The dimensions should be
        time, latitude, longitude, and vertical (pressure) level; these are
        listed at the top of `model_utils.py`.  The available weather variables
        are also listed at the top of `model_utils.py`.
    """

    # Check input args.
    error_checking.assert_file_exists(netcdf_3d_file_name)
    error_checking.assert_file_exists(netcdf_2d_file_name)
    error_checking.assert_is_integer(init_time_unix_sec)

    # Do actual stuff.
    forecast_table_3d_xarray = xarray.open_dataset(netcdf_3d_file_name)
    forecast_table_3d_xarray = _subset_data_1init_time(
        forecast_table_xarray=forecast_table_3d_xarray,
        init_time_unix_sec=init_time_unix_sec
    )

    forecast_table_2d_xarray = xarray.open_dataset(netcdf_2d_file_name)
    forecast_table_2d_xarray = _subset_data_1init_time(
        forecast_table_xarray=forecast_table_2d_xarray,
        init_time_unix_sec=init_time_unix_sec
    )

    ft3d = forecast_table_3d_xarray
    ft3d[model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY].values = numpy.maximum(
        ft3d[model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY].values,
        1e-10
    )
    forecast_table_3d_xarray = ft3d

    forecast_table_3d_xarray = forecast_table_3d_xarray.transpose(
        model_utils.VALID_TIME_DIM,
        model_utils.LATITUDE_DEG_NORTH_DIM,
        model_utils.LONGITUDE_DEG_EAST_DIM,
        model_utils.PRESSURE_HPA_DIM
    )
    forecast_table_2d_xarray = forecast_table_2d_xarray.rename_vars({
        't2m': model_utils.TEMPERATURE_2METRES_KELVINS_KEY
    })

    for this_var_name in [
            model_utils.TEMPERATURE_2METRES_KELVINS_KEY,
            model_utils.SEA_LEVEL_PRESSURE_PASCALS_KEY,
            model_utils.ZONAL_WIND_10METRES_M_S01_KEY,
            model_utils.MERIDIONAL_WIND_10METRES_M_S01_KEY
    ]:
        forecast_table_3d_xarray = forecast_table_3d_xarray.assign({
            this_var_name: (
                forecast_table_2d_xarray[this_var_name].dims,
                forecast_table_2d_xarray[this_var_name].values
            )
        })

    print(forecast_table_3d_xarray)
    return forecast_table_3d_xarray
