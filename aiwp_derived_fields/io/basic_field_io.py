"""Input/output methods for basic model fields.

'Basic model fields' = the raw AIWP output, not including derived fields.
"""

import os
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.gg_utils import moisture_conversions as moisture_conv
from aiwp_derived_fields.utils import model_utils
from aiwp_derived_fields.utils import derived_field_utils

TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'
INIT_TIME_INTERVAL_SEC = 6 * 3600


def find_file(directory_name, model_name, init_time_unix_sec,
              raise_error_if_missing=True):
    """Finds NetCDF file with basic model fields.

    This will be one file, with all fields at all forecast times, for one model
    at one init time.

    :param directory_name: Directory path (string).
    :param model_name: Name of model (string).  Must be accepted by
        `model_utils.check_model_name`.
    :param init_time_unix_sec: Model-initialization time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: netcdf_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    # TODO(thunderhoser): When the models are driven by ERA5, the file format
    # is grib.  I don't know if I want to fuck with this.

    # TODO(thunderhoser): Are different models available at different init
    # times?  Probably not, since the driving model (GFS) is always the same.

    # Check input args.
    error_checking.assert_is_string(directory_name)
    model_utils.check_model_name(model_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    error_checking.assert_is_integer(init_time_unix_sec)
    init_time_unix_sec = int(number_rounding.round_to_nearest(
        init_time_unix_sec, INIT_TIME_INTERVAL_SEC
    ))

    # Determine where the file should be.
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, '%Y%m%d%H'
    )
    init_year_string = init_time_string[:4]
    init_month_day_string = init_time_string[4:8]

    netcdf_file_name = (
        '{0:s}/{1:s}/{2:s}/{3:s}_GFS_{4:s}_f000_f240_06.nc'
    ).format(
        directory_name,
        init_year_string,
        init_month_day_string,
        model_name,
        init_time_string
    )

    # Determine whether the file is actually there.
    if os.path.isfile(netcdf_file_name) or not raise_error_if_missing:
        return netcdf_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        netcdf_file_name
    )
    raise ValueError(error_string)


def read_file(netcdf_file_name):
    """Reads basic model fields from NetCDF file.

    This will be one file, with all fields at all forecast times, for one model
    at one init time.

    :param netcdf_file_name: Path to input file.
    :return: forecast_table_xarray: xarray table.  The dimensions should be
        time, latitude, longitude, and vertical (pressure) level; these are
        listed at the top of `model_utils.py`.  The potentially available
        weather variables are also listed at the top of `model_utils.py`.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    forecast_table_xarray = xarray.open_dataset(netcdf_file_name)

    valid_time_strings = [
        str(t) for t in forecast_table_xarray.coords['time'].values
    ]
    valid_time_strings = [s.split('.')[0] for s in valid_time_strings]

    valid_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in valid_time_strings
    ], dtype=int)

    ftx = forecast_table_xarray
    if model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY not in ftx.data_vars:
        relative_humidity_matrix = 0.01 * ftx['r'].values
        relative_humidity_matrix = numpy.maximum(relative_humidity_matrix, 0.)
        relative_humidity_matrix = numpy.minimum(relative_humidity_matrix, 1.)

        pressure_matrix_pascals = (
            derived_field_utils._get_model_pressure_matrix(
                forecast_table_xarray=ftx,
                vertical_axis_first=True
            )
        )

        num_times = len(ftx.coords['time'].values)
        pressure_matrix_pascals = numpy.expand_dims(
            pressure_matrix_pascals, axis=0
        )
        pressure_matrix_pascals = numpy.repeat(
            pressure_matrix_pascals, axis=0, repeats=num_times
        )

        dewpoint_matrix_kelvins = moisture_conv.relative_humidity_to_dewpoint(
            relative_humidities=relative_humidity_matrix,
            temperatures_kelvins=
            ftx[model_utils.TEMPERATURE_KELVINS_KEY].values,
            total_pressures_pascals=pressure_matrix_pascals
        )

        spec_humidity_matrix_kg_kg01 = (
            moisture_conv.dewpoint_to_specific_humidity(
                dewpoints_kelvins=dewpoint_matrix_kelvins,
                temperatures_kelvins=
                ftx[model_utils.TEMPERATURE_KELVINS_KEY].values,
                total_pressures_pascals=pressure_matrix_pascals
            )
        )

        these_dims = (
            'time',
            model_utils.PRESSURE_HPA_DIM,
            model_utils.LATITUDE_DEG_NORTH_DIM,
            model_utils.LONGITUDE_DEG_EAST_DIM
        )
        forecast_table_xarray = forecast_table_xarray.assign({
            model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY: (
                these_dims, spec_humidity_matrix_kg_kg01
            )
        })

    forecast_table_xarray[
        model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY
    ].values = numpy.maximum(
        forecast_table_xarray[model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY].values,
        1e-10
    )

    forecast_table_xarray = forecast_table_xarray.rename_dims({
        'time': model_utils.VALID_TIME_DIM
    })
    forecast_table_xarray = forecast_table_xarray.assign_coords({
        'time': valid_times_unix_sec
    })
    return forecast_table_xarray.assign_coords({
        model_utils.VALID_TIME_DIM: valid_times_unix_sec
    })
