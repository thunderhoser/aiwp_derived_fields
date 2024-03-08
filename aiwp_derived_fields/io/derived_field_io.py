"""Input/output methods for derived model fields.

'Derived fields' = the ones computed with this library, not the raw AIWP output.
"""


import os
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from aiwp_derived_fields.utils import model_utils
from aiwp_derived_fields.utils import derived_field_utils

INIT_TIME_INTERVAL_SEC = 6 * 3600

VALID_TIME_DIM = 'valid_time_unix_sec'
LATITUDE_DIM = 'latitude_deg_n'
LONGITUDE_DIM = 'longitude_deg_e'
FIELD_DIM = 'field_name'

DATA_KEY = 'data'


def find_file(directory_name, model_name, init_time_unix_sec,
              raise_error_if_missing=True):
    """Finds NetCDF file with derived model fields.

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

    netcdf_file_name = '{0:s}/{1:s}/{2:s}_GFS_{3:s}_derived_fields.nc'.format(
        directory_name,
        init_year_string,
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
    """Reads derived model fields from NetCDF file.

    This will be one file, with all fields at all forecast times, for one model
    at one init time.

    :param netcdf_file_name: Path to input file.
    :return: forecast_table_xarray: xarray table.  The dimensions should be
        time, latitude, longitude, and vertical (pressure) level; these are
        listed at the top of `model_utils.py`.  The potentially available
        weather variables are also listed at the top of `model_utils.py`.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)


def write_file(
        netcdf_file_name, derived_field_matrix,
        valid_times_unix_sec, latitudes_deg_n, longitudes_deg_e,
        derived_field_names):
    """Writes derived model fields to NetCDF file.

    T = number of time steps
    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    F = number of derived fields

    :param netcdf_file_name: Path to output file.
    :param derived_field_matrix: T-by-M-by-N-by-F numpy array of data values.
    :param valid_times_unix_sec: length-T numpy array of valid times.
    :param latitudes_deg_n: length-M numpy array of grid-point latitudes (deg
        north).
    :param longitudes_deg_e: length-N numpy array of grid-point longitudes (deg
        east).
    :param derived_field_names: length-F list of field names.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(valid_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)

    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg_n, allow_nan=False
    )

    error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)
    longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg_e, allow_nan=False
    )

    error_checking.assert_is_string_list(derived_field_names)
    for f in derived_field_names:
        _ = derived_field_utils.parse_field_name(
            derived_field_name=f, is_field_to_compute=False
        )

    expected_dim = numpy.array([
        len(valid_times_unix_sec), len(latitudes_deg_n),
        len(longitudes_deg_e), len(derived_field_names)
    ], dtype=int)

    error_checking.assert_is_numpy_array(
        derived_field_matrix, exact_dimensions=expected_dim
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    # Write file.
    coord_dict = {
        VALID_TIME_DIM: valid_times_unix_sec,
        LATITUDE_DIM: latitudes_deg_n,
        LONGITUDE_DIM: longitudes_deg_e,
        FIELD_DIM: derived_field_names
    }

    these_dim = (VALID_TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM, FIELD_DIM)
    main_data_dict = {
        DATA_KEY: (these_dim, derived_field_matrix)
    }

    derived_field_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )
    derived_field_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF4'
    )
