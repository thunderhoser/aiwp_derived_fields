"""Computes derived fields."""

import os
import copy
import argparse
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from aiwp_derived_fields.io import basic_field_io
from aiwp_derived_fields.io import era5_constants_io
from aiwp_derived_fields.utils import model_utils
from aiwp_derived_fields.utils import derived_field_utils

# TODO(thunderhoser): Need derived_field_io.py.

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_basic_field_dir_name'
MODEL_ARG_NAME = 'model_name'
INIT_TIME_ARG_NAME = 'init_time_string'
DO_MULTIPROCESSING_ARG_NAME = 'do_multiprocessing'
DERIVED_FIELDS_ARG_NAME = 'derived_field_names'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  The input file (containing basic model fields) '
    'will be found therein by `basic_field_io.find_file` and read by  '
    '`basic_field_io.read_file`.'
)
MODEL_HELP_STRING = (
    'Name of AIWP model.  Must be accepted by `model_utils.check_model_name`.'
)
INIT_TIME_HELP_STRING = (
    'Model-initialization time.  Must be in format "yyyy-mm-dd-HH".  Derived '
    'fields will be computed for a single model run (one model @ one init '
    'time) for all forecast hours.'
)
DO_MULTIPROCESSING_HELP_STRING = (
    'Boolean flag.  If 1 (0), will do parallel (sequential) processing.  I '
    'highly recommend parallel.'
)
DERIVED_FIELDS_HELP_STRING = (
    'List with names of derived fields.  Each name must be accepted by '
    '`derived_field_utils.parse_field_name`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Derived fields will be written to a NetCDF '
    'file here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_ARG_NAME, type=str, required=True, help=MODEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DO_MULTIPROCESSING_ARG_NAME, type=int, required=False, default=1,
    help=DO_MULTIPROCESSING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DERIVED_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=DERIVED_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, model_name, init_time_string, do_multiprocessing,
         derived_field_names, output_dir_name):
    """Computes derived fields.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of script.
    :param model_name: Same.
    :param init_time_string: Same.
    :param do_multiprocessing: Same.
    :param derived_field_names: Same.
    :param output_dir_name: Same.
    """

    # Check/process input args.
    metadata_dict_by_field = [
        derived_field_utils.parse_field_name(
            derived_field_name=f, is_field_to_compute=True
        )
        for f in derived_field_names
    ]
    derived_field_names = [
        derived_field_utils.create_field_name(m)
        for m in metadata_dict_by_field
    ]

    _, unique_indices = numpy.unique(
        numpy.array(derived_field_names), return_index=True
    )
    derived_field_names = [derived_field_names[k] for k in unique_indices]
    metadata_dict_by_field = [metadata_dict_by_field[k] for k in unique_indices]
    num_fields = len(derived_field_names)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read inputs (basic weather fields).
    input_file_name = basic_field_io.find_file(
        directory_name=input_dir_name,
        model_name=model_name,
        init_time_unix_sec=time_conversion.string_to_unix_sec(
            init_time_string, TIME_FORMAT
        ),
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    forecast_table_xarray = basic_field_io.read_file(input_file_name)

    # Read surface-geopotential field.  This might be needed to compute some
    # derived fields.
    surface_geopotential_matrix_m2_s02 = (
        era5_constants_io.read_surface_geopotential(forecast_table_xarray)
    )
    surface_pressure_matrix_pascals = None
    surface_dewpoint_matrix_kelvins = None

    # Compute the derived fields.
    new_derived_field_names = []
    new_derived_field_matrices = []

    for j in range(num_fields):
        if derived_field_names[j] in new_derived_field_names:
            continue

        this_meta_dict = metadata_dict_by_field[j]
        this_basic_field_name = (
            this_meta_dict[derived_field_utils.BASIC_FIELD_KEY]
        )

        if this_basic_field_name in derived_field_utils.CAPE_CIN_NAMES:
            (
                this_cape_matrix_j_kg01,
                this_cin_matrix_j_kg01,
                surface_pressure_matrix_pascals,
                surface_dewpoint_matrix_kelvins
            ) = derived_field_utils.get_cape_and_cin(
                forecast_table_xarray=forecast_table_xarray,
                do_multiprocessing=do_multiprocessing,
                parcel_source_string=
                this_meta_dict[derived_field_utils.PARCEL_SOURCE_KEY],
                mixed_layer_depth_metres=
                this_meta_dict[derived_field_utils.MIXED_LAYER_DEPTH_KEY],
                surface_geopotential_matrix_m2_s02=
                surface_geopotential_matrix_m2_s02,
                surface_pressure_matrix_pascals=surface_pressure_matrix_pascals,
                surface_dewpoint_matrix_kelvins=surface_dewpoint_matrix_kelvins
            )

            this_ps = this_meta_dict[derived_field_utils.PARCEL_SOURCE_KEY]
            if this_ps == derived_field_utils.MIXED_LAYER_PARCEL_SOURCE_STRING:
                this_meta_dict[derived_field_utils.BASIC_FIELD_KEY] = (
                    derived_field_utils.MIXED_LAYER_CAPE_NAME
                )
            elif this_ps == derived_field_utils.MOST_UNSTABLE_PARCEL_SOURCE_STRING:
                this_meta_dict[derived_field_utils.BASIC_FIELD_KEY] = (
                    derived_field_utils.MOST_UNSTABLE_CAPE_NAME
                )
            else:
                this_meta_dict[derived_field_utils.BASIC_FIELD_KEY] = (
                    derived_field_utils.SURFACE_BASED_CAPE_NAME
                )

            new_field_name = derived_field_utils.create_field_name(
                this_meta_dict
            )
            new_derived_field_names.append(new_field_name)
            new_derived_field_matrices.append(this_cape_matrix_j_kg01)

            if this_ps == derived_field_utils.MIXED_LAYER_PARCEL_SOURCE_STRING:
                this_meta_dict[derived_field_utils.BASIC_FIELD_KEY] = (
                    derived_field_utils.MIXED_LAYER_CIN_NAME
                )
            elif this_ps == derived_field_utils.MOST_UNSTABLE_PARCEL_SOURCE_STRING:
                this_meta_dict[derived_field_utils.BASIC_FIELD_KEY] = (
                    derived_field_utils.MOST_UNSTABLE_CIN_NAME
                )
            else:
                this_meta_dict[derived_field_utils.BASIC_FIELD_KEY] = (
                    derived_field_utils.SURFACE_BASED_CIN_NAME
                )

            new_field_name = derived_field_utils.create_field_name(
                this_meta_dict
            )
            new_derived_field_names.append(new_field_name)
            new_derived_field_matrices.append(this_cin_matrix_j_kg01)

        elif this_basic_field_name == derived_field_utils.LIFTED_INDEX_NAME:
            (
                this_li_matrix_kelvins,
                surface_pressure_matrix_pascals
            ) = derived_field_utils.get_lifted_index(
                forecast_table_xarray=forecast_table_xarray,
                do_multiprocessing=do_multiprocessing,
                final_pressure_pascals=
                this_meta_dict[derived_field_utils.TOP_PRESSURE_KEY],
                surface_geopotential_matrix_m2_s02=
                surface_geopotential_matrix_m2_s02,
                surface_pressure_matrix_pascals=surface_pressure_matrix_pascals
            )

            new_derived_field_names.append(derived_field_names[j])
            new_derived_field_matrices.append(this_li_matrix_kelvins)

        elif this_basic_field_name == derived_field_utils.PRECIPITABLE_WATER_NAME:
            (
                this_pw_matrix_kg_m02,
                surface_pressure_matrix_pascals,
                surface_dewpoint_matrix_kelvins
            ) = derived_field_utils.get_precipitable_water(
                forecast_table_xarray=forecast_table_xarray,
                do_multiprocessing=do_multiprocessing,
                top_pressure_pascals=
                this_meta_dict[derived_field_utils.TOP_PRESSURE_KEY],
                surface_geopotential_matrix_m2_s02=
                surface_geopotential_matrix_m2_s02,
                surface_pressure_matrix_pascals=surface_pressure_matrix_pascals,
                surface_dewpoint_matrix_kelvins=surface_dewpoint_matrix_kelvins
            )

            new_derived_field_names.append(derived_field_names[j])
            new_derived_field_matrices.append(this_pw_matrix_kg_m02)

        elif this_basic_field_name == derived_field_utils.SCALAR_WIND_SHEAR_NAME:
            (
                this_zonal_shear_matrix_m_s01,
                this_merid_shear_matrix_m_s01,
                surface_pressure_matrix_pascals
            ) = derived_field_utils.get_wind_shear(
                forecast_table_xarray=forecast_table_xarray,
                do_multiprocessing=do_multiprocessing,
                bottom_pressure_pascals=
                this_meta_dict[derived_field_utils.BOTTOM_PRESSURE_KEY],
                top_pressure_pascals=
                this_meta_dict[derived_field_utils.TOP_PRESSURE_KEY],
                surface_geopotential_matrix_m2_s02=
                surface_geopotential_matrix_m2_s02,
                surface_pressure_matrix_pascals=surface_pressure_matrix_pascals
            )

            this_meta_dict[derived_field_utils.BASIC_FIELD_KEY] = (
                derived_field_utils.ZONAL_WIND_SHEAR_NAME
            )
            new_field_name = derived_field_utils.create_field_name(
                this_meta_dict
            )
            new_derived_field_names.append(new_field_name)
            new_derived_field_matrices.append(this_zonal_shear_matrix_m_s01)

            this_meta_dict[derived_field_utils.BASIC_FIELD_KEY] = (
                derived_field_utils.MERIDIONAL_WIND_SHEAR_NAME
            )
            new_field_name = derived_field_utils.create_field_name(
                this_meta_dict
            )
            new_derived_field_names.append(new_field_name)
            new_derived_field_matrices.append(this_merid_shear_matrix_m_s01)

    derived_field_names = copy.deepcopy(new_derived_field_names)
    derived_field_matrix = numpy.stack(new_derived_field_matrices, axis=-1)

    coord_dict = {
        'latitude_deg_n': forecast_table_xarray.coords[
            model_utils.LATITUDE_DEG_NORTH_DIM
        ].values,
        'longitude_deg_e': forecast_table_xarray.coords[
            model_utils.LONGITUDE_DEG_EAST_DIM
        ].values,
        'field_name': derived_field_names
    }

    these_dim = ('latitude_deg_n', 'longitude_deg_e', 'field_name')
    main_data_dict = {
        'data': (these_dim, derived_field_matrix)
    }

    derived_field_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )
    output_file_name = '{0:s}/{1:s}'.format(
        output_dir_name,
        os.path.split(input_file_name)[1]
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    derived_field_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF3_64BIT'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        model_name=getattr(INPUT_ARG_OBJECT, MODEL_ARG_NAME),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        do_multiprocessing=bool(
            getattr(INPUT_ARG_OBJECT, DO_MULTIPROCESSING_ARG_NAME)
        ),
        derived_field_names=getattr(INPUT_ARG_OBJECT, DERIVED_FIELDS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
