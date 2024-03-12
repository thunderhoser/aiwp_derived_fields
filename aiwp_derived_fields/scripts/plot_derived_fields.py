"""Plots derived fields."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from aiwp_derived_fields.io import border_io
from aiwp_derived_fields.io import derived_field_io
from aiwp_derived_fields.utils import model_utils
from aiwp_derived_fields.plotting import plotting_utils
from aiwp_derived_fields.plotting import derived_field_plotting

TOLERANCE = 1e-6

TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600

SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
DIVERGING_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
NAN_COLOUR = numpy.full(3, 152. / 255)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_dir_name'
MODEL_ARG_NAME = 'model_name'
INIT_TIME_ARG_NAME = 'init_time_string'
DERIVED_FIELDS_ARG_NAME = 'derived_field_names'
FORECAST_HOURS_ARG_NAME = 'forecast_hours'
MIN_VALUES_ARG_NAME = 'min_colour_values'
MAX_VALUES_ARG_NAME = 'max_colour_values'
MIN_PERCENTILES_ARG_NAME = 'min_colour_percentiles'
MAX_PERCENTILES_ARG_NAME = 'max_colour_percentiles'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  Relevant files therein will be found by '
    '`derived_field_io.find_file` and read by `derived_field_io.read_file`.'
)
MODEL_HELP_STRING = (
    'Name of AIWP model.  Must be accepted by `model_utils.check_model_name`.'
)
INIT_TIME_HELP_STRING = (
    'Initialization time of model run (format "yyyy-mm-dd-HH").  Derived '
    'fields will be plotted for one or more forecast hours from this model run.'
)
DERIVED_FIELDS_HELP_STRING = (
    'List of derived fields to plot.  Each name must be accepted by '
    '`derived_field_utils.parse_field_name`.'
)
FORECAST_HOURS_HELP_STRING = (
    'List of forecast hours.  Each field in the list `{0:s}` will be plotted '
    'at each forecast hour in this list.'
).format(DERIVED_FIELDS_ARG_NAME)

MIN_VALUES_HELP_STRING = (
    'List of minimum values for each colour scheme (one per derived field in '
    'the list `{0:s}`).  If you would prefer to set colour limits by '
    'percentile instead, leave this argument alone.'
).format(DERIVED_FIELDS_ARG_NAME)

MAX_VALUES_HELP_STRING = (
    'List of max values for each colour scheme (one per derived field in the '
    'list `{0:s}`).  If you would prefer to set colour limits by percentile '
    'instead, leave this argument alone.'
).format(DERIVED_FIELDS_ARG_NAME)

MIN_PERCENTILES_HELP_STRING = (
    'List of minimum percentiles for each colour scheme (one per derived field '
    'in the list `{0:s}`).  For example, suppose that the second value in the '
    'list `{0:s}` is "most_unstable_cape_j_kg01" and the second value in this '
    'list is 1 -- then, at each forecast hour, the minimum value in the colour '
    'scheme for MUCAPE will be the 1st percentile over values in the spatial '
    'grid.  If you would prefer to set colour limits by raw value instead, '
    'leave this argument alone.'
).format(DERIVED_FIELDS_ARG_NAME)

MAX_PERCENTILES_HELP_STRING = (
    'List of max percentiles for each colour scheme (one per derived field in '
    'the list `{0:s}`).  For example, suppose that the second value in the '
    'list `{0:s}` is "most_unstable_cape_j_kg01" and the second value in this '
    'list is 99 -- then, at each forecast hour, the max value in the colour '
    'scheme for MUCAPE will be the 99th percentile over values in the spatial '
    'grid.  If you would prefer to set colour limits by raw value instead, '
    'leave this argument alone.'
).format(DERIVED_FIELDS_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
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
    '--' + DERIVED_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=DERIVED_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FORECAST_HOURS_ARG_NAME, type=int, nargs='+', required=True,
    help=FORECAST_HOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_VALUES_ARG_NAME, type=float, nargs='+', required=False,
    default=[1.], help=MIN_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=MAX_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PERCENTILES_ARG_NAME, type=float, nargs='+', required=False,
    default=[1.], help=MIN_PERCENTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=MAX_PERCENTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_field(
        data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        derived_field_name, min_colour_value, max_colour_value,
        title_string, output_file_name):
    """Plots one field.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border file

    :param data_matrix: M-by-N numpy array of data values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param derived_field_name: Name of field to plot.
    :param min_colour_value: Min value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param title_string: Title.
    :param output_file_name: Path to output file.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if derived_field_plotting.use_diverging_colour_scheme(derived_field_name):
        colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
    else:
        colour_map_object = SEQUENTIAL_COLOUR_MAP_OBJECT

    colour_map_object.set_bad(NAN_COLOUR)
    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )

    is_longitude_positive_in_west = derived_field_plotting.plot_field(
        data_matrix=data_matrix,
        grid_latitudes_deg_n=grid_latitudes_deg_n,
        grid_longitudes_deg_e=grid_longitudes_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=True
    )

    if is_longitude_positive_in_west:
        border_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
            grid_longitudes_deg_e
        )
    else:
        border_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            border_longitudes_deg_e
        )
        grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            grid_longitudes_deg_e
        )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object,
        line_colour=numpy.full(3, 0.)
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=grid_latitudes_deg_n,
        plot_longitudes_deg_e=grid_longitudes_deg_e,
        axes_object=axes_object,
        meridian_spacing_deg=20.,
        parallel_spacing_deg=10.
    )

    axes_object.set_xlim(
        numpy.min(grid_longitudes_deg_e),
        numpy.max(grid_longitudes_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(grid_latitudes_deg_n),
        numpy.max(grid_latitudes_deg_n)
    )
    axes_object.set_title(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(input_dir_name, model_name, init_time_string, derived_field_names,
         forecast_hours, min_colour_values, max_colour_values,
         min_colour_percentiles, max_colour_percentiles, output_dir_name):
    """Plots derived fields.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param model_name: Same.
    :param init_time_string: Same.
    :param derived_field_names: Same.
    :param forecast_hours: Same.
    :param min_colour_values: Same.
    :param max_colour_values: Same.
    :param min_colour_percentiles: Same.
    :param max_colour_percentiles: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(forecast_hours, 0)
    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )

    if (
            len(min_colour_values) == 1 and
            len(max_colour_values) == 1 and
            max_colour_values[0] < min_colour_values[0]
    ):
        min_colour_values = None
        max_colour_values = None

    if (
            len(min_colour_percentiles) == 1 and
            len(max_colour_percentiles) == 1 and
            max_colour_percentiles[0] < min_colour_percentiles[0]
    ):
        min_colour_percentiles = None
        max_colour_percentiles = None

    use_raw_values_for_colour_limits = not (
        min_colour_values is None or max_colour_values is None
    )
    use_percentiles_for_colour_limits = not (
        min_colour_percentiles is None or max_colour_percentiles is None
    )
    assert use_raw_values_for_colour_limits or use_percentiles_for_colour_limits

    num_fields = len(derived_field_names)

    if use_raw_values_for_colour_limits:
        error_checking.assert_is_numpy_array(
            max_colour_values,
            exact_dimensions=numpy.array([num_fields], dtype=int)
        )
        error_checking.assert_is_greater_numpy_array(
            max_colour_values - min_colour_values, 0.
        )
    else:
        error_checking.assert_is_numpy_array(
            max_colour_percentiles,
            exact_dimensions=numpy.array([num_fields], dtype=int)
        )
        error_checking.assert_is_greater_numpy_array(
            max_colour_percentiles - min_colour_percentiles, 0.
        )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    input_file_name = derived_field_io.find_file(
        directory_name=input_dir_name,
        model_name=model_name,
        init_time_unix_sec=init_time_unix_sec,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    derived_field_table_xarray = derived_field_io.read_file(input_file_name)
    dftx = derived_field_table_xarray

    for this_forecast_hour in forecast_hours:
        valid_time_unix_sec = (
            init_time_unix_sec + this_forecast_hour * HOURS_TO_SECONDS
        )
        time_index = numpy.where(
            dftx.coords[derived_field_io.VALID_TIME_DIM].values ==
            valid_time_unix_sec
        )[0][0]

        valid_time_string = time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT
        )

        for j in range(len(derived_field_names)):
            this_field_name_fancy = derived_field_plotting.field_name_to_fancy(
                derived_field_names[j]
            )

            field_index = numpy.where(
                dftx.coords[derived_field_io.FIELD_DIM].values ==
                derived_field_names[j]
            )[0][0]

            data_matrix = dftx[derived_field_io.DATA_KEY].values[
                time_index, ..., field_index
            ]

            data_matrix, unit_string = (
                derived_field_plotting.field_to_plotting_units(
                    data_matrix_default_units=data_matrix + 0.,
                    derived_field_name=derived_field_names[j]
                )
            )

            title_string = (
                '{0:s}{1:s} ({2:s})\n'
                '{3:s} init {4:s}, valid {5:s}'
            ).format(
                this_field_name_fancy[0].upper(),
                this_field_name_fancy[1:],
                unit_string,
                model_utils.model_name_to_fancy(model_name),
                init_time_string,
                valid_time_string
            )

            output_file_name = '{0:s}/{1:s}_{2:s}_{3:s}_{4:s}.jpg'.format(
                output_dir_name,
                model_name,
                init_time_string,
                valid_time_string,
                derived_field_names[j].replace('_', '-')
            )

            if use_raw_values_for_colour_limits:
                this_min = min_colour_values[j] + 0.
                this_max = max_colour_values[j] + 0.
            else:
                if derived_field_plotting.use_diverging_colour_scheme(
                        derived_field_names[j]
                ):
                    this_max = numpy.nanpercentile(
                        numpy.absolute(data_matrix), max_colour_percentiles[j]
                    )
                    if numpy.isnan(this_max):
                        this_max = TOLERANCE

                    this_max = max([this_max, TOLERANCE])
                    this_min = -1 * this_max
                else:
                    this_min = numpy.nanpercentile(
                        data_matrix, min_colour_percentiles[j]
                    )
                    this_max = numpy.nanpercentile(
                        data_matrix, max_colour_percentiles[j]
                    )
                    if numpy.isnan(this_min):
                        this_min = 0.
                        this_max = TOLERANCE

                    this_max = max([this_max, this_min + TOLERANCE])

            _plot_one_field(
                data_matrix=data_matrix,
                grid_latitudes_deg_n=
                dftx.coords[derived_field_io.LATITUDE_DIM].values,
                grid_longitudes_deg_e=
                dftx.coords[derived_field_io.LONGITUDE_DIM].values,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                derived_field_name=derived_field_names[j],
                min_colour_value=this_min,
                max_colour_value=this_max,
                title_string=title_string,
                output_file_name=output_file_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        model_name=getattr(INPUT_ARG_OBJECT, MODEL_ARG_NAME),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        derived_field_names=getattr(INPUT_ARG_OBJECT, DERIVED_FIELDS_ARG_NAME),
        forecast_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, FORECAST_HOURS_ARG_NAME), dtype=int
        ),
        min_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_VALUES_ARG_NAME), dtype=float
        ),
        max_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_VALUES_ARG_NAME), dtype=float
        ),
        min_colour_percentiles=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_PERCENTILES_ARG_NAME), dtype=float
        ),
        max_colour_percentiles=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_PERCENTILES_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
