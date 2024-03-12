"""Methods for plotting derived fields."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from aiwp_derived_fields.utils import derived_field_utils
from aiwp_derived_fields.outside_code import gg_plotting_utils

METRES_PER_SECOND_TO_KT = 3.6 / 1.852
METRES_TO_KM = 0.001
PASCALS_TO_HPA = 0.01

FIELD_TO_CONV_FACTOR = {
    derived_field_utils.MOST_UNSTABLE_CAPE_NAME: 1.,
    derived_field_utils.MOST_UNSTABLE_CIN_NAME: 1.,
    derived_field_utils.SURFACE_BASED_CAPE_NAME: 1.,
    derived_field_utils.SURFACE_BASED_CIN_NAME: 1.,
    derived_field_utils.MIXED_LAYER_CAPE_NAME: 1.,
    derived_field_utils.MIXED_LAYER_CIN_NAME: 1.,
    derived_field_utils.LIFTED_INDEX_NAME: 1.,
    derived_field_utils.PRECIPITABLE_WATER_NAME: 1.,
    derived_field_utils.SCALAR_WIND_SHEAR_NAME: METRES_PER_SECOND_TO_KT,
    derived_field_utils.ZONAL_WIND_SHEAR_NAME: METRES_PER_SECOND_TO_KT,
    derived_field_utils.MERIDIONAL_WIND_SHEAR_NAME: METRES_PER_SECOND_TO_KT,
    derived_field_utils.SCALAR_STORM_MOTION_NAME: METRES_PER_SECOND_TO_KT,
    derived_field_utils.ZONAL_STORM_MOTION_NAME: METRES_PER_SECOND_TO_KT,
    derived_field_utils.MERIDIONAL_STORM_MOTION_NAME: METRES_PER_SECOND_TO_KT,
    # derived_field_utils.HELICITY_NAME: 1.,
    derived_field_utils.POSITIVE_HELICITY_NAME: 1.,
    derived_field_utils.NEGATIVE_HELICITY_NAME: 1.,
    derived_field_utils.PBL_HEIGHT_NAME: METRES_TO_KM
}

FIELD_TO_PLOTTING_UNIT_STRING = {
    derived_field_utils.MOST_UNSTABLE_CAPE_NAME: r'J kg$^{-1}$',
    derived_field_utils.MOST_UNSTABLE_CIN_NAME: r'J kg$^{-1}$',
    derived_field_utils.SURFACE_BASED_CAPE_NAME: r'J kg$^{-1}$',
    derived_field_utils.SURFACE_BASED_CIN_NAME: r'J kg$^{-1}$',
    derived_field_utils.MIXED_LAYER_CAPE_NAME: r'J kg$^{-1}$',
    derived_field_utils.MIXED_LAYER_CIN_NAME: r'J kg$^{-1}$',
    derived_field_utils.LIFTED_INDEX_NAME: r'K or $^{\circ}$C',
    derived_field_utils.PRECIPITABLE_WATER_NAME: r'kg m$^{-2}$ or mm',
    derived_field_utils.SCALAR_WIND_SHEAR_NAME: 'kt',
    derived_field_utils.ZONAL_WIND_SHEAR_NAME: 'kt',
    derived_field_utils.MERIDIONAL_WIND_SHEAR_NAME: 'kt',
    derived_field_utils.SCALAR_STORM_MOTION_NAME: 'kt',
    derived_field_utils.ZONAL_STORM_MOTION_NAME: 'kt',
    derived_field_utils.MERIDIONAL_STORM_MOTION_NAME: 'kt',
    # derived_field_utils.HELICITY_NAME: r'm$^{2}$ s$^{-2}$ or J kg$^{-1}$',
    derived_field_utils.POSITIVE_HELICITY_NAME:
        r'm$^{2}$ s$^{-2}$ or J kg$^{-1}$',
    derived_field_utils.NEGATIVE_HELICITY_NAME:
        r'm$^{2}$ s$^{-2}$ or J kg$^{-1}$',
    derived_field_utils.PBL_HEIGHT_NAME: 'km'
}

FIELD_NAME_TO_FANCY = {
    derived_field_utils.MOST_UNSTABLE_CAPE_NAME: 'most unstable CAPE',
    derived_field_utils.MOST_UNSTABLE_CIN_NAME: 'most unstable CIN',
    derived_field_utils.SURFACE_BASED_CAPE_NAME: 'surface-based CAPE',
    derived_field_utils.SURFACE_BASED_CIN_NAME: 'surface-based CIN',
    derived_field_utils.MIXED_LAYER_CAPE_NAME: 'mixed-layer CAPE',
    derived_field_utils.MIXED_LAYER_CIN_NAME: 'mixed-layer CIN',
    derived_field_utils.LIFTED_INDEX_NAME: 'lifted index',
    derived_field_utils.PRECIPITABLE_WATER_NAME: 'precipitable water',
    derived_field_utils.SCALAR_WIND_SHEAR_NAME: 'wind-shear magnitude',
    derived_field_utils.ZONAL_WIND_SHEAR_NAME: 'zonal wind shear',
    derived_field_utils.MERIDIONAL_WIND_SHEAR_NAME: 'meridional wind shear',
    derived_field_utils.SCALAR_STORM_MOTION_NAME: 'storm-motion magnitude',
    derived_field_utils.ZONAL_STORM_MOTION_NAME: 'zonal storm motion',
    derived_field_utils.MERIDIONAL_STORM_MOTION_NAME: 'meridional storm motion',
    # derived_field_utils.HELICITY_NAME: 'storm-relative helicity',
    derived_field_utils.POSITIVE_HELICITY_NAME: 'positive SRH',
    derived_field_utils.NEGATIVE_HELICITY_NAME: 'negative SRH',
    derived_field_utils.PBL_HEIGHT_NAME: 'PBL height'
}


def field_to_plotting_units(data_matrix_default_units, derived_field_name):
    """Converts field to plotting units.

    :param data_matrix_default_units: numpy array of data values in default
        units.
    :param derived_field_name: Field name (must be accepted by
        `derived_field_utils.parse_field_name`).
    :return: data_matrix_plotting_units: Same as input but in plotting units.
    :return: plotting_units_string: String describing plotting units.
    """

    error_checking.assert_is_numpy_array(data_matrix_default_units)

    try:
        metadata_dict = derived_field_utils.parse_field_name(
            derived_field_name=derived_field_name, is_field_to_compute=False
        )
    except ValueError:
        metadata_dict = derived_field_utils.parse_field_name(
            derived_field_name=derived_field_name, is_field_to_compute=True
        )

    basic_field_name = metadata_dict[derived_field_utils.BASIC_FIELD_KEY]
    data_matrix_plotting_units = (
        FIELD_TO_CONV_FACTOR[basic_field_name] * data_matrix_default_units
    )

    return (
        data_matrix_plotting_units,
        FIELD_TO_PLOTTING_UNIT_STRING[basic_field_name]
    )


def use_diverging_colour_scheme(derived_field_name):
    """Determines whether to use diverging colour scheme.

    :param derived_field_name: Field name (must be accepted by
        `derived_field_utils.parse_field_name`).
    :return: use_diverging_scheme: Boolean flag.
    """

    try:
        metadata_dict = derived_field_utils.parse_field_name(
            derived_field_name=derived_field_name, is_field_to_compute=False
        )
    except ValueError:
        metadata_dict = derived_field_utils.parse_field_name(
            derived_field_name=derived_field_name, is_field_to_compute=True
        )

    basic_field_name = metadata_dict[derived_field_utils.BASIC_FIELD_KEY]

    return basic_field_name in [
        derived_field_utils.LIFTED_INDEX_NAME,
        derived_field_utils.ZONAL_WIND_SHEAR_NAME,
        derived_field_utils.MERIDIONAL_WIND_SHEAR_NAME,
        derived_field_utils.ZONAL_STORM_MOTION_NAME,
        derived_field_utils.MERIDIONAL_STORM_MOTION_NAME
    ]


def field_name_to_fancy(derived_field_name):
    """Converts field name to fancy version.

    :param derived_field_name: Pythonic field name (must be accepted by
        `derived_field_utils.parse_field_name`).
    :return: fancy_field_name: Fancy name (suitable for figure titles and shit).
    """

    try:
        metadata_dict = derived_field_utils.parse_field_name(
            derived_field_name=derived_field_name, is_field_to_compute=False
        )
    except ValueError:
        metadata_dict = derived_field_utils.parse_field_name(
            derived_field_name=derived_field_name, is_field_to_compute=True
        )

    basic_field_name = metadata_dict[derived_field_utils.BASIC_FIELD_KEY]
    fancy_field_name = FIELD_NAME_TO_FANCY[basic_field_name]

    if (
            basic_field_name in
            derived_field_utils.STORM_MOTION_NAMES +
            [derived_field_utils.PBL_HEIGHT_NAME]
    ):
        return fancy_field_name

    mixed_layer_names = [
        derived_field_utils.MIXED_LAYER_CAPE_NAME,
        derived_field_utils.MIXED_LAYER_CIN_NAME
    ]

    if (
            basic_field_name in derived_field_utils.CAPE_CIN_NAMES
            and basic_field_name not in mixed_layer_names
    ):
        return fancy_field_name

    if basic_field_name in mixed_layer_names:
        return '{0:.0f}-m {1:s}'.format(
            metadata_dict[derived_field_utils.MIXED_LAYER_DEPTH_KEY],
            fancy_field_name
        )

    if basic_field_name in derived_field_utils.HELICITY_NAMES:
        return '0--{0:.0f}-m {1:s}'.format(
            metadata_dict[derived_field_utils.TOP_HEIGHT_KEY],
            fancy_field_name
        )

    if basic_field_name in derived_field_utils.WIND_SHEAR_NAMES:
        if metadata_dict[derived_field_utils.BOTTOM_PRESSURE_KEY] == 'surface':
            return 'surface--{0:.0f}-hPa {1:s}'.format(
                PASCALS_TO_HPA *
                metadata_dict[derived_field_utils.TOP_PRESSURE_KEY],
                fancy_field_name
            )

        return '{0:.0f}--{1:.0f}-hPa {2:s}'.format(
            PASCALS_TO_HPA *
            metadata_dict[derived_field_utils.BOTTOM_PRESSURE_KEY],
            PASCALS_TO_HPA *
            metadata_dict[derived_field_utils.TOP_PRESSURE_KEY],
            fancy_field_name
        )

    if basic_field_name == derived_field_utils.LIFTED_INDEX_NAME:
        return '{0:.0f}-hPa {1:s}'.format(
            PASCALS_TO_HPA *
            metadata_dict[derived_field_utils.TOP_PRESSURE_KEY],
            fancy_field_name
        )

    if basic_field_name == derived_field_utils.PRECIPITABLE_WATER_NAME:
        return 'surface--{0:.0f}-hPa {1:s}'.format(
            PASCALS_TO_HPA *
            metadata_dict[derived_field_utils.TOP_PRESSURE_KEY],
            fancy_field_name
        )

    return fancy_field_name


def plot_field(data_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
               colour_map_object, colour_norm_object, axes_object,
               plot_colour_bar, plot_in_log2_scale=False):
    """Plots one field on a lat/long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param data_matrix: M-by-N numpy array of data values.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg north).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg east).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap` or similar).
    :param colour_norm_object: Colour-normalizer, used to map from physical
        values to colours (instance of `matplotlib.colors.BoundaryNorm` or
        similar).
    :param axes_object: Will plot on this set of axes (instance of
        `matplotlib.axes._subplots.AxesSubplot` or similar).
    :param plot_colour_bar: Boolean flag.
    :param plot_in_log2_scale: Boolean flag.
    :return: is_longitude_positive_in_west: Boolean flag.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    num_grid_rows = data_matrix.shape[0]
    num_grid_columns = data_matrix.shape[1]

    error_checking.assert_is_valid_lat_numpy_array(
        grid_latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        grid_latitudes_deg_n,
        exact_dimensions=numpy.array([num_grid_rows], dtype=int)
    )

    grid_longitudes_to_plot_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitudes_deg_e + 0, allow_nan=False
    )
    is_longitude_positive_in_west = True

    if numpy.any(numpy.diff(grid_longitudes_to_plot_deg_e) < 0):
        grid_longitudes_to_plot_deg_e = (
            lng_conversion.convert_lng_negative_in_west(
                grid_longitudes_to_plot_deg_e
            )
        )

        is_longitude_positive_in_west = False

    error_checking.assert_is_numpy_array(
        grid_longitudes_to_plot_deg_e,
        exact_dimensions=numpy.array([num_grid_columns], dtype=int)
    )

    error_checking.assert_is_boolean(plot_colour_bar)
    error_checking.assert_is_boolean(plot_in_log2_scale)

    # Do actual stuff.
    (
        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e
    ) = grids.latlng_vectors_to_matrices(
        unique_latitudes_deg=grid_latitudes_deg_n,
        unique_longitudes_deg=grid_longitudes_to_plot_deg_e
    )

    if plot_in_log2_scale:
        data_matrix_to_plot = numpy.log2(data_matrix + 1.)
        colour_norm_object = pyplot.Normalize(
            vmin=numpy.log2(colour_norm_object.vmin + 1),
            vmax=numpy.log2(colour_norm_object.vmax + 1)
        )
    else:
        data_matrix_to_plot = data_matrix + 0.

    data_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
    )

    axes_object.pcolor(
        grid_longitude_matrix_deg_e, grid_latitude_matrix_deg_n,
        data_matrix_to_plot,
        cmap=colour_map_object, norm=colour_norm_object,
        edgecolors='None', zorder=-1e11
    )

    if plot_colour_bar:
        colour_bar_object = gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

        if plot_in_log2_scale:
            tick_values = colour_bar_object.get_ticks()
            tick_strings = [
                '{0:.0f}'.format(numpy.power(2., v) - 1) for v in tick_values
            ]

            colour_bar_object.set_ticks(tick_values)
            colour_bar_object.set_ticklabels(tick_strings)

    return is_longitude_positive_in_west
