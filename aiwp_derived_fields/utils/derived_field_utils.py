"""Methods for computing derived fields."""

import time
from itertools import repeat
from multiprocessing import Pool
import numpy
from xcape import core
from scipy.integrate import simpson
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import moisture_conversions as moisture_conv
from aiwp_derived_fields.utils import model_utils
from aiwp_derived_fields.outside_code import sharppy_thermo

NUM_SLICES_FOR_MULTIPROCESSING = 8
HEIGHT_SPACING_FOR_HELICITY_METRES = 250.

HPA_TO_PASCALS = 100.
PASCALS_TO_HPA = 0.01
METRES_TO_MM = 1000.

GRAVITY_M_S02 = 9.80655
WATER_DENSITY_KG_M03 = 1000.
EARTH_RADIUS_METRES = 6371222.9

MOST_UNSTABLE_CAPE_NAME = 'most_unstable_cape_j_kg01'
MOST_UNSTABLE_CIN_NAME = 'most_unstable_cin_j_kg01'
SURFACE_BASED_CAPE_NAME = 'surface_based_cape_j_kg01'
SURFACE_BASED_CIN_NAME = 'surface_based_cin_j_kg01'
MIXED_LAYER_CAPE_NAME = 'mixed_layer_cape_j_kg01'
MIXED_LAYER_CIN_NAME = 'mixed_layer_cin_j_kg01'
LIFTED_INDEX_NAME = 'lifted_index_kelvins'
PRECIPITABLE_WATER_NAME = 'precipitable_water_kg_m02'
SCALAR_WIND_SHEAR_NAME = 'wind_shear_m_s01'
ZONAL_WIND_SHEAR_NAME = 'zonal_wind_shear_m_s01'
MERIDIONAL_WIND_SHEAR_NAME = 'meridional_wind_shear_m_s01'
SCALAR_STORM_MOTION_NAME = 'bunkers_right_mover_storm_motion_m_s01'
ZONAL_STORM_MOTION_NAME = 'bunkers_right_mover_zonal_storm_motion_m_s01'
MERIDIONAL_STORM_MOTION_NAME = (
    'bunkers_right_mover_meridional_storm_motion_m_s01'
)
HELICITY_NAME = 'storm_relative_helicity_m2_s02'
POSITIVE_HELICITY_NAME = 'storm_relative_positive_helicity_m2_s02'
NEGATIVE_HELICITY_NAME = 'storm_relative_negative_helicity_m2_s02'
PBL_HEIGHT_NAME = 'planetary_boundary_layer_height_m_agl'

CAPE_CIN_NAMES = [
    MOST_UNSTABLE_CAPE_NAME, MOST_UNSTABLE_CIN_NAME,
    SURFACE_BASED_CAPE_NAME, SURFACE_BASED_CIN_NAME,
    MIXED_LAYER_CAPE_NAME, MIXED_LAYER_CIN_NAME
]
BASIC_FIELD_NAMES_TO_COMPUTE = CAPE_CIN_NAMES + [
    LIFTED_INDEX_NAME, PRECIPITABLE_WATER_NAME,
    SCALAR_WIND_SHEAR_NAME, SCALAR_STORM_MOTION_NAME,
    HELICITY_NAME, PBL_HEIGHT_NAME
]
BASIC_FIELD_NAMES_COMPUTED = CAPE_CIN_NAMES + [
    LIFTED_INDEX_NAME, PRECIPITABLE_WATER_NAME,
    ZONAL_WIND_SHEAR_NAME, MERIDIONAL_WIND_SHEAR_NAME,
    ZONAL_STORM_MOTION_NAME, MERIDIONAL_STORM_MOTION_NAME,
    POSITIVE_HELICITY_NAME, NEGATIVE_HELICITY_NAME,
    PBL_HEIGHT_NAME
]
WIND_SHEAR_NAMES = [
    SCALAR_WIND_SHEAR_NAME, ZONAL_WIND_SHEAR_NAME, MERIDIONAL_WIND_SHEAR_NAME
]
STORM_MOTION_NAMES = [
    SCALAR_STORM_MOTION_NAME, ZONAL_STORM_MOTION_NAME,
    MERIDIONAL_STORM_MOTION_NAME
]
HELICITY_NAMES = [HELICITY_NAME, POSITIVE_HELICITY_NAME, NEGATIVE_HELICITY_NAME]

BASIC_FIELD_KEY = 'basic_field_name'
PARCEL_SOURCE_KEY = 'parcel_source_string'
MIXED_LAYER_DEPTH_KEY = 'mixed_layer_depth_metres'
TOP_PRESSURE_KEY = 'top_pressure_pascals'
TOP_HEIGHT_KEY = 'top_height_m_agl'
BOTTOM_PRESSURE_KEY = 'bottom_pressure_pascals'

SURFACE_PARCEL_SOURCE_STRING = 'surface'
MIXED_LAYER_PARCEL_SOURCE_STRING = 'mixed-layer'
MOST_UNSTABLE_PARCEL_SOURCE_STRING = 'most-unstable'
PARCEL_SOURCE_STRINGS = [
    SURFACE_PARCEL_SOURCE_STRING, MIXED_LAYER_PARCEL_SOURCE_STRING,
    MOST_UNSTABLE_PARCEL_SOURCE_STRING
]


def __starmap_with_kwargs(pool_object, function_object, args_iter, kwargs_iter):
    """Allows `multiprocessing.Pool.starmap` to work with keyword args.

    :param pool_object: Instance of `multiprocessing.Pool`.
    :param function_object: Function handle.
    :param args_iter: Non-keyword arguments to function.
    :param kwargs_iter: Keyword arguments to function.
    :return: starmap_object: Instance of `multiprocessing.Pool.starmap`.
    """

    args_for_starmap = zip(repeat(function_object), args_iter, kwargs_iter)
    return pool_object.starmap(__apply_args_and_kwargs, args_for_starmap)


def __apply_args_and_kwargs(function_object, args, kwargs):
    return function_object(*args, **kwargs)


def __check_for_matching_grids(forecast_table_xarray, aux_data_matrix):
    """Checks for matching grids between xarray table and numpy matrix.

    This method can only check for matching grid *shapes* -- not precisely
    matching grids -- since numpy arrays do not carry metadata about the grid.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param aux_data_matrix: Should be M-by-N numpy array.  If not, this method
        will throw an error.
    """

    num_grid_rows = len(
        forecast_table_xarray.coords[model_utils.LATITUDE_DEG_NORTH_DIM].values
    )
    num_grid_columns = len(
        forecast_table_xarray.coords[model_utils.LONGITUDE_DEG_EAST_DIM].values
    )

    expected_dim = numpy.array([num_grid_rows, num_grid_columns], dtype=int)
    error_checking.assert_is_numpy_array_without_nan(aux_data_matrix)
    error_checking.assert_is_numpy_array(
        aux_data_matrix, exact_dimensions=expected_dim
    )


def __pressure_level_to_index(forecast_table_xarray, desired_pressure_pascals):
    """Returns array index for a given pressure level in the model.

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param desired_pressure_pascals: Desired pressure.
    :return: pressure_index: Array index (non-negative integer).
    """

    all_pressures_pascals = numpy.round(
        HPA_TO_PASCALS *
        forecast_table_xarray.coords[model_utils.PRESSURE_HPA_DIM].values
    ).astype(int)

    desired_pressure_pascals = int(numpy.round(desired_pressure_pascals))

    return numpy.where(all_pressures_pascals == desired_pressure_pascals)[0][0]


def _height_to_geopotential(height_metres):
    """Converts height to geopotential.

    The input may be a scalar or an array with any shape.

    If the input is surface-relative (i.e., metres above ground level), the
    output will also be surface-relative (i.e., geopotential with respect to
    ground level).

    If the input is sea-level-relative (i.e., metres above sea level), the
    output will also be sea-level-relative (i.e., geopotential with respect to
    sea level).

    :param height_metres: Input (scalar or array of any shape).
    :return: geopotential_m2_s02: Geopotential (units of m^2 s^-2).
    """

    numerator = GRAVITY_M_S02 * EARTH_RADIUS_METRES * height_metres
    denominator = EARTH_RADIUS_METRES + height_metres
    return numerator / denominator


def _geopotential_to_height(geopotential_m2_s02):
    """Converts geopotential to height.

    This method is the inverse of `_height_to_geopotential`.

    :param geopotential_m2_s02: See documentation for
        `_height_to_geopotential`.
    :return: height_metres: Same.
    """

    numerator = EARTH_RADIUS_METRES * geopotential_m2_s02
    denominator = GRAVITY_M_S02 * EARTH_RADIUS_METRES - geopotential_m2_s02
    return numerator / denominator


def _get_slices_for_multiprocessing(num_grid_rows):
    """Returns slices for multiprocessing.

    Each "slice" consists of several grid rows.

    K = number of slices

    :param num_grid_rows: Total number of grid rows.
    :return: start_rows: length-K numpy array with index of each start row.
    :return: end_rows: length-K numpy array with index of each end row.
    """

    slice_indices_normalized = numpy.linspace(
        0, 1, num=NUM_SLICES_FOR_MULTIPROCESSING + 1, dtype=float
    )

    start_rows = numpy.round(
        num_grid_rows * slice_indices_normalized[:-1]
    ).astype(int)

    end_rows = numpy.round(
        num_grid_rows * slice_indices_normalized[1:]
    ).astype(int)

    return start_rows, end_rows


def _get_model_pressure_matrix(forecast_table_xarray, vertical_axis_first):
    """Returns matrix of model pressures.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    V = number of vertical levels in grid

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param vertical_axis_first: Boolean flag.  If True, the output array will
        have shape V x M x N.  If False, the output array will have shape
        M x N x V.
    :return: pressure_matrix_pascals: Output array.
    """

    num_grid_rows = len(
        forecast_table_xarray.coords[model_utils.LATITUDE_DEG_NORTH_DIM].values
    )
    num_grid_columns = len(
        forecast_table_xarray.coords[model_utils.LONGITUDE_DEG_EAST_DIM].values
    )
    pressure_levels_pascals = HPA_TO_PASCALS * forecast_table_xarray.coords[
        model_utils.PRESSURE_HPA_DIM
    ].values.astype(float)

    if not vertical_axis_first:
        pressure_matrix_pascals = numpy.repeat(
            numpy.expand_dims(pressure_levels_pascals, axis=0),
            axis=0, repeats=num_grid_columns
        )
        pressure_matrix_pascals = numpy.repeat(
            numpy.expand_dims(pressure_matrix_pascals, axis=0),
            axis=0, repeats=num_grid_rows
        )

        return pressure_matrix_pascals

    pressure_matrix_pascals = numpy.repeat(
        numpy.expand_dims(pressure_levels_pascals, axis=-1),
        axis=-1, repeats=num_grid_rows
    )
    pressure_matrix_pascals = numpy.repeat(
        numpy.expand_dims(pressure_matrix_pascals, axis=-1),
        axis=-1, repeats=num_grid_columns
    )

    return pressure_matrix_pascals


def _height_agl_to_nearest_pressure_level(
        geopotential_matrix_m2_s02, surface_geopotential_matrix_m2_s02,
        desired_height_m_agl, find_nearest_level_beneath):
    """At every horizontal grid point, finds nearest p-level to a given height.

    This is a ground-relative height, specifically.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    V = number of vertical levels in grid

    :param geopotential_matrix_m2_s02: V-by-M-by-N numpy array of geopotentials.
    :param surface_geopotential_matrix_m2_s02: M-by-N numpy array of surface
        geopotentials.
    :param desired_height_m_agl: Desired height (metres above ground level).
    :param find_nearest_level_beneath: Boolean flag.  If True, will find nearest
        pressure level beneath the desired height (i.e., closer to the surface).
        If False, will just find the nearest pressure level.
    :return: vertical_index_matrix: M-by-N numpy array of indices.  These are
        non-negative array indices into the first axis of
        `geopotential_matrix_m2_s02`, indicating the vertical level nearest to
        the desired ground-relative height.
    """

    desired_sfc_relative_geoptl_m2_s02 = _height_to_geopotential(
        desired_height_m_agl
    )
    desired_geopotential_matrix_m2_s02 = (
        surface_geopotential_matrix_m2_s02 + desired_sfc_relative_geoptl_m2_s02
    )

    if not find_nearest_level_beneath:
        return numpy.argmin(
            numpy.absolute(
                geopotential_matrix_m2_s02 -
                numpy.expand_dims(desired_geopotential_matrix_m2_s02, axis=0)
            ),
            axis=0
        )

    new_geopotential_matrix_m2_s02 = geopotential_matrix_m2_s02 + 0.
    new_geopotential_matrix_m2_s02[
        new_geopotential_matrix_m2_s02 >
        numpy.expand_dims(desired_geopotential_matrix_m2_s02, axis=0)
    ] = 1e12

    return numpy.argmin(
        numpy.absolute(
            geopotential_matrix_m2_s02 -
            numpy.expand_dims(desired_geopotential_matrix_m2_s02, axis=0)
        ),
        axis=0
    )


def _get_mean_wind(
        zonal_wind_matrix_m_s01, meridional_wind_matrix_m_s01,
        bottom_index_matrix, top_index_matrix, pressure_weighted,
        pressure_matrix_pascals, surface_pressure_matrix_pascals):
    """At every horizontal grid point, computes mean wind between two levels.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    V = number of vertical levels in grid

    :param zonal_wind_matrix_m_s01: V-by-M-by-N numpy array of zonal wind speeds
        (metres per second).
    :param meridional_wind_matrix_m_s01: V-by-M-by-N numpy array of meridional
        wind speeds (metres per second).
    :param bottom_index_matrix: M-by-N numpy array of non-negative integers,
        indexing the bottom of the layer at each horizontal grid point.
    :param top_index_matrix: Same but for top of layer.
    :param pressure_weighted: Boolean flag.  If True (False), will compute
        pressure-weighted (straight-up) mean.
    :param pressure_matrix_pascals: V-by-M-by-N numpy array of pressures.
    :param surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.
    :return: mean_zonal_wind_matrix_m_s01: M-by-N numpy array of mean zonal wind
        speeds.
    :return: mean_meridional_wind_matrix_m_s01: M-by-N numpy array of mean
        meridional wind speeds.
    """

    # TODO(thunderhoser): This method computes the mean wind between two
    # vertical levels that exist in the model.  It cannot compute mean wind over
    # arbitrary layers, e.g., 900.7 to 850.5 hPa.  This would involve
    # interpolation, which is computationally expensive!

    # At each horizontal grid point, mask out levels that aren't in the desired
    # layer (between bottom_index and top_index).
    num_vertical_levels = zonal_wind_matrix_m_s01.shape[0]
    vertical_indices = numpy.linspace(
        0, num_vertical_levels - 1, num=num_vertical_levels, dtype=int
    )
    vertical_index_matrix = numpy.expand_dims(vertical_indices, axis=-1)
    vertical_index_matrix = numpy.expand_dims(vertical_index_matrix, axis=-1)

    bottom_index_matrix_3d = numpy.expand_dims(bottom_index_matrix, axis=0)
    top_index_matrix_3d = numpy.expand_dims(top_index_matrix, axis=0)

    mask_out_matrix = numpy.logical_or(
        vertical_index_matrix < bottom_index_matrix_3d,
        vertical_index_matrix > top_index_matrix_3d
    )

    # At each horizontal grid point, also mask out levels below the surface.
    mask_out_matrix = numpy.logical_or(
        mask_out_matrix,
        pressure_matrix_pascals - 0.1 >
        numpy.expand_dims(surface_pressure_matrix_pascals, axis=0)
    )

    # At each horizontal grid point, compute the mean wind using only the
    # unmasked vertical levels.
    if pressure_weighted:
        masked_pressure_matrix_pascals = numpy.ma.array(
            data=pressure_matrix_pascals, mask=mask_out_matrix
        )

        mean_zonal_wind_matrix_m_s01 = numpy.ma.average(
            zonal_wind_matrix_m_s01,
            weights=masked_pressure_matrix_pascals, axis=0
        )
        mean_meridional_wind_matrix_m_s01 = numpy.ma.average(
            meridional_wind_matrix_m_s01,
            weights=masked_pressure_matrix_pascals, axis=0
        )
    else:
        masked_zonal_wind_matrix_m_s01 = numpy.ma.array(
            data=zonal_wind_matrix_m_s01, mask=mask_out_matrix
        )
        masked_meridional_wind_matrix_m_s01 = numpy.ma.array(
            data=meridional_wind_matrix_m_s01, mask=mask_out_matrix
        )

        mean_zonal_wind_matrix_m_s01 = numpy.ma.mean(
            masked_zonal_wind_matrix_m_s01, axis=0
        )
        mean_meridional_wind_matrix_m_s01 = numpy.ma.mean(
            masked_meridional_wind_matrix_m_s01, axis=0
        )

    mean_zonal_wind_matrix_m_s01 = numpy.where(
        numpy.ma.getmask(mean_zonal_wind_matrix_m_s01) == False,
        numpy.ma.getdata(mean_zonal_wind_matrix_m_s01),
        numpy.nan
    )
    mean_meridional_wind_matrix_m_s01 = numpy.where(
        numpy.ma.getmask(mean_meridional_wind_matrix_m_s01) == False,
        numpy.ma.getdata(mean_meridional_wind_matrix_m_s01),
        numpy.nan
    )

    return mean_zonal_wind_matrix_m_s01, mean_meridional_wind_matrix_m_s01


def _get_pbl_height(
        theta_v_matrix_kelvins, geopotential_matrix_m2_s02,
        theta_v_deviation_threshold_kelvins=0.5):
    """At every horizontal grid point, computes PBL height.

    PBL = planetary boundary layer

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    V = number of pressure levels
    V + 1 = total number of vertical levels, including surface

    :param theta_v_matrix_kelvins: (V + 1)-by-M-by-N numpy array of virtual
        potential temperatures.  theta_v_matrix_kelvins[0, ...] is the surface,
        and theta_v_matrix_kelvins[1:, ...] is everywhere aloft.
    :param geopotential_matrix_m2_s02: Same but with geopotential values
        (m^2 s^-2).
    :param theta_v_deviation_threshold_kelvins: Deviation threshold used to
        define top of PBL.  Specifically, the PBL top is the lowest height where
        theta_v >= theta_v_surface + `theta_v_deviation_threshold_kelvins`.
    :return: pbl_height_matrix_m_agl: M-by-N numpy array of PBL heights (metres
        above ground level).
    """

    error_checking.assert_is_greater(theta_v_deviation_threshold_kelvins, 0.)

    # At every horizontal grid point, find lowest height where theta_v exceeds
    # critical value (surface theta_v + `theta_v_deviation_threshold_kelvins`).
    theta_v_threshold_matrix_kelvins = (
        theta_v_matrix_kelvins[0, ...] +
        theta_v_deviation_threshold_kelvins
    )
    exceedance_flag_matrix = (
        theta_v_matrix_kelvins >=
        numpy.expand_dims(theta_v_threshold_matrix_kelvins, axis=0)
    )

    # Mask out theta_v values below the surface.
    mask_out_matrix = (
        geopotential_matrix_m2_s02 <
        geopotential_matrix_m2_s02[[0], ...] - 0.1
    )
    exceedance_flag_matrix = numpy.logical_and(
        exceedance_flag_matrix,
        numpy.invert(mask_out_matrix)
    )
    top_index_matrix = numpy.argmax(exceedance_flag_matrix, axis=0)

    # At any horizontal grid point where theta_v does not exceed the threshold
    # at any vertical level, this matrix will have a False value.
    bad_flag_matrix = numpy.invert(exceedance_flag_matrix[
        top_index_matrix,
        numpy.arange(top_index_matrix.shape[0])[:, None],
        numpy.arange(top_index_matrix.shape[1])
    ])

    # Create 2-by-M-by-N matrix of theta_v values, where the two vertical levels
    # sandwich the PBL top.
    top_theta_v_matrix_kelvins = theta_v_matrix_kelvins[
        top_index_matrix,
        numpy.arange(top_index_matrix.shape[0])[:, None],
        numpy.arange(top_index_matrix.shape[1])
    ]
    bottom_theta_v_matrix_kelvins = theta_v_matrix_kelvins[
        top_index_matrix - 1,
        numpy.arange(top_index_matrix.shape[0])[:, None],
        numpy.arange(top_index_matrix.shape[1])
    ]

    # Also create 2-by-M-by-N matrix of geopotential values, where the two
    # vertical levels sandwich the PBL top.
    top_geopotential_matrix_m2_s02 = geopotential_matrix_m2_s02[
        top_index_matrix,
        numpy.arange(top_index_matrix.shape[0])[:, None],
        numpy.arange(top_index_matrix.shape[1])
    ]
    bottom_geopotential_matrix_m2_s02 = geopotential_matrix_m2_s02[
        top_index_matrix - 1,
        numpy.arange(top_index_matrix.shape[0])[:, None],
        numpy.arange(top_index_matrix.shape[1])
    ]

    # Now apply the linear-interpolation formula to find, at every horizontal
    # grid point, the geopotential where theta_v crosses the critical value.
    y0_term = bottom_geopotential_matrix_m2_s02
    x_minus_x0_term = (
        theta_v_threshold_matrix_kelvins - bottom_theta_v_matrix_kelvins
    )
    y1_minus_y0_term = (
        top_geopotential_matrix_m2_s02 - bottom_geopotential_matrix_m2_s02
    )
    x1_minus_x0_term = (
        top_theta_v_matrix_kelvins - bottom_theta_v_matrix_kelvins
    )
    pbl_top_geopotential_matrix_m2_s02 = (
        y0_term + x_minus_x0_term * y1_minus_y0_term / x1_minus_x0_term
    )

    pbl_top_geopotential_matrix_m2_s02[bad_flag_matrix] = numpy.nan

    # At every horizontal grid point where the bottom of the sandwich is
    # sub-surface, do the interpolation again, using the surface itself as the
    # bottom of the sandwich.
    bad_flag_matrix = (
        bottom_geopotential_matrix_m2_s02 <
        geopotential_matrix_m2_s02[0, ...] - 0.1
    )

    bottom_geopotential_matrix_m2_s02 = geopotential_matrix_m2_s02[0, ...]
    bottom_theta_v_matrix_kelvins = theta_v_matrix_kelvins[0, ...]

    y0_term = bottom_geopotential_matrix_m2_s02
    x_minus_x0_term = (
        theta_v_threshold_matrix_kelvins - bottom_theta_v_matrix_kelvins
    )
    y1_minus_y0_term = (
        top_geopotential_matrix_m2_s02 - bottom_geopotential_matrix_m2_s02
    )
    x1_minus_x0_term = (
        top_theta_v_matrix_kelvins - bottom_theta_v_matrix_kelvins
    )
    second_pbl_top_geopotential_matrix_m2_s02 = (
        y0_term + x_minus_x0_term * y1_minus_y0_term / x1_minus_x0_term
    )

    pbl_top_geopotential_matrix_m2_s02[bad_flag_matrix] = (
        second_pbl_top_geopotential_matrix_m2_s02[bad_flag_matrix]
    )

    # Convert PBL geopotential to height above ground.
    pbl_top_sfc_relative_geoptl_matrix_m2_s02 = (
        pbl_top_geopotential_matrix_m2_s02 - geopotential_matrix_m2_s02[0, ...]
    )
    pbl_height_matrix_m_agl = _geopotential_to_height(
        pbl_top_sfc_relative_geoptl_matrix_m2_s02
    )

    # Because of the "minus 0.1" in my comparisons with the surface, sometimes
    # the PBL height is slightly negative.
    pbl_height_matrix_m_agl = numpy.maximum(pbl_height_matrix_m_agl, 0.)
    return pbl_height_matrix_m_agl


def _interp_pressure_to_surface(
        log10_pressure_matrix_pascals, geopotential_matrix_m2_s02,
        surface_geopotential_matrix_m2_s02, use_spline, test_mode=False):
    """At every horizontal grid point, interpolates pressure to surface.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    V = number of vertical levels in grid

    :param log10_pressure_matrix_pascals: V-by-M-by-N numpy array of
        log(pressure) values.
    :param geopotential_matrix_m2_s02: V-by-M-by-N numpy array of geopotentials
        (m^2 s^-2).
    :param surface_geopotential_matrix_m2_s02: M-by-N numpy array of surface
        geopotentials.
    :param use_spline: Boolean flag.
    :param test_mode: Leave this alone.
    :return: surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.
    """

    log10_surface_pressure_matrix_pascals = numpy.full(
        surface_geopotential_matrix_m2_s02.shape, numpy.nan
    )

    num_grid_rows = surface_geopotential_matrix_m2_s02.shape[0]
    num_grid_columns = surface_geopotential_matrix_m2_s02.shape[1]

    if test_mode:
        rounded_geopotential_matrix_m2_s02 = geopotential_matrix_m2_s02
    else:
        rounded_geopotential_matrix_m2_s02 = number_rounding.round_to_nearest(
            geopotential_matrix_m2_s02, 1000.
        )

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            _, inds = numpy.unique(
                rounded_geopotential_matrix_m2_s02[:, i, j], return_index=True
            )

            if use_spline:
                interp_object = InterpolatedUnivariateSpline(
                    x=geopotential_matrix_m2_s02[:, i, j][inds],
                    y=log10_pressure_matrix_pascals[:, i, j][inds],
                    k=1, ext='extrapolate', check_finite=False
                )
            else:
                interp_object = interp1d(
                    x=geopotential_matrix_m2_s02[:, i, j][inds],
                    y=log10_pressure_matrix_pascals[:, i, j][inds],
                    kind='linear',
                    axis=0,
                    bounds_error=False,
                    fill_value='extrapolate',
                    assume_sorted=True
                )

            log10_surface_pressure_matrix_pascals[i, j] = interp_object(
                surface_geopotential_matrix_m2_s02[i, j]
            )

    if test_mode:
        return log10_surface_pressure_matrix_pascals

    return 10 ** log10_surface_pressure_matrix_pascals


def _interp_wind_to_heights_agl(
        wind_matrix_m_s01, geopotential_matrix_m2_s02,
        surface_geopotential_matrix_m2_s02, target_heights_m_agl, use_spline,
        test_mode=False):
    """At every horizontal grid pt, interpolates wind to ground-relative hgts.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    V = number of vertical levels in grid
    T = number of target heights

    :param wind_matrix_m_s01: V-by-M-by-N numpy array of wind speeds (metres per
        second).  These could be scalar wind speeds, representing the magnitude
        of the wind vector, or single-component wind speeds, representing either
        the zonal or meridional part.
    :param geopotential_matrix_m2_s02: V-by-M-by-N numpy array of geopotentials
        (m^2 s^-2).
    :param surface_geopotential_matrix_m2_s02: M-by-N numpy array of surface
        geopotentials.
    :param target_heights_m_agl: length-T numpy array of target heights (metres
        above ground level).  Will interpolate to these heights.
    :param use_spline: Boolean flag.
    :param test_mode: Leave this alone.
    :return: interp_wind_matrix_m_s01: M-by-N numpy array of interpolated wind
        speeds.
    """

    target_sfc_relative_geoptls_m2_s02 = _height_to_geopotential(
        target_heights_m_agl
    )
    target_geopotential_matrix_m2_s02 = numpy.stack([
        surface_geopotential_matrix_m2_s02 + g
        for g in target_sfc_relative_geoptls_m2_s02
    ], axis=0)

    num_grid_rows = surface_geopotential_matrix_m2_s02.shape[0]
    num_grid_columns = surface_geopotential_matrix_m2_s02.shape[1]
    num_target_heights = len(target_heights_m_agl)
    interp_wind_matrix_m_s01 = numpy.full(
        (num_target_heights, num_grid_rows, num_grid_columns), numpy.nan
    )

    if test_mode:
        rounded_geopotential_matrix_m2_s02 = geopotential_matrix_m2_s02
    else:
        rounded_geopotential_matrix_m2_s02 = number_rounding.round_to_nearest(
            geopotential_matrix_m2_s02, 1000.
        )

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            _, inds = numpy.unique(
                rounded_geopotential_matrix_m2_s02[:, i, j], return_index=True
            )

            if use_spline:
                interp_object = InterpolatedUnivariateSpline(
                    x=geopotential_matrix_m2_s02[:, i, j][inds],
                    y=wind_matrix_m_s01[:, i, j][inds],
                    k=1, ext='extrapolate', check_finite=False
                )
            else:
                interp_object = interp1d(
                    x=geopotential_matrix_m2_s02[:, i, j][inds],
                    y=wind_matrix_m_s01[:, i, j][inds],
                    kind='linear',
                    axis=0,
                    bounds_error=False,
                    fill_value='extrapolate',
                    assume_sorted=True
                )

            interp_wind_matrix_m_s01[:, i, j] = interp_object(
                target_geopotential_matrix_m2_s02[:, i, j]
            )

    return interp_wind_matrix_m_s01


def _interp_humidity_to_surface(
        log10_pressure_matrix_pascals, spec_humidity_matrix_kg_kg01,
        log10_surface_pressure_matrix_pascals, use_spline=True):
    """At every horizontal grid point, interpolates specific humidity to sfc.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    V = number of vertical levels in grid

    :param log10_pressure_matrix_pascals: V-by-M-by-N numpy array of
        log(pressure) values.
    :param spec_humidity_matrix_kg_kg01: V-by-M-by-N numpy array of specific
        humidities (kg/kg).
    :param log10_surface_pressure_matrix_pascals: M-by-N numpy array of
        log(surface_pressure) values.
    :param use_spline: Boolean flag.
    :return: surface_spec_humidity_matrix_kg_kg01: M-by-N numpy array of
        surface specific humidities.
    """

    # TODO(thunderhoser): Didn't write unit tests for this method, since it's so
    # similar to _interp_pressure_to_surface and _interp_wind_to_heights_agl,
    # for which I did write unit tests.

    surface_spec_humidity_matrix_kg_kg01 = numpy.full(
        log10_surface_pressure_matrix_pascals.shape, numpy.nan
    )

    num_grid_rows = log10_surface_pressure_matrix_pascals.shape[0]
    num_grid_columns = log10_surface_pressure_matrix_pascals.shape[1]

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if use_spline:
                interp_object = InterpolatedUnivariateSpline(
                    x=log10_pressure_matrix_pascals[:, i, j][::-1],
                    y=spec_humidity_matrix_kg_kg01[:, i, j][::-1],
                    k=1, ext='extrapolate', check_finite=False
                )
            else:
                interp_object = interp1d(
                    x=log10_pressure_matrix_pascals[:, i, j][::-1],
                    y=spec_humidity_matrix_kg_kg01[:, i, j][::-1],
                    kind='linear',
                    axis=0,
                    bounds_error=False,
                    fill_value='extrapolate',
                    assume_sorted=True
                )

            surface_spec_humidity_matrix_kg_kg01[i, j] = interp_object(
                log10_surface_pressure_matrix_pascals[i, j]
            )

    return numpy.maximum(surface_spec_humidity_matrix_kg_kg01, 0.)


def _integrate_to_precipitable_water(
        pressure_matrix_pascals, spec_humidity_matrix_kg_kg01, test_mode=False):
    """At every horizontal grid point, integrates to get precipitable water.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    V = number of vertical levels in grid

    :param pressure_matrix_pascals: V-by-M-by-N numpy array of pressure values.
    :param spec_humidity_matrix_kg_kg01: V-by-M-by-N numpy array of specific
        humidities.
    :param test_mode: Leave this alone.
    :return: precipitable_water_matrix_kg_m02: M-by-N numpy array of
        precipitable-water values.
    """

    precipitable_water_matrix_kg_m02 = numpy.full(
        pressure_matrix_pascals.shape[1:], numpy.nan
    )

    num_grid_rows = pressure_matrix_pascals.shape[1]
    num_grid_columns = pressure_matrix_pascals.shape[2]
    pressure_sort_index_matrix = numpy.argsort(-pressure_matrix_pascals, axis=0)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            inds = pressure_sort_index_matrix[:, i, j]
            subinds = numpy.where(numpy.invert(
                numpy.isnan(spec_humidity_matrix_kg_kg01[:, i, j][inds])
            ))[0]

            if len(subinds) < 2:
                continue

            if test_mode:
                these_humid_kg_kg01 = (
                    spec_humidity_matrix_kg_kg01[:, i, j][inds][subinds]
                )
                these_pressures_pascals = (
                    pressure_matrix_pascals[:, i, j][inds][subinds]
                )
                precipitable_water_matrix_kg_m02[i, j] = -numpy.sum(
                    0.5 *
                    (these_humid_kg_kg01[:-1] + these_humid_kg_kg01[1:]) *
                    (these_pressures_pascals[:-1] - these_pressures_pascals[1:])
                )

                continue

            try:
                precipitable_water_matrix_kg_m02[i, j] = simpson(
                    y=spec_humidity_matrix_kg_kg01[:, i, j][inds][subinds],
                    x=pressure_matrix_pascals[:, i, j][inds][subinds],
                    axis=0,
                    even='simpson'
                )
            except:
                precipitable_water_matrix_kg_m02[i, j] = simpson(
                    y=spec_humidity_matrix_kg_kg01[:, i, j][inds][subinds],
                    x=pressure_matrix_pascals[:, i, j][inds][subinds],
                    axis=0,
                    even='avg'
                )

    coefficient = -METRES_TO_MM / (WATER_DENSITY_KG_M03 * GRAVITY_M_S02)
    return coefficient * precipitable_water_matrix_kg_m02


def _estimate_surface_dewpoint(
        forecast_table_xarray, surface_pressure_matrix_pascals,
        do_multiprocessing, use_spline=True):
    """Estimates surface dewpoint at every grid point.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.  This method simply trusts that the grid setup is correct --
        i.e., that the order of latitudes and longitudes in this array is the
        same as in `forecast_table_xarray`.  If you just use the
        `surface_pressure_matrix_pascals` created by
        `_estimate_surface_pressure`, there should be no problem with the grids
        lining up.
    :param do_multiprocessing: See documentation for
        `_estimate_surface_pressure`.
    :param use_spline: Same.
    :return: surface_dewpoint_matrix_kelvins: M-by-N numpy array of estimated
        surface dewpoints.
    """

    # TODO(thunderhoser): Didn't write unit tests for this method, since it's a
    # thin wrapper for _interp_humidity_to_surface.

    pressure_matrix_pascals = _get_model_pressure_matrix(
        forecast_table_xarray=forecast_table_xarray,
        vertical_axis_first=True
    )
    spec_humidity_matrix_kg_kg01 = forecast_table_xarray[
        model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY
    ].values[0, ...]

    exec_start_time_unix_sec = time.time()

    if do_multiprocessing:
        start_rows, end_rows = _get_slices_for_multiprocessing(
            num_grid_rows=pressure_matrix_pascals.shape[1]
        )

        argument_list = []

        for s, e in zip(start_rows, end_rows):
            argument_list.append((
                numpy.log10(pressure_matrix_pascals[:, s:e, :]),
                spec_humidity_matrix_kg_kg01[:, s:e, :],
                numpy.log10(surface_pressure_matrix_pascals[s:e, :]),
                use_spline
            ))

        surface_spec_humidity_matrix_kg_kg01 = numpy.full(
            pressure_matrix_pascals.shape[1:], numpy.nan
        )

        with Pool() as pool_object:
            submatrices = pool_object.starmap(
                _interp_humidity_to_surface, argument_list
            )

            for k in range(len(start_rows)):
                s = start_rows[k]
                e = end_rows[k]
                surface_spec_humidity_matrix_kg_kg01[s:e, :] = submatrices[k]

        assert not numpy.any(numpy.isnan(surface_spec_humidity_matrix_kg_kg01))
    else:
        surface_spec_humidity_matrix_kg_kg01 = _interp_humidity_to_surface(
            log10_pressure_matrix_pascals=numpy.log10(pressure_matrix_pascals),
            spec_humidity_matrix_kg_kg01=spec_humidity_matrix_kg_kg01,
            log10_surface_pressure_matrix_pascals=numpy.log10(
                surface_pressure_matrix_pascals
            ),
            use_spline=use_spline
        )

    print('Estimating surface specific humidity took {0:.1f} seconds.'.format(
        time.time() - exec_start_time_unix_sec
    ))

    surface_temp_matrix_kelvins = forecast_table_xarray[
        model_utils.TEMPERATURE_2METRES_KELVINS_KEY
    ].values[0, ...]

    return moisture_conv.specific_humidity_to_dewpoint(
        specific_humidities_kg_kg01=surface_spec_humidity_matrix_kg_kg01,
        temperatures_kelvins=surface_temp_matrix_kelvins,
        total_pressures_pascals=surface_pressure_matrix_pascals
    )


def _estimate_surface_pressure(
        forecast_table_xarray, surface_geopotential_matrix_m2_s02,
        do_multiprocessing, use_spline=True, test_mode=False):
    """Estimates surface pressure at every grid point.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param surface_geopotential_matrix_m2_s02: Same.
    :param do_multiprocessing: Boolean flag.  If True, will parallelize the
        interpolation into many processes.  If False, will do it all
        sequentially.
    :param use_spline: Boolean flag.  If True, will use spline interpolation
        with degree = 1.  If False, will use straight-up linear interpolation.
        Spline interpolation is faster in scipy!
    :param test_mode: Leave this alone.
    :return: surface_pressure_matrix_pascals: M-by-N numpy array of estimated
        surface pressures.
    """

    # Create pressure matrix with dimensions V x M x N, where V = number of
    # model levels.
    pressure_matrix_pascals = _get_model_pressure_matrix(
        forecast_table_xarray=forecast_table_xarray,
        vertical_axis_first=True
    )

    # Create pressure matrix with dimensions (V + 1) x M x N, using sea-level
    # pressures from model.
    sea_level_pressure_matrix_pascals = forecast_table_xarray[
        model_utils.SEA_LEVEL_PRESSURE_PASCALS_KEY
    ].values[0, ...]

    pressure_matrix_pascals = numpy.concatenate([
        pressure_matrix_pascals,
        numpy.expand_dims(sea_level_pressure_matrix_pascals, axis=0)
    ], axis=0)

    # Create geopotential matrix with dimensions (V + 1) x M x N, using the fact
    # that geopotential = 0 at sea level by definition.
    geopotential_matrix_m2_s02 = forecast_table_xarray[
        model_utils.GEOPOTENTIAL_M2_S02_KEY
    ].values[0, ...]

    sea_level_geopotential_matrix_m2_s02 = numpy.zeros_like(
        geopotential_matrix_m2_s02[[0], ...]
    )
    geopotential_matrix_m2_s02 = numpy.concatenate(
        [geopotential_matrix_m2_s02, sea_level_geopotential_matrix_m2_s02],
        axis=0
    )

    # Do the interpolation.
    exec_start_time_unix_sec = time.time()

    if do_multiprocessing:
        start_rows, end_rows = _get_slices_for_multiprocessing(
            num_grid_rows=pressure_matrix_pascals.shape[1]
        )

        argument_list = []

        for s, e in zip(start_rows, end_rows):
            if test_mode:
                argument_list.append((
                    pressure_matrix_pascals[:, s:e, :],
                    geopotential_matrix_m2_s02[:, s:e, :],
                    surface_geopotential_matrix_m2_s02[s:e, :],
                    use_spline,
                    True
                ))
            else:
                argument_list.append((
                    numpy.log10(pressure_matrix_pascals[:, s:e, :]),
                    geopotential_matrix_m2_s02[:, s:e, :],
                    surface_geopotential_matrix_m2_s02[s:e, :],
                    use_spline
                ))

        surface_pressure_matrix_pascals = numpy.full(
            pressure_matrix_pascals.shape[1:], numpy.nan
        )

        with Pool() as pool_object:
            submatrices = pool_object.starmap(
                _interp_pressure_to_surface, argument_list
            )

            for k in range(len(start_rows)):
                s = start_rows[k]
                e = end_rows[k]
                surface_pressure_matrix_pascals[s:e, :] = submatrices[k]

        assert not numpy.any(numpy.isnan(surface_pressure_matrix_pascals))
    else:
        if test_mode:
            surface_pressure_matrix_pascals = _interp_pressure_to_surface(
                log10_pressure_matrix_pascals=pressure_matrix_pascals,
                geopotential_matrix_m2_s02=geopotential_matrix_m2_s02,
                surface_geopotential_matrix_m2_s02=
                surface_geopotential_matrix_m2_s02,
                use_spline=use_spline,
                test_mode=True
            )
        else:
            surface_pressure_matrix_pascals = _interp_pressure_to_surface(
                log10_pressure_matrix_pascals=
                numpy.log10(pressure_matrix_pascals),
                geopotential_matrix_m2_s02=geopotential_matrix_m2_s02,
                surface_geopotential_matrix_m2_s02=
                surface_geopotential_matrix_m2_s02,
                use_spline=use_spline
            )

    print('Estimating surface pressure took {0:.1f} seconds.'.format(
        time.time() - exec_start_time_unix_sec
    ))

    return surface_pressure_matrix_pascals


def create_field_name(metadata_dict):
    """Creates name for derived field.

    :param metadata_dict: See output documentation for `parse_field_name`.
    :return: derived_field_name: String.
    """

    basic_field_name = metadata_dict[BASIC_FIELD_KEY]
    derived_field_name = basic_field_name.replace('_', '-')

    if basic_field_name in STORM_MOTION_NAMES:
        return derived_field_name
    if basic_field_name == PBL_HEIGHT_NAME:
        return derived_field_name

    if basic_field_name in HELICITY_NAMES:
        derived_field_name += '_top-height-m-agl={0:.4f}'.format(
            metadata_dict[TOP_HEIGHT_KEY]
        )
        return derived_field_name

    if (
            basic_field_name in CAPE_CIN_NAMES
            and basic_field_name not in
            [MIXED_LAYER_CAPE_NAME, MIXED_LAYER_CIN_NAME]
    ):
        return derived_field_name

    if basic_field_name in [MIXED_LAYER_CAPE_NAME, MIXED_LAYER_CIN_NAME]:
        derived_field_name += '_ml-depth-metres={0:.4f}'.format(
            metadata_dict[MIXED_LAYER_DEPTH_KEY]
        )
        return derived_field_name

    if basic_field_name in (
            [LIFTED_INDEX_NAME, PRECIPITABLE_WATER_NAME] + WIND_SHEAR_NAMES
    ):
        derived_field_name += '_top-pressure-pascals={0:d}'.format(
            int(numpy.round(metadata_dict[TOP_PRESSURE_KEY]))
        )

        if basic_field_name not in WIND_SHEAR_NAMES:
            return derived_field_name

    # If execution makes it to this point,
    # `basic_field_name in WIND_SHEAR_NAMES`.
    if metadata_dict[BOTTOM_PRESSURE_KEY] == 'surface':
        derived_field_name += '_bottom-pressure-pascals={0:s}'.format(
            metadata_dict[BOTTOM_PRESSURE_KEY]
        )
    else:
        derived_field_name += '_bottom-pressure-pascals={0:d}'.format(
            int(numpy.round(metadata_dict[BOTTOM_PRESSURE_KEY]))
        )

    return derived_field_name


def parse_field_name(derived_field_name, is_field_to_compute):
    """Parses name of derived field.

    :param derived_field_name: Name of derived field.
    :param is_field_to_compute: Boolean flag.  If True, the field is something
        to be computed but not yet computed.  If False, the field is something
        already computed.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict["basic_field_name"]: Name of basic field, without any options.
    metadata_dict["parcel_source_string"]: Parcel source (surface, mixed-layer,
        or most unstable).  This variable is only for CAPE and CIN; for other
        fields, it is None.
    metadata_dict["mixed_layer_depth_metres"]: Mixed-layer depth (only for
        mixed-layer CAPE or CIN; otherwise, None).
    metadata_dict["top_pressure_pascals"]: Top pressure level (only for certain
        fields; otherwise, None).
    metadata_dict["top_height_m_agl"]: Top height in metres above ground level
        (only for certain fields; otherwise, None).
    metadata_dict["bottom_pressure_pascals"]: Bottom pressure level (only for
        certain fields; otherwise, None).
    """

    error_checking.assert_is_string(derived_field_name)
    error_checking.assert_is_boolean(is_field_to_compute)

    valid_basic_field_names = (
        BASIC_FIELD_NAMES_TO_COMPUTE if is_field_to_compute
        else BASIC_FIELD_NAMES_COMPUTED
    )

    basic_field_name = derived_field_name.split('_')[0].replace('-', '_')
    if basic_field_name not in valid_basic_field_names:
        error_string = (
            'Cannot find basic field name in string "{0:s}".  The string '
            'should start with one of the following:\n{1:s}'
        ).format(
            derived_field_name,
            str(valid_basic_field_names)
        )

        raise ValueError(error_string)

    metadata_dict = {
        BASIC_FIELD_KEY: basic_field_name,
        PARCEL_SOURCE_KEY: None,
        MIXED_LAYER_DEPTH_KEY: None,
        TOP_PRESSURE_KEY: None,
        TOP_HEIGHT_KEY: None,
        BOTTOM_PRESSURE_KEY: None
    }

    if basic_field_name in STORM_MOTION_NAMES + [PBL_HEIGHT_NAME]:
        return metadata_dict

    if basic_field_name in [MOST_UNSTABLE_CAPE_NAME, MOST_UNSTABLE_CIN_NAME]:
        metadata_dict[PARCEL_SOURCE_KEY] = MOST_UNSTABLE_PARCEL_SOURCE_STRING
    if basic_field_name in [MIXED_LAYER_CAPE_NAME, MIXED_LAYER_CIN_NAME]:
        metadata_dict[PARCEL_SOURCE_KEY] = MIXED_LAYER_PARCEL_SOURCE_STRING
    if basic_field_name in [SURFACE_BASED_CAPE_NAME, SURFACE_BASED_CIN_NAME]:
        metadata_dict[PARCEL_SOURCE_KEY] = SURFACE_PARCEL_SOURCE_STRING

    if (
            basic_field_name in CAPE_CIN_NAMES
            and basic_field_name not in
            [MIXED_LAYER_CAPE_NAME, MIXED_LAYER_CIN_NAME]
    ):
        return metadata_dict

    if basic_field_name in [MIXED_LAYER_CAPE_NAME, MIXED_LAYER_CIN_NAME]:
        this_word = derived_field_name.split('_')[1]
        assert this_word.startswith('ml-depth-metres=')

        mixed_layer_depth_metres = float(
            this_word.replace('ml-depth-metres=', '', 1)
        )
        assert mixed_layer_depth_metres > 0.

        metadata_dict[MIXED_LAYER_DEPTH_KEY] = mixed_layer_depth_metres
        return metadata_dict

    if basic_field_name in HELICITY_NAMES:
        this_word = derived_field_name.split('_')[1]
        assert this_word.startswith('top-height-m-agl=')

        top_height_m_agl = float(
            this_word.replace('top-height-m-agl=', '', 1)
        )
        assert top_height_m_agl > 0.

        metadata_dict[TOP_HEIGHT_KEY] = top_height_m_agl
        return metadata_dict

    if basic_field_name in (
            [LIFTED_INDEX_NAME, PRECIPITABLE_WATER_NAME] + WIND_SHEAR_NAMES
    ):
        this_word = derived_field_name.split('_')[1]
        assert this_word.startswith('top-pressure-pascals=')

        top_pressure_pascals = int(
            this_word.replace('top-pressure-pascals=', '', 1)
        )
        assert top_pressure_pascals > 0

        metadata_dict[TOP_PRESSURE_KEY] = top_pressure_pascals

        if basic_field_name not in WIND_SHEAR_NAMES:
            return metadata_dict

    # If execution makes it to this point,
    # `basic_field_name in WIND_SHEAR_NAMES`.
    this_word = derived_field_name.split('_')[2]
    assert this_word.startswith('bottom-pressure-pascals=')

    this_word = this_word.replace('bottom-pressure-pascals=', '', 1)
    if this_word == 'surface':
        bottom_pressure_pascals = this_word
    else:
        bottom_pressure_pascals = int(this_word)
        assert bottom_pressure_pascals > metadata_dict[TOP_PRESSURE_KEY]

    metadata_dict[BOTTOM_PRESSURE_KEY] = bottom_pressure_pascals
    return metadata_dict


def get_cape_and_cin(
        forecast_table_xarray, do_multiprocessing, parcel_source_string,
        mixed_layer_depth_metres=None,
        surface_geopotential_matrix_m2_s02=None,
        surface_pressure_matrix_pascals=None,
        surface_dewpoint_matrix_kelvins=None):
    """Computes CAPE and CIN at every grid point.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid

    :param forecast_table_xarray: xarray table in format returned by
        `model_io.read_file`.
    :param do_multiprocessing: Boolean flag.  If True, will parallelize the
        calculation into many processes.  If False, will do it all sequentially.
    :param parcel_source_string: Parcel source.  This string must belong to the
        list `PARCEL_SOURCE_STRINGS`.
    :param mixed_layer_depth_metres: Mixed-layer depth.  If
        `parcel_source_string` is not the mixed layer, just leave this argument
        alone.
    :param surface_geopotential_matrix_m2_s02: M-by-N numpy array of surface
        geopotentials (m^2 s^-2).  If you decide to pass
        `surface_pressure_matrix_pascals`, this can be None.  But if you do not
        pass `surface_pressure_matrix_pascals`, this matrix is needed to compute
        `surface_pressure_matrix_pascals`.
    :param surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.  If this is None, `surface_geopotential_matrix_m2_s02` will
        be used to compute surface pressures.
    :param surface_dewpoint_matrix_kelvins: M-by-N numpy array of surface
        dewpoints.  If this is None, `surface_pressure_matrix_pascals` will
        be used to compute surface dewpoints.
    :return: cape_matrix_j_kg01: M-by-N numpy array of CAPE values (Joules per
        kilogram).
    :return: cin_matrix_j_kg01: M-by-N numpy array of CIN values (Joules per
        kilogram).
    :return: surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.
    :return: surface_dewpoint_matrix_kelvins: M-by-N numpy array of surface
        dewpoints.
    """

    # Check input args.
    if surface_geopotential_matrix_m2_s02 is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_geopotential_matrix_m2_s02
        )
    if surface_pressure_matrix_pascals is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_pressure_matrix_pascals
        )
    if surface_dewpoint_matrix_kelvins is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_dewpoint_matrix_kelvins
        )

    error_checking.assert_is_boolean(do_multiprocessing)
    error_checking.assert_is_string(parcel_source_string)

    if parcel_source_string not in PARCEL_SOURCE_STRINGS:
        error_string = (
            'Parcel source ("{0:s}") should be one of the following:\n{1:s}'
        ).format(
            parcel_source_string,
            str(PARCEL_SOURCE_STRINGS)
        )

        raise ValueError(error_string)

    if parcel_source_string != MIXED_LAYER_PARCEL_SOURCE_STRING:
        mixed_layer_depth_metres = None

    if mixed_layer_depth_metres is not None:
        error_checking.assert_is_greater(mixed_layer_depth_metres, 0.)

    # Extract pressure, temperature, and humidity data from model.
    pressure_levels_pascals = HPA_TO_PASCALS * forecast_table_xarray.coords[
        model_utils.PRESSURE_HPA_DIM
    ].values.astype(float)

    temp_matrix_kelvins = forecast_table_xarray[
        model_utils.TEMPERATURE_KELVINS_KEY
    ].values[0, ...]

    spec_humidity_matrix_kg_kg01 = forecast_table_xarray[
        model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY
    ].values[0, ...]

    surface_temp_matrix_kelvins = forecast_table_xarray[
        model_utils.TEMPERATURE_2METRES_KELVINS_KEY
    ].values[0, ...]

    # Current array shape is pressure x lat x long.
    # xcape requires lat x long x pressure.
    temp_matrix_kelvins = numpy.swapaxes(temp_matrix_kelvins, 0, 1)
    temp_matrix_kelvins = numpy.swapaxes(temp_matrix_kelvins, 1, 2)
    spec_humidity_matrix_kg_kg01 = numpy.swapaxes(
        spec_humidity_matrix_kg_kg01, 0, 1
    )
    spec_humidity_matrix_kg_kg01 = numpy.swapaxes(
        spec_humidity_matrix_kg_kg01, 1, 2
    )

    # Convert specific humidity to dewpoint.
    pressure_matrix_pascals = _get_model_pressure_matrix(
        forecast_table_xarray=forecast_table_xarray,
        vertical_axis_first=False
    )
    dewpoint_matrix_kelvins = moisture_conv.specific_humidity_to_dewpoint(
        specific_humidities_kg_kg01=spec_humidity_matrix_kg_kg01,
        temperatures_kelvins=temp_matrix_kelvins,
        total_pressures_pascals=pressure_matrix_pascals
    )

    # Estimate surface pressure and dewpoint, if necessary.
    if surface_pressure_matrix_pascals is None:
        surface_pressure_matrix_pascals = _estimate_surface_pressure(
            forecast_table_xarray=forecast_table_xarray,
            surface_geopotential_matrix_m2_s02=
            surface_geopotential_matrix_m2_s02,
            do_multiprocessing=do_multiprocessing,
            use_spline=True
        )

    if surface_dewpoint_matrix_kelvins is None:
        surface_dewpoint_matrix_kelvins = _estimate_surface_dewpoint(
            forecast_table_xarray=forecast_table_xarray,
            surface_pressure_matrix_pascals=surface_pressure_matrix_pascals,
            do_multiprocessing=do_multiprocessing,
            use_spline=True
        )

    # Compute CAPE and CIN.
    exec_start_time_unix_sec = time.time()

    if do_multiprocessing:
        start_rows, end_rows = _get_slices_for_multiprocessing(
            num_grid_rows=surface_temp_matrix_kelvins.shape[0]
        )

        argument_list = []
        keyword_argument_list = []

        for s, e in zip(start_rows, end_rows):
            argument_list.append((
                PASCALS_TO_HPA * pressure_levels_pascals,
                temperature_conv.kelvins_to_celsius(
                    temp_matrix_kelvins[s:e, ...]
                ),
                temperature_conv.kelvins_to_celsius(
                    dewpoint_matrix_kelvins[s:e, ...]
                ),
                PASCALS_TO_HPA * surface_pressure_matrix_pascals[s:e, ...],
                temperature_conv.kelvins_to_celsius(
                    surface_temp_matrix_kelvins[s:e, ...]
                ),
                temperature_conv.kelvins_to_celsius(
                    surface_dewpoint_matrix_kelvins[s:e, ...]
                )
            ))

            keyword_argument_list.append({
                'source': parcel_source_string,
                'ml_depth': (
                    500. if mixed_layer_depth_metres is None
                    else mixed_layer_depth_metres
                ),
                'adiabat': 'pseudo-liquid',
                'vertical_lev': 'pressure'
            })

        cape_matrix_j_kg01 = numpy.full(
            pressure_matrix_pascals.shape[:-1], numpy.nan
        )
        cin_matrix_j_kg01 = numpy.full(
            pressure_matrix_pascals.shape[:-1], numpy.nan
        )

        with Pool() as pool_object:
            if parcel_source_string == MOST_UNSTABLE_PARCEL_SOURCE_STRING:
                cape_submatrices, cin_submatrices, _, _ = zip(
                    *__starmap_with_kwargs(
                        pool_object, core.calc_cape,
                        argument_list, keyword_argument_list
                    )
                )
            else:
                cape_submatrices, cin_submatrices = zip(*__starmap_with_kwargs(
                    pool_object, core.calc_cape,
                    argument_list, keyword_argument_list
                ))

            for k in range(len(start_rows)):
                s = start_rows[k]
                e = end_rows[k]
                cape_matrix_j_kg01[s:e, :] = cape_submatrices[k]
                cin_matrix_j_kg01[s:e, :] = cin_submatrices[k]
    else:
        cape_matrix_j_kg01, cin_matrix_j_kg01 = core.calc_cape(
            PASCALS_TO_HPA * pressure_levels_pascals,
            temperature_conv.kelvins_to_celsius(temp_matrix_kelvins),
            temperature_conv.kelvins_to_celsius(dewpoint_matrix_kelvins),
            PASCALS_TO_HPA * surface_pressure_matrix_pascals,
            temperature_conv.kelvins_to_celsius(surface_temp_matrix_kelvins),
            temperature_conv.kelvins_to_celsius(
                surface_dewpoint_matrix_kelvins
            ),
            source=parcel_source_string,
            ml_depth=(
                500. if mixed_layer_depth_metres is None
                else mixed_layer_depth_metres
            ),
            adiabat='pseudo-liquid',
            vertical_lev='pressure'
        )[:2]

    assert not numpy.any(numpy.isnan(cape_matrix_j_kg01))
    assert not numpy.any(numpy.isnan(cin_matrix_j_kg01))

    print((
        'Took xcape {0:.1f} seconds to compute CAPE and CIN for {1:d} grid '
        'points.'
    ).format(
        time.time() - exec_start_time_unix_sec,
        surface_pressure_matrix_pascals.size
    ))

    return (
        cape_matrix_j_kg01, cin_matrix_j_kg01,
        surface_pressure_matrix_pascals, surface_dewpoint_matrix_kelvins
    )


def get_lifted_index(
        forecast_table_xarray, do_multiprocessing, final_pressure_pascals,
        surface_geopotential_matrix_m2_s02=None,
        surface_pressure_matrix_pascals=None):
    """Computes lifted index at every grid point.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param do_multiprocessing: Same.
    :param final_pressure_pascals: Final pressure.  Lifted index will be
        computed, at every horizontal grid point, for a parcel lifted from the
        surface to this pressure.
    :param surface_geopotential_matrix_m2_s02: See documentation for
        `get_cape_and_cin`.
    :param surface_pressure_matrix_pascals: Same.
    :return: lifted_index_matrix_kelvins: M-by-N numpy array of lifted-index
        values.
    :return: surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.
    """

    # TODO(thunderhoser): Still need to get fancy with time dimension.

    # Check input args.
    if surface_geopotential_matrix_m2_s02 is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_geopotential_matrix_m2_s02
        )
    if surface_pressure_matrix_pascals is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_pressure_matrix_pascals
        )

    error_checking.assert_is_boolean(do_multiprocessing)

    p_index = __pressure_level_to_index(
        forecast_table_xarray=forecast_table_xarray,
        desired_pressure_pascals=final_pressure_pascals
    )

    # Estimate surface pressure, if necessary.
    if surface_pressure_matrix_pascals is None:
        surface_pressure_matrix_pascals = _estimate_surface_pressure(
            forecast_table_xarray=forecast_table_xarray,
            surface_geopotential_matrix_m2_s02=
            surface_geopotential_matrix_m2_s02,
            do_multiprocessing=do_multiprocessing,
            use_spline=True
        )

    # Compute temperature of lifted parcel at each horizontal grid point.
    surface_temp_matrix_kelvins = forecast_table_xarray[
        model_utils.TEMPERATURE_2METRES_KELVINS_KEY
    ].values[0, ...]

    final_pressure_matrix_pascals = numpy.full(
        surface_temp_matrix_kelvins.shape, final_pressure_pascals
    )

    do_multiprocessing_for_sharppy = False
    exec_start_time_unix_sec = time.time()

    if do_multiprocessing_for_sharppy:
        start_rows, end_rows = _get_slices_for_multiprocessing(
            num_grid_rows=surface_temp_matrix_kelvins.shape[0]
        )

        argument_list = []

        for s, e in zip(start_rows, end_rows):
            argument_list.append((
                PASCALS_TO_HPA * surface_pressure_matrix_pascals[s:e, ...],
                temperature_conv.kelvins_to_celsius(
                    surface_temp_matrix_kelvins[s:e, ...]
                ),
                PASCALS_TO_HPA * final_pressure_matrix_pascals[s:e, ...]
            ))

        lifted_temp_matrix_celsius = numpy.full(
            surface_temp_matrix_kelvins.shape, numpy.nan
        )

        with Pool() as pool_object:
            submatrices = pool_object.starmap(
                sharppy_thermo.wetlift, argument_list
            )

            for k in range(len(start_rows)):
                s = start_rows[k]
                e = end_rows[k]
                lifted_temp_matrix_celsius[s:e, :] = submatrices[k]
    else:
        lifted_temp_matrix_celsius = sharppy_thermo.wetlift(
            p=PASCALS_TO_HPA * surface_pressure_matrix_pascals,
            t=temperature_conv.kelvins_to_celsius(surface_temp_matrix_kelvins),
            p2=PASCALS_TO_HPA * final_pressure_matrix_pascals
        )

    assert not numpy.any(numpy.isnan(lifted_temp_matrix_celsius))

    print((
        'Took SHARPpy {0:.1f} seconds to lift parcels for {1:d} grid points.'
    ).format(
        time.time() - exec_start_time_unix_sec,
        surface_pressure_matrix_pascals.size
    ))

    lifted_temp_matrix_kelvins = temperature_conv.celsius_to_kelvins(
        lifted_temp_matrix_celsius
    )

    # Convert lifted temperature to lifted virtual temperature, assuming 100%
    # saturation.
    lifted_vapour_pressure_matrix_pascals = (
        moisture_conv.dewpoint_to_vapour_pressure(
            dewpoints_kelvins=lifted_temp_matrix_kelvins,
            temperatures_kelvins=lifted_temp_matrix_kelvins,
            total_pressures_pascals=final_pressure_matrix_pascals
        )
    )

    lifted_virtual_temp_matrix_kelvins = (
        moisture_conv.temperature_to_virtual_temperature(
            temperatures_kelvins=lifted_temp_matrix_kelvins,
            total_pressures_pascals=final_pressure_matrix_pascals,
            vapour_pressures_pascals=lifted_vapour_pressure_matrix_pascals
        )
    )

    # Mask out horizontal grid points where the parcel is "lifted" to a level
    # below the surface.
    bad_layer_flag_matrix = (
        final_pressure_matrix_pascals >= surface_pressure_matrix_pascals
    )
    lifted_virtual_temp_matrix_kelvins[bad_layer_flag_matrix] = numpy.nan

    # Determine actual (not lifted) virtual temperature at final pressure level.
    unlifted_spec_humidity_matrix_kg_kg01 = (
        forecast_table_xarray[model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY].values[
            0, p_index, ...
        ]
    )

    unlifted_mixing_ratio_matrix_kg_kg01 = (
        moisture_conv.specific_humidity_to_mixing_ratio(
            unlifted_spec_humidity_matrix_kg_kg01
        )
    )

    unlifted_vapour_pressure_matrix_pascals = (
        moisture_conv.mixing_ratio_to_vapour_pressure(
            mixing_ratios_kg_kg01=unlifted_mixing_ratio_matrix_kg_kg01,
            total_pressures_pascals=final_pressure_matrix_pascals
        )
    )

    unlifted_temp_matrix_kelvins = forecast_table_xarray[
        model_utils.TEMPERATURE_KELVINS_KEY
    ].values[0, p_index, ...]

    unlifted_virtual_temp_matrix_kelvins = (
        moisture_conv.temperature_to_virtual_temperature(
            temperatures_kelvins=unlifted_temp_matrix_kelvins,
            total_pressures_pascals=final_pressure_matrix_pascals,
            vapour_pressures_pascals=unlifted_vapour_pressure_matrix_pascals
        )
    )

    # Lifted index = actual minus lifted virtual temp at final pressure level.
    lifted_index_matrix_kelvins = (
        unlifted_virtual_temp_matrix_kelvins -
        lifted_virtual_temp_matrix_kelvins
    )
    return lifted_index_matrix_kelvins, surface_pressure_matrix_pascals


def get_precipitable_water(
        forecast_table_xarray, do_multiprocessing, top_pressure_pascals,
        surface_geopotential_matrix_m2_s02=None,
        surface_pressure_matrix_pascals=None,
        surface_dewpoint_matrix_kelvins=None):
    """Computes precipitable water at every grid point.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param do_multiprocessing: Same.
    :param top_pressure_pascals: Top pressure.  Precipitable water will be
        computed, at every horizontal grid point, by integrating from the
        surface to this pressure.
    :param surface_geopotential_matrix_m2_s02: See documentation for
        `get_cape_and_cin`.
    :param surface_pressure_matrix_pascals: Same.
    :param surface_dewpoint_matrix_kelvins: Same.
    :return: precipitable_water_matrix_kg_m02: M-by-N numpy array of
        precipitable-water values.  The units are kg m^-2 or, equivalently, mm
        of accumulation.
    :return: surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.
    :return: surface_dewpoint_matrix_kelvins: M-by-N numpy array of surface
        dewpoints.
    """

    # One of the AIWP models (I can't remember which) already has precipitable
    # water as an output variable, but I want to allow for options, specifically
    # the layer top via `top_pressure_pascals`.

    # Check input args.
    if surface_geopotential_matrix_m2_s02 is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_geopotential_matrix_m2_s02
        )
    if surface_pressure_matrix_pascals is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_pressure_matrix_pascals
        )
    if surface_dewpoint_matrix_kelvins is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_dewpoint_matrix_kelvins
        )

    top_p_index = 1 + __pressure_level_to_index(
        forecast_table_xarray=forecast_table_xarray,
        desired_pressure_pascals=top_pressure_pascals
    )

    # Estimate surface pressure and dewpoint, if necessary.
    if surface_pressure_matrix_pascals is None:
        surface_pressure_matrix_pascals = _estimate_surface_pressure(
            forecast_table_xarray=forecast_table_xarray,
            surface_geopotential_matrix_m2_s02=
            surface_geopotential_matrix_m2_s02,
            do_multiprocessing=do_multiprocessing,
            use_spline=True
        )

    if surface_dewpoint_matrix_kelvins is None:
        surface_dewpoint_matrix_kelvins = _estimate_surface_dewpoint(
            forecast_table_xarray=forecast_table_xarray,
            surface_pressure_matrix_pascals=surface_pressure_matrix_pascals,
            do_multiprocessing=do_multiprocessing,
            use_spline=True
        )

    # Convert surface dewpoint to surface specific humidity.
    surface_temp_matrix_kelvins = forecast_table_xarray[
        model_utils.TEMPERATURE_2METRES_KELVINS_KEY
    ].values[0, ...]

    surface_spec_humidity_matrix_kg_kg01 = (
        moisture_conv.dewpoint_to_specific_humidity(
            dewpoints_kelvins=surface_dewpoint_matrix_kelvins,
            temperatures_kelvins=surface_temp_matrix_kelvins,
            total_pressures_pascals=surface_pressure_matrix_pascals
        )
    )

    # Extract pressure and specific humidity at levels above the surface.
    pressure_matrix_pascals = _get_model_pressure_matrix(
        forecast_table_xarray=forecast_table_xarray,
        vertical_axis_first=True
    )
    pressure_matrix_pascals = pressure_matrix_pascals[:top_p_index, ...]

    spec_humidity_matrix_kg_kg01 = 0. + forecast_table_xarray[
        model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY
    ].values[0, :top_p_index, ...]

    # Mask out specific humidity below the surface.
    surface_pressure_matrix_3d_pascals = numpy.expand_dims(
        surface_pressure_matrix_pascals, axis=0
    )
    spec_humidity_matrix_kg_kg01[
        pressure_matrix_pascals >= surface_pressure_matrix_3d_pascals
    ] = numpy.nan

    # Current array shape is V x M x N, where M = number of rows and N = number
    # of columns and V = number of pressure levels.
    # We want (V + 1) x M x N, where the extra vertical level is the surface.
    pressure_matrix_pascals = numpy.concatenate([
        pressure_matrix_pascals,
        numpy.expand_dims(surface_pressure_matrix_pascals, axis=0)
    ], axis=0)

    spec_humidity_matrix_kg_kg01 = numpy.concatenate([
        spec_humidity_matrix_kg_kg01,
        numpy.expand_dims(surface_spec_humidity_matrix_kg_kg01, axis=0)
    ], axis=0)

    # Compute precipitable water.
    exec_start_time_unix_sec = time.time()

    if do_multiprocessing:
        start_rows, end_rows = _get_slices_for_multiprocessing(
            num_grid_rows=pressure_matrix_pascals.shape[1]
        )

        argument_list = []

        for s, e in zip(start_rows, end_rows):
            argument_list.append((
                pressure_matrix_pascals[:, s:e, :],
                spec_humidity_matrix_kg_kg01[:, s:e, :]
            ))

        precipitable_water_matrix_kg_m02 = numpy.full(
            pressure_matrix_pascals.shape[1:], numpy.nan
        )

        with Pool() as pool_object:
            submatrices = pool_object.starmap(
                _integrate_to_precipitable_water, argument_list
            )

            for k in range(len(start_rows)):
                s = start_rows[k]
                e = end_rows[k]
                precipitable_water_matrix_kg_m02[s:e, :] = submatrices[k]
    else:
        precipitable_water_matrix_kg_m02 = _integrate_to_precipitable_water(
            pressure_matrix_pascals=pressure_matrix_pascals,
            spec_humidity_matrix_kg_kg01=spec_humidity_matrix_kg_kg01
        )

    print(numpy.mean(precipitable_water_matrix_kg_m02 < 0.))

    row_indices, column_indices = numpy.where(precipitable_water_matrix_kg_m02 < 0.)
    row_index = row_indices[0]
    column_index = column_indices[0]

    print(pressure_matrix_pascals[:, row_index, column_index])
    print(spec_humidity_matrix_kg_kg01[:, row_index, column_index])

    inds = numpy.argsort(-pressure_matrix_pascals[:, row_index, column_index])
    subinds = numpy.where(numpy.invert(
        numpy.isnan(spec_humidity_matrix_kg_kg01[:, row_index, column_index][inds])
    ))[0]

    print(pressure_matrix_pascals[:, row_index, column_index][inds][subinds])
    print(spec_humidity_matrix_kg_kg01[:, row_index, column_index][inds][subinds])
    print(precipitable_water_matrix_kg_m02[row_index, column_index])

    this_coeff = -METRES_TO_MM / (WATER_DENSITY_KG_M03 * GRAVITY_M_S02)
    this_pwat = this_coeff * simpson(
        y=spec_humidity_matrix_kg_kg01[:, row_index, column_index][inds][subinds],
        x=pressure_matrix_pascals[:, row_index, column_index][inds][subinds],
        axis=0,
        even='simpson'
    )
    print(this_pwat)

    this_coeff = -METRES_TO_MM / (WATER_DENSITY_KG_M03 * GRAVITY_M_S02)
    this_pwat = this_coeff * simpson(
        y=spec_humidity_matrix_kg_kg01[:, row_index, column_index][inds][subinds],
        x=pressure_matrix_pascals[:, row_index, column_index][inds][subinds],
        axis=0,
        even='avg'
    )
    print(this_pwat)

    assert not numpy.any(precipitable_water_matrix_kg_m02 < 0.)

    print('Estimating precipitable water took {0:.1f} seconds.'.format(
        time.time() - exec_start_time_unix_sec
    ))

    return (
        precipitable_water_matrix_kg_m02,
        surface_pressure_matrix_pascals,
        surface_dewpoint_matrix_kelvins
    )


def get_wind_shear(
        forecast_table_xarray, do_multiprocessing,
        bottom_pressure_pascals, top_pressure_pascals,
        surface_geopotential_matrix_m2_s02=None,
        surface_pressure_matrix_pascals=None):
    """Computes vertical wind shear between two levels.

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param do_multiprocessing: Same.
    :param bottom_pressure_pascals: Pressure at bottom of layer.  This can also
        be "surface" (a string).
    :param top_pressure_pascals: Pressure at top of layer.
    :param surface_geopotential_matrix_m2_s02: See documentation for
        `get_cape_and_cin`.
    :param surface_pressure_matrix_pascals: Same.
    :return: zonal_wind_shear_matrix_m_s01: M-by-N numpy array of zonal wind
        shears (metres per second).
    :return: merid_wind_shear_matrix_m_s01: M-by-N numpy array of meridional
        wind shears (metres per second).
    :return: surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.
    """

    # Check input args.
    if surface_geopotential_matrix_m2_s02 is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_geopotential_matrix_m2_s02
        )
    if surface_pressure_matrix_pascals is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_pressure_matrix_pascals
        )

    error_checking.assert_is_boolean(do_multiprocessing)

    top_index = __pressure_level_to_index(
        forecast_table_xarray=forecast_table_xarray,
        desired_pressure_pascals=top_pressure_pascals
    )

    is_bottom_surface = False

    if isinstance(bottom_pressure_pascals, str):
        assert bottom_pressure_pascals == 'surface'
        bottom_index = None
        is_bottom_surface = True
    else:
        bottom_index = __pressure_level_to_index(
            forecast_table_xarray=forecast_table_xarray,
            desired_pressure_pascals=bottom_pressure_pascals
        )
        assert top_index > bottom_index

    # Estimate surface pressure, if necessary.
    if surface_pressure_matrix_pascals is None and is_bottom_surface:
        surface_pressure_matrix_pascals = _estimate_surface_pressure(
            forecast_table_xarray=forecast_table_xarray,
            surface_geopotential_matrix_m2_s02=
            surface_geopotential_matrix_m2_s02,
            do_multiprocessing=do_multiprocessing,
            use_spline=True
        )

    # Do actual stuff.
    if is_bottom_surface:
        bottom_zonal_wind_matrix_m_s01 = 0. + forecast_table_xarray[
            model_utils.ZONAL_WIND_10METRES_M_S01_KEY
        ].values[0, ...]

        bottom_merid_wind_matrix_m_s01 = 0. + forecast_table_xarray[
            model_utils.MERIDIONAL_WIND_10METRES_M_S01_KEY
        ].values[0, ...]

        # Mask out locations where layer top is below surface.
        pressure_matrix_pascals = _get_model_pressure_matrix(
            forecast_table_xarray=forecast_table_xarray,
            vertical_axis_first=True
        )
        top_pressure_matrix_pascals = pressure_matrix_pascals[top_index, ...]

        bad_layer_flag_matrix = (
            top_pressure_matrix_pascals >= surface_pressure_matrix_pascals
        )
        bottom_zonal_wind_matrix_m_s01[bad_layer_flag_matrix] = numpy.nan
        bottom_merid_wind_matrix_m_s01[bad_layer_flag_matrix] = numpy.nan
    else:
        bottom_zonal_wind_matrix_m_s01 = forecast_table_xarray[
            model_utils.ZONAL_WIND_M_S01_KEY
        ].values[0, bottom_index, ...]

        bottom_merid_wind_matrix_m_s01 = forecast_table_xarray[
            model_utils.MERIDIONAL_WIND_M_S01_KEY
        ].values[0, bottom_index, ...]

    top_zonal_wind_matrix_m_s01 = forecast_table_xarray[
        model_utils.ZONAL_WIND_M_S01_KEY
    ].values[0, top_index, ...]

    top_merid_wind_matrix_m_s01 = forecast_table_xarray[
        model_utils.MERIDIONAL_WIND_M_S01_KEY
    ].values[0, top_index, ...]

    return (
        top_zonal_wind_matrix_m_s01 - bottom_zonal_wind_matrix_m_s01,
        top_merid_wind_matrix_m_s01 - bottom_merid_wind_matrix_m_s01,
        surface_pressure_matrix_pascals
    )


def get_bunkers_storm_motion(
        forecast_table_xarray, do_multiprocessing,
        surface_geopotential_matrix_m2_s02,
        surface_pressure_matrix_pascals=None):
    """Computes Bunkers right-mover storm motion at every horizontal grid point.

    Emulating non_parcel_bunkers_motion from winds.py in the SHARPpy library,
    except they use non-pressure-weighted mean wind from 0-6 km AGL, while I use
    pressure-weighted.

    https://github.com/sharppy/SHARPpy/blob/
    05bb6f3b415f4c52046179dd080485709603a535/sharppy/sharptab/winds.py#L247

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param do_multiprocessing: Same.
    :param surface_geopotential_matrix_m2_s02: Same.
    :param surface_pressure_matrix_pascals: Same.
    :return: zonal_storm_motion_matrix_m_s01: M-by-N numpy array with zonal
        (eastward) component of storm motion (metres per second).
    :return: meridional_storm_motion_matrix_m_s01: M-by-N numpy array with
        meridional (northward) component of storm motion (metres per second).
    :return: surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.
    """

    # Check input args.
    if surface_geopotential_matrix_m2_s02 is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_geopotential_matrix_m2_s02
        )
    if surface_pressure_matrix_pascals is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_pressure_matrix_pascals
        )

    error_checking.assert_is_boolean(do_multiprocessing)

    # Estimate surface pressure and dewpoint, if necessary.
    if surface_pressure_matrix_pascals is None:
        surface_pressure_matrix_pascals = _estimate_surface_pressure(
            forecast_table_xarray=forecast_table_xarray,
            surface_geopotential_matrix_m2_s02=
            surface_geopotential_matrix_m2_s02,
            do_multiprocessing=do_multiprocessing,
            use_spline=True
        )

    # At each horizontal grid point, find pressure level corresponding to 6 km
    # above ground level.
    geopotential_matrix_m2_s02 = forecast_table_xarray[
        model_utils.GEOPOTENTIAL_M2_S02_KEY
    ].values[0, ...]

    top_index_matrix = _height_agl_to_nearest_pressure_level(
        geopotential_matrix_m2_s02=geopotential_matrix_m2_s02,
        surface_geopotential_matrix_m2_s02=surface_geopotential_matrix_m2_s02,
        desired_height_m_agl=6000.,
        find_nearest_level_beneath=False
    )

    # At each horizontal grid point, get 0--6-km mean wind.  The first three
    # input matrices to _get_mean_wind -- whose names do not start with
    # "surface" -- have dimensions (V + 1) x M x N, where V is the number of
    # vertical model levels and M is the number of rows and N is the number of
    # columns.  On the first axis, index 0 is the surface and remaining indices
    # are pressure levels.
    ftx = forecast_table_xarray

    zonal_wind_matrix_m_s01 = numpy.concatenate([
        numpy.expand_dims(
            ftx[model_utils.ZONAL_WIND_10METRES_M_S01_KEY].values[0, ...],
            axis=0
        ),
        ftx[model_utils.ZONAL_WIND_M_S01_KEY].values[0, ...]
    ], axis=0)

    meridional_wind_matrix_m_s01 = numpy.concatenate([
        numpy.expand_dims(
            ftx[model_utils.MERIDIONAL_WIND_10METRES_M_S01_KEY].values[0, ...],
            axis=0
        ),
        ftx[model_utils.MERIDIONAL_WIND_M_S01_KEY].values[0, ...]
    ], axis=0)

    pressure_matrix_pascals = _get_model_pressure_matrix(
        forecast_table_xarray=forecast_table_xarray,
        vertical_axis_first=True
    )
    pressure_matrix_pascals = numpy.concatenate([
        numpy.expand_dims(surface_pressure_matrix_pascals, axis=0),
        pressure_matrix_pascals
    ], axis=0)

    top_index_matrix += 1

    (
        mean_zonal_wind_matrix_m_s01, mean_meridional_wind_matrix_m_s01
    ) = _get_mean_wind(
        zonal_wind_matrix_m_s01=zonal_wind_matrix_m_s01,
        meridional_wind_matrix_m_s01=meridional_wind_matrix_m_s01,
        bottom_index_matrix=numpy.full(
            surface_pressure_matrix_pascals.shape, 0, dtype=int
        ),
        top_index_matrix=top_index_matrix,
        pressure_weighted=True,
        pressure_matrix_pascals=pressure_matrix_pascals,
        surface_pressure_matrix_pascals=surface_pressure_matrix_pascals
    )

    # At each horizontal grid point, get 0--6-km wind shear.  Here, we do NOT
    # need to worry about masking grid points below the surface, because we
    # have already done this for 0--6-km mean wind.  Thus, at any horizontal
    # grid point where 6 km is below the surface (Everest?), the two mean-wind
    # matrices have NaN.
    top_zonal_wind_matrix_m_s01 = zonal_wind_matrix_m_s01[
        top_index_matrix,
        numpy.arange(top_index_matrix.shape[0])[:, None],
        numpy.arange(top_index_matrix.shape[1])
    ]
    zonal_shear_matrix_m_s01 = (
        top_zonal_wind_matrix_m_s01 - zonal_wind_matrix_m_s01[0, ...]
    )

    top_meridional_wind_matrix_m_s01 = meridional_wind_matrix_m_s01[
        top_index_matrix,
        numpy.arange(top_index_matrix.shape[0])[:, None],
        numpy.arange(top_index_matrix.shape[1])
    ]
    meridional_shear_matrix_m_s01 = (
        top_meridional_wind_matrix_m_s01 - meridional_wind_matrix_m_s01[0, ...]
    )

    # Now apply the calculation for Bunkers right-mover storm motion.
    scalar_shear_matrix_m_s01 = numpy.sqrt(
        zonal_shear_matrix_m_s01 ** 2 + meridional_shear_matrix_m_s01 ** 2
    )
    multiplier_matrix = 7.5 / scalar_shear_matrix_m_s01

    zonal_storm_motion_matrix_m_s01 = (
        mean_zonal_wind_matrix_m_s01 +
        (multiplier_matrix * meridional_shear_matrix_m_s01)
    )
    meridional_storm_motion_matrix_m_s01 = (
        mean_meridional_wind_matrix_m_s01 -
        (multiplier_matrix * zonal_shear_matrix_m_s01)
    )

    return (
        zonal_storm_motion_matrix_m_s01,
        meridional_storm_motion_matrix_m_s01,
        surface_pressure_matrix_pascals
    )


def get_storm_relative_helicity(
        forecast_table_xarray, do_multiprocessing, top_heights_m_agl,
        surface_geopotential_matrix_m2_s02,
        surface_pressure_matrix_pascals=None):
    """At each horizontal grid point, converts SRH from sfc to a given height.

    Emulating helicity() from winds.py in the SHARPpy library.

    https://github.com/sharppy/SHARPpy/blob/
    05bb6f3b415f4c52046179dd080485709603a535/sharppy/sharptab/winds.py#L289

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid
    L = number of layers for which to compute helicity

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param do_multiprocessing: Same.
    :param top_heights_m_agl: length-L numpy array of top heights (metres above
        ground level).  Will convert SRH from surface up to each height.
    :param surface_geopotential_matrix_m2_s02: See documentation for
        `get_cape_and_cin`.
    :param surface_pressure_matrix_pascals: Same.
    :return: positive_helicity_matrix_m2_s02: L-by-M-by-N numpy array with
        integrated positive helicity at each horizontal grid point.
    :return: negative_helicity_matrix_m2_s02: L-by-M-by-N numpy array with
        integrated negative helicity at each horizontal grid point.
    :return: zonal_storm_motion_matrix_m_s01: M-by-N numpy array with eastward
        storm motion (metres per second) at each horizontal grid point.
    :return: meridional_storm_motion_matrix_m_s01: M-by-N numpy array with
        northward storm motion (metres per second) at each horizontal grid
        point.
    :return: surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.
    """

    # Check input args.
    if surface_geopotential_matrix_m2_s02 is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_geopotential_matrix_m2_s02
        )
    if surface_pressure_matrix_pascals is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_pressure_matrix_pascals
        )

    error_checking.assert_is_boolean(do_multiprocessing)
    error_checking.assert_is_numpy_array(top_heights_m_agl, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(top_heights_m_agl, 0.)

    # Get rid of non-unique top heights.
    top_heights_m_agl = numpy.round(top_heights_m_agl).astype(int)
    top_heights_m_agl = numpy.unique(top_heights_m_agl)

    # At every horizontal grid point, compute Bunkers right-mover storm motion.
    (
        zonal_storm_motion_matrix_m_s01,
        meridional_storm_motion_matrix_m_s01,
        surface_pressure_matrix_pascals
    ) = get_bunkers_storm_motion(
        forecast_table_xarray=forecast_table_xarray,
        do_multiprocessing=do_multiprocessing,
        surface_geopotential_matrix_m2_s02=surface_geopotential_matrix_m2_s02,
        surface_pressure_matrix_pascals=surface_pressure_matrix_pascals
    )

    # Create three matrices with dimensions (V + 1) x M x N, where V = number
    # of vertical model levels; M = num rows; N = num columns.  On the first
    # axis, index 0 is the surface and remaining indices are pressure levels.
    ftx = forecast_table_xarray

    zonal_wind_matrix_m_s01 = numpy.concatenate([
        numpy.expand_dims(
            ftx[model_utils.ZONAL_WIND_10METRES_M_S01_KEY].values[0, ...],
            axis=0
        ),
        ftx[model_utils.ZONAL_WIND_M_S01_KEY].values[0, ...]
    ], axis=0)

    meridional_wind_matrix_m_s01 = numpy.concatenate([
        numpy.expand_dims(
            ftx[model_utils.MERIDIONAL_WIND_10METRES_M_S01_KEY].values[0, ...],
            axis=0
        ),
        ftx[model_utils.MERIDIONAL_WIND_M_S01_KEY].values[0, ...]
    ], axis=0)

    geopotential_matrix_m2_s02 = numpy.concatenate([
        numpy.expand_dims(surface_geopotential_matrix_m2_s02, axis=0),
        ftx[model_utils.GEOPOTENTIAL_M2_S02_KEY].values[0, ...]
    ], axis=0)

    # Determine target heights, to which wind will be interpolated.
    num_interp_heights = int(numpy.ceil(
        float(numpy.max(top_heights_m_agl)) / HEIGHT_SPACING_FOR_HELICITY_METRES
    ))
    interp_heights_m_agl = numpy.linspace(
        0, numpy.max(top_heights_m_agl), num=num_interp_heights + 1, dtype=int
    )[1:]

    for this_top_height_m_agl in top_heights_m_agl:
        if this_top_height_m_agl in interp_heights_m_agl:
            continue

        interp_heights_m_agl = numpy.concatenate([
            interp_heights_m_agl,
            numpy.array([this_top_height_m_agl], dtype=int)
        ])

    interp_heights_m_agl = numpy.unique(interp_heights_m_agl).astype(float)
    num_interp_heights = len(interp_heights_m_agl)

    # At every horizontal grid point, interpolate the wind to all target
    # heights.
    exec_start_time_unix_sec = time.time()

    if do_multiprocessing:
        start_rows, end_rows = _get_slices_for_multiprocessing(
            num_grid_rows=geopotential_matrix_m2_s02.shape[1]
        )

        argument_list = []
        for s, e in zip(start_rows, end_rows):
            argument_list.append((
                zonal_wind_matrix_m_s01[:, s:e, ...],
                geopotential_matrix_m2_s02[:, s:e, ...],
                surface_geopotential_matrix_m2_s02[s:e, ...],
                interp_heights_m_agl,
                True
            ))

        interp_zonal_wind_matrix_m_s01 = numpy.full(
            (num_interp_heights,) + surface_geopotential_matrix_m2_s02.shape,
            numpy.nan
        )

        with Pool() as pool_object:
            submatrices = pool_object.starmap(
                _interp_wind_to_heights_agl, argument_list
            )

            for k in range(len(start_rows)):
                s = start_rows[k]
                e = end_rows[k]
                interp_zonal_wind_matrix_m_s01[:, s:e, :] = submatrices[k]

        argument_list = []
        for s, e in zip(start_rows, end_rows):
            argument_list.append((
                meridional_wind_matrix_m_s01[:, s:e, ...],
                geopotential_matrix_m2_s02[:, s:e, ...],
                surface_geopotential_matrix_m2_s02[s:e, ...],
                interp_heights_m_agl,
                True
            ))

        interp_meridional_wind_matrix_m_s01 = numpy.full(
            (num_interp_heights,) + surface_geopotential_matrix_m2_s02.shape,
            numpy.nan
        )

        with Pool() as pool_object:
            submatrices = pool_object.starmap(
                _interp_wind_to_heights_agl, argument_list
            )

            for k in range(len(start_rows)):
                s = start_rows[k]
                e = end_rows[k]
                interp_meridional_wind_matrix_m_s01[:, s:e, :] = submatrices[k]
    else:
        interp_zonal_wind_matrix_m_s01 = _interp_wind_to_heights_agl(
            wind_matrix_m_s01=zonal_wind_matrix_m_s01,
            geopotential_matrix_m2_s02=geopotential_matrix_m2_s02,
            surface_geopotential_matrix_m2_s02=
            surface_geopotential_matrix_m2_s02,
            target_heights_m_agl=interp_heights_m_agl,
            use_spline=True
        )

        interp_meridional_wind_matrix_m_s01 = _interp_wind_to_heights_agl(
            wind_matrix_m_s01=meridional_wind_matrix_m_s01,
            geopotential_matrix_m2_s02=geopotential_matrix_m2_s02,
            surface_geopotential_matrix_m2_s02=
            surface_geopotential_matrix_m2_s02,
            target_heights_m_agl=interp_heights_m_agl,
            use_spline=True
        )

    assert not numpy.any(numpy.isnan(interp_zonal_wind_matrix_m_s01))
    assert not numpy.any(numpy.isnan(interp_meridional_wind_matrix_m_s01))

    print('Interpolating wind to {0:s} m AGL took {1:.1f} seconds.'.format(
        str(interp_heights_m_agl),
        time.time() - exec_start_time_unix_sec
    ))

    # Add surface wind to both matrices.
    interp_zonal_wind_matrix_m_s01 = numpy.concatenate([
        zonal_wind_matrix_m_s01[[0], ...],
        interp_zonal_wind_matrix_m_s01
    ], axis=0)

    interp_meridional_wind_matrix_m_s01 = numpy.concatenate([
        meridional_wind_matrix_m_s01[[0], ...],
        interp_meridional_wind_matrix_m_s01
    ], axis=0)

    interp_heights_m_agl = numpy.concatenate([
        numpy.array([0.]),
        interp_heights_m_agl
    ])

    # Now apply the calculation for storm-relative helicity.
    sr_interp_zonal_wind_matrix_m_s01 = (
        interp_zonal_wind_matrix_m_s01 -
        numpy.expand_dims(zonal_storm_motion_matrix_m_s01, axis=0)
    )
    sr_interp_meridional_wind_matrix_m_s01 = (
        interp_meridional_wind_matrix_m_s01 -
        numpy.expand_dims(meridional_storm_motion_matrix_m_s01, axis=0)
    )

    num_top_heights = len(top_heights_m_agl)
    num_grid_rows = zonal_storm_motion_matrix_m_s01.shape[0]
    num_grid_columns = zonal_storm_motion_matrix_m_s01.shape[1]

    these_dim = (num_top_heights, num_grid_rows, num_grid_columns)
    positive_helicity_matrix_m2_s02 = numpy.full(these_dim, numpy.nan)
    negative_helicity_matrix_m2_s02 = numpy.full(these_dim, numpy.nan)

    for k in range(num_top_heights):
        this_top_index = numpy.argmin(numpy.absolute(
            top_heights_m_agl[k] - interp_heights_m_agl
        ))

        layerwise_helicity_matrix_m2_s02 = (
            sr_interp_zonal_wind_matrix_m_s01[1:(this_top_index + 1), ...] *
            sr_interp_meridional_wind_matrix_m_s01[:this_top_index, ...] -

            sr_interp_zonal_wind_matrix_m_s01[:this_top_index, ...] *
            sr_interp_meridional_wind_matrix_m_s01[1:(this_top_index + 1), ...]
        )

        positive_helicity_matrix_m2_s02[k, ...] = numpy.sum(
            numpy.maximum(layerwise_helicity_matrix_m2_s02, 0.), axis=0
        )
        negative_helicity_matrix_m2_s02[k, ...] = numpy.sum(
            numpy.minimum(layerwise_helicity_matrix_m2_s02, 0.), axis=0
        )

    return (
        positive_helicity_matrix_m2_s02,
        negative_helicity_matrix_m2_s02,
        zonal_storm_motion_matrix_m_s01,
        meridional_storm_motion_matrix_m_s01,
        surface_pressure_matrix_pascals
    )


def get_pbl_height(
        forecast_table_xarray, do_multiprocessing,
        surface_geopotential_matrix_m2_s02,
        surface_pressure_matrix_pascals=None,
        surface_dewpoint_matrix_kelvins=None):
    """At each horizontal grid point, computes hgt of planetary boundary layer.

    Defined as lowest height where theta_v (virtual potential temperature) >=
    theta_v*, where theta_v* = surface theta_v + 0.5 Kelvins.

    Emulating pbl_top() from params.py in the SHARPpy library.

    https://github.com/sharppy/SHARPpy/blob/
    05bb6f3b415f4c52046179dd080485709603a535/sharppy/sharptab/params.py#L2997

    M = number of rows (latitudes) in grid
    N = number of columns (longitudes) in grid

    :param forecast_table_xarray: See documentation for `get_cape_and_cin`.
    :param do_multiprocessing: Same.
    :param surface_geopotential_matrix_m2_s02: Same.
    :param surface_pressure_matrix_pascals: Same.
    :param surface_dewpoint_matrix_kelvins: Same.
    :return: pbl_height_matrix_m_agl: M-by-N numpy array of PBL heights (metres
        above ground level).
    :return: surface_pressure_matrix_pascals: M-by-N numpy array of surface
        pressures.
    :return: surface_dewpoint_matrix_kelvins: M-by-N numpy array of surface
        dewpoints.
    """

    # Check input args.
    if surface_geopotential_matrix_m2_s02 is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_geopotential_matrix_m2_s02
        )
    if surface_pressure_matrix_pascals is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_pressure_matrix_pascals
        )
    if surface_dewpoint_matrix_kelvins is not None:
        __check_for_matching_grids(
            forecast_table_xarray=forecast_table_xarray,
            aux_data_matrix=surface_dewpoint_matrix_kelvins
        )

    error_checking.assert_is_boolean(do_multiprocessing)

    # Estimate surface pressure and dewpoint, if necessary.
    if surface_pressure_matrix_pascals is None:
        surface_pressure_matrix_pascals = _estimate_surface_pressure(
            forecast_table_xarray=forecast_table_xarray,
            surface_geopotential_matrix_m2_s02=
            surface_geopotential_matrix_m2_s02,
            do_multiprocessing=do_multiprocessing,
            use_spline=True
        )

    if surface_dewpoint_matrix_kelvins is None:
        surface_dewpoint_matrix_kelvins = _estimate_surface_dewpoint(
            forecast_table_xarray=forecast_table_xarray,
            surface_pressure_matrix_pascals=surface_pressure_matrix_pascals,
            do_multiprocessing=do_multiprocessing,
            use_spline=True
        )

    # Compute theta_v everywhere in the 3-D grid.
    surface_temp_matrix_kelvins = forecast_table_xarray[
        model_utils.TEMPERATURE_2METRES_KELVINS_KEY
    ].values[0, ...]

    surface_vapour_pressure_matrix_pascals = (
        moisture_conv.dewpoint_to_vapour_pressure(
            dewpoints_kelvins=surface_dewpoint_matrix_kelvins,
            temperatures_kelvins=surface_temp_matrix_kelvins,
            total_pressures_pascals=surface_pressure_matrix_pascals
        )
    )

    surface_virtual_temp_matrix_kelvins = (
        moisture_conv.temperature_to_virtual_temperature(
            temperatures_kelvins=surface_temp_matrix_kelvins,
            total_pressures_pascals=surface_pressure_matrix_pascals,
            vapour_pressures_pascals=surface_vapour_pressure_matrix_pascals
        )
    )

    surface_theta_v_matrix_kelvins = (
        temperature_conv.temperatures_to_potential_temperatures(
            temperatures_kelvins=surface_virtual_temp_matrix_kelvins,
            total_pressures_pascals=surface_pressure_matrix_pascals
        )
    )

    aloft_spec_humidity_matrix_kg_kg01 = forecast_table_xarray[
        model_utils.SPECIFIC_HUMIDITY_KG_KG01_KEY
    ].values[0, ...]

    aloft_mixing_ratio_matrix_kg_kg01 = (
        moisture_conv.specific_humidity_to_mixing_ratio(
            aloft_spec_humidity_matrix_kg_kg01
        )
    )

    aloft_pressure_matrix_pascals = _get_model_pressure_matrix(
        forecast_table_xarray=forecast_table_xarray,
        vertical_axis_first=True
    )
    aloft_vapour_pressure_matrix_pascals = (
        moisture_conv.mixing_ratio_to_vapour_pressure(
            mixing_ratios_kg_kg01=aloft_mixing_ratio_matrix_kg_kg01,
            total_pressures_pascals=aloft_pressure_matrix_pascals
        )
    )

    aloft_temp_matrix_kelvins = forecast_table_xarray[
        model_utils.TEMPERATURE_KELVINS_KEY
    ].values[0, ...]

    aloft_virtual_temp_matrix_kelvins = (
        moisture_conv.temperature_to_virtual_temperature(
            temperatures_kelvins=aloft_temp_matrix_kelvins,
            total_pressures_pascals=aloft_pressure_matrix_pascals,
            vapour_pressures_pascals=aloft_vapour_pressure_matrix_pascals
        )
    )

    aloft_theta_v_matrix_kelvins = (
        temperature_conv.temperatures_to_potential_temperatures(
            temperatures_kelvins=aloft_virtual_temp_matrix_kelvins,
            total_pressures_pascals=aloft_pressure_matrix_pascals
        )
    )

    full_theta_v_matrix_kelvins = numpy.concatenate([
        numpy.expand_dims(surface_theta_v_matrix_kelvins, axis=0),
        aloft_theta_v_matrix_kelvins
    ], axis=0)

    aloft_geopotential_matrix_m2_s02 = forecast_table_xarray[
        model_utils.GEOPOTENTIAL_M2_S02_KEY
    ].values[0, ...]

    full_geopotential_matrix_m2_s02 = numpy.concatenate([
        numpy.expand_dims(surface_geopotential_matrix_m2_s02, axis=0),
        aloft_geopotential_matrix_m2_s02
    ], axis=0)

    pbl_height_matrix_m_agl = _get_pbl_height(
        theta_v_matrix_kelvins=full_theta_v_matrix_kelvins,
        geopotential_matrix_m2_s02=full_geopotential_matrix_m2_s02,
        theta_v_deviation_threshold_kelvins=0.5
    )

    return (
        pbl_height_matrix_m_agl,
        surface_pressure_matrix_pascals,
        surface_dewpoint_matrix_kelvins
    )
