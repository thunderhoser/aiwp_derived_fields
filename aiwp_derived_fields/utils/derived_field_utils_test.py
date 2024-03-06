"""Unit tests for derived_field_utils.py"""

import unittest
import numpy
import xarray
from aiwp_derived_fields.utils import model_utils
from aiwp_derived_fields.utils import derived_field_utils

TOLERANCE = 1e-6

# The following constants are used to test parse_field_name and
# create_field_name.
DERIVED_FIELD_NAMES = [
    'most-unstable-cape-j-kg01',
    'most-unstable-cin-j-kg01',
    'surface-based-cape-j-kg01',
    'surface-based-cin-j-kg01',
    'mixed-layer-cape-j-kg01_ml-depth-metres=500.0000',
    'mixed-layer-cin-j-kg01_ml-depth-metres=499.9999',
    'lifted-index-kelvins_top-pressure-pascals=50000',
    'precipitable-water-kg-m02_top-pressure-pascals=5000',
    'wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=surface',
    'zonal-wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=85000',
    'meridional-wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=100000',

    'bunkers-right-mover-storm-motion-m-s01',
    'bunkers-right-mover-zonal-storm-motion-m-s01',
    'bunkers-right-mover-meridional-storm-motion-m-s01',
    'storm-relative-helicity-m2-s02_top-height-m-agl=3000.0000',
    'planetary-boundary-layer-height-m-agl',

    'mixed-layer-cape-j-kg01_ml-depth-metres=FOO',
    'wind-shear-m-s01_top-pressure-pascals=surface_bottom-pressure-pascals=surface',
    'meridional-wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=40000',

    'wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=surface',
    'zonal-wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=85000',
    'meridional-wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=100000',
    'bunkers-right-mover-storm-motion-m-s01',
    'bunkers-right-mover-zonal-storm-motion-m-s01',
    'bunkers-right-mover-meridional-storm-motion-m-s01',
    'storm-relative-helicity-m2-s02_top-height-m-agl=0000.0000'
]

FIRST_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.MOST_UNSTABLE_CAPE_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY:
        derived_field_utils.MOST_UNSTABLE_PARCEL_SOURCE_STRING,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

SECOND_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.MOST_UNSTABLE_CIN_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY:
        derived_field_utils.MOST_UNSTABLE_PARCEL_SOURCE_STRING,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

THIRD_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.SURFACE_BASED_CAPE_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY:
        derived_field_utils.SURFACE_PARCEL_SOURCE_STRING,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

FOURTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.SURFACE_BASED_CIN_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY:
        derived_field_utils.SURFACE_PARCEL_SOURCE_STRING,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

FIFTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.MIXED_LAYER_CAPE_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY:
        derived_field_utils.MIXED_LAYER_PARCEL_SOURCE_STRING,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: 500.,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

SIXTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.MIXED_LAYER_CIN_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY:
        derived_field_utils.MIXED_LAYER_PARCEL_SOURCE_STRING,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: 499.9999,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

SEVENTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.LIFTED_INDEX_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: 50000,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

EIGHTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.PRECIPITABLE_WATER_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: 5000,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

NINTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.SCALAR_WIND_SHEAR_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: 50000,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: 'surface'
}

TENTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.ZONAL_WIND_SHEAR_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: 50000,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: 85000
}

ELEVENTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.MERIDIONAL_WIND_SHEAR_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: 50000,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: 100000
}

TWELFTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.SCALAR_STORM_MOTION_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

THIRTEENTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.ZONAL_STORM_MOTION_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

FOURTEENTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.MERIDIONAL_STORM_MOTION_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

FIFTEENTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.HELICITY_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: 3000.,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

SIXTEENTH_METADATA_DICT = {
    derived_field_utils.BASIC_FIELD_KEY:
        derived_field_utils.PBL_HEIGHT_NAME,
    derived_field_utils.PARCEL_SOURCE_KEY: None,
    derived_field_utils.MIXED_LAYER_DEPTH_KEY: None,
    derived_field_utils.TOP_PRESSURE_KEY: None,
    derived_field_utils.TOP_HEIGHT_KEY: None,
    derived_field_utils.BOTTOM_PRESSURE_KEY: None
}

METADATA_DICTS = [
    FIRST_METADATA_DICT,
    SECOND_METADATA_DICT,
    THIRD_METADATA_DICT,
    FOURTH_METADATA_DICT,
    FIFTH_METADATA_DICT,
    SIXTH_METADATA_DICT,
    SEVENTH_METADATA_DICT,
    EIGHTH_METADATA_DICT,
    NINTH_METADATA_DICT,
    TENTH_METADATA_DICT,
    ELEVENTH_METADATA_DICT,
    TWELFTH_METADATA_DICT,
    THIRTEENTH_METADATA_DICT,
    FOURTEENTH_METADATA_DICT,
    FIFTEENTH_METADATA_DICT,
    SIXTEENTH_METADATA_DICT,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None
]

IS_FIELD_TO_COMPUTE_FLAGS = numpy.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    dtype=bool
)

# The following constants are used to test _height_to_geopotential and
# _geopotential_to_height.
HEIGHTS_METRES = numpy.array([0, 1, 10, 100, numpy.nan])
GEOPOTENTIALS_M2_S02 = numpy.array([
    0, 9.80654846, 98.06534608, 980.6396083, numpy.nan
])

# The following constants are used to test _get_slices_for_multiprocessing.
NUM_GRID_ROWS = 721
START_ROWS_FOR_MULTIPROCESSING = numpy.array(
    [0, 90, 180, 270, 360, 451, 541, 631], dtype=int
)
END_ROWS_FOR_MULTIPROCESSING = numpy.array(
    [90, 180, 270, 360, 451, 541, 631, 721], dtype=int
)

# The following constants are used to test _get_model_pressure_matrix.
PRESSURE_LEVELS_HPA = numpy.array(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50], dtype=int
)
PRESSURE_LEVELS_PASCALS = PRESSURE_LEVELS_HPA * 100

COORD_DICT = {
    model_utils.VALID_TIME_DIM: numpy.array([0], dtype=int),
    model_utils.LATITUDE_DEG_NORTH_DIM: numpy.array([0, 1, 2, 3], dtype=float),
    model_utils.LONGITUDE_DEG_EAST_DIM: numpy.array([100, 110, 120], dtype=float),
    model_utils.PRESSURE_HPA_DIM: PRESSURE_LEVELS_HPA
}

RANDOM_DATA_MATRIX = numpy.random.normal(loc=0., scale=1., size=(1, 13, 4, 3))
THESE_DIM = (
    model_utils.VALID_TIME_DIM, model_utils.PRESSURE_HPA_DIM,
    model_utils.LATITUDE_DEG_NORTH_DIM, model_utils.LONGITUDE_DEG_EAST_DIM
)
MAIN_DATA_DICT = {
    model_utils.SEA_LEVEL_PRESSURE_PASCALS_KEY: (THESE_DIM, RANDOM_DATA_MATRIX)
}
FORECAST_TABLE_XARRAY = xarray.Dataset(
    coords=COORD_DICT, data_vars=MAIN_DATA_DICT
)

PRESSURE_MATRIX_PASCALS_VERT_AXIS_FIRST = numpy.expand_dims(
    PRESSURE_LEVELS_PASCALS, axis=-1
)
PRESSURE_MATRIX_PASCALS_VERT_AXIS_FIRST = numpy.repeat(
    PRESSURE_MATRIX_PASCALS_VERT_AXIS_FIRST, axis=-1, repeats=4
)
PRESSURE_MATRIX_PASCALS_VERT_AXIS_FIRST = numpy.expand_dims(
    PRESSURE_MATRIX_PASCALS_VERT_AXIS_FIRST, axis=-1
)
PRESSURE_MATRIX_PASCALS_VERT_AXIS_FIRST = numpy.repeat(
    PRESSURE_MATRIX_PASCALS_VERT_AXIS_FIRST, axis=-1, repeats=3
)

PRESSURE_MATRIX_PASCALS_VERT_AXIS_LAST = numpy.expand_dims(
    PRESSURE_LEVELS_PASCALS, axis=0
)
PRESSURE_MATRIX_PASCALS_VERT_AXIS_LAST = numpy.repeat(
    PRESSURE_MATRIX_PASCALS_VERT_AXIS_LAST, axis=0, repeats=3
)
PRESSURE_MATRIX_PASCALS_VERT_AXIS_LAST = numpy.expand_dims(
    PRESSURE_MATRIX_PASCALS_VERT_AXIS_LAST, axis=0
)
PRESSURE_MATRIX_PASCALS_VERT_AXIS_LAST = numpy.repeat(
    PRESSURE_MATRIX_PASCALS_VERT_AXIS_LAST, axis=0, repeats=4
)

# The following constants are used to test
# _height_agl_to_nearest_pressure_level and _interp_pressure_to_surface.
SURFACE_GEOPOTENTIAL_MATRIX_M2_S02 = numpy.array([
    [0, 9.80654846],
    [98.06534608, 980.6396083],
    [0, 0]
])

GEOPOTENTIAL_MATRIX_LEVEL1_M2_S02 = numpy.array([
    [10, 11],
    [12, 13],
    [14, 15]
], dtype=float)

GEOPOTENTIAL_MATRIX_LEVEL2_M2_S02 = numpy.array([
    [150, 140],
    [130, 120],
    [110, 100]
], dtype=float)

GEOPOTENTIAL_MATRIX_LEVEL3_M2_S02 = numpy.array([
    [1000, 500],
    [700, 900],
    [600, 800]
], dtype=float)

GEOPOTENTIAL_MATRIX_LEVEL4_M2_S02 = numpy.array([
    [1500, 1500],
    [1500, 1500],
    [1500, 1500]
], dtype=float)

GEOPOTENTIAL_MATRIX_M2_S02 = numpy.stack([
    GEOPOTENTIAL_MATRIX_LEVEL1_M2_S02, GEOPOTENTIAL_MATRIX_LEVEL2_M2_S02,
    GEOPOTENTIAL_MATRIX_LEVEL3_M2_S02, GEOPOTENTIAL_MATRIX_LEVEL4_M2_S02,
], axis=0)

# This corresponds to a geopotential of
# [[ 500.          509.80654846]
#  [ 598.06534608 1480.6396083 ]
#  [ 500.                   500.]]
# at the six grid points.
DESIRED_HEIGHT_M_AGL = derived_field_utils._geopotential_to_height(
    geopotential_m2_s02=500.
)

NEAREST_PRESSURE_INDEX_MATRIX = numpy.array([
    [1, 2],
    [2, 3],
    [2, 2]
], dtype=int)

PRESSURE_MATRIX_LEVEL1_PASCALS = 100 * numpy.array([
    [1000, 1000],
    [1010, 1010],
    [1015, 990]
], dtype=float)

PRESSURE_MATRIX_LEVEL2_PASCALS = 100 * numpy.array([
    [975, 985],
    [990, 995],
    [1000, 970]
], dtype=float)

PRESSURE_MATRIX_LEVEL3_PASCALS = 100 * numpy.array([
    [900, 950],
    [940, 920],
    [955, 910]
], dtype=float)

PRESSURE_MATRIX_LEVEL4_PASCALS = 100 * numpy.array([
    [850, 850],
    [860, 865],
    [875, 835]
], dtype=float)

PRESSURE_MATRIX_FOR_INTERP_PASCALS = numpy.stack([
    PRESSURE_MATRIX_LEVEL1_PASCALS, PRESSURE_MATRIX_LEVEL2_PASCALS,
    PRESSURE_MATRIX_LEVEL3_PASCALS, PRESSURE_MATRIX_LEVEL4_PASCALS
], axis=0)

FIRST_SURFACE_PRESSURE_PASCALS = (
    PRESSURE_MATRIX_FOR_INTERP_PASCALS[0, 0, 0] -
    numpy.diff(PRESSURE_MATRIX_FOR_INTERP_PASCALS[:2, 0, 0])[0] / 14
)
SECOND_SURFACE_PRESSURE_PASCALS = (
    PRESSURE_MATRIX_FOR_INTERP_PASCALS[0, 0, 1] -
    1.19345154 * numpy.diff(PRESSURE_MATRIX_FOR_INTERP_PASCALS[:2, 0, 1])[0] / 129
)
THIRD_SURFACE_PRESSURE_PASCALS = (
    PRESSURE_MATRIX_FOR_INTERP_PASCALS[0, 1, 0] +
    86.06534608 * numpy.diff(PRESSURE_MATRIX_FOR_INTERP_PASCALS[:2, 1, 0])[0] / 118
)
FOURTH_SURFACE_PRESSURE_PASCALS = (
    PRESSURE_MATRIX_FOR_INTERP_PASCALS[2, 1, 1] +
    80.6396083 * numpy.diff(PRESSURE_MATRIX_FOR_INTERP_PASCALS[-2:, 1, 1])[0] / 600
)
FIFTH_SURFACE_PRESSURE_PASCALS = (
    PRESSURE_MATRIX_FOR_INTERP_PASCALS[0, 2, 0] -
    14 * numpy.diff(PRESSURE_MATRIX_FOR_INTERP_PASCALS[:2, 2, 0])[0] / 96
)
SIXTH_SURFACE_PRESSURE_PASCALS = (
    PRESSURE_MATRIX_FOR_INTERP_PASCALS[0, 2, 1] -
    15 * numpy.diff(PRESSURE_MATRIX_FOR_INTERP_PASCALS[:2, 2, 1])[0] / 85
)

INTERP_SURFACE_PRESSURE_MATRIX_PASCALS = numpy.array([
    [FIRST_SURFACE_PRESSURE_PASCALS, SECOND_SURFACE_PRESSURE_PASCALS],
    [THIRD_SURFACE_PRESSURE_PASCALS, FOURTH_SURFACE_PRESSURE_PASCALS],
    [FIFTH_SURFACE_PRESSURE_PASCALS, SIXTH_SURFACE_PRESSURE_PASCALS],
])

# The following constants are used to test _estimate_surface_pressure.
THESE_PRESSURE_LEVELS_PASCALS = 100 * numpy.array(
    [1000, 925, 850, 700], dtype=int
)

COORD_DICT = {
    model_utils.VALID_TIME_DIM: numpy.array([0], dtype=int),
    model_utils.LATITUDE_DEG_NORTH_DIM: numpy.array([0, 1, 2], dtype=float),
    model_utils.LONGITUDE_DEG_EAST_DIM: numpy.array([100, 110], dtype=float),
    model_utils.PRESSURE_HPA_DIM: numpy.array([1000, 925, 850, 700], dtype=int)
}

SEA_LEVEL_PRESSURE_MATRIX_PASCALS = 100 * numpy.array([
    [1039, 1017],
    [1018, 1044],
    [1001, 1007]
], dtype=float)

THESE_DIM = (
    model_utils.VALID_TIME_DIM, model_utils.PRESSURE_HPA_DIM,
    model_utils.LATITUDE_DEG_NORTH_DIM, model_utils.LONGITUDE_DEG_EAST_DIM
)
MAIN_DATA_DICT = {
    model_utils.GEOPOTENTIAL_M2_S02_KEY: (
        THESE_DIM, numpy.expand_dims(GEOPOTENTIAL_MATRIX_M2_S02, axis=0)
    )
}

THESE_DIM = (
    model_utils.VALID_TIME_DIM,
    model_utils.LATITUDE_DEG_NORTH_DIM, model_utils.LONGITUDE_DEG_EAST_DIM
)
MAIN_DATA_DICT.update({
    model_utils.SEA_LEVEL_PRESSURE_PASCALS_KEY: (
        THESE_DIM, numpy.expand_dims(SEA_LEVEL_PRESSURE_MATRIX_PASCALS, axis=0)
    )
})
FORECAST_TABLE_FOR_SFC_PRESSURE_XARRAY = xarray.Dataset(
    coords=COORD_DICT, data_vars=MAIN_DATA_DICT
)

FIRST_SURFACE_PRESSURE_PASCALS = (
    THESE_PRESSURE_LEVELS_PASCALS[0] -
    numpy.diff(numpy.array([
        SEA_LEVEL_PRESSURE_MATRIX_PASCALS[0, 0],
        THESE_PRESSURE_LEVELS_PASCALS[0]
    ]))[0]
)

SECOND_SURFACE_PRESSURE_PASCALS = (
    THESE_PRESSURE_LEVELS_PASCALS[0] - 1.19345154 *
    numpy.diff(numpy.array([
        SEA_LEVEL_PRESSURE_MATRIX_PASCALS[0, 1],
        THESE_PRESSURE_LEVELS_PASCALS[0]
    ]))[0] / 11
)

THIRD_SURFACE_PRESSURE_PASCALS = (
    THESE_PRESSURE_LEVELS_PASCALS[1] -
    31.93465392 * numpy.diff(THESE_PRESSURE_LEVELS_PASCALS[:2])[0] / 118
)

FOURTH_SURFACE_PRESSURE_PASCALS = (
    THESE_PRESSURE_LEVELS_PASCALS[3] -
    519.3603917 * numpy.diff(THESE_PRESSURE_LEVELS_PASCALS[2:4])[0] / 600
)

FIFTH_SURFACE_PRESSURE_PASCALS = (
    THESE_PRESSURE_LEVELS_PASCALS[0] -
    numpy.diff(numpy.array([
        SEA_LEVEL_PRESSURE_MATRIX_PASCALS[2, 0],
        THESE_PRESSURE_LEVELS_PASCALS[0]
    ]))[0]
)

SIXTH_SURFACE_PRESSURE_PASCALS = (
    THESE_PRESSURE_LEVELS_PASCALS[0] -
    numpy.diff(numpy.array([
        SEA_LEVEL_PRESSURE_MATRIX_PASCALS[2, 1],
        THESE_PRESSURE_LEVELS_PASCALS[0]
    ]))[0]
)

ESTIMATED_SURFACE_PRESSURE_MATRIX_PASCALS = numpy.array([
    [FIRST_SURFACE_PRESSURE_PASCALS, SECOND_SURFACE_PRESSURE_PASCALS],
    [THIRD_SURFACE_PRESSURE_PASCALS, FOURTH_SURFACE_PRESSURE_PASCALS],
    [FIFTH_SURFACE_PRESSURE_PASCALS, SIXTH_SURFACE_PRESSURE_PASCALS]
])

# The following constants are used to test _get_mean_wind and
# _interp_wind_to_heights_agl.
ZONAL_WIND_MATRIX_LEVEL1_M_S01 = numpy.array([
    [0, -5],
    [5, 1],
    [-7, 7]
], dtype=float)

ZONAL_WIND_MATRIX_LEVEL2_M_S01 = numpy.array([
    [2, -1],
    [6, 4],
    [-5, 15]
], dtype=float)

ZONAL_WIND_MATRIX_LEVEL3_M_S01 = numpy.array([
    [8, -1],
    [3, 7],
    [0, 25]
], dtype=float)

ZONAL_WIND_MATRIX_LEVEL4_M_S01 = numpy.array([
    [10, 7],
    [10, 10],
    [12, 40]
], dtype=float)

MERIDIONAL_WIND_MATRIX_LEVEL1_M_S01 = numpy.array([
    [0, 1],
    [-1, 0],
    [-1, -1]
], dtype=float)

MERIDIONAL_WIND_MATRIX_LEVEL2_M_S01 = numpy.array([
    [2, 2],
    [-2, -2],
    [-3, -3]
], dtype=float)

MERIDIONAL_WIND_MATRIX_LEVEL3_M_S01 = numpy.array([
    [0, 0],
    [0, 0],
    [0, 0]
], dtype=float)

MERIDIONAL_WIND_MATRIX_LEVEL4_M_S01 = numpy.array([
    [4, 3],
    [2, 1],
    [5, 6]
], dtype=float)

ZONAL_WIND_MATRIX_M_S01 = numpy.stack([
    ZONAL_WIND_MATRIX_LEVEL1_M_S01, ZONAL_WIND_MATRIX_LEVEL2_M_S01,
    ZONAL_WIND_MATRIX_LEVEL3_M_S01, ZONAL_WIND_MATRIX_LEVEL4_M_S01
], axis=0)

MERIDIONAL_WIND_MATRIX_LEVEL_M_S01 = numpy.stack([
    MERIDIONAL_WIND_MATRIX_LEVEL1_M_S01, MERIDIONAL_WIND_MATRIX_LEVEL2_M_S01,
    MERIDIONAL_WIND_MATRIX_LEVEL3_M_S01, MERIDIONAL_WIND_MATRIX_LEVEL4_M_S01
], axis=0)

BOTTOM_INDEX_MATRIX = numpy.array([
    [0, 1],
    [2, 1],
    [3, 2]
], dtype=int)

TOP_INDEX_MATRIX = numpy.array([
    [3, 3],
    [3, 1],
    [3, 3]
], dtype=int)

SURFACE_PRESSURE_MATRIX_PASCALS = 100 * numpy.array([
    [1013.25, 1013.25],
    [1000, 850],
    [1000, 1000]
])

PRESSURE_MATRIX_LEVEL1_PASCALS = 100 * numpy.array([
    [1000, 1000],
    [1000, 1000],
    [1000, 1000]
])

PRESSURE_MATRIX_LEVEL2_PASCALS = 100 * numpy.array([
    [900, 900],
    [900, 900],
    [900, 900]
])

PRESSURE_MATRIX_LEVEL3_PASCALS = 100 * numpy.array([
    [800, 800],
    [800, 800],
    [800, 800]
])

PRESSURE_MATRIX_LEVEL4_PASCALS = 100 * numpy.array([
    [700, 700],
    [700, 700],
    [700, 700]
])

PRESSURE_MATRIX_FOR_WIND_PASCALS = numpy.stack([
    PRESSURE_MATRIX_LEVEL1_PASCALS, PRESSURE_MATRIX_LEVEL2_PASCALS,
    PRESSURE_MATRIX_LEVEL3_PASCALS, PRESSURE_MATRIX_LEVEL4_PASCALS
], axis=0)

MEAN_ZONAL_WIND_MATRIX_LEVEL_M_S01 = numpy.array([
    [5, 5. / 3],
    [6.5, numpy.nan],
    [12, 32.5]
])

MEAN_MERIDIONAL_WIND_MATRIX_LEVEL_M_S01 = numpy.array([
    [1.5, 5. / 3],
    [1, numpy.nan],
    [5, 3]
])

FIRST_ZONAL_WIND_M_S01 = (
    ZONAL_WIND_MATRIX_M_S01[0, 0, 0] -
    numpy.diff(ZONAL_WIND_MATRIX_M_S01[:2, 0, 0])[0] / 14
)
SECOND_ZONAL_WIND_M_S01 = (
    ZONAL_WIND_MATRIX_M_S01[0, 0, 1] -
    1.19345154 * numpy.diff(ZONAL_WIND_MATRIX_M_S01[:2, 0, 1])[0] / 129
)
THIRD_ZONAL_WIND_M_S01 = (
    ZONAL_WIND_MATRIX_M_S01[0, 1, 0] +
    86.06534608 * numpy.diff(ZONAL_WIND_MATRIX_M_S01[:2, 1, 0])[0] / 118
)
FOURTH_ZONAL_WIND_M_S01 = (
    ZONAL_WIND_MATRIX_M_S01[2, 1, 1] +
    80.6396083 * numpy.diff(ZONAL_WIND_MATRIX_M_S01[-2:, 1, 1])[0] / 600
)
FIFTH_ZONAL_WIND_M_S01 = (
    ZONAL_WIND_MATRIX_M_S01[0, 2, 0] -
    14 * numpy.diff(ZONAL_WIND_MATRIX_M_S01[:2, 2, 0])[0] / 96
)
SIXTH_ZONAL_WIND_M_S01 = (
    ZONAL_WIND_MATRIX_M_S01[0, 2, 1] -
    15 * numpy.diff(ZONAL_WIND_MATRIX_M_S01[:2, 2, 1])[0] / 85
)

INTERP_SURFACE_ZONAL_WIND_MATRIX_M_S01 = numpy.array([
    [FIRST_ZONAL_WIND_M_S01, SECOND_ZONAL_WIND_M_S01],
    [THIRD_ZONAL_WIND_M_S01, FOURTH_ZONAL_WIND_M_S01],
    [FIFTH_ZONAL_WIND_M_S01, SIXTH_ZONAL_WIND_M_S01],
])

# The following constants are used to test _get_pbl_height.
THETA_V_MATRIX_LEVEL1_KELVINS = numpy.array([
    [358, 358.5],
    [359, 357.8],
    [356, 356]
])

THETA_V_MATRIX_LEVEL2_KELVINS = numpy.array([
    [359, 353.5],
    [400, 353.2],
    [352.5, 352.5]
])

THETA_V_MATRIX_LEVEL3_KELVINS = numpy.array([
    [351, 350],
    [352.5, 350.2],
    [350, 350]
])

THETA_V_MATRIX_LEVEL4_KELVINS = numpy.array([
    [350, 349],
    [351, 350],
    [350, 350]
], dtype=float)

THETA_V_MATRIX_LEVEL5_KELVINS = numpy.array([
    [360, 350],
    [357, 352],
    [356.51, 351.5]
])

THETA_V_MATRIX_LEVEL6_KELVINS = numpy.array([
    [300, 353.5],
    [360, 357.9],
    [365, 357]
])

THETA_V_MATRIX_LEVEL7_KELVINS = numpy.array([
    [300, 360],
    [360, 357.9],
    [365, 357]
])

THETA_V_MATRIX_KELVINS = numpy.stack([
    THETA_V_MATRIX_LEVEL1_KELVINS, THETA_V_MATRIX_LEVEL2_KELVINS,
    THETA_V_MATRIX_LEVEL3_KELVINS, THETA_V_MATRIX_LEVEL4_KELVINS,
    THETA_V_MATRIX_LEVEL5_KELVINS, THETA_V_MATRIX_LEVEL6_KELVINS,
    THETA_V_MATRIX_LEVEL7_KELVINS
], axis=0)

HEIGHT_MATRIX_LEVEL1_M_ASL = numpy.array([
    [0, 0],
    [1900, 0],
    [0, 0]
], dtype=float)

HEIGHT_MATRIX_LEVEL2_M_ASL = numpy.array([
    [1000, 1000],
    [1000, 1000],
    [1000, 1000]
], dtype=float)

HEIGHT_MATRIX_LEVEL3_M_ASL = numpy.array([
    [2000, 2000],
    [2000, 2000],
    [2000, 2000]
], dtype=float)

HEIGHT_MATRIX_LEVEL4_M_ASL = numpy.array([
    [3000, 3000],
    [3000, 3000],
    [3000, 3000]
], dtype=float)

HEIGHT_MATRIX_LEVEL5_M_ASL = numpy.array([
    [4000, 4000],
    [4000, 4000],
    [4000, 4000]
], dtype=float)

HEIGHT_MATRIX_LEVEL6_M_ASL = numpy.array([
    [5000, 5000],
    [5000, 5000],
    [5000, 5000]
], dtype=float)

HEIGHT_MATRIX_LEVEL7_M_ASL = numpy.array([
    [6000, 6000],
    [6000, 6000],
    [6000, 6000]
], dtype=float)

THIS_HEIGHT_MATRIX_M_ASL = numpy.stack([
    HEIGHT_MATRIX_LEVEL1_M_ASL, HEIGHT_MATRIX_LEVEL2_M_ASL,
    HEIGHT_MATRIX_LEVEL3_M_ASL, HEIGHT_MATRIX_LEVEL4_M_ASL,
    HEIGHT_MATRIX_LEVEL5_M_ASL, HEIGHT_MATRIX_LEVEL6_M_ASL,
    HEIGHT_MATRIX_LEVEL7_M_ASL
], axis=0)

GEOPOT_MATRIX_FOR_PBL_M2_S02 = derived_field_utils._height_to_geopotential(
    THIS_HEIGHT_MATRIX_M_ASL
)

FIRST_GEOPTL_M2_S02 = (
    (1. / 2) * GEOPOT_MATRIX_FOR_PBL_M2_S02[0, 0, 0] +
    (1. / 2) * GEOPOT_MATRIX_FOR_PBL_M2_S02[1, 0, 0]
)
SECOND_GEOPTL_M2_S02 = (
    (2. / 13) * GEOPOT_MATRIX_FOR_PBL_M2_S02[5, 0, 1] +
    (11. / 13) * GEOPOT_MATRIX_FOR_PBL_M2_S02[6, 0, 1]
)
THIRD_GEOPTL_M2_S02 = (
    (1. / 6) * GEOPOT_MATRIX_FOR_PBL_M2_S02[4, 1, 0] +
    (5. / 6) * GEOPOT_MATRIX_FOR_PBL_M2_S02[5, 1, 0]
)
FOURTH_GEOPTL_M2_S02 = numpy.nan
FIFTH_GEOPTL_M2_S02 = (
    (1. / 651) * GEOPOT_MATRIX_FOR_PBL_M2_S02[3, 2, 0] +
    (650. / 651) * GEOPOT_MATRIX_FOR_PBL_M2_S02[4, 2, 0]
)
SIXTH_GEOPTL_M2_S02 = (
    (1. / 11) * GEOPOT_MATRIX_FOR_PBL_M2_S02[4, 2, 1] +
    (10. / 11) * GEOPOT_MATRIX_FOR_PBL_M2_S02[5, 2, 1]
)

PBL_TOP_GEOPTL_MATRIX_M2_S02 = numpy.array([
    [FIRST_GEOPTL_M2_S02, SECOND_GEOPTL_M2_S02],
    [THIRD_GEOPTL_M2_S02, FOURTH_GEOPTL_M2_S02],
    [FIFTH_GEOPTL_M2_S02, SIXTH_GEOPTL_M2_S02]
])

PBL_HEIGHT_MATRIX_M_AGL = derived_field_utils._geopotential_to_height(
    PBL_TOP_GEOPTL_MATRIX_M2_S02 - GEOPOT_MATRIX_FOR_PBL_M2_S02[0, ...]
)

# The following constants are used to test _integrate_to_precipitable_water.
PRESSURE_MATRIX_LEVEL1_PASCALS = 100 * numpy.array([
    [1006, 1009],
    [986, 995],
    [1019, 1002]
])

PRESSURE_MATRIX_LEVEL2_PASCALS = 100 * numpy.array([
    [892, 943],
    [913, 906],
    [925, 892]
])

PRESSURE_MATRIX_LEVEL3_PASCALS = 100 * numpy.array([
    [788, 799],
    [670, 776],
    [831, 815]
])

PRESSURE_MATRIX_LEVEL4_PASCALS = 100 * numpy.array([
    [734, 718],
    [759, 756],
    [745, 684]
])

PRESSURE_MATRIX_FOR_PWAT_PASCALS = numpy.stack([
    PRESSURE_MATRIX_LEVEL1_PASCALS, PRESSURE_MATRIX_LEVEL2_PASCALS,
    PRESSURE_MATRIX_LEVEL3_PASCALS, PRESSURE_MATRIX_LEVEL4_PASCALS
], axis=0)

SPEC_HUMIDITY_MATRIX_LEVEL1_KG_KG01 = 0.001 * numpy.array([
    [32.4, 36.8],
    [10.9, 6.0],
    [numpy.nan, 28.4]
])

SPEC_HUMIDITY_MATRIX_LEVEL2_KG_KG01 = 0.001 * numpy.array([
    [numpy.nan, numpy.nan],
    [33.6, numpy.nan],
    [11.6, 14.5]
])

SPEC_HUMIDITY_MATRIX_LEVEL3_KG_KG01 = 0.001 * numpy.array([
    [numpy.nan, numpy.nan],
    [2.1, 10.1],
    [22.6, numpy.nan]
])

SPEC_HUMIDITY_MATRIX_LEVEL4_KG_KG01 = 0.001 * numpy.array([
    [numpy.nan, 0.2],
    [12.4, 2.8],
    [numpy.nan, 0.0]
])

SPEC_HUMIDITY_MATRIX_KG_KG01 = numpy.stack([
    SPEC_HUMIDITY_MATRIX_LEVEL1_KG_KG01, SPEC_HUMIDITY_MATRIX_LEVEL2_KG_KG01,
    SPEC_HUMIDITY_MATRIX_LEVEL3_KG_KG01, SPEC_HUMIDITY_MATRIX_LEVEL4_KG_KG01
], axis=0)

THIS_COEFF = (
    0.1 * derived_field_utils.METRES_TO_MM /
    (derived_field_utils.WATER_DENSITY_KG_M03 * derived_field_utils.GRAVITY_M_S02)
)
FIRST_PRECIP_WATER_KG_M02 = numpy.nan
SECOND_PRECIP_WATER_KG_M02 = THIS_COEFF * 18.5 * 291
THIRD_PRECIP_WATER_KG_M02 = THIS_COEFF * ((22.25 * 73) + (23.0 * 154) + (7.25 * 89))
FOURTH_PRECIP_WATER_KG_M02 = THIS_COEFF * ((8.05 * 219) + (6.45 * 20))
FIFTH_PRECIP_WATER_KG_M02 = THIS_COEFF * 17.1 * 94
SIXTH_PRECIP_WATER_KG_M02 = THIS_COEFF * ((21.45 * 110) + (7.25 * 208))

PRECIPITABLE_WATER_MATRIX_KG_M02 = numpy.array([
    [FIRST_PRECIP_WATER_KG_M02, SECOND_PRECIP_WATER_KG_M02],
    [THIRD_PRECIP_WATER_KG_M02, FOURTH_PRECIP_WATER_KG_M02],
    [FIFTH_PRECIP_WATER_KG_M02, SIXTH_PRECIP_WATER_KG_M02]
])


def _compare_metadata_dicts(first_dict, second_dict):
    """Compares metadata dictionaries produced by `parse_field_name`.

    :param first_dict: First dictionary.
    :param second_dict: Second dictionary.
    :return: are_dicts_same: Boolean flag.
    """

    these_keys = [
        derived_field_utils.BASIC_FIELD_KEY,
        derived_field_utils.PARCEL_SOURCE_KEY,
        derived_field_utils.TOP_PRESSURE_KEY,
        derived_field_utils.BOTTOM_PRESSURE_KEY
    ]

    for this_key in these_keys:
        if first_dict[this_key] != second_dict[this_key]:
            return False

    float_keys = [
        derived_field_utils.MIXED_LAYER_DEPTH_KEY,
        derived_field_utils.TOP_HEIGHT_KEY
    ]

    for this_key in float_keys:
        if first_dict[this_key] is None and second_dict[this_key] is None:
            continue

        if not numpy.isclose(
                first_dict[this_key], second_dict[this_key], atol=TOLERANCE
        ):
            return False

    return True


class DerivedFieldUtilsTests(unittest.TestCase):
    """Each method is a unit test for derived_field_utils.py."""

    def test_parse_field_name(self):
        """Ensures correct output from parse_field_name."""

        for i in range(len(DERIVED_FIELD_NAMES)):
            if METADATA_DICTS[i] is None:
                with self.assertRaises((AssertionError, ValueError)):
                    derived_field_utils.parse_field_name(
                        derived_field_name=DERIVED_FIELD_NAMES[i],
                        is_field_to_compute=IS_FIELD_TO_COMPUTE_FLAGS[i]
                    )
            else:
                this_metadata_dict = derived_field_utils.parse_field_name(
                    derived_field_name=DERIVED_FIELD_NAMES[i],
                    is_field_to_compute=IS_FIELD_TO_COMPUTE_FLAGS[i]
                )

                self.assertTrue(_compare_metadata_dicts(
                    this_metadata_dict, METADATA_DICTS[i]
                ))

    def test_create_field_name(self):
        """Ensures correct output from create_field_name."""

        for i in range(len(DERIVED_FIELD_NAMES)):
            if METADATA_DICTS[i] is None:
                continue

            this_field_name = derived_field_utils.create_field_name(
                METADATA_DICTS[i]
            )
            self.assertEqual(this_field_name, DERIVED_FIELD_NAMES[i])

    def test_height_to_geopotential(self):
        """Ensures correct output from _height_to_geopotential."""

        these_geopotentials_m2_s02 = (
            derived_field_utils._height_to_geopotential(HEIGHTS_METRES)
        )

        self.assertTrue(numpy.allclose(
            these_geopotentials_m2_s02, GEOPOTENTIALS_M2_S02,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_geopotential_to_height(self):
        """Ensures correct output from _geopotential_to_height."""

        these_heights_metres = (
            derived_field_utils._geopotential_to_height(GEOPOTENTIALS_M2_S02)
        )

        self.assertTrue(numpy.allclose(
            these_heights_metres, HEIGHTS_METRES,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_slices_for_multiprocessing(self):
        """Ensures correct output from _get_slices_for_multiprocessing."""

        these_start_rows, these_end_rows = (
            derived_field_utils._get_slices_for_multiprocessing(
                num_grid_rows=NUM_GRID_ROWS
            )
        )

        self.assertTrue(numpy.array_equal(
            these_start_rows, START_ROWS_FOR_MULTIPROCESSING
        ))
        self.assertTrue(numpy.array_equal(
            these_end_rows, END_ROWS_FOR_MULTIPROCESSING
        ))

    def test_get_model_pressure_matrix_vert_axis_first(self):
        """Ensures correct output from _get_model_pressure_matrix.

        In this case, vertical axis should come first.
        """

        this_pressure_matrix_pascals = (
            derived_field_utils._get_model_pressure_matrix(
                forecast_table_xarray=FORECAST_TABLE_XARRAY,
                vertical_axis_first=True
            )
        )

        self.assertTrue(numpy.allclose(
            this_pressure_matrix_pascals,
            PRESSURE_MATRIX_PASCALS_VERT_AXIS_FIRST,
            atol=TOLERANCE
        ))

    def test_get_model_pressure_matrix_vert_axis_last(self):
        """Ensures correct output from _get_model_pressure_matrix.

        In this case, vertical axis should come last.
        """

        this_pressure_matrix_pascals = (
            derived_field_utils._get_model_pressure_matrix(
                forecast_table_xarray=FORECAST_TABLE_XARRAY,
                vertical_axis_first=False
            )
        )

        self.assertTrue(numpy.allclose(
            this_pressure_matrix_pascals,
            PRESSURE_MATRIX_PASCALS_VERT_AXIS_LAST,
            atol=TOLERANCE
        ))

    def test_height_agl_to_nearest_pressure_level(self):
        """Ensures correct output from _height_agl_to_nearest_pressure_level."""

        this_index_matrix = (
            derived_field_utils._height_agl_to_nearest_pressure_level(
                geopotential_matrix_m2_s02=GEOPOTENTIAL_MATRIX_M2_S02,
                surface_geopotential_matrix_m2_s02=
                SURFACE_GEOPOTENTIAL_MATRIX_M2_S02,
                desired_height_m_agl=DESIRED_HEIGHT_M_AGL,
                find_nearest_level_beneath=False
            )
        )

        self.assertTrue(numpy.array_equal(
            this_index_matrix, NEAREST_PRESSURE_INDEX_MATRIX
        ))

    def test_get_mean_wind(self):
        """Ensures correct output from _get_mean_wind."""

        (
            this_mean_zonal_wind_matrix_m_s01,
            this_mean_merid_wind_matrix_m_s01
        ) = derived_field_utils._get_mean_wind(
            zonal_wind_matrix_m_s01=ZONAL_WIND_MATRIX_M_S01,
            meridional_wind_matrix_m_s01=MERIDIONAL_WIND_MATRIX_LEVEL_M_S01,
            bottom_index_matrix=BOTTOM_INDEX_MATRIX,
            top_index_matrix=TOP_INDEX_MATRIX,
            pressure_weighted=False,
            pressure_matrix_pascals=PRESSURE_MATRIX_FOR_WIND_PASCALS,
            surface_pressure_matrix_pascals=SURFACE_PRESSURE_MATRIX_PASCALS
        )

        self.assertTrue(numpy.allclose(
            this_mean_zonal_wind_matrix_m_s01,
            MEAN_ZONAL_WIND_MATRIX_LEVEL_M_S01,
            atol=TOLERANCE, equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            this_mean_merid_wind_matrix_m_s01,
            MEAN_MERIDIONAL_WIND_MATRIX_LEVEL_M_S01,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_pbl_height(self):
        """Ensures correct output from _get_pbl_height."""

        this_pbl_height_matrix_m_agl = derived_field_utils._get_pbl_height(
            theta_v_matrix_kelvins=THETA_V_MATRIX_KELVINS,
            geopotential_matrix_m2_s02=GEOPOT_MATRIX_FOR_PBL_M2_S02,
            theta_v_deviation_threshold_kelvins=0.5
        )

        self.assertTrue(numpy.allclose(
            this_pbl_height_matrix_m_agl,
            PBL_HEIGHT_MATRIX_M_AGL,
            atol=TOLERANCE, equal_nan=True
        ))

    # def test_interp_pressure_to_surface(self):
    #     """Ensures correct output from _interp_pressure_to_surface."""
    #
    #     this_pressure_matrix_pascals = (
    #         derived_field_utils._interp_pressure_to_surface(
    #             log10_pressure_matrix_pascals=PRESSURE_MATRIX_FOR_INTERP_PASCALS,
    #             geopotential_matrix_m2_s02=GEOPOTENTIAL_MATRIX_M2_S02,
    #             surface_geopotential_matrix_m2_s02=
    #             SURFACE_GEOPOTENTIAL_MATRIX_M2_S02,
    #             use_spline=False,
    #             test_mode=True
    #         )
    #     )
    #
    #     self.assertTrue(numpy.allclose(
    #         this_pressure_matrix_pascals,
    #         INTERP_SURFACE_PRESSURE_MATRIX_PASCALS,
    #         atol=TOLERANCE
    #     ))

    def test_interp_wind_to_heights_m_agl(self):
        """Ensures correct output from _interp_wind_to_heights_m_agl."""

        (
            this_wind_matrix_m_s01
        ) = derived_field_utils._interp_wind_to_heights_agl(
            wind_matrix_m_s01=ZONAL_WIND_MATRIX_M_S01,
            geopotential_matrix_m2_s02=GEOPOTENTIAL_MATRIX_M2_S02,
            surface_geopotential_matrix_m2_s02=
            SURFACE_GEOPOTENTIAL_MATRIX_M2_S02,
            target_heights_m_agl=numpy.array([0.]),
            use_spline=False
        )[0, ...]

        self.assertTrue(numpy.allclose(
            this_wind_matrix_m_s01,
            INTERP_SURFACE_ZONAL_WIND_MATRIX_M_S01,
            atol=TOLERANCE
        ))

    def test_integrate_to_precipitable_water(self):
        """Ensures correct output from _integrate_to_precipitable_water."""

        this_precip_water_matrix_kg_m02 = (
            derived_field_utils._integrate_to_precipitable_water(
                pressure_matrix_pascals=PRESSURE_MATRIX_FOR_PWAT_PASCALS,
                spec_humidity_matrix_kg_kg01=SPEC_HUMIDITY_MATRIX_KG_KG01,
                test_mode=True
            )
        )

        self.assertTrue(numpy.allclose(
            this_precip_water_matrix_kg_m02,
            PRECIPITABLE_WATER_MATRIX_KG_M02,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_estimate_surface_pressure(self):
        """Ensures correct output from _estimate_surface_pressure."""

        this_pressure_matrix_pascals = (
            derived_field_utils._estimate_surface_pressure(
                forecast_table_xarray=FORECAST_TABLE_FOR_SFC_PRESSURE_XARRAY,
                surface_geopotential_matrix_m2_s02=
                SURFACE_GEOPOTENTIAL_MATRIX_M2_S02,
                do_multiprocessing=False,
                use_spline=False,
                test_mode=True
            )
        )

        self.assertTrue(numpy.allclose(
            this_pressure_matrix_pascals,
            ESTIMATED_SURFACE_PRESSURE_MATRIX_PASCALS,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
