"""Unit tests for derived_field_utils.py"""

import unittest
import numpy
from aiwp_derived_fields.utils import derived_field_utils

TOLERANCE = 1e-6

DERIVED_FIELD_NAMES = [
    derived_field_utils.MOST_UNSTABLE_CAPE_NAME.replace('_', '-'),
    derived_field_utils.MOST_UNSTABLE_CIN_NAME.replace('_', '-'),
    derived_field_utils.SURFACE_BASED_CAPE_NAME.replace('_', '-'),
    derived_field_utils.SURFACE_BASED_CIN_NAME.replace('_', '-'),
    '{0:s}_ml-depth-metres=500.0000'.format(
        derived_field_utils.MIXED_LAYER_CAPE_NAME.replace('_', '-')
    ),
    '{0:s}_ml-depth-metres=499.9999'.format(
        derived_field_utils.MIXED_LAYER_CIN_NAME.replace('_', '-')
    ),
    '{0:s}_top-pressure-pascals=50000'.format(
        derived_field_utils.LIFTED_INDEX_NAME.replace('_', '-')
    ),
    '{0:s}_top-pressure-pascals=5000'.format(
        derived_field_utils.PRECIPITABLE_WATER_NAME.replace('_', '-')
    ),
    '{0:s}_top-pressure-pascals=50000_bottom-pressure-pascals=surface'.format(
        derived_field_utils.SCALAR_WIND_SHEAR_NAME.replace('_', '-')
    ),
    '{0:s}_top-pressure-pascals=50000_bottom-pressure-pascals=85000'.format(
        derived_field_utils.ZONAL_WIND_SHEAR_NAME.replace('_', '-')
    ),
    '{0:s}_top-pressure-pascals=50000_bottom-pressure-pascals=100000'.format(
        derived_field_utils.MERIDIONAL_WIND_SHEAR_NAME.replace('_', '-')
    ),

    derived_field_utils.SCALAR_STORM_MOTION_NAME.replace('_', '-'),
    derived_field_utils.ZONAL_STORM_MOTION_NAME.replace('_', '-'),
    derived_field_utils.MERIDIONAL_STORM_MOTION_NAME.replace('_', '-'),
    '{0:s}_top-height-m-agl=3000.0000'.format(
        derived_field_utils.HELICITY_NAME.replace('_', '-')
    ),
    derived_field_utils.PBL_HEIGHT_NAME.replace('_', '-'),

    '{0:s}_ml-depth-metres=FOO'.format(
        derived_field_utils.MIXED_LAYER_CAPE_NAME.replace('_', '-')
    ),
    '{0:s}_top-pressure-pascals=surface_bottom-pressure-pascals=surface'.format(
        derived_field_utils.SCALAR_WIND_SHEAR_NAME.replace('_', '-')
    ),
    '{0:s}_top-pressure-pascals=50000_bottom-pressure-pascals=40000'.format(
        derived_field_utils.MERIDIONAL_WIND_SHEAR_NAME.replace('_', '-')
    ),

    '{0:s}_top-pressure-pascals=50000_bottom-pressure-pascals=surface'.format(
        derived_field_utils.SCALAR_WIND_SHEAR_NAME.replace('_', '-')
    ),
    '{0:s}_top-pressure-pascals=50000_bottom-pressure-pascals=85000'.format(
        derived_field_utils.ZONAL_WIND_SHEAR_NAME.replace('_', '-')
    ),
    '{0:s}_top-pressure-pascals=50000_bottom-pressure-pascals=100000'.format(
        derived_field_utils.MERIDIONAL_WIND_SHEAR_NAME.replace('_', '-')
    ),
    derived_field_utils.SCALAR_STORM_MOTION_NAME.replace('_', '-'),
    derived_field_utils.ZONAL_STORM_MOTION_NAME.replace('_', '-'),
    derived_field_utils.MERIDIONAL_STORM_MOTION_NAME.replace('_', '-'),
    '{0:s}_top-height-m-agl=0000.0000'.format(
        derived_field_utils.HELICITY_NAME.replace('_', '-')
    )
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


if __name__ == '__main__':
    unittest.main()
