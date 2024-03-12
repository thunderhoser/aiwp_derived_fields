"""Helper methods for AIWP models."""

from gewittergefahr.gg_utils import error_checking

FOURCASTNET_VERSION2_NAME = 'FOUR_v200'
GRAPHCAST_VERSION1_NAME = 'GRAP_v100'
PANGU_WEATHER_VERSION1_NAME = 'PANG_v100'

ALL_MODEL_NAMES = [
    FOURCASTNET_VERSION2_NAME, GRAPHCAST_VERSION1_NAME,
    PANGU_WEATHER_VERSION1_NAME
]

VALID_TIME_DIM = 'valid_time_unix_sec'
PRESSURE_HPA_DIM = 'level'
LATITUDE_DEG_NORTH_DIM = 'latitude'
LONGITUDE_DEG_EAST_DIM = 'longitude'

TEMPERATURE_KELVINS_KEY = 't'
TEMPERATURE_2METRES_KELVINS_KEY = 't2'
SPECIFIC_HUMIDITY_KG_KG01_KEY = 'q'
SEA_LEVEL_PRESSURE_PASCALS_KEY = 'msl'
GEOPOTENTIAL_M2_S02_KEY = 'z'
ZONAL_WIND_M_S01_KEY = 'u'
MERIDIONAL_WIND_M_S01_KEY = 'v'
ZONAL_WIND_10METRES_M_S01_KEY = 'u10'
MERIDIONAL_WIND_10METRES_M_S01_KEY = 'v10'


def check_model_name(model_name):
    """Ensures that model name is valid.

    :param model_name: String (must be in list `ALL_MODEL_NAMES`).
    :raises: ValueError: if `model_name not in ALL_MODEL_NAMES`.
    """

    error_checking.assert_is_string(model_name)
    if model_name in ALL_MODEL_NAMES:
        return

    error_string = (
        'Model name "{0:s}" is not in the list of accepted model names '
        '(below):\n{1:s}'
    ).format(
        model_name, str(ALL_MODEL_NAMES)
    )

    raise ValueError(error_string)


def model_name_to_fancy(model_name):
    """Converts model name to fancy version.

    :param model_name: Pythonic name (must be in list `ALL_MODEL_NAMES`).
    :return: fancy_model_name: Fancy name (suitable for figure titles and shit).
    """

    check_model_name(model_name)

    if model_name == FOURCASTNET_VERSION2_NAME:
        return 'FourCast v2'
    if model_name == GRAPHCAST_VERSION1_NAME:
        return 'GraphCast v1'

    return 'Pangu v1'
