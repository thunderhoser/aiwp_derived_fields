"""Downloads AIWP output from Amazon Web Services."""

import argparse
import subprocess
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from aiwp_derived_fields.io import basic_field_io

TIME_FORMAT = '%Y-%m-%d-%H'
INPUT_ONLINE_DIR_NAME = 'https://noaa-oar-mlwp-data.s3.amazonaws.com'

OUTPUT_DIR_ARG_NAME = 'output_local_dir_name'
MODEL_ARG_NAME = 'model_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
WAIT_PERIOD_ARG_NAME = 'wait_period_sec'

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Downloaded files will be saved here on the '
    'local machine, to exact locations determined by '
    '`basic_field_io.find_file`.'
)
MODEL_ARG_HELP_STRING = (
    'Name of model to download.  Must be accepted by '
    '`model_utils.check_model_name`.'
)
FIRST_INIT_TIME_HELP_STRING = (
    'First initialization time (format "yyyy-mm-dd-HH").  This script will '
    'download all model runs initialized in the period `{0:s}`...`{1:s}`.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
WAIT_PERIOD_HELP_STRING = (
    'Waiting period between downloading two files.  The longer the waiting '
    'period, the lower the risk of overwhelming the server and getting denied.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_ARG_NAME, type=str, required=True,
    help=MODEL_ARG_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=LAST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WAIT_PERIOD_ARG_NAME, type=int, required=True,
    help=WAIT_PERIOD_HELP_STRING
)


def _run(output_local_dir_name, model_name, first_init_time_string,
         last_init_time_string, wait_period_sec):
    """Downloads AIWP output from Amazon Web Services.

    This is effectively the main method.

    :param output_local_dir_name: See documentation at top of this script.
    :param model_name: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param wait_period_sec: Same.
    """

    error_checking.assert_is_greater(wait_period_sec, 0)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_local_dir_name
    )

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=basic_field_io.INIT_TIME_INTERVAL_SEC,
        include_endpoint=True
    )

    online_file_names = [
        basic_field_io.find_file(
            directory_name=
            '{0:s}/{1:s}'.format(INPUT_ONLINE_DIR_NAME, model_name),
            model_name=model_name,
            init_time_unix_sec=t,
            raise_error_if_missing=False
        )
        for t in init_times_unix_sec
    ]

    command_string = (
        'wget -vx --no-clobber --wait={0:d} --random-wait '
        '--directory-prefix="{1:s}" '
    ).format(
        wait_period_sec, output_local_dir_name
    )

    command_string += ' '.join(['"{0:s}"'.format(f) for f in online_file_names])
    print(command_string)
    subprocess.call(command_string, shell=True)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_local_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        model_name=getattr(INPUT_ARG_OBJECT, MODEL_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        wait_period_sec=getattr(INPUT_ARG_OBJECT, WAIT_PERIOD_ARG_NAME)
    )
