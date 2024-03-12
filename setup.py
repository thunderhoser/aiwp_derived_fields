"""Setup file for aiwp_derived_fields."""

from setuptools import setup

PACKAGE_NAMES = [
    'aiwp_derived_fields', 'aiwp_derived_fields.io',
    'aiwp_derived_fields.utils', 'aiwp_derived_fields.plotting',
    'aiwp_derived_fields.outside_code'
]
KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'weather prediction', 'numerical weather prediction', 'NWP',
    'AI weather prediction', 'AIWP',
    'severe thunderstorms', 'severe convection', 'convective weather',
    'convective indices', 'sounding indices'
]
SHORT_DESCRIPTION = 'Computes derived fields from AIWP models.'
LONG_DESCRIPTION = (
    'Computes derived fields (e.g., profile-based indices for severe '
    'convective weather) from AIWP (AI-based weather prediction) models.'
)
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3'
]

PACKAGE_REQUIREMENTS = [
    'numpy',
    'scipy',
    'xarray',
    'netCDF4',
    'pyproj',
    'matplotlib',
    'pandas',
    'shapely',
    'geopy',
    'xcape'
]

if __name__ == '__main__':
    setup(
        name='aiwp_derived_fields',
        version='0.1',
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Ryan Lagerquist',
        author_email='ryan.lagerquist@noaa.gov',
        url='https://github.com/thunderhoser/aiwp_derived_fields',
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False,
        install_requires=PACKAGE_REQUIREMENTS
    )
