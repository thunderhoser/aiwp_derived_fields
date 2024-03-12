#!/bin/bash

CODE_DIR_NAME="/home/ralager/aiwp_derived_fields/aiwp_derived_fields/scripts"
INPUT_DIR_NAME="/mnt/mlnas01/ralager/aiwp_derived_fields"

INIT_TIME_STRINGS=("2023-03-28-00" "2023-03-28-12" "2023-03-29-00" "2023-03-29-12" "2023-03-30-00" "2023-03-30-12" "2023-03-31-00" "2023-03-31-12" "2023-04-01-00" "2023-04-01-12" "2023-04-02-00" "2023-04-02-12" "2023-04-03-00" "2023-04-03-12" "2023-04-04-00" "2023-04-04-12" "2023-04-05-00" "2023-04-05-12" "2023-04-06-00" "2023-04-06-12" "2023-04-07-00" "2023-04-07-12" "2023-04-08-00" "2023-04-08-12" "2023-04-09-00" "2023-04-09-12" "2023-04-10-00" "2023-04-10-12" "2023-04-11-00" "2023-04-11-12" "2023-04-12-00" "2023-04-12-12" "2023-04-13-00" "2023-04-13-12" "2023-04-14-00" "2023-04-14-12" "2023-04-15-00" "2023-04-15-12" "2023-04-16-00" "2023-04-16-12" "2023-04-17-00" "2023-04-17-12" "2023-04-18-00" "2023-04-18-12" "2023-04-19-00" "2023-04-19-12" "2023-04-20-00" "2023-04-20-12" "2023-04-21-00" "2023-04-21-12" "2023-04-22-00" "2023-04-22-12" "2023-04-23-00" "2023-04-23-12" "2023-04-24-00" "2023-04-24-12" "2023-04-25-00" "2023-04-25-12" "2023-04-26-00" "2023-04-26-12" "2023-04-27-00" "2023-04-27-12" "2023-04-28-00" "2023-04-28-12" "2023-04-29-00" "2023-04-29-12" "2023-04-30-00" "2023-04-30-12")

i=0

while [ $i -lt ${#INIT_TIME_STRINGS[@]} ]; do
    current_time_string=`date +"%Y-%m-%d-%H%M%S"`
    log_file_name="plot_derived_fields_fourcastnet_${INIT_TIME_STRINGS[$i]}_${current_time_string}.out"

    python3 -u "${CODE_DIR_NAME}/plot_derived_fields.py" &> ${log_file_name} \
    --derived_field_names "most-unstable-cape-j-kg01" "most-unstable-cin-j-kg01" "surface-based-cape-j-kg01" "surface-based-cin-j-kg01" "mixed-layer-cape-j-kg01_ml-depth-metres=500.0000" "mixed-layer-cin-j-kg01_ml-depth-metres=500.0000" "lifted-index-kelvins_top-pressure-pascals=50000" "precipitable-water-kg-m02_top-pressure-pascals=5000" "zonal-wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=surface" "meridional-wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=surface" "zonal-wind-shear-m-s01_top-pressure-pascals=85000_bottom-pressure-pascals=surface" "meridional-wind-shear-m-s01_top-pressure-pascals=85000_bottom-pressure-pascals=surface" "zonal-wind-shear-m-s01_top-pressure-pascals=20000_bottom-pressure-pascals=85000" "meridional-wind-shear-m-s01_top-pressure-pascals=20000_bottom-pressure-pascals=85000" "storm-relative-positive-helicity-m2-s02_top-height-m-agl=3000.0000" "storm-relative-positive-helicity-m2-s02_top-height-m-agl=1000.0000" "planetary-boundary-layer-height-m-agl" \
    --input_dir_name="${INPUT_DIR_NAME}" \
    --model_name="FOUR_v200" \
    --init_time_string="${INIT_TIME_STRINGS[$i]}" \
    --forecast_hours 0 54 96 102 162 \
    --min_colour_percentiles 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
    --max_colour_percentiles 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 \
    --output_dir_name="${INPUT_DIR_NAME}/plots/FOUR_v200/${INIT_TIME_STRINGS[$i]}"
    
    i=$(( $i + 7 ))
done
