#!/bin/bash

CODE_DIR_NAME="/home/ralager/aiwp_derived_fields/aiwp_derived_fields/scripts"
INPUT_DIR_NAME="/home/ralager/era5_data_from_allie"
OUTPUT_DIR_NAME="/home/ralager/era5_data_from_allie/derived_fields"

INIT_TIME_STRINGS=("2023-04-07-00" "2023-04-07-06" "2023-04-07-12" "2023-04-07-18")

i=0

while [ $i -lt ${#INIT_TIME_STRINGS[@]} ]; do
    current_time_string=`date +"%Y-%m-%d-%H%M%S"`
    log_file_name="compute_derived_fields_era5_${INIT_TIME_STRINGS[$i]}_${current_time_string}.out"

    ~/anaconda3/bin/python3.7 -u "${CODE_DIR_NAME}/compute_derived_fields.py" &> ${log_file_name} \
    --derived_field_names "most-unstable-cape-j-kg01" "most-unstable-cin-j-kg01" "surface-based-cape-j-kg01" "surface-based-cin-j-kg01" "mixed-layer-cape-j-kg01_ml-depth-metres=500.0000" "mixed-layer-cin-j-kg01_ml-depth-metres=500.0000" "lifted-index-kelvins_top-pressure-pascals=50000" "precipitable-water-kg-m02_top-pressure-pascals=5000" "wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=surface" "wind-shear-m-s01_top-pressure-pascals=85000_bottom-pressure-pascals=surface" "wind-shear-m-s01_top-pressure-pascals=20000_bottom-pressure-pascals=85000" "storm-relative-helicity-m2-s02_top-height-m-agl=3000.0000" "storm-relative-helicity-m2-s02_top-height-m-agl=1000.0000" "planetary-boundary-layer-height-m-agl" \
    --input_basic_field_dir_name="${INPUT_DIR_NAME}" \
    --model_name="ecmwf_era5" \
    --init_time_string="${INIT_TIME_STRINGS[$i]}" \
    --do_multiprocessing=1 \
    --output_dir_name="${OUTPUT_DIR_NAME}"
    
    i=$(( $i + 1 ))
done
