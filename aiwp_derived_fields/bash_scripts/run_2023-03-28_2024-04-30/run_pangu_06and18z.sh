#!/bin/bash

CODE_DIR_NAME="/home/ralager/aiwp_derived_fields/aiwp_derived_fields/scripts"
INPUT_DIR_NAME="/mnt/mlnas01/ai-models/PANG_v100"
OUTPUT_DIR_NAME="/mnt/mlnas01/ralager/aiwp_derived_fields"

INIT_TIME_STRINGS=("2023-03-28-06" "2023-03-28-18" "2023-03-29-06" "2023-03-29-18" "2023-03-30-06" "2023-03-30-18" "2023-03-31-06" "2023-03-31-18" "2023-04-01-06" "2023-04-01-18" "2023-04-02-06" "2023-04-02-18" "2023-04-03-06" "2023-04-03-18" "2023-04-04-06" "2023-04-04-18" "2023-04-05-06" "2023-04-05-18" "2023-04-06-06" "2023-04-06-18" "2023-04-07-06" "2023-04-07-18" "2023-04-08-06" "2023-04-08-18" "2023-04-09-06" "2023-04-09-18" "2023-04-10-06" "2023-04-10-18" "2023-04-11-06" "2023-04-11-18" "2023-04-12-06" "2023-04-12-18" "2023-04-13-06" "2023-04-13-18" "2023-04-14-06" "2023-04-14-18" "2023-04-15-06" "2023-04-15-18" "2023-04-16-06" "2023-04-16-18" "2023-04-17-06" "2023-04-17-18" "2023-04-18-06" "2023-04-18-18" "2023-04-19-06" "2023-04-19-18" "2023-04-20-06" "2023-04-20-18" "2023-04-21-06" "2023-04-21-18" "2023-04-22-06" "2023-04-22-18" "2023-04-23-06" "2023-04-23-18" "2023-04-24-06" "2023-04-24-18" "2023-04-25-06" "2023-04-25-18" "2023-04-26-06" "2023-04-26-18" "2023-04-27-06" "2023-04-27-18" "2023-04-28-06" "2023-04-28-18" "2023-04-29-06" "2023-04-29-18" "2023-04-30-06" "2023-04-30-18")

i=0

while [ $i -lt ${#INIT_TIME_STRINGS[@]} ]; do
    current_time_string=`date +"%Y-%m-%d-%H%M%S"`
    log_file_name="compute_derived_fields_pangu_${INIT_TIME_STRINGS[$i]}_${current_time_string}.out"

    python3 -u "${CODE_DIR_NAME}/compute_derived_fields.py" &> ${log_file_name} \
    --derived_field_names "most-unstable-cape-j-kg01" "most-unstable-cin-j-kg01" "surface-based-cape-j-kg01" "surface-based-cin-j-kg01" "mixed-layer-cape-j-kg01_ml-depth-metres=500.0000" "mixed-layer-cin-j-kg01_ml-depth-metres=500.0000" "lifted-index-kelvins_top-pressure-pascals=50000" "precipitable-water-kg-m02_top-pressure-pascals=5000" "wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=surface" "wind-shear-m-s01_top-pressure-pascals=85000_bottom-pressure-pascals=surface" "wind-shear-m-s01_top-pressure-pascals=20000_bottom-pressure-pascals=85000" "storm-relative-helicity-m2-s02_top-height-m-agl=3000.0000" "storm-relative-helicity-m2-s02_top-height-m-agl=1000.0000" "planetary-boundary-layer-height-m-agl" \
    --input_basic_field_dir_name="${INPUT_DIR_NAME}" \
    --model_name="PANG_v100" \
    --init_time_string="${INIT_TIME_STRINGS[$i]}" \
    --do_multiprocessing=1 \
    --output_dir_name="${OUTPUT_DIR_NAME}"
    
    i=$(( $i + 1 ))
done
