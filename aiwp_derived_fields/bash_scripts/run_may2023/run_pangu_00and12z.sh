#!/bin/bash

CODE_DIR_NAME="/home/ralager/aiwp_derived_fields/aiwp_derived_fields/scripts"
INPUT_DIR_NAME="/mnt/mlnas01/ai-models/PANG_v100"
OUTPUT_DIR_NAME="/mnt/mlnas01/ralager/aiwp_derived_fields"

INIT_TIME_STRINGS=("2023-05-01-00" "2023-05-01-12" "2023-05-02-00" "2023-05-02-12" "2023-05-03-00" "2023-05-03-12" "2023-05-04-00" "2023-05-04-12" "2023-05-05-00" "2023-05-05-12" "2023-05-06-00" "2023-05-06-12" "2023-05-07-00" "2023-05-07-12" "2023-05-08-00" "2023-05-08-12" "2023-05-09-00" "2023-05-09-12" "2023-05-10-00" "2023-05-10-12" "2023-05-11-00" "2023-05-11-12" "2023-05-12-00" "2023-05-12-12" "2023-05-13-00" "2023-05-13-12" "2023-05-14-00" "2023-05-14-12" "2023-05-15-00" "2023-05-15-12" "2023-05-16-00" "2023-05-16-12" "2023-05-17-00" "2023-05-17-12" "2023-05-18-00" "2023-05-18-12" "2023-05-19-00" "2023-05-19-12" "2023-05-20-00" "2023-05-20-12" "2023-05-21-00" "2023-05-21-12" "2023-05-22-00" "2023-05-22-12" "2023-05-23-00" "2023-05-23-12" "2023-05-24-00" "2023-05-24-12" "2023-05-25-00" "2023-05-25-12" "2023-05-26-00" "2023-05-26-12" "2023-05-27-00" "2023-05-27-12" "2023-05-28-00" "2023-05-28-12" "2023-05-29-00" "2023-05-29-12" "2023-05-30-00" "2023-05-30-12" "2023-05-31-00" "2023-05-31-12")

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
