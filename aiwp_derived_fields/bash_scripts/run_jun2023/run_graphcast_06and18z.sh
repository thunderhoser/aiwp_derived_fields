#!/bin/bash

CODE_DIR_NAME="/home/ralager/aiwp_derived_fields/aiwp_derived_fields/scripts"
INPUT_DIR_NAME="/mnt/mlnas01/ai-models/GRAP_v100"
OUTPUT_DIR_NAME="/mnt/mlnas01/ralager/aiwp_derived_fields"

INIT_TIME_STRINGS=("2023-06-01-06" "2023-06-01-18" "2023-06-02-06" "2023-06-02-18" "2023-06-03-06" "2023-06-03-18" "2023-06-04-06" "2023-06-04-18" "2023-06-05-06" "2023-06-05-18" "2023-06-06-06" "2023-06-06-18" "2023-06-07-06" "2023-06-07-18" "2023-06-08-06" "2023-06-08-18" "2023-06-09-06" "2023-06-09-18" "2023-06-10-06" "2023-06-10-18" "2023-06-11-06" "2023-06-11-18" "2023-06-12-06" "2023-06-12-18" "2023-06-13-06" "2023-06-13-18" "2023-06-14-06" "2023-06-14-18" "2023-06-15-06" "2023-06-15-18" "2023-06-16-06" "2023-06-16-18" "2023-06-17-06" "2023-06-17-18" "2023-06-18-06" "2023-06-18-18" "2023-06-19-06" "2023-06-19-18" "2023-06-20-06" "2023-06-20-18" "2023-06-21-06" "2023-06-21-18" "2023-06-22-06" "2023-06-22-18" "2023-06-23-06" "2023-06-23-18" "2023-06-24-06" "2023-06-24-18" "2023-06-25-06" "2023-06-25-18" "2023-06-26-06" "2023-06-26-18" "2023-06-27-06" "2023-06-27-18" "2023-06-28-06" "2023-06-28-18" "2023-06-29-06" "2023-06-29-18" "2023-06-30-06" "2023-06-30-18" "2023-06-31-06" "2023-06-31-18")

i=0

while [ $i -lt ${#INIT_TIME_STRINGS[@]} ]; do
    current_time_string=`date +"%Y-%m-%d-%H%M%S"`
    log_file_name="compute_derived_fields_graphcast_${INIT_TIME_STRINGS[$i]}_${current_time_string}.out"

    python3 -u "${CODE_DIR_NAME}/compute_derived_fields.py" &> ${log_file_name} \
    --derived_field_names "most-unstable-cape-j-kg01" "most-unstable-cin-j-kg01" "surface-based-cape-j-kg01" "surface-based-cin-j-kg01" "mixed-layer-cape-j-kg01_ml-depth-metres=500.0000" "mixed-layer-cin-j-kg01_ml-depth-metres=500.0000" "lifted-index-kelvins_top-pressure-pascals=50000" "precipitable-water-kg-m02_top-pressure-pascals=5000" "wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=surface" "wind-shear-m-s01_top-pressure-pascals=85000_bottom-pressure-pascals=surface" "wind-shear-m-s01_top-pressure-pascals=20000_bottom-pressure-pascals=85000" "storm-relative-helicity-m2-s02_top-height-m-agl=3000.0000" "storm-relative-helicity-m2-s02_top-height-m-agl=1000.0000" "planetary-boundary-layer-height-m-agl" \
    --input_basic_field_dir_name="${INPUT_DIR_NAME}" \
    --model_name="GRAP_v100" \
    --init_time_string="${INIT_TIME_STRINGS[$i]}" \
    --do_multiprocessing=1 \
    --output_dir_name="${OUTPUT_DIR_NAME}"
    
    i=$(( $i + 1 ))
done
