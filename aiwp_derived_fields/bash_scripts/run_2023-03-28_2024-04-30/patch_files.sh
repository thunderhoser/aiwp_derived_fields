#!/bin/bash

CODE_DIR_NAME="/home/ralager/aiwp_derived_fields/aiwp_derived_fields/scripts"
OUTPUT_DIR_NAME="/mnt/mlnas01/ralager/aiwp_derived_fields"

INIT_TIME_STRINGS=("2023-04-29-12" "2023-04-29-18" "2023-04-30-00" "2023-04-30-06" "2023-04-30-12" "2023-04-30-18" "2023-03-31-12" "2023-03-28-06" "2023-03-31-12" "2023-04-01-12")
ABBREV_MODEL_NAMES=("GRAP_v100" "GRAP_v100" "GRAP_v100" "GRAP_v100" "GRAP_v100" "GRAP_v100" "FOUR_v200" "PANG_v100" "PANG_v100" "PANG_v100")
NICE_MODEL_NAMES=("graphcast" "graphcast" "graphcast" "graphcast" "graphcast" "graphcast" "fourcastnet" "pangu" "pangu" "pangu")

i=0

while [ $i -lt ${#INIT_TIME_STRINGS[@]} ]; do
    current_time_string=`date +"%Y-%m-%d-%H%M%S"`
    log_file_name="compute_derived_fields_${NICE_MODEL_NAMES[$i]}_${INIT_TIME_STRINGS[$i]}_${current_time_string}.out"

    python3 -u "${CODE_DIR_NAME}/compute_derived_fields.py" &> ${log_file_name} \
    --derived_field_names "most-unstable-cape-j-kg01" "most-unstable-cin-j-kg01" "surface-based-cape-j-kg01" "surface-based-cin-j-kg01" "mixed-layer-cape-j-kg01_ml-depth-metres=500.0000" "mixed-layer-cin-j-kg01_ml-depth-metres=500.0000" "lifted-index-kelvins_top-pressure-pascals=50000" "precipitable-water-kg-m02_top-pressure-pascals=5000" "wind-shear-m-s01_top-pressure-pascals=50000_bottom-pressure-pascals=surface" "wind-shear-m-s01_top-pressure-pascals=85000_bottom-pressure-pascals=surface" "wind-shear-m-s01_top-pressure-pascals=20000_bottom-pressure-pascals=85000" "storm-relative-helicity-m2-s02_top-height-m-agl=3000.0000" "storm-relative-helicity-m2-s02_top-height-m-agl=1000.0000" "planetary-boundary-layer-height-m-agl" \
    --input_basic_field_dir_name="${INPUT_DIR_NAME}/${ABBREV_MODEL_NAMES[$i]}" \
    --model_name="${ABBREV_MODEL_NAMES[$i]}" \
    --init_time_string="${INIT_TIME_STRINGS[$i]}" \
    --do_multiprocessing=1 \
    --output_dir_name="${OUTPUT_DIR_NAME}"
    
    i=$(( $i + 1 ))
done
