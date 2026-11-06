#!/usr/bin/env bash
# Runs the benchmark sequentially: link prediction, then node retrieval on its checkpoints, then edge classification.
#
#   DATA_ROOT=/path/to/datasets GPU=0 scripts/run_benchmark.sh
#
# DATASETS, FEATURES and SEEDS can be overridden from the environment. Edit METHODS to select methods.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the directory containing the datasets}
GPU=${GPU:-0}
DATASETS=${DATASETS:-"GDELT GMCT amazon_ind amazon_off Yelp"}
FEATURES=${FEATURES:-"rand text text+image"}
SEEDS=${SEEDS:-"0,1,2"}

# backbone:wrapper. The multimodal fusion methods need two modalities, so they only run on text+image.
METHODS=(
    TGAT:concat CAWN:concat TCL:concat GraphMixer:concat DyGFormer:concat
    FreeDyG:concat TPNet:concat Liquid:concat
    DyGFormer:tst_mf
)
FUSION_METHODS=(DyGFormer:gated_fusion DyGFormer:modality_attention DyGFormer:unigraph2)

run() {
    local dataset=$1 feature=$2 model=${3%%:*} wrapper=${3##*:}
    local common=(--dataset_name "$dataset" --use_feature "$feature" --model_name "$model" --wrapper_name "$wrapper"
                  --seeds "$SEEDS" --gpu "$GPU" --data_root "$DATA_ROOT")
    python train_link_prediction.py "${common[@]}"
    python evaluate_node_retrieval.py "${common[@]}"
    python train_edge_classification.py "${common[@]}"
}

for dataset in $DATASETS; do
    for feature in $FEATURES; do
        for method in "${METHODS[@]}"; do
            run "$dataset" "$feature" "$method"
        done
        if [[ $feature == "text+image" ]]; then
            for method in "${FUSION_METHODS[@]}"; do
                run "$dataset" "$feature" "$method"
            done
        fi
    done
done
