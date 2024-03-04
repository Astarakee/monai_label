BUNDLE="./SpleenBundle"

export PYTHONPATH=$BUNDLE

python -m monai.bundle run evaluator \
    --bundle_root "$BUNDLE" \
    --input_dir "/mnt/workspace/projects/16_MonaiLabel/datasets/Task09_Spleen" \
    --abs_path_checkpoint "$BUNDLE/models/model.pth" \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/evaluate.yaml']"