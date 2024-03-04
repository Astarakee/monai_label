BUNDLE="./SpleenBundle"

export PYTHONPATH=$BUNDLE

python -m monai.bundle run learner \
    --bundle_root "$BUNDLE" \
    --input_dir "/mnt/workspace/projects/16_MonaiLabel/datasets/Task09_Spleen" \
    --abs_path_checkpoint "$BUNDLE/models/model.pth" \
    --fig_save_path: "$BUNDLE/models/plots.png" \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/train.yaml']"