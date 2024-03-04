BUNDLE="./SpleenBundle"

export PYTHONPATH=$BUNDLE

python -m monai.bundle run inferer \
    --bundle_root "$BUNDLE" \
    --test_dir "/mnt/workspace/projects/16_MonaiLabel/datasets/Task09_Spleen/imagesTs/" \
    --pred_dir "/mnt/workspace/projects/16_MonaiLabel/datasets/Task09_Spleen/predTs/" \
    --abs_path_checkpoint "$BUNDLE/models/model.pth" \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/inference.yaml']"