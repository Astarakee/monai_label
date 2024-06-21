BUNDLE="./spleen_custom"

export PYTHONPATH=$BUNDLE

python -m monai.bundle run run  \
    --bundle_root "$BUNDLE" \
    --dataset_dir "/mnt/workspace/projects/16_MonaiLabel/datasets/Task09_Spleen/imagesTs" \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --config_file "$BUNDLE/configs/inference.yaml"
