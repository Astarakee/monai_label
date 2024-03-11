# Simple instruction (will be updated)

The model was originally trained on UPENN GBM dataset. \
The employed Dataset for the training phase contained 4 MRI sequences as separate 3D nifti files. \
However, MonaiLabel is not compatible, yet, with multichannel volumes. \
Therefore, for the inference phase, I modified the dataloader to directly load the 4D Stack volumes. \
The order of the channels is: T1, T1CE, T2, and FLAIR. \
Execution time even on CPU is in the order of a couple of seconds.



To use":

1) put the `glioma_custom` into the following directory:
`<DIR TO MONAI Label>/apps/monaibundle/model`
2) start Monai Label server:
`monailabel start_server --app apps/monaibundle --studies <Abs Path To Data> --conf models glioma_custom`
3) run 3D slicer
