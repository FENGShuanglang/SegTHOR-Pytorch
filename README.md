# SegTHOR-Pytorch
## this is a project on the segmentation of SegTHOR dataset
 
 we transformed the 3D CT data into 2.5D to train our network. Specifically, three adjacent slices were stacked to form a 3-channel input data, and the network was responsible for predicting the segmentation of the middle slice. We also applied multiple augmentations, including left-right flipping, up-down flipping, rotations of -15 degrees to 15 degrees and contrast normalization.
