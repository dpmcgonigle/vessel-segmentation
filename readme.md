# Stage 2 Max

Instead of combining (original image, stage 1 prediction map, edge map of stage 1 prediction map) as the input to stage 2, I'm trying to simply take the maximum value along axis 0 of the 3-channel image to see how that affects training and evaluation.
