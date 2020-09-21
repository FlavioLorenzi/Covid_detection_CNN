import Augmentor

p = Augmentor.Pipeline("dataset/train/covid")
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.

# Add operations to the pipeline as normal:
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.sample(84)
p.flip_left_right(probability=0.5)
p.sample(84)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.sample(84)
p.flip_top_bottom(probability=0.5)
p.sample(84)


#ADD IN THIS WAY 420 samples for training COVID