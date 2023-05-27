import os
import shutil

import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("datasets/Classification", output="datasets/Classification_split",
    seed=420, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values

# Split val/test with a fixed number of items, e.g. `(100, 100)`, for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
# Set 3 values, e.g. `(300, 100, 100)`, to limit the number of training values.
# splitfolders.fixed("input_folder", output="output",
#     seed=1337, fixed=(100, 100), oversample=False, group_prefix=None, move=False) # default values