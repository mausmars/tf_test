import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
# TF1 version
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

model_handle = 'https://tfhub.dev/google/tiny_video_net/tvn2/1'
model = hub.Module(model_handle)

vid_placeholder = tf.placeholder(tf.float32,
                                 shape=(batch_size * num_frames,
                                        image_size, image_size, 3))

# This generates a random video. It should be replaced by a user's video.
# video = load_video(video_path) which should return a video of the above shape.
video = np.random.rand(*vid_placeholder.shape)

predictions = model(video)
