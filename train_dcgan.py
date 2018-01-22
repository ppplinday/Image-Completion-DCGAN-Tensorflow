import os
import scipy.misc
import numpy as np
import config
from model import DCGAN
import tensorflow as tf

if not os.path.exists(config.checkpoint_dir):
	os.makedirs(config.checkpoint_dir)
if not os.path.exists(config.sample_dir):
	os.makedirs(config.sample_dir)

with tf.Session() as sess:
	dcgan = DCGAN(sess)

	dcgan.train_dcgan()