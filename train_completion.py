import os
import scipy.misc
import numpy as np
import config
from model import DCGAN
import tensorflow as tf

with tf.Session() as sess:
	dcgan = DCGAN(sess)

	dcgan.train_completion()

	print('finish to complete images')