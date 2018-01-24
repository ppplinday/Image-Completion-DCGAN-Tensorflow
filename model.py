import os
import time
import math
import random
import itertools
import scipy.misc
import PIL
import numpy as np
from glob import glob
import tensorflow as tf

import config

class Generator:
	def __init__(self, depths=[512, 256, 128, 64], s_size=4):
		self.depths = depths + [3]
		self.s_size = s_size
		self.reuse = False

	def __call__(self, inputs, training=False):
		inputs = tf.convert_to_tensor(inputs)
		inputs = tf.cast(inputs, tf.float32)
		with tf.variable_scope('generator', reuse=self.reuse):
			
			with tf.variable_scope('reshape'):
				outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size, 
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
				outputs = tf.nn.relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('deconv1'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('deconv2'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('deconv3'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('deconv4'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				
			with tf.variable_scope('tanh'):
				outputs = tf.tanh(outputs, name='outputs')
			self.reuse = True
			self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
			return outputs

class Discriminator:
	def __init__(self, depths=[64, 128, 256, 512]):
		self.depths = [3] + depths
		self.reuse = False

	def __call__(self, inputs, training=False):
		outputs = tf.convert_to_tensor(inputs)
		outputs = tf.cast(outputs, tf.float32)
		with tf.variable_scope('discriminator', reuse=self.reuse):
			
			with tf.variable_scope('conv1'):
				outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('conv2'):
				outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('conv3'):
				outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('conv4'):
				outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('classify'):
				batch_size = outputs.get_shape()[0].value
				reshape = tf.reshape(outputs, [-1, 8192])
				outputs = tf.layers.dense(reshape, 1, name='outputs')
				
		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
		return tf.nn.sigmoid(outputs), outputs
'''
(100, 100)
(100, 4, 4, 1024)
(100, 8, 8, 512)
(100, 16, 16, 256)
(100, 32, 32, 128)
(100, 64, 64, 3)
xxxxxx
(100, 64, 64, 3)
(100, 32, 32, 64)
(100, 16, 16, 128)
(100, 8, 8, 256)
(100, 4, 4, 512)
(100, 1)
'''

# a = Generator()
# import numpy as np
# t = np.array(np.zeros(10000))
# tt = t.reshape([100, 100])
# print('xxx')
# print(tt.shape)
# ttt = a(tt)
# b = Discriminator()
# b(ttt)


def imread(path):
	return scipy.misc.imread(path, mode='RGB').astype(np.float)	

def data_index(data_dir, shuffle=False):
	list = os.listdir(data_dir)
	lists = []
	for i in range(0, len(list)):
		imgName = os.path.basename(list[i])
		if (os.path.splitext(imgName)[1] != ".jpg"): continue
		lists.append(list[i])

	if shuffle == True:
		random.shuffle(lists)

	return lists

def read_batch(lists):
	batch = [imread('image/' + a) for a in lists]
	batch =np.array(batch).astype(np.float32)

	# normalization
	batch = batch / 127.5 - 1.

	return batch

# merge picture
def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
	for id, image in enumerate(images):
		i = id // size[0]
		j = id % size[1]
		img[i * h: (i + 1) * h, j * w: (j + 1) * w, :] = image
	return img

# save merge picture
def save_images(images, size, image_path):
	images = (images + 1.) / 2.
	img = merge(images, size)
	return scipy.misc.imsave(image_path, (255*img).astype(np.uint8))

class DCGAN:
	def __init__(self, sess):
		self.sess = sess
		self.batch_size = config.batch_size
		self.z_dim = 100
		self.g = Generator()
		self.d = Discriminator()
		self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

		self.sample_size = 64
		self.model_name = "DCGAN.model"

		self.lam = config.lam

		self.build()

	def build(self):
		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.images = tf.placeholder(tf.float32, [None, 64, 64, 3], name='real_images')
		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
		self.z_sum = tf.summary.histogram("z", self.z)

		# DCGAN
		self.G = self.g(self.z, training=self.is_training)
		self.D, self.D_logits = self.d(self.images, training=self.is_training)
		self.D_, self.D_logits_ = self.d(self.G, training=self.is_training)

		self.d_sum = tf.summary.histogram("d", self.D)
		self.d__sum = tf.summary.histogram("d_", self.D_)
		self.G_sum = tf.summary.image("G", self.G)

		self.d_loss_real = tf.reduce_mean(
			 tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
			 										 labels=tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(
			 tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
			 										 labels=tf.zeros_like(self.D_)))
		self.g_loss = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
			 										labels=tf.ones_like(self.D_)))
		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
		self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
		self.g_vars = [var for var in t_vars if 'generator' in var.name]
		# for var in self.d_vars:
		# 	print(var)
		# print('next!!!!!!')
		# for var in self.g_vars:
		# 	print(var)
		# print('allll')
		# for var in t_vars:
		# 	print(var)
		
		self.checkpoint_dir = config.checkpoint_dir
		self.saver = tf.train.Saver(max_to_keep=1)

		# for completion
		self.lowres_G = tf.reduce_mean(tf.reshape(self.G,
			[self.batch_size, 8, 8, 8, 8, 3]), [2, 4])
		self.lowres_images = tf.reduce_mean(tf.reshape(self.images,
			[self.batch_size, 8, 8, 8, 8, 3]), [2, 4])

		self.mask = tf.placeholder(tf.float32, [None, 64, 64, 3], name='mask')
		self.lowers_mask = tf.placeholder(tf.float32, [None, 8, 8, 3], name='lowres_mask')
		self.contextual_loss = tf.reduce_sum(
			tf.contrib.layers.flatten(
				tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)
		self.contextual_loss += tf.reduce_sum(
			tf.contrib.layers.flatten(
				tf.abs(tf.multiply(self.lowers_mask, self.lowers_G) - tf.multiply(self.lowers_mask, self.lowers_images))), 1)
		self.perceptual_loss = self.g_loss
		self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss
		self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)


	def train_dcgan(self):
		data = data_index(config.dataset)

		d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

		# for sample picture
		sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
		sample_files = data[0:self.sample_size]
		sample_image = read_batch(sample_files)

		start_time = time.time()

		# load check point
		if self.load(self.checkpoint_dir):
			print('load the checkpoint!')
		else:
			print('cannot load the checkpoint and init all the varibale')

		# begin to train
		count = 1
		for epoch in range(config.epoch):
			batch_idx = len(data) // self.batch_size
			
			for idx in range(0, batch_idx):

				# mini_batch
				batch_files = data[idx * self.batch_size : (idx + 1) * self.batch_size]
				batch_image = read_batch(batch_files)
				batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

				# update D and twice G to make sure d_loss do not go to 0
				_, summary_str = self.sess.run([d_optim, self.d_sum], 
					feed_dict={self.images: batch_image, self.z: batch_z, self.is_training: True})
				self.writer.add_summary(summary_str, count)

				_, summary_str = self.sess.run([g_optim, self.g_sum],
					feed_dict={self.z: batch_z, self.is_training: True})
				self.writer.add_summary(summary_str, count)

				_, summary_str = self.sess.run([g_optim, self.g_sum],
					feed_dict={self.z: batch_z, self.is_training: True})
				self.writer.add_summary(summary_str, count)

				err_d_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
				err_d_real = self.d_loss_real.eval({self.images: batch_image, self.is_training: False})
				err_g = self.g_loss.eval({self.z: batch_z, self.is_training: False})

				count += 1
				print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
					epoch, idx, batch_idx, time.time() - start_time, err_d_fake + err_d_real, err_g))

				if count % 100 == 1:
					samples, d_loss, g_loss = self.sess.run(
						[self.G, self.d_loss, self.g_loss],
						feed_dict={self.z: sample_z, self.images: sample_image, self.is_training: False}
					)
					save_images(samples, [8, 8], './samples/train_{:02d}_{:04d}.png'.format(epoch, count-1))
					print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

				if count % 500 == 0:
					self.save(self.checkpoint_dir)
					print("Save the checkpoint for the count: {:4d}".format(count))

	def completion(self):
		if not os.path.exists(config.completion_dir):
			os.makedirs(config.completion_dir)
		p = os.path.join(config.completion_dir, 'hats_imgs')
		if not os.path.exists(p):
			os.makedirs(p)
		p = os.path.join(config.completion_dir, 'completed')
		if not os.path.exists(p):
			os.makedirs(p)
		p = os.path.join(config.completion_dir, 'logs')
		if not os.path.exists(p):
			os.makedirs(p)

		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		isloaded = self.load(self.checkpoint_dir)
		assert(isLoaded)

		num_img = data_index(config.dataset)
		batch_idxs = int(np.ceil(num_img/self.batch_size))
		lowres_mask = np.zeros([8, 8, 3])

		# center mask
		mask = np.ones([64, 64, 3])
		l = int(64 * config.centerScale)
		u = int(64 * (1. - config.centerScale))
		mask[l:u, l:u, :] = 0.0

		for idx in xrange(0, batch_idxs):
			l = idx * self.batch_size
			u = min((idx + 1) * self.batch_size, num_img)
			batch_size_z = u - l
			batch_files = config.img[l : u]
			batch = read_batch(batch_files)
			if batchSz < self.batch_size:
				padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
				batch = np.pad(batch, padSz, 'constant')
				batch = batch.astype(np.float32)

			z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

			nRows = np.ceil(batch_size_z / 8)
			nCols = min(8, batch_size_z)
			save_images(batch_images[:batch_size_z,:,:,:], [nRows, nCols],
				os.path.join(config.completion_dir, 'before.png'))
			masked_images = np.multiply(batch_images, mask)
			save_images(masked_images[:batch_size_z,:,:,:], [nRows, nCols],
				os.path.join(config.completion_dir, 'masked.png'))

			# for Adam
			m = 0
			v = 0

			for img in range(batch_size_z):
				with open(os.path.join(config.completion_dir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
					f.write('iter loss ' + ' '.join(['z{}'.format(zi) for zi in range(self.z_dim)]) + '\n')

			for i in xrange(config.nIter):
				loss, g, G_imgs, lowres_G_imgs = self.sess.run(
					[self.complete_loss, self.grad_complete_loss, self.G, self.lowres_G],
					feed_dict={
						self.z: z,
						self.mask: mask,
						self.lowres_mask: lowres_mask,
						self.images: batch,
						self.is_training: False
					})

				for img in range(batch_size_z):
					with open(os.path.join(config.completion_dir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
						f.write('{} {} '.format(i, loss[img]).encode())
						np.savetxt(f, z[img:img+1])

				if i % 50 == 0:
					print(i, np.mean(loss[0:batch_size_z]))
					imgName = os.path.join(config.completion_dir, 'hats_imgs/{:04d}.png'.format(i))
					nRows = np.ceil(batchSz/8)
					nCols = min(8, batchSz)
					save_images(G_imgs[:batch_size_z,:,:,:], [nRows,nCols], imgName)

					inv_masked_hat_images = np.multiply(G_imgs, 1.0 - mask)
					completed = masked_images + inv_masked_hat_images

					imgName = os.path.join(config.completion_dir, 'completed/{:04d}.png'.format(i))
					save_images(completed[:batch_size_z,:,:,:], [nRows,nCols], imgName)

				# optim Adam
				m_prev = np.copy(m)
				v_prev = np.copy(v)
				m = config.c_beta1 * m_prev + (1 - config.c_beta1) * g[0]
				v = config.c_beta2 * v_prev + (1 - config.c_beta2) * np.multiply(g[0], g[0])
				m_hat = m / (1 - config.c_beta1 ** (i + 1))
				v_hat = v / (1 - config.c_beta2 ** (i + 1))
				z += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
				z = np.clip(z, -1, 1)


	def save(self, checkpoint_dir):
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name))

	def load(self, checkpoint_dir):
		print('Being to load the checkpoint')

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			return True
		else:
			return False

# lists = data_index()
# print(lists)
# batch = read_batch(lists)
# print(batch.shape)
# print(batch[2])

# sess = tf.Session()
# a = DCGAN(sess)
# a.build()
# print('okok')