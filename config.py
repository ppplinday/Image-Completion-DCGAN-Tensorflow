# dcgan
learning_rate = 0.0002
beta1 = 0.5
epoch = 20
batch_size = 64
dataset = 'image'
checkpoint_dir = 'checkpoint'
sample_dir = 'samples'

# completion
lam = 0.1
centerScale = 0.25 # should <= 0.5
completion_dir = 'completions'
nIter = 1000
c_beta1 = 0.9
c_beta2 = 0.999
lr = 0.01
eps = 1e-8
uncompletion_image_dir = ''