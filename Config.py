import tensorflow as tf

class Config:
	e_epochs = 300
	a_epochs = 3000
	e_dim = 300
	a_dim = 100
	act_func = tf.nn.relu
	alpha = 0.1
	beta = 0.3
	gamma = 1.0  # margin based loss
	theta = 0.7
	k = 125  # number of negative samples for each positive one
	seed = 3  # 30% of seeds
	rate = 0.001
