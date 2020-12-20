import sys

from absl import logging
from ferminet.utils import system
from ferminet import train
from ferminet import networks
import tensorflow.compat.v1 as tf
import numpy as np
from ferminet.qmc import H5Writer

# Optional, for also printing training progress to STDOUT
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)


mcmc_config = train.MCMCConfig(
      move_width=0.05,
)

optim_config = train.OptimConfig(
	iterations=50000,	
	learning_rate_delay=5000,
)

# Define Neon Atom
molecule, spins = system.atom("Ne")
print (spins)


def fix_coords(batch_numpy, fixed_coords_up, fixed_coords_down, spins):
	spin_up = spins[0]
	spin_down = spins[1]

	for i, coords in enumerate(fixed_coords_up):
		batch_numpy[:, 3*i:3*(i+1)] = coords
	for i, coords in enumerate(fixed_coords_down):
		batch_numpy[:, 3*(i+spin_up):3*(i+spin_up+1)] = coords
	
	mask = np.ones(3*(spins[0] + spins[1]))
	for i, coords in enumerate(fixed_coords_up):
		mask[3*i:3*(i+1)] = 0
	for i, coords in enumerate(fixed_coords_down):
		mask[3*(i+spin_up):3*(i+spin_up+1)] = 0

	batch = tf.convert_to_tensor(batch_numpy, dtype = (tf.float32).base_dtype)	
	mask = tf.constant(mask, dtype = (tf.float32).base_dtype )

	return batch,  mask


def mcmc_step(batch, psi, network, mask):
	
	new_batch = batch + tf.random.normal(shape = batch.shape, stddev = mcmc_step_std)*mask
	new_psi = network(new_batch)[0]
	pmove = tf.squeeze(2*(new_psi-psi))
	pacc = tf.log(tf.random_uniform(shape = batch.shape.as_list()[:1]))
	decision = tf.less(pacc, pmove)
	with tf.control_dependencies([decision]):
		new_batch = tf.where(decision, new_batch, batch)
		new_psi = tf.where(decision, new_psi, psi)
	move_acc = tf.reduce_mean(tf.cast(decision, tf.float32))
	return new_batch, new_psi, move_acc


def main(fixed_coords_up, fixed_coords_down, mcmc_step_std, init_sigma, batch_size, iterations):
	checkpoint_path = "Neon/checkpoints"

	latest = tf.train.latest_checkpoint(checkpoint_path)


	ferminet = networks.FermiNet(
		atoms = molecule,
		nelectrons = spins,
		slater_dets = 16,
		hidden_units = ((256, 32),) * 4,
		after_det    =  (1,),
		pretrain_iterations = 0,
		logdet  = True,
		envelope = True,
		residual = True,
		name = "model/det_net"
	)
	
	init_means = train.assign_electrons(molecule, spins)
	batch =  np.concatenate([ 
						np.random.normal( 
							size=(batch_size, 1),
							loc = mu,
							scale = init_sigma,
						) for mu in init_means
					], axis = -1)

	batch, mask = fix_coords(batch, fixed_coords_up, fixed_coords_down, spins)

	print ("mask: ", mask)
	psi = ferminet(batch)[0]

	saver = tf.train.Saver(max_to_keep=10, save_relative_paths=True)


	with tf.Session() as sess:
		saver.restore(sess, latest)
		sess.run(psi)

		psi = sess.run(psi)

	#	for itr in range(iterations):
	#		
	#		new_batch = batch + tf.random.normal(shape = batch.shape, stddev = mcmc_step_std)*mask
	#		update_psi = ferminet(new_batch)[0]
	#		new_psi = sess.run(update_psi)
	#		pmove = tf.squeeze(2*(new_psi-psi))
	#		pacc = tf.log(tf.random_uniform(shape = batch.shape.as_list()[:1]))
	#		decision = tf.less(pacc, pmove)
	#		with tf.control_dependencies([decision]):
	#			new_batch = tf.where(decision, new_batch, batch)
	#			new_psi = tf.where(decision, new_psi, psi)
	#		move_acc = tf.reduce_mean(tf.cast(decision, tf.float32))
	#		batch = new_batch
	#		psi = new_psi
	#		print (sess.run(psi), sess.run(move_acc))

		print (type(batch.shape))
		print ("Burn in")
		for itr in range(20):
			new_batch, new_psi, move_acc = sess.run(mcmc_step(batch, psi, ferminet, mask))		
			batch = tf.constant(new_batch , (tf.float32).base_dtype)
			psi = new_psi
			print (itr, " pacc: ", move_acc)
		print ("Burn in done")
		h5_schema  = {"walkers": batch.shape.as_list() }
		with H5Writer(name="neon_data_z.h5", schema = h5_schema, directory = "./") as h5_writer:
			for itr in range(iterations):
				for _ in range(2):
					new_batch, new_psi, move_acc = sess.run(mcmc_step(batch, psi, ferminet, mask))		
					batch = tf.constant(new_batch , (tf.float32).base_dtype)
					psi = new_psi
				print (itr, " pacc: ",move_acc)
				out = {"walkers": new_batch}
				h5_writer.write(itr, out)

#fixed_coords_up = [[0.0,0.0,0.0]]
fixed_coords_up = []
fixed_coords_down = []
mcmc_step_std = 0.02
init_sigma = 0.05
batch_size = 1024
iterations = 100

main(fixed_coords_up, fixed_coords_down, mcmc_step_std, init_sigma, batch_size, iterations)
   



