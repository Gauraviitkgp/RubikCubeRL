import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import environment as Env

try:
	xrange = xrange
except:
	xrange = range


env = Env.RubikCube()


GAMMA				= 0.99
TOTAL_EPISODES		= 100000 #Set total number of episodes to train agent on.
MAX_EP_LENGTH		= 7
UPDATE_FREQUENCY	= 5
STATE_SIZE			= 54*6
ACTION_SIZE			= 12
LR_RATE 			= 1e-3
HIDDEN_LYR_SIZE     = 8
TENSORBOARD_PATH    = 'tensorboard_training/4_L4_mscr_e-2LR_CContRewards'
CHECKPOINT_PATH     = "saved_training/saved_model"
CHECKPOINT_FREQ     = 5000
MAX_MODELS_TO_SAVE  = 5

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
global_summary = tf.compat.v1.summary.FileWriter(TENSORBOARD_PATH)

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		running_add = running_add * GAMMA + r[t]
		discounted_r[t] = running_add
	return discounted_r

class agent():
	def __init__(self, lr, s_size,a_size,h_size):
		#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
		#Input State 
		self.state_in= tf.compat.v1.placeholder(shape=[None,s_size],dtype=tf.float32)
		hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
		self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
		self.chosen_action = tf.argmax(self.output,1)

		#NN is [s_size->(nn.relu) hidden_layer_size->(nn.softmax) action_size-> max_probablity]
		#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
		#to compute the loss, and use it to update the network.
		#Placeholders for reward and action
		self.reward_holder = tf.compat.v1.placeholder(shape=[None],dtype=tf.float32)
		self.action_holder = tf.compat.v1.placeholder(shape=[None],dtype=tf.int32)
		
		#Outputs can be sometimes 2D, thus the next two lines establishes indexing of outputs. First 0..m*n indexing is done for mxn size of outputs
		#Then an for each outputs in self.output indexing is done.

		self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
		self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

		# Loss = mean(log(outputs)*rewards) #Reduce mean computes mean of given set of vecotrs
		self.loss = -tf.reduce_mean(tf.math.log(self.responsible_outputs)*self.reward_holder)
		
		tvars = tf.compat.v1.trainable_variables()
		#Gradient Holder
		self.gradient_holders = []
		for idx,var in enumerate(tvars):
			#Placeholder for every trainable variable
			placeholder = tf.compat.v1.placeholder(tf.float32,name=str(idx)+'_holder')
			self.gradient_holders.append(placeholder)
		
		#Computes gradient of loss wrt tvars
		self.gradients = tf.gradients(self.loss,tvars)
		
		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
		self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


tf.compat.v1.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=LR_RATE,s_size=STATE_SIZE,a_size=ACTION_SIZE,h_size=HIDDEN_LYR_SIZE) #Load the agent.


init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver(max_to_keep = MAX_MODELS_TO_SAVE)

# Launch the tensorflow graph
with tf.compat.v1.Session() as sess:
	sess.run(init)
	i = 0
	total_reward = []
	total_length = []
		
	gradBuffer = sess.run(tf.compat.v1.trainable_variables())
	for ix,grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0
	
	while i < TOTAL_EPISODES:
		s = env.reset(4) #Maximum 10 Steps
		running_reward = 0
		ep_history = []
		for j in range(MAX_EP_LENGTH):
			#Probabilistically pick an action given our network outputs.
			a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
			a = np.random.choice(a_dist[0],p=a_dist[0])
			a = np.argmax(a_dist == a)

			s1,r,d = env.step(a) #Get our reward for taking an action given a bandit.
			ep_history.append([s,a,r,s1])
			s = s1
			running_reward += r

			if j==MAX_EP_LENGTH-1:
				d=True
			# print(r)
			if d == True:
				#Update the network.
				ep_history = np.array(ep_history)
				ep_history[:,2] = discount_rewards(ep_history[:,2])
				feed_dict={myAgent.reward_holder:ep_history[:,2],
						myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
				grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
				for idx,grad in enumerate(grads):
					gradBuffer[idx] += grad

				if i % UPDATE_FREQUENCY == 0 and i != 0:
					feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
					_ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
					for ix,grad in enumerate(gradBuffer):
						gradBuffer[ix] = grad * 0
				
				total_reward.append(running_reward)
				total_length.append(j)
				break

			#Update our running tally of scores.
		summary = tf.compat.v1.Summary()
		summary.value.add(tag='Perf/total_reward', simple_value = running_reward)
		#summary.value.add(tag='Data/Gradients', simple_value = grads)
		global_summary.add_summary(summary, int(i))
		global_summary.flush()
		if i % 1000 == 0:
			print(i,":",np.mean(total_reward[-100:]))
		if i % CHECKPOINT_FREQ == 0:
			saver.save(sess, CHECKPOINT_PATH, global_step = i)
		i += 1
