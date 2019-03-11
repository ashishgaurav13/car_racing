import gym
import numpy as np
from utils import obs2img
from utils import downsample as ds
from utils import ExperienceBuffer
from torchsummary import summary
from torch import nn, optim
import torch
import glob, datetime, sys
from utils import Tee
from constants import *
from utils import key_press, key_release

# if LOGGING:
# 	log_file_name = datetime.datetime.now().strftime("log_%Y_%m_%d_%H_%M_%S.txt")
# 	log_file = open(log_file_name, "w")
# 	backup = sys.stdout
# 	sys.stdout = Tee(sys.stdout, log_file)

class QNetwork(nn.Module):

	def __init__(self, env, lr, eps, decay):
		super().__init__()
		self.env = env
		self.fc = nn.Sequential(
			nn.Linear(4, 32),
			nn.ReLU(),
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, env.action_space.n)
		)
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.eps = eps
		self.decay = decay

	def forward(self, x):
		x = self.fc(x)
		return x

	def backprop(self, x, y):
		self.optimizer.zero_grad() # clear gradients
		output = self.forward(x)
		loss = self.loss(output, y)
		loss.backward()
		self.optimizer.step()
		self.eps *= self.decay
		return loss.item()

	def predict(self, x):
		if type(x) == list: x = np.array(x)
		x = torch.from_numpy(x).float()
		return self.forward(x)

	def choose_action(self, x):
		if np.random.rand() <= self.eps:
			return np.random.choice(self.env.action_space.n)
		else:
			[Q] = self.predict(np.resize(x, (1, *x.shape)))
			values, indices = torch.max(Q, 0)
			return indices.item()

	def train(self, batch):
		i, o = [], []
		for e in batch:
			state, action, r, next_state, done = e
			[Q] = self.predict([state])
			[Q_ns] = self.predict([next_state])
			target = Q
			values, indices = torch.max(Q_ns, 0)
			target[action] = r+GAMMA*values.item() if not done else r
			i.append(state)
			o.append(target)
		ii = torch.FloatTensor(i)
		oo = torch.stack(o)
		return self.backprop(ii, oo)

def play_episode(env, eb, dqn, i):
	global EXPERIENCE_BUFFER_EXPLORE, MINIBATCH_SIZE
	global STEP, STEP_MOD, SAVE_MODELS

	# (1) Reset
	obs = env.reset()

	# (2) What is the shape?
	# print(obs)

	# (3) Downsample, see image
	# print(ds(obs).shape)
	# print(ds(obs, save='test.png'))

	# (4) Assign key_press and key_release functions
	action = np.array([0.0, 0.0, 0.0])
	env.render()
	env.env.viewer.window.on_key_press = lambda k, mod: key_press(k, mod, action)
	env.env.viewer.window.on_key_release = lambda k, mod: key_release(k, mod, action)

	ep_reward = 0
	losses = []

	while True:

		env.render()

		# choose action, eps greedy and step
		# action = dqn.choose_action(obs)
		next_obs, r, done, info = env.step(action)

		# add experience tuple
		experience = (obs, action, r, next_obs, done)
		eb.add(experience)

		# train
		if len(eb) > EXPERIENCE_BUFFER_EXPLORE:
			batch = eb.sample(MINIBATCH_SIZE)
			loss = dqn.train(batch)
			losses += [loss]

		# save model
		# if SAVE_MODELS and STEP % STEP_MOD == 0:
		# 	print('Saving model: model%d.pt' % (STEP//STEP_MOD))
		# 	torch.save(dqn, 'model%d.pt' % (STEP//STEP_MOD))
		# Misc

		STEP += 1
		obs = next_obs
		ep_reward += r
		if done:
			break

	avg_loss = np.mean(losses) if len(losses) > 0 else np.nan
	print('Episode %d: total_reward = %g, avg_loss = %g, @ STEP %d' % (i,
		ep_reward, avg_loss, STEP))


# https://github.com/openai/atari-reset/blob/master/test_atari.py
env = gym.make('CarRacing-v0')
eb = ExperienceBuffer(size=EXPERIENCE_BUFFER_SIZE)

# Load latest model
# saved_models = glob.glob('model*.pt')
# if len(saved_models) > 0:
# 	nums = map(int, [item.lstrip('model').rstrip('.pt') for item in saved_models])
# 	max_num = max(nums)
# 	print('Loading: model%d.pt' % max_num)
# 	dqn = torch.load('model%d.pt' % max_num)
# 	STEP += max_num*STEP_MOD
# else:
dqn = None # QNetwork(env, lr=0.001, eps=EPSILON_START, decay=EPSILON_DECAY)
# summary(dqn, (1, 4))

# Infinite episodes
EP = 1
while True: 
	play_episode(env, eb, dqn, EP)
	EP += 1