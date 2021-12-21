from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
seed = ['1992']
coeff = [
	'0.0',
	'0.1',
	'0.2',
	'0.3',
	'0.4',
	'0.5',
	'0.6',
	'0.7',
	'0.8',
	'0.9',
	'1.0',
	'2.0',
]
float_coeff = [float(f) for f in coeff]
acc = 0
splitRMSE = []
for s in seed:
	for w in coeff:
		acc = 0
		c = 0
		fname = 'data/digital_music/Results/latent_factor_100/mode_4/user_' + w + '_item_' + w + '/seed_' + s + '/test_error_bounded.csv'
		with open(fname) as file:
			for i, line in enumerate(file):
				if i == 0:
					continue
				line = line[:-1]
				_, _, r = line.split(',')
				acc += float(r)
				c += 1
			splitRMSE.append(acc / c)
print("===========================")
print("DIGITAL MUSIC Overall RMSE Over Autoencoder Weights")
print(splitRMSE)
print("===========================")
plt.figure(1)
plt.title('NAECF RMSE on Amazon Digital Music Dataset')
plt.plot(float_coeff, splitRMSE,'-o')
plt.xlabel('Autoencoder Weight (w)')
plt.ylabel('Root Mean Square Error Over Ratings')
acc = 0
splitRMSE = []
for s in seed:
	for w in coeff:
		acc = 0
		c = 0
		fname = 'data/instant_video/Results/latent_factor_100/mode_4/user_' + w + '_item_' + w + '/seed_' + s + '/test_error_bounded.csv'
		with open(fname) as file:
			for i, line in enumerate(file):
				if i == 0:
					continue
				line = line[:-1]
				_, _, r = line.split(',')
				acc += float(r)
				c += 1
			splitRMSE.append(acc / c)
print("===========================")
print("INSTANT VIDEO Overall RMSE Over Autoencoder Weights")
print(splitRMSE)
print("===========================")
plt.figure(2)
plt.title('NAECF RMSE on Amazon Instant Video Dataset')
plt.plot(float_coeff, splitRMSE,'-o')
plt.xlabel('Autoencoder Weight (w)')
plt.ylabel('Root Mean Square Error Over Ratings')
acc = 0
uRMSE = []
for s in seed:
	for w in coeff:
		userToError = defaultdict(list)
		acc = 0
		c = 0
		fname = 'data/digital_music/Results/latent_factor_100/mode_4/user_' + w + '_item_' + w + '/seed_' + s + '/test_error_bounded.csv'
		with open(fname) as file:
			for i, line in enumerate(file):
				if i == 0:
					continue
				line = line[:-1]
				u, _, r = line.split(',')
				acc += float(r)
				c += 1
				userToError[u].append(float(r))
		for u, errors in userToError.items():
			userToError[u] = sum(userToError[u]) / len(userToError[u])
		RMSEs = userToError.values()
		uRMSE.append(sum(RMSEs) / len(RMSEs))
print("===========================")
print("DIGITAL MUSIC USER RMSE Over Autoencoder Weights")
print(uRMSE)
print("===========================")
plt.figure(3)
plt.title('NAECF User RMSE on Digital Music Dataset')
plt.plot(float_coeff, uRMSE,'-o')
plt.xlabel('Autoencoder Weight (w)')
plt.ylabel('User Root Mean Square Error Over Ratings')

acc = 0
uRMSE = []
for s in seed:
	for w in coeff:
		userToError = defaultdict(list)
		acc = 0
		c = 0
		fname = 'data/instant_video/Results/latent_factor_100/mode_4/user_' + w + '_item_' + w + '/seed_' + s + '/test_error_bounded.csv'
		with open(fname) as file:
			for i, line in enumerate(file):
				if i == 0:
					continue
				line = line[:-1]
				u, _, r = line.split(',')
				acc += float(r)
				c += 1
				userToError[u].append(float(r))
		for u, errors in userToError.items():
			userToError[u] = sum(userToError[u]) / len(userToError[u])
		RMSEs = userToError.values()
		uRMSE.append(sum(RMSEs) / len(RMSEs))
print("===========================")
print("INSTANT VIDEO USER RMSE Over Autoencoder Weights")
print(uRMSE)
print("===========================")
plt.figure(4)
plt.title('NAECF User RMSE on Instant Video Dataset')
plt.plot(float_coeff, uRMSE,'-o')
plt.xlabel('Autoencoder Weight (w)')
plt.ylabel('User Root Mean Square Error Over Ratings')
plt.show()