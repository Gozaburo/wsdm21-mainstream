from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
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
filepath = './data/digital_music/NAECF_1992/full_rating.csv'
out = './surprise_clean'

df = pd.read_csv(filepath)
with open(out, "w") as f:
	for i, row in df.iterrows():
		uid = row['user_id']
		iid = row['item_id']
		rating = row['rating']
		f.write(str(uid) + ' ' + str(iid) + ' ' + str(rating) + '\n')
reader = Reader(line_format='user item rating', sep=' ')
data = Dataset.load_from_file(out, reader=reader)
trainset, testset = train_test_split(data, test_size=.20)
algorithm = SVD()
digital_svd = algorithm.fit(trainset).test(testset)

sim_options = {
	'user_based': True
}
algorithm = KNNBasic(sim_options = sim_options)
digital_uucf = algorithm.fit(trainset).test(testset)

sim_options = {
	'user_based': False
}
algorithm = KNNBasic(sim_options = sim_options)
digital_iicf = algorithm.fit(trainset).test(testset)
acc = 0
splitRMSE = []
coeff = ['0.5']
naecf_digital_user_errors = defaultdict(list)
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
				u, _, r = line.split(',')
				acc += float(r)
				c += 1
				naecf_digital_user_errors[u].append(float(r))
			splitRMSE.append(acc / c)
x = ['SVD', 'UU KNN', 'II KNN', 'NAECF (w = .5)']
rmse = [
	accuracy.rmse(digital_svd),
	accuracy.rmse(digital_uucf),
	accuracy.rmse(digital_iicf),
	splitRMSE[0]
]
for algo, e in zip(x, rmse):
	print(algo, e)

x_pos = [i for i, _ in enumerate(x)]
plt.figure(1)
ax = plt.gca()
ax.set_ylim([.70, 1.1])
plt.bar(x_pos, rmse, color='blue')
plt.xlabel("Algorithm")
plt.ylabel("Overall Root Mean Square Error")
plt.title("Overall Root Mean Square Error Across Algorithms on Digital Music Dataset")

plt.xticks(x_pos, x)
filepath = './data/instant_video/NAECF_1992/full_rating.csv'
out = './surprise_clean'

df = pd.read_csv(filepath)
with open(out, "w") as f:
	for i, row in df.iterrows():
		uid = row['user_id']
		iid = row['item_id']
		rating = row['rating']
		f.write(str(uid) + ' ' + str(iid) + ' ' + str(rating) + '\n')
reader = Reader(line_format='user item rating', sep=' ')
data = Dataset.load_from_file(out, reader=reader)
trainset, testset = train_test_split(data, test_size=.20)
algorithm = SVD()
video_svd = algorithm.fit(trainset).test(testset)
sim_options = {
	'user_based': True
}
algorithm = KNNBasic(sim_options = sim_options)
video_uucf = algorithm.fit(trainset).test(testset)

sim_options = {
	'user_based': False
}
algorithm = KNNBasic(sim_options = sim_options)
video_iicf = algorithm.fit(trainset).test(testset)


acc = 0
splitRMSE = []
coeff = ['0.3']
naecf_video_user_errors = defaultdict(list)
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
				u, _, r = line.split(',')
				acc += float(r)
				c += 1
				naecf_digital_user_errors[u].append(float(r))
			splitRMSE.append(acc / c)
x = ['SVD', 'UU KNN', 'II KNN', 'NAECF (w = .3)']
rmse = [
	accuracy.rmse(video_svd),
	accuracy.rmse(video_uucf),
	accuracy.rmse(video_iicf),
	splitRMSE[0]
]

for algo, e in zip(x, rmse):
	print(algo, e)

x_pos = [i for i, _ in enumerate(x)]
plt.figure(2)
ax = plt.gca()
ax.set_ylim([.70, 1.2])
plt.bar(x_pos, rmse, color='blue')
plt.xlabel("Algorithm")
plt.ylabel("Overall Root Mean Square Error")
plt.title("Overall Root Mean Square Error Across Algorithms on Instant Video Dataset")

plt.xticks(x_pos, x)

digital_svd_user_errors = defaultdict(list)
for pred in digital_svd:
	digital_svd_user_errors[pred.uid].append((pred.r_ui - pred.est)**2)
for uid, errors in digital_svd_user_errors.items():
	digital_svd_user_errors[uid] = sum(errors) / len(errors)

digital_uu_user_errors = defaultdict(list)
for pred in digital_uucf:
	digital_uu_user_errors[pred.uid].append((pred.r_ui - pred.est)**2)
for uid, errors in digital_uu_user_errors.items():
	digital_uu_user_errors[uid] = sum(errors) / len(errors)

digital_ii_user_errors = defaultdict(list)
for pred in digital_iicf:
	digital_ii_user_errors[pred.uid].append((pred.r_ui - pred.est)**2)
for uid, errors in digital_ii_user_errors.items():
	digital_ii_user_errors[uid] = sum(errors) / len(errors)

for uid, errors in naecf_digital_user_errors.items():
	naecf_digital_user_errors[uid] = sum(errors) / len(errors)
digital_svd_user_error = list(digital_svd_user_errors.values())
digital_uu_user_error = list(digital_uu_user_errors.values())
digital_ii_user_error = list(digital_ii_user_errors.values())
digital_naecf_user_error = list(naecf_digital_user_errors.values())
bins = [.05*i for i in range(30)]
kwargs = dict(histtype='step', density=True, bins=bins)
plt.figure(3)
plt.title('User RMSE SVD vs NAECF Digital Music')
plt.xlabel('User Root Mean Square Error')
plt.ylabel('Density of Uses With RMSE')
plt.hist(digital_svd_user_error,histtype='step', density=True, bins=bins,color='red',label='SVD')
plt.hist(digital_naecf_user_error,histtype='step', density=True, bins=bins,color='blue',label='NAECF')
plt.legend()
plt.figure(4)
plt.title('User RMSE UU CF vs NAECF Digital Music')
plt.xlabel('User Root Mean Square Error')
plt.ylabel('Density of Uses With RMSE')
plt.hist(digital_uu_user_error,histtype='step', density=True, bins=bins,color='red',label='UU CF')
plt.hist(digital_naecf_user_error,histtype='step', density=True, bins=bins,color='blue',label='NAECF')
plt.legend()
plt.figure(5)
plt.title('User RMSE II CF vs NAECF Digital Music')
plt.xlabel('User Root Mean Square Error')
plt.ylabel('Density of Uses With RMSE')
plt.hist(digital_ii_user_error,histtype='step', density=True, bins=bins,color='red', label='II CF')
plt.hist(digital_naecf_user_error,histtype='step', density=True, bins=bins,color='blue', label = 'NAECF')
plt.legend()




video_svd_user_errors = defaultdict(list)
for pred in video_svd:
	video_svd_user_errors[pred.uid].append((pred.r_ui - pred.est)**2)
for uid, errors in video_svd_user_errors.items():
	video_svd_user_errors[uid] = sum(errors) / len(errors)

video_uu_user_errors = defaultdict(list)
for pred in video_uucf:
	video_uu_user_errors[pred.uid].append((pred.r_ui - pred.est)**2)
for uid, errors in video_uu_user_errors.items():
	video_uu_user_errors[uid] = sum(errors) / len(errors)

video_ii_user_errors = defaultdict(list)
for pred in video_iicf:
	video_ii_user_errors[pred.uid].append((pred.r_ui - pred.est)**2)
for uid, errors in video_ii_user_errors.items():
	video_ii_user_errors[uid] = sum(errors) / len(errors)

for uid, errors in naecf_video_user_errors.items():
	naecf_video_user_errors[uid] = sum(errors) / len(errors)
video_svd_user_error = list(video_svd_user_errors.values())
video_uu_user_error = list(video_uu_user_errors.values())
video_ii_user_error = list(video_ii_user_errors.values())
video_naecf_user_error = list(naecf_video_user_errors.values())
bins = [.05*i for i in range(30)]
kwargs = dict(histtype='step', density=True, bins=bins)
plt.figure(6)
plt.title('User RMSE SVD vs NAECF Instant Video')
plt.xlabel('User Root Mean Square Error')
plt.ylabel('Density of Uses With RMSE')
plt.hist(digital_svd_user_error,histtype='step', density=True, bins=bins,color='red',label='SVD')
plt.hist(digital_naecf_user_error,histtype='step', density=True, bins=bins,color='blue',label='NAECF')
plt.legend()
plt.figure(7)
plt.title('User RMSE UU CF vs NAECF Instant Video')
plt.xlabel('User Root Mean Square Error')
plt.ylabel('Density of Uses With RMSE')
plt.hist(digital_uu_user_error,histtype='step', density=True, bins=bins,color='red',label='UU CF')
plt.hist(digital_naecf_user_error,histtype='step', density=True, bins=bins,color='blue',label='NAECF')
plt.legend()
plt.figure(8)
plt.title('User RMSE II CF vs NAECF Instant Video')
plt.xlabel('User Root Mean Square Error')
plt.ylabel('Density of Uses With RMSE')
plt.hist(digital_ii_user_error,histtype='step', density=True, bins=bins,color='red', label='II CF')
plt.hist(digital_naecf_user_error,histtype='step', density=True, bins=bins,color='blue', label = 'NAECF')
plt.legend()



plt.show()