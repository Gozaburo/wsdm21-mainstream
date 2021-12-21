predictionDict = {}
actualDict = {}
# seed = seed[:1]
seed = ['1992']
acc = 0
c = 0
splitRMSE = []
for s in seed:
	acc = 0
	c = 0
	fname = 'data/digital_music/Results/latent_factor_100/mode_4/user_0.5_item_0.5/seed_' + s + '/test_error_bounded.csv'
	# fname = 'data/instant_video/Results/latent_factor_100/mode_5_/seed_' + s + '/test_error_bounded.csv'
	with open(fname) as file:
		for i, line in enumerate(file):
			if i == 0:
				continue
			line = line[:-1]
			_, _, r = line.split(',')
			acc += float(r)
			c += 1
		splitRMSE.append((acc, c))
totalRMSE = 0
for i in range(len(splitRMSE)):
	totalRMSE += (splitRMSE[i][0] / splitRMSE[i][1])**.5
avgRMSE = totalRMSE / len(splitRMSE)
print(avgRMSE)
