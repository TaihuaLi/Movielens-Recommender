from math import sqrt
from numpy import *
from numpy import linalg as la
import numpy as np
import math

# User based functions
	
# ----------------------------------------------#	
# ---------------input type: numpy--------------#
# ----------------------------------------------#
	
def getcorrelation(dat, u1, u2):
	test1 = np.nonzero(dat[u1])[0]
	test2 = np.nonzero(dat[u2])[0]
	
	# find overlap indices
	items = []
	for i in test1:
		if i in test2:
			items.append(i)
	if len(items) == 0: return 0
	
	# calculate correlations on items that are mutually rated
	to_calc_1 = np.array(dat[0][items])
	to_calc_2 = np.array(dat[1][items])

	correlation = np.corrcoef(to_calc_1, to_calc_2)[0][1] # returns only a number
	
	if math.isnan(correlation): 
		return 0
	elif correlation <= 0:
		correlation = 0
	
	return correlation

def findsimusers(dat, user, n = 5, similarity=getcorrelation):
	scores = []
	for other in range(len(dat)):
		if other != user-1:
			scores.append([similarity(dat, user-1, other), other])
	# sort the list
	scores.sort()
	scores.reverse()
	
	return scores[:n]

def getmemovies(dat, user, movies, n=5, similarity=getcorrelation):
	
	to_rec = []
	
	for other in range(len(dat)):
		if other == user-1: continue
		sim = similarity(dat, user-1, other)
	 	if sim <= 0: continue
	 	for item in np.where(dat[other] != 0)[0]:
	 		if item not in np.nonzero(dat[user])[0] or dat[user][item] == 0:
	 			item_rating = dat[other][item]*sim
	 			temp = [item_rating, item]
				to_rec.append(temp)
	# sort
	to_rec.sort()
	to_rec.reverse()
	movie_rec = []
	
	for i in range(n):
		name = movies[to_rec[i][1]]
		movie_rec.append(name)
	
	return movie_rec

def cv_user(dat, test_ratio, similarity = getcorrelation):
	number_of_users = np.shape(dat)[0]
	user_list = np.array(range(0, number_of_users))
	test_user_size = test_ratio*number_of_users
	test_user_indices = np.random.randint(0, number_of_users, test_user_size)
	witheld_users = user_list[test_user_indices]

	tot = 0
	
	for user in witheld_users:
		number_of_items = np.shape(dat)[1]
		rated_items_by_user = np.array(np.nonzero(dat[user]))[0]
		test_size = test_ratio * len(rated_items_by_user)
		test_indices = np.random.randint(0, len(rated_items_by_user), test_size)
		witheld_items = rated_items_by_user[test_indices]
		original_user_profile = np.copy(dat[user])
		dat[user, witheld_items] = 0 # set to 0

		simuser = findsimusers(dat, user, n=1)[0][1]
		to_test = []
		sim = similarity(dat, user, simuser)
		if sim <= 0: sim = 0
		
		for item in np.where(dat[simuser]!= 0)[0]:
	 		if item not in np.nonzero(dat[user])[0] or dat[user][item] == 0:
	 			item_rating = dat[simuser][item]*sim
	 			temp = [item_rating, item]
				to_test.append(temp)
		
		error_u = 0
		count_u = len(witheld_items)
		
		for i in to_test:
			item_name = i[1]
			item_pred_rating = i[0]
			error_u = error_u + abs(dat[user][item_name] - item_pred_rating)
		maes = error_u/count_u
		tot += maes

	MAE = tot/len(witheld_users)
	
	print "The MAE  is for user-based collaborative filtering is: %0.5f" % MAE
	
	return MAE
			
# Item Based Functions

# -----------------------------------------------#	
# ---------------input type: numpy---------------#
# -----------------------------------------------#	
	
def pearsSim(inA,inB):
    if len(inA) < 3 : 
    	return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar = 0)[0][1]
	
def getsim(dataMat, simMeas = pearsSim):
	n = np.shape(dataMat)[1] # number of items
	dat = dataMat.T # movie-user format
	simL = {}

	for item in range(n):
		temp = {}
		simL[item] = {}
		for j in range(n):
			test1 = np.nonzero(dat[item])[0]
			test2 = np.nonzero(dat[j])[0]
			
			items = []
			for i in test1:
				if i in test2: 
					items.append(i)
				
			to_calc_1 = np.array(dat[item][items])
			to_calc_2 = np.array(dat[j][items])

			if len(items) != 0:
				correlation = np.corrcoef(to_calc_1, to_calc_2)[0][1]
			else:
				correlation = 0
				
			if math.isnan(correlation): 
				correlation = 0
			
			temp[j] = correlation
				# returns r between item and j
			if item not in simL:
				simL[item] = temp
			else:
				simL[item].update(temp)

	return simL
    

def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    
    for j in range(n):
        userRating = dataMat[user,j]
        
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:,item]>0, dataMat[:,j]>0))[0]
        
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j])
        
        #print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
        
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def standEstTest(dataMat, user, similarity, item):
	n = np.shape(dataMat)[1] # number of items
	simTotal = 0.0
	ratSimTotal = 0.0
	
	for j in range(n):
		userRating = dataMat[user, j]
		if userRating == 0: 
			continue
		
		sim = similarity[item][j]
		if sim <= 0: sim = 0
		
		simTotal += sim
		ratSimTotal += sim * userRating
		
	if simTotal == 0: 
		return 0
	else: 
		return ratSimTotal/simTotal

def recommend(dataMat, user, simMeas, N=3, estMethod=standEstTest):
    unratedItems = nonzero(np.matrix(dataMat[user,:]).A==0)[1] #find unrated items 
    
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def cross_validate_user(dataMat, user, test_ratio, simMeas, estMethod=standEstTest):
	number_of_items = np.shape(dataMat)[1]
	rated_items_by_user = np.array([i for i in range(number_of_items) if dataMat[user,i]>0])
	test_size = test_ratio * len(rated_items_by_user)
	test_indices = np.random.randint(0, len(rated_items_by_user), test_size)
	withheld_items = rated_items_by_user[test_indices]
	original_user_profile = np.copy(dataMat[user])
	dataMat[user, withheld_items] = 0 # So that the withheld test items is not used in the rating estimation below
	error_u = 0.0
	count_u = len(withheld_items)

	# Compute absolute error for user u over all test items
	for item in withheld_items:
		# Estimate rating on the withheld item
		estimatedScore = estMethod(dataMat, user, simMeas, item)
		error_u = error_u + abs(estimatedScore - original_user_profile[item])	
	
	# Now restore ratings of the withheld items to the user profile
	for item in withheld_items:
		dataMat[user, item] = original_user_profile[item]
		
	# Return sum of absolute errors and the count of test cases for this user
	# Note that these will have to be accumulated for each user to compute MAE
	return error_u, count_u
	
def test(dataMat, test_ratio, simMeas):
	error = 0
	count = 0
	for i in range(np.shape(dataMat)[0]):
		er, ct = cross_validate_user(dataMat, i, test_ratio, simMeas)
		error += er
		count += ct
	MAE = error/count
	print "The MAE  is for user-based collaborative filtering is: %0.5f" % MAE
	return MAE