#Installed Package : 
#C:\Installed_Softwares\anaconda3\Lib\site-packages\implicit


#conda install -c conda-forge implicit
#conda install -c conda-forge/label/gcc7 implicit 
#conda install -c conda-forge/label/cf201901 implicit 
#conda install -c conda-forge/label/cf202003 implicit


import os
import sys

import logging
import time

import numpy as np
import tqdm

import codecs
import h5py
from scipy.sparse import coo_matrix, csr_matrix
from scipy import sparse

import implicit

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, TFIDFRecommender, bm25_weight

import sqlite3
from sqlite3 import Error


from implicit.datasets.lastfm import get_lastfm



def predictByCollaborativeFiltering(inputMatrix):
	print("predictByCollaborativeFiltering")
	
	conn = sqlite3.connect('CONP.db')
	cur = conn.cursor()
	cur.execute("SELECT * FROM summary_results")
	print("cur : ")
	print(cur)
	rows = cur.fetchall()
	
	model = AlternatingLeastSquares(factors = 64, dtype = np.float32)
	print("model : ")
	print(model)

	model.fit(data, show_progress=False)

	fittedData = np.copy(data.toarray())
	print(fittedData)
	return

	# row indices
	row_ind = np.array([0, 1, 1, 3, 4])
	# column indices
	col_ind = np.array([0, 2, 4, 3, 4])
	# data to be stored in COO sparse matrix
	data = np.array([-100, -200, 300, -4000, 50000], dtype=float)
	data = sparse.csr_matrix((data, (row_ind, col_ind)))
	print(data.toarray())
	
	model = AlternatingLeastSquares(factors = 64, dtype = np.float32)
	model.fit(data, show_progress=False)
	
	fittedData = np.copy(data.toarray())
	for rowIndex in range(5):
		score = model.recommend(rowIndex, data)
		#print(rowIndex,  score)
		for item in score:
			columnNumber = item[0]
			fittedData[rowIndex, item[0]] = round(item[1], 2)
	print(fittedData)
	
	sys.exit(0)


	""" Generates artist recommendations for each user in the dataset """
	# train the model based off input params
	
	#INFO:implicit:Using cached dataset at 'C:\Users\mmNabi\implicit_datasets\lastfm_360k.hdf5'
	#More info @ https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html
	artists, users, plays = get_lastfm()
	'''
	with h5py.File('../../../Datasets/lastfm_360k.hdf5', 'r') as f:
		#<KeysViewHDF5 ['artist', 'artist_user_plays', 'user']>
		print(f.keys())
		m = f.get('artist_user_plays')
		plays = csr_matrix((m.get('data'), m.get('indices'), m.get('indptr')))
		artists, users, plays =  np.array(f['artist']), np.array(f['user']), plays
	'''

	conn = sqlite3.connect('../../../Datasets/CONP.db')
	cur = conn.cursor()
	cur.execute("SELECT * FROM summary_results")
	rows = cur.fetchall()
	#(1, 'a7f11079d50f7a5d5582986486866d42', None, 'zenodo.3240521', 'fslstats', '0')
	#Dataset Name : None
	#Pipeline : fslstats
	#Result : 0
	print(rows[0])

	print("artists : ", len (artists))
	print(artists[0:10])
	print("users : ", len (users))
	print(users[0:10])
	print("plays : ", plays.get_shape())
	print(plays[0:10, 0:10].toarray())



	

	# create a model from the input data
	'''
	factors=100, regularization=0.01, dtype=np.float32,
				 use_native=True, use_cg=True, use_gpu=implicit.cuda.HAS_CUDA,
				 iterations=15, calculate_training_loss=False, num_threads=0,
				 random_state=None
	'''
	model = AlternatingLeastSquares(factors = 64, dtype = np.float32)

	logging.debug("weighting matrix by bm25_weight")
	plays = bm25_weight(plays, K1=100, B=0.8)
	print("plays : ")
	print(plays)

	# also disable building approximate recommend index
	model.approximate_similar_items = False

	# this is actually disturbingly expensive:
	plays = plays.tocsr()
	print("plays :")
	print(plays)

	logging.debug("training model ALS")
	start = time.time()

	'''
	DEBUG:implicit:Calculated transpose in 0.913s
	DEBUG:implicit:Initialized factors in 0.8054990768432617
	DEBUG:implicit:Running 15 ALS iterations
	'''
	model.fit(plays)			#Takes time to fit the model

	'''
	DEBUG:root:trained model ALS in 85.18s
	'''
	logging.debug("trained model ALS in %0.2fs", time.time() - start)

	# generate recommendations for each user and write out to a file
	start = time.time()
	user_plays = plays.T.tocsr()
	with tqdm.tqdm(total=len(users)) as progress:
		with codecs.open(output_filename, "w", "utf8") as o:
			for userid, username in enumerate(users):
				for artistid, score in model.recommend(userid, user_plays):
					o.write("%s\t%s\t%s\n" % (username, artists[artistid], score))
				progress.update(1)
	logging.debug("generated recommendations in %0.2fs",  time.time() - start)



if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)

	predictByCollaborativeFiltering(inputMatrix = Null)
