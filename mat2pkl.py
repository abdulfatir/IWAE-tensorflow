import scipy.io
import pickle
import sys

mat = scipy.io.loadmat(sys.argv[1])
with open('data.pkl', 'wb') as pkl:
    pickle.dump(mat, pkl)
