from sklearn import preprocessing
import numpy as np
X = np.array([[-2,2],
              [-1,1]]
			,dtype=np.float)
scaler= preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X)
X_scaled = scaler.transform(X)
print("X\n",X)
print("X_scaled\n",X_scaled)
X_back=scaler.inverse_transform(X_scaled)
print("X_back\n",X_back)