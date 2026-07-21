import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import shap

real = np.load('preprocessing/fft_signals_real.npy')
fake = np.load('preprocessing/fft_signals_fake.npy')

ones = np.ones((len(real))).reshape(-1, 1)
zeros = np.zeros((len(real))).reshape(-1, 1)

real = np.hstack((real.reshape(len(real), -1), zeros))
fake = np.hstack((fake.reshape(len(real), -1), ones))
df = np.vstack((real, fake))

y = df[:, -1]
X = df[:, :-1]
y = y.ravel()

pipeline = Pipeline([("pca", PCA(n_components=10)), ("regression", LogisticRegression(max_iter=1000))])
pipeline.fit(X, y)
joblib.dump(pipeline, 'ai_detector.joblib')

X_explainer = pipeline.named_steps['pca'].transform(X)
explainer = shap.LinearExplainer(pipeline.named_steps['regression'], X_explainer)
shap_values = explainer(X_explainer)

img_shape = (256, 256)
f3_spect = pipeline.named_steps['pca'].components_[3].reshape(img_shape)
# plt.imshow(f3_spect, cmap='coolwarm', interpolation='bicubic')

# plt.show()
shap.plots.bar(shap_values)