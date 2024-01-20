from keras.models import model_from_json
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

file_base = "Models/00_model_0_score_test_0.5228"
json_file = open(file_base + '.json', 'r')
saved_model = json_file.read()
# close the file as good practice
json_file.close()
model_ = model_from_json(saved_model)
# load weights into new model
model_.load_weights(file_base + ".h5")

#%%
import pandas as pd

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")
#%%
Yhat_test = model_.predict(X_test)

#%%

score = r2_score(y_test, Yhat_test)
print(score)

#%%
fig = plt.figure(figsize=(15, 8))
plt.scatter(y_test, Yhat_test)
plt.plot(y_test, y_test, "k--", linewidth=3)
plt.axis("square")
plt.title("Prediction: R^2=%0.4f" % score)
plt.xlabel("Y")
plt.ylabel("Yhat")
plt.grid()
plt.show()

#%%
