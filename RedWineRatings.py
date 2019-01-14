# Red wine quality regression with random forest and ANN

#%%
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Input, Dropout
import matplotlib.pyplot as plt


# Import the data
raw_data = pd.read_csv('winequality-red.csv')

# Create features and label
X = raw_data.iloc[:,:-1]
y = raw_data.iloc[:, -1]

# Normalized the data
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled)

# Train and split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Create Random Forest Regressor model
rfr = RandomForestRegressor(n_estimators=200, random_state=4)
rfr.fit(X_train, y_train)


# Create Neural Net for Regression
def build_NN_model(input_shape):
    # Input Layer
    inputs = Input(shape=input_shape)
    # Hidden Layers
    X = Dense(128, activation='relu')(inputs)
    X = Dropout(0.25)(X)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.10)(X)
    X = Dense(32, activation='relu')(X)
    X = Dropout(0.25)(X)
    X = Dense(16, activation='relu')(X)
    X = Dropout(0.10)(X)
    X = Dense(8, activation='relu')(X)
    X = Dense(4, activation='relu')(X)
    predictions = Dense(1, activation='linear')(X)

    model = Model(inputs, predictions)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
    return model

# Create the NN model
ann_model = build_NN_model((11,))
history = ann_model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=0, validation_split=0.2)
    
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
   
# Evaluation Random Forest Regressor
rf_preds = rfr.predict(X_test)
errors = abs(rf_preds - y_test)
print('Metrics for Random Forest Trained on Expanded Data')
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_test))

# Calculate and display accuracy
accuracy = 100 - mape
print('Random Forest Accuracy:', round(accuracy, 2), '%.')

# Evaluation Artifical Neural Network
ann_preds = ann_model.predict(X_test)
ann_preds = ann_preds.reshape(320,)
errors = abs(ann_preds - y_test)
print('Metrics for Neural Network Trained on Expanded Data')
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_test))

# Calculate and display accuracy
accuracy = 100 - mape
print('Neural Network Accuracy:', round(accuracy, 2), '%.')
