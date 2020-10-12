from FileFeatureExtracter import FileFeatureManager
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler;
from sklearn.preprocessing import OneHotEncoder;

RANDOM_SEED = 42
RELEVANT_ACTIVITIES = ["Walking"]
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

fileFeatureManager = FileFeatureManager();
fileFeatureManager.readKayakDataIntoMemory(["527-8.csv"])
# "Walking", "Jogging", "Standing", "Downstairs", "Upstairs", "Sitting"
fileFeatureManager.readActivityDataIntoMemory(RELEVANT_ACTIVITIES, "WISDM_ar_v1.1_raw.txt");
fileFeatureManager.writeMemoryToFile("CompleteData.txt")

df_complete = fileFeatureManager.createDataFrameFromFile("CompleteData.txt")

RELEVANT_ACTIVITIES.append("Sailing")
fileFeatureManager.count_plot(df_complete, "Records pr. activity")
fileFeatureManager.plot_activities(RELEVANT_ACTIVITIES, df_complete)

fileFeatureManager.writeMemoryToTrainingAndTest("LearningData.txt", "TestData.txt", 80)
df_train = fileFeatureManager.createDataFrameFromFile("LearningData.txt")
df_test = fileFeatureManager.createDataFrameFromFile("TestData.txt")

scale_columns = ['x_axis', 'y_axis', 'z_axis']

scaler = RobustScaler()
scaler = scaler.fit(df_train[scale_columns])

df_train.loc[:, scale_columns] = scaler.transform(df_train[scale_columns].to_numpy())
df_test.loc[:, scale_columns] = scaler.transform(df_test[scale_columns].to_numpy())

TIME_STEPS = 200
STEP = 40

X_train, y_train = fileFeatureManager.create_dataset(
    df_train[['x_axis', 'y_axis', 'z_axis']],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = fileFeatureManager.create_dataset(
    df_test[['x_axis', 'y_axis', 'z_axis']],
    df_test.activity,
    TIME_STEPS,
    STEP
)

print(X_train.shape, y_train.shape)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

print(X_train.shape, y_train.shape)

# Actual machine learning LSMT model
model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=128,
          input_shape=[X_train.shape[1], X_train.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Train network
history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    shuffle=True
)

# Plot training progress
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Evaluate the model
model.evaluate(X_test, y_test)

# Do model predictions
y_pred = model.predict(X_test)

# Take note on how the categories are arranged this will need to be used in evaluation
print(enc.categories_[0])

# Plot the final confusion matrix
fileFeatureManager.plot_cm(
  enc.inverse_transform(y_test),
  enc.inverse_transform(y_pred),
  enc.categories_[0]
)

# Save AI to disk
model.save("HAR_LSTM")
