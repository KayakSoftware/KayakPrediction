from FileFeatureExtracter import FileFeatureManager
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import RobustScaler;

RELEVANT_ACTIVITIES = ["Walking"]

# Instantiate FileFeaturerManager and read data into memory
fileFeatureManager = FileFeatureManager();
fileFeatureManager.readKayakDataIntoMemory(["527-8.csv"])
fileFeatureManager.readActivityDataIntoMemory(RELEVANT_ACTIVITIES, "WISDM_ar_v1.1_raw.txt");

# Create a dataframe from the currennt fileFeatureManager memory and initiate respectiive RobustScaler from training Reference
dataFrame_TrainingReference = fileFeatureManager.createDataFromFromMemory();
trainingScaler = RobustScaler().fit(dataFrame_TrainingReference[['x_axis', 'y_axis', 'z_axis']])

# Sample test data from fileFeatureManager
testDataWalking = fileFeatureManager.extractActivitySet("Walking", 2000)
testDataSailing = fileFeatureManager.extractActivitySet("Sailing", 2000)
#testDataJogging = fileFeatureManager.extractActivitySet("Jogging", 2000)
testData = fileFeatureManager.combineData([testDataWalking, testDataSailing])

# Use the RobustScaler from the training reference to transform test data
transformedInput = trainingScaler.transform(np.array(testData))

# Convert the transformed input into predictionSets
predictionSets = fileFeatureManager.convertActivitySetsToDataSet([transformedInput], 200, 200)

# Actual machine learning LSMT model
model = keras.models.load_model("HAR_LSTM")
y_pred = model.predict(predictionSets)

translatedPredictions = fileFeatureManager.interpretPrediction(y_pred, ['Sailing', 'Walking'])

for translation in translatedPredictions:
    print(translation)