from FileFeatureExtracter import FileFeatureManager
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import RobustScaler;


class Har_Brain:

    def __init__(self):
        print("Initializing Brain")
        RELEVANT_ACTIVITIES = ["Walking"]

        # Instantiate FileFeaturerManager and read data into memory
        print("Reading Data into memory...")
        self.fileFeatureManager = FileFeatureManager();
        print("Reading kayak data...")
        self.fileFeatureManager.readKayakDataIntoMemory(["527-8.csv"])
        print("Reading activity data...")
        self.fileFeatureManager.readActivityDataIntoMemory(RELEVANT_ACTIVITIES, "WISDM_ar_v1.1_raw.txt");


        print("Creating dataframe trainingReference")
        # training reference vector for input transformation
        dataFrame_TrainingReference = self.fileFeatureManager.createDataFromFromMemory();
        print("Constructing training reference vector")
        self.trainingScaler = RobustScaler().fit(dataFrame_TrainingReference[['x_axis', 'y_axis', 'z_axis']])

        # Machine learning model
        print("Loading saved HAR_LSTM model...")
        self.brain = keras.models.load_model("HAR_LSTM")
        print("Brain initialized")

    def predict(self, dataList):
        transform = self.trainingScaler.transform(np.array(dataList))
        predictionSet = []
        predictionSet.append(transform)
        prediction = self.brain.predict(np.array(predictionSet))
        translatedPredictions = self.fileFeatureManager.interpretPrediction(prediction, ["Sailing", "Walking"])
        return translatedPredictions[0]


