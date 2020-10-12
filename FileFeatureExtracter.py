import math
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pylab import rcParams
from scipy import stats

class FileFeatureManager:

    def __init__(self):
        self.sailData = []
        self.activityData = []

        # This registers pandas formatters and converters with matplotlib more specifically
        # modifies global matplotlib.units.registry for timeStamp, period, dateTime etc...
        register_matplotlib_converters()

        # diagram styles style=Plot layout, palleter=Plot color, font_scale=diagram font size
        sns.set(style="whitegrid", palette="muted", font_scale=1.5)

        # more diagram plot styles
        rcParams["figure.figsize"] = 20, 10

    def readKayakDataIntoMemory(self, filePaths):

        accelerometerData = []
        gyroscopeData = []

        # be aware that the chronological order of different sets are obeyed? Or give each file an id similar to the other training data
        for filePath in filePaths:
            file = open(filePath, "r")

            lines = file.read().split("\n");

            rawimusLines = []

            for line in lines:
                if "rawimus" in line:
                    rawimusLines.append(line)

            for line in rawimusLines:
                columns = line.split(",")
                dataColumns = []
                for column in columns:
                    if column != "":
                        dataColumns.append(column.replace(",", "."))

                accelerometerData.append(
                    TemporalThreeAxisData("Sailing", dataColumns[0].replace(".", "")[0:12], dataColumns[2], dataColumns[3],
                                          dataColumns[4]))
                gyroscopeData.append(
                    TemporalThreeAxisData("Sailing", dataColumns[0].replace(".", "")[0:12], dataColumns[5], dataColumns[6],
                                          dataColumns[7]))
            file.close();

        self.sailData = accelerometerData;

    def readActivityDataIntoMemory(self, featureType, filePath):

        file = open(filePath, "r")
        lines = file.read().split("\n")
        featureTypeLines = []

        for line in lines:
            if isinstance(featureType, list):
                for feature in featureType:
                    if feature in line:
                        featureTypeLines.append(line.replace(";", ""))
                        break
            else:
                if featureType in line:
                    featureTypeLines.append(line.replace(";", ""))

        accelerometerData = []

        for line in featureTypeLines:
            columns = line.split(",")
            dataColumns = []
            for column in columns:
                if column != "":
                    dataColumns.append(column)

            if len(dataColumns) == 6:
                accelerometerData.append(
                    TemporalThreeAxisData(dataColumns[1], dataColumns[2], dataColumns[3], dataColumns[4],
                                          dataColumns[5]))
        file.close()

        self.activityData = accelerometerData;

    # Sample an equal fraction of all current types of data
    def sampleFractionOfData(self, percentage, replaceCurrentState = False):

        sailDataFraction = []

        # Take the first percentage of data and return if replaceCurrentState all replace the current state with that fraction
        if(len(self.sailData) > 0):
            sailingDataMaxCount = math.floor(len(self.sailData) / 100 * percentage)
            for i in range(sailingDataMaxCount):
                sailDataFraction.append(self.sailData[i])

        # We have to take an equal fraction from each type of activity data
        # sort each type of data into their own list and extract the fraction from each

        sorted = {}
        for data in self.activityData:
            if data.type in sorted:
                sorted[data.type].append(data)
            else:
                sorted[data.type] = [data];

        activityFractions = {}
        for type in sorted:
            activityFractions[type] = [];
            activityDataMaxCount = math.floor(len(sorted[type]) / 100 * percentage)
            for i in range(activityDataMaxCount):
                activityFractions[type].append(sorted[type][i])

        activityDataFraction = []
        for type in activityFractions:
            for activity in activityFractions[type]:
                activityDataFraction.append(activity)

        if replaceCurrentState:
            self.sailData = sailDataFraction
            self.activityData = activityDataFraction

        return sailDataFraction, activityDataFraction

    # Split the data into a training and test set file, based on percentage cut-off
    def writeMemoryToTrainingAndTest(self, learningDataFile, testDataFile, learningFraction):

        sailLearnFraction = []
        sailTestFraction = []

        activityLearnFraction = []
        activityTestFraction = []

        sailingDataMaxCount = math.floor(len(self.sailData) / 100 * learningFraction)
        for i in range(len(self.sailData) -1):
            if(i < sailingDataMaxCount):
                sailLearnFraction.append(self.sailData[i])
            else:
                sailTestFraction.append(self.sailData[i])

        sorted = {}
        for data in self.activityData:
            if data.type in sorted:
                sorted[data.type].append(data)
            else:
                sorted[data.type] = [data];

        activityLearnFractions = {}
        activityTestFractions = {}
        for type in sorted:
            activityLearnFractions[type] = [];
            activityTestFractions[type] = [];
            activityDataMaxCount = math.floor(len(sorted[type]) / 100 * learningFraction)
            for i in range(len(sorted[type])):
                if i < activityDataMaxCount:
                    activityLearnFractions[type].append(sorted[type][i])
                else:
                    activityTestFractions[type].append(sorted[type][i])

        for type in activityLearnFractions:
            for activity in activityLearnFractions[type]:
                activityLearnFraction.append(activity)

        for type in activityTestFractions:
            for activity in activityTestFractions[type]:
                activityTestFraction.append(activity)

        self.writeFile(learningDataFile, self.combineData([sailLearnFraction, activityLearnFraction]))
        self.writeFile(testDataFile, self.combineData([sailTestFraction, activityTestFraction]))

    # Overwrite the specified filePath with the temporalThreeAxisData
    def writeMemoryToFile(self, filePath):
        self.writeFile(filePath, self.combineData([self.activityData, self.sailData]))

    def writeFile(self, filePath, dataList):
        file = open(filePath, "w")

        for data in dataList:
            file.write(
                data.type + "," + str(data.time) + "," + str(data.x) + "," + str(data.y) + "," + str(data.z) + ";\n")
        file.close();

    # Combine different lists of TemporalThreeAxisData into one
    def combineData(self, dataLists):
        combinedList = []

        for list in dataLists:
            for data in list:
                combinedList.append(data)
        return combinedList

    # Read the csv into a dataFrame, remove z-axis trailing ";"
    # also convert to float and drop all rows with null values
    def createDataFrameFromFile(self, filePath, column_names = ['activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis']):
        df_complete = pd.read_csv(filePath, header=None, names=column_names)
        df_complete.z_axis.replace(regex=True, inplace=True, to_replace=r';', value=r'')
        df_complete['z_axis'] = df_complete.z_axis.astype(np.float64)
        df_complete.dropna(axis=0, how='any', inplace=True)

        return df_complete

    def createDataFromFromMemory(self, column_names=["x_axis", "y_axis", "z_axis"]):
        combined = self.combineData([self.activityData, self.sailData])
        dataList = []
        for data in combined:
            dataList.append([data.x, data.y, data.z])

        dataFrame = pd.DataFrame(dataList, columns=column_names)
        return dataFrame


    def plot_activities(self, activities, dataFrame, subset = 400):
        for activity in activities:
            self.plot_activity(activity, dataFrame, subset)

    # plot a certain activity
    def plot_activity(self, activity, dataFrame, subset = 400):
        data = dataFrame[dataFrame['activity'] == activity][['x_axis', 'y_axis', 'z_axis']][:subset]  # <--- Subset to plot
        axis = data.plot(subplots=True, figsize=(16, 12), title=activity)
        for ax in axis:
            ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
        plt.show()

    def plot_cm(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(18, 16))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.diverging_palette(220, 20, n=7),
            ax=ax
        )

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        plt.show()  # ta-da!

    def count_plot(self, data_frame, title):
        # Create a count plot of the number of records for each activity
        sns.countplot(x="activity", data=data_frame, order=data_frame.activity.value_counts().index)
        plt.title(title)

    def create_dataset(self, X, y, time_steps=1, step=1):
        Xs, ys = [], []
        for i in range(0, len(X) - time_steps, step):
            v = X.iloc[i:(i + time_steps)].values
            labels = y.iloc[i: i + time_steps]
            Xs.append(v)
            ys.append(stats.mode(labels)[0][0])
        return np.array(Xs), np.array(ys).reshape(-1, 1)


    def extractActivitySet(self, activity, size):
        combined = self.combineData([self.activityData, self.sailData])

        observationSets = [];

        for data in combined:
            if len(observationSets) == size:
                return observationSets

            if data.type == activity:
                observationSets.append([data.x, data.y, data.z])

        return observationSets

    def convertActivitySetsToDataSet(self, activitySets, timeSteps=1, step=1):

        collectiveSet = []
        for activitySet in activitySets:
            for data in activitySet:
                collectiveSet.append(data)

        dataSet = []
        for i in range(0, len(collectiveSet) - timeSteps, step):
            data = collectiveSet[i: i + timeSteps]
            dataSet.append(data)

        return np.array(dataSet)

    def interpretPrediction(self, pred, categories):

        translatedPredictions = []
        for i in range(len(pred)):
            maxIndex = 0;
            currentMax = 0;
            for j in range(len(pred[i])):
                if pred[i][j] > currentMax:
                    currentMax = pred[i][j]
                    maxIndex = j
            translatedPredictions.append(categories[maxIndex])

        return translatedPredictions


class TemporalThreeAxisData:

    def __init__(self, type, time, x, y, z):
        self.time = time
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.type = type

    def toString(self):
        return "Type: " + str(self.type) + " - Time: " + str(self.time) + " - X: " + str(self.x) + " - Y: " + str(self.y) + " - Z: " + str(self.z)