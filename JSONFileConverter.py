file = open("527-8.csv", "r")

lines = file.read().split("\n");

rawimusLines = []

for line in lines:
    if "rawimus" in line:
        rawimusLines.append(line)

file.close()

newFile = open("527-8.json", "w")

newFile.write("{")
newFile.write("\"data\": [")

index = 0;

for line in rawimusLines:

    columns = line.split(",")
    dataColumns = []
    for column in columns:
        if column != "":
            dataColumns.append(column.replace(",", "."))

    newFile.write("{")
    newFile.write("\"xAxis\":" + str(dataColumns[2]) + ",")
    newFile.write("\"yAxis\":" + str(dataColumns[3]) + ",")
    newFile.write("\"zAxis\":" + str(dataColumns[4]))
    newFile.write("}")
    index += 1;
    if(index != len(rawimusLines)):
        newFile.write(",")


newFile.write("]}")

newFile.close();