import pandas

#convert features_3_sec.csv into a dataframe, and give me a list of all unique items in the column "label"
#dont make it a function

data = pandas.read_csv(r'..\data\Data\features_3_sec.csv')
#print what the unique items are in the column "label" and how many there are
print(data['label'].value_counts())

#give me all the columns in the dataframe
print(data.columns)