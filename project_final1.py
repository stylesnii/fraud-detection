import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
data = pd.read_csv('creditcard.csv')

# Plot histograms
sns.histplot(data['Amount'], kde=True)
sns.histplot(data['Time'], kde=True)
plt.show()

# Joint plot of 'Time' and 'Amount'
sns.jointplot(x='Time', y='Amount', data=data)
plt.show()

# Data class distribution
class0 = data[data['Class'] == 0]
class1 = data[data['Class'] == 1]

print("Number of Class 0 instances:", len(class0))
print("Number of Class 1 instances:", len(class1))

# Shuffle and sample data
from sklearn.utils import shuffle

temp = shuffle(class0)
d1 = temp.iloc[:2000, :]
frames = [d1, class1]
df_temp = pd.concat(frames)

# Shuffle and save the sampled data
df = shuffle(df_temp)
df.to_csv('creditcardsampling.csv', index=False)

# Count plot
sns.countplot(x='Class', data=df)
plt.show()
