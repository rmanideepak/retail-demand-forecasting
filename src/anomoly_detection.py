import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("/Users/manideepak/Desktop/PycharmProjects/Retail_Demand_Project/data/archive-2/train.csv")

iso = IsolationForest(contamination=0.01)

df['anomaly'] = iso.fit_predict(df[['Weekly_Sales']])

anomalies = df[df['anomaly'] == -1]

print("Anomalies detected:")
print(anomalies.head())
