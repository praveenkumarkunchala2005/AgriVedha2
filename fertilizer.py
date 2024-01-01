import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('/Users/praveenkumar/Desktop/CropRecomendation 2/Fertilizer Prediction.csv')

df.columns = [col.strip() for col in df.columns]
df.drop(['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type'], axis=1, inplace=True)
df.head(6)


x = df.drop(['Fertilizer Name'], axis=1)
y = df['Fertilizer Name']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

ferti = RandomForestClassifier()

ferti.fit(x_train, y_train)

y_pred = ferti.predict(x_test)


random_acc = accuracy_score(y_test, y_pred)
print("Accuracy is " + str(random_acc))


filename = 'fertilizer.pkl'
with open(filename, 'wb') as model_file:
    pickle.dump(ferti, model_file)

print("Model saved successfully.")
