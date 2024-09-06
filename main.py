import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#person 1

#loading the dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

#Remove null or NA values
df = df.dropna()

#Remove any redundant rows
df = df.drop_duplicates()



#Convert categorical variables to numerical variables
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}) #Gender
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'no': 0, 'yes': 1}) #Has a family member suffered or suffers from overweight?
df['FAVC'] = df['FAVC'].map({'no': 0, 'yes': 1}) #Do you eat high caloric food frequently?
df['SMOKE'] = df['SMOKE'].map({'no': 0, 'yes': 1}) #Do you smoke?
df['CAEC'] = df['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}) #Do you eat any food between meals?
df['SCC'] = df['SCC'].map({'no': 0, 'yes': 1})  
df['CALC'] = df['CALC'].map({'no': 0, 'Sometimes': 1,'Frequently': 2,'Always': 3})  
df['MTRANS'] = df['MTRANS'].map({'Public_Transportation': 0, 'Automobile': 1, 'Bike': 2, 'Walking': 3})  
df['NObeyesdad'] = df['NObeyesdad'].astype('category').cat.codes


#Splitting the dataset into features and targets
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Ensure no NaN values are present
if X.isnull().sum().sum() > 0:
    # Impute missing values if necessary
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

#standardizing the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

#spliting test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialing the linear regression model using gradient descent
model = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, random_state=42)

#training the model
model.fit(X_train, y_train)

#making predictions
y_pred = model.predict(X_test)

#calculating the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#plot mse vs number of iterations
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.legend()
plt.show()

#report coefficients and evaluation statistics
print(f"Weight Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

#r2 score
r2_score = model.score(X_test, y_test)
print("R2 Score:", r2_score)

#save results and plots
with open('model_results.txt', 'w') as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"Weight Coefficients: {model.coef_}\n")
    f.write(f"Intercept: {model.intercept_}\n")
    f.write(f"R2 Score: {r2_score}\n")

#save plot
plt.savefig('predicted_vs_actual.png')
