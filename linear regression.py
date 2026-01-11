import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("D:/TEACHING/2024 teaching academic research/R/programs/datasets/LungCap.csv")

# View data structure and summary
print(data.info())
print(data.describe())

# Normalization
#nor <-function(x) { (x -min(x))/(max(x)-min(x))}  
scaler = MinMaxScaler()
data[['Age', 'Height', 'LungCap']] = scaler.fit_transform(data[['Age', 'Height', 
                                                                'LungCap']])
print("\nAfter Normalization:")
print(data[['Age', 'Height', 'LungCap']].describe())
# Scatter plot
plt.scatter(data['Age'], data['LungCap'])
plt.xlabel('Age (Normalized)')
plt.ylabel('Lung Capacity')
plt.title('Scatterplot: Age vs Lung Capacity')
plt.grid(True)
plt.show()
# Linear Regression Model
X = data[['Age']]
y = data[['LungCap']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# Predict 
checklungs = pd.DataFrame({'Age': [25]}) 
result = model.predict(checklungs) 
print("Predicted Lung Capacity for Age 25:", result)  
# Plot regression line 
plt.scatter(X, y, color='blue') 
plt.plot(X, model.predict(X), color='red', linewidth=3) 
plt.xlabel('Age') 
plt.ylabel('Lung Capacity') 
plt.title('Linear Regression Fit') 
plt.show()  
# Mean Squared Error
#Measures average squared difference between actual and predicted values
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

#use built in "cancer" dataset to pridict the type of dignosis with 80:20 train 
#test ratio using naive bayes and decision tree model. evaluate and visualize the 
#performance of model to mention the outperforming model.


#on test data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("D:/TEACHING/2024 teaching academic research/R/programs/datasets/LungCap.csv")

# View data structure and summary
print(data.info())
print(data.describe())

# Normalize selected columns
scaler = MinMaxScaler()
data[['Age', 'Height', 'LungCap']] = scaler.fit_transform(data[['Age', 'Height', 'LungCap']])
print("\nAfter Normalization:")
print(data[['Age', 'Height', 'LungCap']].describe())

# Scatter plot (Age vs Lung Capacity)
plt.scatter(data['Age'], data['LungCap'])
plt.xlabel('Age (Normalized)')
plt.ylabel('Lung Capacity (Normalized)')
plt.title('Scatterplot: Age vs Lung Capacity')
plt.grid(True)
plt.show()

# Define predictor (X) and target (y)
X = data[['Age']]
y = data[['LungCap']]

# Split dataset: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining Data: {X_train.shape[0]} samples")
print(f"Testing Data: {X_test.shape[0]} samples")

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Lung Capacity for test data (20%)
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE) on Test Data: {mse:.4f}")

# Optional: Predict for a specific Age (e.g., 25 years, normalized scale)
checklungs = pd.DataFrame({'Age': [0.25]})  # scaled value for Age=25 if original max=100
predicted_lung = model.predict(checklungs)
print("Predicted (normalized) Lung Capacity for Age 25:", predicted_lung)

# Plot regression line using training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X, model.predict(X), color='red', linewidth=3, label='Regression Line')
plt.xlabel('Age (Normalized)')
plt.ylabel('Lung Capacity (Normalized)')
plt.title('Linear Regression Fit (Train/Test Split)')
plt.legend()
plt.show()









#hands on SLR(How diabetes disease develops or worsens 
#over time in a patient with respect to BMI.)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load built-in diabetes dataset
diabetes = load_diabetes()
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data['target'] = diabetes.target

# View data structure and summary
print(data.info())
print(data.describe())

# Normalization
scaler = MinMaxScaler()
data[['age', 'bmi', 'target']] = scaler.fit_transform(data[['age', 'bmi', 'target']])
print("\nAfter Normalization:")
print(data[['age', 'bmi', 'target']].describe())

# Scatter plot: BMI vs Target
plt.scatter(data['bmi'], data['target'])
plt.xlabel('BMI (Normalized)')
plt.ylabel('Disease Progression (Normalized)')
plt.title('Scatterplot: BMI vs Disease Progression')
plt.grid(True)
plt.show()

# Linear Regression Model
X = data[['bmi']]
y = data[['target']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict for a new BMI value (e.g., 0.08 - normalized scale)
check_bmi = pd.DataFrame({'bmi': [0.08]})
result = model.predict(check_bmi)
print("Predicted Disease Progression for BMI 0.08 (Normalized):", result)

# Plot regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=3)
plt.xlabel('BMI (Normalized)')
plt.ylabel('Disease Progression (Normalized)')
plt.title('Linear Regression Fit')
plt.show()

# Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

#Logistic regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load breast cancer dataset
cancer = load_breast_cancer()
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data['target'] = cancer.target

# View structure
print(data.info())
print(data.describe())

# Normalize selected features
scaler = MinMaxScaler()
data[['mean radius', 'mean texture']] = scaler.fit_transform(data[['mean radius', 'mean texture']])
print("\nAfter Normalization:")
print(data[['mean radius', 'mean texture']].describe())

# Logistic Regression using one feature: mean radius
X = data[['mean radius']]
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = log_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", conf_mat)

# Plot decision curve
sns.scatterplot(x=X['mean radius'], y=y, hue=y, palette='Set1', alpha=0.6)
radius_vals = np.linspace(X['mean radius'].min(), X['mean radius'].max(), 100).reshape(-1, 1)
probs = log_model.predict_proba(radius_vals)[:, 1]
plt.plot(radius_vals, probs, color='black', linewidth=2)
plt.xlabel('Mean Radius (Normalized)')
plt.ylabel('Probability of Malignant (target = 1)')
plt.title('Logistic Regression: Cancer Prediction Curve')
plt.grid(True)
plt.show()



#MLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("D:/TEACHING/2024 teaching academic research/R/programs/datasets/LungCap.csv")

# View data structure and summary
print(data.info())
print(data.describe())

# Normalization (Min-Max Scaling)
scaler = MinMaxScaler()
data[['Age', 'Height', 'LungCap']] = scaler.fit_transform(data[['Age', 'Height', 'LungCap']])
print("\nAfter Normalization:")
print(data[['Age', 'Height', 'LungCap']].describe())

# Multiple Linear Regression Model
X = data[['Age', 'Height']]  # Now using both Age and Height
y = data[['LungCap']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict lung capacity for new data (example: Age=25, Height=70 inches)
# First normalize input values based on the original scaler
# Note: Must use same scaler used during training

# Create raw input and normalize it
input_df = pd.DataFrame({'Age': [25], 'Height': [70], 'LungCap': [0]})  # Dummy LungCap value for shape
input_scaled = scaler.transform(input_df)  # returns array of scaled values
input_scaled_df = pd.DataFrame(input_scaled, columns=['Age', 'Height', 'LungCap'])

checklungs = input_scaled_df[['Age', 'Height']]  # Only use predictors
result = model.predict(checklungs)
print("Predicted Lung Capacity for Age 25 and Height 70 inches (normalized input):", result[0][0])

checklungs = pd.DataFrame({'Age': [25], 'Height': [70]}) 
result = model.predict(checklungs) 
print("Predicted Lung Capacity for Age 25:", result)  
# Plot: Actual vs Predicted values
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred, color='purple')
plt.plot([0, 1], [0, 1], 'r--')  # ideal prediction line
plt.xlabel('Actual Lung Capacity')
plt.ylabel('Predicted Lung Capacity')
plt.title('Actual vs Predicted Lung Capacity (Multiple Linear Regression)')
plt.grid(True)
plt.show()

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

