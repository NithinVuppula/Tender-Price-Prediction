import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


df = pd.read_csv(r"C:/360DigiTMG/Project/linearRegression/projectnew(Sheet1).csv")  # Read CSV file into a DataFrame
print(df.columns)

names = {
    'Tender no' : 'tender_no', 
    'Name of Railway' : 'name_of_railway', 
    'Date of\nOpening' : 'date_of_opening', 
    'Nature' : 'nature',
    'Awarded to' : 'awarded_to',
    'L1 Price' : 'l1_price',
    'BASIC' : 'basic', 
    'Qty' : 'qty'
    }

df = df.rename(columns = names)
print(df.columns)

print(df.dtypes)
df = df.applymap(lambda x: str(x).replace(',', '') if isinstance(x, str) else x)
col1 = {
    'tender_no' : 'int', 
    'l1_price' : 'float64',
    'basic' : 'float64', 
    'qty' : 'float64',
    'date_of_opening' : 'datetime64[ns]',
    }
df = df.astype(col1)
df['date_of_opening'] = pd.to_datetime(df['date_of_opening']).dt.normalize()


df['l1_price'] = df['l1_price'].fillna(1786)


comp_columns = ['Comp_A','Comp_B', 'Comp_C', 'Comp_D', 'Comp_E', 'Comp_F', 'Comp_G', 'Comp_H','Comp_I', 'Comp_J', 'Comp_K', 'Comp_L']

df[comp_columns] = df[comp_columns].astype('float64')
print(df.dtypes)




#changing the values for comps into l1_price
# Loop through the columns and check the condition
for col in comp_columns:
    df.loc[df['awarded_to'] == col, col] = df['l1_price']

# Print the updated DataFrame
print(df)


df.info()
df.isnull().sum()
df.describe().T
df.shape






#Heatmap
plt.figure(figsize=(15, 8))
numeric_data = df.select_dtypes(include=np.number)
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



#Histogram
sns.histplot(df['l1_price'], bins=30, kde=False, alpha=0.7); plt.title('l1_price'); plt.show()
sns.histplot(df['basic'], bins=30, kde=False, alpha=0.7); plt.title('basic'); plt.show()
sns.histplot(df['qty'], bins=30, kde=False, alpha=0.7); plt.title('qty'); plt.show()





    
#Scatterplot
sns.scatterplot(x = 'basic', y = 'l1_price', data = df)
plt.title("Relationship between BASIC and L1 Price")
plt.show()

sns.scatterplot(x = 'qty', y = 'l1_price', data = df)
plt.title("Relationship between Qty and L1 Price")
plt.show()







#Barplot
sns.barplot(x='basic', y ='l1_price', data = df)
plt.title('basic vs L1 Price')
plt.show()






#Boxplot
columns_to_plot = ['l1_price', 'basic', 'qty']

# Remove outliers from each column using IQR method
for col in columns_to_plot:
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
 
  IQR = Q3 - Q1
  upper_bound = Q3 + 1.5 * IQR
  lower_bound = Q1 - 1.5 * IQR
  outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
  df[col] = np.clip(df[col], lower_bound, upper_bound)  
    
       
# Plot boxplots for the columns after removing outliers
for col in columns_to_plot:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()
    
    
    
    
df['category'].value_counts()
df['nature'].value_counts()  
df['awarded_to'].value_counts()


df['nature']=df['nature'].replace('Supply', 'supply')   
df['nature']=df['nature'].replace('SUPPLY', 'supply') 


df['category'].unique()
df['awarded_to'].unique()
df['nature'].unique()








# Define the features and target
features_data = ['qty', 'basic', 'Comp_A', 'Comp_B', 'Comp_C', 'Comp_D', 'Comp_E', 'Comp_F', 'Comp_G', 'Comp_H', 'Comp_I', 'Comp_J', 'Comp_K', 'Comp_L']
target_data = 'l1_price'

# Separate features and target
X = df[features_data]
y = df[target_data]

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that includes scaling and regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the data
    ('model', LinearRegression())  # Step 2: Fit Linear Regression model
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on train and test data
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Calculate evaluation metrics for train and test sets
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)


print("\nLinear Regressor Results:")
print("Train R2 Score: {:.2f}%".format(train_r2 * 100))
print("Test R2 Score: {:.2f}%".format(test_r2 * 100))
print("Train MAE: {:.2f}".format(train_mae))
print("Test MAE: {:.2f}".format(test_mae))
print("Train MSE: {:.2f}".format(train_mse))
print("Test MSE: {:.2f}".format(test_mse))
print("Train RMSE: {:.2f}".format(train_rmse))
print("Test RMSE: {:.2f}".format(test_rmse))






import os
import joblib
from sklearn.pipeline import Pipeline

# Ensure that the 'static' folder exists, create it if it doesn't
static_folder = 'static'
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

# Define the path where the model will be saved
model_path = os.path.join(static_folder, 'model.pkl')

# Save the trained model to the 'static' folder
joblib.dump(pipeline, model_path)