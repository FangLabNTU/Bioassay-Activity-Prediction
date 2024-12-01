#%% Import necessary libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import MolStandardize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mordred import Calculator, descriptors
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time
from joblib import dump
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import concurrent.futures

#%% Define binning function
def generate_fixed_bins(min_value, max_value, interval=1):
    bins = []

    # Calculate bin boundaries for negative values
    start = -interval
    while start >= min_value:
        bins.append(start)
        start -= interval

    # Use 0 as the center point
    bins.append(0)

    # Calculate bin boundaries for positive values
    start = interval
    while start <= max_value:
        bins.append(start)
        start += interval

    # Ensure boundaries are ordered and unique
    bins = sorted(set(bins))  # Remove duplicates and sort
    return bins



#%% Standardize SMILES
class MolClean(object):
    def __init__(self):
        self.normizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()
 
    def clean(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            # Note: Parameters 'isomericSmiles' and 'canonical' should be carefully set here
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None

mc = MolClean()

#%% Convert SMILES to Morgan fingerprints
def smiles_to_fingerprints(smiles, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    except:
        return np.zeros(n_bits)
    
#%% Mordred fingerprint data with Random Forest

# Create a descriptor calculator
calc = Calculator(descriptors, ignore_3D=True)

# Read CSV file
df = pd.read_csv(r"D:/Desktop/regression/wide_all_dfs_filter.csv")

# Filter out invalid SMILES strings
df = df[df['smiles'].apply(lambda x: isinstance(x, str) and Chem.MolFromSmiles(x) is not None)]

# Apply MolClean to the SMILES column in the DataFrame
df['clean_smiles'] = df['smiles'].apply(mc.clean)

# Remove rows containing NA values
df = df.dropna(subset=['clean_smiles'])

# Add a new column with RDKit molecule objects created from SMILES strings
PandasTools.AddMoleculeColumnToFrame(df, 'clean_smiles', 'Molecule')

# Remove invalid molecule objects
df = df[df['Molecule'].notnull()]

# Compute fingerprints
X = pd.DataFrame(calc.pandas(df['Molecule']))
X = X.dropna()
X = X._get_numeric_data()

# Remove corresponding rows from df
df = df.loc[X.index]
df.drop(columns=["Molecule"], inplace=True)

# Select all columns except specific ones
columns_to_keep = ['smiles', 'Molecule', 'clean_smiles', 'casn']
# columns_to_process = df.columns.difference(['casn', 'smiles', 'clean_smiles'])

for col in df.columns.difference(columns_to_keep):
    if df[col].notna().sum() < 50:
        df.drop(columns=[col], inplace=True)

# Determine the minimum and maximum values across the DataFrame
min_value = df[df.columns.difference(columns_to_keep)].min().min()
max_value = df[df.columns.difference(columns_to_keep)].max().max()

# Generate fixed bin boundaries
fixed_bins = generate_fixed_bins(min_value, max_value)

# Apply binning to each column (excluding specific columns)
for col in df.columns.difference(columns_to_keep):
    df[col] = pd.cut(df[col], fixed_bins, labels=np.arange(len(fixed_bins)-1), right=False)

# Initialize DataFrame to store accuracy data
accuracy_df = pd.DataFrame(columns=['Column', 'Accuracy'])
filled_df = df.copy()

# Define parameter grid
param_grid = {
    'n_estimators': [10, 30, 50, 80, 100, 200],
    'max_depth': [3, 5, 8, 10, 15],
    'min_samples_split': [2, 5, 8]
}

# Iterate through each column
for col in tqdm(df.columns.difference(columns_to_keep), desc="Processing"):
    start_time = time.time()

    # Identify known and unknown data points
    known = df[col].notna()
    unknown = df[col].isna()

    # Skip the current column if there are no missing or known values
    if unknown.sum() == 0 or known.sum() == 0:
        continue

    # Prepare training data
    X_train = X.loc[known].values if known.sum() > 0 else np.array([])
    y_train = df.loc[known, col]
    
    # Check if there are enough samples for cross-validation
    # if X_train.shape[0] < 5 or any(y_train.value_counts() < 5):
        # Check if total sample count is less than 5 or if any class has fewer than 5 samples
    #    continue

    # Use XGBoost classifier model
    # model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    # model.fit(X_train, y_train)
    
    # Use Random Forest classifier model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    # Create GridSearchCV instance
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameter model
    # best_model = grid_search.best_estimator_

    # Evaluate accuracy
    try:
        accuracy = cross_val_score(rf, X_train, y_train, cv=5).mean()
        accuracy_df.loc[len(accuracy_df)] = [col, accuracy]
    except ValueError:
        # Skip accuracy calculation if cross-validation fails due to insufficient samples
        continue

    # Predict missing values
    X_pred = X.loc[unknown]
    if not X_pred.empty:
        filled_df.loc[unknown, col] = rf.predict(X_pred)

    end_time = time.time()
    print(f"Completed {col} in {end_time - start_time:.2f} seconds")

# Save results
accuracy_df.to_csv("accuracy_df_regression_mord-R.csv", index=False)
filled_df.to_csv("filled_df_regression_mord-R.csv", index=False)

# Save model
dump(rf, 'regression_model_mord-R.joblib')


#%% Mordred fingerprint data with XGBoost

# Create descriptor calculator
calc = Calculator(descriptors, ignore_3D=True)

# Read CSV file
df = pd.read_csv(r"/home/cy/wide_all_dfs_filter.csv")

# Filter out invalid SMILES strings
df = df[df['smiles'].apply(lambda x: isinstance(x, str) and Chem.MolFromSmiles(x) is not None)]

# Apply MolClean to the SMILES column in the DataFrame
df['clean_smiles'] = df['smiles'].apply(mc.clean)

# Remove rows containing NA values
df = df.dropna(subset=['clean_smiles'])

# Add a new column with RDKit molecule objects created from SMILES strings
PandasTools.AddMoleculeColumnToFrame(df, 'clean_smiles', 'Molecule')

# Remove invalid molecule objects
df = df[df['Molecule'].notnull()]

# Compute fingerprints
X = pd.DataFrame(calc.pandas(df['Molecule']))
X = X.dropna()
X = X._get_numeric_data()

# Remove corresponding rows from df
df = df.loc[X.index]
df.drop(columns=["Molecule"], inplace=True)

# Select all columns except specific ones
columns_to_keep = ['smiles', 'Molecule', 'clean_smiles', 'casn']
# columns_to_process = df.columns.difference(['casn', 'smiles', 'clean_smiles'])

for col in df.columns.difference(columns_to_keep):
    if df[col].notna().sum() < 50:
        df.drop(columns=[col], inplace=True)

# Initialize DataFrame to store accuracy data
accuracy_df = pd.DataFrame(columns=['Column', 'Accuracy'])
filled_df = df.copy()

# Define parameter grid for search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize dictionaries to store encoders and bin boundaries for each column
label_encoders = {}
bins_mapping = {}

# Perform binning and encoding on each column
for col in df.columns.difference(columns_to_keep):
    # Apply custom binning function
    non_null_data = df[col].dropna()
    bins = generate_fixed_bins(non_null_data.min(), non_null_data.max())
    bins_mapping[col] = bins  # Save bin boundaries
    df[col] = pd.cut(df[col], bins=bins, labels=False, right=False)

    # Apply LabelEncoder and save the encoder
    le = LabelEncoder()

    # Encode non-missing values
    df.loc[df[col].notna(), col] = le.fit_transform(df.loc[df[col].notna(), col])
    label_encoders[col] = le

# Iterate through each column
for col in tqdm(df.columns.difference(columns_to_keep), desc="Processing"):

    start_time = time.time()

    # Identify known and unknown data points
    known = df[col].notna()
    unknown = df[col].isna()

    # Skip the current column if there are no missing or known values
    if unknown.sum() == 0 or known.sum() == 0:
        continue

    # Prepare training data
    X_train = X.loc[known].values if known.sum() > 0 else np.array([])
    y_train = df.loc[known, col]

    # Create an XGBoost model instance
    model = xgb.XGBClassifier(random_state=42, tree_method='hist', device='cuda')

    # Create grid search instance
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate accuracy
    try:
        accuracy = cross_val_score(best_model, X_train, y_train, cv=5).mean()
        accuracy_df.loc[len(accuracy_df)] = [col, accuracy]
    except ValueError:
        # Skip accuracy calculation if cross-validation fails due to insufficient samples
        continue

    # Predict and reverse-transform predictions
    X_pred = X.loc[unknown]
    if not X_pred.empty:
        predicted_labels = best_model.predict(X_pred)
        original_labels = label_encoders[col].inverse_transform(predicted_labels)
        
        # Convert predicted labels back to bin intervals
        predicted_bins = [bins_mapping[col][int(label)] for label in original_labels]
        filled_df.loc[unknown, col] = predicted_bins

    end_time = time.time()
    print(f"Completed {col} in {end_time - start_time:.2f} seconds")

# Save results
accuracy_df.to_csv("accuracy_df_regression_mord-X.csv", index=False)
filled_df.to_csv("filled_df_regression_mord-X.csv", index=False)

# Save model
dump(best_model, 'regression_model_mord-X.joblib')


#%% Mordred fingerprint data with XGBoost, parallel execution

# Create descriptor calculator
calc = Calculator(descriptors, ignore_3D=True)

# Read CSV file
df = pd.read_csv(r"D:/Desktop/regression/wide_all_dfs_filter.csv")

# Filter out invalid SMILES strings
df = df[df['smiles'].apply(lambda x: isinstance(x, str) and Chem.MolFromSmiles(x) is not None)]

# Apply MolClean to the SMILES column in the DataFrame
df['clean_smiles'] = df['smiles'].apply(mc.clean)

# Remove rows containing NA values
df = df.dropna(subset=['clean_smiles'])

# Add a new column with RDKit molecule objects created from SMILES strings
PandasTools.AddMoleculeColumnToFrame(df, 'clean_smiles', 'Molecule')

# Remove invalid molecule objects
df = df[df['Molecule'].notnull()]

# Compute fingerprints
X = pd.DataFrame(calc.pandas(df['Molecule']))
X = X.dropna()
X = X._get_numeric_data()

# Remove corresponding rows from df
df = df.loc[X.index]
df.drop(columns=["Molecule"], inplace=True)

# Select all columns except specific ones
columns_to_keep = ['smiles', 'Molecule', 'clean_smiles', 'casn']
# columns_to_process = df.columns.difference(['casn', 'smiles', 'clean_smiles'])

for col in df.columns.difference(columns_to_keep):
    if df[col].notna().sum() < 50:
        df.drop(columns=[col], inplace=True)

# Initialize DataFrame to store accuracy data
accuracy_df = pd.DataFrame(columns=['Column', 'Accuracy'])
filled_df = df.copy()

# Define parameter grid for search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize dictionaries to store encoders and bin boundaries for each column
label_encoders = {}
bins_mapping = {}

# Perform binning and encoding on each column
for col in df.columns.difference(columns_to_keep):
    # Apply custom binning function
    non_null_data = df[col].dropna()
    bins = generate_fixed_bins(non_null_data.min(), non_null_data.max())
    bins_mapping[col] = bins  # Save bin boundaries
    df[col] = pd.cut(df[col], bins=bins, labels=False, right=False)

    # Apply LabelEncoder and save the encoder
    le = LabelEncoder()
    
    # Encode non-missing values
    df.loc[df[col].notna(), col] = le.fit_transform(df.loc[df[col].notna(), col])
    label_encoders[col] = le

# Process each column in parallel
def process_column(col, X, df, param_grid, label_encoders):
    start_time = time.time()

    # Identify known and unknown data points
    known = df[col].notna()
    unknown = df[col].isna()

    # Skip the current column if there are no missing or known values
    if unknown.sum() == 0 or known.sum() == 0:
        return None

    # Prepare training data
    X_train = X.loc[known].values if known.sum() > 0 else np.array([])
    y_train = df.loc[known, col]

    # Create an XGBoost model instance
    model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

    # Create grid search instance
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate accuracy
    try:
        accuracy = cross_val_score(best_model, X_train, y_train, cv=5).mean()
        accuracy_df.loc[len(accuracy_df)] = [col, accuracy]
    except ValueError:
        # Skip accuracy calculation if cross-validation fails due to insufficient samples
        return None

    # Predict and reverse-transform predictions
    X_pred = X.loc[unknown]
    if not X_pred.empty:
        predicted_labels = best_model.predict(X_pred)
        original_labels = label_encoders[col].inverse_transform(predicted_labels)
        
        # Convert predicted labels back to bin intervals
        predicted_bins = [bins_mapping[col][int(label)] for label in original_labels]
        filled_df.loc[unknown, col] = predicted_bins

    end_time = time.time()
    print(f"Completed {col} in {end_time - start_time:.2f} seconds")

# Use ProcessPoolExecutor for parallel processing
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_column, col, X, df, param_grid, label_encoders) 
               for col in df.columns.difference(columns_to_keep)]
    
    for future in concurrent.futures.as_completed(futures):
        col, accuracy, predictions = future.result()
        if accuracy is not None:
            accuracy_df.loc[len(accuracy_df)] = [col, accuracy]
        if predictions is not None:
            filled_df.loc[df[col].isna(), col] = predictions

# Save results
accuracy_df.to_csv("accuracy_df_regression_mord-XP.csv", index=False)
filled_df.to_csv("filled_df_regression_mord-XP.csv", index=False)

# Save model
dump(best_model, 'regression_model_mord-XP.joblib')

#%% Mordred fingerprint data with SVM
# Create a descriptor calculator
calc = Calculator(descriptors, ignore_3D=True)

# Read CSV file
df = pd.read_csv(r"D:/Desktop/regression/wide_all_dfs_filter.csv")

# Filter out invalid SMILES strings
df = df[df['smiles'].apply(lambda x: isinstance(x, str) and Chem.MolFromSmiles(x) is not None)]

# Apply MolClean to the SMILES column in the DataFrame
df['clean_smiles'] = df['smiles'].apply(mc.clean)

# Remove rows containing NA values
df = df.dropna(subset=['clean_smiles'])

# Add a new column with RDKit molecule objects created from SMILES strings
PandasTools.AddMoleculeColumnToFrame(df, 'clean_smiles', 'Molecule')

# Remove invalid molecule objects
df = df[df['Molecule'].notnull()]

# Compute fingerprints
X = pd.DataFrame(calc.pandas(df['Molecule']))
X = X.dropna()
X = X._get_numeric_data()

# Remove corresponding rows from df
df = df.loc[X.index]
df.drop(columns=["Molecule"], inplace=True)

# Select all columns except specific ones
columns_to_keep = ['smiles', 'Molecule', 'clean_smiles', 'casn']
# columns_to_process = df.columns.difference(['casn', 'smiles', 'clean_smiles']) 

for col in df.columns.difference(columns_to_keep):
    if df[col].notna().sum() < 50:
        df.drop(columns=[col], inplace=True)

# Determine the minimum and maximum values across the DataFrame
min_value = df[df.columns.difference(columns_to_keep)].min().min()
max_value = df[df.columns.difference(columns_to_keep)].max().max()

# Generate fixed bin boundaries
fixed_bins = generate_fixed_bins(min_value, max_value)

# Apply binning to each column (excluding specific columns)
for col in df.columns.difference(columns_to_keep):
    df[col] = pd.cut(df[col], fixed_bins, labels=np.arange(len(fixed_bins)-1), right=False)

# Initialize DataFrame to store accuracy data
accuracy_df = pd.DataFrame(columns=['Column', 'Accuracy'])
filled_df = df.copy()

# Define parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel type
    'gamma': ['scale', 'auto']  # Kernel coefficient
}

# Iterate through each column
for col in tqdm(df.columns.difference(columns_to_keep), desc="Processing"):
    start_time = time.time()

    # Identify known and unknown data points
    known = df[col].notna()
    unknown = df[col].isna()

    # Skip the current column if there are no missing or known values
    if unknown.sum() == 0 or known.sum() == 0:
        continue

    # Prepare training data
    X_train = X.loc[known].values if known.sum() > 0 else np.array([])
    y_train = df.loc[known, col]

    # Check if there are enough samples for cross-validation
    # if X_train.shape[0] < 5 or any(y_train.value_counts() < 5):
        # Check if total sample count is less than 5 or if any class has fewer than 5 samples
    #    continue

    # Use Support Vector Machine (SVM) model
    svm = SVC(random_state=42)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameter model
    best_model = grid_search.best_estimator_

    # Evaluate accuracy
    try:
        accuracy = cross_val_score(best_model, X_train, y_train, cv=5).mean()
        accuracy_df.loc[len(accuracy_df)] = [col, accuracy]
    except ValueError:
        # Skip accuracy calculation if cross-validation fails due to insufficient samples
        continue

    # Predict missing values
    X_pred = X.loc[unknown]
    if not X_pred.empty:
        filled_df.loc[unknown, col] = best_model.predict(X_pred)

    end_time = time.time()
    print(f"Completed {col} in {end_time - start_time:.2f} seconds")

# Save results
accuracy_df.to_csv("accuracy_df_regression_mord-S.csv", index=False)
filled_df.to_csv("filled_df_regression.csv_mord-S", index=False)

# Save model
dump(best_model, 'regression_model_mord-S.joblib')

#%% Morgan fingerprint data preparation
df = pd.read_csv(r"D:/Desktop/regression/test.csv") 

# Apply MolClean to the SMILES column in the DataFrame
df['clean_smiles'] = df['smiles'].apply(mc.clean)

# Convert SMILES to Morgan fingerprints
df['Morgan_Fingerprint'] = df['clean_smiles'].apply(smiles_to_fingerprints)

# Ensure Morgan_Fingerprint is in a format suitable for model input, such as each fingerprint being a numerical array

# Prepare features and target variables
df = df.dropna().dropna(axis=1)

# Generate bin boundaries
min_value = df['ATG_PXRE_CIS_up'].min()
max_value = df['ATG_PXRE_CIS_up'].max()
bins = generate_fixed_bins(min_value, max_value)

# Apply binning
df['ATG_PXRE_CIS_up'] = pd.cut(df['ATG_PXRE_CIS_up'], bins, labels=np.arange(len(bins)-1), right=False)

# Remove rows with missing values in the 'ATG_PXRE_CIS_up' column
df = df.dropna(subset=['ATG_PXRE_CIS_up'])

X = np.array(list(df['Morgan_Fingerprint']))  # Assumes Morgan_Fingerprint is already a list of numeric arrays
y = df['ATG_PXRE_CIS_up']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

#%% Morgan fingerprint with Random Forest

# Create an instance of the Random Forest classifier
classifier = RandomForestClassifier(random_state=42)

# Perform 10-fold cross-validation
scores = cross_val_score(classifier, X, y, cv=10)

# Output accuracy for each fold
print("Accuracy scores for each fold:")
print(scores)

# Output average accuracy
print("Average accuracy:", scores.mean())

# Define parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [10, 20, 30],        # Maximum depth of the trees
    'min_samples_split': [2, 4, 6],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required at a leaf node
}

# Create a GridSearchCV instance
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)

# Execute grid search
grid_search.fit(X, y)

# Output the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Output the accuracy of the best model
print("Best accuracy: ", grid_search.best_score_)

# Save the best model
best_model = grid_search.best_estimator_
dump(best_model, 'best_model_morg-R.joblib')

# Fill missing values using the best model (from grid search)
# For each feature column with missing values, we use the trained model to predict the missing values.
# We assume 'X' contains the features and 'y' contains the target labels.

# Identify missing data (NaNs)
missing_data = df[df.isna().any(axis=1)]

# Fill missing values for each column
filled_df = df.copy()
for col in missing_data.columns:
    known = df[col].notna()  # Identify known (non-missing) values
    unknown = df[col].isna()  # Identify missing values
    
    # Prepare training data
    X_train = X.loc[known].values  # Use only known data points for training
    y_train = df.loc[known, col]
    
    # Train the best model from grid search
    best_model.fit(X_train, y_train)
    
    # Predict missing values
    X_pred = X.loc[unknown]
    if not X_pred.empty:
        filled_df.loc[unknown, col] = best_model.predict(X_pred)

# Save the accuracy data
accuracy_df.to_csv("accuracy_scores_morg-R.csv", index=False)

# Save the filled dataframe (with predicted values for missing data)
filled_df.to_csv("filled_df_morg-R.csv", index=False)

# Record the end time
end_time = time.time()

# Print the total execution time
print(f"Execution time: {end_time - start_time:.2f} seconds")
#%% Morgan fingerprint with XGBoost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Create GridSearchCV instance
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Execute grid search
grid_search.fit(X_train, y_train)

# Output the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Output the accuracy of the best model
print("Best accuracy: ", grid_search.best_score_)

# Save the best model
best_model = grid_search.best_estimator_
dump(best_model, 'best_model_morg-X.joblib')

# Fill missing values using the best model (from grid search)
# For each feature column with missing values, we use the trained model to predict the missing values.
# We assume 'X' contains the features and 'y' contains the target labels.

# Identify missing data (NaNs)
missing_data = df[df.isna().any(axis=1)]

# Fill missing values for each column
filled_df = df.copy()
for col in missing_data.columns:
    known = df[col].notna()  # Identify known (non-missing) values
    unknown = df[col].isna()  # Identify missing values
    
    # Prepare training data
    X_train = X.loc[known].values  # Use only known data points for training
    y_train = df.loc[known, col]
    
    # Train the best model from grid search
    best_model.fit(X_train, y_train)
    
    # Predict missing values
    X_pred = X.loc[unknown]
    if not X_pred.empty:
        filled_df.loc[unknown, col] = best_model.predict(X_pred)

# Save the accuracy data
accuracy_df.to_csv("accuracy_scores_morg-X.csv", index=False)

# Save the filled dataframe (with predicted values for missing data)
filled_df.to_csv("filled_df_morg-X.csv", index=False)

# Record the end time
end_time = time.time()

# Print the total execution time
print(f"Execution time: {end_time - start_time:.2f} seconds")
#%% Morgan fingerprint with SVM
svm_classifier = SVC(random_state=42)

# Define parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10],           # Regularization parameter
    'svm__gamma': ['scale', 'auto'],   # Kernel coefficient
    'svm__kernel': ['rbf', 'linear']   # Kernel type
}

# Create pipeline with standard scaling and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm_classifier)
])

# Create GridSearchCV instance
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

# Execute grid search
grid_search.fit(X_train, y_train)

# Output the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Output the accuracy of the best model
print("Best accuracy: ", grid_search.best_score_)

# Save the best model
best_model = grid_search.best_estimator_
dump(best_model, 'best_model_morg-S.joblib')

# Fill missing values using the best model (from grid search)
# For each feature column with missing values, we use the trained model to predict the missing values.
# We assume 'X' contains the features and 'y' contains the target labels.

# Identify missing data (NaNs)
missing_data = df[df.isna().any(axis=1)]

# Fill missing values for each column
filled_df = df.copy()
for col in missing_data.columns:
    known = df[col].notna()  # Identify known (non-missing) values
    unknown = df[col].isna()  # Identify missing values
    
    # Prepare training data
    X_train = X.loc[known].values  # Use only known data points for training
    y_train = df.loc[known, col]
    
    # Train the best model from grid search
    best_model.fit(X_train, y_train)
    
    # Predict missing values
    X_pred = X.loc[unknown]
    if not X_pred.empty:
        filled_df.loc[unknown, col] = best_model.predict(X_pred)

# Save the accuracy data
accuracy_df.to_csv("accuracy_scores_morg-S.csv", index=False)

# Save the filled dataframe (with predicted values for missing data)
filled_df.to_csv("filled_df_morg-S.csv", index=False)

# Record the end time
end_time = time.time()

# Print the total execution time
print(f"Execution time: {end_time - start_time:.2f} seconds")