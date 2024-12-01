#%% Import necessary libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import time
from rdkit.Chem import MolStandardize
from joblib import dump
from rdkit.Chem import PandasTools
from mordred import Calculator, descriptors
#%% SMILES standardization function
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
            # Note: Pay attention to the parameters 'isomericSmiles' and 'canonical' here; explained further below
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None

mc = MolClean()

#%% Function to convert SMILES to Morgan fingerprints

def smiles_to_fingerprints(smiles, radius=2, n_bits=1024):
    """Convert SMILES string to Morgan fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    except:
        # Return a fixed-length zero vector if molecule generation from SMILES fails
        return np.zeros(n_bits)


#%% Morgan fingerprint with Random Forest
df = pd.read_csv(r"/home/cy/wide_all_dfs_filter.csv")

# Filter out invalid SMILES strings
df = df[df['smiles'].apply(lambda x: isinstance(x, str) and Chem.MolFromSmiles(x) is not None)]

# Apply MolClean to the SMILES column in the DataFrame
df['clean_smiles'] = df['smiles'].apply(mc.clean)

# Remove any null values generated during cleaning
df.dropna(subset=['clean_smiles'], inplace=True)

# Apply the smiles_to_fingerprints function to each SMILES string
df["fingerprints"] = df['clean_smiles'].apply(smiles_to_fingerprints)

# Create a new DataFrame to store accuracy
accuracy_df = pd.DataFrame(columns=['Assay', 'Accuracy'])

# Create a new DataFrame to store filled data
filled_df = df.copy()

# Initialize a DataFrame to store accuracy
accuracy_df = pd.DataFrame(columns=['Column', 'Accuracy'])
filled_df = df.copy()

# Get all columns except specific ones
columns_to_process = df.columns.difference(['casn', 'smiles', 'clean_smiles', 'fingerprints'])

# Iterate through each column
for col in tqdm(columns_to_process, desc="Processing"):
    start_time = time.time()

    # Identify known and unknown data points
    known = df[col].notna()
    unknown = df[col].isna()

    # Skip the current column if there are no missing or known values
    if unknown.sum() == 0 or known.sum() == 0:
        continue

    # Prepare training data
    X_train = np.stack(df.loc[known, 'fingerprints']) if known.sum() > 0 else np.array([])
    y_train = df.loc[known, col]

    # Check if there are enough samples for cross-validation
    if X_train.shape[0] < 5 or any(y_train.value_counts() < 5):
        # Skip if total sample count is less than 5 or any class has fewer than 5 samples
        continue

    # Use Random Forest classifier
    rf = RandomForestClassifier(random_state=2023, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Evaluate accuracy
    try:
        accuracy = cross_val_score(rf, X_train, y_train, cv=5).mean()
        accuracy_df.loc[len(accuracy_df)] = [col, accuracy]
    except ValueError:
        # Skip accuracy calculation if cross-validation fails due to insufficient samples
        continue

    # Predict missing values
    X_pred = np.stack(df.loc[unknown, 'fingerprints']) if unknown.sum() > 0 else np.array([])
    if X_pred.size > 0:
        filled_df.loc[unknown, col] = rf.predict(X_pred)

    end_time = time.time()
    print(f"Completed {col} in {end_time - start_time:.2f} seconds")

accuracy_df.to_csv("accuracy_df.csv", index=False)
filled_df.to_csv("filled_df_classification.csv", index=False)

# Save the model to file
dump(rf, 'classification_model.joblib')

#%% Mordred fingerprint with Random Forest

# Initialize Mordred calculator
calc = Calculator(descriptors, ignore_3D=True)

# Load data
df = pd.read_csv(r"/home/cy/wide_all_dfs_classification.csv")

# Apply MolClean
df['clean_smiles'] = df['smiles'].apply(mc.clean)

# Remove null values
df.dropna(subset=['clean_smiles'], inplace=True)

# Add molecule column
PandasTools.AddMoleculeColumnToFrame(df, 'clean_smiles', 'Molecule')

# Remove invalid molecule objects
df = df[df['Molecule'].notnull()]

# Compute fingerprints
X = pd.DataFrame(calc.pandas(df['Molecule']))
X = X.dropna()
X = X._get_numeric_data()

# Remove corresponding rows from df
df = df.loc[X.index]

# Get all columns except specific ones
columns_to_process = df.columns.difference(['casn', 'smiles', 'clean_smiles', 'Molecule'])

# Remove columns with fewer than 100 non-null values
for col in columns_to_process:
    if df[col].notna().sum() < 100:
        df.drop(columns=[col], inplace=True)

# Update columns to process
columns_to_process = df.columns.difference(['casn', 'smiles', 'clean_smiles', 'Molecule'])

# Initialize DataFrame to store accuracy
accuracy_df = pd.DataFrame(columns=['Column', 'Accuracy'])
filled_df = df.copy()

# Iterate through each column
for col in tqdm(columns_to_process, desc="Processing"):
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
    if X_train.shape[0] < 5 or any(y_train.value_counts() < 5):
        # Skip if total sample count is less than 5 or any class has fewer than 5 samples
        continue

    # Use Random Forest classifier
    rf = RandomForestClassifier(random_state=2023, n_jobs=-1)
    rf.fit(X_train, y_train)

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
accuracy_df.to_csv("accuracy_df.csv", index=False)
filled_df.to_csv("filled_df_classification.csv", index=False)

# Save the model
dump(rf, 'classification_model.joblib')

