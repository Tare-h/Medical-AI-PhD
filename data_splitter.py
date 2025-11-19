# data_splitter.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml

class MedicalDataSplitter:
    """
    Split medical data for training and validation while maintaining class balance
    """
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.validation_split = self.config['training']['validation_split']
        self.random_state = 42
        
    def split_patients(self, patients_df, test_size=0.2):
        """
        Split patients data while ensuring no data leakage
        """
        # Group by patient to avoid leakage
        patient_ids = patients_df['patient_id'].unique()
        
        train_ids, test_ids = train_test_split(
            patient_ids, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=patients_df.groupby('patient_id')['condition'].first()
        )
        
        train_df = patients_df[patients_df['patient_id'].isin(train_ids)]
        test_df = patients_df[patients_df['patient_id'].isin(test_ids)]
        
        return train_df, test_df
    
    def split_studies(self, studies_df, test_size=0.2):
        """
        Split medical studies data
        """
        train_studies, test_studies = train_test_split(
            studies_df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=studies_df['modality']
        )
        
        return train_studies, test_studies
    
    def create_cross_validation_folds(self, dataframe, n_folds=5):
        """
        Create cross-validation folds for medical data
        """
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        folds = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataframe, dataframe['condition'])):
            fold_data = {
                'fold': fold + 1,
                'train': dataframe.iloc[train_idx],
                'validation': dataframe.iloc[val_idx]
            }
            folds.append(fold_data)
        
        return folds
    
    def balance_dataset(self, dataframe, target_column='condition'):
        """
        Balance medical dataset using oversampling
        """
        from sklearn.utils import resample
        
        # Find the majority class
        value_counts = dataframe[target_column].value_counts()
        max_size = value_counts.max()
        
        balanced_dfs = []
        for condition, count in value_counts.items():
            if count < max_size:
                # Oversample minority class
                condition_df = dataframe[dataframe[target_column] == condition]
                oversampled_df = resample(
                    condition_df,
                    replace=True,
                    n_samples=max_size,
                    random_state=self.random_state
                )
                balanced_dfs.append(oversampled_df)
            else:
                balanced_dfs.append(dataframe[dataframe[target_column] == condition])
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=self.random_state)

# Example usage
if __name__ == "__main__":
    splitter = MedicalDataSplitter()
    
    # Example with sample data
    sample_data = pd.DataFrame({
        'patient_id': [f'PAT_{i}' for i in range(100)],
        'condition': np.random.choice(['Normal', 'Pneumonia', 'Fracture'], 100),
        'age': np.random.randint(25, 80, 100)
    })
    
    train_df, test_df = splitter.split_patients(sample_data)
    print(f"Training patients: {len(train_df)}")
    print(f"Test patients: {len(test_df)}")
