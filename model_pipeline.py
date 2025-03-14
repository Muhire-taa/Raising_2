import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

class RiskAssessmentPipeline:
    def __init__(self, config=None):
        """Initialize the pipeline with configuration parameters."""
        self.config = config or {
            'model_dir': 'models',
            'data_dir': 'data',
            'logs_dir': 'logs',
            'target_col': 'HH_Income_UGX_Day',
            'risk_threshold': 2.0,  # $2/day threshold
            'test_size': 0.25,
            'random_state': 42
        }
        
        # Setup logging
        os.makedirs(self.config['logs_dir'], exist_ok=True)
        logging.basicConfig(
            filename=f"{self.config['logs_dir']}/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('risk_assessment_pipeline')
        
        # Create directories
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['data_dir'], exist_ok=True)
        
        # Initialize model and preprocessors
        self.model = None
        self.imputer = None
        self.tfidf_vectorizer = None
        self.feature_columns = None
        self.categorical_columns = None
        self.numeric_columns = None
        
    def load_data(self, file_path):
        """Load data from file."""
        self.logger.info(f"Loading data from {file_path}")
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df, training=True):
        """Clean and prepare data for modeling."""
        self.logger.info("Starting data preprocessing")
        
        # Define column groups if first run
        if training:
            self.numeric_columns = [
                'Total_Savings_Ugx', 'Loan_Amount_Ugx', 'Interest', '#_HH_Members',
                'VSLA_Profits', 'Vegetable_Income_Ugx', 'Seasonal_Vegetable_Value_Ugx',
                'consumption_exp_monthly', 'consumption_exp_*annual', 'Total_Expenses',
                'Formal_Employment_Ugx', 'Personal_Business_&_Self_Employment_Ugx',
                'Casual_Labour_Ugx', 'Remittances_&_Gifts_Ugx', 'Rent_Income_Property_&_Land_Ugx',
                'Seasonal_Crops_Income_Ugx', 'Perenial_Crops_Income_Ugx', 'Livestock_Income_Ugx',
                'Livestock_Asset_Value', 'Assets', 'Program_Value_UGX_Day'
            ]
            
            self.categorical_columns = [
                '#2-Spouse_can_read_and_write', '#4-Material_on_Walls', '#5-Roofing_Materials',
                '#6-Fuel_Source_for_Cooking', '#7-Type_of_Toilet_Facilty', 
                '#8-Every_Member_at_least_ONE_Pair_of_Shoes', 'STATUS'
            ]
        
        # Filter to available columns
        available_numeric = [col for col in self.numeric_columns if col in df.columns]
        available_categorical = [col for col in self.categorical_columns if col in df.columns]
        
        # Impute missing values
        if training:
            self.imputer = SimpleImputer(strategy='median')
            df[available_numeric] = self.imputer.fit_transform(df[available_numeric])
        else:
            df[available_numeric] = self.imputer.transform(df[available_numeric])
        
        # Create target variable
        if self.config['target_col'] in df.columns:
            df['at_risk'] = (df[self.config['target_col']] < self.config['risk_threshold']).astype(int)
        
        # Create derived features
        self.logger.info("Creating derived features")
        self._create_derived_features(df)
        
        # Process categorical features
        self.logger.info("Processing categorical features")
        df_encoded = pd.get_dummies(df, columns=available_categorical, drop_first=True)
        
        # Process text data
        if 'most_recommend_rtv_program_reason' in df.columns:
            self.logger.info("Processing text features")
            df_encoded = self._process_text_features(df_encoded, training)
        
        # Drop non-feature columns
        columns_to_drop = ['SubmissionDate', 'starttime', 'endtime', 'pre_vid', 
                         'text_audit', 'pre_cohort', 'version', 'duration', 
                         'survey_start', 'intro_start', 'most_recommend_rtv_program_reason', 
                         'least_recommend_rtv_program_reason']
        
        drop_cols = [col for col in columns_to_drop if col in df_encoded.columns]
        df_encoded.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        # Fill any remaining missing values
        df_encoded.fillna(0, inplace=True)
        
        self.logger.info(f"Preprocessing completed. Final shape: {df_encoded.shape}")
        return df_encoded
    
    def _create_derived_features(self, df):
        """Create ratio and interaction features."""
        target_col = self.config['target_col']
        
        # Add 1 to denominators to avoid division by zero
        df['debt_to_savings_ratio'] = df['Loan_Amount_Ugx'] / (df['Total_Savings_Ugx'] + 1)
        if '#_HH_Members' in df.columns and target_col in df.columns:
            df['income_per_member'] = df[target_col] / df['#_HH_Members']
        
        if 'Total_Expenses' in df.columns and target_col in df.columns:
            df['expense_to_income_ratio'] = df['Total_Expenses'] / (df[target_col] + 1)
        
        if 'Loan_Amount_Ugx' in df.columns and target_col in df.columns:
            df['debt_to_income_ratio'] = df['Loan_Amount_Ugx'] / (df[target_col] + 1)
        
        if 'Assets' in df.columns and target_col in df.columns:
            df['assets_to_income_ratio'] = df['Assets'] / (df[target_col] + 1)
        
        if 'Seasonal_Crops_Income_Ugx' in df.columns and target_col in df.columns:
            df['agriculture_dependency'] = df['Seasonal_Crops_Income_Ugx'] / (df[target_col] + 1)
        
        if 'Livestock_Income_Ugx' in df.columns and target_col in df.columns:
            df['livestock_dependency'] = df['Livestock_Income_Ugx'] / (df[target_col] + 1)
        
        # Income source diversity score
        income_sources = [
            'Formal_Employment_Ugx', 'Personal_Business_&_Self_Employment_Ugx',
            'Casual_Labour_Ugx', 'Remittances_&_Gifts_Ugx', 'Rent_Income_Property_&_Land_Ugx',
            'Seasonal_Crops_Income_Ugx', 'Livestock_Income_Ugx'
        ]
        
        available_income_sources = [col for col in income_sources if col in df.columns]
        df['income_source_count'] = (df[available_income_sources] > 0).sum(axis=1)
    
    def _process_text_features(self, df, training=True):
        """Process text data using TF-IDF."""
        # Fill missing values
        df['most_recommend_rtv_program_reason'] = df['most_recommend_rtv_program_reason'].fillna('')
        
        # Create TF-IDF features
        if training:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            text_features = self.tfidf_vectorizer.fit_transform(df['most_recommend_rtv_program_reason'])
        else:
            text_features = self.tfidf_vectorizer.transform(df['most_recommend_rtv_program_reason'])
        
        # Convert to DataFrame
        text_df = pd.DataFrame(
            text_features.toarray(),
            columns=[f'text_feature_{i}' for i in range(text_features.shape[1])]
        )
        
        # Concatenate with main dataframe
        result = pd.concat([df, text_df], axis=1)
        return result
    
    def train_model(self, df):
        """Train the risk assessment model."""
        self.logger.info("Starting model training")
        
        # Define features and target
        all_columns = set(df.columns)
        excluded_cols = {'at_risk', self.config['target_col']}
        feature_cols = list(all_columns - excluded_cols)
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df['at_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'], 
            stratify=y
        )
        
        # Train Random Forest model
        self.logger.info("Training Random Forest model")
        self.model = RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            random_state=self.config['random_state']
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred)
        
        self.logger.info(f"Model training completed. F1 Score: {f1:.4f}")
        self.logger.info(f"Classification Report: {report}")
        
        # Save feature importance
        self._log_feature_importance()
        
        return {
            'model': self.model,
            'metrics': report,
            'f1_score': f1,
            'feature_importance': self._get_feature_importance()
        }
    
    def _log_feature_importance(self):
        """Log feature importance from the model."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            self.logger.info("Top 10 important features:")
            for i, row in feature_importance.head(10).iterrows():
                self.logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    def _get_feature_importance(self):
        """Return feature importance as a dictionary."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        return {}
    
    def predict(self, df):
        """Make predictions using the trained model."""
        self.logger.info(f"Making predictions on data with shape {df.shape}")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, training=False)
        
        # Ensure all model features are present
        missing_cols = set(self.feature_columns) - set(df_processed.columns)
        for col in missing_cols:
            df_processed[col] = 0
            
        # Select only the features used during training
        X = df_processed[self.feature_columns]
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of being at risk
        
        result = pd.DataFrame({
            'at_risk_prediction': predictions,
            'risk_probability': probabilities
        })
        
        self.logger.info(f"Predictions completed. Found {sum(predictions)} households at risk.")
        return result
    
    def save_model(self, version=None):
        """Save the model and preprocessing objects."""
        if not version:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_path = f"{self.config['model_dir']}/risk_model_{version}.joblib"
        imputer_path = f"{self.config['model_dir']}/imputer_{version}.joblib"
        tfidf_path = f"{self.config['model_dir']}/tfidf_{version}.joblib"
        config_path = f"{self.config['model_dir']}/config_{version}.joblib"
        
        # Save objects
        joblib.dump(self.model, model_path)
        joblib.dump(self.imputer, imputer_path)
        joblib.dump(self.tfidf_vectorizer, tfidf_path)
        joblib.dump({
            'feature_columns': self.feature_columns,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'config': self.config
        }, config_path)
        
        self.logger.info(f"Model and preprocessing objects saved with version {version}")
        return version
    
    def load_model(self, version):
        """Load a saved model and preprocessing objects."""
        model_path = f"{self.config['model_dir']}/risk_model_{version}.joblib"
        imputer_path = f"{self.config['model_dir']}/imputer_{version}.joblib"
        tfidf_path = f"{self.config['model_dir']}/tfidf_{version}.joblib"
        config_path = f"{self.config['model_dir']}/config_{version}.joblib"
        
        try:
            self.model = joblib.load(model_path)
            self.imputer = joblib.load(imputer_path)
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            
            saved_config = joblib.load(config_path)
            self.feature_columns = saved_config['feature_columns']
            self.numeric_columns = saved_config['numeric_columns']
            self.categorical_columns = saved_config['categorical_columns']
            
            self.logger.info(f"Model version {version} loaded successfully")
            return True
            self.logger.info(f"Model version {version} loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model version {version}: {str(e)}")
            return False
    
    def run_pipeline(self, data_path, save=True):
        """Run the full pipeline from data loading to model training."""
        self.logger.info("Starting full pipeline run")
        
        # Load data
        df = self.load_data(data_path)
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Train model
        results = self.train_model(df_processed)
        
        # Save model if requested
        version = None
        if save:
            version = self.save_model()
        
        self.logger.info("Pipeline run completed successfully")
        return {
            'version': version,
            'metrics': results['metrics'],
            'f1_score': results['f1_score'],
            'feature_importance': results['feature_importance']
        }


class ModelMonitor:
    """Monitor model performance and data drift."""
    
    def __init__(self, pipeline, config=None):
        """Initialize the monitor with a trained pipeline."""
        self.pipeline = pipeline
        self.config = config or {
            'metrics_dir': 'metrics',
            'drift_threshold': 0.1,  # Threshold for significant distribution drift
            'performance_threshold': 0.05,  # Acceptable performance degradation
        }
        
        # Setup logging and directories
        os.makedirs(self.config['metrics_dir'], exist_ok=True)
        self.logger = logging.getLogger('model_monitor')
        
        # Initialize reference distributions
        self.reference_data = None
        self.reference_predictions = None
        self.reference_metrics = None
    
    def set_reference_data(self, data):
        """Set reference data for drift comparison."""
        self.reference_data = data.copy()
        
        # Process through pipeline and get predictions
        data_processed = self.pipeline.preprocess_data(data)
        X = data_processed[self.pipeline.feature_columns]
        y_true = data_processed['at_risk'] if 'at_risk' in data_processed else None
        
        # Get predictions
        predictions = self.pipeline.model.predict(X)
        probabilities = self.pipeline.model.predict_proba(X)[:, 1]
        
        self.reference_predictions = {
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        # Calculate reference metrics if true labels available
        if y_true is not None:
            self.reference_metrics = {
                'f1': f1_score(y_true, predictions),
                'classification_report': classification_report(y_true, predictions, output_dict=True)
            }
        
        self.logger.info("Reference data and distributions set")
    
    def check_data_drift(self, new_data):
        """Check for drift in feature distributions."""
        if self.reference_data is None:
            self.logger.warning("Reference data not set. Cannot check for drift.")
            return None
        
        drift_report = {}
        
        # Compare distributions of key numeric features
        for col in self.pipeline.numeric_columns:
            if col in self.reference_data.columns and col in new_data.columns:
                # Simple mean difference as drift metric
                ref_mean = self.reference_data[col].mean()
                new_mean = new_data[col].mean()
                
                if ref_mean != 0:
                    rel_diff = abs(ref_mean - new_mean) / abs(ref_mean)
                    
                    drift_report[col] = {
                        'reference_mean': ref_mean,
                        'new_mean': new_mean,
                        'relative_difference': rel_diff,
                        'significant_drift': rel_diff > self.config['drift_threshold']
                    }
        
        # Count significant drifts
        significant_drifts = sum(1 for col in drift_report if drift_report[col]['significant_drift'])
        
        self.logger.info(f"Data drift check completed. Found {significant_drifts} features with significant drift.")
        return drift_report
    
    def check_prediction_drift(self, new_data):
        """Check for drift in model predictions."""
        if self.reference_predictions is None:
            self.logger.warning("Reference predictions not set. Cannot check for prediction drift.")
            return None
        
        # Process through pipeline and get predictions
        data_processed = self.pipeline.preprocess_data(new_data, training=False)
        X = data_processed[self.pipeline.feature_columns]
        
        # Get predictions
        predictions = self.pipeline.model.predict(X)
        probabilities = self.pipeline.model.predict_proba(X)[:, 1]
        
        # Compare prediction distributions
        ref_positive_rate = self.reference_predictions['predictions'].mean()
        new_positive_rate = predictions.mean()
        
        prediction_drift = {
            'reference_positive_rate': ref_positive_rate,
            'new_positive_rate': new_positive_rate,
            'absolute_difference': abs(ref_positive_rate - new_positive_rate),
            'significant_drift': abs(ref_positive_rate - new_positive_rate) > self.config['drift_threshold']
        }
        
        self.logger.info(f"Prediction drift check completed. Positive rate changed from {ref_positive_rate:.3f} to {new_positive_rate:.3f}")
        return prediction_drift
    
    def evaluate_performance(self, data, true_labels=None):
        """Evaluate model performance on new data."""
        # Process through pipeline and get predictions
        data_processed = self.pipeline.preprocess_data(data, training=False)
        X = data_processed[self.pipeline.feature_columns]
        
        # Get true labels if not provided
        if true_labels is None and 'at_risk' in data_processed:
            true_labels = data_processed['at_risk']
        
        if true_labels is None:
            self.logger.warning("No true labels available. Cannot evaluate performance.")
            return None
        
        # Get predictions
        predictions = self.pipeline.model.predict(X)
        
        # Calculate metrics
        f1 = f1_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # Compare with reference metrics
        performance_change = None
        if self.reference_metrics is not None:
            ref_f1 = self.reference_metrics['f1']
            performance_change = {
                'reference_f1': ref_f1,
                'new_f1': f1,
                'absolute_difference': abs(ref_f1 - f1),
                'significant_degradation': (ref_f1 - f1) > self.config['performance_threshold']
            }
        
        result = {
            'f1_score': f1,
            'classification_report': report,
            'performance_change': performance_change
        }
        
        self.logger.info(f"Performance evaluation completed. F1 score: {f1:.4f}")
        if performance_change and performance_change['significant_degradation']:
            self.logger.warning(f"Significant performance degradation detected. F1 score decreased by {performance_change['absolute_difference']:.4f}")
        
        return result
    
    def log_monitoring_results(self, results, timestamp=None):
        """Log monitoring results to file."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        log_path = f"{self.config['metrics_dir']}/monitoring_{timestamp}.joblib"
        joblib.dump(results, log_path)
        
        self.logger.info(f"Monitoring results logged to {log_path}")
        return log_path
    
    def full_monitoring_check(self, new_data, true_labels=None):
        """Run all monitoring checks and return comprehensive report."""
        data_drift = self.check_data_drift(new_data)
        prediction_drift = self.check_prediction_drift(new_data)
        performance = self.evaluate_performance(new_data, true_labels)
        
        monitoring_report = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': data_drift,
            'prediction_drift': prediction_drift,
            'performance': performance,
            'retraining_recommended': self._should_recommend_retraining(data_drift, prediction_drift, performance)
        }
        
        # Log results
        self.log_monitoring_results(monitoring_report)
        
        self.logger.info("Full monitoring check completed")
        if monitoring_report['retraining_recommended']:
            self.logger.warning("Model retraining is recommended based on monitoring results")
        
        return monitoring_report
    
    def _should_recommend_retraining(self, data_drift, prediction_drift, performance):
        """Determine if retraining is recommended based on monitoring results."""
        if data_drift is None or prediction_drift is None:
            return False
        
        # Count significant data drifts
        significant_data_drifts = sum(1 for col in data_drift if data_drift[col]['significant_drift'])
        significant_data_drift_ratio = significant_data_drifts / len(data_drift) if data_drift else 0
        
        # Check prediction drift
        significant_prediction_drift = prediction_drift.get('significant_drift', False)
        
        # Check performance degradation
        significant_degradation = False
        if performance and performance.get('performance_change'):
            significant_degradation = performance['performance_change'].get('significant_degradation', False)
        
        # Recommend retraining if any significant issues found
        return (significant_data_drift_ratio > 0.3 or 
                significant_prediction_drift or 
                significant_degradation)