"""
Training module for political bias detection.
Trains 5 models:
1. Logistic Regression
2. LinearSVC with Calibration
3. Multinomial Naive Bayes
4. XGBoost
5. Soft Voting Ensemble
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

from preprocess import load_or_create_dataset
from features import PoliticalBiasFeatureExtractor


class ModelTrainer:
    """Trains and saves all models."""
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        self.feature_extractor = None
        self.label_encoder = None
    
    def prepare_data(self, df):
        """
        Prepare features and labels from dataset.
        
        Returns:
            X: feature matrix (sparse)
            y: label array
            label_mapping: dict mapping class names to integers
        """
        print("Preparing data...")
        
        texts = df['text'].tolist()
        labels = df['bias'].tolist()
        
        # Initialize feature extractor
        self.feature_extractor = PoliticalBiasFeatureExtractor()
        
        # Extract features
        X, _ = self.feature_extractor.extract_all_features(texts, fit_tfidf=True)
        
        # Encode labels
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        label_to_idx = self.label_encoder
        
        y = np.array([label_to_idx[label] for label in labels])
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Classes: {unique_labels}")
        
        return X, y, unique_labels
    
    def train_logistic_regression(self, X, y):
        """Train Logistic Regression model."""
        print("\n[1/5] Training Logistic Regression...")
        
        model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs',  # Works well with balanced class_weight
            n_jobs=-1
        )
        
        model.fit(X, y)
        self.models['logistic_regression'] = model
        
        train_score = model.score(X, y)
        print(f"   Training accuracy: {train_score:.4f}")
        
        return model
    
    def train_linear_svc(self, X, y):
        """Train LinearSVC with Calibration."""
        print("\n[2/5] Training LinearSVC with Calibration...")
        
        base_model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            max_iter=2000,
            random_state=42,
            dual=False
        )
        
        model = CalibratedClassifierCV(base_model, cv=5)
        model.fit(X, y)
        self.models['linear_svc'] = model
        
        train_score = model.score(X, y)
        print(f"   Training accuracy: {train_score:.4f}")
        
        return model
    
    def train_multinomial_nb(self, X, y):
        """Train Multinomial Naive Bayes (on TF-IDF portion only)."""
        print("\n[3/5] Training Multinomial Naive Bayes...")
        
        # Extract only TF-IDF features (first part of the sparse matrix)
        # TF-IDF features are the first 5000 columns (or less if we reduced)
        tfidf_features_count = min(5000, X.shape[1] - 34)  # 34 hand-crafted features
        X_tfidf = X[:, :tfidf_features_count]
        
        model = MultinomialNB(alpha=0.1)
        model.fit(X_tfidf, y)
        self.models['multinomial_nb'] = model
        self.tfidf_features_end = tfidf_features_count  # Store for later use
        
        train_score = model.score(X_tfidf, y)
        print(f"   Training accuracy: {train_score:.4f}")
        
        return model
    
    def train_xgboost(self, X, y, unique_labels):
        """Train XGBoost model."""
        print("\n[4/5] Training XGBoost...")
        
        # Convert sparse matrix to dense for XGBoost
        X_dense = X.toarray()
        
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=1,
            num_class=len(unique_labels)
        )
        
        model.fit(X_dense, y)
        self.models['xgboost'] = model
        
        train_score = model.score(X_dense, y)
        print(f"   Training accuracy: {train_score:.4f}")
        
        return model
    
    def train_voting_ensemble(self, X, y, unique_labels):
        """Train Soft Voting Ensemble."""
        print("\n[5/5] Training Soft Voting Ensemble...")
        
        # Get trained models
        lr = self.models['logistic_regression']
        svc = self.models['linear_svc']
        xgb = self.models['xgboost']
        
        # Create ensemble without MultinomialNB since it needs different features
        ensemble = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('svc', svc),
                ('xgb', xgb)
            ],
            voting='soft',
            weights=[3, 4, 3]  # Adjusted weights since NB is excluded
        )
        
        ensemble.fit(X, y)
        self.models['voting_ensemble'] = ensemble
        
        train_score = ensemble.score(X, y)
        print(f"   Training accuracy: {train_score:.4f}")
        
        return ensemble
    
    def train_all_models(self, df):
        """Train all models."""
        X, y, unique_labels = self.prepare_data(df)
        
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.train_logistic_regression(X, y)
        self.train_linear_svc(X, y)
        self.train_multinomial_nb(X, y)
        self.train_xgboost(X, y, unique_labels)
        self.train_voting_ensemble(X, y, unique_labels)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        return self.models, unique_labels
    
    def save_models(self):
        """Save all trained models to disk."""
        print("\nSaving models...")
        
        for name, model in self.models.items():
            path = self.model_dir / f"{name}.pkl"
            joblib.dump(model, path)
            print(f"   Saved: {path}")
        
        # Save feature extractor
        path = self.model_dir / "feature_extractor.pkl"
        joblib.dump(self.feature_extractor, path)
        print(f"   Saved: {path}")
        
        # Save label encoder
        path = self.model_dir / "label_encoder.pkl"
        joblib.dump(self.label_encoder, path)
        print(f"   Saved: {path}")
    
    def load_models(self):
        """Load all trained models from disk."""
        print("Loading models...")
        
        model_names = ['logistic_regression', 'linear_svc', 'multinomial_nb', 
                      'xgboost', 'voting_ensemble']
        
        for name in model_names:
            path = self.model_dir / f"{name}.pkl"
            model = joblib.load(path)
            self.models[name] = model
            print(f"   Loaded: {path}")
        
        # Load feature extractor
        path = self.model_dir / "feature_extractor.pkl"
        self.feature_extractor = joblib.load(path)
        print(f"   Loaded: {path}")
        
        # Load label encoder
        path = self.model_dir / "label_encoder.pkl"
        self.label_encoder = joblib.load(path)
        print(f"   Loaded: {path}")


if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    df = load_or_create_dataset()
    
    # Train models
    trainer = ModelTrainer()
    models, unique_labels = trainer.train_all_models(df)
    
    # Save models
    trainer.save_models()
    
    print("\nTraining pipeline complete!")
    print(f"Models saved to: {trainer.model_dir}")
