"""
Evaluation module for political bias detection.
Performs stratified 5-fold cross-validation and generates reports.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from preprocess import load_or_create_dataset
from features import PoliticalBiasFeatureExtractor
from train import ModelTrainer


class ModelEvaluator:
    """Evaluates trained models with cross-validation."""
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.trainer = ModelTrainer(model_dir)
        self.trainer.load_models()
        self.results = {}
    
    def evaluate_with_cv(self, X, y, unique_labels, n_splits=5):
        """
        Perform stratified k-fold cross-validation on all models.
        
        Returns:
            DataFrame with evaluation results
        """
        print("\n" + "="*60)
        print("STRATIFIED 5-FOLD CROSS-VALIDATION")
        print("="*60)
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        results_data = []
        
        for model_name, model in self.trainer.models.items():
            print(f"\nEvaluating: {model_name}")
            print("-" * 40)
            
            # Handle MultinomialNB special case (needs TF-IDF only)
            if model_name == 'multinomial_nb':
                tfidf_features_count = min(5000, X.shape[1] - 34)  # 34 hand-crafted features
                X_eval = X[:, :tfidf_features_count]
            elif model_name == 'xgboost':
                X_eval = X.toarray()
            else:
                X_eval = X
            
            # Cross-validation scores
            scoring = {
                'accuracy': 'accuracy',
                'f1_macro': 'f1_macro',
            }
            
            cv_results = cross_validate(
                model, X_eval, y, cv=cv, scoring=scoring, n_jobs=-1
            )
            
            # Extract results
            accuracy_scores = cv_results['test_accuracy']
            f1_scores = cv_results['test_f1_macro']
            
            print(f"   Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")
            print(f"   F1-macro:  {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
            
            # Train on full data to get per-class F1
            if model_name == 'multinomial_nb':
                tfidf_features_count = min(5000, X.shape[1] - 34)  # 34 hand-crafted features
                X_train = X[:, :tfidf_features_count]
            elif model_name == 'xgboost':
                X_train = X.toarray()
            else:
                X_train = X
            
            model.fit(X_train, y)
            y_pred = model.predict(X_train)
            
            # Per-class F1 scores
            f1_per_class = f1_score(y, y_pred, average=None, labels=range(len(unique_labels)))
            
            for class_idx, class_name in enumerate(unique_labels):
                results_data.append({
                    'Model': model_name,
                    'Class': class_name,
                    'F1-Score': f1_per_class[class_idx]
                })
            
            self.results[model_name] = {
                'accuracy_mean': accuracy_scores.mean(),
                'accuracy_std': accuracy_scores.std(),
                'f1_macro_mean': f1_scores.mean(),
                'f1_macro_std': f1_scores.std(),
                'f1_per_class': f1_per_class,
                'y_pred': y_pred
            }
        
        return pd.DataFrame(results_data)
    
    def plot_confusion_matrix(self, y_true, y_pred, unique_labels, model_name='voting_ensemble'):
        """Plot and save confusion matrix for the ensemble model."""
        print(f"\nGenerating confusion matrix for {model_name}...")
        
        cm = confusion_matrix(y_true, y_pred, labels=range(len(unique_labels)))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_labels, yticklabels=unique_labels,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.model_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")
        plt.close()
    
    def get_top_features(self, X, y, unique_labels, n_features=20):
        """
        Get top predictive TF-IDF features for each class from Logistic Regression.
        """
        print(f"\nTop {n_features} TF-IDF features per class:")
        print("=" * 60)
        
        lr_model = self.trainer.models['logistic_regression']
        feature_names = self.trainer.feature_extractor.get_feature_names()
        
        for class_idx, class_name in enumerate(unique_labels):
            # Get coefficients for this class (one-vs-rest)
            coefficients = lr_model.coef_[class_idx]
            
            # Only consider TF-IDF features (first part of coefficients)
            tfidf_coeffs = coefficients[:len(feature_names)]
            
            # Get indices of top positive and negative features
            top_indices = np.argsort(np.abs(tfidf_coeffs))[-n_features:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_coefs = tfidf_coeffs[top_indices]
            
            print(f"\n{class_name}:")
            for feat, coef in zip(top_features, top_coefs):
                direction = "→" if coef > 0 else "←"
                print(f"  {direction} {feat:30s} ({coef:+.4f})")
        
        return feature_names
    
    def create_model_comparison_table(self, unique_labels):
        """Create summary table comparing all models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON TABLE")
        print("="*60)
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}",
                'F1-Macro': f"{metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + comparison_df.to_string(index=False))
        
        return comparison_df
    
    def print_detailed_report(self, y_true, y_pred, unique_labels, model_name):
        """Print detailed classification report."""
        print(f"\n\nDetailed Report - {model_name}:")
        print("-" * 60)
        
        report = classification_report(
            y_true, y_pred,
            target_names=unique_labels,
            digits=4
        )
        print(report)
    
    def run_full_evaluation(self, df):
        """Run complete evaluation pipeline."""
        print("\nPreparing data for evaluation...")
        
        # Prepare features
        texts = df['text'].tolist()
        labels = df['bias'].tolist()
        
        X, _ = self.trainer.feature_extractor.extract_all_features(texts, fit_tfidf=False)
        
        # Encode labels
        unique_labels = sorted(set(labels))
        label_to_idx = self.trainer.label_encoder
        y = np.array([label_to_idx[label] for label in labels])
        
        # Cross-validation
        cv_results = self.evaluate_with_cv(X, y, unique_labels, n_splits=5)
        
        # Confusion matrix for ensemble
        envelope_model = self.trainer.models['voting_ensemble']
        y_pred_ensemble = self.trainer.models['voting_ensemble'].predict(X)
        
        self.plot_confusion_matrix(y, y_pred_ensemble, unique_labels, 'Soft Voting Ensemble')
        
        # Top features
        self.get_top_features(X, y, unique_labels, n_features=20)
        
        # Model comparison
        comparison_table = self.create_model_comparison_table(unique_labels)
        
        # Detailed report for ensemble
        self.print_detailed_report(y, y_pred_ensemble, unique_labels, 'Soft Voting Ensemble')
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        return comparison_table


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_or_create_dataset()
    
    print("Creating evaluator...")
    evaluator = ModelEvaluator()
    
    print("Running evaluation...")
    comparison_table = evaluator.run_full_evaluation(df)
    
    print("\nEvaluation pipeline complete!")
