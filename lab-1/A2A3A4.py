import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings
import time
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    # Load the data
    df = pd.read_csv("lab-1/Custom_CNN_Features.csv")
    
    # Separate features and target
    X = df.filter(regex='^f\d+$')  # Select columns starting with 'f' followed by numbers
    y = df['Class Label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_classifier(y_true, y_pred, y_pred_proba=None):
    """Calculate various evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Calculate AUROC if probability predictions are available
    auroc = None
    if y_pred_proba is not None:
        try:
            auroc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except:
            auroc = "Not applicable"
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUROC': auroc
    }

def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print('='*80)

def print_metrics(metrics, prefix=""):
    """Print metrics in a formatted way"""
    print(f"\n{prefix}Metrics:")
    print('-' * 40)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:.<30} {value:.4f}")
        else:
            print(f"{metric:.<30} {value}")

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple classifiers"""
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print_section_header(f"Training and Evaluating {name}")
        
        # Measure training time
        train_start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        # Measure prediction time on training set
        pred_train_start = time.time()
        y_train_pred = clf.predict(X_train)
        y_train_pred_proba = clf.predict_proba(X_train)
        train_pred_time = time.time() - pred_train_start
        
        train_metrics = evaluate_classifier(y_train, y_train_pred, y_train_pred_proba)
        
        # Measure prediction time on test set
        pred_test_start = time.time()
        y_test_pred = clf.predict(X_test)
        y_test_pred_proba = clf.predict_proba(X_test)
        test_pred_time = time.time() - pred_test_start
        
        test_metrics = evaluate_classifier(y_test, y_test_pred, y_test_pred_proba)
        
        # Store results
        results[name] = {
            'train': train_metrics,
            'test': test_metrics,
            'timing': {
                'training_time': train_time,
                'train_prediction_time': train_pred_time,
                'test_prediction_time': test_pred_time
            }
        }
        
        # Print timing analysis
        print("\nTiming Analysis:")
        print('-' * 40)
        print(f"Training Time................... {train_time:.4f} s")
        print(f"Training Set Prediction Time.... {train_pred_time:.4f} s")
        print(f"Test Set Prediction Time....... {test_pred_time:.4f} s")
        
        # Print metrics
        print_metrics(train_metrics, "Training ")
        print_metrics(test_metrics, "Testing ")
        
        # Analyze fitting
        acc_diff = train_metrics['Accuracy'] - test_metrics['Accuracy']
        print("\nFitting Analysis:")
        print('-' * 40)
        if acc_diff > 0.05:
            status = "Overfitting"
        elif acc_diff < -0.05:
            status = "Underfitting"
        else:
            status = "Good fit"
        print(f"Status.......................... {status}")
        print(f"Accuracy Difference............. {acc_diff:.4f}")
        
    return results

def main():
    print_section_header("Loading and Preprocessing Data")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Print final summary
    print_section_header("Final Summary")
    
    # Performance summary
    print("\nAccuracy Summary:")
    print('-' * 80)
    print(f"{'Model':<15} {'Training':<12} {'Testing':<12} {'Difference':<12} {'Status':<15}")
    print('-' * 80)
    for name, metrics in results.items():
        train_acc = metrics['train']['Accuracy']
        test_acc = metrics['test']['Accuracy']
        diff = train_acc - test_acc
        status = "Overfitting" if diff > 0.05 else "Underfitting" if diff < -0.05 else "Good fit"
        print(f"{name:<15} {train_acc:>11.4f} {test_acc:>11.4f} {diff:>11.4f} {status:<15}")
    
    # Timing summary
    print("\nTiming Summary (seconds):")
    print('-' * 80)
    print(f"{'Model':<15} {'Training':<12} {'Train Pred':<12} {'Test Pred':<12}")
    print('-' * 80)
    for name, metrics in results.items():
        timing = metrics['timing']
        print(f"{name:<15} {timing['training_time']:>11.4f} {timing['train_prediction_time']:>11.4f} {timing['test_prediction_time']:>11.4f}")

if __name__ == "__main__":
    main() 