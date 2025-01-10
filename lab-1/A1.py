import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_data_types(df):
    """Analyze data types of attributes"""
    print("\n=== Data Types Analysis ===")
    print(df.dtypes)

def analyze_missing_values(df):
    """Analyze missing values in the dataset and suggest imputation strategies"""
    print("\n=== Missing Values Analysis ===")
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if len(missing_cols) > 0:
        print("Columns with missing values:")
        print(missing_cols)
        
        # Suggest imputation strategies
        numeric_cols = df[missing_cols.index].select_dtypes(include=[np.number]).columns
        categorical_cols = df[missing_cols.index].select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) > 0:
            print("\nSuggested imputation strategies for numeric columns:")
            print("- Mean imputation")
            print("- Median imputation (for skewed distributions)")
            print("- KNN imputation")
        
        if len(categorical_cols) > 0:
            print("\nSuggested imputation strategies for categorical columns:")
            print("- Mode imputation")
            print("- Create a new 'missing' category")
    else:
        print("No missing values found")

def analyze_class_balance(df):
    """Analyze class balance nature of data"""
    print("\n=== Class Balance Analysis ===")
    class_distribution = df['Class Label'].value_counts()
    print("Class Distribution:")
    print(class_distribution)
    
    # Calculate percentages
    class_percentages = (class_distribution / len(df) * 100).round(2)
    print("\nClass Distribution (%):")
    print(class_percentages)
    
    # Plot class distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.title('Class Distribution')
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')
    plt.close()

def handle_zero_imputation(df):
    """Handle zero values in features which might represent missing data"""
    print("\n=== Zero Value Analysis and Imputation ===")
    
    # Skip 'Filename' and 'Class Label' columns
    feature_cols = [col for col in df.columns if col.startswith('f')]
    
    # Calculate zero percentages per feature
    zero_percentages = (df[feature_cols] == 0).mean() * 100
    features_with_zeros = zero_percentages[zero_percentages > 0].sort_values(ascending=False)
    
    print("Features with zero values (showing top 10):")
    print(features_with_zeros.head(10).round(2))
    
    # Impute zeros with median for features that have > 50% non-zero values
    for col in feature_cols:
        non_zero_percent = (df[col] != 0).mean() * 100
        if non_zero_percent > 50:  # Only impute if majority of values are non-zero
            median_val = df[df[col] != 0][col].median()
            df[col] = df[col].replace(0, median_val)
    
    return df

def analyze_correlation(df):
    """Analyze correlation between features"""
    print("\n=== Correlation Analysis ===")
    
    # Skip 'Filename' column for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Print highly correlated features (|correlation| > 0.8)
    high_corr = np.where(np.abs(corr_matrix) > 0.8)
    high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                 for x, y in zip(*high_corr) if x != y]
    
    if high_corr:
        print("\nHighly correlated features (|correlation| > 0.8):")
        for feat1, feat2, corr in high_corr:
            print(f"{feat1} -- {feat2}: {corr:.3f}")
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def analyze_matrix_rank(df):
    """Study matrix rank"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].values
    rank = np.linalg.matrix_rank(X)
    print("\n=== Matrix Rank Analysis ===")
    print(f"Matrix rank: {rank}")
    print(f"Number of features: {len(numeric_cols)}")

def analyze_data_range(df):
    """Analyze data range and normalization aspects"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    ranges = df[numeric_cols].max() - df[numeric_cols].min()
    
    print("\n=== Data Range Analysis ===")
    print("\nFeature Ranges:")
    print(ranges)
    print("\nFeatures that might need normalization (range > 10):")
    print(ranges[ranges > 10])

def analyze_descriptive_stats(df):
    """Analyze descriptive statistics of numeric features"""
    print("\n=== Descriptive Statistics ===")
    print(df.describe())

def analyze_feature_distributions(df):
    """Visualize distributions of numeric features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'distribution_{col}.png')
        plt.close()

def main():
    df = pd.read_csv("lab-1/Custom_CNN_Features.csv")
    
    analyze_data_types(df)
    analyze_class_balance(df)
    
    # Handle zero imputation
    df = handle_zero_imputation(df)
    
    analyze_correlation(df)
    analyze_matrix_rank(df)
    analyze_data_range(df)

if __name__ == "__main__":
    main()
