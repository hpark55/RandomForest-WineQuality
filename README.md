# RandomForest-WineQualityAnalysis

In this project, I applied a Random Forest Classifier to analyze and predict the quality of red wine based on its chemical properties. The dataset contains physicochemical attributes and quality labels for different red wines.

# Dataset

I used the `winequality-red.csv` dataset, which includes the following attributes:
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

# Objective

My goals for this project were:
1. To train a Random Forest model to classify wines into two categories:
   - Low Quality (0) for wines with quality ratings 3-6.
   - High Quality (1) for wines with quality ratings 7-10.
2. To evaluate the model's performance using metrics such as accuracy, precision, recall, and a confusion matrix.
3. To explore hyperparameter optimization using GridSearchCV.
4. To analyze feature importance and how it influences wine quality predictions.

# Steps in Analysis

1. Data Preprocessing
   - Handled class imbalance by converting the multi-class quality ratings into binary categories: 0 for low quality and 1 for high quality.
   - Split the dataset into training (70%) and testing (30%) sets.
   - To ensure features were on the same scale, I normalized them using StandardScaler.

2. Random Forest Classifier
   - First trained a Random Forest model with 10 estimators using default parameters.
   - Evaluated its performance by checking the accuracy, precision, recall, and F1 score.

3. Hyperparameter Tuning
   - Used GridSearchCV to identify the best hyperparameters. Specifically, I tested:
     - Number of estimators (n_estimators)
     - Maximum depth (max_depth)
     - Maximum leaf nodes (max_leaf_nodes)
     - Splitting criterion (gini or entropy)
     - Maximum features considered for splitting (auto, sqrt, log2)
   - The best hyperparameters I found were:
     {'criterion': 'entropy', 'max_depth': 15, 'max_features': 'auto', 'max_leaf_nodes': 100, 'n_estimators': 47}

4. Performance Evaluation
   - Compared the training and testing accuracy of the model before and after hyperparameter tuning.
   - Visualized the confusion matrix to better understand the model’s predictions.

5. Feature Importance
   - Identified alcohol and sulphates as the most important features in predicting wine quality.
   - To investigate further, I analyzed the relationship between alcohol levels and wine quality using scatter plots.

# Results

- Initial Model Performance:
  - Training Accuracy: 99.2%
  - Testing Accuracy: 89.0%

- Optimized Model Performance:
  - Training Accuracy: 99.9%
  - Testing Accuracy: 90.8%

- Key Findings:
  - Discovered that wines with higher alcohol levels and sulphates were more likely to be classified as high quality.

# Visualizations

1. Confusion Matrix
   - I plotted the confusion matrix to show how well the model distinguished between low and high-quality wines.

2. Feature Importance
   - I created a bar plot to highlight the contribution of each feature to the model’s decisions.

3. Scatter Plot
   - I visualized the relationship between alcohol levels and wine quality to see how higher alcohol content correlates with higher quality.

# Tools and Libraries Used

- Python for programming
- pandas for data manipulation
- numpy for numerical computations
- scikit-learn for machine learning algorithms and evaluation metrics
- matplotlib for data visualization


