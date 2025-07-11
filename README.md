# iris-K-Nearest-Neighbors
#  Iris Flower Classification using K-Nearest Neighbors (KNN)

##  Project Title  
*Iris Flower Classification using K-Nearest Neighbors (KNN)*

##  Week 3 – Mini Project (AI & ML)  
This project develops a simple machine learning model to classify the species of iris flowers based on their sepal and petal dimensions.  
The model uses the *K-Nearest Neighbors (KNN)* algorithm, a basic supervised learning method for classification.

---

##  Objective  
To:
- Understand the basics of the KNN classification algorithm  
- Train and test a model using the popular Iris dataset  
- Predict flower species based on feature measurements  
- Visualize the data and evaluate model performance  

This project is designed for beginners exploring fundamental ML techniques.

---

##  Dataset Description  

The *Iris dataset* is built into the sklearn library and contains the following:

| Feature Name      | Description                         |
|-------------------|-------------------------------------|
| Sepal Length (cm) | Length of the flower’s sepal        |
| Sepal Width (cm)  | Width of the flower’s sepal         |
| Petal Length (cm) | Length of the flower’s petal        |
| Petal Width (cm)  | Width of the flower’s petal         |
| Species           | Iris Setosa, Versicolor, or Virginica |

###  Sample Data

| Sepal Length | Sepal Width | Petal Length | Petal Width | Species    |
|--------------|-------------|--------------|-------------|------------|
| 5.1          | 3.5         | 1.4          | 0.2         | Setosa     |
| 6.7          | 3.1         | 4.7          | 1.5         | Versicolor |
| 7.2          | 3.6         | 6.1          | 2.5         | Virginica  |

---

## 🛠 Tools and Libraries Used  

| Library      | Purpose                              |
|--------------|--------------------------------------|
| pandas       | Data handling and analysis           |
| matplotlib   | Data visualization                   |
| seaborn      | Advanced data visualization          |
| sklearn      | ML model creation and evaluation     |

##```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

---
 Machine Learning Workflow
 Dataset Loading
The Iris dataset is imported using sklearn.datasets.load_iris()

Converted into a pandas DataFrame for easy manipulation

 Data Visualization
Pairplot: Visualize the relationship between features and species

Histogram: Understand feature distributions

 Data Splitting
Split into features (X) and labels (y)

Use train_test_split() to create training and test datasets

 Model Training
Create a KNeighborsClassifier with n_neighbors=3

Train using the .fit() method

 Model Evaluation
Make predictions using .predict()

---
Evaluate performance using:

accuracy_score

confusion_matrix

ConfusionMatrixDisplay

 K-value Optimization
Experiment with different values of k (e.g., 3, 5, 7) for best accuracy

 Results
 Model Accuracy
High accuracy (typically 95%–100%) on test data

Confusion Matrix
Clearly shows correct and incorrect predictions for each class

 Project Files
File Name	Description
iris_knn_classification.ipynb	Complete code and explanations
README.md	Project documentation (this file)

---
 How to Run This Project
Open the notebook in Google Colab, Jupyter Notebook, or a local Python IDE.

Install required libraries if not already installed:

bash
Copy
Edit
pip install pandas matplotlib seaborn scikit-learn
Run each cell step-by-step to:

Visualize the data

Train and test the model

Make predictions

Evaluate results

Try changing the k value in the KNN model to improve accuracy.

⚠ Important Notes
The Iris dataset is small and used for learning/demo purposes.

KNN is simple but may not be optimal for large or complex datasets.

This project is focused on concept understanding and visualization.

---
 Author
Name: AKBAR ALI 
