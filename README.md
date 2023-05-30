# Body Performance

### Overview:
This project is designed to predict the body performance level based on age, gender, and other exercise performance data from kaggle. The goal is to build a model that accurately predicts the body performance index using the available data. 

### Data Analysis:
The first step is to load the dataset from a CSV file into a DataFrame. Missing values are checked, and the dataset is balanced based on gender. Data distribution in the dataset is analyzed, and categorical values are converted to numerical values.

### Data Preprocessing:
To prepare the data for the model, the training and testing data are scaled using StandardScaler to normalize their values. Scaling is performed when the dataset values have a large variation.

### Correlation Matrix:
A correlation matrix is created to identify the relationships between variables in the dataset. If there are strong dependencies (correlation above a set threshold), some variables may be considered for removal. However, in this case, all the data is included.

### GridSearchCV:
Grid Search is used to find optimal hyperparameters for machine learning models. Cross-validation is also performed to achieve the best results for the models. The Decision Tree Classifier and Random Forest Classifier models are used in this project.

### Machine Learning Model Results:
The best result is achieved by the Random Forest Classifier model, with an accuracy of 0.741 using the hyperparameters: entropy, max_depth=20, max_features=7. The Decision Tree Classifier model achieves an accuracy of 0.681. The neural network model achieves an accuracy of 0.692.

### Model Performance Evaluation:
The models are evaluated using metrics such as accuracy and the area under the ROC curve (ROC AUC). The Random Forest Classifier model achieves an accuracy of 0.735 and a ROC AUC score of 0.918. The Decision Tree Classifier model achieves an accuracy of 0.682 and a ROC AUC score of 0.856. The neural network model achieves an accuracy of 0.693 and a ROC AUC score of 0.898.

### Summary:
This project focuses on predicting body performance levels based on age, gender, and other parameters. The Random Forest Classifier model achieves the best results, but all models achieve high accuracy in predicting the body performance class. The project utilizes various libraries such as Pandas, Sklearn, Matplotlib, Seaborn, TensorFlow, Keras, and Numpy for data analysis, data preprocessing, machine learning model building, and result evaluation.
