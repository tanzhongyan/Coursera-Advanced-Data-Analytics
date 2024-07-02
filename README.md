# Google Advanced Data Analytics Professional [Certificate](https://coursera.org/share/2f497e1a478694f351bc06e78e173ed6)

This repository contains all projects and assignments from the Google Data Analytics Professional Certificate on Coursera. Over a period of six months and 240 hours, this course equipped me with robust analytical skills, focusing on data processing, visualisation, and sophisticated statistical analysis.

## Technical Skills Developed

- **Programming Languages**: Python
- **Python Packages**: Numpy, Pandas, SciPy, Seaborn, Matplotlib, Statsmodels, Scikit-Learn
- **Statistical Methods**: ANOVA, MANOVA, ANCOVA, MANCOVA, Chi-Square Goodness of Fit, Chi-Square Test of Independence
- **Machine Learning Models**: Linear and Logistic Regression, Naive Bayes, Decision Trees, Random Forest, AdaBoost, XGBoost

## Table of Contents

### Capstone Project: [Certificate](https://coursera.org/share/72684cadfbbe503015a46a8a4c35f716)

[Salifort Motors HR Retention](./Activity_Course_7_Salifort_Motors_project_lab)
### Foundations of Data Science: [Certificate](https://coursera.org/share/9b12149e670866754fe1779bbec8345f)

### Get Started with Python: [Certificate](https://coursera.org/share/a8fbbcc5bc5d9531f848547328c95fec)

### Go Beyond the Numbers: Translate Data into Insights: [Certificate](https://coursera.org/share/2a00d3cf1fddfe92d8f9b0b9563d68e8)
- **COURSE CONCLUSION**: [Activity_Course 3 TikTok project lab](Go%20Beyond%20the%20Numbers%20Translate%20Data%20into%20Insights/Activity_Course%203%20TikTok%20project%20lab.ipynb)
- [Activity: Address Missing Data](Go%20Beyond%20the%20Numbers%20Translate%20Data%20into%20Insights/Activity_Address_missing_data.ipynb)
- [Activity: Discover What is in Your Dataset](Go%20Beyond%20the%20Numbers%20Translate%20Data%20into%20Insights/Activity_Discover_what_is_in_your_dataset.ipynb)
- [Activity: Structure Your Data](Go%20Beyond%20the%20Numbers%20Translate%20Data%20into%20Insights/Activity_Structure_your_data.ipynb)
- [Activity: Validate and Clean Your Data](Go%20Beyond%20the%20Numbers%20Translate%20Data%20into%20Insights/Activity_Validate_and_clean_your_data.ipynb)

### The Power of Statistics: [Certificate](https://coursera.org/share/107827b0e43be6b23e2b470b5c649e6c)
- **COURSE CONCLUSION**: [Activity_Course 4 TikTok project lab](The%20Power%20of%20Statistics/Activity_Course%204%20TikTok%20project%20lab.ipynb)
- [Activity: Explore Confidence Intervals](The%20Power%20of%20Statistics/Activity_Explore_confidence_intervals.ipynb)
- [Activity: Explore Descriptive Statistics](The%20Power%20of%20Statistics/Activity_Explore_descriptive_statistics.ipynb)
- [Activity: Explore Hypothesis Testing](The%20Power%20of%20Statistics/Activity_Explore_hypothesis_testing.ipynb)
- [Activity: Explore Probability Distributions](The%20Power%20of%20Statistics/Activity_Explore_probability_distributions.ipynb)
- [Activity: Explore Sampling](The%20Power%20of%20Statistics/Activity_Explore_sampling.ipynb)

### Regression Analysis: [Certificate](https://coursera.org/share/81960a967d0946e89152d4cfd80dfa38)
- **COURSE CONCLUSION**: [Activity_Course 5 TikTok project lab](Regression%20Analysis%20Simplify%20Complex%20Data%20Relationships/Activity_Course%205%20TikTok%20project%20lab.ipynb)
- [Activity: Evaluate Simple Linear Regression](Regression%20Analysis%20Simplify%20Complex%20Data%20Relationships/Activity_Evaluate_simple_linear_regression.ipynb)
- [Activity: Hypothesis Testing with Python](Regression%20Analysis%20Simplify%20Complex%20Data%20Relationships/Activity_Hypothesis_testing_with_Python.ipynb)
- [Activity: Perform Logistic Regression](Regression%20Analysis%20Simplify%20Complex%20Data%20Relationships/Activity_Perform_logistic_regression.ipynb)
- [Activity: Perform Multiple Linear Regression](Regression%20Analysis%20Simplify%20Complex%20Data%20Relationships/Activity_Perform_multiple_linear_regression.ipynb)
- [Activity: Run Simple Linear Regression](Regression%20Analysis%20Simplify%20Complex%20Data%20Relationships/Activity_Run_simple_linear_regression.ipynb)

### Machine Learning Techniques: [Certificate](https://coursera.org/share/7e4f62c9e56fc9be9771885226a05d64)
- **COURSE CONCLUSION**: [Activity_Course 6 TikTok project lab](The%20Nuts%20and%20Bolts%20of%20Machine%20Learning/Activity_Course%206%20TikTok%20project%20lab)
- [Activity: Build a Naive Bayes Model](The%20Nuts%20and%20Bolts%20of%20Machine%20Learning/Activity_Build_a_Naive_Bayes_model)
- [Activity: Build a K-means Model](The%20Nuts%20and%20Bolts%20of%20Machine%20Learning/Activity_Build_a_K_means_model)
- [Activity: Build a Decision Tree](The%20Nuts%20and%20Bolts%20of%20Machine%20Learning/Activity_Build_a_decision_tree)
- [Activity: Build a Random Forest Model](The%20Nuts%20and%20Bolts%20of%20Machine%20Learning/Activity_Build_a_random_forest_model)
- [Activity: Build an XGBoost Model](The%20Nuts%20and%20Bolts%20of%20Machine%20Learning/Activity_Build_an_XGBoost_model)

## Key Learnings and Tools
Throughout this course, I learnt many useful skills, here is a summary of what I have learnt:

**Data Manipulation and Analysis**:
- EDA using Pandas and Numpy.
  ```
  data.shape()
  data.info()
  data.dtypes()
  data.describe()
  data.astype(dtype)
  data.value_counts()
  data.isnull()
  data.duplicated()
  data.drop()
  ```

**Data Visualisation**:
- Visualisations using Seaborn and Matplotlib:
  ```
  sns.histplot(data)
  plt.scatterplot(x="x_variable", y="y_variable", data=data)
  sns.countplot(x="variable", data=data)
  plt.bar(x="categories", height="values")
  sns.heatmap(data.corr())
  ```
  ![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

- Confusion matrix using sklearn:
  ```
  from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
  cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
  ```
  ![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

- plot_tree using sklearn:
  ```
  from sklearn.tree import plot_tree
  plt.figure(figsize=(15,12))
  plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns, class_names={0:'stayed', 1:'churned'}, filled=True);
  plt.show()
  ```
  ![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

- Decision Tree Gini Importance plot
  ```
  importance = pd.DataFrame(decision_tree.feature_importances_, 
                                 columns=['gini_importance'], 
                                 index=X.columns
                                )
  importance = importance.sort_values(by='gini_importance', ascending=False)

  # Only extract the features with importances > 0
  importance = importance[importance['gini_importance'] != 0]
  importance

  ```
  ![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

- Feature importance using XGBoost
  ```
  from xgboost import plot_importance
  plot_importance(xgb_cv.best_estimator_);
  ```
  ![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

**Machine Learning**:
- Logistic regression
  ```
  # Load in sci-kit learn functions for constructing logistic regression
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LogisticRegression

  # Save X and y data into variables
  X = activity[["Acc (vertical)"]]
  y = activity[["LyingDown"]]

  # Split dataset into training and holdout datasets
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

  clf = LogisticRegression().fit(X_train,y_train)

  # Print the coefficient (Beta 1)
  Clf.coef_ #clf means classifier

  # Print the intercept (Beta 0)
  clf.intercept_

  # Save predictions
  y_pred = clf.predict(X_test)
  ```
- Naive Bayes
  ```
  from sklearn.model_selection import train_test_split
  from sklearn.naive_bayes import GaussianNB

  # Define the y (target) variable
  y = churn_df['Exited']

  # Define the X (predictor) variables
  X = churn_df.copy()
  X = X.drop('Exited', axis=1)

  # Split into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, \
                                                      stratify=y, random_state=42)
  # Fit the model
  gnb = GaussianNB()
  gnb.fit(X_train, y_train)

  # Get the predictions on test data
  y_preds = gnb.predict(X_test)
  ```
- Decision Tree
  ```
  from sklearn.model_selection import train_test_split
  from sklearn.tree import DecisionTreeClassifier

  # This function displays the splits of the tree
  from sklearn.tree import plot_tree

  # Split into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=0.25, stratify=y, 
                                                      random_state=42)

  # Instantiate the model
  decision_tree = DecisionTreeClassifier(random_state=0)

  # Fit the model to training data
  decision_tree.fit(X_train, y_train)

  # Make predictions on test data
  dt_pred = decision_tree.predict(X_test)
  ```
- Random Forest
  ```
  from sklearn.model_selection import GridSearchCV, train_test_split
  from sklearn.ensemble import RandomForestClassifier
  rf = RandomForestClassifier(random_state=0)

  cv_params = {'max_depth': [2,3,4,5, None], 
              'min_samples_leaf': [1,2,3],
              'min_samples_split': [2,3,4],
              'max_features': [2,3,4],
              'n_estimators': [75, 100, 125, 150]
              }  

  scoring = {'accuracy', 'precision', 'recall', 'f1'}

  rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='f1')

  rf_cv.fit(X_train, y_train)
  
  rf_cv.best_params_

  rf_cv.best_score_

  rf_cv.best_estimator_ #use the best model.
  ```
- XGBoost
  ```
  # This is the classifier
  from xgboost import XGBClassifier

  # This is the function that helps plot feature importance 
  from xgboost import plot_importance
  from sklearn.model_selection import GridSearchCV, train_test_split
  xgb = XGBClassifier(objective='binary:logistic', random_state=0) 

  cv_params = {'max_depth': [4,5,6,7,8], 
              'min_child_weight': [1,2,3,4,5],
              'learning_rate': [0.1, 0.2, 0.3],
              'n_estimators': [75, 100, 125]
              }    

  scoring = {'accuracy', 'precision', 'recall', 'f1'}

  xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='f1')

  plot_importance(xgb_cv.best_estimator_);
  ```
**Saving models**
- Pickle
  ```
  import pickle

  with open('model.pkl', 'wb') as file:
      pickle.dump(model, file)
  ```
**Hyperparameter tuning**
- Decision Tree
  | Hyperparameter       | Explanation & Range of Values                                                   |
  |----------------------|---------------------------------------------------------------------------------|
  | `max_depth`          | The maximum depth of the tree, controlling overfitting. Typical values: 3, 5, 10, None (unlimited depth). |
  | `min_samples_split`  | The minimum number of samples required to split an internal node. Typical values: 2, 5, 10. |
  | `min_samples_leaf`   | The minimum number of samples required to be at a leaf node. Typical values: 1, 2, 4. |
- Random Forest
  | Hyperparameter  | Explanation & Range of Values                                                   |
  |-----------------|---------------------------------------------------------------------------------|
  | `n_estimators`  | The number of trees in the forest. Higher numbers typically lead to better performance but increase computation. Typical values: 100, 200, 500. |
  | `max_features`  | The number of features to consider when looking for the best split. Typical values: 'auto', 'sqrt', 'log2', where 'auto' uses all features. |
  | `max_depth`     | The maximum depth of each tree. Setting this value too high can cause overfitting. Typical values: 10, 20, 30, None. |
- XGBoost
  | Hyperparameter   | Explanation & Range of Values                                                   |
  |------------------|---------------------------------------------------------------------------------|
  | `learning_rate`  | Step size shrinkage used to prevent overfitting. Range is typically 0.01 to 0.3. |
  | `max_depth`      | Maximum tree depth for base learners. Helps in controlling over-fitting. Typical values: 3, 5, 7, 9. |
  | `n_estimators`   | Number of gradient boosted trees. More can lead to better performance, but may also cause overfitting. Typical values: 100, 200, 300. |
  | `subsample`      | Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees and this will prevent overfitting. Typical values: 0.5, 0.75, 1. |

## Reflection

While taking this course, I discovered that data science was not as easy as I thought it was. Here are the three take aways that I have from this cours

1. **There's no one size fit all solution, but we can rely on our intuition.**  Sometimes, there isn't even any clear direction or intuitive reason behind why some data are they way they are. Data science, like software deelopment, relies on experience and intuition, often through hard core practicing and messing around with code.

2. **Using machine learning models can be easy; it can be copy pasting code from a github. But a good understanding of how to model the data before the creation of model is vital to the model's performance.** As a common saying in my company during my internship: "Garbage in, Garbage out". This is especially true if we do not consider class imbalances and stratification of data. Sampling methods like oversampling and undersampling assist us in ensuring that our ML model doesn't churn out garbage. Additionally, taking note of missing data, fixing outliers, and normalising features are small little changes we can make to improve our model metrics. These are things that novice ML enthusiast may miss out as it seems that the creation of ML models are so easy. Just being slightly more obsessive regarding your data cleanliness and preperation can sigificantly improve model performance.

3. **You don't have to remember all the code, just refer to notes and documentations online.** Often times, we believe that we should memorise how to write code to prove that we know what we are doing. However, in the field of ML, there are so many models, so much math, and countless other methos of doing a single task. This can get overwhelming at times, especially when you're starting out. Sometimes, using a cheatsheet or documentation doesn't make you less of a professional. Instead, it leaves space for you to consider and learn other aspects of your job. Relying on ChatGPT to write your code isn't unprofessional or shows your lack of understanding, it just means that you're writing code more efficiently, while understanding how the code is operated.


## Conclusion

Throughout this course, I significantly enhanced my data analytics capabilities, particularly in understanding complex data relationships and making data-driven decisions. These projects not only solidified my technical skills but also improved my ability to translate data insights into actionable recommendations.

## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://sg.linkedin.com/in/zhong-yan-tan)
