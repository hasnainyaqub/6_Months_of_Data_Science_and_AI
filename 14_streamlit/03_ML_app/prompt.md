# promt 
Hey Chat Gpt, act as an application developer expert, in python usind streamlit, and build a machine learning app using scikit-learn with the following requirements:

1. Greet the user. with a welcome message. and a short description of the app.
2. Ask the user if he wants to upload a dataset or use the example dataset.
3. if the user select the upload option then ask the user to upload the dataset  in csv, xlsx, tsv or any other possible data format.
4. If the user do not upload the data then provide a default dataset selection box. this selection box should download the data from sns.load_dataset() function. the datasets should include titanic, tips, or iris.
5. Print the basic data information such as , data head, data shape , data description, datainfo and column names
6. Ask from user to select the columns as features and also the coulumns as target.
7. Identify the problem if the target column is a continuos numeric column then print  the message that this is a regression problem, otherwise print the message that this is a classification problem.
8. Pre-process the data , if the data contains any missing values then fill the missing values with the with the iterative imputer function of scikit-learn, if the features are not in the same scale then scale the features using the standard scaler function of scikit-learn, if the features are categorical variables then encode the categorical variables then encode the categorical variables using the encoder function of scikit-learn. Keep in mind to keep the encoder separate for each column as we need to inverse transform the data at the end .
9. Ask the user to provide the train test split size via slider or user input functio.
10. Ask the user to select the model from the sidebar, the model should include vector machines and some classes of models for the classification problem.
11. Train the model on the training data and evaluate  on the test data.
12. If the problem is a regression problem. use the mean-squared-error, RMSE, MAE, AUROC curve and r2-score for evaluation. if the problem is a classification problem. use the accuracy score, precision, recall, f1-score and confusion matrix for evaluation.
13, Print the evaluation metrics for each model.
14. Highlifht the best model based on the evaluation matrics.
15. Ask the user if he wants to save the model, if the user select yes then save the model using the joblib function of scikit-learn.
16. Ask the user if he wants to use the model for prediction, if the user select yes then ask the user to provide the input data and predict the output using the model.




# App Description
Built a Full-Stack ML App Using Only AI Prompts 💡🤖



Proud to share something different:

 I didn’t hand-code this app line-by-line — I created it entirely using AI prompts! 

App Flow: Dataset upload → Data Exploration → Feature/target selection →Preprocessing→ Model training → Result display → Prediction.

The app lets users:

- Upload their own datasets (CSV)

- Choose columns for training

- Train classification/regression models

- See model accuracy/results instantly

- The goal: Learn how far I could go just by prompting an AI and refining its output.

 This was a hands-on experiment in AI-assisted software development, prompt engineering, and building full-stack apps from zero using natural language.



-  Watch the demo and let me know what you think!

App[https://ml-app-a-z.netlify.app/]

#PromptEngineering #AIDevelopment #ChatGPT #MachineLearning #FullStack #ReactJS #NodeJS #Python #NoCode #LowCode #AI #DataScience