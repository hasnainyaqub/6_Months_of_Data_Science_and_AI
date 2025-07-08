import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import io

# Utility function to safely display dataframes
def safe_display(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include='object').columns:
        df_copy[col] = df_copy[col].astype(str)
    st.dataframe(df_copy)

# Step 1: Greeting and App Description
st.title("Welcome to the Machine Learning App")
st.write("This app allows you to build and evaluate ML models using your own dataset or an example dataset.")

# Step 2: Dataset Selection
data_option = st.radio("Choose your data source:", ("Upload your dataset", "Use example dataset"))

# Step 3: Upload Dataset
if data_option == "Upload your dataset":
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "tsv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith("csv"):
                for enc in ["utf-8", "latin1", "ISO-8859-1", "cp1252"]:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=enc)
                        if enc != "utf-8":
                            st.warning(f"File encoding wasn't UTF-8. Used '{enc}' instead.")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("Could not read CSV file with common encodings.")
            elif uploaded_file.name.endswith("xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith("tsv"):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                st.error("Unsupported file format.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    example = st.selectbox("Select an example dataset", ["titanic", "tips", "iris"])
    df = sns.load_dataset(example)

# Step 5: Basic Data Information
if 'df' in locals():
    st.subheader("Data Preview")
    safe_display(df.head())
    st.write("Shape:", df.shape)
    st.write("Description (numeric columns only):")
    st.write(df.describe())
    try:
        st.write("Description (all columns):")
        describe_all = df.copy()
        object_cols = describe_all.select_dtypes(include=['object']).columns
        describe_all[object_cols] = describe_all[object_cols].astype(str)
        st.dataframe(describe_all.describe(include='all').astype(str))
    except Exception as e:
        st.warning(f"Could not display full description: {e}")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("Columns:", df.columns.tolist())

    # Step 6: Feature and Target Selection
    features = st.multiselect("Select features", df.columns.tolist())
    target = st.selectbox("Select target", df.columns.tolist())

    if features and target:
        # Step 7: Problem Type Identification
        if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
            problem_type = "regression"
            st.info("This is a regression problem.")
        else:
            problem_type = "classification"
            st.info("This is a classification problem.")

        # Step 8: Pre-processing
        X = df[features]
        y = df[target]

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []
        if numeric_features:
            transformers.append(('num', Pipeline([
                ('imputer', IterativeImputer()),
                ('scaler', StandardScaler())
            ]), numeric_features))

        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        X_processed = preprocessor.fit_transform(X)
        processed_feature_names = (
            preprocessor.named_transformers_['cat']
            .get_feature_names_out(categorical_features)
            if categorical_features else []
        )
        all_feature_names = numeric_features + list(processed_feature_names) + [col for col in X.columns if col not in numeric_features + categorical_features]

        # Step 9: Train-Test Split
        test_size = st.slider("Test size (in %)", min_value=10, max_value=50, value=20) / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=42)

        # Step 10: Model Selection
        st.sidebar.title("Model Selection")
        models = {
            "SVM": SVC() if problem_type == "classification" else SVR(),
            "Random Forest": RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor(),
            "Logistic/Linear Regression": LogisticRegression() if problem_type == "classification" else LinearRegression()
        }

        selected_model_name = st.sidebar.selectbox("Select a model", list(models.keys()))
        model = models[selected_model_name]

        # Step 11: Train and Evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Evaluation")
        if problem_type == "classification":
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
            st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
            st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write("MSE:", mse)
            st.write("RMSE:", rmse)
            st.write("MAE:", mae)
            st.write("R2 Score:", r2)

        # Step 15: Save Model
        if st.checkbox("Do you want to save the model?"):
            joblib.dump(model, "trained_model.pkl")
            st.success("Model saved successfully as trained_model.pkl")

        # Step 16: Prediction
        if st.checkbox("Do you want to use the model for prediction?"):
            input_data = {}
            for feature in features:
                input_data[feature] = st.text_input(f"Enter value for {feature}")

            if all(val for val in input_data.values()):
                input_df = pd.DataFrame([input_data])
                for col in numeric_features:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

                input_processed = preprocessor.transform(input_df)
                prediction = model.predict(input_processed)
                st.write("Prediction:", prediction[0])

                st.write("Processed Features Used for Prediction:")
                st.write(all_feature_names)
