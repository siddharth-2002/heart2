import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Set up Streamlit page
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "Upload Data", "Data Cleaning", "Data Visualization", "Prediction"])

if page == "Home":
    st.title("Welcome to Heart Disease Prediction")
    st.write("This app allows you to upload data, clean and visualize it, and make predictions regarding heart disease.")
    st.write("Use the navigation menu on the left to get started:")
    st.write("- **Upload Data**: Upload your CSV file with heart disease data.")
    st.write("- **Data Cleaning**: Clean the data by handling missing values and feature selection.")
    st.write("- **Data Visualization**: Visualize the data before and after balancing.")
    st.write("- **Prediction**: Use the cleaned and balanced data to predict the risk of heart disease.")
    st.write("Click on 'Upload Data' to begin.")

elif page == "Upload Data":
    st.title("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Overview")
        st.write(df.head())
        st.session_state.df = df  # Save DataFrame to session state

elif page == "Data Cleaning":
    if 'df' not in st.session_state:
        st.warning("Please upload data first.")
    else:
        st.title("Data Cleaning")
        df = st.session_state.df

        st.write("### Initial Data Overview")
        st.write(df.head())

        # Data Preprocessing
        df.drop(['education'], axis=1, inplace=True)
        missing_data = df.isnull().sum()
        total_percentage = (missing_data.sum() / df.shape[0]) * 100
        st.write(f'Total percentage of missing data: {round(total_percentage, 2)}%')
        
        # Handle missing values
        df.dropna(axis=0, inplace=True)
        st.session_state.df = df  # Save cleaned DataFrame to session state
        
        st.write("### Cleaned Data")
        st.write(df.head())

elif page == "Data Visualization":
    if 'df' not in st.session_state:
        st.warning("Please clean the data first.")
    else:
        st.title("Data Visualization")
        df = st.session_state.df

        # Feature selection
        X = df.iloc[:, :-1]
        y = df['TenYearCHD']
        
        # Fit a RandomForest model to the data
        model = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')
        model.fit(X, y)
        
        # Select important features
        selector = SelectFromModel(model, threshold="mean", prefit=True)
        X_new = selector.transform(X)
        top_features = X.columns[selector.get_support()].tolist()
        
        st.write("### Top Features Selected:")
        st.write(top_features)
        
        X = df[top_features]
        y = df['TenYearCHD']
        
        # SMOTE for balancing
        over = SMOTE(sampling_strategy=0.8)
        under = RandomUnderSampler(sampling_strategy=0.8)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_smote, y_smote = pipeline.fit_resample(X, y)
        
        num_before = dict(Counter(y))
        num_after = dict(Counter(y_smote))
        
        st.write(f"### Numbers Before SMOTE:")
        st.write(num_before)
        st.write(f"### Numbers After SMOTE:")
        st.write(num_after)
        
        # Data Visualization
        labels = ["Negative Cases", "Positive Cases"]
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x=labels, y=list(num_before.values()), ax=ax)
        ax.set_title("Numbers Before Balancing")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x=labels, y=list(num_after.values()), ax=ax)
        ax.set_title("Numbers After Balancing")
        st.pyplot(fig)

elif page == "Prediction":
    if 'df' not in st.session_state:
        st.warning("Please clean the data first.")
    else:
        st.title("Heart Disease Prediction")
        df = st.session_state.df

        # Feature selection
        X = df.iloc[:, :-1]
        y = df['TenYearCHD']
        
        # Fit a RandomForest model to the data
        model = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')
        model.fit(X, y)
        
        # Select important features
        selector = SelectFromModel(model, threshold="mean", prefit=True)
        X_new = selector.transform(X)
        top_features = X.columns[selector.get_support()].tolist()
        
        X = df[top_features]
        y = df['TenYearCHD']
        
        # SMOTE for balancing
        over = SMOTE(sampling_strategy=0.8)
        under = RandomUnderSampler(sampling_strategy=0.8)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_smote, y_smote = pipeline.fit_resample(X, y)
        
        # Train-test split and scaling
        X_new = pd.DataFrame(X_smote, columns=top_features)
        X_new['TenYearCHD'] = y_smote
        X = X_new[top_features]
        y = X_new['TenYearCHD']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model selection
        model_type = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree"])
        
        if model_type == "Logistic Regression":
            st.subheader("Logistic Regression")
            penalty = st.selectbox("Penalty", ['l1', 'l2'])
            C = st.slider("C", 0.01, 100.0, 1.0)
            class_weight = st.selectbox("Class Weight", ['balanced', None])
            
            params = {'penalty': [penalty], 'C': [C], 'class_weight': [class_weight]}
            logistic_clf = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid=params, cv=10)
            logistic_clf.fit(X_train_scaled, y_train)
            st.write("### Best Parameters:")
            st.write(logistic_clf.best_params_)
            
            logistic_predict = logistic_clf.predict(X_test_scaled)
            log_accuracy = accuracy_score(y_test, logistic_predict)
            st.write(f"### Accuracy: {round(log_accuracy * 100, 2)}%")
            
            cm = confusion_matrix(y_test, logistic_predict)
            conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
            
            fpr, tpr, _ = roc_curve(y_test, logistic_clf.predict_proba(X_test_scaled)[:, 1])
            roc_auc = roc_auc_score(y_test, logistic_clf.predict_proba(X_test_scaled)[:, 1])
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc='lower right')
            st.pyplot(fig)
            
        elif model_type == "Decision Tree":
            st.subheader("Decision Tree")
            max_features = st.selectbox("Max Features", ['auto', 'sqrt', 'log2'])
            min_samples_split = st.slider("Min Samples Split", 2, 15, 2)
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 11, 1)
            
            params = {'max_features': [max_features], 'min_samples_split': [min_samples_split], 'min_samples_leaf': [min_samples_leaf]}
            decision_clf = GridSearchCV(DecisionTreeClassifier(random_state=7), param_grid=params, n_jobs=-1)
            decision_clf.fit(X_train_scaled, y_train)
            st.write("### Best Parameters:")
            st.write(decision_clf.best_params_)
            
            decision_predict = decision_clf.predict(X_test_scaled)
            decision_accuracy = accuracy_score(y_test, decision_predict)
            st.write(f"### Accuracy: {round(decision_accuracy * 100, 2)}%")
            
            cm = confusion_matrix(y_test, decision_predict)
            conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
        
        # User Input Section
        st.markdown('<div class="subheader">Predict Your Risk</div>', unsafe_allow_html=True)

        user_input = {}
        
        # Generate input fields for each feature
        for feature in top_features:
            if df[feature].dtype in [np.float64, np.int64]:
                user_input[feature] = st.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
            else:
                user_input[feature] = st.selectbox(f"{feature}", df[feature].unique())
        
        # Convert user input into a DataFrame
        user_data = pd.DataFrame([user_input])
        
        # Display user input data for review (optional)
        st.write("### Your Input Data:")
        st.dataframe(user_data)
        
        # Prediction button
        if st.button("Predict"):
            # Scale user input
            user_data_scaled = scaler.transform(user_data)
            
            # Predict using the selected model
            if model_type == "Logistic Regression":
                user_prediction = logistic_clf.predict(user_data_scaled)
                user_proba = logistic_clf.predict_proba(user_data_scaled)[:, 1]
            elif model_type == "Decision Tree":
                user_prediction = decision_clf.predict(user_data_scaled)
                user_proba = decision_clf.predict_proba(user_data_scaled)[:, 1]
            
            # Display the prediction result
            result = "High Risk of Heart Disease" if user_prediction[0] == 1 else "Low Risk of Heart Disease"
            st.markdown(f'### **Prediction:** {result}')
            
            # Display probability
            st.markdown(f'### **Probability of Heart Disease:** {user_proba[0] * 100:.2f}%')
            
            # Plotting the probability with colors
            fig, ax = plt.subplots()
            risk_label = "High Risk" if user_proba[0] > 0.7 else "Borderline Risk" if user_proba[0] > 0.3 else "Normal Risk"
            color_map = {"High Risk": "red", "Borderline Risk": "orange", "Normal Risk": "green"}
            color = color_map[risk_label]
            
            sns.barplot(x=[risk_label], y=[user_proba[0]], palette=[color], ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title('Probability of Heart Disease')
            st.pyplot(fig)
