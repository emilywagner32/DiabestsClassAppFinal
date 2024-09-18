#import needed Libraries
import time
import imblearn as imb
from imblearn import over_sampling
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from io import StringIO



warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Diabetes Classifier Application",
    layout='wide')
# Title of application
st.title("Diabetes Classifier Application")
st.info("This application aims to provide insight to the Diabetes Health Indicators Dataset through data exploration. Additionally this application showcases three classificationn models used in conjunction with the data set as the goal of the project is so show how ML models can help in the classification of diabetes, which can lead to early detection and beter care.")

HELP = """
The Diabetes Health Indicators Dataset is preloaded into the application. 
Users may explore the application thorough clicking the various tabs.
In the Exploratory Data Analysis tab the user will find 
data exploration componentes such as the dataframe, insightful graphs and visuals, 
and relative information about the data such as missing values and data types. Each model has a tab dedicated to run that model and present its performance metrics.
For example the K-Nearest Neighbors tab will allow users to run the model by simply clicking the button. Upon click the model will run and the results will populate within the tab.
"""


tab5, tab1, tab2, tab3, tab4, = st.tabs(["Home","Exploratory Data Analysis ", "K-Nearest Neighbors Classifier", "Random Forest Classifier", "MLP Classifier"])


def stream_data():
    for word in HELP.split(" "):
        yield word + " "
        time.sleep(0.20)

with tab5:
    if st.button("CLICK ME FOR HELP"):
        st.write_stream(stream_data)


sentiment_mapping = ["one", "two", "three", "four", "five"]
selected = tab5.feedback("stars")

tab5.write("Rating and Feedback")
tab5.write('Please provide a rating and feedback to let us know how we can improve the application for future users')
form = tab5.form(key='my_form')
form.text_input(label='Feedback:')

tab5.submit_button = form.form_submit_button(label='Submit')

with tab5:
    if selected is not None:
        st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")

tab1.header('Diabetes Health Indicators Data Set')

@st.cache_data(ttl=3600)
def load_data(csv):
    data = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
    return data

data = load_data('diabetes_012_health_indicators_BRFSS2015.csv')
tab1.dataframe(data)


if tab1.button("Data Shape"):
    tab1.write(data.shape)

if tab1.button("Null Values within the Diabetes Health Indicators Dataset"):
    tab1.write(data.isnull().sum())

if tab1.button("Descriptive Statistics"):
    tab1.write(data.describe(include='all').T)

if tab1.button("Duplicate Values in the diabetes Health Indicators Dataset"):
    tab1.write(data.duplicated().sum())

if tab1.button("Drop duplicate values found in the Diabetes Health Indicators Dataset"):
    data.drop_duplicates(keep='first', inplace=True)
#Changing the data type
data = data.rename(columns={'Diabetes_012':'Diabetes_Class', 'CholCheck': 'Had_Chol_Check_In_Last_5_Years', 'PhysActivity': 'Phys_Activity_In_Last_30_Days', 'Fruits':'Consumes_Fruits_Daily', 'Veggies':'Consumes_Vegetables_Daily'})
with tab1.expander("Updated column names", expanded=False):
    st.write(data.columns)

df = data.astype('int64')

columns = list(data.columns)

with tab1.expander("Variable Data Types", expanded=False):
        st.write(df.dtypes)

with tab1.expander("Diabetes Health Indicators Dataframe with cleaned column names", expanded=False):
        st.dataframe(df)

print("Missing values distribution: ")
print(data.isnull().mean())
print("")

print("Column datatypes: ")


print(df)

print("Column datatypes: ")
print(df.dtypes)



# Describing the variables witth in the dataset
variable_descriptions = {
    'Diabetes_Class': ['Diabetes status', '0 = no diabetes 1 = pre-diabetes 2 = diabetes'],
    'HighBP': ['High blood pressure?', 'Yes(1)/No(0)'],
    'HighChol': ['High cholesterol? (>240 mg/dL)', 'Yes(1)/No(0)'],
    'CholCheck': ['Checked cholesterol in the past 5 years?', 'Yes(1)/No(0)'],
    'BMI': ['Body mass index', 'Continuous'],
    'Smoker': ['Smoked at least 100 cigarettes? (5 packs)', 'Yes(1)/No(0)'],
    'Stroke': ['Had a stroke or been told so?', 'Yes(1)/No(0)'],
    'HeartDiseaseorAttack': ['Had a coronary heart disease (CHD) or myocardial infarction (MI)?', 'Yes(1)/No(0)'],
    'PhysActivity': ['Done physical activity in past 30 days? (not including job)', 'Yes(1)/No(0)'],
    'Fruits': ['Consumes at least 1 fruit per day', 'Yes(1)/No(0)'],
    'Veggies': ['Consumes at least 1 vegetable per day', 'Yes(1)/No(0)'],
    'HvyAlcoholConsump': ['Heavy drinker?', 'Yes(1)/No(0)'],
    'AnyHealthcare': ['Have any kind of health care coverage', 'Yes(1)/No(0)'],
    'NoDocbcCost': ['Unable to see doctor because of cost in the past year?', 'Yes(1)/No(0)'],
    'GenHlth': ['General health description', 'Excellent/Very good/Good/Fair/Poor'],
    'MentHlth': ['Days with poor mental health in last month', 'Discrete scale: 1-30 days'],
    'PhysHlth': ['Days with poor physical health in last month', 'Discrete scale: 1-30 days'],
    'DiffWalk': ['Difficulty walking or climbing stairs?', 'Yes(1)/No(0)'],
    'Sex': ['Sex', 'Male(1)/Female(0)'],
    'Age': ['13-level age category (Split increments of 5 years)', 'Age groups description'],
    'Education': ['Education level (categorized 1-6)', 'Education levels description'],
    'Income': ['Income scale (1-8)', 'Income levels description'],
}

# Create a DataFrame from the dictionary
df_desc = pd.DataFrame.from_dict(variable_descriptions, orient='index', columns=['Description', 'Responses'])

# Add a count of each variable's occurrences in the dataset
df_desc['Data Length'] = df.count()

# Styling the DataFrame
df_styled = df_desc.fillna(0).style.format({"Data Length": "{:,.0f}"}).set_properties(**{
    'text-align': 'left',
    'white-space': 'pre-wrap',
}).set_table_styles([
    dict(selector='th', props=[('text-align', 'left')])
])

# Output the styled DataFrame
print(df_desc)

with tab1.expander('Description of the variables found in the Diabetes Health Indicators dataset'):
        st.caption("Description of features for easier comprehension of variables")
        chart_data1 = pd.DataFrame(df_desc)
        chart_data1

#Count the number of each class
class_count = df['Diabetes_Class'].value_counts()
print("Diabetes Class Count: ", class_count)

fig,ax = plt.subplots(figsize=[14,6])
class_count.plot(kind='pie', autopct='%1.1f%%', colors=['pink', 'lavender'], startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Class Counts')
plt.ylabel('')
plt.xlabel('')
plt.show()

with tab1.expander('Original Class Distribution'):
    st.caption("Distribution of classes before over sampling is used")
    visual = st.pyplot(fig)
#Balance the classes using random over sampler
ros = imb.over_sampling.RandomOverSampler(random_state=42)
data_resampled, target_resampled = ros.fit_resample(df, df['Diabetes_Class'])


print("Target Resampled")
print(target_resampled)

resampled_counts = pd.Series(target_resampled).value_counts()
print("Counts After Random Over-Sampling:")
print(resampled_counts)

#Pie chart representing the balanced data
fig1,ax = plt.subplots(figsize=[14,6])
resampled_counts.plot(kind='pie', autopct='%1.1f%%', colors=['pink', 'lavender'], startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Resampled Counts')
plt.ylabel('')
plt.xlabel('')
plt.show()

with tab1.expander('Class Distribution after Random Over-Sampling'):
    st.caption("Distribution of classes after over sampling is used to balance the classes")
    visual1 = st.pyplot(fig1)
#Create a correlation matrix used for feature selection
corr_df=data_resampled.corr()
print(corr_df)

fig2,ax = plt.subplots(figsize=[14,6])
sns.heatmap(corr_df, annot=True, fmt = '.2f', ax=ax)
sns.color_palette("rocket", as_cmap=True)
ax.set_title("Correlation Heatmap", fontsize=12)
plt.show()

with tab1.expander('Feature Selection via Correlation Matrix'):
    st.caption("Correlation matrix used to determine feature selection. Features are selected at with a value of .2 or greater, unless the variable was noted as an indicator by the WHO, CDC, or American Diabetes Association")
    visual2 = st.pyplot(fig2)

#Drop low correlated variables
Selected_df = data_resampled.drop(['Had_Chol_Check_In_Last_5_Years', 'Consumes_Fruits_Daily', 'Consumes_Vegetables_Daily', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'HvyAlcoholConsump', 'MentHlth', 'Education', 'Income', 'Smoker', 'Stroke', 'Sex', 'PhysHlth', "Age"], axis=1)

corr_df1 =Selected_df.corr()
print(corr_df1)


fig8,ax = plt.subplots(figsize=[14,6])
sns.heatmap(corr_df1, annot=True, fmt = '.2f', ax=ax)
sns.color_palette("rocket", as_cmap=True)
ax.set_title("Correlation Heatmap of only selected features", fontsize=12)
plt.show()

with tab1.expander('Selected features and their correlation'):
    st.caption("Correlation matrix to show the correlation of the selected features")
    visual2 = st.pyplot(fig8)


#Display density bar charts
fig3,ax = plt.subplots(4,2, figsize=(10, 16))
sns.distplot(Selected_df.HighBP, bins=20, ax=ax[0, 0], color="red")
sns.distplot(Selected_df.HighChol, bins=20, ax=ax[0, 1], color="red")
sns.distplot(Selected_df.BMI, bins=20, ax=ax[1, 0], color="blue")
sns.distplot(Selected_df.Phys_Activity_In_Last_30_Days, bins=20, ax=ax[1, 1], color="blue")
sns.distplot(Selected_df.HeartDiseaseorAttack, bins=20, ax=ax[2, 1], color="green")
sns.distplot(Selected_df.GenHlth, bins=20, ax=ax[3, 0], color="#b640d4")
sns.distplot(Selected_df.Diabetes_Class, bins=20, ax=ax[3, 1], color="#b640d4")
plt.show()



#Remove outliers
fig4,ax = plt.subplots(figsize=(10, 5))
sns.boxplot(Selected_df['BMI'])
plt.show()

with tab1.expander('BMI Outliers'):
        visual3 = st.pyplot(fig4)

Q1_BMI = Selected_df['BMI'].quantile(0.25)
Q3_BMI = Selected_df['BMI'].quantile(0.75)
IQR = Q3_BMI - Q1_BMI
lower_bmi = Q1_BMI - 1.5*IQR
upper_bmi = Q3_BMI + 1.5*IQR

print("Lower BMI: ", lower_bmi)
print("Upper BMI: ", upper_bmi)

upper_array_BMI = np.where(Selected_df['BMI'] >= upper_bmi)[0]
lower_array_BMI = np.where(Selected_df['BMI'] <= lower_bmi)[0]

Selected_df.drop(index=upper_array_BMI, inplace=True)
Selected_df.drop(index=lower_array_BMI, inplace=True)


fig12, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(Selected_df['BMI'])
plt.show()

with tab1.expander('BMI Outliers Removed'):
    visual5 = st.pyplot(fig12)

with tab1.expander('Bar charts used to show the density of important selected variables'):
    st.caption("Density charts for highly correlated data points")
    visual9 = st.pyplot(fig3)

with tab1.expander('Final Diabetes Health Indicators Dataframe'):
    st.caption("Cleaned, Balanced Data with feature selection applied")
    st.caption("For security purposes all identifying patient information has been removed")
    chart_data = pd.DataFrame(Selected_df)
    chart_data




# Separate X(Independent) and Y (Dependent)
x = Selected_df.drop('Diabetes_Class', axis=1)
y = Selected_df.Diabetes_Class

# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=51)


print(x)
print(y)
print(x_train.shape, x_test.shape,   y_train.shape,  y_test.shape)


# Data Scaling
scaler = StandardScaler()
scaler.fit(x_train)
X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)


@st.cache_data
def knn_classifier():
    classifier_one = KNeighborsClassifier(n_neighbors=3)
    classifier_one.fit(X_train, y_train)
    y_test_pred_one = classifier_one.predict(X_test)

    accuracy = st.write("Testing Accuracy Score", accuracy_score(y_test, y_test_pred_one))
    recall = st.write("Testing Recall", recall_score(y_test, y_test_pred_one, average='weighted'))
    f1 = st.write("Testing F1_Score:", f1_score(y_test, y_test_pred_one, average=None))
    precision = st.write("Testing Precision", precision_score(y_test, y_test_pred_one, average='micro'))
    cm_KNN = confusion_matrix(y_test, y_test_pred_one)
    fig5, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cm_KNN,
                annot=True,
                fmt='g')
    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()

    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
    plt.show()

    visual4 = st.pyplot(fig5)

    return accuracy, recall, f1, precision, cm_KNN, visual4,


@st.cache_data
def rf_classifier():
    classifier_two = RandomForestClassifier(max_depth=15, n_estimators=10, random_state=1)
    classifier_two.fit(x_train, y_train)
    y_pred_two = classifier_two.predict(X_test)
    f1_1 = st.write("F1_Score:", f1_score(y_test, y_pred_two, average=None))
    accuracy_1 = st.write("Accuracy KNN: ", accuracy_score(y_test, y_pred_two))
    precision_1 = st.write("Precision KNN: ", precision_score(y_test, y_pred_two, average='micro'))
    recall_1= st.write("Recall KNN: ", recall_score(y_test, y_pred_two, average='weighted'))
    cm_RF = confusion_matrix(y_test, y_pred_two)

    fig6, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cm_RF,
                annot=True,
                fmt='g')
    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()

    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
    plt.show()

    visual5 = st.pyplot(fig6)

    return accuracy_1, precision_1, recall_1, cm_RF, visual5, f1_1

@st.cache_data
def mlp_classifier():
    classifier_three = MLPClassifier(activation='relu', solver='lbfgs', alpha=0.0001, max_iter=1000,
                                     hidden_layer_sizes=(10,))
    classifier_three.fit(X_train, y_train)
    y_pred_three = classifier_three.predict(X_test)
    f1_3= st.write("F1_Score:", f1_score(y_test, y_pred_three, average=None))
    accuracy_3 = st.write("Accuracy KNN: ", accuracy_score(y_test, y_pred_three))
    precision_3 = st.write("Precision KNN: ", precision_score(y_test, y_pred_three, average='micro'))
    recall_3 = st.write("Recall KNN: ", recall_score(y_test, y_pred_three, average='weighted'))

    cm_MLP = confusion_matrix(y_test, y_pred_three)

    fig6, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cm_MLP,
                annot=True,
                fmt='g')
    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()

    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
    plt.show()

    visual5 = st.pyplot(fig6)

    return accuracy_3, precision_3, recall_3, cm_MLP, visual5, f1_3

#Model Building inside of Streamlit
with tab2:



    if st.button("Run KNN Classifier", key="Run KNN Classifier"):
        st.header("KNN Precision Metrics", divider=True)
        model_1 = knn_classifier(X_train, X_test, y_train, y_test)




with tab3:

    if st.button("Run Random Forest Classifier", key="Run Random Forest Classifier"):
        st.header("Random Forest Performance Metrics", divider=True)
        model_2 = rf_classifier(X_train, X_test, y_train, y_test)


with tab4:
    if st.button("Run MLP Classifier", key="Run MLP Classifier"):
        st.header("MLP Classifier Performance Metrics", divider=True)
        model_3 = mlp_classifier(X_train, X_test, y_train, y_test)

with st.sidebar:
    st.header("Diabetes Classifier Input Tool", divider=True)
    bmi = st.sidebar.slider("BMI",15, 25, 30, 35)
    high_bp = st.sidebar.slider("High Blood Pressure", 0, 1)
    high_chol = st.sidebar.slider("High Cholesterol", 0, 1)
    heart_disease_or_attack = st.sidebar.slider("Heart Disease or Attack", 0, 1)
    phsical_activity = st.sidebar.slider("Phisical Activity in the Last 30 Days", 0, 1)
    gen_hlth = st.sidebar.slider("General Health", 0, 1)

@st.cache_data
def rf_classifier_prediction(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred_tool_rf = st.sidebar.write("Random Forest Classification Prediction", rf.predict([[bmi, high_bp, high_chol, heart_disease_or_attack, phsical_activity, gen_hlth]]))

    return y_pred_tool_rf

if st.sidebar.button("Run Random Forest Classification based on input features", key="Run Random Forest Classification based on input features"):
    model_rf = rf_classifier_prediction(X_train, X_test, y_train, y_test)

@st.cache_data
def knn_classifier_prediction(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_tool_knn = st.sidebar.write("KNN Classification Predication", knn.predict([[bmi, high_bp, high_chol, heart_disease_or_attack, phsical_activity, gen_hlth]]))

    return y_pred_tool_knn

if st.sidebar.button("Run KNN Classification based on input features", key="Run KNN Classification based on input features"):
       model_knn = knn_classifier_prediction(X_train, X_test, y_train, y_test)


@st.cache_data
def mlp_classifier_prediction(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    y_pred_tool_mlp = st.sidebar.write("MLP Classification Prediction", mlp.predict([[bmi, high_bp, high_chol, heart_disease_or_attack, phsical_activity, gen_hlth]]))
    return y_pred_tool_mlp

if st.sidebar.button("Run MLP Classification based on input features", key="Run MLP Classification based on input features"):
    model_mlp = mlp_classifier_prediction(X_train, X_test, y_train, y_test)