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
from sklearn.preprocessing import StandardScaler,MinMaxScaler, QuantileTransformer
from scipy import stats
from io import StringIO



warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Diabetes Classifier Application",
    layout='wide')
# Title of application
st.title("Diabetes Classifier Application")
st.info("This application aims to provide insight to the Diabetes Health Indicators Dataset. This application showcases three classificationn models used in conjunction with the data set as the goal of the project is so show how ML models can help in the classification of diabetes, which can lead to early detection and better care.")

HELP = """
The Diabetes Health Indicators Dataset is preloaded into the application. 
Users may explore the application thorough clicking the various tabs.
In the Exploratory Data Analysis tab the user will find 
data exploration componentes such as the dataframe, insightful graphs and visuals, 
and relative information about the data such as missing values and data types. Each model has a tab dedicated to run that model and present its performance metrics.
For example the K-Nearest Neighbors tab will allow users to run the model by simply clicking the button. Upon click the model will run and the results will populate within the tab.
"""


tab5, tab1, tab2, tab3, tab4 = st.tabs(["Home","Exploratory Data Analysis ", "K-Nearest Neighbors Classifier", "Random Forest Classifier", "MLP Classifier"])


pdf_file_path = "User Guide.pdf"  # Replace with your actual PDF file path

# Create a download button
# Path to your PDF file
with open(pdf_file_path, "rb") as pdf_file:
    tab5.download_button(
        label="Download Application User Guide Here",
        data=pdf_file,
        file_name="userguide.pdf",  # The name the file will have when downloaded
        mime="application/pdf"  # Specify the MIME type
    )
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
tab1.caption("all personally identifying information has been removed prior to data usage to ensure participant privacy")

@st.cache_data(ttl=3600)
def load_data(csv):
    data = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
    return data
# Load the data
data = load_data('diabetes_012_health_indicators_BRFSS2015.csv')
tab1.dataframe(data)

left,  right = tab1.columns(2, vertical_alignment="top")

left.subheader("Diabetes Health Indicators Data Set  Interactive Buttons")
right.subheader("Diabetes Health Indicators Data Set Interactive Drop-downs")
# EDA
if left.button("Data Shape"):
    left.write(data.shape)

if left.button("Null values within the Diabetes Health Indicators Dataset"):
    left.write(data.isnull().sum())

if left.button("Descriptive Statistics of original Diabetes Health Indicators Dataset"):
    left.write(data.describe(include='all').T)

if left.button("Number of duplicate values in the original Diabetes Health Indicators Dataset"):
    left.write(data.duplicated().sum())


# Describing the variables with in the dataset
data = data.rename(columns={'Diabetes_012':'Diabetes_Class', 'CholCheck': 'Had_Chol_Check_In_Last_5_Years', 'PhysActivity': 'Phys_Activity_In_Last_30_Days', 'Fruits':'Consumes_Fruits_Daily', 'Veggies':'Consumes_Vegetables_Daily', 'HeartDiseaseorAttack':'Heart_Disease_or_Attack', "HighBP":'High_BP', 'HighChol':'High_Chol', 'HvyAlcoholConsump':'Hvy_Alcohol_Consump', 'AnyHealthcare':'Any_Health_care', 'NoDocbcCost':'No_Doc_bc_Cost', 'GenHlth':'Gen_Hlth', 'MentHlth':'Ment_Hlth', 'PhysHlth':'Phys_Hlth', 'DiffWalk':'Diff_Walk'})
with right.expander("Cleaned column names", expanded=False):
    st.caption('Column names were cleaned for easier comprehension')
    st.write(data.columns)

df = data.astype('int64')

columns = list(data.columns)

with right.expander("Variable Data Types", expanded=False):
    st.write(df.dtypes)

print("Missing values distribution: ")
print(data.isnull().mean())
print("")

print("Column datatypes: ")


print(df)

print("Column datatypes: ")
print(df.dtypes)



# Describing the variables with in the dataset
variable_descriptions = {
    'Diabetes_Class': ['Diabetes status', '0 = no diabetes 1 = pre-diabetes 2 = diabetes'],
    'HighBP': ['High blood pressure', 'Yes(1)/No(0)'],
    'HighChol': ['High cholesterol: (>240 mg/dL)', 'Yes(1)/No(0)'],
    'CholCheck': ['Checked cholesterol in the past 5 years?', 'Yes(1)/No(0)'],
    'BMI': ['Body mass index', 'Continuous'],
    'Smoker': ['Smoked at least 100 cigarettes in lifetime. (5 packs)', 'Yes(1)/No(0)'],
    'Stroke': ['Had stroke or told by physician individual had a stroke', 'Yes(1)/No(0)'],
    'HeartDiseaseorAttack': ['Has coronary heart disease (CHD) or had a myocardial infarction (MI)?', 'Yes(1)/No(0)'],
    'PhysActivity': ['Participated in any physical activity in past 30 days? (not including job)', 'Yes(1)/No(0)'],
    'Fruits': ['Consumes a minimum of 1 serving of fruit per day', 'Yes(1)/No(0)'],
    'Veggies': ['Consumes a minimum of 1 serving of vegetables per day', 'Yes(1)/No(0)'],
    'HvyAlcoholConsump': ['Heavy drinker?', 'Yes(1)/No(0)'],
    'AnyHealthcare': ['Have any kind of health care coverage', 'Yes(1)/No(0)'],
    'NoDocbcCost': ['Unable to see doctor because of cost in the past year?', 'Yes(1)/No(0)'],
    'GenHlth': ['General health description', 'Excellent(1)/Very good(2)/Good(3)/Fair(4)/Poor(5)'],
    'MentHlth': ['Days with poor mental health in last month.', 'Discrete scale: 1-30 days'],
    'PhysHlth': ['Days with poor physical health in last month.', 'Discrete scale: 1-30 days'],
    'DiffWalk': ['Difficulty walking or climbing stairs.', 'Yes(1)/No(0)'],
    'Sex': ['Sex', 'Male(1)/Female(0)'],
    'Age': ['13-level age groups', 'Age groups description as indicated by the data card found on Kaggle 1: 18-24 years of age, 9: 60-64, 13: 80 or older'],
    'Education': ['Education level (categorized 1-6)', 'Education levels description as indicated by the data card found on Kaggle; 1: Never atteneded school (only Kindergarden), 2: Previous criteria and attended grades 1 through 8, 3: Previous Criteria and grades 9 through 11, 4: Previous criteria and grade 12 or similar GED, 5: Previous criteria and College (1 to 3 years), 6: Previous criteria and College Graduate '],
    'Income': ['Income scale (1-8)', 'Income levels as indicated by the data card found on Kaggle; 1: Less than $10,000, 5: less than $35,000, 8: $75,000 or more'],
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

with right.expander('Description of the variables found in the Diabetes Health Indicators dataset'):
        st.caption("Description of features for easier comprehension of variables")
        chart_data1 = pd.DataFrame(df_desc)
        chart_data1

df= df.drop_duplicates()

print(df['Diabetes_Class'].value_counts(normalize=1))

with right.expander('Diabetes Class counts'):
    st.write(df["Diabetes_Class"].value_counts(normalize=1))
# boxplots with outliers
fig4,ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data =df)
plt.xticks(rotation=90)
plt.show()


tab1.subheader('Interactive Drop-Downs with visuals')


with tab1.expander('Box plots of all features of  the Diabetes Health Indicators Dataset'):
        visual3 = st.pyplot(fig4)

# Dropping major outliers
def drop_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


df = drop_outliers_iqr(df, ['BMI', 'Ment_Hlth', 'Phys_Hlth'])

#boxplot without outliers
fig12, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()




with tab1.expander('Boxplots with extreme outliers removed'):
    visual5 = st.pyplot(fig12)


if left.button("Number of duplicate values in the cleaned Diabetes Health Indicators Dataset"):
    left.write(df.duplicated().sum())



corr_df =df.corr()
print(corr_df)





# Creation of correlation heatmap visual with all features
fig2,ax = plt.subplots(figsize=[14,6])
sns.heatmap(corr_df, annot=True, fmt = '.2f', ax=ax)
sns.color_palette("rocket", as_cmap=True)
ax.set_title("Correlation Heatmap", fontsize=12)
plt.show()

# Create a correlation bar chart so that users who are not technically savy can understand the relationship between all features and the class
fig19, ax = plt.subplots(figsize=[14,6])
plt.bar(corr_df.columns, corr_df['Diabetes_Class'])
plt.xlabel("Health Indicators")
plt.xticks(rotation=90)
plt.ylabel('Correlation with Diabetes Class')
plt.title("Correlation Bar Chart", fontsize=12)
plt.show()

# Drop low correlated variables
Selected_df = df.drop(['Had_Chol_Check_In_Last_5_Years', 'Consumes_Fruits_Daily', 'Consumes_Vegetables_Daily', 'Any_Health_care', 'No_Doc_bc_Cost', 'Education', 'Income'], axis=1)

corr_df1 =Selected_df.corr()
print(corr_df1)

# Creation of correlation heatmap visual with selected features
fig8,ax = plt.subplots(figsize=[14,6])
sns.heatmap(corr_df1, annot=True, fmt = '.2f', ax=ax)
sns.color_palette("rocket", as_cmap=True)
ax.set_title("Correlation Heatmap of only selected features", fontsize=12)
plt.show()

# Create a correlation bar chart so that users who are not technically savy can understand the relationship between selected features and the class
fig20, ax = plt.subplots(figsize=[14,6])
plt.bar(corr_df1.columns, corr_df1['Diabetes_Class'])
plt.xlabel("Health Indicators")
plt.xticks(rotation=90)
plt.ylabel('Correlation with Diabetes Class')
plt.title("Correlation Bar Chart with Selected Features", fontsize=12)
plt.show()

print(Selected_df)

# Scale Data to enhance model performance
ST = StandardScaler()

ST.fit(Selected_df)

# Preform Quantile distribution to transform data distribution to standard
QT=QuantileTransformer(n_quantiles=500,output_distribution='normal')

QT.fit(Selected_df)

# Denote X and Y variables
X = Selected_df.drop('Diabetes_Class', axis=1)
y = Selected_df.Diabetes_Class


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
ros = imb.over_sampling.RandomOverSampler(random_state=42,sampling_strategy='not majority' )
X, y = ros.fit_resample(X, y)


print("Target Resampled")
print(y)

resampled_counts = pd.Series(y).value_counts()
print("Counts After Random Over-Sampling:")
print(resampled_counts)

#Pie chart representing the balanced data
fig1,ax = plt.subplots(figsize=[14,6])
resampled_counts.plot(kind='pie', autopct='%1.1f%%', colors=['pink', 'lavender'], startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Resampled Counts')
plt.ylabel('')
plt.xlabel('')
plt.show()

with right.expander('Resampled Diabetes Class counts'):
    st.write(y.value_counts(normalize=1))

with tab1.expander('Class Distribution after Random Over-Sampling'):
    st.caption("Distribution of classes after over sampling is used to balance the classes")
    visual1 = st.pyplot(fig1)

# Split data into training and testing sets at a 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)


print(X)
print(y)
print(X_train.shape, X_test.shape,   y_train.shape,  y_test.shape)


with tab1.expander('Feature Selection via Correlation Matrix'):
    st.caption("Correlation matrix used to determine feature selection. Features are selected at with a value of .2 or greater, unless the variable was noted as an indicator by the WHO, CDC, or American Diabetes Association")
    visual2 = st.pyplot(fig2)

with tab1.expander('Bar chart showing correlation between Diabetes Class and other variables(easier comprehension for non technical users)'):
    visual19 = st.pyplot(fig19)

#Drop low correlated variables
with tab1.expander('Selected features and their correlation'):
    st.caption("Correlation matrix to show the correlation of the selected features")
    visual2 = st.pyplot(fig8)

with tab1.expander('Bar chart showing correlation between Diabetes Class and selected variables(easier comprehension for non technical users)'):
    visual20 = st.pyplot(fig20)


#Display density bar charts
fig3,ax = plt.subplots(4,2, figsize=(10, 16))
sns.distplot(Selected_df.High_BP, bins=20, ax=ax[0, 0], color="red")
sns.distplot(Selected_df.High_Chol, bins=20, ax=ax[0, 1], color="red")
sns.distplot(Selected_df.BMI, bins=20, ax=ax[1, 0], color="blue")
sns.distplot(Selected_df.Phys_Activity_In_Last_30_Days, bins=20, ax=ax[1, 1], color="blue")
sns.distplot(Selected_df.Heart_Disease_or_Attack, bins=20, ax=ax[2, 0], color="green")
sns.distplot(Selected_df.Stroke, bins=20, ax=ax[2, 1], color="green")
sns.distplot(Selected_df.Gen_Hlth, bins=20, ax=ax[3, 0], color="pink")
sns.distplot(Selected_df.Diabetes_Class, bins=20, ax=ax[3, 1], color="pink")
plt.show()



#Remove outliers

with tab1.expander('Bar charts used to show the density of important selected variables'):
    st.caption("Density charts for highly correlated data points")
    visual9 = st.pyplot(fig3)

tab1.subheader('Final Diabetes Health Indicators Dataframe')
tab1.caption("Cleaned, Balanced Data with feature selection applied")
tab1.caption("For security purposes all identifying patient information has been removed")
tab1.dataframe(data=Selected_df)






@st.cache_data
def knn_classifier(X_train, X_test, y_train, y_test):
    classifier_one = KNeighborsClassifier(n_neighbors=5, weights='distance')
    classifier_one.fit(X_train, y_train)
    y_test_pred_one = classifier_one.predict(X_test)

    accuracy = st.write("KNN Accuracy Score", accuracy_score(y_test, y_test_pred_one)*100)
    recall = st.write("KNN Recall", recall_score(y_test, y_test_pred_one, average='micro')*100)
    f1 = st.write("KNN F1_Score:", f1_score(y_test, y_test_pred_one, average='micro'))
    precision = st.write("KNN Precision", precision_score(y_test, y_test_pred_one, average='micro')*100)
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
def rf_classifier(X_train, X_test, y_train, y_test):
    classifier_two = RandomForestClassifier(max_depth=50, n_estimators=40, random_state=1)
    classifier_two.fit(X_train, y_train)
    y_pred_two = classifier_two.predict(X_test)
    accuracy_1 = st.write("Accuracy RF: ", accuracy_score(y_test, y_pred_two)*100)
    precision_1 = st.write("Precision RF: ", precision_score(y_test, y_pred_two, average='micro')*100)
    recall_1= st.write("Recall RF: ", recall_score(y_test, y_pred_two, average='weighted')*100)
    f1_1 = st.write("F1_Score:", f1_score(y_test, y_pred_two, average='micro'))
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
def mlp_classifier(X_train, X_test, y_train, y_test):
    classifier_three = MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, max_iter=1000,
                                     hidden_layer_sizes=(300,100,100,50,50,30,20,10,5))
    classifier_three.fit(X_train, y_train)
    y_pred_three = classifier_three.predict(X_test)
    accuracy_3 = st.write("Accuracy MLP: ", accuracy_score(y_test, y_pred_three)*100, '%')
    precision_3 = st.write("Precision MLP: ", precision_score(y_test, y_pred_three, average='micro')*100, '%')
    recall_3 = st.write("Recall MLP: ", recall_score(y_test, y_pred_three, average='weighted')*100, '%')
    f1_3 = st.write("F1_Score:", f1_score(y_test, y_pred_three, average='micro'))

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
    st.caption("Note: Values 0=Healthy, 1=Pre-Diabetic, 2=Diabetic")


def user_input_features():
    High_BP = st.sidebar.slider("High BP", X.High_BP.min(), X.High_BP.max())
    High_Chol = st.sidebar.slider("High Chol", X.High_Chol.min(), X_train.High_Chol.max())
    BMI = st.sidebar.slider("BMI", X_train.BMI.min(), X_train.BMI.max())
    Smoker = st.sidebar.slider("Smoker", X_train.Smoker.min(), X_train.Smoker.max())
    Stroke = st.sidebar.slider("Stroke", X_train.Stroke.min(), X_train.Stroke.max())
    Heart_Disease_or_Attack = st.sidebar.slider("Heart Disease or Attack", X_train.Heart_Disease_or_Attack.min(), X_train.Heart_Disease_or_Attack.max())
    Phys_Activity_In_Last_30_Days = st.sidebar.slider("Phys Activity in Last 30 Days", X_train.Phys_Activity_In_Last_30_Days.min(), X_train.Phys_Activity_In_Last_30_Days.max())
    Hvy_Alcohol_Consump = st.sidebar.slider("Hvy Alcohol Consump", X_train.Hvy_Alcohol_Consump.min(),
                                            X_train.Hvy_Alcohol_Consump.max())
    Gen_Hlth = st.sidebar.slider("General Health", X_train.Gen_Hlth.min(), X_train.Gen_Hlth.max())
    Ment_Hlth = st.sidebar.slider("Mental Health", X_train.Ment_Hlth.min(), X_train.Ment_Hlth.max())
    Phys_Hlth = st.sidebar.slider("Physical Health", X_train.Phys_Hlth.min(), X_train.Phys_Hlth.max())
    Diff_Walk = st.sidebar.slider("Difficulty Walking", X_train.Diff_Walk.min(), X_train.Diff_Walk.max())
    Sex = st.sidebar.slider("Sex", X_train.Sex.min(), X_train.Sex.max())
    Age = st.sidebar.slider("Age", X_train.Age.min(), X_train.Age.max())
    data_sidebar = {
            'High_BP': High_BP,
            'High_Chol': High_Chol,
            'BMI': BMI,
            'Smoker': Smoker,
            'Stroke': Stroke,
            'Heart_Disease_or_Attack': Heart_Disease_or_Attack,
            'Phys_Activity_In_Last_30_Days': Phys_Activity_In_Last_30_Days,
            'Hvy_Alcohol_Consump': Hvy_Alcohol_Consump,
            'Gen_Hlth': Gen_Hlth,
            'Ment_Hlth': Ment_Hlth,
            'Phys_Hlth': Phys_Hlth,
            'Diff_Walk': Diff_Walk,
            'Sex': Sex,
            'Age': Age}
    features = pd.DataFrame(data_sidebar, index=[0])
    return features
dfs = user_input_features()


@st.cache_data
def rf_classifier_prediction():
    rf = RandomForestClassifier(max_depth=15, n_estimators=10, random_state=1)
    rf.fit(X_train, y_train)
    y_pred_tool_rf = st.sidebar.write("Random Forest Classification Prediction", rf.predict(dfs))

    return y_pred_tool_rf

if st.sidebar.button("Run Random Forest Classification based on input features", key="Run Random Forest Classification based on input features"):
    model_rf = rf_classifier_prediction()

@st.cache_data
def knn_classifier_prediction(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
    knn.fit(X_train, y_train)
    y_pred_tool_knn = st.sidebar.write("KNN Classification Predication", knn.predict(dfs))

    return y_pred_tool_knn

if st.sidebar.button("Run KNN Classification based on input features", key="Run KNN Classification based on input features"):
       model_knn = knn_classifier_prediction(X_train, X_test, y_train, y_test)


@st.cache_data
def mlp_classifier_prediction(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, max_iter=1000,
                                     hidden_layer_sizes=(300,100,100,50,50,30,20,10,5))
    mlp.fit(X_train, y_train)
    y_pred_tool_mlp = st.sidebar.write("MLP Classification Prediction", mlp.predict(dfs))
    return y_pred_tool_mlp

if st.sidebar.button("Run MLP Classification based on input features", key="Run MLP Classification based on input features"):
    model_mlp = mlp_classifier_prediction(X_train, X_test, y_train, y_test)


