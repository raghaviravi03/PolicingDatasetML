# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:51:01 2023

@author: raghavi

Layout of the web app:
    1. Introduction to the App - What to expect (a. Intro to dataset: Explain what the dataset is 
    about and its significance. Highlight key features like the area for which this dataset is,
    time period covered and types of data recorded. 
    b. Data Summary: statistics of the dataset, such as the total number of records, 
    unique categories of stops, top offenses, and demographic distribution.
    c. Sample Data Exploration: Explore a small sample of the dataset interactively. Include 
    filters for attributes like date, time, race, gender allowing users to see how the filters
    affect the displayed data. Show a few sample rows from the dataset to give users an idea 
    of the data's structure.
    d. Overview of the App: What we can achieve from this app?
    e. Educational Resources: 
    Stanford Open Policing Project (https://openpolicing.stanford.edu/))
    2. Demographic Analysis: a. Age Wise b. Race Wise c. Gender Wise
    Date Range Selector: Allow users to filter data within a specific date range.
    3. Time Series Trends - Search prediction, Arrest rates prediction, Drug related stop prediction
    4. Driver Behaviour Analysis - and also feature selection tab
    5. Stop Outcome Prediction
    6. 
"""

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from streamlit_option_menu import option_menu
import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet as ProphetModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error, r2_score
from prince import MCA
from sklearn.ensemble import IsolationForest
import pickle
st.set_page_config(layout="wide")

df = pd.read_csv("police_project.csv")
df['stop_date'] = pd.to_datetime(df['stop_date'], format='%Y-%m-%d')
df["stop_time"] = pd.to_datetime(df.stop_time, format="%H:%M").dt.hour
df.drop('county_name', axis=1, inplace=True)
df['search_type'].replace([np.nan], 'Search not Conducted', inplace=True)
df.dropna(inplace=True)
df["year"] = df.stop_date.dt.year
df = df.assign(bins = pd.cut(df["driver_age"], [0,5,18,64,100], 
                        labels=['0 to 5', '6 to 18', '19 to 64', '65 to 100']))
df.rename(columns={'bins':'driver_age_group'}, inplace=True)
a=[i.split(',')[0] for i in df.search_type]
df['search_type_agg']=a
race_pop={'driver_race':['White', 'Black', 'Asian', 'Hispanic', 'Other'],
    'race_population':[863105, 95783, 38945, 178936, 124202]}
df_race_pop = pd.DataFrame(race_pop)
df=df.merge(df_race_pop, on='driver_race', how='left')

age_pop={'driver_age_group':['6 to 18', '19 to 64', '65 to 100'],
    'age_population':[195777, 607331, 198935]}
df_age_pop = pd.DataFrame(age_pop)
df=df.merge(df_age_pop, on='driver_age_group', how='left')

gender_pop={'driver_gender':['M', 'F'],
    'gender_population':[516810, 535757]}

df_gender_pop = pd.DataFrame(gender_pop)
df=df.merge(df_gender_pop, on='driver_gender', how='left')

cl_df = df.copy()
out_df = df.copy()

file = open('gender_map', 'rb')
gender_map = pickle.load(file)
file.close()

file = open('age_map', 'rb')
age_map = pickle.load(file)
file.close()

file = open('race_map', 'rb')
race_map = pickle.load(file)
file.close()

file = open('violation_raw_map', 'rb')
violation_raw_map = pickle.load(file)
file.close()

file = open('violation_map', 'rb')
violation_map = pickle.load(file)
file.close()

file = open('search_conducted_map', 'rb')
search_conducted_map = pickle.load(file)
file.close()

file = open('stop_duration_map', 'rb')
stop_duration_map = pickle.load(file)
file.close()

file = open('stop_outcome_map', 'rb')
stop_outcome_map = pickle.load(file)
file.close()

file = open('drugs_related_stop_map', 'rb')
drugs_related_stop_map = pickle.load(file)
file.close()

def classfication(clf, test_percent, features, pred, var_map=None, random_state=None):
    
    resamp = pd.read_csv(f"{pred}_Resampled.csv")
    x_cols = resamp.drop("y_resampled",axis=1)
    y_resampled = resamp["y_resampled"]
    
    y_resampled=y_resampled[x_cols.stop_duration!=3]
    x_cols=x_cols[x_cols.stop_duration!=3]
    
    
    xtrain, xtest, ytrain, ytest = train_test_split(x_cols[features], y_resampled, test_size=test_percent, random_state=random_state)
    model = clf()
    
    # Initialize the MCA model
    mca = MCA(n_components=2)  # Reduce to 2 dimensions for visualization
    
    # Fit and transform the MCA model on the combined data
    xtrain = mca.fit_transform(xtrain)
    xtest = mca.transform(xtest)
    
    model.fit(xtrain, ytrain)
    
    ypreds = model.predict(xtest)
    #y_probs = model.predict_proba(xtest)[:, 1]
    
    accuracy = accuracy_score(ytest, ypreds)
    f1score = f1_score(ytest, ypreds, average='weighted')
    st.write("#### Watch how Decision Boundaries shift with different selections!")
    if len(features) >= 2:
        formatter = plt.FuncFormatter(lambda i, *args: var_map[i])
        colors=['#e56e1b','#38761d','#037ffc', '#fb1dac','#301014', '#f1c232', '#cc0000', '#79ff9a']
        colors=colors[:len(ytrain.unique())]
        custom_cmap = ListedColormap(colors)
        
        x_index = 0
        y_index = 1
        
        xtrain=np.array(xtrain)
        xtest=np.array(xtest)
        
        x0, x1 = np.meshgrid(
            np.linspace(xtrain[:, x_index].min(), xtrain[:, x_index].max(), 500).reshape(-1, 1),
            np.linspace(xtrain[:, y_index].min(), xtrain[:, y_index].max(), 500).reshape(-1, 1)
         )
         
        X_new = np.c_[x0.ravel(), x1.ravel()]
        y_pred = model.predict(X_new)
        zz = y_pred.reshape(x0.shape)
        fig, ax = plt.subplots(figsize=(10, 5))
        contour=ax.contourf(x0, x1, zz, cmap=custom_cmap, alpha=0.4)
        scatter=ax.scatter(xtrain[:, x_index], xtrain[:, y_index], c=ytrain, cmap=custom_cmap, edgecolor="black")
        colorbar = plt.colorbar(scatter, ax=ax, ticks=list(range(len(ytrain.unique()))), format=formatter)
        st.pyplot()
        
    else:
        st.write("#### Please select atleast 2 features to visualize the decision boundary")
    
    le, ri = st.columns(2, gap='large')
    st.write(f"#### Accuracy: {round(accuracy, 2)}")
    st.write(f"#### F1-Score: {round(f1score, 2)}")
    mapped=pd.Series(y_resampled.unique()).replace(var_map)
    
    conf_matrix = confusion_matrix(ytest, ypreds)
    plt.figure(figsize=(3, 3))
    sns.set(font_scale=0.5)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='PiYG',
    xticklabels=mapped, 
    yticklabels=mapped)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    le.pyplot()
    

    model2 = clf()
    model2.fit(x_cols[features], y_resampled)
    if clf != LogisticRegression and clf != GaussianNB and clf != SVC and clf != KNeighborsClassifier:
        feature_importances = model2.feature_importances_
        sorted_indices = feature_importances.argsort()[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        
        fig = go.Figure(go.Bar(
            y=sorted_features,
            x=feature_importances[sorted_indices],
            orientation='h'
        ))
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=500,
            width=500,
            margin=dict(l=100, r=20, t=50, b=50)
        )
        ri.plotly_chart(fig)

st.title("Police Stops Explorer") # App Title
with st.sidebar:
    selected = option_menu("Menu", ["Homepage","Summary Statistics","Demographic Analysis","Time Series Forecasting","Drug Related Stops Classification", "Search Occurrence Classification", "Stop Outcome Classification", "Traffic Violation Classification", "Customized Predictions"],
                           icons=['house-fill', 'gear', 'people-fill', 'clock', 'capsule-pill', 'search', 'sign-stop-fill', 'stoplights-fill', 'person-fill-gear'], menu_icon="cast",
                          default_index=0,
                          orientation="vertical")

if selected == "Homepage":
    # App Description
    selected_home = option_menu(None, ["Goal",
    "Overview", "Data Explorer", "About Me", "Resources"],
    icons=['house-fill', 'search', 'globe', 'person-circle', 'book-fill'], menu_icon="cast",
    default_index=0, orientation="horizontal")
    
    if selected_home == "Goal":
       # img = Image.open()
        st.image("AI_Policing_Top.jpg", caption="Traffic Police Stops Analysis")
        st.subheader("Motivation behind the web app")
        st.markdown('<div style="text-align: justify;"> Every day, law enforcement in the U.S. conducts over 50,000 traffic stops. According to the Time Magazine, while many of these stops are conducted to ensure road safety, a significant portion is initiated for reasons unrelated to drivers behavior on the road. </div>', unsafe_allow_html=True)
        st.markdown('#####')
        st.subheader("Significance of the web app")
        st.markdown('<div style="text-align: justify;"> By visualizing and summarizing policing data, the app raises awareness about law enforcement activities, including patterns in stops, demographics, and potential biases. This awareness can lead to informed discussions and actions for positive change. </div>', unsafe_allow_html=True)
        st.markdown('#####')
        st.markdown('<div style="text-align: justify;"> Utilizing data-driven insights, the web app can aid in identifying potential patterns and trends related to criminal activities. By understanding these patterns, law enforcement agencies and communities can work together to implement proactive strategies and interventions aimed at preventing crimes. This approach focuses on community-wide safety initiatives promoting a safer environment for everyone </div>', unsafe_allow_html=True)
        st.markdown('######')
        st.subheader("Brief description of the dataset")
        st.markdown('<div style="text-align: justify;"> This web app displays information and insights about the traffic stops in the state of Rhode Island. The dataset encapsulates a decade worth of information, covering the period from January 2005 to December 2015. The data recorded in this project includes, </div>', unsafe_allow_html=True)
        st.markdown("- Date and Time: Information about the date and time when the police stop occurred.")
        st.markdown("- Driver Demographics: Information about the driver, such as race, gender, age, and ethnicity.")
        st.markdown("- Reason for Stop: The primary reason for the police stop, such as speeding, traffic violation, suspicious behavior, etc.")
        st.markdown("- Stop Outcome: The outcome of the stop, including whether a citation was issued, a warning given, or an arrest made.")
        st.markdown("- Search Details: If a vehicle or individual was searched, the reasons for the search and whether any contraband or illegal items were found.")
        
        st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        url_home = "https://dornsife.usc.edu/news/stories/ai-meets-policing/"
        st.write("[Credits: The above image was taken from this website](%s)" % url_home)
        
    elif selected_home == "Overview":
        images = [
            'de.png',
            'ta.png',
            'ss.png',
            'od.png',
            'tsf.png',
            'clal.png',
            'cp.png',
            'te.jpg'
        ]
        
        image_descriptions = {
        'de.png': "Get familiarized with the dataset",
        'ta.png': "Guide to the technical concepts used in the web app",
        'ss.png': "Summary of dataset",
        'od.png': "Understanding odd data points",
        'tsf.png': "Jump to this section to perform time series analysis",
        'clal.png': "Explore this section to perform classification",
        'cp.png': "Customize the inputs and make predictions",
        'te.jpg': "Move to the next tab"
        }
        
        current_image_index = st.session_state.get('image_index', 0)
        
        col1, col2 = st.columns([9.2, 0.8])
        
        col1.markdown(f'<p style="text-align: center;background-color:#f4cccc;font-size:35px;font-color:#ffffff;"> {image_descriptions.get(images[current_image_index])}</p>', unsafe_allow_html=True)
        col1.image(images[current_image_index], caption=f'Image {current_image_index + 1}', use_column_width=True)

        
        #col1.image(images[current_image_index], caption=f'Image {current_image_index + 1}', use_column_width=True)
        st.write("Check out the tab highlighted in red to identify the section")
        if col2.button('Next'):
            current_image_index = (current_image_index + 1) % len(images)
            st.session_state['image_index'] = current_image_index
        
        url_te = "https://www.backstage.com/magazine/article/a-guide-to-movie-credits-75625/"
        st.write("[Credits: The end image was taken from this website](%s)" % url_te)
        #selected_image = images[current_image_index]
        #description = image_descriptions.get(selected_image, "Explore")
        #st.write(f"### {description}")
        
    elif selected_home == "Data Explorer":
        st.write("**Welcome to the Data Explorer section, where you have the power to navigate the dataset according to your preferences.**")
        st.write("**Export the selected data:** Need to analyze the data offline or share your findings? Export your customized results in CSV format, for further analysis or presentation!")
        selected_de = st.multiselect('Select columns to see the data present in it', df.columns,
                                     ['stop_date', 'driver_gender', 'driver_age', 'driver_race', 'violation'])

        choose_year = st.slider('Choose a year to see the data', min_value=2005, max_value=2015, value=2010)

        st.dataframe(df[selected_de][df.year==choose_year].head(10))        
        
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df[selected_de][df.year==choose_year])
        customized_button = st.markdown("""
                                <style>
                                .stDownloadButton, div.stButton {text-align:right}
                                .stDownloadButton button, div.stButton > button:first-child {
                                background-color: #000000;
                                color:#FFFFFF;
                                padding-left: 20px;
                                padding-right: 20px;
                                }
    
                            .stDownloadButton button:hover, div.stButton > button:hover {
                            background-color: #ADD8E6;
                            color:#000000;
                            }
                        </style>""", unsafe_allow_html=True)
                        
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='open_policing_dataset_rhode_island.csv',
            mime='text/csv',
            )
    
    elif selected_home == "About Me":
        st.write("I am **Raghavi Ravi**, a driven Data Science explorer with a knack for translating complex data into impactful insights. Currently pursuing a Master's in Data Science at Michigan State University, I’m on a mission to inspire positive change through data.")
        
        st.write("My journey includes steering pivotal studies where I've strategically crafted product design roadmaps, conducted comprehensive customer behavior analyses, and crafted compelling, data-driven narratives for marketing strategies. My contributions led to substantial profit increases and a notable enhancement in user satisfaction.")
        st.write("I've actively engaged stakeholders, bridging the gap between technical insights and practical business applications. Proficient in Python, SQL, R, Tableau and Machine Learning, I've honed my skills to translate complex data into actionable insights. Leveraging these tools, I've navigated intricate datasets, extracted meaningful patterns, and formulated data-driven strategies.")
        st.write("Eager to contribute my expertise in predictive analytics, market research, and strategic decision-making, I'm passionate about harnessing data to innovate and drive impactful change.")
        
        url_lp = "https://www.linkedin.com/in/raghaviravi/"
        st.write("[Check out my LinkedIn page for more details!](%s)" % url_lp)
        
    
    elif selected_home == "Resources":
        st.write("For additional insights into this dataset, click the link below,")
        url1 = "https://openpolicing.stanford.edu/data/"
        st.write("[Stanford Open Policing Project](%s)" % url1)
        
        st.subheader("Articles and Research Papers")
        st.write("Comprehensive research papers discussing racial disparities in policing practices.")
        url2 = "https://5harad.com/papers/policing-the-police.pdf"
        st.write("[Combatting Police Discrimination in the age of Big Data](%s)" % url2)
        url3 = "https://5harad.com/papers/simple-rules.pdf"
        st.write("[Simple Rules to guide expert classifications](%s)" % url3)
        
        st.subheader("Community Safety Programs")
        st.write("Information about neighborhood watch programs and how communities can work together for safety.")
        url4 = "https://bja.ojp.gov/sites/g/files/xyckuh186/files/Publications/NSA_NW_Manual.pdf"
        st.write("[Neighborhood Watch Program](%s)" % url4)
        
        st.subheader("Educational Videos")
        st.write("An educational video explaining legal rights during police stops and interactions.")
        url5 = "https://www.youtube.com/watch?v=f26QJKREYB8&t=8s"
        st.write("[Know your rights!](%s)" % url5)
        
        st.subheader("References")
        st.write("Data used for this analysis is obtained from the below references:")
        st.write("https://openpolicing.stanford.edu/publications/")
        st.write("https://github.com/stanford-policylab/opp")
        st.write("https://www.kaggle.com/datasets/yassershrief/dataset-of-traffic-stops-in-rhode-island")
        
elif selected == "Summary Statistics":    
    selected_stat = option_menu(None, ["Tech Alert !!!",
    "Stat Snap", "Outliers Detection"],
    icons=['cpu-fill', 'percent', 'people-fill', 'bar-chart-line-fill'], menu_icon="cast",
    default_index=0, orientation="horizontal")
    
    if selected_stat == "Tech Alert !!!":
        st.write("**Welcome to the Tech Alert Section, your comprehensive guide to metrics, machine learning and forecasting models, and the distinctions between classification and regression. Explore this section to unlock the meaning behind different algorithms.**")
        
        st.write("## What is Time Series Forecasting?")
        st.write("Time series forecasting is a powerful method used to predict future values based on patterns derived from historical data collected at consistent time intervals. It involves analyzing past trends, seasonality, and other patterns to make predictions about future outcomes.")
    
        l, r = st.columns(2)
        l.image("timeseries.png", width=500)
        r.markdown('<p style="background-color:#efefef;color:#000000;font-size:25px;border-radius:2%;">Ever heard of FBProphet?</p>', unsafe_allow_html=True)
        r.markdown('<div style="text-align: justify;"> It is like a crystal ball for your time-based data! Developed by Facebook, this tool simplifies predicting future values by breaking down trends, seasonal patterns, and even holiday effects in your data. It is user-friendly and handles irregularities like a pro. Ready to uncover insights and forecast trends? Dive into the time series forecasting section of the web app and let FBProphet do its magic!</div>', unsafe_allow_html=True)
        r.markdown('####')
        
        url_time = "https://www.turing.com/kb/comprehensive-guide-to-time-series-analysis-in-Python"
        r.write("[Credits: This image was taken from this website](%s)" % url_time)
        
        lt, rt = st.columns(2, gap="large")
        rt.write("## Classification vs Regression")
        st.write("Think of classification as sorting data into specific categories, like deciding if an email is spam or not. Regression, on the other hand, is like predicting actual numbers - think house prices based on various details. They are both methods machines use to interpret data differently!")
        
        l, r = st.columns(2)
        l.markdown('<p style="background-color:#efefef;font-size:25px;">Figuring out if our models are doing a good job!</p>', unsafe_allow_html=True)
        #l.markdown('<p style="font-size:18px;">Accuracy</p>', unsafe_allow_html=True)
        l.markdown('<div style="text-align: justify;"> <b>Accuracy:</b> It tells us how well a model sorts items correctly. It is like counting how many times it guessed right out of everything it tried.</div>', unsafe_allow_html=True)
        #l.markdown('<p style="font-size:18px;">F1 Score</p>', unsafe_allow_html=True)
        l.markdown('<div style="text-align: justify;"> <b>F1 Score:</b> Imagine you are into gaming and you want to get a high score in a game that combines precision and speed. The F1 score is like aiming for a great balance between those two. You want to be quick, but also accurate. It blends precision and recall to find a balanced score where you are both fast and accurate to win the game! </div>', unsafe_allow_html=True)
        #l.markdown('<p style="font-size:18px;">RMSE and R-squared Error</p>', unsafe_allow_html=True)
        l.markdown('<div style="text-align: justify;"> <b>RMSE and R-squared Error:</b> In regression, RMSE shows how far off predictions are on average, while R-squared gives a big-picture view of how much the model captures the data variability.</div>', unsafe_allow_html=True)
        l.markdown("######")
        l.markdown('<div style="text-align: justify;"> These metrics help us see where our model shines and where it might need a little tune-up.</div>', unsafe_allow_html=True)
        r.image("regvsclass.png", width=585)
        url_reg = "https://www.ris-ai.com/regression-and-classification"
        r.write("[Credits: This image was taken from this website](%s)" % url_reg)
        r.markdown('####')
        st.write("## Confusion matrix??")
        st.write("A confusion matrix is like a report card for a classifier—a way to see how well it's doing with its predictions. Picture a grid that shows where the model gets things right and where it falters.")
        
        l, r = st.columns(2, gap="large")
        l.image("conf.png", width=520)
        url_c = "https://www.biosymetrics.com/blog/binary-classification-metrics"
        l.write("[Credits: This image was taken from this website](%s)" % url_c)
        r.markdown('<p style="background-color:#efefef;font-size:25px;">Decoding a confusion matrix</p>', unsafe_allow_html=True)
        r.markdown('<div style="text-align: justify;"> In this grid, you have got four sections: true positives, true negatives, false positives, and false negatives. </div>', unsafe_allow_html=True)
        r.markdown("- **True Positives - TP:** When the model correctly identifies the presence of the disease in a person")
        r.markdown("- **True Negatives - TN:** When the model correctly determines that a person does not have the disease")
        r.markdown("- **False Positives - FP:** When the model incorrectly detects the presence of the disease in a person who actually doesn't have it")
        r.markdown("- **False Negatives - FN:** When the model incorrectly indicates that a person doesn't have the disease when, in reality, they do have it.")
        
        lt, rt = st.columns(2, gap="large")
        rt.write("## What are Decision Boundaries?")
        st.write("Look at this image where a machine learning model has drawn a border, like lines on a map, to separate Class A from Class B. These borders precisely indicate where the model decides Class A from Class B, helping categorize data points accurately.")
        
        l, r = st.columns(2)
        l.markdown('<p style="background-color:#efefef;font-size:25px;">Understanding how models divide data!</p>', unsafe_allow_html=True)
        l.markdown('<div style="text-align: justify;">For instance, in a scenario where you are distinguishing between cats and dogs based on their weights and heights, the decision boundary could be a line that separates areas where the model predicts "cat" from where it predicts "dog." The scatter plot in this image shows data points, each marked as either Class Dog or Class Cat. </div>', unsafe_allow_html=True)
        l.markdown('<div style="text-align: justify;"> But, here is the catch: sometimes the line is not exactly spot-on. It might wiggle a bit or not perfectly divide the points into clear "this side is Dog" and "that side is Cat" sections. That is why you can see some points on the wrong side of the line</div>', unsafe_allow_html=True)
        #l.markdown('<p style="font-size:18px;">RMSE and R-squared Error</p>', unsafe_allow_html=True)
        l.markdown('<div style="text-align: justify;"> <b>RMSE and R-squared Error:</b> In regression, RMSE shows how far off predictions are on average, while R-squared gives a big-picture view of how much the model captures the data variability.</div>', unsafe_allow_html=True)
        l.markdown("######")
        l.markdown('<div style="text-align: justify;"> These metrics help us see where our model shines and where it might need a little tune-up.</div>', unsafe_allow_html=True)
        r.image("boun.png", width=585)
        url_b = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdKpVQg8HtvXiwOwpotTaoaGd6bKRysFPFNN-OhMnC-60P0ULlF9W1agjKyI_3JBIorGM&usqp=CAU"
        r.write("[Credits: This image was taken from this website](%s)" % url_b)
        r.markdown('####')
        
    elif selected_stat == "Stat Snap":
        st.write("**Explore the Summary Statistics page, your gateway to key insights about the dataset. Here, you'll find a concise overview that provides some context about the data at your fingertips.**")
        
        st.subheader("What You'll Discover:")
        
        st.markdown("1. **Total Number of Records:** Get an understanding of the dataset's scale. Learn how many records are available, giving you an idea of the dataset's scope and depth.")
        st.markdown("- The dataset comprises 91,741 recorded stops. However, this analysis is based on 86,113 usable rows due to the presence of missing values in certain fields.")
        st.markdown("2. **Common Values for some columns like:**")
        st.markdown("- **Unique Categories of Stops:** The reason behind the stops include *Speeding*, *Moving Violation*, *Equipment*, *Registration*, and *Seat Belt*")
        st.markdown("- **Demographic Information:** *Race* includes values such as White, Black, Hispanic, Asian, and other. While, *gender* which includes values such as Male and Female. Also, *age* ranging from 15 to 99.")
        st.markdown("3. **Column Overview:** **Reason for Stops:**")
        st.markdown("- *Speeding - 56%*")
        st.markdown("- *Moving Violation - 18.7%*")
        st.markdown("- *Equipment - 12.7%*")
        st.markdown("- *Registration/Plates - 3.9%*")
        st.markdown("- *Seat belt - 3.4%*")
        st.markdown("- *Other - 4.9%*")
        st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        st.markdown("4. **Column Overview:** **Outcome of Stops:**")
        st.markdown("- *Citation - 89%*")
        st.markdown("- *Warning - 6%*")
        st.markdown("- *Arrest Driver - 2.9%*")
        st.markdown("- *Other - 3%*")
        st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        
    elif selected_stat == "Outliers Detection":
        def outliers_mca(data, feat):
            if len(feat) >= 3:
                X = data[feat]
                label_encoder = LabelEncoder()
                for column in X.columns:
                    X[column] = label_encoder.fit_transform(X[column])
                
                mca = MCA(n_components=2)  # Reduce to 2 dimensions for visualization
                
                mca_data = mca.fit_transform(X)
                model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
                model.fit(X)
                
                # Predict outliers on the combined dataset
                outliers = model.predict(X)
                
                # Visualize the MCA results with outliers
                plt.figure(figsize=(12, 6))
                
                # Plot normal data points
                plt.scatter(mca_data.iloc[:, 0], mca_data.iloc[:, 1], label='Normal', alpha=0.5)
                
                # Highlight outliers
                outlier_indices = X.index[outliers == -1]
                outliers_mca = mca_data.loc[outlier_indices]
                plt.scatter(outliers_mca.iloc[:, 0], outliers_mca.iloc[:, 1], label='Outliers', color='red', alpha=0.7)
                
                plt.title('Visualization of Outliers')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.legend()
                plt.show()
                st.pyplot()
                violation_outliers = data.loc[outlier_indices].groupby(feat).size().reset_index(name='count')
                violation_outliers = violation_outliers.sort_values(by='count', ascending=False)
                st.write("Each row in the table represents a outlier data point, allowing you to delve into these outliers, understand their context, and grasp their influence on the dataset. Get ready to embrace the unique insights that outliers reveal about the data!")
                st.table(violation_outliers.head(10))
            else:
                st.write("## Please select atleast 3 features to visualize the outliers")
        
        st.write("### Step into the Outliers Detection Section")
        st.write("**This is where the data gets intriguing. You have the power to select specific features, visualize the data, and uncover outliers effortlessly.**")
        st.write("Explore your dataset visually, handpick features, and witness outliers standing out in the interactive table. Select your parameters, visualize, and dive into the outlier table to unveil the hidden stories within your dataset.")
        feat=st.multiselect("Play with the options below to find any irregularities in traffic stops that might indicate unusual or potentially problematic policing behavior. Select atleast 3 features!", ['driver_gender','driver_race','violation','search_type','stop_outcome','is_arrested','stop_duration','driver_age_group'], ['stop_outcome','driver_age_group','search_type','driver_race','is_arrested'])
        outliers_mca(out_df, feat)
        st.write("**Note:** I have applied both MCA and Isolation Forest in this analysis")
        exp = st.checkbox('**What is MCA and Isolation Forest?**')
        
        if exp:
            st.markdown("* **MCA: Multiple Correspondence Analysis:** Think of it as a detective helping us uncover relationships in categorical data. Imagine we have data not in numbers but in categories like violation type, stop outcome, true-or-false values. MCA steps in to find patterns in this kind of data, showing us which categories are linked or tend to go together.")
            st.markdown("- **Why can't we use PCA?** PCA is great with numerical data, looking for patterns and reducing dimensions. But when it comes to categories or things that can't be measured numerically PCA might get confused. MCA steps in for this special kind of detective work, making sense of these categorical puzzles where numbers alone wouldn't cut it.")
            st.markdown("* **Isolation Forests:** This is like a security guard for outliers in your dataset. It specializes in pinpointing those rare instances that stand out or behave differently from the majority.")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)

        
elif selected == "Demographic Analysis":
    selected_dem = option_menu(None, ["Searches vs Stops", "Stop Duration", "Arrests vs Stops", "Speeding trends"], 
    default_index=0, orientation="horizontal")
    
    if selected_dem == "Searches vs Stops":
        st.subheader("Percentage of searches per stops")
        st.markdown('<div style="text-align: justify;"> Analyze the percentage of searches concerning the total number of stops. You have the flexibility to customize your analysis based on specific range of year and different demographic factors, such as age, gender, or race. </div>', unsafe_allow_html=True)
        st.markdown("#####")
        st.write("*Choose the range of years you are interested in analyzing. You can focus on a range of years to observe trends over time.*")
        min_max_year = st.slider('Select the Year Range:', 2005, 2015, (2007, 2012))
        st.markdown("####")
        st.write("*Select whether you want to analyze the data based on age, gender, or race. This choice will determine how the data is categorized and visualized.*")
        choose_hue = st.selectbox('Choose Demographic Factor:', ('driver_age_group', 'driver_race', 'driver_gender'))
        
        if choose_hue == 'driver_age_group':
            pop = 'age_population'
        elif choose_hue == 'driver_race':
            pop = 'race_population'
        else:
            pop = 'gender_population'
        
        temp = 100*df.groupby(by=['year',choose_hue])['search_conducted'].sum()/(df.groupby(by=['year',choose_hue]).max()[pop])
        temp = temp.reset_index().set_index('year')
        temp = temp[(temp.index>=min_max_year[0]) & (temp.index<=min_max_year[1])]
        
        fig = px.line(temp,color=choose_hue)
        fig.update_xaxes(showgrid=False, linecolor = "#BCCCDC")
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_yaxes(showgrid=True,gridcolor='#BCCCDC',gridwidth=0.3, linecolor = "#BCCCDC")
        if min_max_year[0]<2009:
            fig.add_vline(x=2009, line_dash="dash")
        st.write("**You can notice that the trends shift in the year 2009**")
        fig.update_layout(yaxis_title='Search rates')
        st.plotly_chart(fig)
        st.write("**Note:** The figures depicted in this graph have been standardized according to the census data representing the different racial, age and gender groups in Rhode Island.")
        
        st.subheader("Let's uncover the story behind the graph!")
        st.markdown("1. We observe a notable trend indicating that individuals within the age range of 65 to 100 years are consistently exempt from searches.")
        st.markdown("2. Individuals between the ages of 19 and 64 are searched more frequently compared to those in the age brackets of 6 to 18 or 65 to 100.")
        st.markdown("3. The graph reveals that black individuals are the most frequently searched, followed by Hispanics, and then Asians and whites.")
        st.markdown("The observed trend where individuals in the age groups 65 to 100 are never searched can be influenced by several factors such as,")
        st.markdown("- **Assumed Lower Risk:** Law enforcement officers might perceive elderly individuals as lower-risk in terms of criminal activity, leading to fewer searches in this age group.")
        st.markdown("- **Respect for Elderly:** There might be a cultural or societal norm that emphasizes respect for the elderly, resulting in fewer searches out of courtesy and regard for their age.")
        st.markdown("- **Health Considerations:** Older individuals may have health issues, and conducting searches could be physically challenging or potentially harmful, leading officers to avoid such procedures.")
        
        st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        
    elif selected_dem == "Arrests vs Stops":
        st.subheader("Percentage of arrests per stops")
        st.markdown('<div style="text-align: justify;"> Explore the percentage of arrests concerning the total number of police stops. </div>', unsafe_allow_html=True)
        st.markdown("#####")
        
        st.write("*Choose the range of years you are interested in analyzing. You can focus on a range of years to observe trends over time.*")
        min_max_year = st.slider('Select the Year Range:', 2005, 2015, (2005, 2008))
        st.markdown("####")
        st.write("*Select whether you want to analyze the data based on age, gender, or race. This choice will determine how the data is categorized and visualized.*")
        choose_hue = st.selectbox('Choose the demographic factor:', ('driver_age_group', 'driver_race', 'driver_gender'))
        
        if choose_hue == 'driver_age_group':
            pop = 'age_population'
        elif choose_hue == 'driver_race':
            pop = 'race_population'
        else:
            pop = 'gender_population'
        
        temp = 100*df.groupby(by=['year',choose_hue])['is_arrested'].sum()/(df.groupby(by=['year',choose_hue]).max()[pop])
        temp = temp.reset_index().set_index('year')
        temp = temp[(temp.index>=min_max_year[0]) & (temp.index<=min_max_year[1])]
        fig = px.line(temp,color=choose_hue)
        fig.update_xaxes(showgrid=False, linecolor = "#BCCCDC")
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_yaxes(showgrid=True,gridcolor='#BCCCDC',gridwidth=0.3, linecolor = "#BCCCDC")
        fig.update_layout(yaxis_title='Arrest rate')
        st.plotly_chart(fig)
        st.write("**Note:** The figures depicted in this graph have been standardized according to the census data representing the different racial, age and gender groups in Rhode Island.")
        
        st.subheader("Let's uncover the story behind the graph!")
        st.markdown("1. In 2006, the arrest rates for Black individuals notably decreased, although they continue to be higher than those for Hispanics, Whites, and Asians, with Hispanics having the second highest arrest rates among these groups.")
        st.markdown("2. The arrest rates for men significantly exceed those for women, standing at nearly three to four times the rate at which women are arrested.")
        st.markdown("3. Individuals between the ages of 19 and 64 exhibit higher arrest rates compared to any other age group, while those in the age group of 65 to 100 are rarely subject to arrest.")
        
        st.markdown("The reduction in arrest rates for Black individuals could be influenced by a range of factors like,")
        st.markdown("- **Policy Reforms:** Law enforcement agencies might have implemented policy changes, emphasizing community engagement, diversion programs, or rehabilitation over strict enforcement, leading to reduced arrests.")
        st.markdown("- **Community Outreach:** Increased efforts in community policing and engagement programs might have fostered trust and cooperation between law enforcement and minority communities, reducing confrontational interactions and subsequent arrests.")
        st.markdown("- **Legal Reforms:** Changes in laws or reforms related to non-violent offenses, drug offenses, or sentencing guidelines might have reduced the number of arrests across all racial groups.")
        
        st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        
        
    elif selected_dem == "Stop Duration":
        st.subheader("Trends in average stop duration across years")
        st.markdown('<div style="text-align: justify;"> Delve into the average stop duration concerning different periods. This interactive feature allows you to examine how the average duration of police stops has evolved over the chosen time frame. </div>', unsafe_allow_html=True)
        st.markdown("#####")
        
        st.write("*Choose the range of years you are interested in analyzing. You can focus on a range of years to observe trends over time.*")
        min_max_year = st.slider('Select the Year Range:', 2005, 2015, (2005, 2010))
        st.markdown("####")
        st.write("*Select whether you want to analyze the data based on age, gender, or race. This choice will determine how the data is categorized and visualized.*")
        choose_hue = st.selectbox('Choose the demographic factor:', ('driver_age_group', 'driver_race', 'driver_gender'))
        
        temp = df.groupby(by=['year',choose_hue])['stop_time'].mean()
        temp = temp.reset_index().set_index('year')
        temp = temp[(temp.index>=min_max_year[0]) & (temp.index<=min_max_year[1])]
        fig = px.line(temp,color=choose_hue)
        fig.update_xaxes(showgrid=False, linecolor = "#BCCCDC")
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_yaxes(showgrid=True,gridcolor='#BCCCDC',gridwidth=0.3, linecolor = "#BCCCDC")
        fig.update_layout(yaxis_title='Stop duration')
        st.plotly_chart(fig)
        
        st.subheader("Let's uncover the story behind the graph!")
        st.markdown("1. For individuals within the age group of 6 to 18, the duration of police stops is higher in comparison to other age groups.")
        st.markdown("2. For elderly individuals, specifically those aged between 65 and 100, the duration of police stops is higher than people belonging to the age groups 19 to 64.")
        st.markdown("3. The duration of police stops is longer for women in comparison to men.")
        
        st.markdown("The longer stop durations for individuals aged 6 to 18 could be influenced by several factors:")
        st.markdown("- **Parental Involvement:** Police officers might spend more time ensuring the safety and well-being of minors, which often involves interacting with parents or guardians, checking their identities, and confirming the child's relationship with them.")
        st.markdown("- **Document Verification:** Officers may need to verify identification documents and contact parents or legal guardians to confirm the minor's identity, which can prolong the stop duration.")
        st.markdown("- **Child Safety Protocols:** Law enforcement officers could be following specific protocols and procedures when dealing with minors, requiring additional time to confirm the child's safety, identity, and accompanying adult's authorization.")
        
        st.markdown("The fluctuations in stop duration for individuals aged 65 to 100 could be influenced by various factors such as,")
        st.markdown("- **Health Conditions:** Older individuals may have health issues or mobility challenges, leading to longer stop durations as they might need more time to provide information or exit the vehicle.")
        st.markdown("- **Communication Difficulties:** Seniors might experience hearing or communication difficulties, requiring officers to spend more time ensuring clear understanding, thereby increasing the stop duration.")
        st.markdown("- **Assistance Requirements:** Older individuals might need assistance in retrieving documents or complying with requests, contributing to extended stop durations.")
        
        st.markdown("The longer stop durations for women compared to men could be influenced by a variety of factors like,")
        st.markdown("- **Documentation Verification:** Officers might spend more time verifying identification documents or licenses for women, especially if there are name changes due to marriage or other reasons.")
        st.markdown("- **Safety Concerns:** Law enforcement officers could exercise extra precautions with women, particularly during late hours, to ensure their safety, which might involve more thorough checks and inquiries.")
        st.markdown("- **Search Procedures:** If a search is required, female officers might be called to conduct the search on women, which could take additional time.")
        
        st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        
    elif selected_dem == "Speeding trends":
        st.subheader("Understanding Speeding Patterns Across Demographics!")
        st.markdown('<div style="text-align: justify;"> Compare speeding incidents between different races or genders that you select. Explore the disparities in speeding patterns by comparing one race to another or analyzing gender categories comprehensively. </div>', unsafe_allow_html=True)
        st.markdown("#####")
        
        st.write("*Choose whether you want to analyze speeding patterns based on races or genders. This selection will determine the demographic factor that forms the basis of your analysis.*")
        choose_column = st.selectbox('Select Speeding Analysis Type', ('driver_race', 'driver_gender'))
        
        column1 = st.selectbox('Pick a parameter to define the x-axis variable for your analysis', df[choose_column].unique())
        column2 = st.selectbox('Pick a parameter to define the y-axis variable for your analysis', df[choose_column].unique())
        
        if choose_column == 'driver_age_group':
            pop = 'age_population'
        elif choose_column == 'driver_race':
            pop = 'race_population'
        else:
            pop = 'gender_population'
        
        d = {}
        for i in df[choose_column].unique():
            dataframe = df[(df[choose_column] == i)]
            dataframe['Speeding_violation'] = dataframe.violation=='Speeding'
            across_ethnicities = dataframe.groupby('driver_age')['Speeding_violation'].sum()/dataframe.groupby(['driver_age',choose_column])[pop].max()
            across_ethnicities=across_ethnicities.to_frame().reset_index()
            data = pd.DataFrame(columns=["age", choose_column]) 
            data.loc[:, 'age'] = np.arange(101) 
            data[choose_column] = i
            data1=data.merge(across_ethnicities, left_on='age', right_on='driver_age')
            d[i] = data1

        temp=d[column1].merge(d[column2],how='outer',on='age')
        fig=px.scatter(temp,x='0_x',y='0_y',size='age',width=400,height=400, 
                   labels={'0_x':column1, '0_y':column2})
        
        if temp['0_x'].max()>temp['0_y'].max():
            val=temp['0_x'].max()
        else:
            val=temp['0_y'].max()
            
        if temp['0_x'].min()<temp['0_y'].min():
            val2=temp['0_x'].min()
        else:
            val2=temp['0_y'].min()
            
        if temp['0_x'].std()>temp['0_y'].std():
            sd=temp['0_x'].std()
        else:
            sd=temp['0_y'].std()
        
        fig.update_yaxes(range=[val2-sd, val+sd])
        fig.update_xaxes(range=[val2-sd, val+sd])
        fig.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}])
        st.plotly_chart(fig)
        
elif selected == "Time Series Forecasting":
    st.write("#### Welcome to the Forecasting Section!")
    
    selected_tsf = option_menu(None, ["Search Frequency Forecast", "Arrest Frequency Forecast", "Drugs Related Stop Forecast"], 
    default_index=0, orientation="horizontal")
    
    st.write("**Firstly, you have two options: Test and Forecast**")
    
    st.markdown("1. **Test:** If you are keen on checking things out before diving into forecasting, this is the place to start.")
    st.markdown("2. **Forecast:** If your test results look good and you are ready to forecast, this is where the magic happens!")
    st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
    
    if selected_tsf == "Search Frequency Forecast":

        def plot(data, freq_level='Weekly', n_intervals=100, col='driver_race', var='White', forecast=True, test_percentage=None, start_year=2012):
            
            data = data[data[col] == var]
            
            if freq_level == 'Daily':
                frequency = 'D'
            elif freq_level == 'Weekly':
                frequency = 'W'
            elif freq_level == 'Monthly':
                frequency = 'M'
            elif freq_level == 'Quarterly':
                frequency = 'Q'
            elif freq_level == 'Half-yearly':
                frequency = '6M'
            elif freq_level == 'Yearly':
                frequency = 'Y'
            else:
                st.error("Please choose an appropriate value at which you want to forecast the values")
                return
            
            data['date'] = pd.to_datetime(data['stop_date'], format='%Y%'+frequency)
            daily_data = data.groupby([pd.Grouper(key='date', freq=frequency), col])['search_conducted'].sum().reset_index()
            daily_data = daily_data.set_index('date')['search_conducted']
            
            daily_data.index = pd.to_datetime(daily_data.index)
            daily_data = daily_data.asfreq(frequency)
            daily_data = daily_data.fillna(0)
            dates = daily_data.index
            
            start_date = str(start_year)
            
            #daily_data = daily_data[start_date:]  # Filtering data from the selected start year
            
            if not forecast:
                test_size = int(len(daily_data) * (test_percentage / 100))
                
                # Testing the model for specified test set size
                train_data = daily_data[:-test_size]
                test_data = daily_data[-test_size:]
                
                train_data = train_data.reset_index()
                train_data.columns = ['ds', 'y']
                
                model = ProphetModule()
                model.fit(train_data)
                
                future = model.make_future_dataframe(periods=test_size, freq=frequency)
                forecast = model.predict(future)
                
                forecast_test = forecast[-test_size:]
                
                train_data=train_data.set_index('ds')
                train_data=train_data[start_date:]
                train_data = train_data.reset_index()
                train_data.columns = ['ds', 'y']
                
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_data['ds'], y=train_data['y'], mode='lines', line=dict(color='blue'), name='Actual values of Train'))
                fig.add_trace(go.Scatter(x=test_data.index, y=test_data.values, mode='lines', line=dict(color='green'), name='Actual values of Test'))
                fig.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'], mode='lines', line=dict(color='red'), name='Forecasted values of Test'))
                
                fig.update_layout(title='Estimated Search Occurrences', xaxis=dict(title='Date'), yaxis=dict(title=f'Search Frequency for {var}'), width=1100)
                st.plotly_chart(fig)
                
                rmse=np.sqrt(mean_squared_error(test_data.values,forecast_test['yhat']))
                    
                st.write(f'#### The RMSE is {np.round(rmse,2)}')
            
            else:            
                # Forecasting for specified intervals
                forecasted_data = pd.DataFrame((pd.date_range(dates[-1] + dates.freq, periods=n_intervals, freq=dates.freq)), columns=['ds'])
            
                daily_data = daily_data.reset_index()
                daily_data.columns = ['ds', 'y']
                
                model = ProphetModule()
                model.fit(daily_data)
                
                y_pred = model.predict(forecasted_data)
                
                daily_data=daily_data.set_index('ds')
                daily_data=daily_data[start_date:]
                daily_data = daily_data.reset_index()
                daily_data.columns = ['ds', 'y']
        
                fig = go.Figure()
        
                fig.add_trace(go.Scatter(x=daily_data['ds'], y=daily_data['y'], mode='lines', line=dict(color='blue'), name='Historical Data'))
                fig.add_trace(go.Scatter(x=y_pred['ds'], y=y_pred['yhat'], mode='lines', line=dict(color='red'), name='Forecast'))
                fig.add_trace(go.Scatter(x=y_pred['ds'], y=y_pred['trend'], mode='lines', line=dict(color='green'), name='Trend Line'))
                
                fig.add_trace(go.Scatter(
                    x=y_pred['ds'],
                    y=y_pred['yhat_upper'],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    name='Upper Bound',
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=y_pred['ds'],
                    y=y_pred['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Bands',
                    hoverinfo='skip'
                ))
                
                fig.update_layout(title=f'Prediction of Search Frequency for people belonging to the {col} - {var}', xaxis=dict(title='Date'), yaxis=dict(title=f'Search Frequency Forecast for {var}'), width=1100)
                st.plotly_chart(fig)
        
        choice = st.radio("Choose an option:", ('Test', 'Forecast'))
        
        if choice == 'Test':
            st.write("Pick the 'Column' you want to test and the 'Value' within that column you're curious about. Then choose your 'Test Size' - how much of the data you'd like to test")
            
            left1, right1 = st.columns(2)
            col = left1.selectbox("Choose the column across which you would like to filter", ["driver_race", "driver_age_group", "driver_gender"])
            var = right1.selectbox("Choose the category: ", df[col].unique())
            left, right = st.columns(2, gap="large")
            start_year = left.slider("Enter the year from which you'd like to see", min_value=2005, max_value=2016, value=2009)
            test_percentage = right.slider("Select test set size (%)", min_value=1, max_value=50, value=15)
            plot(df, forecast=False, col=col, var=var, start_year=start_year, test_percentage=test_percentage)
        else:
            st.write("Pick the 'Column' you want to forecast and the 'Value' within that column you're curious about. Then choose the frequency and the number of periods you would like to forecast i.e., how much into the future you'd like to forecast")
            
            left1, right1 = st.columns(2)
            col = left1.selectbox("Choose the column across which you would like to filter", ["driver_race", "driver_age_group", "driver_gender"])
            var = right1.selectbox("Choose the category: ", df[col].unique())
            
            freq_level = st.selectbox("Choose the frequency at which you would like to forecast", ["Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly", "Yearly"], index=2)
            
            left, right = st.columns(2, gap="large")
            n_intervals = left.slider("Select number of intervals to forecast", min_value=1, max_value=1000, value=24)
            start_year = right.slider("Enter the year from which you'd like to see", min_value=2005, max_value=2016, value=2009)
            plot(df, forecast=True, col=col, var=var, freq_level=freq_level, n_intervals=n_intervals, start_year=start_year)
            
    elif selected_tsf == "Arrest Frequency Forecast":

        def plot(data, freq_level='Weekly', n_intervals=100, col='driver_race', var='White', forecast=True, test_percentage=None, start_year=2012):
            
            data = data[data[col] == var]
            
            if freq_level == 'Daily':
                frequency = 'D'
            elif freq_level == 'Weekly':
                frequency = 'W'
            elif freq_level == 'Monthly':
                frequency = 'M'
            elif freq_level == 'Quarterly':
                frequency = 'Q'
            elif freq_level == 'Half-yearly':
                frequency = '6M'
            elif freq_level == 'Yearly':
                frequency = 'Y'
            else:
                st.error("Please choose an appropriate value at which you want to forecast the values")
                return
            
            data['date'] = pd.to_datetime(data['stop_date'], format='%Y%'+frequency)
            daily_data = data.groupby([pd.Grouper(key='date', freq=frequency), col])['is_arrested'].sum().reset_index()
            daily_data = daily_data.set_index('date')['is_arrested']
            
            daily_data.index = pd.to_datetime(daily_data.index)
            daily_data = daily_data.asfreq(frequency)
            daily_data = daily_data.fillna(0)
            dates = daily_data.index
            
            start_date = str(start_year)
            
            #daily_data = daily_data[start_date:]  # Filtering data from the selected start year
            
            if not forecast:
                test_size = int(len(daily_data) * (test_percentage / 100))
                
                # Testing the model for specified test set size
                train_data = daily_data[:-test_size]
                test_data = daily_data[-test_size:]
                
                train_data = train_data.reset_index()
                train_data.columns = ['ds', 'y']
                
                model = ProphetModule()
                model.fit(train_data)
                
                future = model.make_future_dataframe(periods=test_size, freq=frequency)
                forecast = model.predict(future)
                
                forecast_test = forecast[-test_size:]
                
                train_data=train_data.set_index('ds')
                train_data=train_data[start_date:]
                train_data = train_data.reset_index()
                train_data.columns = ['ds', 'y']
                
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_data['ds'], y=train_data['y'], mode='lines', line=dict(color='blue'), name='Actual values of Train'))
                fig.add_trace(go.Scatter(x=test_data.index, y=test_data.values, mode='lines', line=dict(color='green'), name='Actual values of Test'))
                fig.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'], mode='lines', line=dict(color='red'), name='Forecasted values of Test'))
                
                fig.update_layout(title='Estimated Arrests', xaxis=dict(title='Date'), yaxis=dict(title=f'Arrest Frequency for {var}'), width=1100)
                st.plotly_chart(fig)
                
                rmse=np.sqrt(mean_squared_error(test_data.values,forecast_test['yhat']))
                    
                st.write(f'#### The RMSE is {np.round(rmse,2)}')
            
            else:            
                # Forecasting for specified intervals
                forecasted_data = pd.DataFrame((pd.date_range(dates[-1] + dates.freq, periods=n_intervals, freq=dates.freq)), columns=['ds'])
            
                daily_data = daily_data.reset_index()
                daily_data.columns = ['ds', 'y']
                
                model = ProphetModule()
                model.fit(daily_data)
                
                y_pred = model.predict(forecasted_data)
                
                daily_data=daily_data.set_index('ds')
                daily_data=daily_data[start_date:]
                daily_data = daily_data.reset_index()
                daily_data.columns = ['ds', 'y']
        
                fig = go.Figure()
        
                fig.add_trace(go.Scatter(x=daily_data['ds'], y=daily_data['y'], mode='lines', line=dict(color='blue'), name='Historical Data'))
                fig.add_trace(go.Scatter(x=y_pred['ds'], y=y_pred['yhat'], mode='lines', line=dict(color='red'), name='Forecast'))
                fig.add_trace(go.Scatter(x=y_pred['ds'], y=y_pred['trend'], mode='lines', line=dict(color='green'), name='Trend Line'))
                
                fig.add_trace(go.Scatter(
                    x=y_pred['ds'],
                    y=y_pred['yhat_upper'],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    name='Upper Bound',
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=y_pred['ds'],
                    y=y_pred['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Bands',
                    hoverinfo='skip'
                ))
                
                fig.update_layout(title=f'Prediction of Arrests Frequency for people belonging to the {col} - {var}', xaxis=dict(title='Date'), yaxis=dict(title=f'Arrest Frequency Forecast for {var}'), width=1100)
                st.plotly_chart(fig)
        
        choice = st.radio("Choose an option:", ('Test', 'Forecast'))
        
        if choice == 'Test':
            st.write("Pick the 'Column' you want to test and the 'Value' within that column you're curious about. Then choose your 'Test Size' - how much of the data you'd like to test")
            
            left1, right1 = st.columns(2)
            col = left1.selectbox("Choose the column across which you would like to filter", ["driver_race", "driver_age_group", "driver_gender"])
            var = right1.selectbox("Choose the category: ", df[col].unique())
            left, right = st.columns(2, gap="large")
            start_year = left.slider("Enter the year from which you'd like to see", min_value=2005, max_value=2016, value=2009)
            test_percentage = right.slider("Select test set size (%)", min_value=1, max_value=50, value=15)
            plot(df, forecast=False, col=col, var=var, start_year=start_year, test_percentage=test_percentage)
        else:
            st.write("Pick the 'Column' you want to forecast and the 'Value' within that column you're curious about. Then choose the frequency and the number of periods you would like to forecast i.e., how much into the future you'd like to forecast")
            
            left1, right1 = st.columns(2)
            col = left1.selectbox("Choose the column across which you would like to filter", ["driver_race", "driver_age_group", "driver_gender"])
            var = right1.selectbox("Choose the category: ", df[col].unique())
            
            freq_level = st.selectbox("Choose the frequency at which you would like to forecast", ["Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly", "Yearly"], index=2)
            
            left, right = st.columns(2, gap="large")
            n_intervals = left.slider("Select number of intervals to forecast", min_value=1, max_value=1000, value=24)
            start_year = right.slider("Enter the year from which you'd like to see", min_value=2005, max_value=2016, value=2009)
            plot(df, forecast=True, col=col, var=var, freq_level=freq_level, n_intervals=n_intervals, start_year=start_year)
            
            
    elif selected_tsf == "Drugs Related Stop Forecast":

        def plot(data, freq_level='Weekly', n_intervals=100, col='driver_race', var='White', forecast=True, test_percentage=None, start_year=2012):
            
            data = data[data[col] == var]
            
            if freq_level == 'Daily':
                frequency = 'D'
            elif freq_level == 'Weekly':
                frequency = 'W'
            elif freq_level == 'Monthly':
                frequency = 'M'
            elif freq_level == 'Quarterly':
                frequency = 'Q'
            elif freq_level == 'Half-yearly':
                frequency = '6M'
            elif freq_level == 'Yearly':
                frequency = 'Y'
            else:
                st.error("Please choose an appropriate value at which you want to forecast the values")
                return
            
            data['date'] = pd.to_datetime(data['stop_date'], format='%Y%'+frequency)
            daily_data = data.groupby([pd.Grouper(key='date', freq=frequency), col])['drugs_related_stop'].sum().reset_index()
            daily_data = daily_data.set_index('date')['drugs_related_stop']
            
            daily_data.index = pd.to_datetime(daily_data.index)
            daily_data = daily_data.asfreq(frequency)
            daily_data = daily_data.fillna(0)
            dates = daily_data.index
            
            start_date = str(start_year)
            
            #daily_data = daily_data[start_date:]  # Filtering data from the selected start year
            
            if not forecast:
                test_size = int(len(daily_data) * (test_percentage / 100))
                
                # Testing the model for specified test set size
                train_data = daily_data[:-test_size]
                test_data = daily_data[-test_size:]
                
                train_data = train_data.reset_index()
                train_data.columns = ['ds', 'y']
                
                model = ProphetModule()
                model.fit(train_data)
                
                future = model.make_future_dataframe(periods=test_size, freq=frequency)
                forecast = model.predict(future)
                
                forecast_test = forecast[-test_size:]
                
                train_data=train_data.set_index('ds')
                train_data=train_data[start_date:]
                train_data = train_data.reset_index()
                train_data.columns = ['ds', 'y']
                
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_data['ds'], y=train_data['y'], mode='lines', line=dict(color='blue'), name='Actual values of Train'))
                fig.add_trace(go.Scatter(x=test_data.index, y=test_data.values, mode='lines', line=dict(color='green'), name='Actual values of Test'))
                fig.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'], mode='lines', line=dict(color='red'), name='Forecasted values of Test'))
                
                fig.update_layout(title='Estimated Drug Related Stops', xaxis=dict(title='Date'), yaxis=dict(title=f'Drugs Related Stop Frequency for {var}'), width=1100)
                st.plotly_chart(fig)
                
                rmse=np.sqrt(mean_squared_error(test_data.values,forecast_test['yhat']))
                    
                st.write(f'#### The RMSE is {np.round(rmse,2)}')
            
            else:            
                # Forecasting for specified intervals
                forecasted_data = pd.DataFrame((pd.date_range(dates[-1] + dates.freq, periods=n_intervals, freq=dates.freq)), columns=['ds'])
            
                daily_data = daily_data.reset_index()
                daily_data.columns = ['ds', 'y']
                
                model = ProphetModule()
                model.fit(daily_data)
                
                y_pred = model.predict(forecasted_data)
                
                daily_data=daily_data.set_index('ds')
                daily_data=daily_data[start_date:]
                daily_data = daily_data.reset_index()
                daily_data.columns = ['ds', 'y']
        
                fig = go.Figure()
        
                fig.add_trace(go.Scatter(x=daily_data['ds'], y=daily_data['y'], mode='lines', line=dict(color='blue'), name='Histroical Data'))
                fig.add_trace(go.Scatter(x=y_pred['ds'], y=y_pred['yhat'], mode='lines', line=dict(color='red'), name='Forecast'))
                fig.add_trace(go.Scatter(x=y_pred['ds'], y=y_pred['trend'], mode='lines', line=dict(color='green'), name='Trend Line'))
                
                fig.add_trace(go.Scatter(
                    x=y_pred['ds'],
                    y=y_pred['yhat_upper'],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    name='Upper Bound',
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=y_pred['ds'],
                    y=y_pred['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Bands',
                    hoverinfo='skip'
                ))
                
                fig.update_layout(title=f'Prediction of Drug Related Stops for people belonging to the {col} - {var}', xaxis=dict(title='Date'), yaxis=dict(title=f'Drug Related Stop Frequency Forecast for {var}'), width=1100)
                st.plotly_chart(fig)
        
        choice = st.radio("Choose an option:", ('Test', 'Forecast'))
        
        if choice == 'Test':
            st.write("Pick the 'Column' you want to test and the 'Value' within that column you're curious about. Then choose your 'Test Size' - how much of the data you'd like to test")
            
            left1, right1 = st.columns(2)
            col = left1.selectbox("Choose the column across which you would like to filter", ["driver_race", "driver_age_group", "driver_gender"])
            var = right1.selectbox("Choose the category: ", df[col].unique())
            left, right = st.columns(2, gap="large")
            start_year = left.slider("Enter the year from which you'd like to see", min_value=2005, max_value=2016, value=2009)
            test_percentage = right.slider("Select test set size (%)", min_value=1, max_value=50, value=15)
            plot(df, forecast=False, col=col, var=var, start_year=start_year, test_percentage=test_percentage)
        else:
            st.write("Pick the 'Column' you want to forecast and the 'Value' within that column you're curious about. Then choose the frequency and the number of periods you would like to forecast i.e., how much into the future you'd like to forecast")
            
            left1, right1 = st.columns(2)
            col = left1.selectbox("Choose the column across which you would like to filter", ["driver_race", "driver_age_group", "driver_gender"])
            var = right1.selectbox("Choose the category: ", df[col].unique())
            
            freq_level = st.selectbox("Choose the frequency at which you would like to forecast", ["Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly", "Yearly"], index=2)
            
            left, right = st.columns(2, gap="large")
            n_intervals = left.slider("Select number of intervals to forecast", min_value=1, max_value=1000, value=24)
            start_year = right.slider("Enter the year from which you'd like to see", min_value=2005, max_value=2016, value=2009)
            plot(df, forecast=True, col=col, var=var, freq_level=freq_level, n_intervals=n_intervals, start_year=start_year)
        
elif selected == "Traffic Violation Classification":
    selected_cm = option_menu(None, ["Random Forest", "Gradient Boosting", "Logistic Regression", "Naive Bayes", "Decision Tree", "KNN"], 
    default_index=0, orientation="horizontal")
    
    if selected_cm == "Random Forest":
        rf_description = st.checkbox('**What is Random Forest Algorithm?**')
        
        if rf_description:
            st.markdown("* Imagine you're making a big decision and you ask a bunch of friends for advice. Each friend gives their thoughts, and then you go with the most popular opinion.")
            st.markdown("* A random forest classifier is kind of like that—except your friends are different decision-making trees, and together, they help make really accurate predictions by combining their opinions.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) link for a detailed explanation of random forest classifiers!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'], ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'])
        classfication(RandomForestClassifier, 0.2, feat, 'Violation', violation_map, random_state)
        
    elif selected_cm == "Gradient Boosting":
        gb_description = st.checkbox('**What does Gradient Boosting algorithm do?**')
        
        if gb_description:
            st.markdown("* Gradient boosting is like teamwork among different experts. Each new expert focuses on the mistakes the previous ones made, gradually improving the overall prediction. It's powerful and often used in things like predicting sales.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) link for a detailed explanation of gradient boosting classifiers!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'], ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'])
        classfication(GradientBoostingClassifier, 0.2, feat, 'Violation', violation_map, random_state)
        
    elif selected_cm == "Logistic Regression":
        lr_description = st.checkbox('**What is Logistic Regression Algorithm?**')
        
        if lr_description:
            st.markdown("* Imagine you're trying to predict if it'll rain tomorrow based on today's weather. Logistic regression helps with precisely that—it figures out the probability of a yes-or-no outcome, like 'will it rain?' based on various factors, like temperature or humidity.")
            st.markdown("* It's a handy tool for making predictions when you're dealing with two possible outcomes.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) link for a detailed explanation on logistic regression!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'], ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'])
        classfication(LogisticRegression, 0.2, feat, 'Violation', violation_map, random_state)
    
    elif selected_cm == "Naive Bayes":
        nb_description = st.checkbox('**What does Naive Bayes algorithm do?**')
        
        if nb_description:
            st.markdown("* Naive Bayes is like a clever detective—it uses probabilities to make predictions. It assumes that different features in your data are independent of each other, even if they might not be in reality.")
            st.markdown("* Despite its simplification, it's surprisingly good at making predictions, especially in text classification or spam filtering.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/naive_bayes.html) link for a detailed explanation on naive bayes!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'], ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'])
        classfication(GaussianNB, 0.2, feat, 'Violation', violation_map, random_state)
        
    elif selected_cm == "Decision Tree":
        dt_description = st.checkbox('**What is a Decision Tree?**')
        
        if dt_description:
            st.markdown("* Decision trees are like flowcharts that help you make decisions by asking yes-or-no questions, leading to a final choice or prediction.")
            st.markdown("* They break down problems into smaller parts, making them easy to understand and use for making decisions or predictions in various fields, including machine learning.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/tree.html) link for a detailed explanation on decision trees!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'], ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'])
        classfication(DecisionTreeClassifier, 0.2, feat, 'Violation', violation_map, random_state)
        
    elif selected_cm == "KNN":
        knn_description = st.checkbox('**What does KNN do?**')
        
        if knn_description:
            st.markdown("* K-Nearest Neighbors, is like finding your nearest neighbors in a neighborhood. It predicts based on the majority of its closest 'neighbors' in the data. Simple and effective for classification or regression tasks in machine learning.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) link for a detailed explanation on KNN!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'], ['stop_duration', 'driver_gender', 'driver_age_group', 'driver_race', 'search_conducted', 'stop_outcome', 'drugs_related_stop'])
        classfication(KNeighborsClassifier, 0.2, feat, 'Violation', violation_map, random_state)
        
elif selected == "Search Occurrence Classification":
    selected_cm = option_menu(None, ["Random Forest", "Gradient Boosting", "Logistic Regression", "Naive Bayes", "Decision Tree", "KNN"], 
    default_index=0, orientation="horizontal")
    
    if selected_cm == "Random Forest":
        rf_description = st.checkbox('**What is Random Forest Algorithm?**')
        
        if rf_description:
            st.markdown("* Imagine you're making a big decision and you ask a bunch of friends for advice. Each friend gives their thoughts, and then you go with the most popular opinion.")
            st.markdown("* A random forest classifier is kind of like that—except your friends are different decision-making trees, and together, they help make really accurate predictions by combining their opinions.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) link for a detailed explanation of random forest classifiers!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'])
        classfication(RandomForestClassifier, 0.2, feat, 'search_conducted', search_conducted_map, random_state)
        
    elif selected_cm == "Gradient Boosting":
        gb_description = st.checkbox('**What does Gradient Boosting algorithm do?**')
        
        if gb_description:
            st.markdown("* Gradient boosting is like teamwork among different experts. Each new expert focuses on the mistakes the previous ones made, gradually improving the overall prediction. It's powerful and often used in things like predicting sales.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) link for a detailed explanation of gradient boosting classifiers!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'])
        classfication(GradientBoostingClassifier, 0.2, feat, 'search_conducted', search_conducted_map, random_state)
        
    elif selected_cm == "Logistic Regression":
        lr_description = st.checkbox('**What is Logistic Regression Algorithm?**')
        
        if lr_description:
            st.markdown("* Imagine you're trying to predict if it'll rain tomorrow based on today's weather. Logistic regression helps with precisely that—it figures out the probability of a yes-or-no outcome, like 'will it rain?' based on various factors, like temperature or humidity.")
            st.markdown("* It's a handy tool for making predictions when you're dealing with two possible outcomes.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) link for a detailed explanation on logistic regression!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'])
        classfication(LogisticRegression, 0.2, feat, 'search_conducted', search_conducted_map, random_state)
    
    elif selected_cm == "Naive Bayes":
        nb_description = st.checkbox('**What does Naive Bayes algorithm do?**')
        
        if nb_description:
            st.markdown("* Naive Bayes is like a clever detective—it uses probabilities to make predictions. It assumes that different features in your data are independent of each other, even if they might not be in reality.")
            st.markdown("* Despite its simplification, it's surprisingly good at making predictions, especially in text classification or spam filtering.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/naive_bayes.html) link for a detailed explanation on naive bayes!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'])
        classfication(GaussianNB, 0.2, feat, 'search_conducted', search_conducted_map, random_state)
        
    elif selected_cm == "Decision Tree":
        dt_description = st.checkbox('**What is a Decision Tree?**')
        
        if dt_description:
            st.markdown("* Decision trees are like flowcharts that help you make decisions by asking yes-or-no questions, leading to a final choice or prediction.")
            st.markdown("* They break down problems into smaller parts, making them easy to understand and use for making decisions or predictions in various fields, including machine learning.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/tree.html) link for a detailed explanation on decision trees!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'])
        classfication(DecisionTreeClassifier, 0.2, feat, 'search_conducted', search_conducted_map, random_state)
        
    elif selected_cm == "KNN":
        knn_description = st.checkbox('**What does KNN do?**')
        
        if knn_description:
            st.markdown("* K-Nearest Neighbors, is like finding your nearest neighbors in a neighborhood. It predicts based on the majority of its closest 'neighbors' in the data. Simple and effective for classification or regression tasks in machine learning.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) link for a detailed explanation on KNN!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'stop_outcome', 'drugs_related_stop'])
        classfication(KNeighborsClassifier, 0.2, feat, 'search_conducted', search_conducted_map, random_state)
        
elif selected == "Stop Outcome Classification":
    selected_cm = option_menu(None, ["Random Forest", "Gradient Boosting", "Logistic Regression", "Naive Bayes", "Decision Tree", "KNN"], 
    default_index=0, orientation="horizontal")
    
    if selected_cm == "Random Forest":
        rf_description = st.checkbox('**What is Random Forest Algorithm?**')
        
        if rf_description:
            st.markdown("* Imagine you're making a big decision and you ask a bunch of friends for advice. Each friend gives their thoughts, and then you go with the most popular opinion.")
            st.markdown("* A random forest classifier is kind of like that—except your friends are different decision-making trees, and together, they help make really accurate predictions by combining their opinions.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) link for a detailed explanation of random forest classifiers!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'])
        classfication(RandomForestClassifier, 0.2, feat, 'stop_outcome', stop_outcome_map, random_state)
        
    elif selected_cm == "Gradient Boosting":
        gb_description = st.checkbox('**What does Gradient Boosting algorithm do?**')
        
        if gb_description:
            st.markdown("* Gradient boosting is like teamwork among different experts. Each new expert focuses on the mistakes the previous ones made, gradually improving the overall prediction. It's powerful and often used in things like predicting sales.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) link for a detailed explanation of gradient boosting classifiers!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'])
        classfication(GradientBoostingClassifier, 0.2, feat, 'stop_outcome', stop_outcome_map, random_state)
        
    elif selected_cm == "Logistic Regression":
        lr_description = st.checkbox('**What is Logistic Regression Algorithm?**')
        
        if lr_description:
            st.markdown("* Imagine you're trying to predict if it'll rain tomorrow based on today's weather. Logistic regression helps with precisely that—it figures out the probability of a yes-or-no outcome, like 'will it rain?' based on various factors, like temperature or humidity.")
            st.markdown("* It's a handy tool for making predictions when you're dealing with two possible outcomes.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) link for a detailed explanation on logistic regression!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'])
        classfication(LogisticRegression, 0.2, feat, 'stop_outcome', stop_outcome_map, random_state)
    
    elif selected_cm == "Naive Bayes":
        nb_description = st.checkbox('**What does Naive Bayes algorithm do?**')
        
        if nb_description:
            st.markdown("* Naive Bayes is like a clever detective—it uses probabilities to make predictions. It assumes that different features in your data are independent of each other, even if they might not be in reality.")
            st.markdown("* Despite its simplification, it's surprisingly good at making predictions, especially in text classification or spam filtering.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/naive_bayes.html) link for a detailed explanation on naive bayes!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'])
        classfication(GaussianNB, 0.2, feat, 'stop_outcome', stop_outcome_map, random_state)
        
    elif selected_cm == "Decision Tree":
        dt_description = st.checkbox('**What is a Decision Tree?**')
        
        if dt_description:
            st.markdown("* Decision trees are like flowcharts that help you make decisions by asking yes-or-no questions, leading to a final choice or prediction.")
            st.markdown("* They break down problems into smaller parts, making them easy to understand and use for making decisions or predictions in various fields, including machine learning.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/tree.html) link for a detailed explanation on decision trees!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'])
        classfication(DecisionTreeClassifier, 0.2, feat, 'stop_outcome', stop_outcome_map, random_state)
        
    elif selected_cm == "KNN":
        knn_description = st.checkbox('**What does KNN do?**')
        
        if knn_description:
            st.markdown("* K-Nearest Neighbors, is like finding your nearest neighbors in a neighborhood. It predicts based on the majority of its closest 'neighbors' in the data. Simple and effective for classification or regression tasks in machine learning.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) link for a detailed explanation on KNN!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'drugs_related_stop'])
        classfication(KNeighborsClassifier, 0.2, feat, 'stop_outcome', stop_outcome_map, random_state)
        
elif selected == "Drug Related Stops Classification":
    selected_cm = option_menu(None, ["Resampling", "Random Forest", "Gradient Boosting", "Logistic Regression", "Naive Bayes", "Decision Tree", "KNN"], 
    default_index=0, orientation="horizontal")
    
    if selected_cm == "Resampling":
        st.image("sm.png", width=500)
        st.write("You might have noticed something interesting in this dataset — there's quite a difference in the number of incidents or outcomes across different categories. This kind of imbalance can sometimes throw off our predictions.")
        st.write("Imagine this: it's like trying to teach a classroom where most of the students are experts in one subject and only a few in another. It's a bit tricky to ensure everyone learns equally, right?")
        
        st.write("#### How to handle this?")
        
        st.write("To make sure our predictions are fair and accurate across all types of incidents, I used a neat technique called resampling. It's like finding a way to balance the number of examples we have for each type of incident, making sure the model doesn’t just get really good at predicting the most common ones.")
        st.write("**SMOTE-NC** - It is special recipe for making the dataset more balanced. It cooks up new examples of the less frequent incidents by mixing and matching the characteristics of the existing ones. Imagine creating new, yet very similar, scenarios to teach the class about the less common subjects.")
    
        st.write("#### Why SMOTE-NC?")
        st.write("SMOTE-NC addresses imbalance in datasets with both numerical and categorical features, unlike many other methods focused only on numerical data.")
        st.write("#### How did this help with the predictions?")
        st.write("Think of it like this: by giving the model a more balanced view of different incidents, it’s like providing equal learning opportunities to all students in our classroom. This helps our model become fairer and more accurate in predicting all types of incidents, not just the most common ones.")
        
        url_sr = "https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html"
        st.write("[Credits: The above image was taken from this website](%s)" % url_sr)
    
    elif selected_cm == "Random Forest":
        rf_description = st.checkbox('**What is Random Forest Algorithm?**')
        
        if rf_description:
            st.markdown("* Imagine you're making a big decision and you ask a bunch of friends for advice. Each friend gives their thoughts, and then you go with the most popular opinion.")
            st.markdown("* A random forest classifier is kind of like that—except your friends are different decision-making trees, and together, they help make really accurate predictions by combining their opinions.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) link for a detailed explanation of random forest classifiers!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'])
        classfication(RandomForestClassifier, 0.2, feat, 'drug_related', drugs_related_stop_map, random_state)
        
    elif selected_cm == "Gradient Boosting":
        gb_description = st.checkbox('**What does Gradient Boosting algorithm do?**')
        
        if gb_description:
            st.markdown("* Gradient boosting is like teamwork among different experts. Each new expert focuses on the mistakes the previous ones made, gradually improving the overall prediction. It's powerful and often used in things like predicting sales.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) link for a detailed explanation of gradient boosting classifiers!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'])
        classfication(GradientBoostingClassifier, 0.2, feat, 'drug_related', drugs_related_stop_map, random_state)
        
    elif selected_cm == "Logistic Regression":
        lr_description = st.checkbox('**What is Logistic Regression Algorithm?**')
        
        if lr_description:
            st.markdown("* Imagine you're trying to predict if it'll rain tomorrow based on today's weather. Logistic regression helps with precisely that—it figures out the probability of a yes-or-no outcome, like 'will it rain?' based on various factors, like temperature or humidity.")
            st.markdown("* It's a handy tool for making predictions when you're dealing with two possible outcomes.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) link for a detailed explanation on logistic regression!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'])
        classfication(LogisticRegression, 0.2, feat, 'drug_related', drugs_related_stop_map, random_state)
    
    elif selected_cm == "Naive Bayes":
        nb_description = st.checkbox('**What does Naive Bayes algorithm do?**')
        
        if nb_description:
            st.markdown("* Naive Bayes is like a clever detective—it uses probabilities to make predictions. It assumes that different features in your data are independent of each other, even if they might not be in reality.")
            st.markdown("* Despite its simplification, it's surprisingly good at making predictions, especially in text classification or spam filtering.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/naive_bayes.html) link for a detailed explanation on naive bayes!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'])
        classfication(GaussianNB, 0.2, feat, 'drug_related', drugs_related_stop_map, random_state)
        
    elif selected_cm == "Decision Tree":
        dt_description = st.checkbox('**What is a Decision Tree?**')
        
        if dt_description:
            st.markdown("* Decision trees are like flowcharts that help you make decisions by asking yes-or-no questions, leading to a final choice or prediction.")
            st.markdown("* They break down problems into smaller parts, making them easy to understand and use for making decisions or predictions in various fields, including machine learning.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/tree.html) link for a detailed explanation on decision trees!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'])
        classfication(DecisionTreeClassifier, 0.2, feat, 'drug_related', drugs_related_stop_map, random_state)
        
    elif selected_cm == "KNN":
        knn_description = st.checkbox('**What does KNN do?**')
        
        if knn_description:
            st.markdown("* K-Nearest Neighbors, is like finding your nearest neighbors in a neighborhood. It predicts based on the majority of its closest 'neighbors' in the data. Simple and effective for classification or regression tasks in machine learning.")
            st.markdown("* **Curious to know more?** Check out [this](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) link for a detailed explanation on KNN!")
            st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
        left, right = st.columns(2)
        test_perc = left.slider("Select the test set size", min_value=0.1, max_value=0.9, value=0.2)
        random_state = right.slider("Select the random state for reproducibility", min_value=0, max_value=100, value=5)
        feat = st.multiselect("Select the features for the model", ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'], ['driver_gender', 'driver_age_group', 'driver_race', 'violation', 'search_conducted', 'stop_outcome'])
        classfication(KNeighborsClassifier, 0.2, feat, 'drug_related', drugs_related_stop_map, random_state)

if selected == "Customized Predictions":
    
    def custom_model_feature(data, target, model, features):
        m=model()
        x_label=features.columns
        x=data[x_label]
        y=data['y_resampled']
        
        x=pd.concat([x,features],ignore_index=True)
        y=y*1
        y=y.astype(int)
        x_test=x.tail(1)
        x=x.iloc[:-1,:]
        m.fit(x,y)
        pred=m.predict(x_test)
        if target == 'search_conducted':
            if pred == 0:
                st.write("**For the selections you have made, the probability of being searched by a police officer is less**")
            elif pred == 1:
                st.write("**For the selections you have made, the probability of being searched by a police officer is high**")
        elif target == 'is_arrested':
            if pred == 0:
                st.write("**For the selections you have made, the probability of being arrested is less**")
            elif pred == 1:
                st.write("**For the selections you have made, the probability of being arrested is high**")
        elif target == 'drugs_related_stop':
            if pred == 0:
                st.write("**For the selections you have made, the probability of the stop being related to drugs is less**")
            elif pred == 1:
                st.write("**For the selections you have made, the probability of the stop being related to drugs is high**")
        #st.write(pred)
        return pred
    
    st.write("**Welcome to the Prediction Zone! Here, you're in complete control. Choose what you want to predict, select the model, input your data features, and let the magic happen. Make informed predictions at your fingertips! Let's dive in.**")
    l1, r1 = st.columns(2, gap="large")
    target = l1.selectbox("First things first, pick a variable you want to predict", ['search_conducted', 'is_arrested', 'drugs_related_stop'], index=1)
    mod = r1.selectbox("Select the model which you want to use:", ['Logistic Regression','Decision Tree','Random Forest','KNN Classifier','Gradient Boosting', 'Naive Bayes'], index=0)
    model_map={'Logistic Regression': LogisticRegression,'Decision Tree': DecisionTreeClassifier, 'Random Forest':RandomForestClassifier,
               'KNN Classifier':KNeighborsClassifier, 'Gradient Boosting':GradientBoostingClassifier, 'Naive Bayes': GaussianNB}
    model=model_map[mod]
    st.write("**Time to add the info you've got!**")
    l2, r2 = st.columns(2, gap="large")
    gender = l2.selectbox("Gender", list(cl_df.driver_gender.unique()))
    age_group = r2.selectbox("Age group", list(cl_df.driver_age_group.unique()))
    l3, r3 = st.columns(2, gap="large")
    race = l3.selectbox("Race", list(cl_df.driver_race.unique()))
    violation = r3.selectbox("Violation Type", list(cl_df.violation.unique()))
    if target == "search_conducted":
        l4, r4 = st.columns(2, gap="large")
        var1 = l4.selectbox("Are they arrested?", list(cl_df.is_arrested.unique()))
        var2 = r4.selectbox("Was it a drug related stop?", list(cl_df.drugs_related_stop.unique()))
    elif target == "is_arrested":
        l5, r5 = st.columns(2, gap="large")
        var1 = l5.selectbox("Search conducted?", list(cl_df.search_conducted.unique()))
        var2 = r5.selectbox("Drug related stop?", list(cl_df.drugs_related_stop.unique()))  
    elif target == "drugs_related_stop":
        l6, r6 = st.columns(2, gap="large")
        var1 = l6.selectbox("Was search conducted?", list(cl_df.search_conducted.unique()))
        var2 = r6.selectbox("Are they arrested?", list(cl_df.is_arrested.unique()))
    extra=[var1, var2]
    
    extra_f=['search_conducted', 'is_arrested', 'drugs_related_stop']
    extra_f.remove(target)
    
    d={'driver_gender':[gender],'driver_age_group':[age_group],'driver_race':[race],'violation':[violation]}
    for i in range(len(extra)):
        d[extra_f[i]]=[extra[i]]
    d=pd.DataFrame(d)
    d=d[['driver_gender','driver_age_group','driver_race','violation'] + extra_f]
    
    def swap_keys_and_values(input_dict):
        return {value: key for key, value in input_dict.items()}
    
    g_map=swap_keys_and_values(gender_map)
    a_map=swap_keys_and_values(age_map)
    r_map=swap_keys_and_values(race_map)
    v_map=swap_keys_and_values(violation_map)

    
    if target == 'search_conducted':
        custom_df = pd.read_csv("search_conducted_Resampled.csv")
    elif target == 'drugs_related_stop':
        custom_df = pd.read_csv("drug_related_Resampled.csv")
    elif target == 'is_arrested':
        custom_df = pd.read_csv("is_arrested_Resampled.csv")
        
    d.driver_gender.replace(g_map,inplace=True)
    d.driver_age_group.replace(a_map,inplace=True)
    d.driver_race.replace(r_map,inplace=True)
    d.violation.replace(v_map,inplace=True)
    d[extra_f[0]].replace({True:1,False:0},inplace=True)
    d[extra_f[1]].replace({True:1,False:0},inplace=True)
    
    custom_model_feature(custom_df, target, model, d)