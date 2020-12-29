#IMPORT MODULES
import inline
import matplotlib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from colorama import Fore, Back, Style
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.formula.api import ols
import plotly.graph_objs as gobj
import plotly.figure_factory as ff

#SHOW THE LIST OF DIRECTORY
# import os
# for dirname, _, filenames in os.walk('/Users/theresacalangian/Desktop'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

#READ FILE
heart_data = pd.read_csv('/Users/theresacalangian/Desktop/heart_failure_clinical_records_dataset.csv')
# heart_data.head()
print(heart_data)

#1.IS AGE AND SEX AN INDICATOR FOR DEATH EVENT?

#1.a Age Distribution Plot
hist_data =[heart_data["age"].values]
group_labels = ['age']
fig = ff.create_distplot(hist_data, group_labels)
fig.update_layout(title_text='Age Distribution Plot')
fig.show()

#1.b Sex vs Age Distribution Plot (Male = 1 Female =0)
fig = px.box(heart_data, x='sex', y='age', points="all")
fig.update_layout(title_text="Gender vs Age Distribution Plot (Male = 1 Female =0)")
fig.show()

#1.c Analysis on Survival Based on Sex
male = heart_data[heart_data["sex"]==1]
female = heart_data[heart_data["sex"]==0]

male_survived = male[heart_data["DEATH_EVENT"]==0]
male_not = male[heart_data["DEATH_EVENT"]==1]
female_survived = female[heart_data["DEATH_EVENT"]==0]
female_not = female[heart_data["DEATH_EVENT"]==1]

labels = ['Male - Survived','Male - Not Survived', "Female -  Survived", "Female - Not Survived"]
values = [len(male[heart_data["DEATH_EVENT"]==0]),len(male[heart_data["DEATH_EVENT"]==1]),
         len(female[heart_data["DEATH_EVENT"]==0]),len(female[heart_data["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Analysis on Survival Based on Sex")
fig.show()

#2. SEX FACTOR ANALYSIS

#2.a Age vs Death Event Distribution Plot
survived = heart_data[heart_data["DEATH_EVENT"]==0]["age"]
not_survived = heart_data[heart_data["DEATH_EVENT"]==1]["age"]
hist_data = [survived,not_survived]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(title_text="Age vs Death Event Distribution Plot")
fig.show()

#2.b Analysis on Survival Based on Age and Sex
fig = px.violin(heart_data, y="age", x="sex", color="DEATH_EVENT", box=True, points="all", hover_data=heart_data.columns)
fig.update_layout(title_text="Analysis on Survival Based on Age and Sex")
fig.show()

#2.c Analysis on Survival Based on Age and Smoking
fig = px.violin(heart_data, y="age", x="smoking", color="DEATH_EVENT", box=True, points="all", hover_data=heart_data.columns)
fig.update_layout(title_text="Analysis on Survival Based on Age and Smoking")
fig.show()

#2.d Analysis on Survival Based on Age and Diabetes
fig = px.violin(heart_data, y="age", x="diabetes", color="DEATH_EVENT", box=True, points="all", hover_data=heart_data.columns)
fig.update_layout(title_text="Analysis on Survival Based on Age and Diabetes")
fig.show()

#3 OTHER FACTORS (HISTOGRAM PLOTS)

#3.a Creatinine Phosphokinase vs Death Event Distribution Plot
fig = px.histogram(heart_data, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns)
fig.update_layout(title_text="Creatinine Phosphokinase vs Death Event Distribution Plot")
fig.show()

#3.b Ejection Fraction vs Death Event Distribution Plot
fig = px.histogram(heart_data, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns)
fig.update_layout(title_text="Ejection Fraction vs Death Event Distribution Plot")
fig.show()

#3.c Platelets vs Death Event Distribution Plot
fig = px.histogram(heart_data, x="platelets", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns)
fig.update_layout(title_text="Platelets vs Death Event Distribution Plot")
fig.show()

#3.d Serum Creatinine vs Death Event Distribution Plot
fig = px.histogram(heart_data, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns)
fig.update_layout(title_text="Serum Creatinine vs Death Event Distribution Plot")
fig.show()

#3.e Serum Sodium vs Death Event Distribution Plot
fig = px.histogram(heart_data, x="serum_sodium", color="DEATH_EVENT", marginal="violin",hover_data=heart_data.columns)
fig.update_layout(title_text="Serum Sodium vs Death Event Distribution Plot")
fig.show()

#3.f Analysis on Survival Based on Creatinine Phosphokinase
survived = heart_data[heart_data['DEATH_EVENT']==0]['creatinine_phosphokinase']
not_survived = heart_data[heart_data['DEATH_EVENT']==1]['creatinine_phosphokinase']
hist_data = [survived,not_survived]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(title_text="Analysis on Survival Based on Creatinine Phosphokinase")
fig.show()

#3.g Analysis on Survival Based on Ejection Fraction
survived = heart_data[heart_data['DEATH_EVENT']==0]['ejection_fraction']
not_survived = heart_data[heart_data['DEATH_EVENT']==1]['ejection_fraction']
hist_data = [survived,not_survived]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(title_text="Analysis on Survival Based on Ejection Fraction")
fig.show()

#3.h Analysis on Survival Based on Platelets
survived = heart_data[heart_data['DEATH_EVENT']==0]['platelets']
not_survived = heart_data[heart_data['DEATH_EVENT']==1]['platelets']
hist_data = [survived,not_survived]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(title_text="Analysis on Survival Based on Platelets")
fig.show()

#3.i Analysis on Survival Based on Serum Creatinine
survived = heart_data[heart_data['DEATH_EVENT']==0]['serum_creatinine']
not_survived = heart_data[heart_data['DEATH_EVENT']==1]['serum_creatinine']
hist_data = [survived,not_survived]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(title_text="Analysis on Survival Based on Serum Creatinine")
fig.show()

#3.j Analysis on Survival Based on Serum Sodium
survived = heart_data[heart_data['DEATH_EVENT']==0]['serum_sodium']
not_survived = heart_data[heart_data['DEATH_EVENT']==1]['serum_sodium']
hist_data = [survived,not_survived]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(title_text="Analysis on Survival Based on Serum Sodium")
fig.show()

#4 OTHER FACTORS (PIE CHARTS)

#4.a Diabetes Distribution Pie Chart
labels = ['No Diabetes','Diabetes']
diabetes_yes = heart_data[heart_data['diabetes']==1]
diabetes_no = heart_data[heart_data['diabetes']==0]
values = [len(diabetes_no), len(diabetes_yes)]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Diabetes Distribution Pie Chart")
fig.show()

#4.b Analysis on Diabetes vs Death Event Ratio
fig = px.pie(heart_data, values='diabetes',names='DEATH_EVENT', title='Diabetes vs Death Event Pie Chart')
fig.show()

#4.c Analysis on Survival Based on Diabetes
diabetes_yes_survived = diabetes_yes[heart_data["DEATH_EVENT"]==0]
diabetes_yes_not_survived = diabetes_yes[heart_data["DEATH_EVENT"]==1]
diabetes_no_survived = diabetes_no[heart_data["DEATH_EVENT"]==0]
diabetes__no_not_survived = diabetes_no[heart_data["DEATH_EVENT"]==1]

labels = ['Diabetes Yes - Survived','Diabetes Yes - Not Survived', 'Diabetes NO - Survived', 'Diabetes NO - Not Survived']
values = [len(diabetes_yes[heart_data["DEATH_EVENT"]==0]),len(diabetes_yes[heart_data["DEATH_EVENT"]==1]),
         len(diabetes_no[heart_data["DEATH_EVENT"]==0]),len(diabetes_no[heart_data["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Analysis on Survival Based on Diabetes")
fig.show()

#4.d Anemia Distribution Pie Chart
anaemia_yes = heart_data[heart_data['anaemia']==1]
anaemia_no = heart_data[heart_data['anaemia']==0]

labels = ['No Anaemia', 'Anaemia']
values = [len(anaemia_no), len(anaemia_yes)]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Anemia Distribution Pie Chart")
fig.show()

#4.e Analysis on Anemia vs Death Event Ratio
fig = px.pie(heart_data, values='anaemia',names='DEATH_EVENT', title='Analysis on Anemia vs Death Event Ratio')
fig.show()

#4.f Analysis on Survival Based on Anemia
anaemia_yes_survived = anaemia_yes[heart_data["DEATH_EVENT"]==0]
anaemia_yes_not_survived = anaemia_yes[heart_data["DEATH_EVENT"]==1]
anaemia_no_survived = anaemia_no[heart_data["DEATH_EVENT"]==0]
anaemia_no_not_survived = anaemia_no[heart_data["DEATH_EVENT"]==1]

labels = ['Anaemia Yes - Survived','Anaemia Yes - Not Survived', 'Anaemia No - Survived', 'Anaemia NO - Not Survived']
values = [len(anaemia_yes[heart_data["DEATH_EVENT"]==0]),len(anaemia_yes[heart_data["DEATH_EVENT"]==1]),
         len(anaemia_no[heart_data["DEATH_EVENT"]==0]),len(anaemia_no[heart_data["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Analysis on Survival Based on Anemia")
fig.show()

#4.g High Blood Pressure Distribution Pie Chart
hbp_yes = heart_data[heart_data['high_blood_pressure']==1]
hbp_no = heart_data[heart_data['high_blood_pressure']==0]

labels = ["No High BP","High BP"]
values = [len(hbp_no), len(hbp_yes)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="High Blood Pressure Distribution Pie Chart")
fig.show()

#4.h Analysis on High Blood Pressure vs Death Event Ratio
fig = px.pie(heart_data, values='high_blood_pressure',names='DEATH_EVENT', title='Analysis on High Blood Pressure vs Death Event Ratio')
fig.show()

#4.i Analysis on Survival Based on High Blood Pressure
hbp_yes_survived = hbp_yes[heart_data["DEATH_EVENT"]==0]
hbp_yes_not_survived = hbp_yes[heart_data["DEATH_EVENT"]==1]
hbp_no_survived = hbp_no[heart_data["DEATH_EVENT"]==0]
hbp_no_not_survived = hbp_no[heart_data["DEATH_EVENT"]==1]

labels = ['HBP Yes - Survived','HBP Yes - Not Survived', 'HBP No - Survived', 'HBP NO - Not Survived']
values = [len(hbp_yes[heart_data["DEATH_EVENT"]==0]),len(hbp_yes[heart_data["DEATH_EVENT"]==1]),
         len(hbp_no[heart_data["DEATH_EVENT"]==0]),len(hbp_no[heart_data["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Analysis on Survival Based on High Blood Pressure")
fig.show()

#4.j Smoking Distribution Pie Chart
smoking_yes = heart_data[heart_data['smoking']==1]
smoking_no = heart_data[heart_data['smoking']==0]

labels = ['No Smoking','Smoking']
values = [len(smoking_no), len(smoking_yes)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Smoking Distribution Pie Chart")
fig.show()

#4.h Analysis on Smoking vs Death Event Ratio
fig = px.pie(heart_data, values='smoking',names='DEATH_EVENT', title='Analysis on Smoking vs Death Event Ratio')
fig.show()

#4.i Analysis on Survival Based on Smoking
smoking_yes_survived = smoking_yes[heart_data["DEATH_EVENT"]==0]
smoking_yes_not_survived = smoking_yes[heart_data["DEATH_EVENT"]==1]
smoking_no_survived = smoking_no[heart_data["DEATH_EVENT"]==0]
smoking_no_not_survived = smoking_no[heart_data["DEATH_EVENT"]==1]

labels = ['Smoking Yes - Survived','Smoking Yes - Not Survived', 'Smoking No - Survived', 'Smoking NO- Not Survived']
values = [len(smoking_yes[heart_data["DEATH_EVENT"]==0]),len(smoking_yes[heart_data["DEATH_EVENT"]==1]),
         len(smoking_no[heart_data["DEATH_EVENT"]==0]),len(smoking_no[heart_data["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Analysis on Survival Based on Smoking")
fig.show()

#HEAT MAP
plt.figure(figsize=(10,10))
sns.heatmap(heart_data.corr(), vmin=-1, cmap='coolwarm', annot=True);
plt.show()

