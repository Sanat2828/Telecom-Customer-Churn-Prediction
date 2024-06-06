# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

df_1=pd.read_csv("first_Telecom.csv")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    
    '''
    gender
    SeniorCitizen
    Partner
    Dependents
    tenure
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    MonthlyCharges
    TotalCharges
    '''
    
    inputQuery1 = request.form.get('query1', 0)
    inputQuery2 = request.form.get('query2', 0)
    inputQuery3 = request.form.get('query3', 0)
    inputQuery4 = request.form.get('query4', 0)
    inputQuery5 = request.form.get('query5', 0)
    inputQuery6 = request.form.get('query6', 0)
    inputQuery7 = request.form.get('query7', 0)
    inputQuery8 = request.form.get('query8', 0)
    inputQuery9 = request.form.get('query9', 0)
    inputQuery10 = request.form.get('query10', 0)
    inputQuery11 = request.form.get('query11', 0)
    inputQuery12 = request.form.get('query12', 0)
    inputQuery13 = request.form.get('query13', 0)
    inputQuery14 = request.form.get('query14', 0)
    inputQuery15 = request.form.get('query15', 0)
    inputQuery16 = request.form.get('query16', 0)
    inputQuery17 = request.form.get('query17', 0)
    inputQuery18 = request.form.get('query18', 0)
    inputQuery19 = request.form.get('query19', 0)

    model = pickle.load(open("model.sav", "rb"))
    
    data = [[inputQuery4,
             inputQuery1, 
             inputQuery5,     
             inputQuery6, 
             inputQuery19, 
             inputQuery7,  
             inputQuery8, 
             inputQuery9, 
             inputQuery10, 
             inputQuery11, 
             inputQuery12, 
             inputQuery13, 
             inputQuery14, 
             inputQuery15, 
             inputQuery16, 
             inputQuery17, 
             inputQuery18, 
             inputQuery2, 
             inputQuery3]]
    
    new_df = pd.DataFrame(data, columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure_group', 'PhoneService',
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
           'Contract', 'PaperlessBilling', 'PaymentMethod','MonthlyCharges', 'TotalCharges'])
    
    df_2 = pd.concat([df_1, new_df], ignore_index = True) 
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    
    df_2.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_2.dropna(inplace=True)
    # print(df_2.isna())
    print(df_2.isnull().sum())
    print(df_2.head(10))

    df_2['tenure_group'] = pd.cut(df_2['tenure_group'].astype(int), range(1, 80, 12), right=False, labels=labels)
    
    print(df_2.head(10))
    #drop column customerID and tenure
    # df_2.drop(columns= ['tenure_group'], axis=1, inplace=True)   
    # print(df_2.head(10))
    df_2['SeniorCitizen'] = pd.to_numeric(df_2['SeniorCitizen'], errors='coerce')
    df_2['tenure_group'] = pd.to_numeric(df_2['tenure_group'], errors='coerce')
    df_2['MonthlyCharges'] = pd.to_numeric(df_2['MonthlyCharges'], errors='coerce')
    df_2['TotalCharges'] = pd.to_numeric(df_2['TotalCharges'], errors='coerce')
    df_2['SeniorCitizen'] = df_2.SeniorCitizen.astype(int)
    df_2['tenure_group'] = df_2.tenure_group.astype(int)
    df_2['MonthlyCharges'] = df_2.MonthlyCharges.astype(float)
    df_2['TotalCharges'] = df_2.TotalCharges.astype(float)


    new_df__dummies = pd.get_dummies(df_2
    [['gender', 
    'SeniorCitizen', 
    'Partner', 
    'Dependents', 
    'tenure_group', 
    'PhoneService', 
    'MultipleLines', 
    'InternetService', 
    'OnlineSecurity', 
    'OnlineBackup', 
    'DeviceProtection', 
    'TechSupport', 
    'StreamingTV', 
    'StreamingMovies', 
    'Contract', 
    'PaperlessBilling', 
    'PaymentMethod', 
    'MonthlyCharges', 
    'TotalCharges']])
    print(new_df__dummies.head(10))
    #final_df=pd.concat([new_df__dummies, new_dummy], axis=1)
        
    single = model.predict(new_df__dummies.tail(1))
    probablity = model.predict_proba(new_df__dummies.tail(1))[:,1]
    
    if single==1:
        o1 = "This customer is likely to be churned"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('home.html', output1=o1, output2=o2, query1 = request.form.get('query1', 0), 
                           query2 = request.form.get('query2', 0),
                           query3 = request.form.get('query3', 0),
                           query4 = request.form.get('query4', 0),
                           query5 = request.form.get('query5', 0), 
                           query6 = request.form.get('query6', 0), 
                           query7 = request.form.get('query7', 0), 
                           query8 = request.form.get('query8', 0), 
                           query9 = request.form.get('query9', 0), 
                           query10 = request.form.get('query10', 0), 
                           query11 = request.form.get('query11', 0), 
                           query12 = request.form.get('query12', 0), 
                           query13 = request.form.get('query13', 0), 
                           query14 = request.form.get('query14', 0), 
                           query15 = request.form.get('query15', 0), 
                           query16 = request.form.get('query16', 0), 
                           query17 = request.form.get('query17', 0),
                           query18 = request.form.get('query18', 0), 
                           query19 = request.form.get('query19', 0))
    
app.run()