#!/usr/bin/env python3


#########################
### IMPORT EVERYTHING ###
#########################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from nltk.tokenize import TreebankWordTokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, SimpleRNN
import keras_metrics

import gensim.downloader as api
import string
import math
import sys

######################
### SET PARAMETERS ###
######################

DIR = "/Users/markwicks/Desktop/ML/HW/HW4"
FILENAME = "fake_job_postings.csv"


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
np.set_printoptions(suppress=True)

SAMPLE_SIZE = 1 # fraction of dataset to sample
DEV_SHARE  = 0.15 # fraction of dataset for dev set
TEST_SHARE = 0.15 # fraction of dataset for test set

###############################################################################
###############################################################################
# Input is numpy array of strings
# Output is a list of word vectors, indexed by input sentence number
def tokenize_and_vectorize(input_data):
    
    #w2v_embedding = api.load('word2vec-google-news-300')
    w2v_embedding = api.load("glove-wiki-gigaword-100")  

    tokenizer = TreebankWordTokenizer()
    vectorized_data = []

    for element in input_data:

        #cleaned_string = clean_string(element)
        
        #print("  -->" + str(element))
        
        if str(element)=='nan': element=""
        
        tokens = tokenizer.tokenize(element)
        embedding_list = []
        
        for token in tokens:
            
            try:
                embedding_list.append(w2v_embedding[token.lower()])
            except KeyError:
                pass
            
        vectorized_data.append(embedding_list)
        
    del w2v_embedding
        
    return vectorized_data

###############################################################################
# Input is numpy array of strings
# Output is a list of word vectors, indexed by input sentence number
def pad_trunc(data, max_length):
    
    new_data = []
    zero_vector = []
    
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
        
    for sample in data:
        
        if len(sample) > max_length:
            temp = sample[:max_length]
            
        elif len(sample) < max_length:
            temp = sample
            additional_elems = max_length - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
                
        else:
            temp = sample
            
        new_data.append(temp)
        
    return new_data
###############################################################################
# Input is numpy array of strings
# Output is a list of word vectors, indexed by input sentence number
def pad_trunc_2(data):
    
    max_length=100
    
    n,m = data.shape
    
    m_diff = max_length-m
    
    pad = np.zeros([n, m_diff])
    
    new_data = pd.concat([pd.DataFrame(data), pd.DataFrame(pad)], axis=1).values
     
    return new_data

###############################################################################
#################
### READ DATA ###
#################
def READ_DATA(DIR, FILENAME, SAMPLE_SIZE):
    
    print("")
    print(" --> Reading data")
    print("")

    RAW_DATA = pd.read_csv(DIR + '/DATA/' + FILENAME) 
    #RAW_DATA.drop(columns=['job_id'])
    RAW_DATA = RAW_DATA.sample(frac=SAMPLE_SIZE)
    
    return RAW_DATA

###############################################################################
#########################
### CLEAN THE DATASET ###
#########################
def CLEAN_DATA(DATA, BAG_OF_WORDS, INCLUDE_TEXT_VARS):

### VARIABLES ###
# job_id',               <-- junk
# 'title',               <-- string field (1490 different values)
# 'location',            <-- tuple (country, state, city)
# 'department',          <-- string field (271 different values)
# 'salary_range',        <-- string field
# 'company_profile',     <-- string field
# 'description',         <-- string field
# 'requirements',        <-- string field
# 'benefits',            <-- string field
# 'telecommuting',       <-- binary
# 'has_company_logo',    <-- binary
# 'has_questions',       <-- binary
# 'employment_type',     <-- categorical (6 different values)   converted to dummies
# 'required_experience', <-- categorical (8 different values)   converted to dummies
# 'required_education',  <-- categorical (13 different values)  converted to dummies
# 'industry',            <-- string field (98 different values)
# 'function',            <-- categorical (38 different values)  converted to dummies
# 'fraudulent'           <-- response variable    
    
    print("")
    print(" --> Cleaning data")
    print("")
     
    ###############################################
    ### SPLIT LOCATION INTO COUNTY, STATE, CITY ###
    ###############################################
    
    DATA['country'] = ""
    DATA['state'] = ""
    DATA['city'] = ""

    for x in range(DATA.shape[0]):
        
        val = str(DATA.location.values[x]).replace(", ,", ",")
        if val==" ": val=""
 
        count = val.count(",")
        
        if count==2:
            
           COUNTRY, STATE, CITY = val.split(',')
           
           DATA.country.values[x] = COUNTRY
           DATA.state.values[x]   = STATE   
           DATA.city.values[x]    = CITY
                     
           #print("-->" +str(COUNTRY) + "    " + 
           #      str(STATE) + "   " + str(type(CITY)))
           
        elif count==3:
            
           COUNTRY, STATE, CITY, JUNK = val.split(',')
           
        else: continue

        DATA.country.values[x] = COUNTRY.replace("  ", " ")
        DATA.state.values[x]   = STATE.replace("  ", " ") 
        DATA.city.values[x]    = CITY.replace("  ", " ")

        
    ###########################################
    ### CONVERT COUNTRY VARIABLE TO DUMMIES ###
    ###########################################
    
    COUNTRY_DUMMIES = pd.get_dummies(DATA[['country']])[['country_US', "country_",
                                                         "country_GB", "country_GR", 
                                                         "country_CA", "country_DE", 
                                                         "country_NZ", "country_IN", 
                                                         "country_AU"]] 
    
    DATA = pd.concat([DATA, COUNTRY_DUMMIES], axis=1)     
        
    ########################################
    ### CONVERT CITY VARIABLE TO DUMMIES ###
    ########################################
        
    CITY_DUMMIES = pd.get_dummies(DATA[['city']])[['city_ London', "city_", "city_ New York", "city_ Athens", 
                                                   'city_ San Francisco', 'city_ Houston', 'city_ Washington',
                                                   'city_ Chicago', 'city_ Berlin', 'city_ Auckland', 'city_ Los Angeles',
                                                   'city_ Austin', 'city_ San Diego', 'city_ Atlanta', 'city_ Portland',
                                                   'city_ Toronto', 'city_ Boston', 'city_ Philadelphia', 'city_ Detroit']]
    
    CITY_DUMMIES = CITY_DUMMIES.rename(columns={
                                          "city_ ":              "city_null",
                                          "city_ London":        "city_london", 
                                          "city_ New York":      "city_new_york",
                                          "city_ Athens":        "city_athen",
                                          "city_ San Francisco": "city_san_fran",
                                          "city_ Houston":        "city_houson",
                                          "city_ Washington":    "city_washington",
                                          "city_ Chicago":       "city_chicago",
                                          "city_ Berlin":        "city_berlin",
                                          "city_ Auckland":      "city_auckland",
                                          "city_ Los Angeles":   "city_los_angeles",
                                          "city_ Austin":        "city_austin",
                                          "city_ San Diego":     "city_san_diego",
                                          "city_ Atlanta":       "city_atlanta",
                                          "city_ Portland":      "city_portland",
                                          "city_ Toronto":       "city_toronto",
                                          "city_ Boston":        "city_boston",
                                          "city_ Philadelphia":  "city_philadelphia",
                                          "city_ Detroit":       "city_detroit"
                                           })
    
    DATA = pd.concat([DATA, CITY_DUMMIES], axis=1)   
    
    ###################################
    ### SPLIT salary_range VARIABLE ###
    ###################################
    
    DATA['low_salary_range'] = np.NaN
    DATA['high_salary_range'] = np.NaN

    for x in range(DATA.shape[0]):
        val = str(DATA.salary_range.values[x])
        if ('-' in val):
           LOW, HIGH = val.split('-')
           #print("-->" + val + "   " + str(LOW) + "    " + str(HIGH) + "   " + str(type(LOW)))
           if str.isdecimal(LOW) and str.isdecimal(HIGH):
              DATA.low_salary_range.values[x] = LOW
              DATA.high_salary_range.values[x] = HIGH

    ##############################################
    ### CONVERT required_experience TO DUMMIES ###
    ##############################################
    # need to create categories for blank fields

    # dummy_na=True gives us a column for the NAs
    REQ_EXP_DUMMIES = pd.get_dummies(DATA[['required_experience']], dummy_na=True)
    
    REQ_EXP_DUMMIES = REQ_EXP_DUMMIES.rename(columns={
                   "required_experience_Associate":        "req_exp_associate", 
                   "required_experience_Director":         "req_exp_director",
                   "required_experience_Entry level":      "req_exp_entry_level",
                   "required_experience_Executive":        "req_exp_executive",
                   "required_experience_Internship":       "req_exp_internship",
                   "required_experience_Mid-Senior level": "req_exp_mid_senior",
                   "required_experience_Not Applicable":   "req_exp_not_applicable"
                    })
    
    DATA = pd.concat([DATA, REQ_EXP_DUMMIES], axis=1)
    
    ##########################################
    ### CONVERT employment_type TO DUMMIES ###
    ##########################################
    # need to create categories for blank fields

    # dummy_na=True gives us a column for the NAs
    EMP_TYPE_DUMMIES = pd.get_dummies(DATA[['employment_type']], dummy_na=True)
    
    EMP_TYPE_DUMMIES = EMP_TYPE_DUMMIES.rename(columns={
                   "employment_type_Contract":   "emp_type_contract",
                   "employment_type_Full-time":  "emp_type_full_time",
                   "employment_type_Other":      "emp_type_other",
                   "employment_type_Part-time":  "emp_type_part_time",
                   "employment_type_Temporary":  "emp_type_temp"
                   })  
    
    DATA = pd.concat([DATA, EMP_TYPE_DUMMIES], axis=1)   
    
    ##########################################
    ### CONVERT function_type TO DUMMIES ###
    ##########################################   
    
    # Need to add a dummy variable for areas not included in this list
    FUNCTION_DUMMIES = pd.get_dummies(DATA.function)[['Information Technology', 'Sales', 'Engineering', 
                                                      'Customer Service', 'Marketing', 'Administrative', 'Other',
                                                      'Health Care Provider', 'Management', 'Design']]
    
    DATA = pd.concat([DATA, FUNCTION_DUMMIES], axis=1)
    
    DATA=DATA.rename(columns={"Information Technology":  "func_IT",
                              "Sales":                   "func_sales",
                              "Engineering":             "func_engineering",
                              "Customer Service":        "func_customer_serv",
                              "Marketing":               "func_marketing",
                              "Administrative":          "func_admin",
                              "Other":                   "func_other",
                              "Health Care Provider":    "func_healthcare",
                              "Management":              "func_management",
                              "Design":                  "func_design"
                              })
    
    #############################################
    ### CONVERT required_education TO DUMMIES ###
    #############################################  
    
    # Only do so for the most frequent values. Need to create 'other' and blank categories.
    EDUCATION_DUMMIES = pd.get_dummies(DATA.required_education)[["Bachelor's Degree", 'High School or equivalent',
                                                                 'Unspecified', "Master's Degree", "Associate Degree",
                                                                 "Certification", "Some College Coursework Completed"]]
    
    EDUCATION_DUMMIES=EDUCATION_DUMMIES.rename(
            columns={"Bachelor's Degree":                    "edu_bachelors",
                     "High School or equivalent":            "edu_high_school",
                     "Unspecified":                          "edu_unspecified",
                     "Master's Degree":                      "edu_masters",
                     "Associate Degree":                     "edu_associate",
                     "Certification":                        "edu_certification",
                     "Some College Coursework Completed":    "edu_some_college"
                              })    
    
    DATA = pd.concat([DATA, EDUCATION_DUMMIES], axis=1)

    #################################################
    ### CREATE DUMMY VARIABLES FOR NULL VARIABLES ###
    #################################################
    
    DATA['company_profile_is_null']=0
    DATA.loc[pd.isna(DATA['company_profile']), 'company_profile_is_null'] = 1
    DATA.loc[pd.isna(DATA['company_profile']), 'company_profile'] = ""
    
    
    DATA['department_is_null']=0
    DATA.loc[pd.isna(DATA['department']), 'department_is_null'] = 1 
    DATA.loc[pd.isna(DATA['department']), 'department'] = ""  
    
    DATA['company_profile_is_null']=0
    DATA.loc[pd.isna(DATA['company_profile']), 'company_profile_is_null'] = 1
    DATA.loc[pd.isna(DATA['company_profile']), 'company_profile'] = ""  

    # only 1 missing for this one
    #DATA['description_is_null']=0
    #DATA.loc[pd.isna(DATA['description']), 'description_is_null'] = 1
    DATA.loc[pd.isna(DATA['description']), 'description'] = ""

    DATA['requirements_is_null']=0
    DATA.loc[pd.isna(DATA['requirements']), 'requirements_is_null'] = 1
    DATA.loc[pd.isna(DATA['requirements']), 'requirements'] = "" 
    
    DATA['benefits_is_null']=0
    DATA.loc[pd.isna(DATA['benefits']), 'benefits_is_null'] = 1  
    DATA.loc[pd.isna(DATA['benefits']), 'benefits'] = ""
    
    DATA['industry_is_null']=0
    DATA.loc[pd.isna(DATA['industry']), 'industry_is_null'] = 1 
    DATA.loc[pd.isna(DATA['industry']), 'industry'] = ""   

    DATA['low_salary_range_is_null']=0
    DATA.loc[pd.isna(DATA['low_salary_range']), 'low_salary_range_is_null'] = 1  
    DATA.loc[pd.isna(DATA['low_salary_range']), 'low_salary_range'] = 0 

    DATA['high_salary_range_is_null']=0
    DATA.loc[pd.isna(DATA['high_salary_range']), 'high_salary_range_is_null'] = 1  
    DATA.loc[pd.isna(DATA['high_salary_range']), 'high_salary_range'] = 0     
    
    DATA['country_is_null']=0
    DATA.loc[pd.isna(DATA['country']), 'country_is_null'] = 1 
    
    DATA['city_is_null']=0
    DATA.loc[pd.isna(DATA['city']), 'city_is_null'] = 1 
    
    
    ###################################################
    ### GENERATE VARIABLES FOR LENGTH OF TEXT FIELD ###
    ###################################################

    DATA['title_length']           = DATA.title.str.len().fillna(1)
    DATA['company_profile_length'] = DATA.company_profile.str.len().fillna(1)  
    DATA['description_length']     = np.log(DATA.description.str.len().fillna(1)+ 0.0000001) # take log
    DATA['requirements_length']    = np.log(DATA.requirements.str.len().fillna(1)+ 0.0000001) # take log
    DATA['benefits_length']        = np.log(DATA.benefits.str.len().fillna(1)+ 0.0000001)  # take log
    DATA['industry_length']        = DATA.industry.str.len().fillna(1)
    
    #############################################
    ### GENERATE BINARY BAG OF WORDS FEATURES ###
    #############################################    
    
    if INCLUDE_TEXT_VARS==True:
    
       DATA['all_text']= (DATA['title'] + " " + 
                          DATA['company_profile'] + " " + 
                          DATA['description'] + " " + 
                          DATA['requirements'] + " " + 
                          DATA['benefits'] + " " +
                          DATA['industry']
                          )
 
    if BAG_OF_WORDS:
          VECTORIZER = CountVectorizer(binary=True, min_df=30)
          BAG_OF_WORDS = VECTORIZER.fit_transform(DATA.all_text_fields_combined).toarray()
          BAG_OF_WORDS_DF = pd.DataFrame(BAG_OF_WORDS, columns=list(VECTORIZER.vocabulary_.keys()))
          BAG_OF_WORDS_DF = BAG_OF_WORDS_DF.rename(columns={"fraudulent":  "fraudulent_2"})
    
          #VECTORIZER.vocabulary_
          DATA = pd.concat([DATA, BAG_OF_WORDS_DF], axis=1)    
    
    #####################################
    ###  REMOVE STRING FIELDS FOR NOW ###
    #####################################  
    
    # remove string fields
    DATA = DATA.drop(columns=['job_id', 'title', 'department', 'company_profile',
                              'description', 'requirements', 'benefits', 'industry'])
    
    #################################################
    ### DROP COLUMNS CONVERTED TO DUMMY VARIABLES ###
    #################################################
    
    DATA = DATA.drop(columns=['salary_range', 'required_experience', 'employment_type', 
                              'function', 'required_education', 'city', 'country', 
                              'state', 'location', 'low_salary_range', 'high_salary_range'])  
    
    return DATA

###############################################################################
#######################################
### SUBSET DATA INTO TRAIN/DEV/TEST ###
#######################################
def TRAIN_DEV_TEST_SPLIT(INPUT_DATA, DEV_SHARE, TEST_SHARE):
        
    # Sort the data
    INPUT_DATA = INPUT_DATA.sample(frac=1, random_state=1)
    
    # Use the test_share and dev_share variables hard-coded at the top to 
    # compute the number of observations in the train and dev data sets. Then
    # use these values to divide up the raw data set into 3 parts.
    NUM_OBS       = INPUT_DATA.shape[0]
    NUM_TRAIN_OBS = math.ceil(NUM_OBS*(1-TEST_SHARE-DEV_SHARE))
    NUM_DEV_OBS   = math.floor(NUM_OBS*DEV_SHARE)

    TRAIN_DATA = INPUT_DATA[:NUM_TRAIN_OBS]
    DEV_DATA   = INPUT_DATA[NUM_TRAIN_OBS:(NUM_TRAIN_OBS+NUM_DEV_OBS)]
    TEST_DATA  = INPUT_DATA[(NUM_DEV_OBS+NUM_TRAIN_OBS):]
    
    return TRAIN_DATA, DEV_DATA, TEST_DATA

###############################################################################  
# Input a vector of predicted Y values and the actual Y values. Return accuracy
# rate. 
def GET_ACCURACY_RATE(Y_HAT_VECTOR, Y_VECTOR, PRINT_ACCURACY_RATE):
    
    Y_HAT_VECTOR_LEN = Y_HAT_VECTOR.shape[0]
    Y_VECTOR_LEN     = Y_VECTOR.shape[0]
    
    # The two input vectors must have the same length.
    if Y_HAT_VECTOR_LEN != Y_VECTOR_LEN: 
       sys.exit('ERROR: Vectors of different lengths in GET_ACCURACY_RATE')
       
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for x in range(Y_VECTOR_LEN):
        if Y_HAT_VECTOR[x]==1  and Y_VECTOR[x]==1: TP += 1        
        if Y_HAT_VECTOR[x]==1  and Y_VECTOR[x]==0: FP += 1
        if Y_HAT_VECTOR[x]==0 and Y_VECTOR[x]==0: TN += 1
        if Y_HAT_VECTOR[x]==0 and Y_VECTOR[x]==1: FN += 1        
        
    ACCURACY  = (TP+TN)/Y_HAT_VECTOR_LEN*100
    RECALL    = TP/(TP+FN)*100
    PRECISION = TP/(TP+FP)*100
    
    if PRINT_ACCURACY_RATE==True:
       print("")
       print(" --> Accuracy Rate: " + str(round(ACCURACY,2)) + "%")
       print(" --> Recall: " + str(round(RECALL,2)) + "%")
       print(" --> Precision: " + str(round(PRECISION,2)) + "%")
       print("")
       
    return ACCURACY
###############################################################################
def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    #colors = [‘red’ if c < 0 else ‘blue’ for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color='blue')
    feature_names = np.array(feature_names)
    #plt.xticks(np.arange(1, 1 + 2 * top_features),feature_names[top_coefficients], rotation=60, ha=’right’)
    plt.xticks(np.arange(1, 1+2 * top_features), feature_names[top_coefficients], rotation=90)
    plt.show()
###############################################################################
def LOGISTIC_REGRESSION(TRAIN_DATA_X, TRAIN_DATA_Y, DEV_DATA_X, DEV_DATA_Y):
    
    LOGISTIC_REG = (LogisticRegression(penalty='l2', solver='lbfgs', max_iter=250).
                    fit(TRAIN_DATA_X, TRAIN_DATA_Y))

    # Generate the predicted values
    Y_HAT = (LOGISTIC_REG.predict(DEV_DATA_X))
    GET_ACCURACY_RATE(Y_HAT, DEV_DATA_Y.values, True)
    
    COEF = LOGISTIC_REG.coef_
    COEF_DF = pd.DataFrame(COEF, columns=DEV_DATA_X.columns)
    plot_coefficients(LOGISTIC_REG, TRAIN_DATA_X.columns, 25)    
    
###############################################################################
def run_neural_network(X_TRAIN_DATA, X_DEV_DATA, Y_TRAIN, Y_DEV, 
                       num_neurons, max_length, embedding_length, num_epochs):
    
    print(" --> Running neural network")
    print("")

    ###################
    ### FORMAT DATA ###
    ###################
    
    # convert each text string to a sequence of word vectors 
    # Each observation is a vector of vectors.
    X_TRAIN_VECTORIZED = tokenize_and_vectorize(X_TRAIN_DATA.all_text.values)
    X_DEV_VECTORIZED   = tokenize_and_vectorize(X_DEV_DATA.all_text.values)

    ##################################
    ### FORMAT THE INPUT DATA SETS ###
    ##################################

    # Crop if in excess of max_length, and pad with zeros if shorter than max_length
    X_TRAIN_PADDED_DATA = pad_trunc(X_TRAIN_VECTORIZED, max_length)
    X_DEV_PADDED_DATA   = pad_trunc(X_DEV_VECTORIZED, max_length)

    # First dimension is number of observations in dataset, second dimension
    # is max number of words, and third is the embedding length
    # Each observation therefore has input dimension (300, 100)
    X_TRAIN_PADDED_FORMATTED_DATA = np.reshape(X_TRAIN_PADDED_DATA, 
                                              (len(X_TRAIN_PADDED_DATA), 
                                               max_length, 
                                               embedding_length))

    X_DEV_PADDED_FORMATTED_DATA = np.reshape(X_DEV_PADDED_DATA, 
                                            (len(X_DEV_PADDED_DATA), 
                                             max_length, 
                                             embedding_length))
    
    ###########################
    ### CREATE LSTM NETWORK ###
    ###########################

    model = Sequential()
    
    model.add(LSTM(num_neurons, 
                   return_sequences = True, 
                   input_shape = (max_length, embedding_length)))

    
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile('rmsprop', 'binary_crossentropy', 
                  metrics = ['accuracy',
                             keras_metrics.precision(), 
                             keras_metrics.recall()])
    model.summary()

    ##########################
    ### TRAIN LSTM NETWORK ###
    ##########################

    model.fit(X_TRAIN_PADDED_FORMATTED_DATA, 
              Y_TRAIN,
              batch_size = 32,
              epochs = num_epochs,
              validation_data = (X_DEV_PADDED_FORMATTED_DATA, 
                                 Y_DEV.values))
    
    # Value between 0 and 1.
    PRED = model.predict(X_DEV_PADDED_FORMATTED_DATA)

    YHAT = np.where(PRED > 0.5, 1, 0)
    
    GET_ACCURACY_RATE(YHAT, DEV_DATA_Y.values, True)
 
    #######################
    ### RUN ON TEST SET ###
    #######################  
    

    
###############################################################################   
#################
### READ DATA ###
#################
    
DATA = READ_DATA(DIR, FILENAME, SAMPLE_SIZE)

##################
### CLEAN DATA ###
##################

CLEANED_DATA = CLEAN_DATA(DATA, BAG_OF_WORDS=False, INCLUDE_TEXT_VARS=True)

############################
### TRAIN/DEV/TEST SPLIT ###
############################

TRAIN_DATA, DEV_DATA, TEST_DATA = TRAIN_DEV_TEST_SPLIT(CLEANED_DATA, DEV_SHARE, TEST_SHARE)

TRAIN_DATA_X = TRAIN_DATA.drop(columns=['fraudulent'])
TRAIN_DATA_Y = TRAIN_DATA.fraudulent

DEV_DATA_X = DEV_DATA.drop(columns=['fraudulent'])
DEV_DATA_Y = DEV_DATA.fraudulent

TEST_DATA_X = TEST_DATA.drop(columns=['fraudulent'])
TEST_DATA_Y = TEST_DATA.fraudulent

#################
### MODELLING ###
#################

LOGISTIC_REGRESSION(TRAIN_DATA_X, TRAIN_DATA_Y, DEV_DATA_X, DEV_DATA_Y)

run_neural_network(X_TRAIN_DATA     = TRAIN_DATA_X, 
                   X_DEV_DATA       = DEV_DATA_X, 
                   Y_TRAIN          = TRAIN_DATA_Y,
                   Y_DEV            = DEV_DATA_Y,
                   num_neurons      = 32, 
                   max_length       = 300, 
                   embedding_length = 100,
                   num_epochs       = 5)

########################
### DATA EXPLORATION ###
########################

NOT_FAKE = DATA[DATA['fraudulent']==0]

NOT_FAKE.city.value_counts()
NOT_FAKE.employment_type.value_counts()
NOT_FAKE.industry.value_counts()
NOT_FAKE.has_company_logo.value_counts()
NOT_FAKE.required_education.value_counts()

############
### FAKE ###
############

FAKE = DATA[DATA['fraudulent']==1]

FAKE.city.value_counts()
FAKE.employment_type.value_counts()
FAKE.industry.value_counts()
FAKE.has_company_logo.value_counts()
FAKE.required_education.value_counts()



#############################
### GENERATE WORD VECTORS ###
#############################

#WORD_VECTOR = tokenize_and_vectorize(TRAIN_DATA.description)
#WORD_VECTOR_TRUNCATED = pad_trunc(data = WORD_VECTOR, max_length = 300)








###############################################################################
