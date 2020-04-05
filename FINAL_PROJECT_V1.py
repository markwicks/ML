#!/usr/bin/env python3


#########################
### IMPORT EVERYTHING ###
#########################

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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

SAMPLE_SIZE = 0.1 # fraction of dataset to sample
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
#################
### READ DATA ###
#################
def READ_DATA(DIR, FILENAME, SAMPLE_SIZE):
    
    print("")
    print(" --> Reading data")
    print("")

    RAW_DATA = pd.read_csv(DIR + '/DATA/' + FILENAME) 
    RAW_DATA.drop(columns=['job_id'])
    RAW_DATA = RAW_DATA.sample(frac=SAMPLE_SIZE)
    
    return RAW_DATA

###############################################################################
#########################
### CLEAN THE DATASET ###
#########################
def CLEAN_DATA(DATA):

    
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

    ##########################################
    ### CONVERT employment_type TO DUMMIES ###
    ##########################################

    # dummy_na=True gives us a column for the NAs
    DUMMIES = pd.get_dummies(DATA[['required_experience', 'employment_type']], dummy_na=True)
    DATA.drop(columns=['required_experience', 'employment_type'])
    DATA = pd.concat([DATA, DUMMIES], axis=1)
    
    FUNCTION_DUMMIES = pd.get_dummies(DATA.function)[['Information Technology', 'Sales', 'Engineering', 
                                                      'Customer Service', 'Marketing', 'Administrative', 'Other',
                                                      'Health Care Provider', 'Management', 'Design']]
    
    DATA = pd.concat([DATA, FUNCTION_DUMMIES], axis=1)

    ########################
    ### RENAME VARIABLES ###
    ########################
    
    DATA.rename(columns={"required_experience_Associate":        "req_exp_associate", 
                         "required_experience_Director":         "req_exp_director",
                         "required_experience_Entry level":      "req_exp_entry_level",
                         "required_experience_Executive":        "req_exp_executive",
                         "required_experience_Internship":       "req_exp_internship",
                         "required_experience_Mid-Senior level": "req_exp_mid_senior",
                         "required_experience_Not Applicable":   "req_exp_not_applicable",
                         
                         "employment_type_Contract":             "emp_type_contract",
                         "employment_type_Full-time":            "emp_type_full_time",
                         "employment_type_Other":                "emp_type_other",
                         "employment_type_Part-time":            "emp_type_part_time",
                         "employment_type_Temporary":            "emp_type_temp",                         
                         })
    
    return DATA

###############################################################################
#######################################
### SUBSET DATA INTO TRAIN/DEV/TEST ###
#######################################
def TRAIN_DEV_TEST_SPLIT(INPUT_DATA, DEV_SHARE, TEST_SHARE):
        
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
        if Y_HAT_VECTOR[x]==1  and Y_VECTOR[x]==-1: FP += 1
        if Y_HAT_VECTOR[x]==-1 and Y_VECTOR[x]==-1: TN += 1
        if Y_HAT_VECTOR[x]==-1 and Y_VECTOR[x]==1: FN += 1        
        
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

#################
### READ DATA ###
#################
    
DATA = READ_DATA(DIR, FILENAME, SAMPLE_SIZE)

##################
### CLEAN DATA ###
##################

DATA = CLEAN_DATA(DATA)

#DATA.low_salary_range.sort_values().unique()
############################
### TRAIN/DEV/TEST SPLIT ###
############################

TRAIN_DATA, DEV_DATA, TEST_DATA = TRAIN_DEV_TEST_SPLIT(DATA, DEV_SHARE, TEST_SHARE)

TRAIN_DATA_X = TRAIN_DATA.drop(columns=['fraudulent'])
TRAIN_DATA_Y = TRAIN_DATA.fraudulent

DEV_DATA_X = DEV_DATA.drop(columns=['fraudulent'])
DEV_DATA_Y = DEV_DATA.fraudulent

TEST_DATA_X = TEST_DATA.drop(columns=['fraudulent'])
TEST_DATA_Y = TEST_DATA.fraudulent

########################
### DATA EXPLORATION ###
########################

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
# 'employment_type',     <-- categorical (6 different values)
# 'required_experience', <-- categorical (8 different values)
# 'required_education',  <-- categorical (13 different values)
# 'industry',            <-- string field (98 different values)
# 'function',            <-- categorical (38 different values)
# 'fraudulent'           <-- response variable

print(TRAIN_DATA.columns)
print(TRAIN_DATA.shape)
DATA.function.value_counts()

#############################
### GENERATE WORD VECTORS ###
#############################

WORD_VECTOR = tokenize_and_vectorize(TRAIN_DATA.description)
WORD_VECTOR_TRUNCATED = pad_trunc(data = WORD_VECTOR, max_length = 300)


###############################################################################


















