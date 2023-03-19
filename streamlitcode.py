# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 11:01:38 2023

@author: Asus
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import os

# Loading the saved model 
loaded_model = pickle.load(open("rsf2.pkl", 'rb')) # rb = reading binary

# Creating a function for prediction 
def graft_surv_pred(input_data):    
    # changing the input_data to numpy array       
    # input_data = [37,25.84954573,8.855944644,32,25,45.97932939,70,10.32224936,0.138526645,15,1.846902311,13,48.95554566,110,55,2.325456113,0.695345368,8.623085328,0,1,0,5,0,0,1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,3,0]
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    surv_test_cph = loaded_model.predict_survival_function(input_data_reshaped, return_array=False)
    surv = loaded_model.predict_survival_function(input_data_reshaped, return_array=True)

    event_times = loaded_model.event_times_

    lower, upper = event_times[0], event_times[-1]
    y_times = np.arange(lower, upper)

    T1, T2 = surv_test_cph[0].x.min(),surv_test_cph[0].x.max()
    mask = np.logical_or(y_times >= T2, y_times < T1) # mask outer interval
    times = y_times[~mask]

    cph_surv_prob_test = np.row_stack([ fn(times) for fn in surv_test_cph])
    cph_surv_prob_test1 = pd.DataFrame(cph_surv_prob_test)

    six_mon = cph_surv_prob_test1.iloc[ :,180]*100
    one_yr = cph_surv_prob_test1.iloc[ :,360]*100
    two_yr = cph_surv_prob_test1.iloc[ :,730]*100
    three_yr = cph_surv_prob_test1.iloc[ :,1095]*100
    four_yr = cph_surv_prob_test1.iloc[ :,1460]*100
    five_yr = cph_surv_prob_test1.iloc[ :,-1]*100 
    
    plt.step(loaded_model.event_times_,surv[0], where="post", label=str(0))
    plt.ylabel("Survival probability")
    plt.xlabel("Time in days")
    plt.legend()
    plt.grid(True)
    st.pyplot()
    
    s = pd.DataFrame({'survival time' : ['Probability of survival at 6 months in % ',
                                'Probability of survival at 1 years in % ',
                                'Probability of survival at 2 years in % ',
                                'Probability of survival at 3 years in % ',
                                'Probability of survival at 4 years in % ',
                                'Probability of survival at 5 years in % '],
                     'probability' : [six_mon.to_string(index = False),
                                   one_yr.to_string(index = False),
                                   two_yr.to_string(index = False),
                                   three_yr.to_string(index = False),
                                   four_yr.to_string(index = False),
                                   five_yr.to_string(index = False)]})
    return(st.table(s))
    
    

def main():   
# Every good app has a title, so let's add one
    
    
    
    # Getting input data from user
    Blood_Urea_Nitrogen_level = st.text_input("Blood_Urea_Nitrogen_level :0-100")
    
    Body_mass_index = st.text_input('Body_mass_index')
    Calcium_level = st.text_input('Calcium level')
    Diastolic_blood_pressure = st.text_input('Diastolic blood pressure')
    Donor_age = st.text_input('Donor age')
    eGFR = st.text_input('eGFR value')
    Glucose_level = st.text_input('Glucose level')
    Hemoglobin_level = st.text_input('Hemoglobin level')
    Phosphorus_level = st.text_input('Phosphorus level')
    Platelets = st.text_input('Platelets')
    Potassium_level = st.text_input('Potassium level')
    Recipients_age = st.text_input('Recipients age')
    Serum_creatinine_level = st.text_input('Serum_creatinine_level')
    Sodium_level = st.text_input('Sodium_level')
    Systolic_blood_pressure = st.text_input('Systolic_blood_pressure')
    Tacrolimus_Modified_Release = st.text_input('Tacrolimus_Modified_Release')
    White_blood_cell_count = st.text_input('White_blood_cell_count')
    Blood_Urea_Nitrogen_level_to_Serum_creatinine_level_ratio = st.text_input('Blood_Urea_Nitrogen_level_to_Serum_creatinine_level_ratio')
    An_episode_of_acute_rejection = st.text_input('An_episode_of_acute_rejection')
    An_episode_of_chronic_rejection = st.text_input('An_episode_of_chronic_rejection')
    An_episode_of_hyperacute_rejection = st.text_input('An_episode_of_hyperacute_rejection')
    Donor_to_recipient_relationship = st.text_input('Donor_to_recipient_relationship')
    Family_history_of_kidney_disease = st.text_input('Family_history_of_kidney_disease')
    History_of_abdominal_surge = st.text_input('History_of_abdominal_surge')
    History_of_blood_transfusion = st.text_input('History_of_blood_transfusion')
    History_of_dialysis_before_transplant = st.text_input('History_of_dialysis_before_transplant')
    History_of_pre_transplant_comorbid = st.text_input('History_of_pre_transplant_comorbid')
    Post_transplant_Cardio_vascular_complications = st.text_input('Post_transplant_Cardio_vascular_complications')
    Post_transplant_Covid19 = st.text_input('Post_transplant_Covid19')
    Post_transplant_delayed_graft_function = st.text_input('Post_transplant_delayed_graft_function')
    Post_transplant_diabetes = st.text_input('Post_transplant_diabetes')
    Post_transplant_fluid_overload = st.text_input('Post_transplant_fluid_overload')
    Post_transplant_Gastro_intestininal_complications = st.text_input('Post_transplant_Gastro_intestininal_complications')
    Post_transplant_Glomerulonephritis = st.text_input('Post_transplant_Glomerulonephritis')
    Post_transplant_hypertension = st.text_input('Post_transplant_hypertension')
    Post_transplant_Infection = st.text_input('Post_transplant_Infection')
    Post_transplant_malignance = st.text_input('Post_transplant_malignance')
    Post_transplant_Urological_complications = st.text_input('Post_transplant_Urological_complications')
    Post_transplant_Vascular_complications = st.text_input('Post_transplant_Vascular_complications')
    post_water_intake = st.text_input('post_water_intake')
    Pre_transplant_history_of_substance = st.text_input('Pre_transplant_history_of_substance')
    Causes_of_End_Stage_Renal_Disease = st.text_input('Causes_of_End_Stage_Renal_Disease')
    Donor_sex = st.text_input('Donor_sex')
    Number_of_post_transplant_admission = st.text_input('Number_of_post_transplant_admission')
    Post_transplant_non_adherence = st.text_input('Post_transplant_non_adherence')
    Post_transplant_regular_physicale = st.text_input('Post_transplant_regular_physicale')
    Recipients_sex = st.text_input('Recipients_sex')
    
    
    # Code for prediction
    diagnosis = ''
             
    # Creating a button for prediction
    if st.button('Predict graft survival'):
       diagnosis = graft_surv_pred([Blood_Urea_Nitrogen_level, Body_mass_index, Calcium_level,
       Diastolic_blood_pressure, Donor_age, eGFR, Glucose_level,
       Hemoglobin_level, Phosphorus_level, Platelets, Potassium_level,
       Recipients_age, Serum_creatinine_level, Sodium_level,
       Systolic_blood_pressure, Tacrolimus_Modified_Release,
       White_blood_cell_count,
       Blood_Urea_Nitrogen_level_to_Serum_creatinine_level_ratio,
       An_episode_of_acute_rejection, An_episode_of_chronic_rejection,
       An_episode_of_hyperacute_rejection, Donor_to_recipient_relationship,
       Family_history_of_kidney_disease, History_of_abdominal_surge,
       History_of_blood_transfusion, History_of_dialysis_before_transplant,
       History_of_pre_transplant_comorbid,
       Post_transplant_Cardio_vascular_complications,
       Post_transplant_Covid19, Post_transplant_delayed_graft_function,
       Post_transplant_diabetes, Post_transplant_fluid_overload,
       Post_transplant_Gastro_intestininal_complications,
       Post_transplant_Glomerulonephritis, Post_transplant_hypertension,
       Post_transplant_Infection, Post_transplant_malignance,
       Post_transplant_Urological_complications,
       Post_transplant_Vascular_complications, post_water_intake,
       Pre_transplant_history_of_substance, 
       Causes_of_End_Stage_Renal_Disease, 
       Donor_sex, 
       Number_of_post_transplant_admission, 
       Post_transplant_non_adherence, 
       Post_transplant_regular_physicale, 
       Recipients_sex])
        
    st.info(diagnosis)    
    
if __name__ == '__main__':
    main()
    
    