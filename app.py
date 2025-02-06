#!/usr/bin/env python
# coding: utf-8

# In[4]:


##########################################
# Date: February 2025
# Author: Kirk Rose
# Python filename: app.py
##########################################


# In[5]:


#import libraries
import streamlit as st

# Import your EDA module
import notebooks.EDA as eda

# create sidebar
st.sidebar.write('')

####################
# MAIN PANEL
####################

eda. show_heading_and_overview()

# Load dataframe
eda.loadDataframe()

eda.frontEnd_display_control_1()

eda.frontEnd_display_control_2()

# Analysis of Numerical Variables
eda.frontEnd_display_control_3()

# Analysis and Summary
eda.frontEnd_display_control_4() 

# Thank you
st.write(":blue[Hope you enjoyed reading!]")



####################
# SIDEBAR
####################

# Add a header to the sidebar
st.sidebar.title(':green[Utility Rate Finder]')

# Add a text input in the sidebar
zip_code = st.sidebar.text_input("Enter Zip Code")

# Add a search button in the sidebar
search_button = st.sidebar.button("Search")

# Optionally, you can also add other widgets like a button or text display
if search_button:
    zip_code = int(zip_code) if zip_code.isdigit() else None
    if zip_code:
        html_result = eda.get_zipCode_rate(zip_code)
        # st.sidebar.write(f"Zip Code entered: {zip_code}")
        st.sidebar.markdown(html_result, unsafe_allow_html=True)
    else:
        st.sidebar.error("Please enter a valid zip code.")




# In[ ]:





# In[ ]:




