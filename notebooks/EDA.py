#!/usr/bin/env python
# coding: utf-8

<<<<<<< HEAD
# # Python Webapp

# * Date: February 2025
# * Author: Kirk Rose
# * Jupyter Notebook filename: EDA.ipynb
# * Python filename: EDA.py

# In[6]:


##########################################
# Date: February 2025
# Author: Kirk Rose
# Jupyter Notebook filename: EDA.ipynb
# Python filename: EDA.py
##########################################
=======
# In[ ]:


# EDA.py


# In[2]:


import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import scipy.stats as stats
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde


# In[3]:


# Global variables
csv_path = None
df = None
df_loaded = "File not found"
cylinder_fig = None
vehicle_type_fig = None
price_dist_fig = None
top_5_manufacturers_cylinder_fig = None
price_range_fig = None


# In[4]:


def load_data():
    global df
    global csv_path
    global df_loaded

    if df is None:  # Only load if df hasn't been loaded already
        # CSV filename
        csv_filename = 'vehicles_us.csv'

        # Construct the relative path to the CSV file
        script_dir = os.path.dirname(__file__)
        csv_path = os.path.join(script_dir, '..', csv_filename)  # Adjust as needed

        # Print for debugging
        print(f"Constructed CSV path: {csv_path}")

        # Check if the file exists
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df_loaded = "Data loaded successfully"
        else:
            df_loaded = "File not found"
    return df_loaded



# In[5]:


def cleanup_data():
    global df
    # check for duplicates 
    duplicate_rows = df[df.duplicated()]
    num_duplicates = df.duplicated().sum()
    #print(f"Number of duplicate rows: {num_duplicates}")
    print("Duplicates found:",num_duplicates)

    if num_duplicates > 0:
        # Remove duplicates
        df = df.drop_duplicates()

    # Replace NaN with False and non-NaN with True in the 'is_4wd' column
    df['is_4wd'] = df['is_4wd'].notna()

    # Remove all rows with any NaN values
    df = df.dropna()

    # List of columns to capitalize the first letter of each word
    columns_to_capitalize = ['model', 'condition', 'fuel', 'transmission', 'type', 'paint_color']

    # Capitalize the first letter of each word in the specified columns using .loc[]
    for column in columns_to_capitalize:
        df.loc[:, column] = df[column].str.title()

    # Remove rows where price > 100000 or price < 300
    df = df[(df['price'] <= 100000) & (df['price'] >= 300)]

    #Adding column for manufaturer as this will be used in subsequent analyses
    df.loc[:, 'manufacturer'] = df['model'].apply(lambda x: x.split()[0])
    
    
    return


# In[62]:


def analyze_data():
    global cylinder_fig
    global vehicle_type_fig
    global price_dist_fig
    global top_5_manufacturers_cylinder_fig
    global price_range_fig

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Sales by cylinder count >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Create a DataFrame for Plotly
    df_cylinders = df['cylinders'].value_counts().reset_index()
    df_cylinders.columns = ['Cylinders Type', 'Frequency']  # Rename columns for clarity
    
    # Create a Plotly bar plot
    fig = px.bar(df_cylinders,
                 x='Cylinders Type', y='Frequency',
                 title='Cylinders Type Distribution',
                 labels={'Cylinders Type': 'Cylinders Type', 'Frequency': 'Frequency'})
    
    # Assign to Plotly chart variable
    cylinder_fig = fig


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Total Sales by vehicle type >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Show the chart
    #fig.show()

    # Group by type and calculate total sales
    total_sales_by_type = df.groupby('type')['price'].sum()
    
    # Create the bar chart
    fig = px.bar(total_sales_by_type, x=total_sales_by_type.index, y='price',
                title='Total Sales by Vehicle Type',
                labels={'x': 'Vehicle Type', 'y': 'Total Sales'})

    
    # Assign to Plotly chart variable
    vehicle_type_fig = fig




    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Price Distribution Curve >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Create the price distribution plot
    fig = px.histogram(
        df,
        x="price",
        nbins=10,  # Adjust the number of bins as needed
        histfunc="count",  # Use 'count' for the histogram
        marginal="rug",  # Add a rug plot along the x-axis
        title="Price Distribution Curve",
        labels={"price": "Price", "value": "Frequency"},
    )
    
    # Assign to Plotly chart variable
    price_dist_fig = fig




    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Top 5 Manufacturers based on most popular cylinders >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Filter data for cylinders 4, 6, and 8
    filtered_df = df[df['cylinders'].isin([4.0, 6.0, 8.0])]
    
    # Group by manufacturer and cylinder, and count occurrences
    cylinder_counts = filtered_df.groupby(['manufacturer', 'cylinders']).size().unstack(fill_value=0)
    
    # Sum counts for each manufacturer to get total count for sorting
    cylinder_counts['Total'] = cylinder_counts.sum(axis=1)
    
    # Get the top 5 manufacturers based on total count
    top_5_manufacturers = cylinder_counts['Total'].nlargest(5).index
    
    # Filter data for top 5 manufacturers
    top_5_df = cylinder_counts.loc[top_5_manufacturers]
    
    # Convert the DataFrame to long format for Plotly
    top_5_df_long = top_5_df.drop(columns='Total').reset_index().melt(id_vars='manufacturer', var_name='cylinders', value_name='count')
    
    # Plot with Plotly Express
    top_5_manufacturers_cylinder_fig = px.bar(top_5_df_long, 
                                             x='manufacturer', 
                                             y='count', 
                                             color='cylinders', 
                                             title='Top 5 Manufacturers by Cylinder Type',
                                             labels={'count': 'Number of Vehicles', 'manufacturer': 'Manufacturer', 'cylinders': 'Cylinders'},
                                             text='count',
                                             color_discrete_sequence=px.colors.qualitative.Plotly,
                                             category_orders={'cylinders': ['4.0', '6.0', '8.0']}
                                            )
    
    top_5_manufacturers_cylinder_fig.update_layout(barmode='stack', 
                                                   xaxis_title='Manufacturer',
                                                   yaxis_title='Number of Vehicles',
                                                   xaxis_tickangle=-45)


    # number of cars sold by price ranges
    # Define the price bins and labels
    price_bins = [0, 10000, 20000, 30000, 40000, 50000, 60000]
    price_labels = ['0-10000', '10001-20000', '20001-30000', '30001-40000', '40001-50000', '50001-60000']
    
    # Create a new column 'price_range' based on the bins
    df['price_range'] = pd.cut(df['price'], bins=price_bins, labels=price_labels)
    
    # Count the number of cars sold in each price range
    price_range_counts = df['price_range'].value_counts().sort_index()
    
    # Convert the result to a DataFrame for Plotly
    price_range_df = price_range_counts.reset_index()
    price_range_df.columns = ['Price Range', 'Number of Cars Sold']
    
    # Create a bar plot using Plotly
    price_range_fig = px.bar(price_range_df, 
                 x='Price Range', 
                 y='Number of Cars Sold', 
                 title='Number of Cars Sold by Price Range',
                 labels={'Price Range': 'Price Range', 'Number of Cars Sold': 'Number of Cars Sold'},
                 text='Number of Cars Sold',  # Display the number on the bars
                 color_discrete_sequence=['skyblue'])  # Color for bars
    
    # Update layout to rotate x-axis labels and adjust size
    price_range_fig.update_layout(xaxis_tickangle=-45, width=800, height=600)

    return
>>>>>>> origin/main


# In[7]:


<<<<<<< HEAD
# Import libraries
import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
=======
def get_file_path():
    global csv_path
    return csv_path
>>>>>>> origin/main


# In[8]:


<<<<<<< HEAD
# Global variables
utilities_df = None


# ## Data Loading: Functions
=======
def get_df():
    global df
    global df_loaded
    if df is None:
        load_data()  # Load the data if not already loaded
    return


# In[9]:


def get_df_head():
    global df
    return df.head(5)

>>>>>>> origin/main

# In[10]:


<<<<<<< HEAD
# Function to load dataframe
def load_data():
    """Loads CSV file and returns a DataFrame."""
    global utilities_df
    
    csv_filename = 'electric_companies_and rates_2020.csv'
    csv_path = os.path.join(os.getcwd(), csv_filename)  # Get absolute path

    print(f"Constructed CSV path: {csv_path}")  # Debugging output

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.success("Data loaded successfully")
        return df
    else:
        st.error(f"File not found: {csv_path}")
        return pd.DataFrame()  # Return an empty DataFrame instead of None



# ## Data Loading: FrontEnd Controls
=======
def get_df_loaded_message():
    global df_loaded
    return df_loaded


# In[11]:


def get_cylinder():
    global cylinder_fig
    return cylinder_fig


# In[ ]:


def get_vehicle_type():
    global vehicle_type_fig
    return vehicle_type_fig

>>>>>>> origin/main

# In[12]:


<<<<<<< HEAD
# Load data into vehicles_df
def loadDataframe():
    global utilities_df
    
    st.markdown("#### :blue[Attempt to load dataset ... ]")
    
    utilities_df = load_data()


# ## Heading and app overview

# In[14]:


def show_heading_and_overview():
    st.write("Created by: Kirk S. Rose")
    st.write("TripleTen - Data Science Bootcamp Project (February 2025)")
    st.title('U.S. (2020) Electric Utility Companies and Rates: Look-up by Zipcode:', anchor=None)
    
    st.markdown("""
    This app retrieves U.S. Electric Utility Companies and Rates (kWh) for all 50 states. Focused on 2020, it was compiled by NREL using data from ABB, 
    the Velocity Suite and the U.S. Energy Information Administration dataset 861, 
    provides average residential, commercial and industrial electricity rates with 
    likely zip codes for both investor owned utilities (IOU) and non-investor owned utilities. 
    
    **Note:** the files include average rates for each utility (not average rates per zip code), 
    but not the detailed rate structure data found in the OpenEI U.S. Utility Rate Database.
    
    * **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, os, seaborn, plotly
    * **Data source:** [DATA.GOV](https://catalog.data.gov/dataset/u-s-electric-utility-companies-and-rates-look-up-by-zipcode-2020).
    """)

    st.markdown("""Streamlit is an open-source Python library that makes it easy to build and share beautiful, 
    custom web apps for machine learning and data science.
    """)


# ## Data Summary and Initial Checks - Section 1: Functions

# In[16]:


# Function to clean data
def cleanup_data(utilities_df):
    
    # check for duplicates 
    duplicate_rows = utilities_df[utilities_df.duplicated()]
    num_duplicates = utilities_df.duplicated().sum()
    print()
    print("Duplicates found:",num_duplicates)

    # checking for NaN
    missing_counts = utilities_df.isna().sum()
    print()
    print("Missing found:",missing_counts)
    
    # Ensure correct data types
    utilities_df['zip'] = pd.to_numeric(utilities_df['zip'], errors='coerce')
    utilities_df['eiaid'] = pd.to_numeric(utilities_df['eiaid'], errors='coerce')
    utilities_df['comm_rate'] = pd.to_numeric(utilities_df['comm_rate'], errors='coerce')
    utilities_df['ind_rate'] = pd.to_numeric(utilities_df['ind_rate'], errors='coerce')
    utilities_df['res_rate'] = pd.to_numeric(utilities_df['res_rate'], errors='coerce')
    
    utilities_df[['utility_name', 'state', 'service_type', 'ownership']] = utilities_df[['utility_name', 'state', 'service_type', 'ownership']].astype(str)


    assuredDataTypes = """
        utilities_df['zip'] = pd.to_numeric(utilities_df['zip'], errors='coerce')
        utilities_df['eiaid'] = pd.to_numeric(utilities_df['eiaid'], errors='coerce')
        utilities_df['comm_rate'] = pd.to_numeric(utilities_df['comm_rate'], errors='coerce')
        utilities_df['ind_rate'] = pd.to_numeric(utilities_df['ind_rate'], errors='coerce')
        utilities_df['res_rate'] = pd.to_numeric(utilities_df['res_rate'], errors='coerce')
        
        utilities_df[['utility_name', 'state', 'service_type', 'ownership']] = utilities_df[['utility_name', 'state', 'service_type', 'ownership']].astype(str)
    """

    # Select numeric columns
    numeric_cols = utilities_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Check for zero values
    for col in numeric_cols:
        if (utilities_df[col] == 0).any():
            print(f"Column '{col}' contains zero values.")
    
    # Check for NaN values
    for col in numeric_cols:
        if utilities_df[col].isna().any():
            print(f"Column '{col}' contains NaN values.")

    with st.expander("Data Cleaning"):  # Creates the accordion (expander)
        st.markdown("""In this section we focused on validating, cleaning the data, checking for duplicate rows, 
        as weell as ensuring that any missing value are given proper treatment (such as filling based on median, mode, mean)""")
        st.write(f"1. Number of duplicate rows: {num_duplicates}") # display as html
        st.write(f"2. Missing values check: The table below based on the .info() method, shows no missing values for any columns.") # display as html
        st.table(missing_counts) # display as html
        st.write(f"3. Assured data types:")
        st.write({assuredDataTypes})
        st.write(f"4. Post datatype conformity validation: The rows found with 'zero' values for `industrial` and `residential` rates are legimimate")
        st.caption("""Cleaned data to ensure consistency and compliance""")
        
    return


# ## Data Summary and Initial Checks - Section 1: Frontend Control Section 1

# In[18]:


def frontEnd_display_control_1():
    global utilities_df
            
    # title
    st.header(":blue[Data Summary and Initial Checks:]")
    
    num_rows = len(utilities_df)
    st.write(f"Found records:",{num_rows})
    
    # Format numerical columns to avoid comma formatting in zip code
    utilities_df_formatted = utilities_df.copy()
    utilities_df_formatted[['zip','eiaid']] = utilities_df_formatted[['zip','eiaid']].applymap(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
    
    # Display data (handles empty DataFrame)
    if utilities_df.empty:
        st.warning("No data found. Please check the file.")
    else:
        st.write(utilities_df_formatted.sample(n=10))
        with st.expander("Column Explanations"):  # Creates the accordion (expander)
            st.write("""
            *   **zip:** The ZIP code served by the electric utility. This indicates the geographic area where the rates apply. Keep in mind that ZIP codes can sometimes be served by multiple utilities, or a single utility might have different rates in different parts of a ZIP code.
            *   **eiaid:** The unique identifier assigned to the electric utility by the U.S. Energy Information Administration (EIA). This is a standard identifier used for tracking utilities.
            *   **utility_name:** The name of the electric utility company.
            *   **state:** The U.S. state in which the utility operates.
            *   **service_type:** The type of service provided. "Bundled" typically means that the utility provides both electricity generation and distribution. Other service types might exist (e.g., if a utility only distributes electricity but doesn't generate it).
            *   **ownership:** The ownership structure of the utility. "Investor Owned" (IOU) means the utility is a for-profit company owned by shareholders. Other common ownership types include "Publicly Owned" (municipal or government-owned), "Cooperative" (owned by its customers), and "Federal."
            *   **comm_rate:** The average commercial electricity rate in dollars per kilowatt-hour (kWh).
            *   **ind_rate:** The average industrial electricity rate in dollars per kWh.
            *   **res_rate:** The average residential electricity rate in dollars per kWh.
            """)    
            
    # print(utilities_df.head())  # Debugging output (safe even if empty)
    st.caption("Expand/collapse `Column Explanations` section above. It contains general information about the dataset.")
    st.divider()
    
    cleanup_data(utilities_df)
    st.caption("Expand/collapse `Data Cleaning` section above. It details how this dataset was cleaned and validated.")
    st.divider()





# # examine dataframe
# print(utilities_df['zip'].nunique())
# print(utilities_df['eiaid'].nunique())
# print(utilities_df['utility_name'].nunique())
# utilities_df.info()
# 
# num_rows = len(utilities_df)
# st.write(f"Found records:",{num_rows})
# 
# # Format numerical columns to avoid comma formatting in zip code
# utilities_df_formatted = utilities_df.copy()
# utilities_df_formatted[['zip']] = utilities_df_formatted[['zip']].applymap(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
# 
# # Display data (handles empty DataFrame)
# if utilities_df.empty:
#     st.warning("No data found. Please check the file.")
# else:
#     st.write(utilities_df_formatted.head())
#     with st.expander("Column Explanations"):  # Creates the accordion (expander)
#         st.write("""
#         *   **zip:** The ZIP code served by the electric utility. This indicates the geographic area where the rates apply. Keep in mind that ZIP codes can sometimes be served by multiple utilities, or a single utility might have different rates in different parts of a ZIP code.
#         *   **eiaid:** The unique identifier assigned to the electric utility by the U.S. Energy Information Administration (EIA). This is a standard identifier used for tracking utilities.
#         *   **utility_name:** The name of the electric utility company.
#         *   **state:** The U.S. state in which the utility operates.
#         *   **service_type:** The type of service provided. "Bundled" typically means that the utility provides both electricity generation and distribution. Other service types might exist (e.g., if a utility only distributes electricity but doesn't generate it).
#         *   **ownership:** The ownership structure of the utility. "Investor Owned" (IOU) means the utility is a for-profit company owned by shareholders. Other common ownership types include "Publicly Owned" (municipal or government-owned), "Cooperative" (owned by its customers), and "Federal."
#         *   **comm_rate:** The average commercial electricity rate in dollars per kilowatt-hour (kWh).
#         *   **ind_rate:** The average industrial electricity rate in dollars per kWh.
#         *   **res_rate:** The average residential electricity rate in dollars per kWh.
#         """)
# 

# ### Data validation and cleaup

# print(utilities_df.head())  # Debugging output (safe even if empty)
# st.caption("Expand/collapse `Column Explanations` section above. It contains general information about the dataset.")
# st.divider()
# 
# cleanup_data(utilities_df)
# st.caption("Expand/collapse `Data Cleaning` section above. It details how this dataset was cleaned and validated.")
# st.divider()

# ## Utility Rate Finder

# In[ ]:





# # Streamlit UI
# st.header(":green[Utility Rate Finder. Zipcode lookup]")
# 
# # Input for zip code
# zip_code = st.text_input("Enter Zip Code")
# 
# 
# 
# # Button to trigger search
# if st.button('Search'):
#     # Convert the zip code input to an integer for matching
#     zip_code = int(zip_code) if zip_code.isdigit() else None
#     
#     if zip_code:
#         # Filter the dataframe for the entered zip code
#         result = utilities_df[utilities_df['zip'] == zip_code]
#         
#         if not result.empty:
#             # Custom CSS to set the background color for the simulated dialog
#             st.markdown("""
#                 <style>
#                     .custom-container {
#                         background-color: #F7F7DC;
#                         border-radius: 10px;
#                         padding: 20px;
#                         margin-top: 10px;
#                     }
#                     .custom-container h3 {
#                         margin-bottom: 10px;
#                     }
#                     .custom-container p {
#                         margin: 5px 0;
#                     }
#                 </style>
#             """, unsafe_allow_html=True)
#             
#             # Simulating an expander-like dialog with a container
#             st.markdown(f"""
#                 <div class="custom-container">
#                     <h3>Utility Information</h3>
#                     <p><strong>Utility Name:</strong> {result['utility_name'].values[0]}</p>
#                     <p><strong>State:</strong> {result['state'].values[0]}</p>
#                     <p><strong>Service Type:</strong> {result['service_type'].values[0]}</p>
#                     <p><strong>Commercial Rate:</strong> {result['comm_rate'].values[0]:.4f}</p>
#                     <p><strong>Industrial Rate:</strong> {result['ind_rate'].values[0]:.4f}</p>
#                     <p><strong>Residential Rate:</strong> {result['res_rate'].values[0]:.4f}</p>
#                 </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.error("No utility found for the entered zip code.")
#     else:
#         st.error("Please enter a valid zip code.")

# In[ ]:



=======
def get_price_dist():
    global price_dist_fig
    return price_dist_fig
>>>>>>> origin/main


# In[ ]:


<<<<<<< HEAD
def get_zipCode_rate(zip_code):
    result = utilities_df[utilities_df['zip'] == zip_code]
    
    if not result.empty:
        html_output = f"""
            <div class="custom-container">
                <h3>Utility Company Information</h3>
                <p><strong>Utility Name:</strong> {result['utility_name'].values[0]}</p>
                <p><strong>Zip Code:</strong> {result['zip'].values[0]}</p>
                <p><strong>State:</strong> {result['state'].values[0]}</p>
                <p><strong>Service Type:</strong> {result['service_type'].values[0]}</p>
                <p><strong>Commercial Rate:</strong> {result['comm_rate'].values[0]:.4f}</p>
                <p><strong>Industrial Rate:</strong> {result['ind_rate'].values[0]:.4f}</p>
                <p><strong>Residential Rate:</strong> {result['res_rate'].values[0]:.4f}</p>
            </div>
        """
        return html_output
    else:
        return "No utility found for the entered zip code."




# ## Analysis of Categorical Variables - Section 2: Functions

# In[26]:


def operators_per_state(utilities_df):
    # Group by state and count unique utility providers
    utilities_per_state = utilities_df.groupby('state')['utility_name'].nunique().reset_index()
    utilities_per_state.columns = ['state', 'num_utilities']
    
    # Display the grouped data
    #st.write("### Number of Utility Providers per State")
    #st.write(utilities_per_state)
    
    # Create a bar plot
    st.write("#### Number of Utility Providers per State")
    plt.figure(figsize=(18, 7))
    sns.barplot(x='state', y='num_utilities', data=utilities_per_state, palette='viridis')
    plt.title('Number of Utility Providers per State')
    plt.xlabel('State')
    plt.ylabel('Number of Utility Providers')
    plt.xticks(rotation=45)
    st.pyplot(plt)


# In[27]:


def plot_stacked_histogram(df, categorical_columns):
    """
    Creates stacked histograms for given categorical columns.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        categorical_columns (list): List of categorical column names.
    
    Displays:
        Stacked histogram plots for each categorical column.
    """
    global utilities_df
    
    for col in categorical_columns:
        # Frequency count
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, 'count']
        
        # Create stacked histogram
        fig = px.bar(counts, x=col, y='count', color=col, title=f'Distribution of {col}', text='count')
        fig.update_traces(texttemplate='%{text}', textposition='outside')

        # Display using Streamlit
        st.plotly_chart(fig)


# In[28]:


def national_averages(utilities_df):
    # Calculate national averages
    national_avg_comm = utilities_df['comm_rate'].mean()
    national_avg_ind = utilities_df['ind_rate'].mean()
    national_avg_res = utilities_df['res_rate'].mean()
    
    # Group by state and calculate average rates
    state_avg_rates = utilities_df.groupby('state')[['comm_rate', 'ind_rate', 'res_rate']].mean()
    
    # Find states with the highest and lowest overall average rate
    state_avg_rates['overall_avg_rate'] = state_avg_rates.mean(axis=1)
    
    highest_avg_state = state_avg_rates['overall_avg_rate'].idxmax()
    lowest_avg_state = state_avg_rates['overall_avg_rate'].idxmin()
    
    # Get the highest and lowest overall average rates
    highest_avg_rate = state_avg_rates.loc[highest_avg_state, 'overall_avg_rate']
    lowest_avg_rate = state_avg_rates.loc[lowest_avg_state, 'overall_avg_rate']
    
    # Calculate the percentage difference
    percent_difference = ((highest_avg_rate - lowest_avg_rate) / lowest_avg_rate) * 100

    # Filter for rows where comm_rate > res_rate
    states_with_higher_comm_rate = utilities_df[utilities_df['comm_rate'] > utilities_df['res_rate']]['state'].unique()
    
    # Filter states where the average commercial rate is higher than the average residential rate
    avg_rates_by_state = utilities_df.groupby('state')[['comm_rate', 'res_rate']].mean() # Group by state and calculate average rates
    states_with_higher_avg_comm_rate = avg_rates_by_state[avg_rates_by_state['comm_rate'] > avg_rates_by_state['res_rate']].index
    
    # Print results
    st.markdown(f"""
    - **National Average Commercial Rate:** {national_avg_comm:.6f}
    - **National Average Industrial Rate:** {national_avg_ind:.6f}
    - **National Average Residential Rate:** {national_avg_res:.6f}
    - 
    - **State with the Highest Average Rate:** {highest_avg_state} ({highest_avg_rate:.6f})
    - **State with the Lowest Average Rate:** {lowest_avg_state} ({lowest_avg_rate:.6f})
    - **Percentage Difference (Highest vs Lowest):** {percent_difference:.2f}%
    - 
    - **States with some providers having a higher Commercial Rate than Residential Rate:** {", ".join(states_with_higher_comm_rate)}
    - **States where Average Commercial Rate is higher than Average Residential Rate:** {", ".join(states_with_higher_avg_comm_rate)}
    """)
    st.caption(f"OBSERVATIONS:")
    st.caption(f"We found that customers in Hawaii pay more than 6 times {percent_difference:.2f}% the average rate of those living in Nevada.")
    st.caption(f"Despite the national average for Residential Rates being higher than Commercial Rates, there are 4 states where Average Commercial Rates are higher.")


# In[29]:


def top_20_operator_counties(utilities_df):
    """
    Finds and displays the top 20 utility companies that service the most ZIP codes as a bar chart.
    Includes the state in the output.

    Args:
        utilities_df (pd.DataFrame): The utilities dataset containing 'utility_name', 'state', and 'zip' columns.

    Returns:
        pd.DataFrame: A DataFrame of the top 20 utility companies with their ZIP code count and state.
    """
    top_20_utilities = (
        utilities_df.groupby(['utility_name', 'state'])['zip']
        .nunique()  # Count unique ZIP codes per utility and state
        .reset_index(name='zip_count')
        .sort_values(by='zip_count', ascending=False)  # Sort in descending order
        .head(20)  # Select the top 20
    )

    # Display results in Streamlit
    st.write("Top 20 Utility Companies by ZIP Code Coverage (Including State)")
    
    # Create a bar chart using Plotly
    fig = px.bar(
        top_20_utilities,
        x='zip_count',
        y='utility_name',
        color='state',  # Color by state
        orientation='h',  # Horizontal bar chart
        title="Top 20 Utility Companies by ZIP Code Coverage",
        labels={'zip_count': 'Number of ZIP Codes', 'utility_name': 'Utility Company', 'state': 'State'},
        height=700
    )
    
    st.plotly_chart(fig)  # Display plot in Streamlit

    return top_20_utilities


# ## Analysis of Categorical Variables - Section 2: Frontend Control Section 2

# In[31]:


def frontEnd_display_control_2():
    global utilities_df
    
    st.header(":blue[Analysis of Categorical Variables:]")
    
    # Number of Utility Providers per State
    operators_per_state(utilities_df)
    
    # Ownership
    st.divider()
    st.subheader("Ownership:")
    st.write("`utilities_df['ownership'].unique()`")
    st.write(utilities_df['ownership'].unique())
    st.caption("We found that all operators are private companies")
    st.divider()

    # Define categorical columns
    categorical_columns = ['state', 'ownership', 'service_type']
    
    # Call the function
    plot_stacked_histogram(utilities_df, categorical_columns)
    
    st.write(f"Unique zip codes: {utilities_df['zip'].nunique()}")
    st.caption("OBSERVATION: The number of unique zip codes being less that the number of records, indicates that some zip codes may be serviced my multiple operators.")
    
    # National Average Rates
    st.divider()
    st.subheader("National Average Rates:")
    national_averages(utilities_df)
    
    # Utility Analysis Dashboard
    st.divider()
    st.subheader("Utility Analysis Dashboard")
    top_20 = top_20_operator_counties(utilities_df)
    st.caption("OBSERVATION: Each of the Top 20 Utility Providers serviced over 500 Zip codes.")
=======
def get_top_5_manufacturers_cylinder():
    global top_5_manufacturers_cylinder_fig
    return top_5_manufacturers_cylinder_fig
>>>>>>> origin/main


# In[ ]:


<<<<<<< HEAD



# In[ ]:





# ## Analysis of Numerical Variables - Section 3: Functions

# In[33]:


def summary_statistics_by_group(utilities_df, rate_column, group_by_column):
    """
    Summary statistics for a given rate column grouped by a specified categorical column.
    
    Parameters:
    utilities_df (DataFrame): The utilities dataset
    rate_column (str): The column representing the rate to analyze (e.g., 'comm_rate', 'ind_rate', 'res_rate')
    group_by_column (str): The categorical column to group by (e.g., 'ownership', 'state')
    """
    summary_stats = utilities_df.groupby(group_by_column)[rate_column].describe()

    # Display results in Streamlit
    st.write(f"#### Summary Statistics for `{rate_column}` grouped by `{group_by_column}`")
    st.dataframe(summary_stats)


# In[34]:


def create_rate_comparison_plot(utilities_df):
    """
    Creates an interactive scatter plot comparing commercial and residential utility rates.

    Parameters:
    data (pd.DataFrame): DataFrame containing utility rate data
    """
    # Create an interactive scatter plot
    fig = px.scatter(
        utilities_df, 
        x="comm_rate", 
        y="res_rate", 
        color="comm_rate",  # Color by commercial rate
        color_continuous_scale="greens",  # Green for commercial rates
        title="Commercial vs Residential Utility Rates",
        labels={"comm_rate": "Commercial Rate ($/kWh)", "res_rate": "Residential Rate ($/kWh)"},
        hover_data=["utility_name", "state", "service_type"],  # Show extra info on hover
    )

    # Add an equality reference line
    max_rate = max(utilities_df["comm_rate"].max(), utilities_df["res_rate"].max())
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=max_rate, y1=max_rate,
        line=dict(color="red", dash="dash"),
        name="Equal Rates Line"
    )

    # Customize layout
    fig.update_traces(marker=dict(size=8, opacity=0.6))  # Adjust marker size & transparency
    fig.update_layout(coloraxis_colorbar=dict(title="Commercial Rate ($/kWh)"))
    fig.update_xaxes(title="Commercial Rate ($/kWh)")
    fig.update_yaxes(title="Residential Rate ($/kWh)")

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# st.header(":blue[Analysis of Numerical Variables:]")

# ## Analysis of Numerical Variables - Section 3: Frontend Control Section 3

# In[37]:


def frontEnd_display_control_3():
    global utilities_df
    
    st.header(":blue[Analysis of Numerical Variables:]")
    
    # Generate summary statistics for ownership types
    summary_statistics_by_group(utilities_df, 'comm_rate', 'ownership')
    summary_statistics_by_group(utilities_df, 'ind_rate', 'ownership')
    summary_statistics_by_group(utilities_df, 'res_rate', 'ownership')
    
    # Generate state-level statistics
    summary_statistics_by_group(utilities_df, 'comm_rate', 'state')
    summary_statistics_by_group(utilities_df, 'ind_rate', 'state')
    summary_statistics_by_group(utilities_df, 'res_rate', 'state')

    # get Max & Min comparison
    # Extract the max value from the table
    max_rate = utilities_df[['comm_rate', 'res_rate', 'ind_rate']].max().max()
    
    # Extract the min value from the table
    min_rate = utilities_df[['comm_rate', 'res_rate', 'ind_rate']].min().min()
    
    # Display in Streamlit caption
    st.caption(f"OBSERVATION: The highest rate (Commercial or Residential) in the country is {max_rate:.6f}, while the lowest rate is {min_rate:.6f}. A rate of '0' doesn't mean free, but indicates that one of ['comm_rate','ind_rate','res_rate'] is rated at zero for that service area.")

    # Utility Rate Analysis / Comparison
    st.subheader('Utility Rate Analysis')
    st.write("Expand the plot to see better details...")
    create_rate_comparison_plot(utilities_df)


# ## Analysis and Summary - Section 4: Functions and Frontend Control Section 4

# In[39]:


# Analysis and Summary [called by function frontEnd_control_4]

def frontEnd_display_control_4():
    global utilities_df
    
    st.header(":blue[Analysis and Summary:]")
    st.markdown("""
        The U.S. was serviced by 143 Electricty providers as late as 2020. The providers were all private companies. 
        Some states like Georgia and Delaware had only one service provider, while Pennslyvannia and Wisconsin and 11 and 12 respectively.
        
        Of the visualizations used in the analysis, the scatter plot providea great insight:
    
        **Rate Distribution Pattern**
        The plot shows a clear positive correlation between commercial and residential rates, with most data points clustered along a consistent trend. 
        This pattern indicates that utilities tend to maintain proportional relationships between their commercial and residential rates, 
        likely reflecting similar underlying cost structures and market conditions across service areas.
        
        **Residential Premium**
        One of the most striking features is that approximately `87.7%` of utilities charge higher rates to residential customers than commercial customers. 
        This is visualized by the large number of points above the red dashed `"Equal Rates Line."` 
        The mean difference of `0.0243 $/kWh` represents a significant premium that residential customers pay compared to commercial users.
        
        This residential premium can be explained by several factors:
        1. Infrastructure Costs: Residential service requires more extensive distribution networks and individual connections, increasing per-unit delivery costs.
        2. Usage Patterns: Residential consumption tends to be more variable and peak-dependent, requiring utilities to maintain additional capacity.
        3. Administrative Overhead: Managing many small residential accounts typically costs more per kilowatt-hour than fewer, larger commercial accounts.
        4. Commercial and Industrial consumption per customer is usually much higher than residential, and may attract cheaper rates, 
        to decentivise commerial customers from investing in their own power plants.
        
        **Rate Clustering**
        The visualization shows interesting clustering patterns:
        - Most rates fall between `0.05 and 0.25 $/kWh` for both customer classes.
        - There's greater variation in residential rates than commercial rates, shown by the vertical spread of points.
        - The density of points is highest in the `0.10 - 0.15 $/kWh` range for commercial rates.
        
        **Outliers and Variations**
        The scatter plot reveals several noteworthy outliers:
        - Some utilities show rates above `0.30 $/kWh` for both customer classes.
        - A small number of points fall below the equal rates line, representing unusual cases where commercial rates exceed residential rates.
        - The spread increases at higher rate levels, suggesting more variability in pricing strategies among higher-cost utilities.
    
        """)


# In[40]:


# utilities_df.head()


# if __name__ == '__main__':
#     # This block only runs when EDA.py is executed directly.
#     display_eda()
=======
def get_price_range_fig():
    global price_range_fig
    return price_range_fig

>>>>>>> origin/main
