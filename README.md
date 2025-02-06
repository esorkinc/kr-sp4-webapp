# kr-sp4-webapp


# my-webapp
Website based on Software Development Tools


# Project Overview
This project provided me with an opprtunity  to practice some of the tools and coding skills associated with Sprint 4. I was able to present anaylsis of a csv file in a Python (Streamlit) web application, exported the version to Github, then published to the public internet using Render.

## Data source and analysis:
- Based of the flexibility provided in the instructions, I chose to do something in an area of deep interest. I performed analyses of 2020 electricty rates in the USA, and the distribution of utility providers. 
- File Used: electric_companies_and_rates_2020.csv


## Development and Testing
1. These 7 files contain everything needed for the application to run successfully.
  
    |-- README.md
    |-- requirements.txt
    |-- app.py
    |-- electric_companies_and_rates_2020.csv
    |__ notebooks
        |__ EDA.ipynb
        |__ EDA.py
    |__ .streamlit
        |__ config.toml
   
3. Create local Github repository, ensuring license that is selected will make repo accessible to the public
4. Mirror the Github repo on local machine.
5. Use use jupyter to create notebook for app development.
    - As this application will be used in a browser, Plotly was selected over Matplotlib to render the graphs to the screen
    - convert notebook files to python native files with command. Example: `jupyter nbconvert --to python <EDA.ipynb>`. Repeat this for app.ipybn as well.
    - Test on local computer. From commant line run: `streamlit run app.py`. A new browser will be launched to display what you have accomplished.
6. Create requirements.txt and add libraries. This file will ensure that only specific Python libraries are used, optimizing the application's performance.
7. Convert the notebook to a .py file using jupyter nbconvert command. This will place the command in a flat Python file which can then be imported into the app.py file in order to access its functions.
```bash
jupyter nbconvert -- to script notebooks/EDA.ipynb
```
9. Create the streamlit configuration file which will tell the server which port to use to publish the application. If not specified the default is port 10000

## Testing on local machine
Publish site with command below. Open a web browser and enter the following url: localhost:10000
```bash
streamlit run app.py 
```

## Update Github main repo
With the application having worked locally, this creates the confidence that it should also work in a remote environment. Push the files to Guthub.
Create a repository of github server with similar name to the application on your local machine. Push the files from your local machine to update the Github repository

## Deploy to Render

1. Login with existing (or create) account
2. Connect Render to your Github repository
3. From Render, suite of services, locate "Web Services" and click "New Web Service" button
    - Select the repository that contains the python streamlit project and "Connect"
4. Configure the new Render service
    - In lowercase, enter a **Name** for the service
    - Enter "Python3" for the **Environment**
    - Select a **Region** of preference
    - For **Branch** select main (default)
    - In the **Build Command** text input, enter `pip install streamlit & pip install -r requirements.txt`
    - For **Start Command** enter `streamlit run app.py`
5. Click "Deploy" button and wait on Render to complete the build process
6. When build is completed, use **url** link nelow the name of the application to launch application.

```bash
pip install streamlit & pip install -r requirements.txt
```

```bash
streamlit run app.py 
```


## Launching the app
Open a web browser and enter the following url: [Software Development Tools Project](https://kr-sp4-webapp.onrender.com)


## License
MIT License was selected for Github repo as this allows any person that obatained a copy of the software to use without restriction. That is the person is allowed to use, copy, modify publish and distribute the software.



# Stundent Comment V1:
app worked only with the default port 8501 for local testing.