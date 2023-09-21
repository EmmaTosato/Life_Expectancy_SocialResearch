# Importing all the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
import random
import scipy.stats as stats
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix
import streamlit as st
import statsmodels.api as sm
from plotly.subplots import make_subplots


# Style and Layout
plt.style.use('default')

###-----------------------------------------------------------TITLE AND CONTEXT---------------------------------------------------------------###
st.title('Life Expectancy: what does it depend on?  ')
st.write('Statistical Analysis on factors influencing Life Expectancy.')
st.text("\n\n")
st.text("\n\n")
st.header('The dataset')
st.subheader('What is Life Expectancy?')
st.write('''
The term “life expectancy” refers to the number of years a person can expect to live. 
By definition, life expectancy is based on an estimate of the average age members of a particular population group will be when they die.''')
st.subheader('Content')
st.write('''
The dataset focus on immunization factors, mortality factors, economic factors, social factors and other health related factors.
         
The Global Health Observatory (GHO) data repository under World Health Organization (WHO) keeps track of the health status as well as many other related factors for all countries.\n
So, the data-set related to life expectancy and health factors for 193 countries has been collected from the same WHO data repository website; its corresponding economic data was collected from United Nation website. 
Among all categories of factors, only those critical factors were chosen which are more representative. 
         
It has been observed that in the past 15 years , there has been a huge development in health sector resulting in improvement of human mortality rates especially in the developing nations in comparison to the past 30 years. Therefore, in this project we have considered data from year 2000-2015 for 193 countries for further analysis. 
''')

st.text("\n\n")
st.header('Research\'s questions and methods')
st.subheader('Question')
st.write('''
The main question on which we can concentrate on, is how different socioeconomic factors, immunization-related factors, mortality factors and economic factors are related and influence life expectancy (our target variable).
        
We will then go on to view some factors individually in order to answer some questions such as:
- How Adult mortality rates affect life expectancy?
- Does Life Expectancy have positive or negative correlation with eating habits, lifestyle, exercise, smoking, drinking alcohol etc.
- What is the impact of schooling on the lifespan of humans?
- Does Life Expectancy have a positive or negative relationship with drinking alcohol?
- Do densely populated countries tend to have lower life expectancy?
- What is the impact of Immunization coverage on life Expectancy?
- etc...
         
''')

st.subheader('Method')
st.write(''' 
We will study this dataset through a **Quantitative Analysis**, since the dataset contains for the most numerical variables. Thus we can reply to quantitative answers, numerical changes in variables and analyze the some phenomena related to life expectancy.
As said before, the final aim is to assess relationship between various factors and life expectancy. 

We will perform:
- **statistical analyses** such as correlation analysis and hypothesis testing.
- **machine learning models** (regressions models), to understand whether it is indeed possible to predict life expectancy through the factors available to us.
- **dimensionality reduction**. 
''')

st.text("\n\n")
st.text("\n\n") 
st.text("\n\n") 

###---------------------------------------------------- DATA CLEANING  ---------------------------------------------------###
st.header('Data Cleaning')
# From the csv file to the datframe
raw_df = pd.read_csv('/Users/emmatosato/Documents/UNI/Magistrale/Social_Research/Life Expectancy Data.csv')
df= raw_df.copy()

# RENAMING #
# Lowering the characters  
orig_cols = list(df.columns)
new_cols = []
for col in orig_cols:
    new_cols.append(col.strip().replace('  ', ' ').replace(' ', '_').lower())
df.columns = new_cols

# Renaming an incorrect label
df.rename(columns={'thinness_1-19_years':'thinness_10-19_years'}, inplace=True)

st.text("\n\n")


# COLUMNS MEANING #
# QUICK EXPLORATION #
st.subheader('The dataset')
st.write('In order to understand briefly how the dataset is organized, i show below the first 5 rows.')
df_example = df.head(5)
st.dataframe(df_example)

st.text("\n\n")

st.subheader('Columns meaning')
with st.expander("Expand for the specific content of each column and the type of each variable"):
    st.write('''
- country (Nominal) - the country in which the indicators are from (i.e. United States of America or Congo)
- year (Ordinal) - the calendar year the indicators are from (ranging from 2000 to 2015)
- status (Nominal) - whether a country is considered to be 'Developing' or 'Developed' by WHO standards
- life_expectancy (Ratio) - the life expectancy of people in years for a particular country and year
- adult_mortality (Ratio) - the adult mortality rate per 1000 population (i.e. number of people dying between 15 and 60 years per 1000 population); if the rate is 263 then that means 263 people will die out of 1000 between the ages of 15 and 60; another way to think of this is - infant_deaths (Ratio) - number of infant deaths per 1000 population; similar to above, but for infants
- alcohol (Ratio) - a country's alcohol consumption rate measured as liters of pure alcohol consumption per capita
- percentage_expenditure (Ratio) - expenditure on health as a percentage of Gross Domestic Product (gdp)
- hepatitis_b (Ratio) - number of 1 year olds with Hepatitis B immunization over all 1 year olds in population
- measles (Ratio) - number of reported Measles cases per 1000 population
- bmi (Interval/Ordinal) - average Body Mass Index (BMI) of a country's total population
- under-five_deaths (Ratio) - number of people under the age of five deaths per 1000 population
- polio (Ratio) - number of 1 year olds with Polio immunization over the number of all 1 year olds in population
- total_expenditure (Ratio) - government expenditure on health as a percentage of total government expenditure
- diphtheria (Ratio) - Diphtheria tetanus toxoid and pertussis (DTP3) immunization rate of 1 year olds
- hiv/aids (Ratio) - deaths per 1000 live births caused by HIV/AIDS for people under 5; number of people under 5 who die due to HIV/AIDS per 1000 births
- gdp (Ratio) 
    - Gross Domestic Product per capita
    - the standard measure of the value added created through the production of goods and services in a country during a certain period
- population (Ratio) - population of a country
- thinness_1-19_years (Ratio) - rate of thinness among people aged 10-19 (Note: variable should be renamed to thinness_10-19_years to more accurately represent the variable)
- thinness_5-9_years (Ratio) - rate of thinness among people aged 5-9
- income_composition_of_resources (Ratio) 
    - Human Development Index in terms of income composition of resources (index ranging from 0 to 1) 
    - ICOR measures how good a country is at utilizing its resources.
    - The HDI is a summary composite measure of a country's average achievements in three basic aspects of human development: health, knowledge and standard of living. 
- schooling (Ratio) - average number of years of schooling of a population
''')

st.text("\n\n")

# NO SENSE VALUE HANDLING
st.subheader('No sense values handling')
st.write('Descriptive statistics about the dataset:')

# Descriptive statistics about the dataset
df_describe = df.describe()
st.dataframe(df_describe) 
st.write('''Things that may not make sense from above:
Things that may not make sense from above:

- Adult mortality of 1? This is likely an error in measurement, but what values make sense here? May need to change to null if under a certain threshold.
- Infant deaths as low as 0 per 1000? That just isn't plausible - I'm deeming those values to actually be null. Also on the other end 1800 is likely an outlier, but it is possible in a country with very high birthrates and perhaps a not very high population total - this can be dealt with later.
- BMI of 1 and 87.3? A BMI of 15 or lower is seriously underweight and a BMI of 40 or higher is morbidly obese, therefore a large number of these measurements just seem unrealistic.
- Under Five Deaths, similar to infant deaths just isn't likely (perhaps even impossible) to have values at zero.
- GDP per capita as low as 1.68 (USD) possible? Perhaps this low values are outliers.
- Population of 34 for an entire country? Impossibile.
''')   

st.write('''
There are a few of the above that could simply be outliers, but there are some that almost certainly have to be errors of some sort. Of the above variables, changes to **null value** (i will handle them later) will be made for the following since these numbers don't make any sense:
1. Adult mortality rates lower than the 5th percentile
2. Infant deaths of 0 
3. BMI less than 10 and greater than 50
4. Under Five deaths of 0
''')

mort_5_percentile = np.percentile(df['adult_mortality'].dropna(), 5)
df.adult_mortality = df.apply(lambda x: np.nan if x.adult_mortality < mort_5_percentile else x.adult_mortality, axis=1)
df.infant_deaths = df.infant_deaths.replace(0, np.nan)
df.bmi = df.apply(lambda x: np.nan if (x.bmi < 10 or x.bmi > 50) else x.bmi, axis=1)
df['under-five_deaths'] = df['under-five_deaths'].replace(0, np.nan)

st.text("\n\n")

# NULL VALUES HANDLING #
st.subheader('Null values handling')

# Checking for Null Values
temp = df.isnull().sum()
with st.expander("Expand for checking how many null values are in the dataset"):
    st.table(temp)

# Dropping a column
st.write('Since half of the BMI variable\'s values are null, it\'s better to remove this variable.')
df.drop(columns='bmi', inplace=True)

st.write('''
There are a lot of columns containing null values, since this is time series data assorted by country, the best course of action would be to interpolate the data by country. 
However, when attempting to interpolate by country it doesn't fill in any values as the countries' data for all the null values are null for each year. 
Therefore imputation before by country, and after year, may be the best possible method here.
''') 

# Interpolating with the mean
# By country
imputed_data = []
for country in list(df.country.unique()):
    country_data = df[df.country == country].copy()
    for col in list(country_data.columns)[3:]:
        country_data[col] = country_data[col].fillna(country_data[col].dropna().mean()).copy()
    imputed_data.append(country_data)
df = pd.concat(imputed_data).copy()

# By Year
imputed_data = []
for year in list(df.year.unique()):
    year_data = df[df.year == year].copy()
    for col in list(year_data.columns)[3:]:
        year_data[col] = year_data[col].fillna(year_data[col].dropna().mean()).copy()
    imputed_data.append(year_data)
df = pd.concat(imputed_data).copy()

st.text("\n\n")

# OUTLIERS HANDLING #
# Plotting 
st.subheader('Outliers handling')
st.write('First a boxplot and histogram will be created for each continuous variable in order to visually see if outliers exist.')

cont_vars = list(df.columns)[3:]
def outliers_visual(data):
    fig, axes = plt.subplots(nrows=9, ncols=4, figsize=(15, 40))
    i = 0
    for col in cont_vars:
        ax_box = axes[i // 4, i % 4]
        ax_hist = axes[(i + 1) // 4, (i + 1) % 4]
        ax_box.boxplot(data[col])
        ax_box.set_title(col)
        ax_hist.hist(data[col], color = '#33ADA4')
        ax_hist.set_title(col)
        i += 2
    plt.tight_layout()
    st.pyplot(fig)
outliers_visual(df)

# Dealing
st.write('''
Since each variable has a unique amount of outliers, the best route to take is probably *winsorizing (limiting)* the values for each variable on its own until no outliers remain.

***What does it mean to winsorize data?*** \n
For example, a 90% winsorization sets all observations greater than the 95th percentile equal to the value at the 95th percentile and all observations less than the 5th percentile equal to the value at the 5th percentile. 
In effect, to winsorize data means to change extreme values in a dataset to less extreme values
''')

# Winsorizing
def test_wins(col, lower_limit=0, upper_limit=0, show_plot=True):
    wins_data = winsorize(df[col], limits=(lower_limit, upper_limit))
    wins_dict[col] = wins_data
    if show_plot == True:
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.boxplot(df[col])
        plt.title('original {}'.format(col))
        plt.subplot(122)
        plt.boxplot(wins_data)
        plt.title('wins=({},{}) {}'.format(lower_limit, upper_limit, col))
        plt.show()

wins_dict = {}
test_wins(cont_vars[0], lower_limit=.01, show_plot=False) 
test_wins(cont_vars[1], upper_limit=.04, show_plot=False)
test_wins(cont_vars[2], upper_limit=.05, show_plot=False)
test_wins(cont_vars[3], upper_limit=.0025, show_plot=False)
test_wins(cont_vars[4], upper_limit=.135, show_plot=False)
test_wins(cont_vars[5], lower_limit=.1, show_plot=False)
test_wins(cont_vars[6], upper_limit=.19, show_plot=False)
test_wins(cont_vars[7], upper_limit=.05, show_plot=False)
test_wins(cont_vars[8], lower_limit=.1, show_plot=False)
test_wins(cont_vars[9], upper_limit=.02, show_plot=False)
test_wins(cont_vars[10], lower_limit=.105, show_plot=False)
test_wins(cont_vars[11], upper_limit=.185, show_plot=False)
test_wins(cont_vars[12], upper_limit=.105, show_plot=False)
test_wins(cont_vars[13], upper_limit=.07, show_plot=False)
test_wins(cont_vars[14], upper_limit=.035, show_plot=False)
test_wins(cont_vars[15], upper_limit=.035, show_plot=False)
test_wins(cont_vars[16], lower_limit=.05, show_plot=False)
test_wins(cont_vars[17], lower_limit=.025, upper_limit=.005, show_plot=False)

# New dataset with the outliers removed
st.write('''
All the variables have now been winsorized as little as possible in order to keep as much data intact as possible while still being able to eliminate the outliers. 
However, in some cases the distributions are still disproportionate. With a better knowledge of this statistical method, the winsorization could be more effective, 
but for the level at which i am conducting this study, i prefer to mantein the dataset as true to the original as possible.\n
         
Thus keep in mind that in several cases the data are not very attendable.
''')
wins_df = df.iloc[:, 0:3]
for col in cont_vars:
    wins_df[col] = wins_dict[col]


# Sorting by index
wins_df.sort_index(inplace= True)

# All Distributions with winsorized data
with st.expander('We can see the distribution of our features after the winsorization.'):
    cont_vars = list(wins_df.columns)[3:]
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(15, 20))
    for i, col in enumerate(cont_vars, 1):
        ax = axes[(i - 1) // 4, (i - 1) % 4]
        ax.hist(wins_df[col], color='#33ADA4')
        ax.set_title(col)
    fig.tight_layout()
    st.pyplot(fig)
st.text('\n\n')
st.text('\n\n')

# VISUALIZING THE DATASETS
st.markdown('#### Exploring the dataset')
st.write("Select an option (or both) if you want to explore the dataset:")
raw_option = st.checkbox("Raw dataset (before the cleaning)")
clean_option = st.checkbox("Cleaned and with no outliers dataset")

if raw_option:
    st.write('RAW dataset')
    st.dataframe(raw_df)
st.text("")
if clean_option:
    st.write('CLEAN dataset')
    st.dataframe(wins_df)

st.text("\n\n")
st.text("\n\n") 
st.text("\n\n") 


###---------------------------------------------------------- DATA EXPORATION------------------------------------------------------------###
st.header('Data Exploration')

# UNIVARIATE ANALYSIS # 
st.subheader('Univariate Analysis')

# Descriptive statistics #
st.markdown('#### Descriptive statistics')

# Descriptive statistics of continuous variables.
st.write('Descriptive statistics of continuous variables:')
cont_ds = wins_df.describe()
st.dataframe(cont_ds)

# Descriptive statistics of categorical variables.
st.write('Descriptive statistics of categorical variables:')
discrete_ds = wins_df.describe(include='O')
st.dataframe(discrete_ds)
st.text("\n\n")
st.text("\n\n")


# Life expectancy distribution
st.markdown('#### Life expectancy distribution')
fig =  px.histogram(wins_df ,x='life_expectancy', color_discrete_sequence= ['#33ADA4'])
st.plotly_chart(fig)
st.text("\n")


# Developed vs developing countries counts 
st.markdown('#### Developed vs developing countries counts')
st.write('''
This two graphs, though simple, are important. 
They display that the majority of our data comes from countries listed as 'Developing'.
It is likely that any model used will more accurately depict results for 'Developing' countries over 'Developed' countries as the majority of the data lies within countries that are 'Developing' rather than 'Developed'.''')
tab1, tab2 = st.tabs(["Pie chart", "Bar chart"])

with tab1:
    fig = px.pie(wins_df, values=wins_df["status"].value_counts(), names=wins_df["status"].value_counts().index, color=wins_df["status"].value_counts().index,
             color_discrete_map={'Developing': '#33ADA4', 'Developed': '#F87188'})
    fig.update_layout(autosize=False, width=500, height=500)
    st.plotly_chart(fig)

with tab2:
     # Count the status occurrences
    status_counts = wins_df['status'].value_counts()

    # Create the countplot figure using Plotly
    fig = go.Figure(data=[
        go.Bar(
            y=status_counts.index,
            x=status_counts.values,
            orientation='h',
            marker=dict(
                color=['#33ADA4', '#F87188']
            )
        )
    ])

    # Customize the figure layout
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1000,
            gridcolor='lightgray',
            showticklabels=True
        ),
        yaxis=dict(
            gridcolor='lightgray',
            showticklabels=True
        ),
        showlegend=False
    )
    st.plotly_chart(fig)


st.text("\n\n")
st.text("\n\n")


# CHLOROPLETH MAP
st.subheader('Chloropleth map')
st.write('Chloropleth maps or graphs can be a very good way to plot the data points on a map. It provides enhanced visuals and helps us to know about the data more easily.')

# List of attributes 
features = ['status', 'life_expectancy', 'adult_mortality', 'infant_deaths',
             'alcohol', 'percentage_expenditure', 'hepatitis_b', 'measles', 'under-five_deaths',
             'polio', 'total_expenditure', 'diphtheria', 'hiv/aids', 'gdp', 'population',
             'thinness_10-19_years', 'thinness_5-9_years', 'income_composition_of_resources', 'schooling']

# Select the feature 
selected_feature = st.selectbox('Select the feature whose distribution you want to see on the map:', features, index = 1)

# Create the choropleth map with animation
fig = px.choropleth(
    wins_df.sort_values(by='year'),
    locations='country',
    locationmode='country names',
    color=selected_feature,
    hover_name='country',
    animation_frame='year',  
    color_continuous_scale='Tealrose',  
    color_discrete_map={'Developing': '#33ADA4', 'Developed': '#F87188'},
    projection='natural earth',
    range_color=[wins_df[selected_feature].min(), wins_df[selected_feature].max()]  # Set the fixed color range 
)

# Update layout settings
fig.update_layout(
    geo=dict(
        showframe=False,                        # Hide country borders
        showcoastlines=False,                   # Hide coastlines
        projection_type='equirectangular'       # Choose a map projection
    )
)

# Show the choropleth map
st.plotly_chart(fig)
st.text('\n\n')
st.text('\n\n')

# BIVARIATE ANALYSIS 
st.subheader('Bivariate Analysis')

# CORRELATION AND HEAT MAP #
st.markdown('#### Correlation and heatmap')
st.write('''The corelation function and the heatmap can help in discovering the possible correlations between continous variables. ''')

# Plotting the heatmap
fig = plt.figure(figsize=(20,12))
sns.heatmap(wins_df.corr(),annot=True, cmap='PiYG')
plt.title('Correlation Matrix Heatmap')
plt.show()
plt.title('Correlation Matrix Heatmap')
plt.show()
st.pyplot(fig)

# Transforming the categorical variable in a numeric one 
wins_df['status'].replace(['Developing', 'Developed'],[0, 1], inplace=True)

# CORRELATION MATRIX # 
df_corr = wins_df.corr()
with st.expander("Expand for checking the correlation values among the variables"):
    st.table(df_corr)

st.text("\n")
st.write('''
From the above heatmap we can have these general takeaways:
- Life Expectancy (target variable) appears to be relatively highly correlated (negatively or positively) with:
    - Adult Mortality (negative).
    - HIV/AIDS (negative).
    - Income Composition of Resources (positive).
    - Schooling (positive).
- Infant deaths and Under Five deaths are extremely highly correlated.
- Thinness of 5-9 Year olds rate and Thinness of 10-15 Year olds rate is extremely highly correlated.
- Income Composition of Resources and Schooling are very highly correlated.
- Polio vaccine rate and Diphtheria vaccine rate are very positively correlated.
- Hepatitis B vaccine rate is relatively positively correlated with Polio and Diphtheria vaccine rates.
- HIV/AIDS is relatively negatively correlated with Income Composition of Resources.
- Percentage Expenditure and GDP are relatively highly correlated.

''')
st.text("\n\n")
st.text("\n\n")

# SCATTER PLOT #
st.markdown('#### Scatter plots')

# Changing the values of the variable status for a better visualiztion
wins_df['status'].replace([0, 1],['Developing', 'Developed'], inplace=True)

# -------------------Life Expectancy over Years------------------#
st.markdown('##### Life Expectancy over Years')
tab1, tab2 = st.tabs(["By country", "All countries"])

with tab1:
    st.write('This interactive plot shows country wise life expectancy over years:')
    fig1 =px.line(wins_df.sort_values(by='year'),x='year',y='life_expectancy',animation_frame='country',animation_group='year',color='country',markers=True, title = 'Country wise life expectancy over years')
    fig1.update_layout(yaxis_range=[35, 95],showlegend=False, yaxis=dict(tickmode='linear', tick0=30, dtick=5)) 
    st.plotly_chart(fig1)

with tab2:
    st.write('Here we can see how life expectancy has changed over the years, considerig all the countries together.')
    fig2 = plt.figure()
    sns.lineplot(x = 'year', y = 'life_expectancy', data=wins_df, marker='o', color = '#33ADA4')
    plt.title('Life Expectancy by Year')
    st.pyplot(fig2)

st.write('There appears to definitely be a positive trend over time.')
st.text("\n")
st.text("\n")

#-------------------Life_Expectancy w.r.t Status--------------------#
st.markdown('##### Life_Expectancy w.r.t Status')
st.write('''
Althought the correlation value is not so high for these 2 variables, we are still going to check the relation between the variable with a violin plot.
In fact, we can easily see that developing countries exhibit a wider spectrum of values, as far as life expectancy is concerned.
While in more developed countries, these values are more established, in developing countries the situation is still unstable and malleable.          
''')

fig=px.violin( wins_df ,x='status',y='life_expectancy',color='status',box=True, 
              color_discrete_sequence=['#33ADA4', '#F87188'])
st.plotly_chart(fig)
st.text("\n")
st.text("\n")

#-------------------Life Expectancy w.r.t Adult Mortality-------------------# 
st.markdown('##### Life Expectancy w.r.t Adult Mortality')
st.write('''
When there is a high negative correlation between Life expectancy and Adult Mortality, it suggests that countries or regions with higher adult mortality rates tend to have lower life expectancies. 
In other words, higher adult mortality is associated with shorter life expectancy.\n
This correlation may indicate that factors leading to higher adult mortality, such as :
- disease burden (hiv/aids)
- inadequate healthcare systems (total_expenditure, percentage_expenditure)
- socioeconomic challenges (income_composition_of_resources, alcohol)
- immunization coverage (hepatitis_b, polio, diphtheria)
contribute to a decrease in life expectancy.
''')

# Saving the correlation value
corr_le_adult = round(wins_df.corr().loc['life_expectancy']['adult_mortality'], 2)
st.write('Correlation value for these two variables: ', corr_le_adult)

# Plotting 
fig = px.scatter(wins_df,y='adult_mortality',x='life_expectancy',opacity=0.6, color = 'status',title ='<b> Life Expectancy w.r.t Adult Mortality and status',
                 color_discrete_map={'Developing': '#33ADA4', 'Developed': '#F87188'})
st.plotly_chart(fig)

# Life Expectancy + Adult Mortality + HIV/AIDS
fig = px.scatter(wins_df,y='adult_mortality',x='life_expectancy',opacity=0.6,title='<b> Life Expectancy w.r.t Adult Mortality and Hiv/Aids', 
                 color = 'hiv/aids', color_continuous_scale=  'Tealrose')
st.plotly_chart(fig)
st.write('''\n
Here we can see how the hiv/aids factor can lead to increased adult mortality, and thus lower life expectancy (disease burden). 
Higher the level of deaths per 1000 live births caused by HIV/AIDS for people under 5, higher the fact that the country concerned is one with an higher level of adult mortality and low life expectancy.  ''')


# Life Expectancy + Adult Mortality + Income composition of Resources
fig = px.scatter(wins_df,y='adult_mortality',x='life_expectancy',opacity=0.6,title='<b> Life Expectancy w.r.t Adult Mortality and Income composition of Resources', 
                 color = 'income_composition_of_resources', color_continuous_scale= 'Tealrose')
st.plotly_chart(fig)
st.write('''\n
Same reasoning for the Income composition of Resources (Socio-economical factor). 
We can see that most of the cases where mortality is high, life expectancy is low, and the country has also low Income composition of Resources. ''')

st.text("\n")
st.text("\n")

#-------------------Life Expectancy w.r.t HIV/AIDS-------------------# 
st.markdown('##### Life Expectancy w.r.t HIV/AIDS')
st.write('''
All the things stated before hold also for this graph, where we can see the strong negative correlation between life expectancy and HIV/AIDS. 
This means that as the prevalence of HIV/AIDS decreases, the life expectancy tends to increase. ''')

# Saving the correlation value
corr_le_adult = round(wins_df.corr().loc['life_expectancy']['hiv/aids'], 2)
st.write('Correlation value for these two variables: ', corr_le_adult)

# Plotting       
fig = px.scatter(wins_df,y='hiv/aids',x='life_expectancy',opacity=0.6, color = 'status',
                 color_discrete_map={'Developing': '#33ADA4', 'Developed': '#F87188'})
st.plotly_chart(fig)
st.text("\n")
st.text("\n")

#-------------------Life Expectancy w.r.t Income Composition of Resources-------------------# 
st.markdown('##### Life Expectancy w.r.t Income Composition of Resources')
st.write('''
The Income Composition of Resources is a measure of the distribution of income in a country and is here represented by the Human Development Index (HDI). It takes into account factors such as income inequality, access to education, and healthcare resources.

A high correlation between life expectancy and the Income Composition of Resources implies that countries with higher income equality, better access to education and healthcare, and more resources available to the population tend to have longer life expectancies. 
This is because higher income composition often translates to improved living standards, better healthcare services, and overall improved quality of life for the population, which can lead to better health outcomes and longer life expectancy.
         
The difference between developing and developed countries is alwasy the same.
''')

# Saving the correlation value
corr_le_adult = round(wins_df.corr().loc['life_expectancy']['income_composition_of_resources'], 2)
st.write('Correlation value for these two variables: ', corr_le_adult)

# Plotting
fig = px.scatter(wins_df,y='income_composition_of_resources', x='life_expectancy',opacity=0.6, color = 'status',
                 color_discrete_map={'Developing': '#33ADA4', 'Developed': '#F87188'})
st.plotly_chart(fig)
st.text("\n")
st.text("\n")

#-------------------Life Expectancy w.r.t Schooling------------------------------------# 
st.markdown('##### Life Expectancy w.r.t Schooling')
st.write('''
As before, if the level of schooling or education in a population increases, the life expectancy tends to increase as well.

Education and life expectancy are closely interconnected in several ways:
- Improved Health Knowledge: education provides individuals with knowledge about health, hygiene, and disease prevention. 
- Access to Healthcare: higher levels of education are often associated with better access to healthcare services. 
- Socioeconomic Status: sducation is often linked to higher socioeconomic status, which can positively impact health and life expectancy. People with higher education tend to have better job opportunities, higher income, and improved living conditions, all of which contribute to better health outcomes.
- Empowerment can lead to healthier lifestyle choices
''')

# Saving the correlation value
corr_le_adult = round(wins_df.corr().loc['life_expectancy']['schooling'], 2)
st.write('Correlation value for these two variables: ', corr_le_adult)

# Plotting 
fig = px.scatter(wins_df,y='schooling',x='life_expectancy',opacity=0.6, color = 'status',
                 color_discrete_map={'Developing': '#33ADA4', 'Developed': '#F87188'})
st.plotly_chart(fig)
st.text("\n")
st.text("\n")

#-------------------Income composition of resources w.r.t Schooling-----------------------# 
st.markdown('##### Income composition of resources w.r.t Schooling')

# Saving the correlation value
corr_le_adult = round(wins_df.corr().loc['income_composition_of_resources']['schooling'], 2)
st.write('Correlation value for these two variables: ', corr_le_adult)

# Plotting 
fig = px.scatter(wins_df,y='income_composition_of_resources',x='schooling',opacity=0.6, color = 'status',
                 color_discrete_map={'Developing': '#33ADA4', 'Developed': '#F87188'})
st.plotly_chart(fig)
st.text("\n")
st.text("\n")



# PCA #
st.markdown('#### PCA')
st.write('''It may be useful to run a Principal Components Analysis (PCA) on this data to reduce the amount of dimensions (features). But there are a number of assumptions/requirements when it comes to PCA:
- Continuous data: the data used should be of a continuous type
- Outliers: PCA is sensitive to outliers, therefore outliers should not be present
- Sample size: the sample size should have between 5-10 samples per feature
- Correlation: there should be correlation between the features
- Linearity: it is assumed that relationships between features are linear
- Normalized data: the data is generally normally distributed')
''')
st.text('\n\n')

st.markdown('\n##### Steps')
with st.expander("Expand to read the step performed"):
    st.write('''
    - Firstly, PCA is an unsupervised technique, so the target variable is not needed and can be dropped.
    - We remove the categorical variable 'country' and the year variables, because don't have significant differences among life expectancy.
    - Transforming the categorical variable in a numeric one.
    - Next i have to check the correlation values, and the variable that is not very correlated with any of the other is 'Population', so we are going to drop it.
    - We normalize the data.
    - Now the features set satisfies the following above assumptions: sample size, correlation, outliers, normalization and continuous data. 
        - The linearity assumption may not be true, however we take linearity for granted in order to see how the PCA works.
    - We frist proceed with the PCA without a fixed number of components, in order to find the optimal one.
    - We finally applying the fitting.
    ''')

# Copy of the dataset
feat_df = wins_df.copy()

# Dropping the target variable
feat_df.drop(columns='life_expectancy', inplace=True)

# Dropping non useful variables
feat_df.drop(columns=['country', 'year'], inplace=True)

# Transforming the categorical variable in a numeric one
feat_df['status'].replace(['Developing', 'Developed'],[0, 1], inplace=True)

# Dropping no so correlated variable 
feat_df.drop(columns=['population'], inplace=True)

# Standardize the numeric features by scaling them
scaler = StandardScaler()
feat_df_std = scaler.fit_transform(feat_df)

# Fitting
sklearn_pca = PCA()
Y = sklearn_pca.fit_transform(feat_df_std)

st.text('\n\n')


# Evaluation
# Explained variance by Principal Components
# Display the vector horizontally in Streamlit
st.markdown('\n\n##### Results')
st.write('Check the amount of variance that each PC explain in order to understand the importance of each feature.')
st.write('Explained variance by Principal Components: \n', ', '.join(str(num) for num in sklearn_pca.explained_variance_ratio_))

sum = 0
for i in range(10):
    temp = round(sklearn_pca.explained_variance_ratio_[i]*100, 2)
    sum += temp
sum_var = round(sum)
st.write('Summing the first 9 components we obtain a variance of ',sum_var, '''%. So, in order to capture at least 90% of the variance, 9 components are needed.
         This is with the assumption that the variables are linearly related as well.''')

# Plotting Explained variance by Principal Components
fig = plt.figure(figsize= (5,3))
plt.plot(sklearn_pca.explained_variance_)
st.pyplot(fig)
st.write('''\n
Based on the scree plot above, it would suggest that only PC1 be kept, this is likely not a great idea as PC1 only accounts for about the 37% of the total variance of the variables.
In this case, perhaps more features are better then less features for the modelling, and looking at the plot we coluld use from 9 to 16 features.''')

st.markdown('\n\n##### Conclusion')
st.write('''       
So we can conclude that the dimensionality reduction method of PCA didn't seem to garner very useful results. 
It is likely that further transformation of the features should be done, but knowing which model will be used is easier to select the best set of feautures that have to be done''')

st.text('\n\n')

# PCA - Continue #
st.markdown('#### PCA - Continue')
st.write('''
Before applying the PCA, the selection of the best features can be done by removing the *highly correlated to one another* variables. The aim was to keep only the variables which was more highly correlated with the target.
Infact, in some situations, removing highly correlated variables before PCA can be useful if the goal is to reduce multicollinearity or remove redundant information. Highly correlated variables can introduce instability or noise in the PCA results, as they may dominate the principal components' contribution. By removing highly correlated variables, you can focus on capturing unique and independent information in the remaining variables.

Checking the heatmap following are very/extremely highly correlated (correlation > .7 or correlation < -.7):
- Infant Deaths/Under Five Deaths (drop Infant Deaths - Under Five Deaths is more highly correlated to Life Expectancy)
- GDP/Percentage Expenditure (drop Percentage Expenditure - GDP is more higher correlated to Life Expectancy)
- Polio/Diphtheria (drop Polio - Diphtheria is more highly correlated to Life Expectancy)
- Thinness 5-9/Thinness 10-19 (drop Thinness 10-19 as correlations to other variables are slightly higher)
- Income Composition of Resources/Schooling (drop Schooling - Income Composition of Resources is more highly correlated with Life Expectancy)

Then we consider to remove also the 'population' for the reason stated before and the 'status' because is categorical and even numerical, it is not so relevant for the target.

By reasoning in this way, we can consider these 12 variables:
1. Adult Mortality
2. Alcohol
3. Hepatitis B
4. Measles
5. Under-Five Deaths
5. Total Expenditure
7. Diphtheria
8. HIV/AIDS
9. DP
10. Thinness 5-9 Years
11. Income Composition Of Resources
12. Status
''')

# Copy of the dataset
feat_df = wins_df.copy()

# Converting
feat_df['status'].replace(['Developing', 'Developed'],[0, 1], inplace=True)

# Select the relevant features
variables = ['adult_mortality',
             'alcohol',
             'hepatitis_b',
             'measles',
             'under-five_deaths',
             'total_expenditure',
             'diphtheria',
             'hiv/aids',
             'thinness_5-9_years',
             'income_composition_of_resources',
             'status'
            ]
feat_df = feat_df[variables]

# Standardize the numeric features by scaling them
scaler = StandardScaler()
feat_df_std = scaler.fit_transform(feat_df)

# Fitting for finding the n° of compoennts
sklearn_pca = PCA()
pca_components = sklearn_pca.fit_transform(feat_df_std)

# Perform PCA for finding the components
num_components = 9
pca = PCA(n_components=num_components)
pca_components = pca.fit_transform(feat_df_std)

st.text('\n\n')
st.text('\n\n')

# MULTIPLE LINEAR REGRESSION #
st.markdown('#### Multiple Linear regression')
st.write('''
1. Firstly, the Multiple linear regression, helps us understand the relationships between multiple independent variables and a dependent variable, allowing us to identify which variables have significant impacts on the outcome. 
2. Secondly, it enables us to make predictions and estimate the effects of changes in the independent variables on the dependent variable. 
3. Lastly, by evaluating the statistical significance of the coefficients, we can determine the strength of the relationships and identify the most influential predictors in the model. 
3. Overall, multiple linear regression is a powerful tool for understanding, predicting, and interpreting complex relationships in the data.
''')

# Valid also for other regressions
# Define the feature matrix X by selecting the columns (features) you want to include
X = wins_df.copy()
X.drop(['country', 'life_expectancy'],inplace=True,axis=1)

# Converting the categorical to numerical
X['status'].replace(['Developing', 'Developed'],[0, 1], inplace=True)

# Define the target variable y
y = wins_df['life_expectancy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an instance of the LinearRegression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test,y_pred) * 100
mae = mean_absolute_error(y_test,y_pred)

evaluation_df = pd.Series({'rmse':rmse , 'R2':R2, 'mae': mae}, name = 'Linear regression results')
st.dataframe(evaluation_df)
st.write('''
With an MSE of 14.157 and an R-squared score of 0.849, it can be inferred that the multiple linear regression model has reasonably good predictive performance.\n
The MSE indicates the average prediction error, while the R-squared score suggests that a significant portion of the variance in the target variable is explained by the model\'s independent variables.''')
st.text('\n\n')
st.text('\n')

# -------------------- with PCA ----------------------------#
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(pca_components, y, test_size=0.3, random_state=42)

# Create an instance of the LinearRegression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test,y_pred) * 100
mae = mean_absolute_error(y_test,y_pred)

evaluation_df = pd.Series({'rmse':rmse , 'R2':R2, 'mae': mae}, name = 'Linear regression results - PCA')
st.dataframe(evaluation_df)
st.write('''
         I tried the Multiple Linear Regression model after the PCA i\'ve shown before, but the results are worst.
         Myabe the reasons are the ones stated before, namely that several indicators are needed for the prediction and so the dimensionality reduction isn't a good path.

         For the next model i won't perform the PCA because i tried it before and everytime the results aren't improved.\n\n
         ''')


# Plotting
# Create a scatter plot for Actual vs Predicted
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs Predicted',
                         marker=dict(color='#33ADA4')))
fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                         mode='lines', line=dict(color='red', dash='dash'), name='Perfect Prediction Line'))

# Set axes labels and title
fig.update_layout(xaxis_title='Actual Values', yaxis_title='Predicted Values', title='Actual vs Predicted Multiple Linear Regression')

# Show the plot
st.plotly_chart(fig)

st.text("\n")
st.text('\n\n')
st.markdown('##### P-values analaysis')
st.write('''
To evaluate the statistical significance of the coefficients in multiple linear regression and so to identify the most influential predictors in the model, we can assess the p-values associated with each coefficient. 
A lower p-value suggests that the coefficient is statistically significant and that it has a significant impact on the dependent variable (p-value below 0.05)''')


# Add a constant column to X for the intercept
X_with_intercept = sm.add_constant(X)

# Fit the ordinary least squares (OLS) model
model_ols = sm.OLS(y, X_with_intercept).fit()

# Access the p-values
p_values = model_ols.pvalues[1:]  # Exclude the intercept term

# Dictionary of variable names and p-values
p_values = {
    'year': p_values[0],
    'status': p_values[1],
    'adult_mortality': p_values[2],
    'infant_deaths': p_values[3],
    'alcohol': p_values[4],
    'percentage_expenditure': p_values[5],
    'hepatitis_b': p_values[6],
    'measles': p_values[7],
    'under_five_deaths': p_values[8],
    'polio': p_values[9],
    'total_expenditure': p_values[10],
    'diphtheria': p_values[11],
    'hiv_aids': p_values[12],
    'gdp': p_values[13],
    'population': p_values[14],
    'thinness_10_19_years': p_values[15],
    'thinness_5_9_years': p_values[16],
    'income_composition_of_resources': p_values[17],
    'schooling': p_values[18]
}

with st.expander("Expand to explore the p-values"):
    # Check if each p-value is below 0.05 and print the corresponding variable name
    for variable, p_value in p_values.items():
        if p_value < 0.05:
            st.write(f"The p-value for variable '{variable}' is below 0.05")
        else:
            st.write(f"The p-value for variable '{variable}' is not below 0.05")

st.text('\n\n')

# Define the feature matrix X by selecting the columns (features) you want to include
X_p_value = wins_df.copy()
X_p_value.drop(['country', 'life_expectancy', 'infant_deaths', 'gdp' , 'thinness_10-19_years', 'population'],inplace=True,axis=1)

# Converting the categorical to numerical
X_p_value['status'].replace(['Developing', 'Developed'],[0, 1], inplace=True)

# Define the target variable y
y = wins_df['life_expectancy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_p_value, y, test_size=0.3, random_state=42)

# Create an instance of the LinearRegression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test,y_pred) * 100
mae = mean_absolute_error(y_test,y_pred)

evaluation_df = pd.Series({'rmse':rmse , 'R2':R2, 'mae': mae}, name = 'Linear regression results - p-value')
st.dataframe(evaluation_df)

st.write('''
Applying regression with the significant characteristics as independent variables, the mse value and $R^2$ improve a little bit. 
From now we will discard the not significative dependent variables and we will keep the signficative.\n
         
 Discarded: *'country', 'infant_deaths', 'gdp' , 'thinness_10-19_years', 'population'*
''')
st.text("\n")
st.text("\n")


# RIDGE AND LASSO REGRESSION #
st.markdown('#### Ridge and Lasso Regression')
st.write('''Ridge Regression and Lasso Regression are both regularization techniques used in linear regression to prevent overfitting and improve model performance.''')

# Spliting
X_train, X_test, y_train, y_test = train_test_split(X_p_value, y, test_size=0.3, random_state=42)

# The models
ridge_model=Ridge()
lasso_model=Lasso(alpha=0.00000001)

# Fitting
lasso_model.fit(X_train,y_train)
ridge_model.fit(X_train,y_train)

# Predicting
y_pred1 = ridge_model.predict(X_test)
y_pred2 = lasso_model.predict(X_test)

# Evaluate the model
rmse1 = mean_squared_error(y_test, y_pred1)
rmse2 = mean_squared_error(y_test, y_pred2)

R2_1 = r2_score(y_test,y_pred1) * 100
R2_2 = r2_score(y_test,y_pred2) * 100

mae1 = mean_absolute_error(y_test,y_pred1)
mae2 = mean_absolute_error(y_test,y_pred2)

evaluation_df1 = pd.Series({'rmse':rmse1 , 'R2':R2_1, 'mae': mae1}, name = 'Ridge regression Evaluation')
evaluation_df2 = pd.Series({'rmse':rmse2 , 'R2':R2_2, 'mae': mae2}, name = 'Lasso regression Evaluation')

st.dataframe(evaluation_df1)
st.dataframe(evaluation_df2)


# Create subplots 
fig = make_subplots(rows=1, cols=2, subplot_titles=('Ridge Regression Model', 'Lasso Regression Model'))

# Scatter plot for Model 1
fig.add_trace(go.Scatter(x=y_test, y=y_pred1, mode='markers', name='Ridge Regression Model',
                         marker=dict(color='#33ADA4')), row=1, col=1)

# Scatter plot for Model 2
fig.add_trace(go.Scatter(x=y_test, y=y_pred2, mode='markers', name='Ridge Regression Model',
                         marker=dict(color='#F87188')), row=1, col=2)

# Perfect Prediction Line for both subplots
fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                         mode='lines', line=dict(color='red', dash='dash'), name='Perfect Prediction Line'),
                         row=1, col=1)

fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                         mode='lines', line=dict(color='red', dash='dash'), name='Perfect Prediction Line'),
                         row=1, col=2)

# Set axes labels and title
fig.update_xaxes(title_text='Actual Values', row=1, col=1)
fig.update_xaxes(title_text='Actual Values', row=1, col=2)
fig.update_yaxes(title_text='Predicted Values', row=1, col=1)
fig.update_yaxes(title_text='Predicted Values', row=1, col=2)
fig.update_layout(title='Ridge and Lasso Actual vs Predicted', showlegend=False)

# Show the plot
st.plotly_chart(fig)

st.write('We can see no difference and improvements from the previous results. ')
st.text("\n")
st.text("\n")


# POLYNOMIAL REGRESSION #
st.markdown('#### Polynomial regression')
st.write('''
Polynomial Regression is a variation of linear regression that allows us to model nonlinear relationships between the independent variables and the dependent variable. 
Instead of fitting a straight line, Polynomial Regression fits a polynomial equation to the data, allowing for more flexible and curved relationships.
''')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_p_value, y, test_size=0.2, random_state=42)

# Create a PolynomialFeatures object with the degree of polynomial you want
degree = 2  # Change this to the desired degree of the polynomial
poly = PolynomialFeatures(degree=degree)

# Transform the original features into polynomial features
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create and fit the linear regression model on the polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_poly)

# Evaluate the model performance
rmse = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred) * 100
mae = mean_absolute_error(y_test,y_pred)

evaluation_df = pd.Series({'rmse':rmse , 'R2':R2, 'mae': mae}, name = 'Polynomial Regression results')
st.dataframe(evaluation_df)

# Create a scatter plot for Actual vs Predicted
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs Predicted',
                         marker=dict(color='#33ADA4')))
fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                         mode='lines', line=dict(color='red', dash='dash'), name='Perfect Prediction Line'))

# Set axes labels and title
fig.update_layout(xaxis_title='Actual Values', yaxis_title='Predicted Values', title='Actual vs Predicted Polynomial Regression')

# Show the plot
st.plotly_chart(fig)

st.write('The results of the Polynomial regression are a little better, and they may mean that nonlinear relationships exist between the independent variables and the dependent variable.')
st.text("\n")
st.text("\n")

# DECISION TREE AND RANDOM FOREST REGRESSION # 
st.markdown('#### Decision Tree Regression')
st.write('''
Decision Tree Regression is a non-parametric supervised learning algorithm used for regression tasks. The goal is to split the data in a way that minimizes the impurity or maximizes the information gain at each step.
Random Forest Regression is an ensemble learning method that combines multiple decision trees to make predictions. 
The randomness reduces overfitting and improves generalization. ''')


# Assuming you have your feature matrix X and target variable y
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_p_value, y, test_size=0.2, random_state=42)

# Decision Tree Regression
# Create and fit the Decision Tree model
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_decision_tree = decision_tree_model.predict(X_test)

# Evaluate the Decision Tree model performance
rmse_decision_tree = mean_squared_error(y_test, y_pred_decision_tree)
r2_decision_tree = r2_score(y_test, y_pred_decision_tree)* 100

evaluation_df = pd.Series({'rmse':rmse_decision_tree , 'R2':r2_decision_tree}, name = 'Decision Tree Regression Model')
st.dataframe(evaluation_df)

# Random Forest Regression
# Create and fit the Random Forest model
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_random_forest = random_forest_model.predict(X_test)

# Evaluate the Random Forest model performance
rmse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
r2_random_forest = round(r2_score(y_test, y_pred_random_forest) * 100, 1)

evaluation_df = pd.Series({'rmse':rmse_random_forest , 'R2':r2_random_forest}, name = 'Random Forest Regression Model')
st.dataframe(evaluation_df)

# Create subplots for Actual vs Predicted (Model 1) and Actual vs Predicted (Model 2)
fig = make_subplots(rows=1, cols=2, subplot_titles=('Decision Tree Regression Model', 'Random Forest Regression Model'))

# Scatter plot for Model 1
fig.add_trace(go.Scatter(x=y_test, y=y_pred_decision_tree, mode='markers', name='Decision Tree model',
                         marker=dict(color='#33ADA4')), row=1, col=1)

# Scatter plot for Model 2
fig.add_trace(go.Scatter(x=y_test, y=y_pred_random_forest, mode='markers', name='Random Forest model',
                         marker=dict(color='#F87188')), row=1, col=2)

# Perfect Prediction Line for both subplots
fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                         mode='lines', line=dict(color='red', dash='dash'), name='Perfect Prediction Line'),
                         row=1, col=1)

fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                         mode='lines', line=dict(color='red', dash='dash'), name='Perfect Prediction Line'),
                         row=1, col=2)

# Set axes labels and title
fig.update_xaxes(title_text='Actual Values', row=1, col=1)
fig.update_xaxes(title_text='Actual Values', row=1, col=2)
fig.update_yaxes(title_text='Predicted Values', row=1, col=1)
fig.update_yaxes(title_text='Predicted Values', row=1, col=2)
fig.update_layout(title='Decision Tree and Random Forest Actual vs Predicted', showlegend=False)

# Show the plot
st.plotly_chart(fig)

st.write('''
The results with Decision Tree Regressor and Random Forest Regressor might be better compared to simple linear regression because these models have greater flexibility to capture nonlinear relationships between the features and the target variable.
Here below some reasons for the better performance:
- **Nonlinear relationships**: In real-world datasets, the relationships between the features and the target variable are often nonlinear. Decision Tree Regressor and Random Forest Regressor can handle nonlinear relationships more effectively than linear regression.
- **Interactions**: Decision Tree Regressor and Random Forest Regressor can capture interactions between features, which can significantly impact the target variable.
- **Ensemble learning**: Random Forest Regressor is an ensemble model that combines multiple decision trees, reducing overfitting and improving generalization.
- **Ability to learn complex patterns**: Decision Tree Regressor and Random Forest Regressor can learn complex patterns and capture interactions between multiple features, making them more suitable for datasets with high-dimensional feature spaces.
''')
         
st.text("\n\n")
st.text("\n\n")

###------------------------------------------------------------- CONCLUSIONS -----------------------------------------------------------------------###       
st.header('Conclusion')
st.write('''
    In this analysis we have grasped how different aspects influence people\'s life expectancy, and how some of these factors are also interrelated between each other.
    We understand which were the most correlated factors to life expectancy, such as adult mortality, Hiv/Aids, Schooling and the Income Composition of Resources.
         
    We have seeen that is possible to rely on predictive models based, for example, on regression, to support and guide research and studies in this field. The best model was the Random Forest Regression model,
    that has an R-squared of about 96%. We have also viewd how quite all factors are significant for the model, despite the ones we labeled as not significative after the P-value analysis. 
    ''')
