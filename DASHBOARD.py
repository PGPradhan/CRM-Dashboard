#!/usr/bin/env python
# coding: utf-8

# In[3]:


# installlation required
get_ipython().system('pip install Lifetimes')
get_ipython().system('pip install openpyxl')


# In[4]:


# libraries
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler


# In[5]:


get_ipython().system('pip install squarify')


# In[6]:


import squarify


# In[7]:


df = pd.read_csv("demoCRM.csv", encoding='unicode_escape')


# In[8]:


df = df[~df["Invoice"].str.contains("C", na=False)]
df.shape


# In[9]:


def check_df(dataframe):
    print("################ Shape ####################")
    print(dataframe.shape)
    print("############### Columns ###################")
    print(dataframe.columns)
    print("############### Types #####################")
    print(dataframe.dtypes)
    print("############### Head ######################")
    print(dataframe.head())
    print("############### Tail ######################")
    print(dataframe.tail())
    print("############### Describe ###################")
    print(dataframe.describe().T)

check_df(df)


# In[10]:


df.isnull().sum()


# In[11]:


df.dropna(inplace=True)
df.isnull().sum()


# In[12]:


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


# In[13]:


cat_cols = [col for col in df.columns if df[col].dtypes =="O"]
cat_but_car = [col for col in df.columns if df[col].nunique() > 100 and df[col].dtypes == "O"]
cat_cols = [col for col in cat_cols if col not in cat_but_car]
cat_cols


# In[14]:


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        fig_dims = (15, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation = 45, ha = 'right')
        plt.show()

cat_summary(df, "Country", plot=True)


# In[15]:



num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


# In[16]:


# How many sales for each product?
df_product = df.groupby("Description").agg({"Quantity":"count"})
df_product.reset_index(inplace=True)
df_product


# In[17]:


# Top 10 Products
top_pr= df_product.sort_values(by="Quantity",ascending=False).head(10)

sns.barplot(x="Description", y="Quantity", data=top_pr)
plt.xticks(rotation=90)
plt.show()


# In[18]:


# total price per invoice
df["TotalPrice"] = df["Price"] * df["Quantity"]


# In[19]:


# Determining the analysis date for the recency
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)


# In[20]:


rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvıiceDate: (today_date- InvıiceDate.max()).days,
                                    "Invoice": lambda Invoice: Invoice.nunique(),
                                    "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ["recency","frequency","monetary"]
rfm.describe().T


# In[21]:


rfm = rfm[rfm["monetary"] > 0]
rfm.describe().T


# In[22]:


rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
# frequency_score
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
# monetary_score
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4], duplicates='drop')

#  RFM Score
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
rfm.head(10)


# In[23]:


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm.head(10)


# In[24]:


# Let's group RFM mean and frequency values according to segments
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# In[25]:


sgm= rfm["segment"].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=sgm.index,y=sgm.values)
plt.xticks(rotation=45)
plt.title('Customer Segments',color = 'blue',fontsize=15)
plt.show()


# In[26]:


# Treemap Visualization
df_treemap = rfm.groupby('segment').agg('count').reset_index()
df_treemap.head()


# In[27]:


fig, ax = plt.subplots(1, figsize = (10,10))

squarify.plot(sizes=df_treemap['RFM_SCORE'],
              label=df_treemap['segment'],
              alpha=.8,
              color=['tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
             )
plt.axis('off')
plt.show()
#plt.savefig('treemap.png')


# In[28]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[29]:


# Importing libraries
import numpy as np
import pandas as pd
#from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


# visulaisation
from matplotlib.pyplot import xticks
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


# Data display coustomization
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[32]:


lead = pd.read_csv(r"leadsource_md.csv")
lead.head()


# In[33]:


lead_dub = lead.copy()

# Checking for duplicates and dropping the entire duplicate row if any
lead_dub.drop_duplicates(subset=None, inplace=True)
lead_dub.shape


# In[34]:


lead.shape


# In[35]:


lead.info()


# In[36]:


lead.describe()


# In[37]:


lead.isnull().sum()


# In[38]:


# Drop all the columns in which greater than 3000 missing values are present

for col in lead.columns:
    if lead[col].isnull().sum() > 3000:
        lead.drop(col, 1, inplace=True)


# In[39]:


# Check the number of null values again

lead.isnull().sum()


# In[40]:


lead['City'].value_counts()


# In[41]:


# City
lead.City.value_counts()


# In[42]:


lead.head()


# In[43]:


round(100*(lead.isnull().sum()/len(lead.index)), 2)


# In[44]:


# City
lead.City.value_counts()


# In[45]:


lead.City.describe()


# In[46]:


lead['Country'].value_counts()


# In[47]:


round(100*(lead.isnull().sum()/len(lead.index)), 2)


# In[48]:


for column in lead:
    print('___________________________________________________')
    print (column)
    print('___________________________________________________')
    print(lead[column].astype('category').value_counts())
    print('___________________________________________________')


# In[49]:


lead['Lead Profile'].astype('category').value_counts()


# In[50]:


lead['Specialization'].value_counts()


# In[51]:


# Drop the null value rows in the column 'TotalVisits'

lead = lead[~pd.isnull(lead['TotalVisits'])]


# In[52]:


# Check the null values again

lead.isnull().sum()


# In[53]:


# Drop the null values rows in the column 'Lead Source'

leads = lead[~pd.isnull(lead['Lead Source'])]


# In[54]:


# Drop the null values rows in the column 'Specialization'

lead = lead[~pd.isnull(lead['Specialization'])]


# In[55]:


leads.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[56]:


Converted = round((sum(lead['Converted'])/len(lead['Converted'].index))*100,2)

print("We have almost {} %  Converted rate".format(Converted))


# In[57]:


plt.figure(figsize = (10,5))
ax= sns.countplot(x = "Lead Origin", hue = "Converted", data = lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 90)
ax.set_yscale('log')
plt.show()


# In[58]:


plt.figure(figsize = (25,5))
ax= sns.countplot(x = "Lead Source", hue = "Converted", data = lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 90)
ax.set_yscale('log')
plt.show()


# In[59]:


plt.figure(figsize = (10,5))
ax= sns.countplot(x = "Lead Source", hue = "Converted", data = lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 90)
ax.set_yscale('log')
plt.show()


# In[60]:


plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
ax= sns.countplot(x = "Do Not Email", hue = "Converted", data = lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 90)
ax.set_yscale('log')
plt.subplot(1,2,2)
ax= sns.countplot(x = "Do Not Call", hue = "Converted", data = lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 90)
ax.set_yscale('log')
plt.show()


# In[61]:


plt.figure(figsize = (25,5))
ax= sns.countplot(x = "Last Activity", hue = "Converted", data = lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 90)
ax.set_yscale('log')
plt.show()


# In[62]:


plt.figure(figsize = (10,5))
ax= sns.countplot(x = "Last Activity", hue = "Converted", data = lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 90)
ax.set_yscale('log')
plt.show()


# In[63]:


lead.Specialization.describe()


# In[64]:


plt.figure(figsize = (20,6))
ax= sns.countplot(x = "Last Notable Activity", hue = "Converted", data = lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 90)
ax.set_yscale('log')
plt.show()


# In[65]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (10,5))
sns.heatmap(lead.corr(), annot = True, cmap="rainbow")
plt.show()


# In[66]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[67]:


import pandas as pd
import panel as pn
pn.extension()
import seaborn as sns
import matplotlib.pyplot as plt

# # Assuming rfm is your DataFrame

def create_segment_bar_plot():
    sgm = rfm["segment"].value_counts()
    plt.figure(figsize=(10, 7))
    sns.barplot(x=sgm.index, y=sgm.values)
    plt.xticks(rotation=45)
    plt.title('Customer Segments', color='blue', fontsize=15)
    return pn.pane.Matplotlib(plt.gcf(), tight=True)

def create_treemap():
    fig, ax = plt.subplots(1, figsize=(7, 7))
    squarify.plot(sizes=df_treemap['RFM_SCORE'],
                  label=df_treemap['segment'],
                  alpha=.8,
                  color=['tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
                 )
    plt.axis('off')
    plt.close()  # Close the plot to prevent it from being displayed directly
    return pn.pane.Matplotlib(fig, tight=True)

def create_lead_origin_plot():
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x="Lead Origin", hue="Converted", data=lead)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
    plt.xticks(rotation=90)
    ax.set_yscale('log')
    return pn.pane.Matplotlib(plt.gcf(), tight=True)

def create_lead_source_plot():
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x="Lead Source", hue="Converted", data=lead)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
    plt.xticks(rotation=90)
    ax.set_yscale('log')
    return pn.pane.Matplotlib(plt.gcf(), tight=True)

def create_do_not_email_call_plot():
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    ax1 = sns.countplot(x="Do Not Email", hue="Converted", data=lead)
    for p in ax1.patches:
        ax1.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
    plt.xticks(rotation=90)
    ax1.set_yscale('log')
    
    plt.subplot(1, 2, 2)
    ax2 = sns.countplot(x="Do Not Call", hue="Converted", data=lead)
    for p in ax2.patches:
        ax2.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
    plt.xticks(rotation=90)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return pn.pane.Matplotlib(plt.gcf(), tight=True)

def create_last_activity_plot():
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x="Last Activity", hue="Converted", data=lead)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
    plt.xticks(rotation=90)
    ax.set_yscale('log')
    return pn.pane.Matplotlib(plt.gcf(), tight=True)

def create_correlation_heatmap():
    plt.figure(figsize=(15, 10))
    sns.heatmap(lead.corr(), annot=True, cmap="rainbow")
    plt.title('Correlation Matrix')
    return pn.pane.Matplotlib(plt.gcf(), tight=True)



# # Create the Panel dashboard
# dashboard = pn.Column(
#     pn.Row(
#         pn.pane.Markdown("## CRM SCORE SEGMENTATION"),
#         create_segment_bar_plot(),
#         align="center",
#     ),
#     pn.Row(
#         pn.pane.Markdown("## Customer Segments Treemap"),
#         create_treemap(),
#         align="center",
#     ),
#     pn.Row(
#         pn.pane.Markdown("## Lead Origin"),
#         create_lead_origin_plot(),
#         align="center",
#     ),
#     pn.Row(
#         pn.pane.Markdown("## Lead Source"),
#         create_lead_source_plot(),
#         align="center",
        
#     ),
#     pn.Row(
#         pn.pane.Markdown("## Do Not CALL/MAIL "),
#         create_do_not_email_call_plot(),
#         align="center",
#         ),
#     pn.Row(
#         pn.pane.Markdown("## Last Activity"),
#         create_last_activity_plot(),
#         align="center",
#         ),
#     pn.Row(
#         pn.pane.Markdown("## Correlation Heat Map"),
#         create_correlation_heatmap(),
#         align="center"
        
#         ),
    
# )

# # Display the dashboard
# dashboard.servable()
# dashboard.show


# In[ ]:





# In[68]:


pip install panel


# In[ ]:





# In[92]:


dashboard.show()


# In[81]:


import panel as pn
pn.extension()
from panel.interact import interact

# Define a function to select and display graphs based on the dropdown selection
def select_and_display_graph(graph_type):
    if graph_type == " ":
        return 0
    elif graph_type == "Segment Bar Plot":
        return create_segment_bar_plot()
    elif graph_type == "Treemap":
        return create_treemap()
    elif graph_type == "Count Plot - Lead Origin":
        return create_lead_origin_plot()
    elif graph_type == "Lead Source Plot":
        return create_lead_source_plot()
    elif graph_type == "Do Not Email/Call Plot":
        return create_do_not_email_call_plot()
    elif graph_type == "Last Activity Plot":
        return create_last_activity_plot()
    elif graph_type == "Lead Origin Plot":
        return create_lead_origin_plot()
    elif graph_type == "Correlation Heatmap":
        return create_correlation_heatmap()

# Define the dropdown menu options
graph_types = [" ","Segment Bar Plot", "Treemap", "Count Plot - Lead Origin", "Lead Source Plot",
               "Do Not Email/Call Plot", "Last Activity Plot", "Lead Origin Plot", "Correlation Heatmap"]

# Create the dropdown menu using interact
# interact(select_and_display_graph, graph_type=graph_types)


# Create the dropdown menu using interact
dropdown = pn.Row(
    pn.layout.HSpacer(),
    interact(select_and_display_graph, graph_type=graph_types),
    pn.layout.HSpacer(),
    align="center"
)

# Create the Panel dashboard with a specific background color
dashboard = pn.Column(
    pn.pane.Markdown("## AI based CRM System Analysis"),
    dropdown,
    background="#f0f0f0",  # Set the background color
    align="center"
)

# Display the dashboard on localhost
dashboard.servable()
dashboard.show()


# In[75]:


dashboard.show()


# In[82]:


import panel as pn
from panel.interact import interact

# Define a function to select and display graphs based on the dropdown selection
def select_and_display_graph(graph_type):
    if graph_type == "Segment Bar Plot":
        return create_segment_bar_plot()
    elif graph_type == "Treemap":
        return create_treemap()
    elif graph_type == "Count Plot - Lead Origin":
        return create_count_plot()
    elif graph_type == "Lead Source Plot":
        return create_lead_source_plot()
    elif graph_type == "Do Not Email/Call Plot":
        return create_do_not_email_call_plot()
    elif graph_type == "Last Activity Plot":
        return create_last_activity_plot()
    elif graph_type == "Lead Origin Plot":
        return create_lead_origin_plot()
    elif graph_type == "Correlation Heatmap":
        return create_correlation_heatmap()

# Define the dropdown menu options
graph_types = ["Segment Bar Plot", "Treemap", "Count Plot - Lead Origin", "Lead Source Plot",
               "Do Not Email/Call Plot", "Last Activity Plot", "Lead Origin Plot", "Correlation Heatmap"]

# Create the dropdown menu using interact
dropdown = interact(select_and_display_graph, graph_type=graph_types, _manual=True)

# Apply styles to the dropdown
dropdown = pn.Row(dropdown, align="center", background='#f0f0f0', width=900)

# Create the Panel dashboard
dashboard = pn.Column(
    pn.pane.Markdown("## AI based "),
    dropdown,
    align="center"
)

# Display the dashboard
dashboard.show()


# In[ ]:




