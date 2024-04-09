import warnings
warnings.filterwarnings('ignore')
# Importing libraries
import numpy as np
import pandas as pd
#from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

st.title("Lead Source Identification")

# Data display coustomization
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

f1 = st.file_uploader(":file_folder: Upload a File ", type=(["csv", "text", "xlsx", "xls"]))

if f1 is not None:
    filename = f1.name
    st.write(filename)
    lead = pd.read_csv(f1, encoding='unicode_escape')
else:
    os.chdir(r"E:\Projects\CRM\pages")
    lead = pd.read_csv("leadsource_md.csv", encoding='unicode_escape')

st.write(lead)

round(100*(lead.isnull().sum()/len(lead.index)), 2)

# Remove rows with NaN values
lead = lead.dropna(subset=['Converted'])

# Calculate conversion rate
Converted = round((sum(lead['Converted'])/len(lead.index))*100, 2)

print("We have almost {} % Converted rate".format(Converted))

plt.figure(figsize=(10, 5))
ax = sns.countplot(x="Lead Origin", hue="Converted", data=lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
plt.xticks(rotation=90)
ax.set_yscale('log')
st.pyplot(plt)

plt.figure(figsize=(25, 15))
ax = sns.countplot(x="Lead Source", hue="Converted", data=lead)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
plt.xticks(rotation=90)
ax.set_yscale('log')

# Display the plot in Streamlit
st.pyplot(plt)

plt.figure(figsize=(20, 5))

# First subplot
plt.subplot(1, 2, 1)
ax1 = sns.countplot(x="Do Not Email", hue="Converted", data=lead)
for p in ax1.patches:
    ax1.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
plt.xticks(rotation=90)
ax1.set_yscale('log')

# Second subplot
plt.subplot(1, 2, 2)
ax2 = sns.countplot(x="Do Not Call", hue="Converted", data=lead)
for p in ax2.patches:
    ax2.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
plt.xticks(rotation=90)
ax2.set_yscale('log')

# Display the plots in Streamlit
st.pyplot(plt)

plt.figure(figsize=(10, 5))

# Create the countplot
ax = sns.countplot(x="Last Activity", hue="Converted", data=lead)

# Annotate the bars with their heights
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Set y-axis scale to logarithmic
ax.set_yscale('log')

# Display the plot in Streamlit
st.pyplot(plt)


numeric_lead = lead.select_dtypes(include=[float, int])

plt.figure(figsize=(10, 5))

# Create the heatmap
sns.heatmap(numeric_lead.corr(), annot=True, cmap="rainbow")

# Display the plot in Streamlit
st.pyplot(plt)








