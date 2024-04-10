import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import datetime as dt
import os

st.title("Lead Scoring")

# Load the dataset
f1 = st.file_uploader(":file_folder: Upload a File ", type=(["csv", "text", "xlsx", "xls"]))

if f1 is not None:
    filename = f1.name
    st.write(filename)
    df = pd.read_csv(f1, encoding='unicode_escape')
else:
    os.chdir(r"E:\Projects\CRM\pages")
    df = pd.read_csv("demoCRM.csv", encoding='unicode_escape')

df = df[~df["Invoice"].str.contains("C", na=False)]

# Drop missing values
df.dropna(inplace=True)

df["TotalPrice"] = df["Price"] * df["Quantity"]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

st.write(df)
rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvıiceDate: (today_date - InvıiceDate.max()).days,
                                     "Invoice": lambda Invoice: Invoice.nunique(),
                                     "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
rfm.columns = ["recency", "frequency", "monetary"]

rfm = rfm[rfm["monetary"] > 0]


rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5,4, 3, 2], duplicates='drop')
# frequency_score
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
# monetary_score
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4], duplicates='drop')

#  RFM Score
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)+ rfm["monetary_score"].astype(str))


seg_map = {
    r'[1-2][1-2][1-2]': 'hibernating',
    r'[1-2][2-5][1-3]': 'at_Risk',
    r'[1-2][1-5]5': 'cant_loose',
    r'[1-2]3[1-2]': 'about_to_sleep',
    r'[1-2]33': 'need_attention',
    r'3[1-3][1-3]': 'loyal_customers',
    r'4[1-2]1': 'promising',
    r'5[1-4][1-2]': 'new_customers',
    r'[3-4][4-5][2-4]': 'potential_loyalists',
    r'5[3-5][3-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

# Group by 'segment' and calculate the mean and count for 'recency', 'frequency', and 'monetary'
segment_stats = rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# Display the aggregated data using st.write
st.write(segment_stats)
sgm = rfm["segment"].value_counts()

# Create the plot
plt.figure(figsize=(10, 7))
sns.barplot(x=sgm.index, y=sgm.values)
plt.xticks(rotation=45)
plt.title('Customer Segments', color='blue', fontsize=15)

# Display the plot in Streamlit
st.pyplot(plt)

df_treemap = rfm.groupby('segment').size().reset_index(name='count')

# Set up the plot
fig, ax = plt.subplots(1, figsize=(10, 10))

# Plot the TreeMap
squarify.plot(sizes=df_treemap['count'],
              label=df_treemap['segment'],
              alpha=0.8,
              color=['tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray'])

# Configure plot display
plt.axis('off')
plt.title("""Customer Segments TreeMap""", color='blue', fontsize=15)

# Display the plot in Streamlit
st.pyplot(fig)









