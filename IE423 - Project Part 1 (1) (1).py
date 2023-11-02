#!/usr/bin/env python
# coding: utf-8

# # IE423 - PROJECT PART 1
# ### Contributions of group members:
# #### %33.3  -  Emel YAĞAN 2019402066                                                                 
# #### %33.3  -  Mustafa Kutay ALMAK 2019402144                                                 
# #### %33.3  -  Müge ŞENAY 2019402201                                                                
# 
# 

# In[1]:


import pandas as pd
# Read the CSV file
file_path = "/Users/kutay/Downloads/all_ticks_wide.csv"
data = pd.read_csv(file_path)

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import warnings
warnings.filterwarnings('ignore')


# We utilized Python's Pandas library to manage a dataset stored in a CSV file. Initially, we read the CSV file and extracted specific columns, including 'timestamp',and the stocks we are interested in: 'GARAN', 'HALKB', 'TUPRS', 'PETKM', 'MGROS', and 'SISE'. Upon selecting these columns, we conducted an assessment for missing values within the dataset. We filled the missing values with previous data to ensure continuity. Then, we transformed the 'timestamp' column into a DateTime format and introduced a new 'year_month' column to depict time in 'YYYY-MM' format. Finally, we organized the dataset in chronological order, arranging it based on the 'year_month' column. These steps were essential in the initial preparation and structuring of the data.

# In[2]:


new_data = data[["timestamp", "GARAN","HALKB","TUPRS","PETKM","MGROS","SISE"]]
#print(new_data)
selected_rows = new_data.iloc[22132:37685]
#print(selected_rows)
NA_count = selected_rows.isna().sum()

# Print NA count for each stock
#print(NA_count)
data_filled = selected_rows.ffill(axis=0)
#print(data_filled.isna().sum())

# Fill NA values from the previous timestamp
data_filled = selected_rows.ffill(axis=0)

# convert "timestamp" column to the date format
data_filled['timestamp'] = pd.to_datetime(data_filled['timestamp'])

# Create 'year_month' column with format 'YYYY-MM'
data_filled['year_month'] = data_filled['timestamp'].dt.to_period('M')

# Sort the months in chronological order
data_filled = data_filled.sort_values(by='year_month')


# ## PART A - BOXPLOT 

# In[3]:


import matplotlib.pyplot as plt


# We calculated the Interquartile Range (IQR) for each stock values, analyzing them on a monthly basis. The code identified outliers by comparing each month's data to the boundaries established by the IQR. To do this, it computed quartiles for each month and defined specific boundaries, checking for values significantly below or above these limits. Any outliers found were collected and stored in a list. The 'identify_outliers' function individually processed each month's data, checking for extreme values beyond the established limits. Once all months were assessed, the code printed a list of these outliers, detailing their timestamps, respective months, and the values that notably deviate from the overall dataset distribution for chosen stocks.

# In[4]:


stocks = ['TUPRS', 'PETKM', 'GARAN', 'HALKB', 'MGROS', 'SISE']
for stock_boxplot in stocks:
    # Calculate the IQR for stock values by month
    Q1 = data_filled.groupby('year_month')[stock_boxplot].quantile(0.25)
    Q3 = data_filled.groupby('year_month')[stock_boxplot].quantile(0.75)
    IQR = Q3 - Q1

    # Define a list to store the outliers
    outliers_list = []

    # Define a function to identify outliers by month and append them to the list
    def identify_outliers(group):
        month = group['year_month'].max()
        Q1 = group[stock_boxplot].quantile(0.25)
        Q3 = group[stock_boxplot].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = group[(group[stock_boxplot] < lower_bound) | (group[stock_boxplot] > upper_bound)]
        if not outliers.empty:
            outliers_list.append(outliers[['timestamp', 'year_month', stock_boxplot]])

    # Apply the function to identify outliers by month
    data_filled.groupby('year_month').apply(identify_outliers)
    
    # Increase the max display rows to show more rows
    pd.set_option('display.max_rows', None)
    

#     # Print the list of outliers
#     for outlier in outliers_list:
#         print(outlier)

        # Assuming you already have the 'outliers_list' as shown in your code

    # Combine the list of DataFrames into a single DataFrame
    combined_outliers = pd.concat(outliers_list)

    # Convert the 'timestamp' column to datetime
    combined_outliers['timestamp'] = pd.to_datetime(combined_outliers['timestamp'])

    # Sort the DataFrame by the 'timestamp' column
    combined_outliers.sort_values(by='timestamp', inplace=True)

    # Print the sorted DataFrame
    print(combined_outliers)


# The code generates boxplots to display how each stock prices vary across 24 months. Each boxplot represents the monthly distribution of chosen stock prices. The code sets labels and adjusts the visual layout for better clarity. Lastly, it displays the created boxplot for visualization.

# In[5]:


# Create a boxplot for each month 
for stock_boxplot in stocks:
    data_filled.boxplot(column=stock_boxplot, by='year_month', figsize=(12, 6))

    plt.title(f'{stock_boxplot} Stock Prices Boxplot (For 24 Months)')  # Use f-string to include stock_boxplot variable
    plt.xlabel('Year-Month')
    plt.ylabel(f'{stock_boxplot} Stock Prices')
    plt.xticks(rotation=45)
    plt.show()


# ## PART B - 3 SIGMA

# We calculated the monthly mean and standard deviation for each stock within our dataset. We have previously organized the statistics by 'year_month'. After consolidating the monthly statistics, we merged this information back into the primary dataset. Additionally, we developed a function to identify outliers within the data. This function compared the stock values to their respective mean and standard deviation, flagging values that significantly deviated from the standard range. We organized these outliers by stock name, following a specific order as defined. Finally, we printed out the detected outliers for each stock, detailing the timestamps and the values that were identified as outliers.

# In[6]:


import numpy as np

# Calculate monthly mean and standard deviation for each stock
monthly_stats = data_filled.groupby(['year_month']).agg({'GARAN': ['mean', 'std'],
                                                        'HALKB': ['mean', 'std'],
                                                        'TUPRS': ['mean', 'std'],
                                                        'PETKM': ['mean', 'std'],
                                                        'MGROS': ['mean', 'std'],
                                                        'SISE': ['mean', 'std']})

# Flatten the multi-index columns
monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]

# Merge the monthly stats back into the main dataframe
data_filled = data_filled.merge(monthly_stats, left_on='year_month', right_index=True)

# Define a function to identify outliers
def detect_outliers(row):
    outliers = []
    for col in ['GARAN', 'HALKB', 'TUPRS', 'PETKM', 'MGROS', 'SISE']:
        mean = row[f'{col}_mean']
        std = row[f'{col}_std']
        if not (mean - 3 * std <= row[col] <= mean + 3 * std):
            outliers.append((col, row['timestamp'], row[col]))
    return outliers

# Apply the outlier detection function
outliers = data_filled.apply(detect_outliers, axis=1)

# Organize outliers by stock name
outliers_by_stock = {stock: [] for stock in ['GARAN', 'HALKB', 'TUPRS', 'PETKM', 'MGROS', 'SISE']}

for row_outliers in outliers:
    for col, timestamp, value in row_outliers:
        outliers_by_stock[col].append((col, timestamp, value))

# Print outliers by stock name in a specific order
desired_order = ['TUPRS', 'PETKM', 'GARAN', 'HALKB', 'MGROS', 'SISE']
for stock in desired_order:
    for outlier in outliers_by_stock[stock]:
        col, timestamp, value = outlier
        print(f"Outlier detected for {col} at timestamp: {timestamp} with value: {value}")


# 
# 
# 
# ### MGROS Outliers
# May 2nd and 3rd, 2016
# On May 2, 2016, MGROS shares were experiencing a significant drop, as the value, which had been steadily rising until shortly before, suddenly plummeted rapidly. Prior to this decline, MGROS had been rapidly appreciating. In the first week of April, Tera Investment provided a buy recommendation for MGROS, while in the last week, Goldman Sachs still had a target price for MGROS of 26.0 TL, up from 22.8 TL. However, due to a minor global crisis that occurred, causing the dollar to gain value worldwide, all global stock markets experienced a decline due to people wanted to take less risks, and MGROS was also affected, rapidly losing its value. As it lost value, Tera Investment and then Burgan Investment removed MGROS from their portfolios, and this decline gained significant momentum, causing Migros' stock to remain at low values throughout the month of May. Therefore, in the early days of the month, MGROS, which rapidly declined from a high value, appeared as an positive outlier in our boxplot.
# 
# June 10th, 2016
# On June 10th, prior to June 10th, the stock market was going through challenging times, still showing the effects of the sharp decline it had previously experienced. In these uncertain days, Migros acquired Kipa from Tesco. This acquisition led to a brief but significant increase in Migros' stock, and since the impact of this purchase faded quickly, we encountered a positive outlier on June 10th.
# 
# September 23rd and October 3rd,2016
# After the coup attempt on July 15th, the dollar was rapidly continuing its ascent, initially negatively impacting the markets. As the Turkish lira rapidly lost its value, stock prices were well below their expected levels. A few months after the coup attempt, as the markets recovered and the exchange rate stabilized, stock prices began to follow the delayed trend of the dollar. This increase provided a very positive signal, and even Tera Investment reported that it did not receive a negative response from MGROS and did not expect any negativity after the upcoming interest rate decision.
# However, on September 24th, due to the international credit rating agency Moody's downgrading Turkey's credit rating to below investment grade, with the agency lowering Turkey's credit rating from "Baa3" to "Ba1" while maintaining a "stable" outlook, investors increased their selling activities, and the stock market was severely affected. MGROS, which was on the rise, also felt the impact of this decline and became a positive outlier in our charts on September 23rd. This decline continued to be evident at the beginning of October. Although MGROS lost some more value and became more stable, it remained a positive outlier on October 3rd since it was still depreciating.
# 
# December 1st, 2nd and 5th,2016
# By the end of November, with Moody's downgrading Turkey's credit rating, investors began to flee Turkey, leading to a significant depreciation of the Turkish lira. In the early days of December, the lack of confidence that caused this depreciation directly affected the markets, and MGROS also felt the impact of this decline. While this decline was reflected in the charts as a mere fluctuation, it appeared as negative outliers in our boxplot. It took some time for the stock to regain its value. Following this process, once again, MGROS followed the Dollar trend with some delay.
# 
#  August 10th and 17th, 2016
# On August 10th and 17th, 2016, when we examined 15-minute data, we observed only 3 outliers in total. Therefore, we did not embark on extensive research, but we noticed that two major global events during that period were the likely causes of these fluctuations. The first was the tension between the United States and North Korea and the subsequent easing of that tension, and the second was the terrorist attack in Barcelona.
# 
# 
# ### SISE Outliers
# 
# January 4th and 5th, February 3rd, and March 31st.
# In the early days of 2016, the newly announced minimum wage had a direct impact on the markets. This decision led to a depreciation of the Turkish lira and caused declines in some stocks, including SISE. Towards the end of the year, SISE, which had been appreciating, experienced a rapid decline with the arrival of the new year. During this decline, on January 4th and 5th, which were the first days of the month, positive outliers were observed as the stock was in a falling trend.
# 
# This decline continued until February, and in February, unlike January, SISE's value was steadily on the rise, which led to the observation of a negative outlier on February 3rd. One of the main reasons for the increase in February was Goldman Sachs raising its price target for SISE from 4.32 TL to 4.75 TL and giving a buy recommendation.
# 
# This rise continued steadily until April. Due to the consistent increase in March, a positive outlier was observed on the last day of the month, March 31st.
# 
# April 29th, May 2nd, 3rd, and 6th.
# While the rise of SISE had slowed down in April, it continued. Positive news, such as the newly opened glass recycling factory, contributed to SISE's positive image. In fact, on April 28th, Goldman Sachs raised its price target for SISE from 4.75 TL to 4.80 TL, causing the stock to rapidly rise on that day. However, after April 29th, due to the global crisis that ensued, SISE was also affected and began to decline rapidly, stabilizing at the beginning of May. During this period, as the decline continued on April 29th, May 2nd, and May 3rd, positive outliers were observed, while the lowest point of the decline, which occurred on May 6th, resulted in a negative outlier. After May 6th, SISE followed a stable chart, leading to the observation of both negative and positive outliers in the first week of the month.
# 
#  
# December 1st and 2nd 2016
# In the early days of December, just like in all markets, Moody's downgrade of Turkey at the beginning of November had an impact on SISE. However, SISE quickly recovered from this situation, and as December began, it started following the trend of the dollar. The rise it experienced in Turkish lira terms led to the observation of negative outliers in the first days of the month.
# August 16th and 17th, 2017
# By the time August arrived, SISE had recovered from the decline caused by the dividends distributed in May and was back on an upward trend. It was benefiting from the positive movement happening globally. In fact, Ak Investment had listed Şişe Cam among its most preferred stocks. However, SISE was affected by the volatility in mid-August, and the rapid depreciation of its stock price for a few days was partly due to the company's announcement of its 6-month net profit. Although it quickly recovered, during this period, negative outliers were observed in the boxplot for two days.
# 
# 
# ### TUPRS OUTLIERS
# January 4, 5 and 6, 2016
# The negative outlier in Tüpraş shares could be attributed to the announcement of management changes on December 31, 2015, and potential market uncertainties arising from this relationship on January 4, 2016, the next trading day following this announcement. The management alterations and market disclosures might have triggered sudden fluctuations in the stock prices. Additionally, the outliers observed on January 4th, 5th, and 6th could be a result of investor reactions to the previously announced management changes, potentially causing continued market volatility during these subsequent trading days.
# 
# 
# May 2 and 3, 2016
# The positive outliers observed in Tüpraş shares on May 2nd and 3rd, 2016, could potentially be linked to the news announcing an agreement between Tüpraş and Kuwait Petroleum Corporation (KPC) for oil supply. According to information from the Kuwait state news agency KUNA, Nabil Bouresli, the International Marketing Manager of KPC, stated that the agreement with Tüpraş arrived during a period of intense competition among oil-exporting countries for market share in the Mediterranean region. This collaboration signifies a significant move, considering the competitive landscape within the region. In 2015, Turkey purchased 149,227 tons of crude oil from Kuwait, further underlining the significance of this partnership for Tüpraş.
# 
# December 1 and 2, 2016
# Due to the news such as "BNP Paribas raises Tüpraş target price from 68.9 tl to 80.2 tl, recommendation 'buy''" and "Deutsche Bank updates its most favored stocks list in Turkey," the Tüpraş shares witnessed an upward surge. This rise resulted in the emergence of negative outliers on December 1st and 2nd, 2016.
# 
# 
# 
# March 1, 2, and 3, 2017
# In March 2017, we observed upward surges in Tüpraş shares, surpassing the short-term declining trend. Consequently, sudden rises and falls were experienced. The upward movement can be noted from the negative outliers at the beginning of the month. Additionally, the reflections of declines in response to abrupt rises can be visualized from the box plot.
# 
# March 21, 2017
# On March 21, 2017, Tüpraş shares were confirmed to have a stable outlook by Moody's Investor Services. The small interquartile range supports this situation. Instantaneous buying and selling trends may have caused investors to panic, leading to the observation of outliers which can be observed in Box Plot.
# 
# April 3, 4, and 5, 2017
# Due to the announcement of dividend non-payments after April 5, Tüpraş shares experienced a sudden surge. Consequently, the days prior to the announcement, namely April 3, 4, and 5, might have appeared as negative outliers in the Box Plot, remaining lower compared to the rest of the month.
# 
# April 26, 27, and 28, 2017
# Similar to the previously mentioned reason, the presence of a positive outlier towards the end of the month highlights the trend observed in Tüpraş stocks during April. The positive outlier, occurring in the later days of the month, could be indicative of dividend’s effect and sudden upward shift in Tüpraş stock prices. 
# 
# June 1 and 2, 2017
# In June 2017, the trend in Tüpraş shares resulted in negative outliers on June 1st and 2nd, as reflected in the Box Plot. This trend in the stock performance coincided with the news corroborating the industrial highlights of 2016, where Tüpraş, Ford, and Tofaş were positioned at the summit according to information provided by Bloomberg HT. This alignment indicates a possible relationship between the stock's temporary decline and the news surrounding the top industry standings, suggesting a plausible correlation between market reactions and the company's comparative industrial position.
# 
# August 1 and 2, 2017
# In August 2017, a rising trend in Tüpraş shares led to negative outliers on August 1st and 2nd, as evidenced in the Box Plot. This trend coincided with supportive news indicating that in the first half of 2017, Tüpraş managed to increase its sales by 5.1%. Additionally, A1 Capital offered a 'buy' recommendation for Tüpraş with a target price of 139.00 TL. The correlation between these events indicates a potential upward trend in Tüpraş stocks. This is evidenced by the emergence of negative outliers at the beginning of the month, as depicted in the Box Plot.
# 
# December 18, 19, and 21, 2017
# During this period, Tüpraş's lawsuit challenging the Competition Board decision was dismissed, and BNP PARIBAS reduced its target price for TÜPRAŞ from 124.50 TL to 115.25 TL. These news might have led to the emergence of negative outliers in the Box Plot.
# 
# ### PETKM OUTLIERS
# June 24, 2016
# The negative outliers observed in Petkim shares on June 24, 2016, could be attributed to the news regarding to a vacant board seat. Conversely, the positive outliers might be linked to brief periods of positive momentum experienced by the company. These events reflect how the market reacted: the negative outliers potentially stemmed from uncertainties or concerns related to a sudden board appointment, while the positive outliers likely emerged due to temporary bursts of positive market momentum.
# 
# October 5 and 6, 2016
# The reason for Petkim's positive outlier on October 5 and 6 could be related to good news from the U.S. and rising oil prices affecting currency rates. Also, global stock markets looked better due to the higher oil prices and fewer worries about Deutsche Bank. Petkim might have benefited from this positive market mood, causing its stock to stand out positively. This can be seen by positive outliers in the Box Plot.
# 
# November 1, 2, and 3, 2016
# We could not find any significant news on this outliers we observe in the Box Plot.
# 
# December 1 and 2, 2016
# Positive trend in Petkim’s stocks results negative outliers in the beginning of the month. The increase in Petkim's shares can be attributed to the news of rising Brent crude oil prices to $52 per barrel following the OPEC agreement. This upward movement in oil prices sparked a resurgence in global "reflation" pricing dynamics. The surge in the yield of 10-year U.S. Treasury bonds to 2.38% and the global trend of stock gains, particularly in energy companies, could have influenced the rise in Petkim's shares.
# 
# May 2, 3, and 4, 2017
# An optimistic pattern in Petkim’s stock leads to adverse outliers at the start of the month.The continuous positive uptrend in Petkim's shares could be linked to the ongoing strong demand for petrochemical products, which is expected to have a positive impact on Petkim's financial performance. Particularly, the increasing margin between Ethylene and Naphtha prices is anticipated to positively influence the company's profitability in the first quarter. This favorable market scenario could be driving investor confidence and contributing to the rise in Petkim's shares.
# 
# 
# October 31, 2017
# Petkim's decision to seek up to $500 million through foreign bond issuance, as revealed in their announcement to KAP, possibly led to increased investor optimism, contributing to the rise in Petkim's shares and the observed positive outliers in the Box Plot by the end of the month. This news about external borrowing for strategic investment often encourages positive market sentiments, driving investor confidence and potential stock growth. 
# 
# 

# In[ ]:




