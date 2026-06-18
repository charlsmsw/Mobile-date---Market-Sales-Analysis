# Project Title

Mobile Phone Dataset

from google.colab import drive
drive.mount('/content/drive')
# Domain
Retail/ Mobile Phone Market Analysis
# Objective

1.Analyze mobile phone specifications and understand their impact on pricing.

2.Identify the relationship between features such as RAM, storage, battery capacity, and camera quality with the overall price.

3.Compare different brands based on performance, pricing, and customer ratings.

4.To examine how technical specifications influence customer satisfaction levels.

5.Identify pricing patterns across budget, mid-range, and premium segments.

6.Detect trends in smartphone features and market demand.

7.Generate meaningful insights using statistical and visualization techniques.

8.support data-driven decisions for product development and pricing strategies.

9.Recommend optimal feature combinations that provide better market competitiveness.
# Outcome

The outcome of this project is the identification of key mobile phone features such as RAM, storage, battery capacity, camera quality, and processor that significantly influence pricing and customer ratings. The analysis highlights brand-wise comparisons, market positioning, and pricing patterns across different segments. It provides meaningful insights into how technical specifications impact customer satisfaction and overall product value. These findings support data-driven decisions for product development, competitive pricing strategies, and improved market performance.
#Dataset Information

Source: Kaggle

Year / Timeline: Data collected during 2025

Dataset Description:The dataset contains detailed information about various mobile phones, including brand name, model, price, technical specifications (such as RAM, storage, battery capacity, camera, processor, and display size), operating system, and customer ratings. It is designed to analyze how different features influence pricing, performance, and customer satisfaction in the smartphone market.

Name - Mobile Phone Dataset

# Type of Analysis

Descriptive Analysis-Summarizes the basic features of the dataset such as average price, most common RAM and storage, brand distribution, and overall rating trends.

Diagnostic Analysis-Examines relationships between variables to understand why certain phones are priced higher, such as the impact of RAM, battery, and camera on price.

Predictive Analysis-Uses statistical models (like regression) to predict mobile phone price or rating based on features such as RAM, storage, and processor.

Prescriptive Analysis-Provides recommendations for optimal feature combinations and pricing strategies to improve competitiveness and customer satisfaction.
# Stages for DA Project
## Stage 1 – Problem Definition and Dataset Selection

Problem Definition:The smartphone market offers a wide range of mobile phones with varying specifications, brands, and price levels, making it difficult to understand how different features influence pricing and customer satisfaction. Consumers often struggle to determine whether a phone's price accurately reflects its technical specifications, while businesses need data-driven insights to design competitive products and set optimal pricing strategies. Therefore, this project aims to analyze mobile phone data to identify the relationship between features, price, and customer ratings, and to uncover meaningful patterns that support better decision-making in the smartphone market.

Expected Outcome:The expected outcome of this project is to identify the key factors that significantly influence mobile phone pricing and customer ratings. The analysis will reveal how features such as RAM, storage, battery capacity, camera quality, and processor affect overall product value. It will also highlight brand-wise performance comparisons and pricing trends across different market segments. The final outcome will provide meaningful insights that support better business decisions, improved pricing strategies, and enhanced product development planning.


Dataset Selection Name: Mobile Phone Dataset

Source: Kaggle

Size: 500+ user records

Features: Brand, Model, Price, RAM Capacity, Internal Storage, Processor Type, Operating System, Camera Specifications, Battery Capacity, Display Size, and Customer Rating.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/ASSIGNMENT/Final project/Mobile Data.csv'
df = pd.read_csv(file_path)
df
#First Five rows
df.head()
# Last five Rows
df.tail()
#statistical summary
df.describe()

# Check Data Types and Non-Null Counts
df.info()
# Check Shape
print("Dataset(Rows,Coloums)):", df.shape)
#Datatypes of dataset
print("\nData Types:\n", df.dtypes)
#  Check Null Value
print("-Null Values-")
df.isnull()
#Total elements in dataset
print("Count of elements:", df.size)

#Index values
print("Index value:",df.index)
#Columns
print("columns of Dataframe:",df.columns)
#Null Values
print (f" Checking Null values : ")

df.isnull().sum()
# No of unique values
print(f" Number of Unique Values : ")
df.nunique()
#Duplicate
df.duplicated()
# Duplicate Check
df.duplicated().sum()
## Stage 2 – Data Cleaning and Pre-processing

Handle missing values (impute or drop)

Handle duplicates

Treat outliers if required

Check skewness and apply transformations

Convert data types if needed

Feature transformations (date parts, derived fields if required for analysis)
###Check Missing Values

df.isnull().sum()
##Handle missing values (impute or drop)
##Numeric Columns

num_cols = df.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

##Categorical Columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.isnull().sum()
##Handle duplicates

df.duplicated().sum()
df.drop_duplicates(inplace=True)
df
# Check Skewness & Apply Transformation
df.skew(numeric_only=True)
# If skewed → Apply log transformation

df['Number_of_Ratings'] = np.log1p(df['Number_of_Ratings'])
df
##Convert Data Types

df['Price'] = df['Price'].astype('int')
df['Star_Rating'] = df['Star_Rating'].astype('float')
df['RAM_GB'] = df['RAM_GB'].astype('float')
df.dtypes
##Feature transformations (date parts, derived fields if required for analysis

df['Colour'] = df['Name'].str.split().str[3:-2]
df
def price_category(price):
    if price < 10000:
        return "Budget"
    elif price < 30000:
        return "Mid-Range"
    else:
        return "Premium"

df['Price_Category'] = df['Price'].apply(price_category)
df

## Stage 3 – EDA and Visualizations
# New Section
1.   List item
2.   List item



Univariate Analysis → distribution of single variables (countplot, histogram, boxplot)

Bivariate Analysis → relation between two variables (scatterplot, barplot, correlation heatmap)

Multivariate Analysis → relation among 3+ variables (pairplot, grouped analysis, pivot tables, advanced plots)

Interpretation MUST with every visualization

Focus on business story not just charts

Each chart must be in seperate cells.
###countplot
plt.figure()
sns.histplot(df['Price_Category'], bins=30, kde=True)
plt.title("Distribution of Price")
plt.xlabel("Range")
plt.ylabel("Frequency")
plt.show()

Interpretation:(countplot)

This chart is a countplot that shows the number of smartphones available from each company in the dataset.


It shows which companies have more smartphone models and helps understand the presence of different brands in the dataset.


The feature used for this visualization is the Company column.

This visualization shows which smartphone companies have the highest number of models in the dataset.
# histogram
plt.figure(figsize = (8,5))

plt.hist( df["Price"],
         color = "Orange",
         edgecolor = "black",
         linewidth = 1.2,
         alpha=0.8
)
plt.title( "Price Rate" )
plt.xlabel( "Amount" )
plt.ylabel( "Count" )
plt.show()
Interpretation:(histogram)

This chart is a histogram that shows the distribution of smartphone prices in the dataset.


The chart shows how mobile phone prices are spread across different ranges and helps identify whether most phones are budget, mid-range, or premium.


The feature used for this visualization is the Price column.


This visualization shows that most smartphones fall within the mid-price range, with fewer devices in the premium segment.
# boxplot

plt.figure(figsize=(6,4))

sns.boxplot(y = df['Star_Rating'])

plt.title("Phone Rating Distribution")
plt.ylabel("Rating")
plt.show()
Interpretation:(box plot)

This chart is a box plot that shows the distribution of RAM across different smartphone price categories.


The chart helps understand how RAM values vary within each price category and identifies the median, spread, and possible outliers.


The features used for this visualization are Price_Category and RAM_GB from the dataset.


This visualization shows that smartphones in the premium category generally have higher RAM, while budget phones tend to have lower RAM configurations.
# scatterplot
sns.scatterplot(
    data=df,
    x='Price',
    y='Number_of_Ratings',
    hue="Price_Category"
    )

plt.legend(bbox_to_anchor=(1,1))

plt.show()
Interpretation:(Scatter plot)

This chart is a scatter plot that shows the relationship between RAM and smartphone price.


It helps understand how the amount of RAM affects the price of smartphones.

The features used are RAM_GB and Price.


This visualization shows that smartphones with higher RAM generally have higher prices
# barplot
plt.figure(figsize=(15,10))

sns.barplot(x=df['Company'].value_counts().index,
            y=df['Company'].value_counts().values,
palette="bright",)
plt.xticks(rotation=45)
plt.xlabel("Company")
plt.ylabel("Number of Smartphones")
plt.title("Number of Smartphones by Company")

plt.show()
Interpretation:(bar plot)

This chart is a bar plot that shows the average price of smartphones for each company in the dataset.


The chart helps compare the average price of smartphones across different brands and understand which companies focus on premium or budget devices

The features used for this visualization are Company and Price from the dataset.


This visualization shows that some companies have higher average smartphone prices, indicating a focus on premium devices, while others offer more affordable phones.
 ##correlation heatmap
corr=df.corr(numeric_only=True)
corr
plt.figure(figsize=(10,6))

sns.heatmap(corr, annot=True)

plt.xticks(rotation=20)
plt.show()
Interpretation:(heatmap)

This chart is a heatmap that shows the correlation between different numerical features in the dataset.


The chart helps identify relationships between variables such as price, RAM, storage, ratings, and reviews.


The features used for this visualization are numerical columns such as Price, RAM_GB, ROM_GB, Star_Rating, Number_of_Ratings, and Number_of_Reviews.


This visualization helps identify which features have stronger relationships and may influence smartphone price or ratings.
###pairplot
sns.pairplot(
    df[['Price','Star_Rating','RAM_GB','ROM_GB','Company']],
    hue='Company',
    diag_kind='kde'
)

plt.suptitle("Pairwise Relationship Between Smartphone Features", y=1.02)
plt.show()
Interpretation:(pair plot)

This chart is a pair plot that shows the relationships between multiple numerical features in the smartphone dataset.


It displays pairwise relationships between variables and also shows the distribution of each variable on the diagonal.

The features used for this visualization include Price, Star_Rating, RAM_GB, ROM_GB, and Company.


This visualization helps identify patterns and relationships between smartphone features such as price, RAM, storage, and ratings.
##Line plot

plt.figure(figsize=(7,5))
sns.lineplot(x='RAM_GB',
             y='Price',
            data=df)
plt.title("Price vs RAM")
plt.show()
Interpretation:(Line plot)

This chart is a line plot that shows the relationship between RAM size and the average smartphone price.


The chart helps identify the trend between RAM configuration and smartphone pricing.


The features used for this visualization are RAM_GB and Price from the dataset.

This visualization shows that smartphones with higher RAM generally have higher average prices, indicating that RAM plays an important role in determining smartphone cost.
company_counts = df['Company'].value_counts().head(6)

plt.figure(figsize=(6,6))
plt.pie(company_counts,
        labels=company_counts.index,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Top Smartphone Companies Distribution")
plt.show()
Interpretation:(Pie Chart)

This chart is a pie chart that shows the distribution of smartphones among different companies.


The chart helps identify which companies contribute the most smartphones to the dataset.


The feature used for this visualization is the Company column.

This visualization shows the market presence of different smartphone brands in the dataset.
###grouped_data

grouped_data = df.groupby(['Company','Price_Category'])['Price'].mean()

print(grouped_data)
Interpretation:(Grouped_data)

Grouped data analysis organizes the dataset into different categories.

It summarizes data using functions like mean, sum, or count.

This helps compare different groups such as brands or price ranges.

It identifies patterns and differences between categories.

The analysis provides clear insights from large datasets.
###Pivot_table

pivot_table = pd.pivot_table(df, values='Price', index='Company', aggfunc='mean')

print(pivot_table)
Interpretation:(Pivot Table)



A pivot table summarizes large datasets in a structured format.

It arranges data into rows, columns, and values.

It helps compare multiple variables in the dataset.

It highlights totals, averages, and counts for each category.

It makes data analysis easier and helps identify trends.
###Bubble Chart

plt.figure(figsize=(10,12))

plt.scatter(df['Price'], df['Company'],
            s=df['Star_Rating']*50,
            alpha=0.6,
            color='green')

plt.xlabel("RAM (GB)")
plt.ylabel("Price")
plt.title("Bubble Chart: RAM vs Price vs Rating")

plt.show()
Interpretation:(Bubble chart)

This chart is a bubble chart that shows the relationship between smartphone RAM and price, where the size of the bubble represents the star rating.

The chart helps analyze how smartphone performance (RAM) and customer satisfaction (rating) relate to the price of the device.


The features used for this visualization are RAM_GB, Price, and Star_Rating.

This visualization shows that smartphones with higher RAM usually have higher prices and good customer ratings, indicating better performance and user satisfaction.
###Violinplot

sns.violinplot(
    x="Price_Category",
    y="Star_Rating",
    hue="Price_Category",
    data=df,
    split=False,
    inner="point",
    palette="cool"
)
plt.title("Violin Plot of payment and fare")
plt.show()


Interpretation:(Violin plot)

This chart shows the relationship between smartphone price categories and their star ratings using a visualization such as a violin plot.


The chart helps compare customer ratings across different smartphone price segments like budget, mid-range, and premium.


The features used for this visualization are Price_Category and Star_Rating from the dataset.


This visualization shows that star ratings are generally similar across all price categories, indicating that both affordable and premium smartphones can receive good customer ratings.
## Stage 4 – Documentation, Insights and Presentation

---



---



Show a dashboard aligning all the charts in powerbi by connecting python in PowerBI

Summarize findings in plain English (what do the patterns mean)

Highlight 3–5 key insights (trends, anomalies, correlations)

Provide recommendations for business or decision-making

Explain the FINAL STORY WITH THE DASHBOARD

Create a PDF pasting your DASHBOARD, explaining all the above mentioned concepts and submit it

Power BI Dashboard Link:
#Dashboard PDF Link:https://drive.google.com/file/d/13EnB8yYzuPKfCfog0dQ413M2PBCNTwbA/view?usp=sharing
#Report Link:https://drive.google.com/file/d/1OrJesrm8EXCKQPZHfu24SyTJA__2bmSy/view?usp=sharing
# Future Enhancement

Include more features like battery, processor, and camera details for deeper analysis.

Apply advanced visualization techniques to improve data understanding.

Use machine learning models to predict mobile prices or ratings.

Perform real-time data analysis by updating the dataset regularly.

Add user sentiment analysis from reviews for better insights.

Improve data cleaning techniques to handle missing and inconsistent data.

Create an interactive dashboard using tools like Power BI or Tableau.

Enhance feature engineering to derive more meaningful variables.

Compare data across different regions or time periods.

Automate the analysis process using Python scripts.
# Conclusion

The analysis of the mobile phone dataset provided valuable insights into pricing, features, and customer ratings.

Various techniques like data cleaning, grouping, and pivot tables helped in understanding the dataset effectively.

Visualizations made it easier to identify patterns, trends, and comparisons between different categories.

The study highlighted how factors like brand and features influence price and ratings.

Overall, the project demonstrates how data analysis can support better decision-making and extract meaningful insights from raw data.
