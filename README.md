# Analysis of Factors Influencing the Crime Rate

This project identifies the data science problem of “What Promotes Crime?”. Potential factors, both direct and indirect, including moon phase, weather, days of a week, regional education level and adult obesity/smoking rate are taken into consideration to see how different factors affect certain types of crime more closely. Crime types are categorized into eight groups: `Assault`, `Burglary`, `Death`, `Drug`, `Fraud`, `Robbery`, `Theft` and `Sexual Crime`. 2017 crime data for 14 US major cities are closely analyzed, which are `Austin`, `Baton Rouge`, `Boston`, `Chicago`, `Denver`, `Detroit`, `Hartford`, `Las Vegas`, `Los Angeles`, `Philadelphia`, `New Orleans`, `New York`, `San Francisco`, `Washington DC`. Different methodologies have been applied to analyze the relationship between different factors and crime type, such as correlation, clustering, association rule, hypothesis testing, PCA, spatial-temporal model, machine learning and network analysis. The results reveal that `Theft` and `Burglary` crime have a strong correlation between each other. Distinctive temporal and spatial patterns for different days of the week. `Days of the week`, `Education level`, `Obesity rate`, `Smoking rate`, `New moon`, `Rainy Weather` are found to be more related crime types compared to other factors being considered.

This is the final project of ANLY-501, which is also a collaborative project completed by a group of 4 people. It was split into 2 stages:

### Part 1

+ Identify potential analyses
+ Data collection 
+ Data cleaning
+ Feature engineering
+ Data visualization

### Part 2

+ Exploratory analysis
+ Hypothesis test
+ Predictive models
+ Data visualization

Please check our final report at https://lemonning0713.wixsite.com/website/analysis.

## ANLY-501_part_1_moonphase.py
+ Retrieve weather and moon phase data of 14 major cities in the US in 2017 using API from www.worldweatheronline.com.
+ Merge all the datasets of 14 cities into one.

## ANLY-501_part_2_clustering_correlation_hypothesis_hist.py
+ Conduct unsupervised leaning using 3 clustering methods: K-means, DBSCAN and WARD.
+ Conduct correlation analysis and create correlation plots (heatmap).
+ Create histograms.
+ Conduct t-test, linear regression and logistic regression.
