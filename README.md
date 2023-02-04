# Multinomial Logit Analysis for Predicting Affairs
![alt text](https://media.ambito.com/p/ae6267c2d701b13e60c837b603b9152e/adjuntos/239/imagenes/040/417/0040417747/bizarrap-shakira.jpg)

## Little summary
This project aims to predict the likelihood of individuals having affairs based on various factors such as age, years of marriage, number of children, religious beliefs and education level. The Multinomial Logit model is used for this analysis. The data contains information from 6217 individuals and the dependent variable is the number of affairs, with four categories (1, 2, 3 and 4 more affairs). The model results are presented in the form of coefficients, standard errors, Z-scores and P-values for each predictor variable. The final results show which predictor variables have a significant impact on the likelihood of individuals having affairs.

## Status of the project
The project status is completed!

## Dataset and libraries used
The data set used in this project is a simulated data set from the statsmodels library, which contains information about a sample of individuals such as their marriage rating, age, years married, number of children, religiousness, and education level.

## Specific goal
The specific goal of this project is to develop a predictive model using Multinomial Logit to identify the factors that contribute to the likelihood of individuals having affairs. The results of this project will provide insight into the relationships between different predictors and the number of affairs individuals have.

## Steps
1. Data acquisition: The first step is to acquire the data that will be used in the project. In this case, the data is a simulated data set from the statsmodels library, which contains information about a sample of individuals such as their marriage rating, age, years married, number of children, religiousness, and education level.

2. Data preprocessing: The next step is to preprocess the data. This step typically includes cleaning and transforming the data, handling missing values, and creating new features if necessary.

3. Exploratory Data Analysis (EDA): After the data has been preprocessed, an Exploratory Data Analysis (EDA) is performed to gain a deeper understanding of the data. This step involves visualizing the data, identifying patterns and trends, and determining the distribution of the variables.

4. Model building: After the EDA, the model is built. In this case, a multinomial logistic regression model is used to study the effect of Marriage Rating, Age, Years Married, Number of Children, Religiousness, and Education Level on the Number of Affairs. This step includes selecting the independent variables and fitting the model.

5. Model interpretation: The final step is to interpret the model and report the results. This includes analyzing the coefficients and p-values of the model and drawing conclusions about the relationships between the independent variables and the dependent variable. The report should include a summary of the model, including the coefficients, p-values, and other statistics.

## Results
![alt text](https://github.com/begolazoeg/Predicting-the-Number-of-Affairs-using-Multinomial-Logit/blob/main/Images/reuslt%20modelo%20base.PNG?raw=true)

In this model, 'rate_amrriage', 'age', 'yrs_married', 'children', 'religious', 'educ' are taken into account as independent variables. Each category of the dependent variable 'affairs' has an outcome section.

# Rate marriage
It is possible to see that 'rate_marriage' is a significant negative factor for all categories of 'affairs'. This indicates that as the rating of happiness in marriage increases, the likelihood of having extramarital affairs decreases. This indicates that people who are more satisfied with their marriage are less likely to seek emotional or sexual satisfaction outside of their marriage. However, it is important to keep in mind that this relationship may be bidirectional, i.e., it is also possible that people who have extramarital affairs are dissatisfied with their marriage and that this is affecting their happiness rating in marriage.

# Age
Regarding the variable 'age' we see that its coefficient is negative and significant for category 3 of 'affairs'. This means that for each unit increase in age, the probability of having 3 affairs decreases by 0.1124.

# Children and Years married
Regarding the factors 'children' and 'yrs_married', their coefficients will not be taken into account since they lack statistical significance because they do not have a significant p-value.

# Religiousness
This variable appears as significant in all the 'affairs' categories and all its coefficients are negative. That is, the higher the religiosity, the lower the probability of having affairs. This negative effect is stronger in category 1 of 'affairs', which could suggest that a stronger religiosity functions as a prevention mechanism for extramarital affairs (in this dataset). 
