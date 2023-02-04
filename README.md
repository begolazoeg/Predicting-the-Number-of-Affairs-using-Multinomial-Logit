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

En este modelo se toman en cuenta como variables independientes a 'rate_amrriage', 'age', 'yrs_married', 'children', 'religious', 'educ'. Cada categoría de la variable dependiente 'affairs' tiene una sección de resultados.

# Rate marriage
Es posible ver que 'rate_marriage' es un factor negativo significante para todas las categorías de 'affairs'. Esto indica que a medida que aumenta la calificación de la felicidad en el matrimonio, disminuye la probabilidad de tener aventuras extramatrimoniales. Esto indica que las personas que están más satisfechas con su matrimonio tienen menos probabilidades de buscar satisfacción emocional o sexual fuera de su matrimonio. Sin embargo, es importante tener en cuenta que esta relación puede ser bidireccional, es decir, que también es posible que las personas que tienen aventuras extramatrimoniales estén insatisfechas con su matrimonio y que esto esté afectando su calificación de felicidad en el matrimonio.

# Age
En cuanto a la variable 'age' vemos que su coeficiente es negativo y significante para la categoría 3 de 'affairs'. Lo cual significa que por cada unidad de incremento de la edad, la probabilidad de tener 3 'affairs' disminuye 0.1124.

# Children and Years married
Sobre los factores 'children' y 'yrs_married', sus coeficientes no serán tomados en cuenta ya que carecen de significación estadísitca por no tener un valor-p significativo.

# Religiousness
Esta variable aparece como significativa en todas las categorías de 'affairs' y todos sus coeficientes son negativos. Es decir que a mayor religiosidad, disminuye la probabilidad de tener 'affairs'. Este efecto negativo es más fuerte en la categoría 1 de 'affairs', lo cual podría sugerir que una fuerte religiosidad funciona como un mecanismo de prevneción a la aventuras extramaritales (en este dataset). 
