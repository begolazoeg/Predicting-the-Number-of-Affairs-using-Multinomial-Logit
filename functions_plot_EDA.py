#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)
sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px  

from scipy.stats import skew
from scipy.stats import kurtosis

import warnings
warnings.filterwarnings('ignore')


# In[1]:


def plotHist(df,nameOfFeature):
    """
    Function to create univariate histograms to see the distribution of the values of a variable
    
    input: df-> dataframe
           nameOfFeature -> str that may contain the name of the feature
    
    """
    data_array = df[nameOfFeature]
    hist_data = np.histogram(data_array)
    binsize = .5

    trace1 = go.Histogram(
        x = data_array,
        histnorm='probability',
        name='Histogram'.format(nameOfFeature),
        autobinx=False,
        marker_color='coral',
        xbins=dict(
            start=df[nameOfFeature].min()-1,
            end=df[nameOfFeature].max()+1,
            size=binsize
        )
    )

    trace_data = [trace1]
    layout = go.Layout(
        bargroupgap=0.3,
         title='The distribution of ' + nameOfFeature,
        xaxis=dict(
            title=nameOfFeature,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Number of labels',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
    fig = go.Figure(data=trace_data, layout=layout)
    py.iplot(fig)


# In[1]:


#Let's define a function in order to plot the features vs the target variable to identify the possible outliers
def plotBarCat(df,feature,target): 
    """
    Barplot function to plot any feature together with the target var
    """
    x0 = df[df[target]==0][feature]
    x1 = df[df[target]==1][feature]

    trace1 = go.Histogram(
        x=x0,color=['gold'],
        line=dict(color='#000000',width=1.5),
        opacity=0.75
    )
        
    
    trace2 = go.Histogram(
        x=x1,color=['gold'],
        line=dict(color='#000000',width=1.5),
        opacity=0.75
    )

    data = [trace1, trace2]
    layout = go.Layout(barmode='overlay',
                      title=feature,
                       yaxis=dict(title='Count'
        ))
    fig = go.Figure(data=data, layout=layout,color=['gold', 'lightskyblue'])

    py.iplot(fig, filename='overlaid histogram')
    
    # Let's define a function inside this plot function which will assess the skweness and kurtosis
    # of our variables
    
    def DescribeFloatSkewKurt(df,target):
        """
            A fundamental task in many statistical analyses is to characterize
            the location and variability of a data set. A further
            characterization of the data includes skewness and kurtosis.
            Skewness is a measure of symmetry, or more precisely, the lack
            of symmetry. A distribution, or data set, is symmetric if it
            looks the same to the left and right of the center point.
            Kurtosis is a measure of whether the data are heavy-tailed
            or light-tailed relative to a normal distribution. That is,
            data sets with high kurtosis tend to have heavy tails, or
            outliers. Data sets with low kurtosis tend to have light
            tails, or lack of outliers. A uniform distribution would
            be the extreme case
        """
        print('-*-'*25)
        print('\nMetrics for ' + str(target), ',the TARGET variable')
        mean_t = df[target].mean()
        std_t = df[target].std()
        
        skew_t = skew(df[target])
        kurt_t = kurtosis(df[target])
        
        print("{0} mean : ".format(target), round(mean_t,3))
        print("{0} std  : ".format(target),round(std_t,3) )
        print("{0} skew : ".format(target), round(skew_t, 3))
        print("{0} kurt : ".format(target), round(kurt_t,3))
        print('-*-'*25)
    
    DescribeFloatSkewKurt(df,target)


# In[ ]:


def PlotPie(df, feature, feature_labels, colors_list):
    """
    Pie Chart in order to represent the distribution of each category in each Variable.
    
    input: df--> dataframe
           feature --> string containing the name of the variable that we want to plot
           feature_labels -> feature labels
           colors_list --> list containing the colors that we want to give to the diff labels
        
    """
    # Let's create our Pie go 
    pie_obj = go.Pie(labels= df[feature_labels].unique(),values = df[feature])

    # Create the figure
    fig = go.Figure(pie_obj)

    # Styling 
    fig.update(layout_title_text= str(feature))
    fig.update_traces(textinfo='percent+label', textposition='inside', textfont_size=14,
                      marker=dict(colors=colors_list, line=dict(color='#000000', width=2)))
    fig.show()


# In[ ]:


def PlotPie2(df, nameOfFeature):
    """
    Pie Chart in order to represent the distribution of each category in each Variable
    """
    labels = [str(df[nameOfFeature].unique()[i]) for i in range(df[nameOfFeature].nunique())]
    values = [df[nameOfFeature].value_counts()[i] for i in range(df[nameOfFeature].nunique())]

    trace=go.Pie(labels=labels,values=values,marker=dict(colors=['lightskyblue','gold'], 
                           line=dict(color='#000000', width=1.5)))

    py.iplot([trace])


# In[ ]:


def OutLiersBox(df,nameOfFeature):
    """
    Function to create a BoxPlot and visualise:
    - All Points in the Variable
    - Whiskers in the Variable
    - Suspected Outliers in the variable

    """
    trace0 = go.Box(
        y = df[nameOfFeature],
        name = "All Points",
        jitter = 0.3,
        pointpos = -1.8,
        boxpoints = 'all', #define that we want to plot all points
        marker = dict(
            color = 'rgb(7,40,89)'),
        line = dict(
            color = 'rgb(7,40,89)')
    )

    trace1 = go.Box(
        y = df[nameOfFeature],
        name = "Only Whiskers",
        boxpoints = False, # no boxpoints, just the whiskers
        marker = dict(
            color = 'rgb(9,56,125)'),
        line = dict(
            color = 'rgb(9,56,125)')
    )

    trace2 = go.Box(
        y = df[nameOfFeature],
        name = "Suspected Outliers",
        boxpoints = 'suspectedoutliers', # define the suspected Outliers
        marker = dict(
            color = 'rgb(8,81,156)',
            outliercolor = 'rgba(219, 64, 82, 0.6)',
            line = dict(
                outliercolor = 'rgba(219, 64, 82, 0.6)',
                outlierwidth = 2)),
        line = dict(
            color = 'rgb(8,81,156)')
    )


    data = [trace0,trace1,trace2]

    layout = go.Layout(
        title = "{} Outliers".format(nameOfFeature)
    )

    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig, filename = "Outliers")
    


# In[ ]:


def IQR_Outliers(df_out,nameOfFeature,drop=False):

    valueOfFeature = df_out[nameOfFeature]
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(valueOfFeature, 25.)

    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(valueOfFeature, 75.)

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1)*1.5
    # print "Outlier step:", step
    outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()
    feature_outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].values
    # df[~((df[nameOfFeature] >= Q1 - step) & (df[nameOfFeature] <= Q3 + step))]


    # Remove the outliers, if any were specified
    print ("Number of outliers (inc duplicates): {} and outliers: {}".format(len(outliers), feature_outliers))
    if drop:
        good_data = df_out.drop(df_out.index[outliers]).reset_index(drop = True)
        print ("New dataset with removed outliers has {} samples with {} features each.".format(*good_data.shape))
        return good_data
    else: 
        print ("Nothing happens, df.shape = ",df_out.shape)
        return df_out


# In[ ]:


def HeatMap(df,x=True):
        correlations = df.corr()
        
        ## Create color map ranging between two colors
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',square=True, 
                          linewidths=.5, annot=x, cbar_kws={"shrink": .75})
        fig.set_xticklabels(fig.get_xticklabels(), rotation = 90, fontsize = 10)
        fig.set_yticklabels(fig.get_yticklabels(), rotation = 0, fontsize = 10)
        plt.tight_layout()
        plt.show()


# In[ ]:


def corrMatrix(df, color_map):
    """
    Function to create a Heatmap containing the corr coef of all paired NUMERICAL variables.
    Steps:
    1. Filter just the num_variables
    2. Calculate the corr coef of the num_variables
    3.Create the heatmap
    
    inputs: df-> dataframe
            color_map -> color scale
    
    
    """
    # 1. Filter the numerical variablesjust
    num_var = df.select_dtypes(include=['int', 'float']).columns
    data_num = df[num_var]
    
    # 2. Calculate the corr coef
    corrs = data_num.corr()
    
    # 3. Create the corr map with annotations
    figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.columns),
    annotation_text=corrs.round(2).values,
    showscale=True, colorscale=color_map)
    figure.update_yaxes(autorange="reversed")

    figure.show()


# In[ ]:


def corrCoef_Threshold(data, threshold):
    """
    Function aimed to calculate the corrCoef between each pair of variables
    input: data->dataframe
           threshold -> True: we want to keep the variables with a corrCoef higher than the income
                        False: we want to keep all values, no filtering
            """
    num_var = data.select_dtypes(include=['int64', 'float64']).columns
    print(num_var)
    data_num = data[num_var]
    data_corr = abs(data_num.corr())
    data_cols = data_corr.columns
    if threshold == True:
        data_corr= pd.DataFrame(data_corr.unstack().sort_values(ascending = False), columns = ['corrCoef'])
       # threshold that I want to select. I will keep the variables with a corrCoef higher than the threshols
        thr = float(input('Threshold? (in positive sign, please) '))
        data_corr = data_corr[(data_corr.corrCoef >thr)].unstack()
        data_corr = pd.DataFrame(data_corr)
        mask = np.zeros_like(data_corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # Create the plot
        plt.figure(figsize=(10,8))
        sns.heatmap(data_corr,xticklabels = data_cols,
                    yticklabels = data_cols,mask = mask,
                            annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7),
                   annot_kws={"size":10})
    else:
        plt.figure(figsize=(10,8))
        sns.heatmap(data_corr,
                    xticklabels = data_corr.columns.values,
                   yticklabels = data_corr.columns.values,
                   annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7),
                    annot_kws={"size":10})
    return data_corr


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




