
# Machine Learning with Spark

## Introduction

You've now explored how to perform operations on Spark RDDs for simple MapReduce tasks. Luckily, there are far more advanced use cases for Spark, and many of them are found in the `ml` library, which we are going to explore in this lesson.


## Objectives

You will be able to: 

- Load and manipulate data using Spark DataFrames  
- Define estimators and transformers in Spark ML 
- Create a Spark ML pipeline that transforms data and runs over a grid of hyperparameters 



## A Tale of Two Libraries

If you look at the PySpark documentation, you'll notice that there are two different libraries for machine learning, [mllib](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html) and [ml](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html). These libraries are extremely similar to one another, the only difference being that the `mllib` library is built upon the RDDs you just practiced using; whereas, the `ml` library is built on higher level Spark DataFrames, which has methods and attributes similar to pandas. Spark has stated that in the future, it is going to devote more effort to the `ml` library and that `mllib` will become deprecated. It's important to note that these libraries are much younger than pandas and scikit-learn and there are not as many features present in either.

## Spark DataFrames

In the previous lessons, you were introduced to SparkContext as the primary way to connect with a Spark Application. Here, we will be using SparkSession, which is from the [sql](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html) component of PySpark. The SparkSession acts the same way as SparkContext; it is a bridge between Python and the Spark Application. It's just built on top of the Spark SQL API, a higher-level API than RDDs. In fact, a SparkContext object is spun up around which the SparkSession object is wrapped. Let's go through the process of manipulating some data here. For this example, we're going to be using the [Forest Fire dataset](https://archive.ics.uci.edu/ml/datasets/Forest+Fires) from UCI, which contains data about the area burned by wildfires in the Northeast region of Portugal in relation to numerous other factors.

To begin with, let's create a SparkSession so that we can spin up our spark application. 


```python
# importing the necessary libraries
from pyspark import SparkContext
from pyspark.sql import SparkSession
# sc = SparkContext('local[*]')
# spark = SparkSession(sc)
```

To create a SparkSession: 


```python
spark = SparkSession.builder.master('local').getOrCreate()
```

Now, we'll load the data into a PySpark DataFrame: 


```python
## reading in pyspark df
spark_df = spark.read.csv('./forestfires.csv', header='true', inferSchema='true')

## observing the datatype of df
type(spark_df)
```

You'll notice that some of the methods are extremely similar or the same as those found within Pandas.


```python
spark_df.head()
```


```python
spark_df.columns
```

Selecting multiple columns is similar as well: 


```python
spark_df[['month','day','rain']]
```

But selecting one column is different. If you want to maintain the methods of a spark DataFrame, you should use the `.select()` method. If you want to just select the column, you can use the same method you would use in pandas (this is primarily what you would use if you're attempting to create a boolean mask). 


```python
d = spark_df.select('rain')
```


```python
spark_df['rain']
```

Let's take a look at all of our data types in this dataframe


```python
spark_df.dtypes
```

## Aggregations with our DataFrame

Let's investigate to see if there is any correlation between what month it is and the area of fire: 


```python
spark_df_months = spark_df.groupBy('month').agg({'area': 'mean'})
spark_df_months
```

Notice how the grouped DataFrame is not returned when you call the aggregation method. Remember, this is still Spark! The transformations and actions are kept separate so that it is easier to manage large quantities of data. You can perform the transformation by calling `.collect()`: 


```python
spark_df_months.collect()
```

As you can see, there seem to be larger area fires during what would be considered the summer months in Portugal. On your own, practice more aggregations and manipulations that you might be able to perform on this dataset. 

## Boolean Masking 

Boolean masking also works with PySpark DataFrames just like Pandas DataFrames, the only difference being that the `.filter()` method is used in PySpark. To try this out, let's compare the amount of fire in those areas with absolutely no rain to those areas that had rain.


```python
no_rain = spark_df.filter(spark_df['rain'] == 0.0)
some_rain = spark_df.filter(spark_df['rain'] > 0.0)
```

Now, to perform calculations to find the mean of a column, we'll have to import functions from `pyspark.sql`. As always, to read more about them, check out the [documentation](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions).


```python
from pyspark.sql.functions import mean

print('no rain fire area: ', no_rain.select(mean('area')).show(),'\n')

print('some rain fire area: ', some_rain.select(mean('area')).show(),'\n')
```

Yes there's definitely something there! Unsurprisingly, rain plays in a big factor in the spread of wildfire.

Let's obtain data from only the summer months in Portugal (June, July, and August). We can also do the same for the winter months in Portugal (December, January, February).


```python
summer_months = spark_df.filter(spark_df['month'].isin(['jun','jul','aug']))
winter_months = spark_df.filter(spark_df['month'].isin(['dec','jan','feb']))

print('summer months fire area', summer_months.select(mean('area')).show())
print('winter months fire areas', winter_months.select(mean('area')).show())
```

## Machine Learning

Now that we've performed some data manipulation and aggregation, lets get to the really cool stuff, machine learning! PySpark states that they've used scikit-learn as an inspiration for their implementation of a machine learning library. As a result, many of the methods and functionalities look similar, but there are some crucial distinctions. There are three main concepts found within the ML library:

`Transformer`: An algorithm that transforms one PySpark DataFrame into another DataFrame. 

`Estimator`: An algorithm that can be fit onto a PySpark DataFrame that can then be used as a Transformer. 

`Pipeline`: A pipeline very similar to an `sklearn` pipeline that chains together different actions.

The reasoning behind this separation of the fitting and transforming step is because Spark is lazily evaluated, so the 'fitting' of a model does not actually take place until the Transformation action is called. Let's examine what this actually looks like by performing a regression on the Forest Fire dataset. To start off with, we'll import the necessary libraries for our tasks.


```python
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import feature
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator
```

Looking at our data, one can see that all the categories are numerical except for day and month. We saw some correlation between the month and area burned in a fire, so we will include that in our model. The day of the week, however, is highly unlikely to have any effect on fire, so we will drop it from the DataFrame.


```python
fire_df = spark_df.drop('day')
fire_df.head()
```

In order for us to run our model, we need to turn the months variable into a dummy variable. In `ml` this is a 2-step process that first requires turning the categorical variable into a numerical index (`StringIndexer`). Only after the variable is an integer can PySpark create dummy variable columns related to each category (`OneHotEncoderEstimator`). Your key parameters when using these `ml` estimators are: `inputCol` (the column you want to change) and `outputCol` (where you will store the changed column). Here it is in action: 


```python
si = StringIndexer(inputCol='month', outputCol='month_num')
model = si.fit(fire_df)
new_df = model.transform(fire_df)
```

Note the small, but critical distinction between `sklearn`'s implementation of a transformer and PySpark's implementation. `sklearn` is more object oriented and Spark is more functional oriented.


```python
## this is an estimator (an untrained transformer)
type(si)
```


```python
## this is a transformer (a trained transformer)
type(model)
```


```python
model.labels
```


```python
new_df.head(4)
```

As you can see, we have created a new column called `'month_num'` that represents the month by a number. Now that we have performed this step, we can use Spark's version of `OneHotEncoder()` - `OneHotEncoderEstimator()`. Let's make sure we have an accurate representation of the months.


```python
new_df.select('month_num').distinct().collect()
```


```python
## fitting and transforming the OneHotEncoderEstimator
ohe = feature.OneHotEncoderEstimator(inputCols=['month_num'], outputCols=['month_vec'], dropLast=True)
one_hot_encoded = ohe.fit(new_df).transform(new_df)
one_hot_encoded.head()
```

Great, we now have a OneHotEncoded sparse vector in the `'month_vec'` column! Because Spark is optimized for big data, sparse vectors are used rather than entirely new columns for dummy variables because it is more space efficient. You can see in this first row of the DataFrame:  

`month_vec=SparseVector(11, {2: 1.0})` this indicates that we have a sparse vector of size 11 (because of the parameter `dropLast = True` in `OneHotEncoderEstimator()`) and this particular data point is the 2nd index of our month labels (march, based off the labels in the `model` StringEstimator transformer).  

The final requirement for all machine learning models in PySpark is to put all of the features of your model into one sparse vector. This is once again for efficiency sake. Here, we are doing that with the `VectorAssembler()` estimator.


```python
features = ['X',
 'Y',
 'FFMC',
 'DMC',
 'DC',
 'ISI',
 'temp',
 'RH',
 'wind',
 'rain',
 'month_vec']

target = 'area'

vector = VectorAssembler(inputCols=features, outputCol='features')
vectorized_df = vector.transform(one_hot_encoded)
```


```python
vectorized_df.head()
```

Great! We now have our data in a format that seems acceptable for the last step. It's time for us to actually fit our model to data! Let's fit a Random Forest Regression model to our data. Although there are still a bunch of other features in the DataFrame, it doesn't matter for the machine learning model API. All that needs to be specified are the names of the features column and the label column. 


```python
## instantiating and fitting the model
rf_model = RandomForestRegressor(featuresCol='features', 
                                 labelCol='area', predictionCol='prediction').fit(vectorized_df)
```


```python
rf_model.featureImportances
```


```python
## generating predictions
predictions = rf_model.transform(vectorized_df).select('area', 'prediction')
predictions.head(10)
```

Now we can evaluate how well the model performed using `RegressionEvaluator`.


```python
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='area')
```


```python
## evaluating r^2
evaluator.evaluate(predictions,{evaluator.metricName: 'r2'})
```


```python
## evaluating mean absolute error
evaluator.evaluate(predictions,{evaluator.metricName: 'mae'})
```

## Putting it all in a Pipeline

We just performed a whole lot of transformations to our data. Let's take a look at all the estimators we used to create this model:

* `StringIndexer()` 
* `OneHotEnconderEstimator()` 
* `VectorAssembler()` 
* `RandomForestRegressor()` 

Once we've fit our model in the Pipeline, we're then going to want to evaluate it to determine how well it performs. We can do this with:

* `RegressionEvaluator()` 

We can streamline all of these transformations to make it much more efficient by chaining them together in a pipeline. The Pipeline object expects a list of the estimators prior set to the parameter `stages`.


```python
# importing relevant libraries
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml import Pipeline
```


```python
## instantiating all necessary estimator objects

string_indexer = StringIndexer(inputCol='month', outputCol='month_num', handleInvalid='keep')
one_hot_encoder = OneHotEncoderEstimator(inputCols=['month_num'], outputCols=['month_vec'], dropLast=True)
vector_assember = VectorAssembler(inputCols=features, outputCol='features')
random_forest = RandomForestRegressor(featuresCol='features', labelCol='area')
stages = [string_indexer, one_hot_encoder, vector_assember, random_forest]

# instantiating the pipeline with all them estimator objects
pipeline = Pipeline(stages=stages)
```

### Cross-validation 

You might have missed a critical step in the random forest regression above; we did not cross validate or perform a train/test split! Now we're going to fix that by performing cross-validation and also testing out multiple different combinations of parameters in PySpark's `GridSearch()` equivalent. To begin with, we will create a parameter grid that contains the different parameters we want to use in our model.


```python
# creating parameter grid

params = ParamGridBuilder()\
          .addGrid(random_forest.maxDepth, [5, 10, 15])\ 
          .addGrid(random_forest.numTrees, [20 ,50, 100])\ 
          .build()
```

Let's take a look at the params variable we just built.


```python
print('total combinations of parameters: ', len(params))

params[0]
```

Now it's time to combine all the steps we've created to work in a single line of code with the `CrossValidator()` estimator.


```python
## instantiating the evaluator by which we will measure our model's performance
reg_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='area', metricName = 'mae')

## instantiating crossvalidator estimator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=reg_evaluator, parallelism=4)
```


```python
## fitting crossvalidator
cross_validated_model = cv.fit(fire_df)
```

Now, let's see how well the model performed! Let's take a look at the average performance for each one of our 9 models. It looks like the optimal performance is an MAE around 23. Note that this is worse than our original model, but that's because our original model had substantial data leakage. We didn't do a train-test split!


```python
cross_validated_model.avgMetrics
```

Now, let's take a look at the optimal parameters of our best performing model. The `cross_validated_model` variable is now saved as the best performing model from the grid search just performed. Let's look to see how well the predictions performed. As you can see, this dataset has a large number of areas of "0.0" burned. Perhaps, it would be better to investigate this problem as a classification task.


```python
predictions = cross_validated_model.transform(spark_df)
predictions.select('prediction', 'area').show(300)
```

Now let's go ahead and take a look at the feature importances of our random forest model. In order to do this, we need to unroll our pipeline to access the random forest model. Let's start by first checking out the `.bestModel` attribute of our `cross_validated_model`. 


```python
type(cross_validated_model.bestModel)
```

`ml` is treating the entire pipeline as the best performing model, so we need to go deeper into the pipeline to access the random forest model within it. Previously, we put the random forest model as the final "stage" in the stages variable list. Let's look at the `.stages` attribute of the `.bestModel`.


```python
cross_validated_model.bestModel.stages
```

Perfect! There's the RandomForestRegressionModel, represented by the last item in the stages list. Now, we should be able to access all the attributes of the random forest regressor.


```python
optimal_rf_model = cross_validated_model.bestModel.stages[3]
```


```python
optimal_rf_model.featureImportances
```


```python
optimal_rf_model.getNumTrees
```

## Summary

In this lesson, you learned about PySpark's DataFrames, machine learning models, and pipelines. With the use of a pipeline, you can train a huge number of models simultaneously, saving you a substantial amount of time and effort. Up next, you will have a chance to build a PySpark machine learning pipeline of your own with a classification problem!
