
# Machine Learning with Spark

## Introduction

You've now explored how to perform operations on Spark RDDs for simple Map-Reduce tasks. Luckily, there are far more advanced use cases for spark, and many of the are found in the ml library, which we are going to explore in this lesson.


## Objectives
* Describe the use case for Machine Learning with Spark
* Load data with Spark DataFrames
* Train a machine learning model with Spark


## A Tale of Two Libraries

If you look at the pyspark documentation, you'll notice that there are two different libraries for machine learning [mllib](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html) and [ml](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html). These libraries are extremely similar to one another, the only difference being that the mllib library is built upon the RDDs you just practiced using; whereas, the ml library is built on higher level Spark DataFrames, which has methods and attributes very similar to pandas. It's important to note that these libraries are much younger than pandas and many of the kinks are still being worked out. 

## Spark DataFrames

In the previous lessons, you've been introduced to SparkContext as the primary way to connect with a Spark Application. Here, we will be using SparkSession, which is from the [sql](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html) component of pyspark. Let's go through the process of manipulating some data here. For this example, we're going to be using the [Forest Fire dataset](https://archive.ics.uci.edu/ml/datasets/Forest+Fires) from UCI, which contains data about the area burned by wildfires in the Northeast region of Portugal in relation to numerous other factors.



```python
from pyspark import SparkContext
from pyspark.sql import SparkSession
```


```python
spark = SparkSession.builder.master("local").appName("machine learning").getOrCreate()
```


```python
spark_df = spark.read.csv('./forestfires.csv',header='true',inferSchema='true')
```


```python
## observing the datatype of df
type(spark_df)
```




    pyspark.sql.dataframe.DataFrame



You'll notice that some of the methods are extremely similar or the same as those found within Pandas:



```python
spark_df.head()
```




    Row(X=7, Y=5, month='mar', day='fri', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0)




```python
spark_df.columns
```




    ['X',
     'Y',
     'month',
     'day',
     'FFMC',
     'DMC',
     'DC',
     'ISI',
     'temp',
     'RH',
     'wind',
     'rain',
     'area']



Selecting columns is the same


```python
spark_df[['month','day','rain']].head()
```




    Row(month='mar', day='fri', rain=0.0)



But others not so much...


```python
spark_df.info()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-37-3951e8b2005f> in <module>()
    ----> 1 spark_df.info()
    

    ~/anaconda3/lib/python3.6/site-packages/pyspark/sql/dataframe.py in __getattr__(self, name)
       1180         if name not in self.columns:
       1181             raise AttributeError(
    -> 1182                 "'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
       1183         jc = self._jdf.apply(name)
       1184         return Column(jc)


    AttributeError: 'DataFrame' object has no attribute 'info'



```python
## this is better
spark_df.describe()
```




    DataFrame[summary: string, X: string, Y: string, month: string, day: string, FFMC: string, DMC: string, DC: string, ISI: string, temp: string, RH: string, wind: string, rain: string, area: string]



## Let's try some aggregations with our DataFrame


```python
spark_df_months = spark_df.groupBy('month').agg({'area':'mean'})
spark_df_months
```




    DataFrame[month: string, avg(area): double]



Notice how the grouped DataFrame is not returned when you call the aggregation method. Remember, this is still Spark! The transformations and actions are kept separate so that it is easier to manage large quantities of data. You can perform the transformation by making a `collect` method call.


```python
spark_df_months.collect()
```




    [Row(month='jun', avg(area)=5.841176470588234),
     Row(month='aug', avg(area)=12.489076086956521),
     Row(month='may', avg(area)=19.24),
     Row(month='feb', avg(area)=6.275),
     Row(month='sep', avg(area)=17.942616279069753),
     Row(month='mar', avg(area)=4.356666666666667),
     Row(month='oct', avg(area)=6.638),
     Row(month='jul', avg(area)=14.3696875),
     Row(month='nov', avg(area)=0.0),
     Row(month='apr', avg(area)=8.891111111111112),
     Row(month='dec', avg(area)=13.33),
     Row(month='jan', avg(area)=0.0)]



As you can see, there seem to be larger area fires during what would be considered the summer months in Portugal. On your own, practice more aggregations and manipualtions that you might be able to perform on this dataset. Now, we'll move on to using the machine learning applications of pyspark. 

### ML

Pyspark openly admits that they used sklearn as an inspiration for their implementation of a machine learning library. As a result, many of the methods and functionalities look similar, but there are some crucial distinctions. There are four main concepts found within the ML library:

`Transformer`: An algorithm that transforms one pyspark DataFrame into another DataFrame. 

`Estimator`: An algorithm that can be fit onto a pyspark DataFrame that can then be used as a Transformer. 

`Pipeline`: A pipeline very similar to an sklearn pipeline that chains together different actions.

The reasoning behind this separation of the fitting and transforming step is because sklearn is lazily evaluated, so the 'fitting' of a model does not actually take place until the Transformation action is called.


```python
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import feature
from pyspark.ml.feature import StringIndexer, VectorAssembler
```


```python
spark
```





            <div>
                <p><b>SparkSession - in-memory</b></p>
                
        <div>
            <p><b>SparkContext</b></p>

            <p><a href="http://10.128.106.158:4040">Spark UI</a></p>

            <dl>
              <dt>Version</dt>
                <dd><code>v2.3.1</code></dd>
              <dt>Master</dt>
                <dd><code>local</code></dd>
              <dt>AppName</dt>
                <dd><code>machine learning</code></dd>
            </dl>
        </div>
        
            </div>
        




```python
si = StringIndexer(inputCol='month',outputCol='month_num')
model = si.fit(spark_df)
new_df = model.transform(spark_df)
```

Note the small, but critical distinction between sklearn's implementation of a transformer and pyspark's implementation. sklearn is more object oriented and spark is more functionally based programming


```python
type(si)
```




    pyspark.ml.feature.StringIndexer




```python
type(model)
```




    pyspark.ml.feature.StringIndexerModel




```python
model.labels
```




    ['aug',
     'sep',
     'mar',
     'jul',
     'feb',
     'jun',
     'oct',
     'apr',
     'dec',
     'jan',
     'may',
     'nov']




```python
new_df.head(4)
```




    [Row(X=7, Y=5, month='mar', day='fri', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_num=2.0),
     Row(X=7, Y=4, month='oct', day='tue', FFMC=90.6, DMC=35.4, DC=669.1, ISI=6.7, temp=18.0, RH=33, wind=0.9, rain=0.0, area=0.0, month_num=6.0),
     Row(X=7, Y=4, month='oct', day='sat', FFMC=90.6, DMC=43.7, DC=686.9, ISI=6.7, temp=14.6, RH=33, wind=1.3, rain=0.0, area=0.0, month_num=6.0),
     Row(X=8, Y=6, month='mar', day='fri', FFMC=91.7, DMC=33.3, DC=77.5, ISI=9.0, temp=8.3, RH=97, wind=4.0, rain=0.2, area=0.0, month_num=2.0)]



Let's go ahead and remove the day column, as there is almost certainly no correlation between day of the week and areas burned with forest fires.


```python
new_df = new_df.drop('day','month')
new_df.head()
```




    Row(X=7, Y=5, FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_num=2.0)



As you can see, we have created a new column called "month_num" that represents the month by a number. Now that we have performed this step, we can use Spark's version of OneHotEncoder. Let's make sure we have an accurate representation of the months.


```python
new_df.select('month_num').distinct().collect()
```




    [Row(month_num=8.0),
     Row(month_num=0.0),
     Row(month_num=7.0),
     Row(month_num=1.0),
     Row(month_num=4.0),
     Row(month_num=11.0),
     Row(month_num=3.0),
     Row(month_num=2.0),
     Row(month_num=10.0),
     Row(month_num=6.0),
     Row(month_num=5.0),
     Row(month_num=9.0)]




```python
ohe = feature.OneHotEncoderEstimator(inputCols=['month_num'],outputCols=['month_vec'])
```


```python
one_hot_encoded = ohe.fit(new_df).transform(new_df).drop('month_num')
one_hot_encoded.head()
```




    Row(X=7, Y=5, FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_vec=SparseVector(11, {2: 1.0}))




```python
features = ['X',
 'Y',
 'month_vec',
 'FFMC',
 'DMC',
 'DC',
 'ISI',
 'temp',
 'RH',
 'wind',
 'rain',]

target = 'area'

vector = VectorAssembler(inputCols=features,outputCol='features')
vectorized_df = vector.transform(one_hot_encoded)
```


```python
vectorized_df.head()
```




    Row(X=7, Y=5, FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_vec=SparseVector(11, {2: 1.0}), features=SparseVector(21, {0: 7.0, 1: 5.0, 4: 1.0, 13: 86.2, 14: 26.2, 15: 94.3, 16: 5.1, 17: 8.2, 18: 51.0, 19: 6.7}))



Great! We now have our data in a format that seems acceptable for the last step. Now it's time for us to actually fit our model to data! Let's try and fit a Random Forest Regression model our data.


```python
rf_model = RandomForestRegressor(featuresCol='features',labelCol='area',predictionCol="prediction").fit(vectorized_df)
```


```python
predictions = rf_model.transform(vectorized_df).select("area","prediction")
```


```python
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='area')
```


```python
evaluator.evaluate(predictions,{evaluator.metricName:"r2"})
```




    0.5992278842231795




```python
evaluator.evaluate(predictions,{evaluator.metricName:"mae"})
```




    13.753561104153286



### Putting it all in a Pipeline

We just performed a whole lot of transformations to our data, and we can streamline the process to make it much more efficient let's look at how we could take our previous code and combine it to form a pipeline. Let's take a look at all the Esimators we used to create this model:

* StringIndexer
* OneHotEnconderEstimator
* VectorAssembler
* RandomForestRegressor

Let's also be good data scientists and add in a step to the pipeline where we perform a Train-Test-Split


```python
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator

```


```python
ParamGridBuilder().addGrid()

```




    <pyspark.ml.tuning.ParamGridBuilder at 0x1562684e0>


