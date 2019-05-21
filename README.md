
# Machine Learning with Spark

## Introduction

You've now explored how to perform operations on Spark RDDs for simple Map-Reduce tasks. Luckily, there are far more advanced use cases for spark, and many of the are found in the ml library, which we are going to explore in this lesson.


## Objectives
* Describe the use case for Machine Learning with Spark
* Load and manipulate data with Spark DataFrames
* Train a machine learning model with Spark


## A Tale of Two Libraries

If you look at the pyspark documentation, you'll notice that there are two different libraries for machine learning [mllib](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html) and [ml](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html). These libraries are extremely similar to one another, the only difference being that the mllib library is built upon the RDDs you just practiced using; whereas, the ml library is built on higher level Spark DataFrames, which has methods and attributes similar to pandas. It's important to note that these libraries are much younger than pandas and many of the kinks are still being worked out. 

## Spark DataFrames

In the previous lessons, you've been introduced to SparkContext as the primary way to connect with a Spark Application. Here, we will be using SparkSession, which is from the [sql](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html) component of pyspark. The SparkSession acts the same way as SparkContext; it a bridge between pyhon and the Spark Application. It's just built on top of the Spark SQL API, a higher-level API than RDDs. In fact, a SparkContext object is spun up around which the SparkSession object is wrapped. Let's go through the process of manipulating some data here. For this example, we're going to be using the [Forest Fire dataset](https://archive.ics.uci.edu/ml/datasets/Forest+Fires) from UCI, which contains data about the area burned by wildfires in the Northeast region of Portugal in relation to numerous other factors.

To begin with, let's create a SparkSession so that we can spin up our spark application.



```python
# importing the necessary libraries
from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext('local[*]')
spark = SparkSession(sc)
```

An alternate, one liner way to create a SparkSession is below


```python
# spark = SparkSession.builder.master("local").getOrCreate()
```

Now, we'll load the read in our data into the pyspark DataFrame object.


```python
## reading in pyspark df
spark_df = spark.read.csv('./forestfires.csv',header='true',inferSchema='true')

## observing the datatype of df
type(spark_df)
```




    pyspark.sql.dataframe.DataFrame



You'll notice that some of the methods are extremely similar or the same as those found within Pandas.


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



Selecting multiple columns is the same


```python
spark_df[['month','day','rain']]
```




    DataFrame[month: string, day: string, rain: double]



But selecting one column is different. If you want to mantain the methods of a spark DataFrame, you should use the `select` method. If you want to just select the column, you can use the same method you would use in pandas (this is primarily what you would use if you're attempting to create a Boolean mask).



```python
d = spark_df.select('rain')
```


```python
spark_df['rain']
```




    Column<b'rain'>



Let's take a look at all of our data types in this dataframe


```python
spark_df.dtypes
```




    [('X', 'int'),
     ('Y', 'int'),
     ('month', 'string'),
     ('day', 'string'),
     ('FFMC', 'double'),
     ('DMC', 'double'),
     ('DC', 'double'),
     ('ISI', 'double'),
     ('temp', 'double'),
     ('RH', 'int'),
     ('wind', 'double'),
     ('rain', 'double'),
     ('area', 'double')]



## Aggregations with our DataFrame

Let's investigate to see if there is any correlation between what month it is and the area of fire.


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

## Boolean Masking
Boolean masking also works with pyspark DataFrames just like Pandas DataFrames, the only difference being that the `filter` method is used in pyspark. To try this out, let's compare the amount the fire in those areas with absolutely no rain to those areas that had rain.


```python
no_rain = spark_df.filter(spark_df['rain'] == 0.0)
some_rain = spark_df.filter(spark_df['rain'] > 0.0)
```

Now, to perform calculations to find the mean of a column, we'll have to import functions from `pyspark.sql`. As always, to read more about them, check out the [documentation](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions).


```python
from pyspark.sql.functions import mean

print('no rain fire area: ',no_rain.select(mean('area')).show(),'\n')

print('some rain fire area: ',some_rain.select(mean('area')).show(),'\n')
```

    +------------------+
    |         avg(area)|
    +------------------+
    |13.023693516699408|
    +------------------+
    
    no rain fire area:  None 
    
    +---------+
    |avg(area)|
    +---------+
    |  1.62375|
    +---------+
    
    some rain fire area:  None 
    


Yes there's definitely something there! Unsurprisingly, rain plays in a big factor in the spread of wildfire.

Let's try and get obtain data from only the summer months in Portugal (June, July, and August). We can also do the same for the winter months in Portugal (December, January, February).


```python
summer_months = spark_df.filter(spark_df['month'].isin(['jun','jul','aug']))
winter_months = spark_df.filter(spark_df['month'].isin(['dec','jan','feb']))

print('summer months fire area', summer_months.select(mean('area')).show())
print('winter months fire areas', winter_months.select(mean('area')).show())
```

    +------------------+
    |         avg(area)|
    +------------------+
    |12.262317596566525|
    +------------------+
    
    summer months fire area None
    +-----------------+
    |        avg(area)|
    +-----------------+
    |7.918387096774193|
    +-----------------+
    
    winter months fire areas None


## Machine Learning

Now that we've performed some data manipulation and aggregation, lets get to the really cool stuff, machine learning! Pyspark states that they've used sklearn as an inspiration for their implementation of a machine learning library. As a result, many of the methods and functionalities look similar, but there are some crucial distinctions. There are three main concepts found within the ML library:

`Transformer`: An algorithm that transforms one pyspark DataFrame into another DataFrame. 

`Estimator`: An algorithm that can be fit onto a pyspark DataFrame that can then be used as a Transformer. 

`Pipeline`: A pipeline very similar to an sklearn pipeline that chains together different actions.

The reasoning behind this separation of the fitting and transforming step is because sklearn is lazily evaluated, so the 'fitting' of a model does not actually take place until the Transformation action is called. Let's examine what this actually looks like by performing a regression on the Forest Fire Dataset. To start off with, we'll import the necessary libraries for our tasks.


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




    Row(X=7, Y=5, month='mar', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0)



In order for us to run our model, we need to turn the months variable into a dummy variable. In `ml` this is a 2-step process that first requires turning the categorical variable into a numerical index (`StringIndexer`). Only after the variable is an int can pyspark create dummy variable columns related to each category (`OneHotEncoderEstimator`). You key parameters you when using these `ml` estimators are: inputCol (the column you want to change) and outputCol (where you will store the changed column). Here it is in action


```python
si = StringIndexer(inputCol='month',outputCol='month_num')
model = si.fit(fire_df)
new_df = model.transform(fire_df)
```

Note the small, but critical distinction between sklearn's implementation of a transformer and pyspark's implementation. sklearn is more object oriented and spark is more functionally based programming.


```python
## this is an estimator (an untrained transformer)
type(si)
```




    pyspark.ml.feature.StringIndexer




```python
## this is a transformer (a trained transformer)
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




    [Row(X=7, Y=5, month='mar', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_num=2.0),
     Row(X=7, Y=4, month='oct', FFMC=90.6, DMC=35.4, DC=669.1, ISI=6.7, temp=18.0, RH=33, wind=0.9, rain=0.0, area=0.0, month_num=6.0),
     Row(X=7, Y=4, month='oct', FFMC=90.6, DMC=43.7, DC=686.9, ISI=6.7, temp=14.6, RH=33, wind=1.3, rain=0.0, area=0.0, month_num=6.0),
     Row(X=8, Y=6, month='mar', FFMC=91.7, DMC=33.3, DC=77.5, ISI=9.0, temp=8.3, RH=97, wind=4.0, rain=0.2, area=0.0, month_num=2.0)]




```python
#
# new_df = new_df.drop('month')
new_df.head()
```




    Row(X=7, Y=5, month='mar', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_num=2.0)



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
## fitting and transforming the OneHotEncoder
ohe = feature.OneHotEncoderEstimator(inputCols=['month_num'],outputCols=['month_vec'],dropLast=True)
one_hot_encoded = ohe.fit(new_df).transform(new_df)
one_hot_encoded.head()
```




    Row(X=7, Y=5, month='mar', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_num=2.0, month_vec=SparseVector(11, {2: 1.0}))



Great, we now have a OneHotEncoded sparse vector in the month_vec column! Because spark is optimized for big data, sparse vectors are used rather than entirely new columns for dummy variables because it is more space efficient. You can see in this first row of the data frame:  
`month_vec=SparseVector(11, {2: 1.0})` this indicates that we have a sparse vector of size 11 (because of the parameter `dropLast = True` in OneHotEncoderEstimator) and this particular datapoint is the 2nd index of our month labels (march, based off the labels in the `model` StringEstimator transformer)  

The final requirement for all machine learning models in pyspark is to put all of the features of your model into one Sparse Vector. This is once again for efficiency sake. Here, we are doing that with the `VectorAssembler` estimator.


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

vector = VectorAssembler(inputCols=features,outputCol='features')
vectorized_df = vector.transform(one_hot_encoded)
```


```python
vectorized_df.head()
```




    Row(X=7, Y=5, month='mar', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_num=2.0, month_vec=SparseVector(11, {2: 1.0}), features=SparseVector(21, {0: 7.0, 1: 5.0, 2: 86.2, 3: 26.2, 4: 94.3, 5: 5.1, 6: 8.2, 7: 51.0, 8: 6.7, 12: 1.0}))



Great! We now have our data in a format that seems acceptable for the last step. Now it's time for us to actually fit our model to data! Let's try and fit a Random Forest Regression model our data. Although there are still a bunch of other features in the DataFrame, it doesn't matter for the machine learning model API. All that needs to be specified are the names of the features column and the label column. Let's fit use Random Forest Regression to fit the model here.


```python
## instantiating and fitting the model
rf_model = RandomForestRegressor(featuresCol='features',labelCol='area',predictionCol="prediction").fit(vectorized_df)
```


```python
rf_model.featureImportances
```




    SparseVector(21, {0: 0.113, 1: 0.0684, 2: 0.1352, 3: 0.067, 4: 0.1853, 5: 0.0664, 6: 0.1061, 7: 0.0982, 8: 0.0913, 9: 0.0, 10: 0.0291, 11: 0.0082, 12: 0.0, 13: 0.0313, 14: 0.0001, 15: 0.0001, 16: 0.0, 17: 0.0, 18: 0.0001, 20: 0.0001})




```python
## generating predictions
predictions = rf_model.transform(vectorized_df).select("area","prediction")
predictions.head(10)
```




    [Row(area=0.0, prediction=6.5717014062646255),
     Row(area=0.0, prediction=6.949198729352334),
     Row(area=0.0, prediction=6.150849710460925),
     Row(area=0.0, prediction=6.411877842228333),
     Row(area=0.0, prediction=10.033338730862948),
     Row(area=0.0, prediction=12.908861643565777),
     Row(area=0.0, prediction=42.5416935982501),
     Row(area=0.0, prediction=6.956267310579975),
     Row(area=0.0, prediction=7.337916350955155),
     Row(area=0.0, prediction=7.0016239464836705)]



Now we can evaluate how well the model performed using `RegressionEvaluator`.


```python
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='area')
```


```python
## evaluating r^2
evaluator.evaluate(predictions,{evaluator.metricName:"r2"})
```




    0.7324707925897679




```python
## evaluating mean absolute error
evaluator.evaluate(predictions,{evaluator.metricName:"mae"})
```




    13.57062828178921



## Putting it all in a Pipeline

We just performed a whole lot of transformations to our data. Let's take a look at all the estimators we used to create this model:

* StringIndexer
* OneHotEnconderEstimator
* VectorAssembler
* RandomForestRegressor

Once we've fit our model in the Pipeline, we're then going to want to evaluate it to determine how well it performs. We can do this with:

* RegressionEvaluator


We can streamline all of these transformations to make it all much more efficient by chaining them together in a pipeline. The Pipeline object expects a list of the estimators prior set to the parameter `stages`.


```python
# importing relevant libraries
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml import Pipeline
```


```python
## instantiating all necessary estimator objects

string_indexer = StringIndexer(inputCol='month',outputCol='month_num',handleInvalid='keep')
one_hot_encoder = OneHotEncoderEstimator(inputCols=['month_num'],outputCols=['month_vec'],dropLast=True)
vector_assember = VectorAssembler(inputCols=features,outputCol='features')
random_forest = RandomForestRegressor(featuresCol='features',labelCol='area')
stages =  [string_indexer, one_hot_encoder, vector_assember,random_forest]

# instantiating the pipeline with all them estimator objects
pipeline = Pipeline(stages=stages)
```

### Cross Validation
You might have missed a critical step in the Random Forest Regression above; we did not cross validate or perform a train/test split! Now we're going to fix that by performing cross validation and also testing out multiple different combinations of parameters in pyspark's GridSearch equivalent. To begin with, we will create a parameter grid that contains the different parameters we want to use in our model.


```python
# creating parameter grid

params = ParamGridBuilder()\
.addGrid(random_forest.maxDepth, [5,10,15])\
.addGrid(random_forest.numTrees, [20,50,100])\
.build()
```

Let's take a look at the params variable we just built.


```python
print('total combinations of parameters: ',len(params))

params[0]
```

    total combinations of parameters:  9





    {Param(parent='RandomForestRegressor_4db3a47921a1382190cd', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
     Param(parent='RandomForestRegressor_4db3a47921a1382190cd', name='numTrees', doc='Number of trees to train (>= 1).'): 20}



Now it's time to combine all the steps we've created to work in a single line of code with the CrossValidator estimator.


```python
## instantiating the evaluator by which we will measure our model's performance
reg_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='area',metricName = 'mae')
## instantiating crossvalidator estimator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params,evaluator=reg_evaluator,parallelism=4)
```


```python
## fitting crossvalidator
cross_validated_model = cv.fit(fire_df)
```

Now, let's see how well the model performed! Let's take a look at the average performance for each one of our 9 models. It looks like the optimal performance is an MAE around 23. Note that this is worse than our original model, but that's because our original model had substantial data leakage. We didn't do a train-test-split!


```python
cross_validated_model.avgMetrics
```




    [22.692019327283138,
     22.99290558885901,
     21.75767666249615,
     23.73460352694882,
     23.51470661266113,
     22.213816615833665,
     23.831395507010793,
     23.630131822962525,
     22.264771588623375]



Now, let's take a look at the optimal parameters of our best performing model. The cross_validated_model variable is now saved as the best performing model from the grid search just performed. Let's look to see how well the predictions performed. As you can see, this dataset has a large number of areas of "0.0" burned. Perhaps, it would be better to investigate this problem as classification task.


```python
predictions = cross_validated_model.transform(spark_df)
predictions.select('prediction','area').show(300)
```

    +------------------+-------+
    |        prediction|   area|
    +------------------+-------+
    | 7.100449250093918|    0.0|
    | 6.329025486243283|    0.0|
    |6.2857469188185915|    0.0|
    |7.5933126148034304|    0.0|
    | 6.231314264078399|    0.0|
    | 9.503843005958707|    0.0|
    |19.881573717656202|    0.0|
    | 7.280707629414105|    0.0|
    |11.119455564239933|    0.0|
    |17.149194738494558|    0.0|
    | 8.101320247937082|    0.0|
    |6.9044516704314844|    0.0|
    | 7.231903493447473|    0.0|
    |  9.86584096523932|    0.0|
    | 61.04180198158431|    0.0|
    | 8.488386677044286|    0.0|
    | 5.656956468779599|    0.0|
    | 7.901871325260197|    0.0|
    | 5.276142866785552|    0.0|
    | 6.345659968895647|    0.0|
    | 11.84113334516784|    0.0|
    | 5.072105370355027|    0.0|
    | 6.973167887886957|    0.0|
    |10.353204479628452|    0.0|
    |   7.4602566505675|    0.0|
    |6.5820864230510425|    0.0|
    |  8.17490046852117|    0.0|
    |10.626797338224044|    0.0|
    | 46.02630026242872|    0.0|
    |10.166730040391695|    0.0|
    |27.146709209437546|    0.0|
    | 6.966799072311349|    0.0|
    |  8.11626307208421|    0.0|
    | 5.439542310384569|    0.0|
    | 4.895391939686603|    0.0|
    | 6.696759050770261|    0.0|
    | 7.528165232070547|    0.0|
    | 5.822227886091439|    0.0|
    | 6.678710357495442|    0.0|
    | 5.425840893851139|    0.0|
    | 25.86721944448688|    0.0|
    | 6.249452574077862|    0.0|
    | 4.622097972727356|    0.0|
    | 7.362902228635849|    0.0|
    |  6.66240602247443|    0.0|
    |  71.7146308097191|    0.0|
    | 8.693017713265647|    0.0|
    | 5.073899176531336|    0.0|
    | 4.953363363094199|    0.0|
    | 6.921221602310698|    0.0|
    |11.303926049936212|    0.0|
    | 5.013165968777228|    0.0|
    | 4.295139851766004|    0.0|
    | 4.295139851766004|    0.0|
    | 4.690273155419559|    0.0|
    |10.306154824553344|    0.0|
    | 6.019049848237943|    0.0|
    |  5.12536545949031|    0.0|
    | 4.681847331737386|    0.0|
    | 4.624252277416517|    0.0|
    |3.9281747151294595|    0.0|
    | 4.337524692170501|    0.0|
    | 7.485345924938847|    0.0|
    | 4.654086698715685|    0.0|
    | 6.975636001902659|    0.0|
    |10.810096178532097|    0.0|
    | 8.450780769387558|    0.0|
    |10.171902961640196|    0.0|
    |10.420802192873852|    0.0|
    | 4.842594691665265|    0.0|
    |5.1703734220704805|    0.0|
    | 5.210535855821089|    0.0|
    | 5.132862706879212|    0.0|
    |10.622431695425567|    0.0|
    | 6.091782183586295|    0.0|
    | 8.674669061166123|    0.0|
    | 9.992036122483356|    0.0|
    | 6.660357382122063|    0.0|
    |4.7778828900924015|    0.0|
    | 23.83541509451415|    0.0|
    |10.803620210991928|    0.0|
    | 7.649063837483826|    0.0|
    |  7.13882987134132|    0.0|
    | 4.578566280371522|    0.0|
    | 8.488069282814102|    0.0|
    |10.822049401222635|    0.0|
    | 6.260331712133749|    0.0|
    | 10.93753868563672|    0.0|
    |14.625896360027737|    0.0|
    | 4.960481193629052|    0.0|
    | 5.234556043144365|    0.0|
    | 6.012507827448409|    0.0|
    | 14.46485350457554|    0.0|
    |15.344293467975078|    0.0|
    |20.058493300653808|    0.0|
    | 5.219983552770337|    0.0|
    | 4.704232141356683|    0.0|
    | 5.077804514360988|    0.0|
    |5.6909884374959345|    0.0|
    | 5.746659119994579|    0.0|
    | 5.746659119994579|    0.0|
    |7.4733942222324465|    0.0|
    | 3.996603387434346|    0.0|
    |22.552664610779562|    0.0|
    | 5.050599000210284|    0.0|
    | 6.115326725696012|    0.0|
    | 6.244690839684776|    0.0|
    |7.7513838701703675|    0.0|
    | 8.230000454547685|    0.0|
    | 5.944141774815046|    0.0|
    | 6.713595070528162|    0.0|
    | 4.103155764199948|    0.0|
    | 7.523330692569596|    0.0|
    | 5.144767698606991|    0.0|
    | 5.659863096226432|    0.0|
    |  5.76548226576243|    0.0|
    |5.5392157362960255|    0.0|
    | 6.040311081970276|    0.0|
    |  4.72649461635972|    0.0|
    | 5.301184872773998|    0.0|
    | 5.318656115793643|    0.0|
    | 5.715018092698078|    0.0|
    |7.9849859659504405|    0.0|
    | 7.053606235932927|    0.0|
    | 5.535263327886152|    0.0|
    | 7.807145478337959|    0.0|
    | 4.735761777961728|    0.0|
    |10.018259821050323|    0.0|
    | 5.863230321256752|    0.0|
    | 5.596620550991642|    0.0|
    |5.5036613272246795|    0.0|
    | 5.102400635625883|    0.0|
    |5.1237995876124645|    0.0|
    |6.4532531005306994|    0.0|
    | 4.385714296025399|    0.0|
    |  6.05565321865615|    0.0|
    |11.196262408305747|    0.0|
    | 8.679299884922493|    0.0|
    |17.934301181697524|   0.36|
    |15.433987202813485|   0.43|
    |10.841839635553495|   0.47|
    |4.2198062116335056|   0.55|
    | 20.99473462283787|   0.61|
    | 7.294185490103348|   0.71|
    | 4.095360566548178|   0.77|
    |27.305013521916916|    0.9|
    | 5.469318979493656|   0.95|
    |20.345682705054536|   0.96|
    | 4.629386217314927|   1.07|
    | 8.662877287211932|   1.12|
    | 7.439927658772081|   1.19|
    |18.929286735644542|   1.36|
    | 6.842586324268109|   1.43|
    | 6.163413376077684|   1.46|
    | 21.46212110317302|   1.46|
    |   5.1681881718295|   1.56|
    |10.662993731867093|   1.61|
    |  9.31579535766795|   1.63|
    | 3.379898119953403|   1.64|
    |  8.17490046852117|   1.69|
    | 5.856321087053046|   1.75|
    | 6.191359645523288|    1.9|
    | 7.568894151879123|   1.94|
    |25.595047448712002|   1.95|
    |  5.84283860107011|   2.01|
    | 5.869555520492934|   2.14|
    | 4.136385175241743|   2.29|
    | 5.437810436646471|   2.51|
    | 9.940116314151105|   2.53|
    | 7.271272948228257|   2.55|
    | 6.939392912214788|   2.57|
    | 6.766924331722581|   2.69|
    |  8.85082758507536|   2.74|
    | 9.751879838752394|   3.07|
    |3.5086610766638064|    3.5|
    | 4.408980376906173|   4.53|
    | 6.585652690023418|   4.61|
    | 4.957562328328267|   4.69|
    | 6.171828373449757|   4.88|
    |12.373413928353795|   5.23|
    | 12.12622747582432|   5.33|
    | 12.37366235958869|   5.44|
    |  5.53466563773221|   6.38|
    | 9.028525286293604|   6.83|
    | 8.612386584499419|   6.96|
    | 9.806212812478254|   7.04|
    |12.390469487091352|   7.19|
    |13.324380266829241|    7.3|
    | 5.240635580444434|    7.4|
    | 6.340684832051448|   8.24|
    |5.5031006637662765|   8.31|
    | 6.756284512885327|   8.68|
    | 7.645539057608801|   8.71|
    |19.756355723927673|   9.41|
    | 7.645539057608801|  10.01|
    | 5.212734449114215|  10.02|
    | 6.585652690023418|  10.93|
    |11.777726750419307|  11.06|
    | 8.942723220755147|  11.24|
    |11.345110974082006|  11.32|
    |15.226481557905323|  11.53|
    | 5.431429456061093|   12.1|
    | 5.739615669445284|  13.05|
    |13.093951191212843|   13.7|
    | 5.814064172268612|  13.99|
    | 7.584489108710756|  14.57|
    |7.4416040578797595|  15.45|
    | 11.64524899421403|   17.2|
    |7.5848814205733746|  19.23|
    |10.387748600549052|  23.41|
    | 8.238152214427172|  24.23|
    |10.616851167764505|   26.0|
    | 7.269807408346524|  26.13|
    | 8.175716130414095|  27.35|
    | 6.830144118906803|  28.66|
    | 6.830144118906803|  28.66|
    |23.029939442366512|  29.48|
    | 8.786069532360905|  30.32|
    |14.289718158048409|  31.72|
    | 6.850589248832495|  31.86|
    | 7.758531072632805|  32.07|
    | 9.959502749713508|  35.88|
    | 6.924943465454832|  36.85|
    | 22.39254234007707|  37.02|
    | 7.925324567376545|  37.71|
    |11.127485404446151|  48.55|
    |11.221157104646476|  49.37|
    | 16.43948467948461|   58.3|
    | 53.84375521892401|   64.1|
    | 17.74557166391077|   71.3|
    |40.071509981384246|  88.49|
    |  45.6102528576853|  95.18|
    |14.855479084883378| 103.39|
    | 48.80799512404707| 105.66|
    |106.83450531242131| 154.88|
    | 38.25980009365937| 196.48|
    | 83.27985573796573| 200.94|
    | 74.23322506406934| 212.88|
    |   657.76726732383|1090.84|
    | 5.542981032713149|    0.0|
    | 7.163684724339429|    0.0|
    | 6.292692013514795|    0.0|
    | 5.239774830888478|  10.13|
    |  5.83210518582317|    0.0|
    | 6.453796574061019|   2.87|
    |5.4087952821167615|   0.76|
    |4.6994772084564636|   0.09|
    | 3.850420576745567|   0.75|
    |14.110401897959514|    0.0|
    | 4.819956609218735|   2.47|
    |40.342063092364725|   0.68|
    | 6.102676234277047|   0.24|
    | 4.637275714163025|   0.21|
    | 5.103375716298627|   1.52|
    | 9.750140839293481|  10.34|
    | 7.245829418006586|    0.0|
    | 7.519133980485911|   8.02|
    |  5.38972594470606|   0.68|
    | 5.700892589137636|    0.0|
    | 5.147909155009395|   1.38|
    | 6.313139433646153|   8.85|
    | 5.109769263356273|    3.3|
    | 4.483783351024762|   4.25|
    | 7.681293291795142|   1.56|
    | 5.744841137739331|   6.54|
    | 4.836702503626425|   0.79|
    | 6.589432940374435|   0.17|
    | 6.954206755537839|    0.0|
    | 4.390947168363254|    0.0|
    |5.0885856686183075|    4.4|
    | 6.608697781554784|   0.52|
    | 9.903069776402774|   9.27|
    |4.8023728267909265|   3.09|
    | 8.195073858888364|   8.98|
    |  12.6525370887747|  11.19|
    |6.7447992152295715|   5.38|
    |11.074900373878663|  17.85|
    | 10.31350043964992|  10.73|
    |11.074900373878663|  22.03|
    |11.074900373878663|   9.77|
    | 6.683380341603197|   9.27|
    | 11.18096893695042|  24.77|
    | 5.764024133872799|    0.0|
    | 4.891859967125706|    1.1|
    | 7.878436496951975|  24.24|
    | 7.135926606990766|    0.0|
    |13.989644112719969|    0.0|
    | 7.257510879393076|    0.0|
    |7.0996295257062005|    0.0|
    | 7.061818014914833|    0.0|
    | 7.457484128295864|    0.0|
    |24.660424697751186|    8.0|
    | 4.877961110008713|   2.64|
    |  47.5685164058837|  86.45|
    | 6.721145734191943|   6.57|
    |11.805897965237282|    0.0|
    | 5.207581444068171|    0.9|
    | 6.905369708461223|    0.0|
    |16.990850143370483|    0.0|
    | 5.222841507638474|    0.0|
    +------------------+-------+
    only showing top 300 rows
    


Now let's go ahead and take a look at the feature importances of our Random Forest model. In order to do this, we need to unroll our pipeline to access the Random Forest Model. Let's start by first checking out the "bestModel" attribute of our cross_validated_model.


```python
type(cross_validated_model.bestModel)
```




    pyspark.ml.pipeline.PipelineModel



`ml` is treating the entire pipeline as the best performing model, so we need to go deeper into the pipeline to access the Random Forest model within it. Previously, we put the Random Forest Model as the final "stage" in the stages variable list. Let's look at the stages attribute of the bestModel.


```python
cross_validated_model.bestModel.stages
```




    [StringIndexer_419282fc8a3fd67df301,
     OneHotEncoderEstimator_4546abf1bf645cc67301,
     VectorAssembler_4bbd95b5d9e42fccb7e0,
     RandomForestRegressionModel (uid=RandomForestRegressor_4db3a47921a1382190cd) with 100 trees]



Perfect! There's the RandomForestRegressionModel, represented by the last item in the stages list. Now, we should be able to access all the attributes of the Random Forest Regressor.


```python
optimal_rf_model = cross_validated_model.bestModel.stages[3]
```


```python
optimal_rf_model.featureImportances
```




    SparseVector(22, {0: 0.086, 1: 0.0846, 2: 0.119, 3: 0.1334, 4: 0.128, 5: 0.0768, 6: 0.1148, 7: 0.0941, 8: 0.1085, 9: 0.0, 10: 0.0068, 11: 0.036, 12: 0.0002, 13: 0.0097, 14: 0.0003, 15: 0.0004, 16: 0.0001, 17: 0.0007, 18: 0.0001, 20: 0.0006})




```python
optimal_rf_model.getNumTrees
```




    100



## Summary

In this lesson, you have learned about pyspark's DataFrames, machine learning models, and pipelines. With the use of a pipeline, you can train a huge number of models simultaneously, saving you a substantial amount of time and effort. Up next, you will have a chance to build a pyspark machine learning pipeline of your own with a classification problem!
