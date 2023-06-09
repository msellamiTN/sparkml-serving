# -*- coding: utf-8 -*-
"""Predicting Customer Churn Using Logistic Regression.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1plm-aGrS4U3E6TKcwiMhPa7mOKbjhyoU

### Prédire le taux de désabonnement des clients à l'aide de la régression logistique
Qu'est-ce que le taux de désabonnement ?
Le taux de désabonnement , également connu sous le nom de taux d' attrition ou de désabonnement des clients , est le taux auquel les clients cessent de faire affaire avec une entité. Il est le plus souvent exprimé en pourcentage d'abonnés au service qui mettent fin à leur abonnement au cours d'une période donnée. Un taux de désabonnement élevé indique que l'entreprise perd des clients à un rythme alarmant. L'attrition des clients peut être attribuée à une myriade de raisons et c'est l'entreprise qui doit découvrir ces raisons via des modèles et des tendances présents dans les données client.
Bus moderne i nesses emploient aujourd'hui des algorithmes complexes pour prédire les clients qui sont les plus susceptibles de désabonnement, à savoir se éloigner de la société. En utilisant de tels algorithmes, les entreprises peuvent connaître à l'avance les clients les plus susceptibles d'abandonner les services de l'entreprise et, par conséquent, proposer des stratégies de fidélisation de la clientèle pour atténuer les pertes auxquelles l'entreprise pourrait être confrontée.
"""
#!pip install pyspark
"""### Read the data file"""
import findspark
findspark.init()
from pyspark.sql import SparkSession
"""### Build the Pipeline"""
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
spark = SparkSession.builder.appName('churn_log_reg').getOrCreate()
# File location and type
file_location = "customer_churn-1.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

"""### Summary Statistics"""
#df.describe().toPandas().transpose()

assembler = VectorAssembler(inputCols = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites',], outputCol = 'features')
log_reg = LogisticRegression(featuresCol = 'features', labelCol = 'Churn', maxIter=10)
pipeline = Pipeline(stages = [assembler, log_reg])
"""#### Split DataSet"""
train, test = df.randomSplit([0.7, 0.3])
lrModel = pipeline.fit(train)
predictions = lrModel.transform(test)
predictions.printSchema()
"""### Make Predictions
- rawPrediction is equal (w*x + bias) variable coefficients values
- probability is 1/(1+e^(w*x + bias))
- prediction is 0 or 1.
"""
predictions.select('Churn', 'prediction', 'probability', 'rawPrediction').show(4)
"""### Performance Evaluation"""
from pyspark.ml.evaluation import BinaryClassificationEvaluator
eval = BinaryClassificationEvaluator(labelCol = 'Churn', rawPredictionCol = 'rawPrediction')
eval.evaluate(predictions)

#save the model to disk
#Save the model 
lrModel.write().overwrite().save("lrModel")

