
import findspark
findspark.init()
import json
from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression 
from pyspark.ml import Pipeline
import os
from pyspark.sql.types import StructType, StructField, DoubleType
import pandas as pd
spark = SparkSession.builder.appName('churn-prediction-api').getOrCreate()
#charger le modele a partir de dosser de serialisation (resultat de serialisation de phase de training )
model = PipelineModel.load("lrModel")
#developper l'api


app = Flask(__name__, template_folder=os.path.abspath('templates'))

@app.route('/')
def home():
    return render_template('index2.html')
#tester avec une intefrace web voir index2.html
@app.route('/predict2', methods=['POST'])
def predict2():
    # Get the input data as json
    json_data = request.get_json()
    data = json_data['data']
    # Get the data as a pandas dataframe
    data = pd.DataFrame.from_dict(data, orient='index').transpose()
    # Create a spark dataframe from the pandas dataframe
    spark_df = spark.createDataFrame(data)
    # Cast the string columns to double
    spark_df = spark_df.withColumn("Age", spark_df["Age"].cast(DoubleType()))
    spark_df = spark_df.withColumn("Total_Purchase", spark_df["Total_Purchase"].cast(DoubleType()))
    spark_df = spark_df.withColumn("Account_Manager", spark_df["Account_Manager"].cast(DoubleType()))
    spark_df = spark_df.withColumn("Years", spark_df["Years"].cast(DoubleType()))
    spark_df = spark_df.withColumn("Num_Sites", spark_df["Num_Sites"].cast(DoubleType()))
    # Make the prediction using the trained model
    prediction = model.transform(spark_df).head()
    app.logger.info('%s logged in successfully', prediction)
    # Convert the prediction result to a json string
    #result = {'prediction': str(prediction.prediction), 'probability': str(prediction.probability[1])}
    #result = {'prediction': float(prediction.prediction), 'probability': float(prediction.probability[1])}

    # Return the prediction result as a json response
    return jsonify(prediction=float(prediction.prediction), probability=float(prediction.probability[1]))


#tester avec postman/curl...
@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    if not json_data:
        return jsonify({'message': 'No input data provided'}), 400

    data = json_data['data']
    assembler = VectorAssembler(inputCols=['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites'], outputCol='features')
    log_reg = LogisticRegression(featuresCol='features', labelCol='Churn', maxIter=10)


    pipeline = Pipeline(stages=[assembler, log_reg])
    
    df = spark.createDataFrame([data])
    app.logger.info('%s logged in successfully', df)
    prediction = model.transform(df).head()
    result = {'prediction': str(prediction.prediction), 'probability': str(prediction.probability[1])}
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True, passthrough_errors=True,
    use_debugger=False, use_reloader=False,host='0.0.0.0', port=5000)
