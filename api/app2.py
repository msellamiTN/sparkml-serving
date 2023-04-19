import findspark
findspark.init()

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression 
from pyspark.ml import Pipeline
from prometheus_client import start_http_server, Gauge

spark = SparkSession.builder.appName('churn-prediction-api').getOrCreate()
# Charger le modele a partir de dosser de serialisation (resultat de serialisation de phase de training )
model = PipelineModel.load("lrModel")
# Developper l'api
app = Flask(__name__)

# Définir une métrique Prometheus
PREDICTION_ACCURACY = Gauge('prediction_accuracy', 'Accuracy of the churn prediction model')

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
    prediction = model.transform(df).head()
    result = {'prediction': str(prediction.prediction), 'probability': str(prediction.probability[1])}
    
    PREDICTION_ACCURACY.set(prediction.probability[1])
     
        
    return jsonify(result), 200

if __name__ == '__main__':
    # Démarrer le serveur Prometheus HTTP sur le port 8000
    start_http_server(8000)
    
    # Démarrer l'application Flask sur le port 5000
    app.run(host='0.0.0.0', port=5000)
