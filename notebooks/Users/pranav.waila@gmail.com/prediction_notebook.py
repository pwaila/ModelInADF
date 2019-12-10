# Databricks notebook source
from pyspark import SparkContext, SparkConf, SQLContext
import pandas as pd
from pyspark.sql.types import *
import datetime
from pyspark.sql import SparkSession 

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

jdbcHostname= "predictiondb.database.windows.net"
jdbcPort = "1433"
dwJdbcExraOptions = "encryt=true:trustServerCertificate=true;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"
jdbcDatabase = "testsql"
properties = { "user": "adm","password":"#JaiHanuman9"} 

# COMMAND ----------

url= "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname, jdbcPort, jdbcDatabase)

# COMMAND ----------

def connect_to_sql(spark, jdbc_hostname, jdbc_port, database, data_table, username, password):
    jdbc_url = "jdbc:sqlserver://{0}:{1}/{2}".format(jdbc_hostname, jdbc_port, database)
    connection_details = {
        "user": username,
        "password": password,
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
    }
    df = spark.read.jdbc(url=jdbc_url, table=data_table, properties=connection_details)
    return df

# COMMAND ----------

connection_details = {
        "user": "adm@predictiondb",
        "password": "#JaiHanuman9",
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
    }

df = spark.read.jdbc(url=url, table="prediction.dataset", properties=connection_details)

# COMMAND ----------

df.registerTempTable("table_test")

# COMMAND ----------

data = df.toPandas()

# COMMAND ----------

data['SoBookDate']= pd.to_datetime(data['SoBookDate'], errors = 'coerce')
data['CseVesselArrivalDate']= pd.to_datetime(data['CseVesselArrivalDate'], errors = 'coerce')

# COMMAND ----------

data['SOBD_CSEVAD_Difference'] = data['CseVesselArrivalDate'].sub(data['SoBookDate'], axis=0)
data['SOBD_CSEVAD_Difference'] = data['SOBD_CSEVAD_Difference'] / np.timedelta64(1, 'D')

# COMMAND ----------

d=data[['SOBD_CSEVAD_Difference','SoPOR','SoPDel']].dropna()

# COMMAND ----------

from sklearn import preprocessing

# COMMAND ----------

le = preprocessing.LabelEncoder()
le.fit(d['SoPOR'])
d['SoPOR']=le.transform(d['SoPOR'])

# COMMAND ----------

le1 = preprocessing.LabelEncoder()
le1.fit(d['SoPDel'])
d['SoPDel']=le1.transform(d['SoPDel'])

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(d[['SoPOR','SoPDel']], d['SOBD_CSEVAD_Difference'], test_size=0.33, random_state=42)

# COMMAND ----------

# Create linear regression object
regr = RandomForestRegressor(max_depth=10)

# Train the model using the training sets
regr.fit(X_train, y_train)

# COMMAND ----------

# Make predictions using the testing set
y_pred = regr.predict(d[['SoPOR','SoPDel']])

# COMMAND ----------

import pyspark
from pyspark.sql.functions import *
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.ml.feature import *


# COMMAND ----------

dbutils.fs.mkdirs('/mnt/mountdatalake')


configs = {"dfs.adls.oauth2.access.token.provider.type": "ClientCredential",
           "dfs.adls.oauth2.client.id": "0426b48d-5917-4f69-b8e4-337a217567d2",
           "dfs.adls.oauth2.credential": "Wj]811aurp4uWAmnwPpj-lRo[i@Fz1UE",
           "dfs.adls.oauth2.refresh.url": "https://login.microsoftonline.com/630425d1-6e3a-4d09-b20c-1ba934be0135/oauth2/token"}

try:
  dbutils.fs.unmount("/mnt/mountdatalake")
except:
  print('was not already mounted')

dbutils.fs.mount(source = 'adl://etapredictionstorage.azuredatalakestore.net/etaprediction',mount_point = '/mnt/mountdatalake',extra_configs = configs)



# COMMAND ----------

from joblib import dump
import pickle


#dump(regr, "model.joblib")

print('Export the model to model.pkl')
f = open('fwrk.pkl', 'wb')
pickle.dump(y_pred, f)
f.close()



dbutils.fs.rm( "/mnt/mountdatalake/output/output.csv", recurse=True)

dbutils.fs.put("/mnt/mountdatalake/output/output.csv", str(regr))