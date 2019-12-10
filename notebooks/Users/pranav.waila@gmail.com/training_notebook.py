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

# MAGIC %sql
# MAGIC select * from table_test

# COMMAND ----------

data = df.toPandas()

# COMMAND ----------

data['SoBookDate']= pd.to_datetime(data['SoBookDate'], errors = 'coerce')
data['CseVesselArrivalDate']= pd.to_datetime(data['CseVesselArrivalDate'], errors = 'coerce')


# COMMAND ----------

data['SOBD_CSEVAD_Difference'] = data['CseVesselArrivalDate'].astype(dt.timedelta).sub(data['SoBookDate'].astype(dt.timedelta), axis=0)
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
regr.fit(X_train, y_train)e

# COMMAND ----------

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# COMMAND ----------

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# COMMAND ----------

try:
  dbutils.fs.unmount(mount_point = "/mnt/modelloc")
except:
  print("in correct state")
  
try:
  dbutils.fs.mount(source = "wasbs://model@predictionblobstorage.blob.core.windows.net",mount_point = "/mnt/modelloc",extra_configs = {"fs.azure.account.key.predictionblobstorage.blob.core.windows.net":dbutils.secrets.get(scope = "datasciencesecret1", key = "JaiHanuman9")})
except:
  print("Mount problem")

# COMMAND ----------

from sklearn.externals import joblib
joblib.dump(regr,'/mnt/modelloc/model/model_joblib')

try:
  dbutils.fs.unmount(mount_point = "/mnt/modelloc")
except:
  print("in correct state")
