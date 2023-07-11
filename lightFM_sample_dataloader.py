import sys
import getpass
from pyspark.sql import SparkSession

def get_sample_data(spark,percentage,net_id):
#fetching the location from whole training datasets

    train_file =  f'/user/{net_id}/ListenBrainz/train_small_baseline.parquet'
    validation_file = f'/user/{net_id}/ListenBrainz/validation_small_baseline.parquet'
    #test_file=f'/user/{net_id}/ListenBrainz/test_baseline.parquet'

    #defining the schema
    train_data = spark.read.parquet(train_file, header=True)
    train_data.createOrReplaceTempView('train_data')
    validation_data = spark.read.parquet(train_file, header=True)
    validation_data.createOrReplaceTempView('validation_data')
    #test_data = spark.read.parquet(test_file, header=True)
    #test_data.createOrReplaceTempView('test_data')


    #taking the sample data for LightFM testing
    train_data = train_data.sample(withReplacement=False, fraction=percentage)
    validation_data = validation_data.sample(withReplacement=False, fraction=percentage)
    #test_data = test_data.sample(withReplacement=False, fraction=percentage)

    #df_test=f'/user/{net_id}/ListenBrainz/test_sample.parquet'
    df_train =  f'/user/{net_id}/ListenBrainz/train_sample.parquet'
    df_val = f'/user/{net_id}/ListenBrainz/validation_sample.parquet'

    train_data.coalesce(1).write.mode('overwrite').parquet(df_train)
    validation_data.coalesce(1).write.mode('overwrite').parquet(df_val)
    #test_data.coalesce(1).write.mode('overwrite').parquet(df_test)

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('Recommender-DataLoader-GRP7').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    host = "nyu-dataproc-m"
    netID = getpass.getuser()
    get_sample_data(spark,0.01,netID)
    spark.stop()
