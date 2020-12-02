from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler,VectorIndexer,OneHotEncoder,StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.types import StructType,StructField, StringType, IntegerType

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .getOrCreate()

    model = ALSModel.load('hdfs://master.tibame/user/clubs/spark_mllib_101/movies/movies_recommender/')

    user_movies = model.recommendForAllUsers(5)
    user_movies.show(100, truncate=False)
    user_movies.printSchema()