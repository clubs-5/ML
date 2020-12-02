from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler,VectorIndexer,OneHotEncoder,StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.types import StructType,StructField, StringType, IntegerType

import pyarrow
import redis

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .getOrCreate()

    # 讀取已訓練好的 ALS 模型
    model = ALSModel.load('hdfs://master.tibame/user/clubs/spark_mllib_101/movies/movies_recommender/')

    # 推薦給所有使用者 5 部電影
    user_movies = model.recommendForAllUsers(5)
    #user_movies.show(100, truncate=False)
    #user_movies.printSchema()

    # recommendations 欄位只保留 movieId 資訊
    no_ratingsDF = user_movies.select('userId', 'recommendations.movieId')

    # 轉換成 pandas Dataframe
    pandadf = no_ratingsDF.toPandas()

    # 存回 hdfs 並覆寫
    pandadf.write.save('temp_po/recom_df', SaveMode='overwrite')

    # 準備 redis
    r = redis.Redis(host='master.tibame', port=6379, db=0)

    # 序列化 pandadf
    df_compressed = pa.serialize(pandadf).to_buffer().to_pybytes()

    # 建立 redis key 'recomm' 並寫入
    res = r.set('recomm',df_compressed)
