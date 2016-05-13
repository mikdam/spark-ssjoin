import os
import sys


# Set the path for spark installation
# this is the path where you have built spark using sbt/sbt assembly
os.environ['SPARK_HOME'] = '/home/mikdam/spark-1.6.0-bin-hadoop2.6'

# Append to PYTHONPATH so that pyspark could be found
sys.path.append('/home/mikdam/spark-1.6.0-bin-hadoop2.6/python/lib/py4j-0.9-src.zip')
sys.path.append('/home/mikdam/spark-1.6.0-bin-hadoop2.6/python/lib/pyspark.zip')

# Now we are ready to import Spark Modules
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SQLContext
    from pyspark.sql import DataFrame
    from pyspark.sql.functions import pow, col, udf, concat
    from pyspark.sql.types import DateType, StringType, IntegerType, DoubleType, \
        StructField, StructType, DataType, ArrayType

except ImportError as e:
    print ('Error importing Spark Modules', e)
    sys.exit(1)


def jaccard_similarity(x1, y1):
    x = n_grams(2, x1)
    y = n_grams(2, y1)
    if x == [] or y == []:
        return 0
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)


def n_grams(n, text):
    grams_list = []

    try:
        text = text.upper()
    except:
        return []

    for i in range(len(text) - (n - 1)):
        grams_list.append(text[i:i + n])

    return grams_list


def unpack_n_grams(id, match_col, n_gram_list):
    return [(id, match_col, gram) for gram in n_gram_list.split(',')]


def top_n(l, n):
    sl = sorted(l[1], key=lambda r: r.frequency, reverse=True)
    return (l[0], sl[:n])


# Discover and eliminate identical records based on a column 'match_column'
# This function uses SSJoin to join the dataframe to itself in order to discover similarity.
# The function returns a Spark DataFrame of the
def find_unduplicate_ids(data_df, match_column, id_column, sqlContext):
    udf_jaccard_similarity = udf(lambda x, y: jaccard_similarity(x, y), DoubleType())
    udf_n_grams = udf(lambda r: ','.join(n_grams(2, r)), StringType())
    # Change the name of we are working on
    # (id, match_col, col1, col2,..., [ngram1, ngram2,...])
    data_with_ngrams_df = data_df.withColumn('ngrams', udf_n_grams(match_column)).\
        withColumnRenamed(match_column, 'match_col').\
        withColumnRenamed(id_column, 'id').cache()

    # extract the ngrams list flatten it and count the frequency of each
    # we consider the frequency as the weight of each
    ngram_rdd = data_with_ngrams_df.flatMap(lambda r: r.ngrams.split(','))  # (ngram)
    ngram_count_rdd = ngram_rdd.map(lambda r: (r, 1)).reduceByKey(lambda a, b: a + b)  #(ngram, count)

    identical_removed_df = data_with_ngrams_df.dropDuplicates(['match_col'])
    id_ngram_rdd = identical_removed_df.flatMap(lambda r: unpack_n_grams(r.id, r.match_col, r.ngrams))  # (id, match_col,ngram1), (id, match_col, ngram2),...

    id_ngram_schema = StructType([StructField('id', IntegerType(), True),
                                  StructField('match_col', StringType(), True),
                                  StructField('ngram', StringType(), True)])

    ngram_count_schema = StructType([StructField('ngram', StringType(), True),
                                     StructField('frequency', IntegerType(), True)])

    id_ngram_df = sqlContext.createDataFrame(id_ngram_rdd, id_ngram_schema)  # (id, match_col, ngram1)

    ngram_count_df = sqlContext.createDataFrame(ngram_count_rdd, ngram_count_schema)

    # linking the ngrams in the main table with their weights so we can pick the top n ngrams
    id_ngram_count_df = id_ngram_df.join(ngram_count_df, 'ngram', 'left_outer')

    # pick the top n ngram record
    id_ngram_count_topn_rdd = id_ngram_count_df.rdd\
        .groupBy(lambda r: r.id)\
        .map(lambda l: top_n(l, 2))\
        .flatMap(lambda r: r[1])

    id_ngram_count_topn_schema = StructType([StructField('ngram', StringType(), True),
                                             StructField('id', IntegerType(), True),
                                             StructField('match_col', StringType(), True),
                                             StructField('frequency', IntegerType(), True)])

    id_ngram_count_topn_df1 = sqlContext.createDataFrame(id_ngram_count_topn_rdd, id_ngram_count_topn_schema).cache()

    # create a new DF, prepare to link the table to itself to discover duplication
    id_ngram_count_topn_df2 = id_ngram_count_topn_df1.selectExpr('ngram', 'match_col as match_col2', 'id as id2')

    main_df = id_ngram_count_topn_df1.join(id_ngram_count_topn_df2, 'ngram', 'inner')

    # Calculate similarity and add it as a column to the DF
    main_with_similarity_df = main_df.withColumn('similarity', udf_jaccard_similarity('match_col', 'match_col2'))

    # Filter the record that represent different entity and have a high similarity
    main_filtered_df = main_with_similarity_df.filter('similarity >= 0.7 and id != id2')

    duplicated_ids_rdd = main_filtered_df.rdd.map(lambda r: (r.id, 0))
    all_id_r_rdd = id_ngram_count_topn_df1.map(lambda r: (r.id, r))

    remaining_rdd = all_id_r_rdd.subtractByKey(duplicated_ids_rdd).map(lambda r: r[1])
    id_ngram_count_topn_df1.unpersist()
    results_df = sqlContext.createDataFrame(remaining_rdd, id_ngram_count_topn_schema).dropDuplicates(['id']).cache()

    return results_df


# This function returns a list of ids of potential matches.
# It uses SSJoin to narrow the set of potential match before calculating similarity
def find_candidate_matches(match_field_value, data_df, match_column, id_column, sqlContext):
    # Change the name of we are working on
    # (id, match_col, col1, col2,..., [ngram1, ngram2,...])
    udf_n_grams = udf(lambda r: ','.join(n_grams(2, r)), StringType())
    data_with_ngrams_df = data_df.withColumn('ngrams', udf_n_grams(match_column)).\
        withColumnRenamed(match_column, 'match_col').\
        withColumnRenamed(id_column, 'id').cache()

    identical_removed_df = data_with_ngrams_df.dropDuplicates(['match_col'])
    id_ngram_rdd = identical_removed_df.flatMap(lambda r: unpack_n_grams(r.id, r.match_col, r.ngrams))  # (id, match_col,ngram1), (id, match_col, ngram2),...

    id_ngram_schema = StructType([StructField('id', IntegerType(), True),
                                  StructField('match_col', StringType(), True),
                                  StructField('ngram', StringType(), True)])

    id_ngram_df = sqlContext.createDataFrame(id_ngram_rdd, id_ngram_schema)  # (id, match_col, ngram1)

    n_grams_str = ','.join(['\"' + s + '\"' for s in n_grams(2, match_field_value)])

    filtered_id_ngram_df = id_ngram_df.filter('ngram in (%s)'%n_grams_str)\
        .dropDuplicates(['match_col'])\
        .cache()

    sqlContext.registerDataFrameAsTable(filtered_id_ngram_df, 'maintable')
    sqlContext.registerFunction('mySim', lambda x: jaccard_similarity(x, '\"%s\"'%match_field_value), DoubleType())

    candidate_match_df = sqlContext.sql('select id from maintable where mySim(match_col) >=0.5')

    return [r.id for r in candidate_match_df.collect()]

