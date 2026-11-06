"""Lab 6: rdd"""
import re
import time
from operator import add

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("lab-rdd").master("local[*]").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# RDD creation and partitions
users_lines = sc.textFile("users.csv")
orders_lines = sc.textFile("orders.csv")
print("partitions:", users_lines.getNumPartitions(), orders_lines.getNumPartitions())
print("repartitioned:", orders_lines.repartition(8).getNumPartitions())

# Lazy evaluation and lineage
header = orders_lines.first()
orders_data = orders_lines.filter(lambda line: line != header and line.strip() != "")
print(orders_data.toDebugString().decode())
print("orders lines:", orders_data.count())

# Multiline address workaround
UUID_AT_START = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
users_records = users_lines.filter(lambda line: UUID_AT_START.match(line) is not None)
print("raw user lines:", users_lines.count(), "user records:", users_records.count())


# Manual parsing of orders
def parse_order(line):
    order_id, user_id, date, quantity, product = line.split(",")
    return {
        "order_id": order_id,
        "user_id": user_id,
        "date": date,
        "quantity": int(quantity),
        "product": product.strip().lower(),
    }


orders_rdd = orders_data.map(parse_order).cache()
print(orders_rdd.take(3))

# reduceByKey vs groupByKey
user_quantity = orders_rdd.map(lambda o: (o["user_id"], o["quantity"])).reduceByKey(add)
user_quantity_grouped = orders_rdd.map(lambda o: (o["user_id"], o["quantity"])).groupByKey().mapValues(sum)
print(user_quantity.take(5))
print(user_quantity_grouped.take(5))


# Manual deduplication
def most_recent(order_a, order_b):
    return order_a if order_a["date"] > order_b["date"] else order_b


orders_dedup = orders_rdd.map(lambda o: (o["order_id"], o)).reduceByKey(most_recent).values()
print("orders vs dedup:", orders_rdd.count(), orders_dedup.count())


# Join orders with users
def parse_user_for_join(line):
    fields = line.split(",")
    return fields[0], fields[1]  # uuid, username


users_kv = users_records.map(parse_user_for_join)
orders_kv = orders_rdd.map(lambda o: (o["user_id"], o))
enriched = orders_kv.join(users_kv)
print(enriched.take(3))

# Exercise 1: aggregateByKey -> (order_count, total_quantity), then average
user_stats = orders_rdd.map(lambda o: (o["user_id"], o["quantity"])).aggregateByKey(
    (0, 0),
    lambda acc, q: (acc[0] + 1, acc[1] + q),
    lambda a, b: (a[0] + b[0], a[1] + b[1]),
)
print(user_stats.mapValues(lambda s: s[1] / s[0]).take(3))

# Exercise 2: fake user_id + leftOuterJoin
fake = {"order_id": "fake-order-1", "user_id": "00000000-0000-0000-0000-000000000000",
        "date": "2020-01-01 00:00:00+00:00", "quantity": 1, "product": "bread"}
orders_with_fake = orders_rdd.union(sc.parallelize([fake]))
left = orders_with_fake.map(lambda o: (o["user_id"], o)).leftOuterJoin(users_kv)
orphans = left.filter(lambda kv: kv[1][1] is None)
print("orphan orders:", orphans.count())

# Exercise 3: same aggregation with the DataFrame API
orders_df = spark.read.csv("orders.csv", header=True, inferSchema=True).withColumnRenamed("user_uuid", "user_id")
df_quantity = orders_df.groupBy("user_id").sum("quantity")
df_quantity.show(5)
df_quantity.explain()
print(user_quantity.toDebugString().decode())


# Exercise 4: timing with and without cache
def timed(name, rdd):
    t = time.perf_counter()
    rdd.count()
    print(name, round(time.perf_counter() - t, 3), "s")


for i in (1, 2):
    timed(f"cached orders_rdd #{i}", orders_rdd)
    timed(f"cached user_quantity #{i}", user_quantity)

orders_rdd.unpersist()
orders_rdd_nc = orders_data.map(parse_order)
user_quantity_nc = orders_rdd_nc.map(lambda o: (o["user_id"], o["quantity"])).reduceByKey(add)
for i in (1, 2):
    timed(f"no-cache orders_rdd #{i}", orders_rdd_nc)
    timed(f"no-cache user_quantity #{i}", user_quantity_nc)

spark.stop()
