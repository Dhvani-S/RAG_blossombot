from pymongo import MongoClient

import ssl
import certifi
import os
from dotenv import load_dotenv

load_dotenv() 

username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
cluster_name = os.getenv("MONGODB_CLUSTER_NAME")
db_name = os.getenv("MONGODB_DB_NAME")

#... rest of the code remains the same...

def mongo_connect():
    context = ssl.create_default_context(cafile=certifi.where())

    # Create a MongoDB client
    client = MongoClient(f"mongodb+srv://Blossom:monitoring@blossombot-monitoring.tqpbeqm.mongodb.net/?retryWrites=true&w=majority&appName=Blossombot-monitoring",
                        tlsCAFile=certifi.where())

    # Create a database
    db = client[db_name]

    # Create a collection
    collection = db["test_monmon"]
    return collection
"""
# Insert some data
data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
collection.insert_many(data)

# Print the data
for doc in collection.find():
    print(doc)

# Close the client
client.close()
"""