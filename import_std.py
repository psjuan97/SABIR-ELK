from elasticsearch import helpers, Elasticsearch
import csv
import time

auth = ("elastic", "changeme")
es = Elasticsearch(hosts = ["http://localhost:9200"], http_auth = auth)

start = time.time()

es.indices.create(index='std_test', body={
   'settings' : {
         'index' : {
              'number_of_shards':12
         }
   }
})

es.indices.create(index='std_train', body={
   'settings' : {
         'index' : {
              'number_of_shards':12
         }
   }
})


with open('./datasets/testdata.manual.2009.06.14.csv') as f:
    reader = csv.DictReader(f,delimiter=',')
    helpers.bulk(es, reader, index='std_test')

print("end load test.")


time.sleep(5)



with open('./datasets/training.1600000.processed.noemoticon.csv') as t:
    reader_t = csv.DictReader(t)
    helpers.bulk(es, reader_t, index='std_train',chunk_size=10024)
print("end load train.")


end = time.time()
print(end - start)
