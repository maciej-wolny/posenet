import json
from concurrent.futures import ThreadPoolExecutor
import threading
from abc import ABCMeta, abstractmethod
import io
import avro.io

import mqtt_client

# TODO - schema in file
test_schema = '''{
  "name": "TelemetryDataBatch",
  "type": "record",
  "namespace": "com.helsing.v1",
  "fields": [
    {
      "name": "size",
      "type": "int"
    },
    {
      "name": "telemetryDataBatch",
      "type": {
        "type": "array",
        "items": {
          "name": "TelemetryData",
          "type": "record",
          "fields": [
            {
              "name": "timestamp",
              "type": "int"
            },
            {
              "name": "sensorId",
              "type": "int"
            },
            {
              "name": "id",
              "type": "int"
            },
            {
              "name": "version",
              "type": "int"
            },
            {
              "name": "objects",
              "type": {
                "type": "array",
                "items": {
                  "name": "ClassificationResult",
                  "type": "record",
                  "fields": [
                    {
                      "name": "label",
                      "type": "string"
                    },
                    {
                      "name": "score",
                      "type": "int"
                    },
                    {
                      "name": "ymin",
                      "type": "int"
                    },
                    {
                      "name": "ymax",
                      "type": "int"
                    },
                    {
                      "name": "xmin",
                      "type": "int"
                    },
                    {
                      "name": "xmax",
                      "type": "int"
                    }
                  ]
                }
              }
            }
          ]
        }
      }
    }
  ]
}'''
schema = avro.schema.parse(test_schema)
writer = avro.io.DatumWriter(schema)

bytes_writer = io.BytesIO()
encoder = avro.io.BinaryEncoder(bytes_writer)


class MessageClient:
    __metaclass__ = ABCMeta

    @abstractmethod
    def send_messages(self, messages): raise NotImplementedError


class MQTTClient(MessageClient):
    def __init__(self):
        self.client = mqtt_client.connect_mqtt()
        self.client.loop_start()
        self.topic = "topic/test"

    def send_messages(self, messages):
        # TODO - latency metric
        size = 0
        for msg in messages:
            size += len(msg['objects'])
        # TODO - optimize
        writer.write({"telemetryDataBatch": messages, "size": size}, encoder)
        result = self.client.publish(self.topic, bytes_writer.getvalue())
        status = result[0]
        if status == 0:
            print(f"Send `{len(messages)}` to topic `{self.topic}`")
        else:
            # TODO - implement retry
            print(f"Failed to send message to topic {self.topic}")


class MockClient(MessageClient):
    def send_messages(self, messages):
        print('Hello, World 2!')
        print(messages[0])
        print(writer.write({"telemetryDataBatch": messages}, encoder))


class ImageBatchProcessor:
    def __init__(self, config=None):
        self.semaphore = threading.Lock()
        # self.batch_size = config.batch_size
        self.batch_size = 10
        self.batch = []
        self.client = MQTTClient()
        self.send_worker = ThreadPoolExecutor(max_workers=1)

    def insert_data(self, data):
        self.semaphore.acquire()
        try:
            self.batch.append(data)
            if len(self.batch) >= self.batch_size:
                self.send_worker.submit(self.client.send_messages, self.batch.copy())
                self.batch = []
        except IOError:
            print("Batch send failed, saving...")
        finally:
            self.semaphore.release()