import json
from kafka import KafkaConsumer
from kafka import KafkaProducer


def getKafkaConsumer(config):
    return KafkaConsumer(
        config['KAFKA_READY_MESSAGE_TOPIC'],
        group_id=config['KAFKA_READY_MESSAGE_GROUP'],
        bootstrap_servers=[config['KAFKA_URI']],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    )

def getKafkaProducer(config):
    return KafkaProducer(
        bootstrap_servers=[config['KAFKA_URI']],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
