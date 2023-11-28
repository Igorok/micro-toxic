from infrastructure.config import getConfig
from toxic_module.toxic_service import ToxicService
from infrastructure.kafka import getKafkaConsumer, getKafkaProducer


def main():
    config = getConfig()
    consumer = getKafkaConsumer(config)
    producer = getKafkaProducer(config)
    toxic_service = ToxicService()

    print('toxic-service started')

    for message in consumer:
        try:
            if not message.value.get('message'):
                continue

            prediction = toxic_service.is_toxic(message.value.get('message'))
            if prediction['is_toxic']:
                response = {
                    'id': message.value.get('id'),
                    'user_id': message.value.get('user_id'),
                    'analysis': prediction['prediction']
                }
                producer.send(config['KAFKA_ANALYSIS_MESSAGE_TOPIC'], response)

        except Exception as error:
            print('error', error)


if __name__ == '__main__':
    main()
