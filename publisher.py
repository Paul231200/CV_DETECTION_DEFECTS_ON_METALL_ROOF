import json
import os
import threading
import time
import uuid

import pika


class Publisher(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon = True
        self.is_running = True
        self.name = "sauron_amqp_publisher"

        self.exchange = 'ml_events'
        self.routing_key = 'ml_event'

        parameters = pika.URLParameters(os.environ['AMQP_URL'])
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

    def run(self):
        while self.is_running:
            self.connection.process_data_events(time_limit=1)

    def _publish(self, message):
        print(message)
        self.channel.basic_publish(exchange=self.exchange, routing_key=self.routing_key, body=message)

    def publish(self, app, zone, event, event_state):
        body = {
            'uuid': str(uuid.uuid4()),
            'app': app,
            'zone': str(zone),
            'event': event,
            'event_state': str(event_state),
            'timestamp': round(time.time())
        }
        self.connection.add_callback_threadsafe(lambda: self._publish(json.dumps(body)))

    def stop(self):
        self.is_running = False
        self.connection.process_data_events(time_limit=1)
        if self.connection.is_open:
            self.connection.close()

    def start(self):
        print("Starting AMPQ Publisher")
        super().start()
