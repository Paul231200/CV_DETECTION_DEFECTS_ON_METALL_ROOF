import logging
import os
import time
import threading
from typing import List, Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

MQTT_QOS_FAST = 0
MQTT_QOS_RELIABLE = 1


def _topic_variants(topic: str) -> List[str]:
    t = (topic or "").strip()
    if not t:
        return []
    if t.startswith("/"):
        return [t, t.lstrip("/")]
    return ["/" + t, t]

class MQTTPublisher:
    """
    Класс для публикации событий о дефектах в MQTT для Wiren Board
    """
    
    def __init__(
        self,
        broker: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
    ):
        """
        Инициализация MQTT-клиента для публикации событий
        
        :param broker: адрес MQTT-брокера (по умолчанию переменная окружения MQTT_BROKER)
        :param port: порт MQTT-брокера (по умолчанию переменная MQTT_PORT или 8883)
        :param username: имя пользователя (MQTT_USER)
        :param password: пароль (MQTT_PASSWORD)
        :param client_id: ID клиента (MQTT_CLIENT_ID или DefectPilot)
        """
        self.broker = (broker if broker is not None else os.environ.get("MQTT_BROKER", "")).strip()
        self.port = int(port if port is not None else os.environ.get("MQTT_PORT", "8883"))
        self.username = username if username is not None else os.environ.get("MQTT_USER", "")
        self.password = password if password is not None else os.environ.get("MQTT_PASSWORD", "")
        self.client_id = client_id if client_id is not None else os.environ.get("MQTT_CLIENT_ID", "DefectPilot")
        self.topic_k1 = (os.environ.get("MQTT_TOPIC_K1") or "").strip()
        self.topic_k2 = (os.environ.get("MQTT_TOPIC_K2") or "").strip()
        self.client = None
        self.is_connected = False
        self.connect_thread = None
        self._reconnect_timer = None
        self._disabled = True

        if not mqtt:
            logging.warning("Библиотека paho-mqtt не установлена, MQTT функциональность недоступна")
            return

        if not self.broker:
            logging.info("MQTT_BROKER не задан — MQTT выключен (задайте MQTT_BROKER и учётные данные для включения)")
            return

        self._disabled = False
        self._setup_client()
        if self.client:
            self.connect()
            self._start_reconnect_timer()
    
    def _setup_client(self):
        """
        Настройка MQTT клиента
        """
        try:
            # Создаем клиента с указанным ID и persistent сессией
            self.client = mqtt.Client(self.client_id, clean_session=False)
            
            # Устанавливаем обработчики событий
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            # Устанавливаем учетные данные
            self.client.username_pw_set(self.username, self.password)
            
            # Настраиваем параметры подключения для надежности
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)
            
            logging.info(f"MQTT клиент настроен для {self.broker}:{self.port}")
        except Exception as e:
            logging.error(f"Ошибка при настройке MQTT клиента: {e}")
            self.client = None
    
    def _on_connect(self, client, userdata, flags, rc):
        """
        Обработчик события успешного подключения
        """
        if rc == 0:
            self.is_connected = True
            logging.info(f"Подключено к MQTT брокеру {self.broker}:{self.port}")
            # Дополнительная информация о подключении
            logging.info(f"MQTT соединение установлено, client_id={self.client_id}, userdata={userdata}, flags={flags}")
        else:
            self.is_connected = False
            # Более подробная информация об ошибке подключения
            rc_codes = {
                0: "Успешное подключение",
                1: "Неверная версия протокола MQTT",
                2: "Неверный идентификатор клиента",
                3: "Сервер недоступен",
                4: "Неверное имя пользователя или пароль",
                5: "Нет авторизации"
            }
            rc_desc = rc_codes.get(rc, "Неизвестная ошибка")
            logging.error(f"Ошибка подключения к MQTT брокеру, код: {rc} - {rc_desc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """
        Обработчик события отключения
        """
        self.is_connected = False
        if rc == 0:
            logging.info("Штатное отключение от MQTT брокера")
        else:
            logging.warning(f"Нештатное отключение от MQTT брокера, код: {rc}")
    
    def connect(self) -> bool:
        """
        Подключение к MQTT-брокеру
        
        :return: True если подключение инициировано успешно, иначе False
        """
        if getattr(self, "_disabled", False):
            return False
        if not mqtt:
            logging.warning("Библиотека paho-mqtt не установлена, подключение невозможно")
            return False
        
        if self.client is None:
            self._setup_client()
            if self.client is None:
                return False
        
        try:
            # Дополнительное логирование перед подключением
            logging.info(f"Попытка подключения к MQTT брокеру {self.broker}:{self.port} с client_id={self.client_id}")
            
            # Подключаемся к брокеру
            self.client.connect(self.broker, port=self.port, keepalive=60)
            self.client.loop_start()
            
            # Ожидаем подключения с таймаутом
            start_time = time.time()
            max_wait_time = 5  # Максимальное время ожидания в секундах
            
            while not self.is_connected and time.time() - start_time < max_wait_time:
                time.sleep(0.1)
            
            # Проверяем результат
            if not self.is_connected:
                logging.error("Тайм-аут подключения к MQTT брокеру")
                self.client.loop_stop()
            else:
                logging.info(f"MQTT подключение успешно установлено к {self.broker}:{self.port}")
                
            return self.is_connected
        except Exception as e:
            logging.error(f"Ошибка подключения к MQTT брокеру: {e}")
            if self.client:
                try:
                    self.client.loop_stop()
                except Exception as stop_error:
                    logging.error(f"Ошибка при остановке MQTT цикла: {stop_error}")
            return False
    
    def disconnect(self):
        """
        Отключение от MQTT-брокера
        """
        if self.client and self.is_connected:
            try:
                self.client.disconnect()
                self.client.loop_stop()
                logging.info("Отключено от MQTT брокера")
            except Exception as e:
                logging.error(f"Ошибка при отключении от MQTT брокера: {e}")
        
        self.is_connected = False
    
    def _publish_fast_variants(self, topic: str, payload: str) -> bool:
        if getattr(self, "_disabled", False) or not self.topic_available(topic):
            return False
        if not mqtt or not self.client or not self.is_connected:
            return False
        variants = _topic_variants(topic)
        if not variants:
            return False
        ok = False
        for variant in variants:
            try:
                result = self.client.publish(variant, payload, qos=MQTT_QOS_FAST, retain=False)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    ok = True
            except Exception:
                pass
        return ok

    @staticmethod
    def topic_available(topic: str) -> bool:
        return bool((topic or "").strip())

    def publish_defect_event(self, state: int) -> bool:
        """
        Публикация события о дефекте на K1
        
        :param state: состояние (1 - дефект обнаружен, 0 - дефект не обнаружен)
        :return: True если публикация успешна, иначе False
        """
        payload = str(state)
        if not self.topic_available(self.topic_k1):
            return False
        return self.publish(self.topic_k1, payload, qos=MQTT_QOS_RELIABLE)
    
    def publish_defect_event_fast(self, state: int) -> bool:
        """
        Быстрая публикация события о дефекте на K1
        
        :param state: состояние (1 - дефект обнаружен)
        :return: True если публикация запущена, иначе False
        """
        payload = str(state)
        if not self.topic_available(self.topic_k1):
            return False
        return self._publish_fast_variants(self.topic_k1, payload)
    
    def publish_defect_event_k2(self, state: int) -> bool:
        """
        Публикация события о дефекте на K2
        
        :param state: состояние (1 - дефект обнаружен, 0 - дефект не обнаружен)
        :return: True если публикация успешна, иначе False
        """
        payload = str(state)
        if not self.topic_available(self.topic_k2):
            return False
        return self.publish(self.topic_k2, payload, qos=MQTT_QOS_RELIABLE)
    
    def publish_defect_event_fast_k2(self, state: int) -> bool:
        """
        Быстрая публикация события о дефекте на K2
        
        :param state: состояние (1 - дефект обнаружен)
        :return: True если публикация запущена, иначе False
        """
        payload = str(state)
        if not self.topic_available(self.topic_k2):
            return False
        return self._publish_fast_variants(self.topic_k2, payload)
    
    def publish_defect_event_both(self, state: int) -> bool:
        """
        Публикация события о дефекте на K1 и K2 одновременно
        
        :param state: состояние (1 - дефект обнаружен, 0 - дефект не обнаружен)
        :return: True если обе публикации успешны, иначе False
        """
        result_k1 = self.publish_defect_event(state)
        result_k2 = self.publish_defect_event_k2(state)
        return result_k1 and result_k2
    
    def publish_defect_event_fast_both(self, state: int) -> bool:
        """
        Быстрая публикация события о дефекте на K1 и K2 одновременно
        
        :param state: состояние (1 - дефект обнаружен)
        :return: True если обе публикации запущены, иначе False
        """
        result_k1 = self.publish_defect_event_fast(state)
        result_k2 = self.publish_defect_event_fast_k2(state)
        return result_k1 and result_k2
    
    def publish(self, topic: str, payload: str, retain: bool = False, qos: int = MQTT_QOS_FAST) -> bool:
        """
        Публикация сообщения в указанный топик
        
        :param topic: MQTT-топик
        :param payload: сообщение для публикации
        :param retain: флаг retain
        :param qos: качество обслуживания (0, 1 или 2)
        :return: True если публикация успешна, иначе False
        """
        if getattr(self, "_disabled", False):
            return False
        if not mqtt or self.client is None or not self.is_connected:
            return False
        
        try:
            result = self.client.publish(topic, payload, qos=qos, retain=retain)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception:
            return False
    
    def _start_reconnect_timer(self):
        """
        Запуск таймера для проверки и восстановления соединения
        """
        if getattr(self, "_disabled", False) or not self.client:
            return

        def check_connection():
            if not self.is_connected and self.client:
                logging.info("Автоматическое переподключение к MQTT...")
                try:
                    self.connect()
                except Exception as e:
                    logging.error(f"Ошибка при переподключении: {e}")
            
            # Перезапускаем таймер
            if self.client:
                self._reconnect_timer = threading.Timer(5.0, check_connection)
                self._reconnect_timer.daemon = True
                self._reconnect_timer.start()
        
        # Запускаем первый таймер
        self._reconnect_timer = threading.Timer(5.0, check_connection)
        self._reconnect_timer.daemon = True
        self._reconnect_timer.start()

# Пример использования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    publisher = MQTTPublisher()
    
    # Проверяем соединение
    time.sleep(2)
    
    # Публикуем событие "дефект обнаружен" быстрым методом на K1 и K2
    publisher.publish_defect_event_fast_both(1)
    time.sleep(2)
    
    # Публикуем событие "дефект обнаружен" надежным методом на K1 и K2
    publisher.publish_defect_event_both(1)
    time.sleep(1)
    
    publisher.disconnect()