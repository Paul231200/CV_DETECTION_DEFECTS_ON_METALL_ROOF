import os
import json
import base64
from urllib.parse import urlparse

import requests as req
import clickhouse_connect
from clickhouse_connect.driver import Client


def get_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Не найдена переменная окружения: {name}")
    return value

DB_HOST = get_env_var('DB_HOST')
DB_USER = get_env_var('DB_USER')
DB_PASS = get_env_var('DB_PASS')
DB_DATABASE = get_env_var('DB_DATABASE')
MODEL_UUID = get_env_var('MODEL_UUID')


def load_model(model):
    os.makedirs("models", exist_ok=True)

    headers = {'PRIVATE-TOKEN': model['token']}
    filename = os.path.basename(urlparse(model['url']).path)

    print(f"Проверка модели: {filename}")
    print(f"URL модели: {model['url']}")
    resp = req.head(model['url'], headers=headers)

    print(f"HTTP статус: {resp.status_code}")
    if resp.status_code != 200:
        print("Ошибка при получении информации о модели")
        return

    commit_id = resp.headers.get('X-Gitlab-Commit-Id', '')
    print(f"Commit ID: {commit_id}")

    local_commit_path = f'models/{filename}.commit'
    local_model_path = f'models/{filename}'

    need_download = True

    if os.path.exists(local_commit_path):
        with open(local_commit_path, 'r') as f:
            current_commit = f.read().strip()
        if current_commit == commit_id and os.path.exists(local_model_path):
            print("Модель актуальна, загрузка не требуется")
            print(f"Размер модели: {os.path.getsize(local_model_path)} байт")
            need_download = False

    if need_download:
        print("Загрузка новой версии модели...")
        resp = req.get(model['url'], headers=headers)
        if resp.status_code == 200:
            print(f"Получен ответ от сервера, размер контента: {len(resp.content)} байт")
            try:
                content = json.loads(resp.content)
                print(f"JSON успешно загружен, размер контента: {len(content.get('content', ''))} байт")
                
                decoded_data = base64.b64decode(content.get('content'))
                print(f"Данные декодированы, размер: {len(decoded_data)} байт")
                
                with open(local_model_path, 'wb') as f:
                    f.write(decoded_data)

                print(f"Модель сохранена по пути: {local_model_path}")
                print(f"Размер файла: {os.path.getsize(local_model_path)} байт")
                
                with open(local_commit_path, 'w') as f:
                    f.write(commit_id)

                print(f"Модель {filename} загружена")
            except Exception as e:
                print(f"Ошибка при обработке модели: {e}")
        else:
            print(f"Ошибка при загрузке модели: HTTP {resp.status_code}")


def get_model_from_db(model_uuid):
    client = clickhouse_connect.create_client(
        host=DB_HOST,
        username=DB_USER,
        password=DB_PASS,
        database=DB_DATABASE
    )

    params = {'uuid': model_uuid}
    result = client.query(
        '''
        SELECT url, class_names, token
        FROM ml_models FINAL
        WHERE uuid = {uuid:UUID}
        ''',
        parameters=params
    )

    rows = list(result.named_results())
    return rows[0] if rows else None


def main():
    model = get_model_from_db(MODEL_UUID)
    if model:
        load_model(model)
    else:
        print("Модель с таким UUID не найдена в базе данных.")


if __name__ == "__main__":
    main()
