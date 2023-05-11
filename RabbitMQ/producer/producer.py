import argparse
import os
import pika
from dotenv import load_dotenv

load_dotenv("../")
ip = os.environ.get("ip")
port = os.environ.get("port")
vhost = os.environ.get("vhost")
id = os.environ.get("id")
pw = os.environ.get("pw")
exchange = os.environ.get("exchange")
routing_key = os.environ.get("routing_key")
queue = os.environ.get("queue")


def push(message, priority=0):
    # RabbitMQ 브로커 연결
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(ip, port, vhost, pika.PlainCredentials(id, pw))
    )

    # 채널 및 큐 생성
    channel = connection.channel()
    channel.queue_declare(queue=queue, durable=True, arguments={"x-max-priority": 1})

    # 큐에 메시지 전송 (utf-8 encoding)
    channel.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=message.encode("utf-8"),
        properties=pika.BasicProperties(priority=priority),
    )
    print(f"[x] Sent Success")

    # 연결 종료
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--priority", "-p", type=int, default=0, required=False)
    args = parser.parse_args()

    # 전송할 yaml 파일 로드
    with open(args.config, "r") as f:
        config = f.read()

    # RabbitMQ 브로커로 전송
    push(config, args.priority)
