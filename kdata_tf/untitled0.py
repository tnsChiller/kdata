import socket
from uuid import uuid4

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
HOST = "127.0.0.1"
s.connect((HOST,"3939"))

data_dct = {
        "action":"train",
        "mod_pk": "093b8171-2505-4204-9add-dcd0cd09379d",
        "new_pk": str(uuid4()),
        "num": "100",
        "it": "5",
        "n_epochs": "5",
        "btc_size": "32",
        "learning_rate":"0.001",
        "shuffle": "1"
    }
payload = ""
for i in data_dct:
    payload += data_dct[i]+"\n"
s.send(payload.encode())
data = s.recv(1024)
print(data.decode())
