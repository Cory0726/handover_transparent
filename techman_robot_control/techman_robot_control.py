import socket

robot_ip = "192.168.50.49"
port = 5891

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((robot_ip, port))

print("Connected. Receiving data...")


data = sock.recv(4096)
print(data.decode("utf-8"))
