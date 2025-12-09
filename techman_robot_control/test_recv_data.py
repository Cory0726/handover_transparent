import socket

def run(robot_ip):
    port = 5891

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((robot_ip, port))

    print("Connected. Receiving data...")

    data = sock.recv(4096)
    sock.close()
    print(data.decode("utf-8"))
    print("Finished receiving data.")


if __name__ == "__main__":
    ROBOT_IP = "192.168.50.49"
    run(ROBOT_IP)