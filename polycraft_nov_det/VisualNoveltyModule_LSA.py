import socket, random, time, json
import sys
import os

from VisualNoveltyDetector_LSA import *
import ast
import argparse

parser = argparse.ArgumentParser(description='Novlety detector operation mode')
parser.add_argument('--host', metavar='H',default="127.0.0.1", type=str, help='Host')
parser.add_argument('--port', metavar='P',default=8000, type=int, help='Port to listen for data')
#parser.add_argument('--mode', metavar='M',default=1, type=int, help='1: binary, 2:item, 3: item-location')

args = parser.parse_args()
#mode = args.mode
PORT = args.port
HOST = args.host

def recv_json(sock):
    BUFF_SIZE = 4096  # 4 KiB
    data = b''
    while True:
        time.sleep(0.00001)
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    
    return data


detector = VisualNoveltyDetector(
        state_dict_path='saved_statedict/LSA_polycraft_no_est_075_980.pt', scale_factor=0.75)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
for i in range(10):
    try:
        sock.connect((HOST, PORT))
        break
    except:
        time.sleep(0.5)
        if i == 9:
            exit()

print("connected")

while True:
    data = recv_json(sock).decode()
    time.sleep(1)

    
    
    if len(data)<1: #Not receiving img or state data
        continue

    #Parse and check for novelties

    data = data.split("@")
    if len(data) ==1: #Receive close message from server
        print("exiting...")
        sock.close()
        break
    imgj=data[0]
    statej=data[1]
    
    
    res = detector.check_for_novelty(imgj)
    print(res)
   
    sock.sendall(str.encode(str(res)+"\n"))

