''' -------------------------------------------

interfaces, etc

------------------------------------------- '''

import tensorflow as tf
import sys

DIVISION_LINE_LENGTH = 62

def showDivision():
    print('=' * DIVISION_LINE_LENGTH)

def showDivisionSingle():
    print('-' * DIVISION_LINE_LENGTH)

def showError(msg):
    print('ERROR : ' + msg)

def showProcess(msg, isShow = True):
    if isShow:
        showDivision()
        print(msg)
        showDivision()

def showDevice(): 
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def input(msg):
    showDivisionSingle()
    print(msg)
    response = sys.stdin.readline().strip()
    print('>>> ' + response)
    return response

def inputInt(msg):
    return int(input(msg))