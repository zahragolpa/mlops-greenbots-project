import configparser
import os

def loadConfig():
    path = os.path.dirname(__file__)
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(path + '/../config.ini')
    return config