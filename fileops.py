import os

def listdir_nohidden(path):
    return [i for i in os.listdir(path) if not i.startswith('.')]
