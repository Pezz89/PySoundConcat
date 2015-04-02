import os

def listdir_nohidden(path):
    return [i for i in os.listdir(path) if not i.startswith('.')]

def must_exist(directory, msg="Directory already exists"):
    try:
        os.mkdir(directory)
    except OSError as err:
        if os.path.exists(directory):
            print msg
        else:
            raise err
