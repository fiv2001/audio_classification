import configparser

def create_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']

def log(x):
    print(x)
    with open('log.txt', 'a') as log:
        log.write(x + '\n')
