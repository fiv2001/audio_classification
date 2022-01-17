import configparser

def create_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']
