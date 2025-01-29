STATE = {
    'config': None
}

def get_current_config():
    if STATE['config'] is None:
        from .config import Config
        STATE['config'] = Config()

    return STATE['config']

def set_current_config(config):
    STATE['config'] = config
