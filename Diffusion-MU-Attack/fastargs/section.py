from .state import get_current_config

class Section:
    def __init__(self, ns, desc=None,
                 config_descriptor=None):
        self.ns = tuple(ns.split('.'))

        if config_descriptor is None:
            config_descriptor = get_current_config()
        self.config_descriptor = config_descriptor

        self.desc = desc

        self.config_descriptor.add_section(self)
        self.condition = None

    def enable_if(self, condition):
        self.condition = condition
        return self

    def is_enabled(self, config):
        if self.condition is None:
            return True
        try:
            return self.condition(config)
        except:
            return False


    def params(self, **kwargs):
        for name, param in kwargs.items():
            param.section = self
            self.config_descriptor.add_entry(self.ns, name, param)
        return self


