from .state import get_current_config

class WrappedFunction:

    def __init__(self, func):
        self.func = func
        self.arg_paths = []

    def add_arg(self, arg, alias):
        self.arg_paths.append([None, arg, alias])

    def set_section(self, section):
        for i in reversed(range(len(self.arg_paths))):
            current_ns, path, alias = self.arg_paths[i]
            if current_ns is None:
                self.arg_paths[i][0] = section
            else:
                break

    def __call__(self, *args, **kwargs):
        config = get_current_config()
        filled_args = {}
        for ns, path, alias in self.arg_paths:
            if ns is not None:
                path = ns + path
            if alias in kwargs:  # User overrode this argument
                continue

            value = config[path]

            if value is not None:
                filled_args[alias] = value

        filled_args.update(kwargs)

        try:
            return self.func(*args, **filled_args)
        except TypeError as e:
            if 'multiple values for argument' in e.args[0]:
                raise TypeError("""Ambiguity overriding config arguments, use \
named parameter to resolve it""") from None
            else:
                raise e

def extract_function(func):
    if hasattr(func, '__fastarg_wrapper'):
        return getattr(func, '__fastarg_wrapper')
    else:
        return WrappedFunction(func)

def param(parameter, alias=None):
    if isinstance(parameter, str):
        parameter = tuple(parameter.split('.'))

    if alias is None:
        alias = parameter[-1]

    def wrapper(func):

        func = extract_function(func)

        func.add_arg(parameter, alias)

        def result(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(result, '__fastarg_wrapper', func)
        return result

    return wrapper

def section(section):
    if isinstance(section, str):
        section = tuple(section.split('.'))

    def wrapper(func):
        func = extract_function(func)
        func.set_section(section)
        return func
    return wrapper
