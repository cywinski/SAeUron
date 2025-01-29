import argparse
from collections import defaultdict
import json
import sys
import os

from terminaltables import SingleTable

from .param import Param
from .section import Section
from .exceptions import MissingValueError, ValidationError
from .dict_utils import (
    fix_dict, expand_keys, recursive_get, rec_dd, recursive_set,
    NestedNamespace)


class Config:
    def __init__(self):
        self.sections = defaultdict(lambda: None)
        self.sections_to_entries = defaultdict(list)
        self.entries = {}
        self.content = {}

    def add_section(self, section):
        self.sections[section.ns] = section

    def add_entry(self, ns, name, param):
        path = ns + tuple(name.split('.'))
        self.sections_to_entries[ns].append(path)
        self.entries[path] = param

    def collect(self, config):
        config = fix_dict(expand_keys(config))
        entries = list(self.entries.items())
        # We repeat until the list of entries doesn't change
        while True:
            for path, param in entries:
                try:
                    value = recursive_get(config, path)
                    if value is not None:
                        self.content[path] = value
                        # We try to validate the parameter to trigger an
                        # import in the case the param contains a module
                        try:
                            param.validate(value)
                        except:
                            pass
                except:
                    pass
            new_entries = list(self.entries.items())
            if len(new_entries) == len(entries):
                break
            else:
                entries = new_entries
        return self

    def augment_argparse(self, parser):
        parser.add_argument('--config-file', '-C', action='append', default=[],
                            help='Integrate a config file (json or yaml, can be repeated)')

        parser.formatter_class = argparse.RawTextHelpFormatter

        epilogStart = """
Arguments:
----------

Each argument can be defined from a JSON file, a YAML file, env variable
or from CLI arguments. For CLI just use:

--PATH.TO.ARG=value

"""
        previous_epilog = ""
        while True:
            epilog = ""
            for sec_path, entries in self.sections_to_entries.items():
                table_content = [['Name', 'Default', 'Constraint', 'Description']]
                for path in entries:
                    param = self.entries[path]
                    # need to accept command line input for disabled sections
                    # if not param.section.is_enabled(self):
                    #    continue
                    argname = '.'.join(path)
                    # We do not want to show the args since we have our nice table after
                    if argname == 'help' or argname == 'h':
                        raise ValueError(f"Argument {argname} is reserved for argparse help")
                    try:
                        additional_args = {}
                        if param.is_flag:
                            additional_args['action'] = 'store_true'
                        parser.add_argument(f'--{argname}',
                                            help=argparse.SUPPRESS,
                                            **additional_args)
                    except argparse.ArgumentError:
                        pass  # We might have tried to add this one already
                    default = param.default
                    if param.required:
                        default = 'Requried!'
                    table_content.append([argname, default, param.checker.help(), param.desc])
                section_desc = self.sections[sec_path].desc
                epilog += SingleTable(table_content, section_desc).table + "\n\n"

            entry_count = len(self.entries)
            self.collect_argparse_args(parser, disable_help=True)
            # If we have more entries in the config we have to regenerate
            # the help with the new settings
            if len(self.entries) == entry_count and epilog == previous_epilog:
                break
            previous_epilog = epilog


        parser.epilog = epilogStart + epilog

        return self


    def collect_config_file(self, fname):
        try:
            self.collect_json(fname)
        except json.decoder.JSONDecodeError:
            self.collect_yaml(fname)

        return self


    def collect_json(self, fname):
        with open(fname) as handle:
            self.collect(json.load(handle))

        return self

    def collect_yaml(self, fname):
        import yaml
        with open(fname) as handle:
            content = handle.read()
        self.collect(yaml.safe_load(content))

        return self

    def collect_env_variables(self):
        self.collect(dict(os.environ))

        return self

    def collect_argparse_args(self, parser, disable_help=False):
        cli_args = sys.argv[1:]
        if disable_help:
            cli_args = [x for x in cli_args if x != '--help']
        args = parser.parse_args(cli_args)
        for fname in args.config_file:
            self.collect_config_file(fname)

        args = vars(args)
        del args['config_file']

        self.collect(args)
        self.collect_env_variables()

        return self

    def __getitem__(self, path):
        if isinstance(path, str):
            path = tuple(path.split('.'))

        try:
            param = self.entries[path]
        except KeyError:
            raise KeyError(f"{'.'.join(path)} not defined")

        if not param.section.is_enabled(self):
            return None

        try:
            value = self.content[path]
        except KeyError:
            value = None

        if value is None and param.default is not None:
            value = param.default

        if value is None and not param.required:
            return value

        try:
            return param.validate(value)
        except ValidationError as e:
            print(f'Issue when typechecking argument psyh `{".".join(path)}`:')
            raise e


    def get(self):
        result = rec_dd()
        for path in self.entries.keys():
            value = self[path]
            if value is not None:
                recursive_set(result, path, value)

        return NestedNamespace(fix_dict(result))


    def get_section(self, section):
        result = {}
        section = tuple(section.split('.'))
        for path in self.entries.keys():
            i = 0
            while i < len(section):
                if section[i] != path[i]:
                    break
                i += 1
            if i == len(section) and len(path) == i + 1:
                result['.'.join(path[i:])] = self[path]
        return result


    def get_all_config(self, dump_path=None, ignore = []):
        config_dict = rec_dd()
        for path in self.entries.keys():
            if '.'.join(path) not in ignore and self.entries[path].section.is_enabled(self):
                recursive_set(config_dict, path, self[path])
        config_dict = fix_dict(config_dict)
        if dump_path is not None:
            if os.path.split(dump_path)[-1].split('.')[-1] == 'json':
                with open(dump_path, "w") as f:
                    json.dump(config_dict, f)
            elif os.path.split(dump_path)[-1].split('.')[-1] in ['yaml', 'yml']:
                import yaml
                with open(dump_path, "w") as f:
                    yaml.dump(config_dict, f)
            else:
                raise ValueError("Only support json and yaml format")
        return config_dict
    

    def validate(self, mode='stderr'):
        errors = {}
        for path, param in self.entries.items():
            try:
                self[path]
            except (MissingValueError, ValidationError) as e:
                errors[path] = e

        if mode == 'stderr':
            if len(errors) > 0:
                table = [['Param', 'Issue', 'Got']]

                for path, error in errors.items():
                    if isinstance(error, MissingValueError):
                        issue = "Required!"
                        got = ""
                    elif isinstance(error, ValidationError):
                        issue = self.entries[path].checker.help()
                        got = self.content[path]
                    table.append(['.'.join(path), issue, got])

                print(SingleTable(table, 'Argument validation errors').table,
                      file=sys.stderr)
                sys.exit()

        elif mode == 'errordict':
            return errors

    def summary(self, target=sys.stderr):
        table = [['Parameter', 'Value']]

        for path in self.entries.keys():
            try:
                value = self[path]
                if value is not None:
                    table.append(['.'.join(path), self[path]])
            except:
                pass

        print(SingleTable(table, ' Arguments defined').table, file=target)

        return self

