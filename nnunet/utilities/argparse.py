import argparse
import json


class ParseJson(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.parse_dict(values))

    @staticmethod
    def parse_dict(dict_str):
        return json.loads(dict_str)
