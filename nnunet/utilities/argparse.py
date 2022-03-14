import argparse
import re


class ParseKwargs(argparse.Action):
    int_regex = re.compile(r"^[0-9]+$")
    float_regex = re.compile(r"\d+\.\d+")
    list_regex = re.compile(r"^\[.+]")

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = self.parse_type(value)

    def parse_type(self, value):
        if re.match(self.list_regex, value):
            values = list(map(self.parse_type, value.strip("[]").split(",")))
            return values
        elif re.match(self.float_regex, value):
            return float(value)
        elif re.match(self.int_regex, value):
            return int(value)
        else:
            return value
