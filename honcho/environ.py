import os
import re
import shlex
from collections import OrderedDict, defaultdict, namedtuple

PROCFILE_LINE = re.compile(r'^([A-Za-z0-9_-]+):\s*(.+)$')


class Env(object):

    def __init__(self, config):
        self._c = config

    @property
    def port(self):
        try:
            return int(self._c['port'])
        except ValueError:
            raise ValueError(f"invalid value for port: '{self._c['port']}'")

    @property
    def procfile(self):
        return os.path.join(self._c['app_root'], self._c['procfile'])

    def load_procfile(self):
        with open(self.procfile) as f:
            content = f.read()

        return parse_procfile(content)


class Procfile(object):
    """A data structure representing a Procfile"""

    def __init__(self):
        self.processes = OrderedDict()

    def add_process(self, name, command):
        assert name not in self.processes, \
            "process names must be unique within a Procfile"
        self.processes[name] = command


def parse_procfile(contents):
    p = Procfile()
    for line in contents.splitlines():
        m = PROCFILE_LINE.match(line)
        if m:
            p.add_process(m.group(1), m.group(2))
    return p


def parse(content):
    """
    Parse the content of a .env file (a line-delimited KEY=value format) into a
    dictionary mapping keys to values.
    """
    values = {}
    for line in content.splitlines():
        lexer = shlex.shlex(line, posix=True)
        tokens = list(lexer)

        # parses the assignment statement
        if len(tokens) < 3:
            continue

        name, op = tokens[:2]
        value = ''.join(tokens[2:])

        if op != '=':
            continue
        if not re.match(r'[A-Za-z_][A-Za-z_0-9]*', name):
            continue

        value = value.replace(r'\n', '\n')
        value = value.replace(r'\t', '\t')
        values[name] = value

    return values


ProcessParams = namedtuple("ProcessParams", "name cmd quiet env")


def expand_processes(processes, concurrency=None, env=None, quiet=None, port=None):
    """
    Get a list of the processes that need to be started given the specified
    list of process types, concurrency, environment, quietness, and base port
    number.

    Returns a list of ProcessParams objects, which have `name`, `cmd`, `env`,
    and `quiet` attributes, corresponding to the parameters to the constructor
    of `honcho.process.Process`.
    """
    if env is not None and env.get("PORT") is not None:
        port = int(env.get("PORT"))

    if quiet is None:
        quiet = []

    con = defaultdict(lambda: 1)
    if concurrency is not None:
        con.update(concurrency)

    out = []

    for name, cmd in processes.items():
        for i in range(con[name]):
            n = "{0}.{1}".format(name, i + 1)
            c = cmd
            q = name in quiet
            e = {'HONCHO_PROCESS_NAME': n}
            if env is not None:
                e.update(env)
            if port is not None:
                e['PORT'] = str(port + i)

            params = ProcessParams(n, c, q, e)
            out.append(params)
        if port is not None:
            port += 100

    return out
