import re
import shlex
import sys

try:
    import jinja2
except ImportError:
    print("honcho's 'export' command requires the jinja2 package,\n"
          "which you don't appear to have installed.\n"
          "\n"
          "To fix this, install honcho with the 'export' extra selected:\n"
          "\n"
          "    pip install honcho[export]\n",
          file=sys.stderr)
    sys.exit(1)


class BaseExport(object):
    def __init__(self, template_dir=None, template_env=None):
        if template_env is not None:
            self._template_env = template_env
            return

        if template_dir is not None:
            loader = jinja2.FileSystemLoader([template_dir])
        else:
            loader = self.get_template_loader()

        self._template_env = _default_template_env(loader)

    def get_template(self, path):
        """
        Retrieve the template at the specified path. Returns an instance of
        :py:class:`Jinja2.Template` by default, but may be overridden by
        subclasses.
        """
        return self._template_env.get_template(path)

    def get_template_loader(self):
        raise NotImplementedError("You must implement a get_template_loader "
                                  "method.")

    def render(self, processes, context):
        raise NotImplementedError("You must implement a render method.")


class File(object):
    def __init__(self, name, content, executable=False):
        self.name = name
        self.content = content
        self.executable = executable


def dashrepl(value):
    """
    Replace any non-word characters with a dash.
    """
    patt = re.compile(r'\W', re.UNICODE)
    return re.sub(patt, '-', value)


def percentescape(value):
    """
    Double any % signs.
    """
    return value.replace('%', '%%')


def _default_template_env(loader):
    env = jinja2.Environment(loader=loader)
    env.filters['shellquote'] = shlex.quote
    env.filters['dashrepl'] = dashrepl
    env.filters['percentescape'] = percentescape
    return env
