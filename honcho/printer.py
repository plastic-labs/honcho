import sys
from collections import namedtuple

from .compat import ON_WINDOWS

Message = namedtuple("Message", "type data time name colour")


class Printer(object):
    """
    Printer is where Honcho's user-visible output is defined. A Printer
    instance receives typed messages and prints them to its output (usually
    STDOUT) in the Honcho format.
    """

    def __init__(self,
                 output=sys.stdout,
                 time_format="%H:%M:%S",
                 width=0,
                 colour=True,
                 prefix=True):
        self.output = output
        self.time_format = time_format
        self.width = width
        self.colour = colour
        self.prefix = prefix

        try:
            # We only want to print coloured messages if the given output supports
            # ANSI escape sequences. Usually, testing if it is a TTY is safe enough.
            self._colours_supported = self.output.isatty()
        except AttributeError:
            # If the given output does not implement isatty(), we assume that it
            # is not able to handle ANSI escape sequences.
            self._colours_supported = False

    def write(self, message):
        if message.type != 'line':
            raise RuntimeError('Printer can only process messages of type "line"')

        name = message.name if message.name is not None else ""
        name = name.ljust(self.width)
        if name:
            name += " "

        # When encountering data that cannot be interpreted as UTF-8 encoded
        # Unicode, Printer will replace the unrecognisable bytes with the
        # Unicode replacement character (U+FFFD).
        if isinstance(message.data, bytes):
            string = message.data.decode("utf-8", "replace")
        else:
            string = message.data

        for line in string.splitlines():
            prefix = ''
            if self.prefix:
                time_formatted = message.time.strftime(self.time_format)
                prefix = '{time} {name}| '.format(time=time_formatted, name=name)
                if self.colour and self._colours_supported and message.colour:
                    prefix = _colour_string(message.colour, prefix)
            print(prefix + line, file=self.output, flush=True)


def _ansi(code):
    return '\033[{0}m'.format(code)


def _colour_string(colour, s):
    return '{0}{1}{2}{3}'.format(_ansi(0), _ansi(colour), s, _ansi(0))


if ON_WINDOWS:
    # The colorama package provides transparent support for ANSI colour codes
    # on Win32 platforms. We try and import and configure that, but fall back
    # to no colour if we fail.
    try:
        import colorama
    except ImportError:
        def _colour_string(colour, s):
            return s
    else:
        colorama.init()
