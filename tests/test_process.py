import datetime

import pytest

from honcho.process import Process


class FakeClock(object):

    def now(self):
        return datetime.datetime(2012, 8, 11, 12, 42)


class FakePopen(object):

    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.stdout = FakeOutput()
        self.pid = 0x42
        self.returncode = 0

    def wait(self):
        return self.returncode


class FakePrinter(object):

    def __init__(self, width=0):
        self.lines = []
        self.width = width

    def write(self, message):
        self.lines.append(message)

    def got_line(self, data):
        return self.find_line(data) is not None

    def find_line(self, data):
        for line in self.lines:
            if line.data == data:
                return line
        return None


class FakeQueue(object):

    def __init__(self):
        self.messages = []

    def put(self, msg):
        self.messages.append(msg)

    def got_message(self, data):
        return self.find_message(data) is not None

    def find_message(self, data):
        for msg in self.messages:
            if msg.data == data:
                return msg
        return None


class FakeOutput(object):
    """
    FakeOutput accepts a set of lines to emit in response to a call to
    #readline(), after which it will emit b''.
    """

    def __init__(self, lines=None):
        self.lines = lines if lines is not None else []

    def readline(self):
        try:
            return self.lines.pop(0)
        except IndexError:
            return b''

    def close(self):
        pass


class TestProcess(object):

    @pytest.fixture(autouse=True)
    def queue(self):
        self.q = FakeQueue()

    def test_ctor_cmd(self):
        proc = Process('echo 123')
        assert proc.cmd == 'echo 123'

    def test_ctor_name(self):
        proc = Process('echo 123', name='foo')
        assert proc.name == 'foo'

    def test_ctor_colour(self):
        proc = Process('echo 123', colour='red')
        assert proc.colour == 'red'

    def test_ctor_quiet(self):
        proc = Process('echo 123', quiet=True)
        assert proc.quiet

    def test_output_receives_start_with_pid(self):
        proc = Process('echo 123')
        proc._child_ctor = FakePopen
        proc.run(self.q)
        msg = self.q.messages[0]
        assert msg.type == 'start'
        assert msg.data == {'pid': 66}

    def test_message_contains_name(self):
        proc = Process('echo 123', name="barry")
        proc._child_ctor = FakePopen
        proc.run(self.q)
        msg = self.q.messages[0]
        assert msg.name == "barry"

    def test_message_contains_time(self):
        proc = Process('echo 123')
        proc._clock = FakeClock()
        proc._child_ctor = FakePopen
        proc.run(self.q)
        msg = self.q.messages[0]
        assert msg.time == datetime.datetime(2012, 8, 11, 12, 42)

    def test_message_contains_colour(self):
        proc = Process('echo 123', colour="red")
        proc._child_ctor = FakePopen
        proc.run(self.q)
        msg = self.q.messages[0]
        assert msg.colour == "red"

    def test_output_receives_lines(self):
        def _ctor(*args, **kwargs):
            popen = FakePopen(*args, **kwargs)
            popen.stdout = FakeOutput(lines=[b"hello\n", b"world\n"])
            return popen

        proc = Process('echo 123')
        proc._child_ctor = _ctor
        proc.run(self.q)
        assert self.q.got_message(b"hello\n")
        assert self.q.got_message(b"world\n")

    def test_output_receives_lines_invalid_utf8(self):
        def _ctor(*args, **kwargs):
            popen = FakePopen(*args, **kwargs)
            popen.stdout = FakeOutput(lines=[b"\xfe\xff\n"])
            return popen

        proc = Process('echo 123')
        proc._child_ctor = _ctor
        proc.run(self.q)
        assert self.q.got_message(b"\xfe\xff\n")

    def test_output_does_not_receive_lines_when_quiet(self):
        def _ctor(*args, **kwargs):
            popen = FakePopen(*args, **kwargs)
            popen.stdout = FakeOutput(lines=[b"hello\n", b"world\n"])
            return popen

        proc = Process('echo 123', quiet=True)
        proc._child_ctor = _ctor
        proc.run(self.q)
        assert not self.q.got_message(b"hello\n")
        assert not self.q.got_message(b"world\n")

    def test_output_receives_stop(self):
        proc = Process('echo 123')
        proc._child_ctor = FakePopen
        proc.run(self.q)
        msg = self.q.messages[-1]
        assert msg.type == 'stop'

    def test_output_receives_stop_with_returncode(self):
        def _ctor(*args, **kwargs):
            popen = FakePopen(*args, **kwargs)
            popen.returncode = 42
            return popen

        proc = Process('echo 123')
        proc._child_ctor = _ctor
        proc.run(self.q)
        msg = self.q.find_message({'returncode': 42})
        assert msg.type == 'stop'

    def test_cwd_passed_along(self):
        proc = Process('echo 123', cwd='fake-dir')
        proc._child_ctor = FakePopen
        proc.run(self.q)
        assert proc._child.kwargs['cwd'] == 'fake-dir'
