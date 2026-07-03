import datetime

import pytest

from honcho.printer import Message, Printer


def fake_message(data, **kwargs):
    defaults = {
        'type': 'line',
        'data': data,
        'time': datetime.datetime(2012, 8, 11, 12, 42),
        'name': None,
        'colour': None,
    }
    defaults.update(kwargs)
    return Message(**defaults)


class FakeOutput(object):

    def __init__(self):
        self.out = []
        self.flushcount = 0

    def flush(self):
        self.flushcount += 1

    def write(self, data):
        self.out.append(data)

    def string(self):
        return "".join(self.out)


class FakeTTY(FakeOutput):

    def isatty(self):
        return True


class TestPrinter(object):
    def test_write(self):
        out = FakeOutput()
        p = Printer(output=out)
        p.write(fake_message("monkeys\n"))
        assert out.string() == "12:42:00 | monkeys\n"

    def test_write_wrong_type(self):
        out = FakeOutput()
        p = Printer(output=out)
        with pytest.raises(RuntimeError):
            p.write(fake_message("monkeys\n", type="error"))

    def test_write_invalid_utf8(self):
        out = FakeOutput()
        p = Printer(output=out)
        p.write(fake_message(b"\xfe\xff\n"))
        assert out.string() == "12:42:00 | \ufffd\ufffd\n"

    def test_write_no_newline(self):
        out = FakeOutput()
        p = Printer(output=out)
        p.write(fake_message("monkeys"))
        assert out.string() == "12:42:00 | monkeys\n"

    def test_write_multiline(self):
        out = FakeOutput()
        p = Printer(output=out)
        p.write(fake_message("one\ntwo\nthree\n"))
        expect = "12:42:00 | one\n12:42:00 | two\n12:42:00 | three\n"
        assert out.string() == expect

    def test_write_with_name(self):
        out = FakeOutput()
        p = Printer(output=out)
        p.write(fake_message("quiescent\n", name="Robert Louis Stevenson"))
        assert out.string() == "12:42:00 Robert Louis Stevenson | quiescent\n"

    def test_write_with_set_width(self):
        out = FakeOutput()
        p = Printer(output=out, width=6)
        p.write(fake_message("giraffe\n"))
        assert out.string() == "12:42:00        | giraffe\n"

    def test_write_with_name_and_set_width(self):
        out = FakeOutput()
        p = Printer(output=out, width=6)
        p.write(fake_message("narcissist\n", name="oop"))
        assert out.string() == "12:42:00 oop    | narcissist\n"

    def test_write_with_colour_tty(self):
        out = FakeTTY()
        p = Printer(output=out)
        p.write(fake_message("conflate\n", name="foo", colour="31"))
        assert out.string() == "\033[0m\033[31m12:42:00 foo | \033[0mconflate\n"

    def test_write_with_colour_non_tty(self):
        out = FakeOutput()
        p = Printer(output=out)
        p.write(fake_message("conflate\n", name="foo", colour="31"))
        assert out.string() == "12:42:00 foo | conflate\n"

    def test_write_without_prefix_tty(self):
        out = FakeTTY()
        p = Printer(output=out, prefix=False, colour=True)
        p.write(fake_message("paranoid android\n", name="foo", colour="31"))
        assert out.string() == "paranoid android\n"

    def test_write_without_prefix_and_colour_tty(self):
        out = FakeTTY()
        p = Printer(output=out, prefix=False, colour=False)
        p.write(fake_message("paranoid android\n", name="foo", colour="31"))
        assert out.string() == "paranoid android\n"

    def test_write_without_colour_tty(self):
        out = FakeTTY()
        p = Printer(output=out, prefix=True, colour=False)
        p.write(fake_message("paranoid android\n", name="foo", colour="31"))
        assert out.string() == "12:42:00 foo | paranoid android\n"

    def test_write_without_prefix_non_tty(self):
        out = FakeOutput()
        p = Printer(output=out, prefix=False)
        p.write(fake_message("paranoid android\n", name="foo", colour="31"))
        assert out.string() == "paranoid android\n"

    def test_write_flushes_output(self):
        out = FakeOutput()
        p = Printer(output=out, prefix=False)
        p.write(fake_message("paranoid android\n"))
        assert out.flushcount == 1
