# coding=utf-8

import collections
import textwrap

import pytest

from honcho import environ


@pytest.mark.parametrize('content,commands', [
    [
        """
        FOO=bar
        """,
        {'FOO': 'bar'}
    ],
    [
        """
        FOO=bar
        BAZ=qux
        """,
        {'FOO': 'bar', 'BAZ': 'qux'}
    ],
    [
        # No newline at EOF
        """
        FOO=bar""",
        {'FOO': 'bar'}
    ],
    [
        # Comments
        """
        #commented: command
        """,
        {}
    ],
    [
        # Invalid characters
        """
        -foo=command
        """,
        {}
    ],
    [
        # Single quoted
        """
        MYVAR='hello"world'
        """,
        {'MYVAR': 'hello"world'}
    ],
    [
        # Double quoted
        """
        MYVAR="hello'world"
        """,
        {'MYVAR': "hello'world"}
    ],
    [
        # Quotation mark surrounded
        r"""
        MYVAR='"surrounded"'
        """,
        {'MYVAR': '"surrounded"'}
    ],
    [
        # Escaped quotation mark surrounded
        r"""
        MYVAR=\"escaped\"
        """,
        {'MYVAR': '"escaped"'}
    ],
    [
        # At-sign in value
        r"""
        MYVAR=user@domain.com
        """,
        {'MYVAR': 'user@domain.com'}
    ],
    [
        # Much punctuation in value
        r"""
        MYVAR=~pun|u@|0n$=
        """,
        {'MYVAR': '~pun|u@|0n$='}
    ],
    [
        # Unicode values
        r"""
        MYVAR=⋃ñᴉ—☪ó∂ǝ
        """,  # noqa: RUF001
        {'MYVAR': '⋃ñᴉ—☪ó∂ǝ'}  # noqa: RUF001
    ],
    [
        # Unicode keys
        r"""
        ṀẎṾẠṚ=value
        """,
        {}
    ],
    [
        # Quoted space in value
        r"""
        MYVAR='sp ace'
        """,
        {'MYVAR': 'sp ace'}
    ],
    [
        # Escaped characters in value
        r"""
        TABS='foo\tbar'
        NEWLINES='foo\nbar'
        DOLLAR='foo\$bar'
        """,
        {'TABS': 'foo\tbar',
         'NEWLINES': 'foo\nbar',
         'DOLLAR': 'foo\\$bar'}
    ],
])
def test_environ_parse(content, commands):
    content = textwrap.dedent(content)
    result = environ.parse(content)
    assert result == commands


@pytest.mark.parametrize('content,processes', [
    [
        # Simple
        """
        web: command
        """,
        {'web': 'command'}
    ],
    [
        # Simple 2
        """
        foo: python foo.py
        bar: python bar.py
        """,
        {'foo': 'python foo.py', 'bar': 'python bar.py'}
    ],
    [
        # No newline at EOF
        """
        web: command""",
        {'web': 'command'}
    ],
    [
        # Comments
        """
        #commented: command
        """,
        {}
    ],
    [
        # Invalid characters
        """
        +foo: command
        """,
        {}
    ],
    [
        # Valid -/_ characters
        """
        -foo_bar: command
        """,
        {'-foo_bar': 'command'}
    ],
    [
        # Shell metacharacters
        """
        web: sh -c "echo $FOOBAR" >/dev/null 2>&1
        """,
        {'web': 'sh -c "echo $FOOBAR" >/dev/null 2>&1'}
    ],
])
def test_parse_procfile(content, processes):
    content = textwrap.dedent(content)
    p = environ.parse_procfile(content)
    assert p.processes == processes


def test_parse_procfile_ordered():
    content = textwrap.dedent("""
    one: onecommand
    two: twocommand
    three: twocommand
    four: fourcommand
    """)

    p = environ.parse_procfile(content)
    order = [k for k in p.processes]
    assert order == ['one', 'two', 'three', 'four']


class TestProcfile(object):
    def test_has_no_processes_after_init(self):
        p = environ.Procfile()
        assert len(p.processes) == 0

    def test_add_process(self):
        p = environ.Procfile()
        p.add_process('foo', 'echo 123')
        assert 'echo 123' == p.processes['foo']

    def test_add_process_ensures_unique_name(self):
        p = environ.Procfile()
        p.add_process('foo', 'echo 123')
        with pytest.raises(AssertionError):
            p.add_process('foo', 'echo 123')


def ep(*args, **kwargs):
    return environ.expand_processes(collections.OrderedDict(args), **kwargs)


def test_expand_processes_name():
    p = ep(("foo", "some command"))
    assert len(p) == 1
    assert p[0].name == "foo.1"


def test_expand_processes_name_multiple():
    p = ep(("foo", "some command"), ("bar", "another command"))
    assert len(p) == 2
    assert p[0].name == "foo.1"
    assert p[1].name == "bar.1"


def test_expand_processes_name_concurrency():
    p = ep(("foo", "some command"), concurrency={"foo": 3})
    assert len(p) == 3
    assert p[0].name == "foo.1"
    assert p[1].name == "foo.2"
    assert p[2].name == "foo.3"


def test_expand_processes_name_concurrency_multiple():
    p = ep(("foo", "some command"), ("bar", "another command"),
           concurrency={"foo": 3, "bar": 2})
    assert len(p) == 5
    assert p[0].name == "foo.1"
    assert p[1].name == "foo.2"
    assert p[2].name == "foo.3"
    assert p[3].name == "bar.1"
    assert p[4].name == "bar.2"


def test_expand_processes_command():
    p = ep(("foo", "some command"))
    assert p[0].cmd == "some command"


def test_expand_processes_port_not_defaulted():
    p = ep(("foo", "some command"))
    assert "PORT" not in p[0].env


def test_expand_processes_port():
    p = ep(("foo", "some command"), port=8000)
    assert p[0].env["PORT"] == "8000"


def test_expand_processes_port_multiple():
    p = ep(("foo", "some command"),
           ("bar", "another command"),
           port=8000)
    assert p[0].env["PORT"] == "8000"
    assert p[1].env["PORT"] == "8100"


def test_expand_processes_port_from_env():
    p = ep(("foo", "some command"),
           ("bar", "another command"),
           env={"PORT": 8000})
    assert p[0].env["PORT"] == "8000"
    assert p[1].env["PORT"] == "8100"


def test_expand_processes_port_from_env_coerced_to_number():
    p = ep(("foo", "some command"), env={"PORT": "5000"})
    assert p[0].env["PORT"] == "5000"


def test_expand_processes_port_from_env_overrides():
    p = ep(("foo", "some command"), env={"PORT": 5000}, port=8000)
    assert p[0].env["PORT"] == "5000"


def test_expand_processes_port_concurrency():
    p = ep(("foo", "some command"),
           ("bar", "another command"),
           concurrency={"foo": 3, "bar": 2},
           port=4000)
    assert p[0].env["PORT"] == "4000"
    assert p[1].env["PORT"] == "4001"
    assert p[2].env["PORT"] == "4002"
    assert p[3].env["PORT"] == "4100"
    assert p[4].env["PORT"] == "4101"


def test_expand_processes_quiet():
    p = ep(("foo", "some command"), quiet=["foo", "bar"])
    assert p[0].quiet


def test_expand_processes_quiet_multiple():
    p = ep(("foo", "some command"),
           ("bar", "another command"),
           quiet=["foo"])
    assert p[0].quiet
    assert not p[1].quiet


def test_expand_processes_env():
    p = ep(("foo", "some command"),
           env={"ANIMAL": "giraffe", "DEBUG": "false"})
    assert p[0].env["ANIMAL"] == "giraffe"
    assert p[0].env["DEBUG"] == "false"


def test_expand_processes_env_multiple():
    p = ep(("foo", "some command"),
           ("bar", "another command"),
           env={"ANIMAL": "giraffe", "DEBUG": "false"})
    assert p[0].env["ANIMAL"] == "giraffe"
    assert p[0].env["DEBUG"] == "false"
    assert p[1].env["ANIMAL"] == "giraffe"
    assert p[1].env["DEBUG"] == "false"


def test_set_env_process_name():
    p = ep(("foo", "some command"),
           ("bar", "another command"),
           concurrency={"foo": 3, "bar": 2})
    assert p[0].env["HONCHO_PROCESS_NAME"] == "foo.1"
    assert p[1].env["HONCHO_PROCESS_NAME"] == "foo.2"
    assert p[2].env["HONCHO_PROCESS_NAME"] == "foo.3"
    assert p[3].env["HONCHO_PROCESS_NAME"] == "bar.1"
    assert p[4].env["HONCHO_PROCESS_NAME"] == "bar.2"
