import pytest

from honcho import command


# All of the following groups of commands should parse to the same result
@pytest.mark.parametrize('commands', [
    # Regression test for #173
    [
        ['start', '-f', 'Procfile'],
        ['-f', 'Procfile', 'start'],
    ],
    # Regression tests for #223
    [
        ['start', '--no-prefix'],
        ['--no-prefix', 'start'],
    ],
    [
        ['start', '--no-colour'],
        ['--no-colour', 'start'],
    ],
    [
        ['start', '--no-prefix', '--no-colour'],
        ['--no-colour', 'start', '--no-prefix'],
        ['--no-prefix', 'start', '--no-colour'],
        ['--no-prefix','--no-colour', 'start'],
    ]
])
def test_command_equivalence(commands):
    if len(commands) < 2:
        pytest.fail("Must supply at least two commands")

    reference_args = commands.pop(0)
    reference_result = command.parser.parse_args(reference_args)

    for args in commands:
        result = command.parser.parse_args(args)
        assert result == reference_result


def test_port_precedence(monkeypatch):
    # Nothing specified -- should return default
    args = command.parser.parse_args(['start'])
    result = command.map_from(args)['port']
    assert result == '5000'

    # OS environment override
    monkeypatch.setenv('PORT', '4200')
    result = command.map_from(args)['port']
    assert result == '4200'

    # App environment override
    # FIXME: find a better test seam for this
    def _read_env(app_root, env):
        return {'PORT': '4300'}
    monkeypatch.setattr(command, '_read_env', _read_env)
    args = command.parser.parse_args(['start'])
    result = command.map_from(args)['port']
    assert result == '4300'

    # CLI override
    args = command.parser.parse_args(['start', '-p', '4400'])
    result = command.map_from(args)['port']
    assert result == '4400'


def test_procfile_precedence(monkeypatch):
    # Nothing specified -- should return default
    args = command.parser.parse_args(['start'])
    result = command.map_from(args)['procfile']
    assert result == 'Procfile'

    # OS environment override
    monkeypatch.setenv('PROCFILE', 'Procfile.osenv')
    result = command.map_from(args)['procfile']
    assert result == 'Procfile.osenv'

    # App environment override
    # FIXME: find a better test seam for this
    def _read_env(app_root, env):
        return {'PROCFILE': 'Procfile.appenv'}
    monkeypatch.setattr(command, '_read_env', _read_env)
    args = command.parser.parse_args(['start'])
    result = command.map_from(args)['procfile']
    assert result == 'Procfile.appenv'

    # CLI override
    args = command.parser.parse_args(['start', '-f', 'Procfile.cli'])
    result = command.map_from(args)['procfile']
    assert result == 'Procfile.cli'
