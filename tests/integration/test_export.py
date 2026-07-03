import os

import pytest


@pytest.mark.parametrize('testenv', [{
    'Procfile': "web: python web.py"
}], indirect=True)
def test_export_supervisord(testenv):
    ret, out, err = testenv.run_honcho([
        'export',
        'supervisord',
        testenv.path('giraffe'),
        '-a', 'neck',
    ])

    expected = testenv.path('giraffe', 'neck.conf')

    assert ret == 0
    assert os.path.exists(expected)


@pytest.mark.parametrize('testenv', [{
    'Procfile': "web: python web.py"
}], indirect=True)
def test_export_upstart(testenv):
    ret, out, err = testenv.run_honcho([
        'export',
        'upstart',
        testenv.path('elephant'),
        '-a', 'trunk',
    ])

    assert ret == 0
    for filename in ('trunk.conf',
                     'trunk-web.conf',
                     'trunk-web-1.conf'):
        expected = testenv.path('elephant', filename)
        assert os.path.exists(expected)


@pytest.mark.parametrize('testenv', [{
    'Procfile': "web: python web.py",
    '.env': """
NORMAL=ok
SQ_SPACES='sqspace sqspace'
DQ_SPACES="dqspace dqspace"
SQ="it's got single quotes"
DQ='it has "double" quotes'
EXCL='an exclamation mark!'
SQ_DOLLAR='costs $UNINTERPOLATED amount'
DQ_DOLLAR="costs $UNINTERPOLATED amount"
"""
}], indirect=True)
def test_export_upstart_environment(testenv):
    ret, out, err = testenv.run_honcho([
        'export',
        'upstart',
        testenv.path('test'),
        '-a', 'envvars',
    ])

    assert ret == 0

    lines = open(testenv.path('test', 'envvars-web-1.conf')).readlines()
    assert 'env NORMAL=ok\n' in lines
    assert "env SQ_SPACES='sqspace sqspace'\n" in lines
    assert "env DQ_SPACES='dqspace dqspace'\n" in lines
    assert "env SQ='it'\"'\"'s got single quotes'\n" in lines
    assert "env DQ='it has \"double\" quotes'\n" in lines
    assert "env EXCL='an exclamation mark!'\n" in lines
    assert "env SQ_DOLLAR='costs $UNINTERPOLATED amount'\n" in lines
    assert "env DQ_DOLLAR='costs $UNINTERPOLATED amount'\n" in lines
