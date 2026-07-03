import textwrap

import pytest


@pytest.mark.parametrize('testenv', [{
    'Procfile': textwrap.dedent("""
        foo: python web.py
        bar: ruby worker.rb
        baz: node socket.js
    """)
}], indirect=True)
def test_check(testenv):
    ret, out, err = testenv.run_honcho(['check'])

    assert ret == 0
    assert 'Valid procfile detected' in err
    assert 'foo, bar, baz' in err
