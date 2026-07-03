import textwrap

import pytest


@pytest.mark.parametrize('testenv', [{
    'Procfile': textwrap.dedent("""
        foo: python web.py
        bar: ruby worker.rb
    """)
}], indirect=True)
def test_main(testenv):
    result = testenv.run(['python', '-m', 'honcho', 'check'], check=True)

    assert 'Valid procfile detected' in result.stderr
    assert 'foo, bar' in result.stderr
