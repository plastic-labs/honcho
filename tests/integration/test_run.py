import sys
import textwrap

import pytest

python_bin = sys.executable

script = textwrap.dedent("""
    import os
    import sys
    print(os.environ.get("ANIMAL", "elephant"))
    print("error output", file=sys.stderr)
""")


@pytest.mark.parametrize('testenv', [{
    'test.py': script
}], indirect=True)
def test_run(testenv):
    ret, out, err = testenv.run_honcho(['run', python_bin, 'test.py'])

    assert ret == 0
    assert out == 'elephant\n'
    assert 'error output\n' in err


@pytest.mark.parametrize('testenv', [{
    '.env': 'ANIMAL=giraffe',
    'test.py': script,
}], indirect=True)
def test_run_env(testenv):
    ret, out, err = testenv.run_honcho(['run', python_bin, 'test.py'])

    assert ret == 0
    assert out == 'giraffe\n'


@pytest.mark.parametrize('testenv', [{
    '.env.x': 'ANIMAL=giraffe',
    'test.py': script,
}], indirect=True)
def test_run_args_before_command(testenv):
    # Regression test for #122 -- ensure that common args can be given
    # before the subcommand.
    ret, out, err = testenv.run_honcho(['-e', '.env.x',
                                        'run', python_bin, 'test.py'])

    assert ret == 0
    assert out == 'giraffe\n'


@pytest.mark.parametrize('testenv', [{
    'test.py': script
}], indirect=True)
def test_run_options_positionals_separator(testenv):
    # Regression test for #159 -- ensure that honcho handles the '--'
    # options/positionals separator correctly.
    ret, out, err = testenv.run_honcho(['run', '--', python_bin, 'test.py'])

    assert ret == 0
    assert out == 'elephant\n'
