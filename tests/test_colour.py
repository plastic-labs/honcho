from honcho import colour


def test_colours():
    assert colour.red == '31'
    assert colour.intense_red == '31;1'
    assert colour.cyan == '36'
    assert colour.intense_cyan == '36;1'


def test_get_colours():
    gen = colour.get_colours()

    expect = ['cyan', 'yellow', 'green', 'magenta', 'red', 'blue']
    expect = [getattr(colour, x) for x in expect]
    actual = [next(gen) for _ in range(6)]

    assert expect == actual
