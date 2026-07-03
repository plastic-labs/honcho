ANSI_COLOURS = [
    'grey',
    'red',
    'green',
    'yellow',
    'blue',
    'magenta',
    'cyan',
    'white'
]

for i, name in enumerate(ANSI_COLOURS):
    globals()[name] = str(30 + i)
    globals()['intense_' + name] = str(30 + i) + ';1'


def get_colours():
    cs = ['cyan', 'yellow', 'green', 'magenta', 'red', 'blue',
          'intense_cyan', 'intense_yellow', 'intense_green',
          'intense_magenta', 'intense_red', 'intense_blue']
    cs = [globals()[c] for c in cs]

    i = 0
    while True:
        yield cs[i % len(cs)]
        i += 1
