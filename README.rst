::

         ___           ___           ___           ___           ___           ___
        /\__\         /\  \         /\__\         /\  \         /\__\         /\  \
       /:/  /        /::\  \       /::|  |       /::\  \       /:/  /        /::\  \
      /:/__/        /:/\:\  \     /:|:|  |      /:/\:\  \     /:/__/        /:/\:\  \
     /::\  \ ___   /:/  \:\  \   /:/|:|  |__   /:/  \:\  \   /::\  \ ___   /:/  \:\  \
    /:/\:\  /\__\ /:/__/ \:\__\ /:/ |:| /\__\ /:/__/ \:\__\ /:/\:\  /\__\ /:/__/ \:\__\
    \/__\:\/:/  / \:\  \ /:/  / \/__|:|/:/  / \:\  \  \/__/ \/__\:\/:/  / \:\  \ /:/  /
         \::/  /   \:\  /:/  /      |:/:/  /   \:\  \            \::/  /   \:\  /:/  /
         /:/  /     \:\/:/  /       |::/  /     \:\  \           /:/  /     \:\/:/  /
        /:/  /       \::/  /        /:/  /       \:\__\         /:/  /       \::/  /
        \/__/         \/__/         \/__/         \/__/         \/__/         \/__/

|PyPI| |Build Status|

Honcho is a Python port of Foreman_, a tool for managing Procfile-based applications.

`Why a port? <//honcho.readthedocs.io/en/latest/#why-did-you-port-foreman>`_

.. _Foreman: https://ddollar.github.io/foreman

.. |Build Status| image:: https://github.com/nickstenning/honcho/actions/workflows/main.yml/badge.svg
   :target: https://github.com/nickstenning/honcho/actions
   :alt: Build Status

.. |PyPI| image:: https://img.shields.io/pypi/v/honcho.svg
   :target: https://pypi.python.org/pypi/honcho/
   :alt: Latest Version on PyPI

Installing Honcho
-----------------

::

    pip install honcho

How to use Honcho
-----------------

The 30-second version:

1. Write `a Procfile`_::

    $ cat >Procfile <<EOM
    web: python serve.py
    redis: redis-server
    EOM

2. *Optional:* write a .env file `to configure your app`_::

    $ cat >.env <<EOM
    PORT=6000
    REDIS_URI=redis://localhost:6789/0
    EOM

3. Run the app with Honcho::

    $ honcho start

.. _a Procfile: https://devcenter.heroku.com/articles/procfile
.. _to configure your app: https://www.12factor.net/config

For more detail and an explanation of the circumstances in which Honcho might
be useful, consult the `Honcho documentation`_.

.. _Honcho documentation: https://honcho.readthedocs.io/

License
-------

Honcho is released under the terms of the MIT license, a copy of which can be
found in ``LICENSE``.
