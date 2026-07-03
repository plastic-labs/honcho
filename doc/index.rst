Honcho: manage Procfile-based applications
==========================================

.. _introduction:

Welcome! This is the home of Honcho and its documentation. Honcho is:

1. A Python port of `David Dollar's`_ Foreman_: a command-line application which
   helps you manage and run `Procfile-based applications`_. It helps you
   simplify deployment and configuration of your applications in both
   development and production environments.
2. Secondarily, Honcho is a Python library/API for running multiple external
   processes and multiplexing their output.

The current version of Honcho is |release| and it can be downloaded from GitHub_
or installed using pip: see :ref:`installing_honcho`.

.. _David Dollar's: https://github.com/ddollar
.. _GitHub: https://github.com/nickstenning/honcho

Documentation index
-------------------

.. toctree::
   :maxdepth: 4

   Introduction <self>
   using_procfiles
   export
   contributing
   api
   authors

.. _what_are_procfiles:

What are Procfiles?
-------------------

A `Procfile`_ is a file which describes how to run your application. If you need
to run a simple web application, you might have a Procfile that looks like
this::

    web: python myapp.py

You'd then be able to run your application using the following command::

    $ honcho start

Now, if running your application is as simple as typing ``python myapp.py``,
then perhaps Honcho isn't that useful. But imagine that a few months have
passed, and running your application is now substantially more complicated. You
need to have the following running in parallel: a web server, a high priority
job queue worker, and a low priority job queue worker. In addition, you've
established that you need to run your application under a proper web server like
`gunicorn`_. Now the Procfile starts to be useful::

    web: gunicorn -b "0.0.0.0:$PORT" -w 4 myapp:app
    worker: python worker.py --priority high,med,low
    worker_low: python worker.py --priority med,low

Again, you can start all three processes with a single command::

    $ honcho start

Honcho also supports `runpy`_ invocation::

    $ python -m honcho start

As you add features to your application, you shouldn't be forced to bundle
everything up into a single process just to make the application easier to run.
The Procfile format allows you to specify how to run your application, even when
it's made up of multiple independent components. Honcho (and `Foreman`_, and
`Heroku`_) can parse the Procfile format and run your application.

.. _Procfile: https://devcenter.heroku.com/articles/procfile
.. _Procfile-based applications: Procfile_
.. _Foreman: https://ddollar.github.io/foreman/
.. _Heroku: https://heroku.com/
.. _gunicorn: http://gunicorn.org/
.. _runpy: https://docs.python.org/3/library/runpy.html

Why did you port Foreman?
-------------------------

`Foreman`_ is a great tool! The fact I chose to port it to Python shouldn't be
interpreted as saying anything negative about Foreman. But I have often worked
in Python-only development environments where I didn't want to install Ruby just
so I could run Procfile applications. Python, on the other hand, is part of the
`Linux Standard Base`_, so even in "Ruby-only" environments it will typically
still be installed.

(Oh, and I also I wanted to learn about `asynchronous I/O`_ `in Python`_.)

.. _Linux Standard Base: http://en.wikipedia.org/wiki/Linux_Standard_Base
.. _asynchronous I/O: http://docs.python.org/library/select.html
.. _in Python: http://docs.python.org/library/queue.html

.. _installing_honcho:

Installing Honcho
-----------------

If you have a working Python and `pip`_ installation, you should be able to
simply

::

    pip install honcho

and get a working installation of Honcho. You can probably also ``easy_install
honcho``. But please, don't: `get with the program`_.

.. _pip: http://www.pip-installer.org/en/latest/index.html
.. _get with the program: pip_

Further reading and assistance
------------------------------

For more about the Procfile format, ``.env`` files, and command-line options to
Honcho, see :ref:`using_procfiles`.

If you have any difficulty using Honcho or this documentation, please get in
touch with me, Nick Stenning, on Twitter at `@nickstenning
<https://twitter.com/nickstenning>`_ or by email: ``<my first name> at whiteink
dot com``.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

