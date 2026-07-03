.. _using_procfiles:

Using Procfiles
===============

As described in :ref:`what_are_procfiles`, Procfiles are simple text files that
describe the components required to run an application. This document describes
some of the more advanced features of Honcho and the Procfile ecosystem.

Syntax
------

The basic syntax of a Procfile is described in `the Heroku Procfile
documentation
<https://devcenter.heroku.com/articles/procfile#declaring-process-types>`_. In
summary, a Procfile is a plain text file placed at the root of your applications
source tree that contains zero or more lines of the form::

    <process type>: <command>

The ``process type`` is a string which may contain alphanumerics as well as
underscores and dashes (``[A-Za-z0-9_-]+``), and uniquely identifies one type
of process which can be run to form your application. For example: ``web``,
``worker``, or ``my_process_123``.

``command`` is a shell commandline which will be executed to spawn a process of
the specified type.

Environment files
-----------------

You can also create a ``.env`` file alongside your Procfile which contains
environment variables which will be available to all processes started by
Honcho::

    $ cat >.env <<EOF
    RACK_ENV=production
    ASSET_ROOT=https://myapp.s3.amazonaws.com/assets
    PROCFILE=Procfile
    EOF

In addition to the variables specified in your ``.env`` file, the subprocess
environment will also contain a ``HONCHO_PROCESS_NAME`` variable that will be
set to a unique string composed of the process name as defined in the Procfile
and an integer counter that is incremented for each concurrent process of the
same type, for example: ``web.1``, ``web.2``, ``queue.1``, etc.

As shown, you may choose to specify your Procfile in the ``.env`` file.  This
takes priority over the default Procfile, but you can still use ``-f`` to replace
which Procfile to use.

Typically, you should not commit your ``.env`` file to your version control
repository, but you might wish to create a ``.env.example`` so that others
checking out your code can see what environment variables your application uses.

For more on why you might want to use environment variables to configure your
application, see Heroku's article on `configuration variables`_ and The
Twelve-Factor App's `guidance on configuration`_.

.. _configuration variables: https://devcenter.heroku.com/articles/config-vars
.. _guidance on configuration: http://12factor.net/config

Using Honcho
------------

To see the command line arguments accepted by Honcho, run it with the ``--help``
option::

    $ honcho --help
    usage: honcho [-h] [-e ENV] [-d DIR] [--no-colour] [--no-prefix] [-f FILE]
                  [-v]
                  {check,export,help,run,start,version} ...

    Manage Procfile-based applications

    optional arguments:
      -h, --help            show this help message and exit
      -e ENV, --env ENV     environment file[,file] (default: .env)
      -d DIR, --app-root DIR
                            procfile directory (default: .)
      --no-colour           disable coloured output
      --no-prefix           disable logging prefix
      -f FILE, --procfile FILE
                            procfile path (default: Procfile)
      -v, --version         show program's version number and exit

    tasks:
      {check,export,help,run,start,version}
        check               validate a Procfile
        export              export a Procfile to another format
        help                describe available tasks or one specific task
        run                 run a command using your application's environment
        start               start the application (or a specific PROCESS)
        version             display honcho version


You will notice that by default, Honcho will read a Procfile called
``Procfile`` from the current working directory, and will read environment from
a file called ``.env`` if one exists. You can override these options at the
command line if necessary. For example, if your application root is a level
above the current directory and your Procfile is called ``Procfile.dev``, you
could invoke Honcho thus::

    $ honcho -d .. -f Procfile.dev start
    16:14:49 web.1 | started with pid 1234
    ...

If you supply multiple comma-separated arguments to the ``-e`` option, Honcho will merge the environments provided by each of the files::

    $ echo 'ANIMAL_1=giraffe' >.env.one
    $ echo 'ANIMAL_2=elephant' >.env.two
    $ honcho -e .env.one,.env.two run sh -c 'env | grep -i animal'
    ANIMAL_1=giraffe
    ANIMAL_2=elephant

Differences to Foreman
----------------------

While Honcho is based heavily on the Foreman_ project, it is not necessarily our
intent to make Honcho feature-for-feature compatible with Foreman.

There are some important differences between the two tools, some of which are
simply the result of differences between Ruby and Python, and others are design
choices. The following is a non-exhaustive list of these differences:

.. _Foreman: https://github.com/ddollar/foreman

No `honcho run {target}`
''''''''''''''''''''''''

Foreman allows you to specify a Procfile target to both the `start` and `run`
subcommands. To me, it seems obvious that this functionality belongs only in
`honcho start`, a command that always reads the Procfile and has no other use
for its ARGV, as opposed to `honcho run`, which is intended for running a
shell command in the environment provided by Honcho and `.env` files. Because
I don't have to guess at whether or not ARGV is a process name or a shell
command, `honcho start` even supports multiple processes:
`honcho start web worker`.

Buffered output
'''''''''''''''

By default, Python will buffer a program's output more aggressively than Ruby
when a process has ``STDOUT`` connected to something other than a TTY. This can
catch people out when running Python programs through Honcho: if the program
only generates small amounts of output, it will be buffered, unavailable to
Honcho, and will not display.

One way around this is to set the ``PYTHONUNBUFFERED`` environment variable in
your ``Procfile`` or your ``.env`` file. Be sure you understand the performance
implications of unbuffered I/O if you do this.

For example::

    myprogram: PYTHONUNBUFFERED=true python myprogram.py
