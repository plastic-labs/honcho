============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/nickstenning/honcho/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Honcho could always use more documentation, whether as part of the
official honcho docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/nickstenning/honcho/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `honcho` for local development.

1. Fork the `honcho` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/honcho.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv honcho
    $ cd honcho/
    $ pip install -e .[export] tox

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the tests, including testing other Python versions with tox and just run::

    $ tox


6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function or class with a docstring.
3. The pull request should work for all supported Python versions. Check
   https://travis-ci.org/nickstenning/honcho/pull_requests and make sure that
   all the tests pass.

Tips
----

If you'd like to run a specific tox environment just use ``-e`` flag e.g.::

    tox -e py39

This will run tests using python3.9 interpreter.

To list all available tox environments run::

    tox -l

Honcho's tox setup uses `pytest`_ to run the test suite. You can pass positional
arguments to a pytest command within tox. For example, if you'd like to use
pytest's ``-x`` flag (stop after first error) with a PyPy interpreter you could
do this::

    tox -e pypy -- -x

.. _pytest: https://pytest.org/
