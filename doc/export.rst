Exporting
=========

Honcho allows you to export your Procfile configuration into other formats.
Basic usage::

  $ honcho export FORMAT LOCATION

Exporters for ``upstart``, ``supervisord``, ``runit`` and ``systemd`` formats are shipped with Honcho.


Examples
--------

The following command will create a :file:`myapp.conf` file in the
:file:`/etc/supervisor/conf.d` directory::

    $ honcho export -a myapp supervisord /etc/supervisor/conf.d

Or, for the upstart exporter::

    $ honcho export -a myapp upstart /etc/init

By default, one of each process type will be started. You can change this by
specifying the ``--concurrency`` option to ``honcho export``.


Adding support for new export formats
-------------------------------------

You can add support for new export formats by writing plugins. Honcho discovers
export plugins with the `entry points mechanism`_ of setuptools. Export plugins
take the form of a class with ``render`` and ``get_template_loader`` methods
that inherits from :class:`honcho.export.base.BaseExport`. Inside the
:meth:`~honcho.export.base.BaseExport.render` method, you can fetch templates
using the `~honcho.export.base.BaseExport.get_template` method.

For example, here is a hypothetical exporter that writes out simple shell
scripts for each process:

.. code-block:: python

    import jinja2

    from honcho.export.base import BaseExport

    class SimpleExport(BaseExport):
        def get_template_loader(self):
            return jinja2.PackageLoader(package_name=__package__,
                                        package_path='templates')

        def render(self, processes, context):
            tpl = get_template('run.sh')

            for p in processes:
                filename = 'run-{0}.sh'.format(p.name)
                ctx = context.copy()
                ctx['process'] = p
                script = tpl.render(ctx)

By writing an exporter in this way (specifically, by inheriting
:class:`~honcho.export.base.BaseExport`), you make it possible for users of your
exporter to override the exporter's default templates using the
``--template-dir`` option to ``honcho export``.

In order for your export plugin to be detected by Honcho, you will need to
register your exporter class under the ``honcho_exporters`` entrypoint. If we
were shipping our hypothetical ``SimpleExport`` class in a package called
``honcho_export_simple``, our ``setup.py`` might look something like the
following:

.. code-block:: python

    from setuptools import setup

    setup(
        name='honcho_export_simple',
        ...
        entry_points={
            'honcho_exporters': [
                'simple=honcho_export_simple:SimpleExport',
            ],
        },
    )

After installing the package, the new export format will be shown by the
``honcho export`` command.

.. _`entry points mechanism`: https://pythonhosted.org/setuptools/setuptools.html#dynamic-discovery-of-services-and-plugins
