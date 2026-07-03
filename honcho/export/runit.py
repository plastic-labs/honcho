import os

import jinja2

from honcho.export.base import BaseExport, File, dashrepl


class Export(BaseExport):
    """
    Exporter for runit_, the cross-platform Unix init system.

    .. _runit: http://smarden.org/runit/
    """
    def get_template_loader(self):
        return jinja2.PackageLoader(__name__, 'templates/runit')

    def render(self, processes, context):
        run_tpl = self.get_template('run')
        log_tpl = self.get_template('log/run')

        for p in processes:
            ctx = context.copy()
            ctx.update({'process': p})

            name = '{0}-{1}'.format(ctx['app'], dashrepl(p.name))

            # Create an envdir
            for k, v in p.env.items():
                yield File(os.path.join(name, 'env', k), v)

            # Startup file
            yield File(os.path.join(name, 'run'),
                       run_tpl.render(ctx),
                       executable=True)

            # Logger
            yield File(os.path.join(name, 'log', 'run'),
                       log_tpl.render(ctx),
                       executable=True)
