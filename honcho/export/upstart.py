from itertools import groupby

import jinja2

from honcho.export.base import BaseExport, File, dashrepl


class Export(BaseExport):
    def get_template_loader(self):
        return jinja2.PackageLoader(__name__, 'templates/upstart')

    def render(self, processes, context):
        master_tpl = self.get_template('master.conf')
        process_master_tpl = self.get_template('process_master.conf')
        process_tpl = self.get_template('process.conf')

        groups = groupby(processes, lambda p: p.name.split('.')[0])

        master = "{0}.conf".format(context['app'])
        yield File(master, master_tpl.render(context))

        for name, procs in groups:
            group_name = "{0}-{1}".format(context['app'], name)

            ctx = context.copy()
            ctx.update({'group_name': group_name})

            process_master = "{0}.conf".format(group_name)
            yield File(process_master, process_master_tpl.render(ctx))

            for p in procs:
                ctx = context.copy()
                ctx.update({'group_name': group_name,
                            'process': p})
                process = "{0}-{1}.conf".format(context['app'],
                                                dashrepl(p.name))
                yield File(process, process_tpl.render(ctx))
