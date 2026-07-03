from itertools import groupby

import jinja2

from honcho.export.base import BaseExport, File, dashrepl


class Export(BaseExport):
    """
    Exporter for Systemd

    """

    def get_template_loader(self):
        return jinja2.PackageLoader(__name__, 'templates/systemd')

    def render(self, processes, context):
        master_tpl = self.get_template('master.target')
        process_master_tpl = self.get_template('process_master.target')
        process_tpl = self.get_template('process.service')

        app_name = context['app']
        process_groups = groupby(processes, lambda proc: proc.name.split(".")[0])
        process_groups = [
            ("-".join([app_name, group_name]),
             [("-".join([app_name, dashrepl(proc.name)]), proc) for proc in procs])
            for group_name, procs in process_groups]

        master_wants = [".".join([group[0], "target"]) for group in process_groups]
        context['master_wants'] = " ".join(master_wants)
        context['process_groups'] = process_groups
        yield File("{0}.target".format(app_name), master_tpl.render(context))

        for process_master_name, proc_groups in process_groups:
            process_master_wants = [".".join([p[0], "service"]) for p in proc_groups]
            context['process_master_wants'] = " ".join(process_master_wants)
            context['process_master_name'] = process_master_name
            yield File("{0}.target".format(process_master_name), process_master_tpl.render(context))

            for process_name, process in proc_groups:
                context['process'] = process
                yield File("{0}.service".format(process_name), process_tpl.render(context))
