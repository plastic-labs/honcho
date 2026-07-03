import jinja2

from honcho.export.base import BaseExport, File


class Export(BaseExport):
    def get_template_loader(self):
        return jinja2.PackageLoader(__name__, 'templates/supervisord')

    def render(self, processes, context):
        context['processes'] = processes
        filename = "{0}.conf".format(context['app'])
        template = self.get_template('supervisord.conf')
        return [File(filename, template.render(context))]
