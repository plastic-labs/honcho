from mock import MagicMock, patch

from honcho.export.supervisord import Export


class TestExportSupervisord():
    @patch.object(Export, 'get_template')
    def test_render_adds_processes_to_context(self, fake_get_template):
        export = Export()
        p1, p2 = object(), object()

        export.render([p1, p2], {'app': 'elephant'})

        fake_get_template.return_value.render.assert_called_with(
            {'app': 'elephant',
             'processes': [p1, p2]})

    def test_render_uses_app_name_as_filename(self):
        export = Export()
        export.get_template = MagicMock()

        results = export.render([], {'app': 'elephant'})

        assert results[0].name == 'elephant.conf'
