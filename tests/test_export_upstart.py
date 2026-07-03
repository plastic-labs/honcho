import collections

import pytest
from mock import MagicMock, call, patch

from honcho.export.upstart import Export

FakeProcess = collections.namedtuple('FakeProcess', 'name')

FIX_1PROC = [FakeProcess('web.1')]
FIX_NPROC = [FakeProcess('web.1'),
             FakeProcess('worker.1'),
             FakeProcess('worker.2')]


class TestExportUpstart(object):
    @pytest.fixture(autouse=True)
    def exporter(self, request):
        self.export = Export()

        self.master = MagicMock()
        self.process_master = MagicMock()
        self.process = MagicMock()

        def _get_template(name):
            if name.endswith('process_master.conf'):
                return self.process_master
            elif name.endswith('process.conf'):
                return self.process
            elif name.endswith('master.conf'):
                return self.master
            else:
                raise RuntimeError("tests don't know about that template")

        get_template_patcher = patch.object(Export, 'get_template')
        self.get_template = get_template_patcher.start()
        self.get_template.side_effect = _get_template
        request.addfinalizer(get_template_patcher.stop)

    @pytest.fixture(autouse=True)
    def file(self, request):
        file_patcher = patch('honcho.export.upstart.File')
        self.File = file_patcher.start()
        request.addfinalizer(file_patcher.stop)

    def test_render_master(self):
        out = list(self.export.render(FIX_1PROC, {'app': 'elephant'}))

        master = self.File('elephant.conf', self.master.render.return_value)
        assert master in out

        self.master.render.assert_called_once_with({'app': 'elephant'})

    def test_render_process_master(self):
        out = list(self.export.render(FIX_1PROC, {'app': 'elephant'}))

        process_master = self.File('elephant-web.conf',
                                   self.process_master.render.return_value)
        assert process_master in out

        expected = {'app': 'elephant',
                    'group_name': 'elephant-web'}
        self.process_master.render.assert_called_once_with(expected)

    def test_render_process(self):
        out = list(self.export.render(FIX_1PROC, {'app': 'elephant'}))

        process = self.File('elephant-web-1.conf',
                            self.process.render.return_value)
        assert process in out

        expected = {'app': 'elephant',
                    'group_name': 'elephant-web',
                    'process': FIX_1PROC[0]}
        self.process.render.assert_called_once_with(expected)

    def test_render_multiple_process_groups(self):
        out = list(self.export.render(FIX_NPROC, {'app': 'elephant'}))

        assert self.File('elephant-web.conf',
                         self.process_master.render.return_value) in out
        assert self.File('elephant-worker.conf',
                         self.process_master.render.return_value) in out

        expected = [call({'app': 'elephant',
                          'group_name': 'elephant-web'}),
                    call({'app': 'elephant',
                          'group_name': 'elephant-worker'})]
        assert self.process_master.render.call_args_list == expected

    def test_render_multiple_processes(self):
        out = list(self.export.render(FIX_NPROC, {'app': 'elephant'}))

        assert self.File('elephant-web-1.conf',
                         self.process.render.return_value) in out
        assert self.File('elephant-worker-1.conf',
                         self.process.render.return_value) in out
        assert self.File('elephant-worker-2.conf',
                         self.process.render.return_value) in out

        expected = [call({'app': 'elephant',
                          'group_name': 'elephant-web',
                          'process': FIX_NPROC[0]}),
                    call({'app': 'elephant',
                          'group_name': 'elephant-worker',
                          'process': FIX_NPROC[1]}),
                    call({'app': 'elephant',
                          'group_name': 'elephant-worker',
                          'process': FIX_NPROC[2]})]
        assert self.process.render.call_args_list == expected
