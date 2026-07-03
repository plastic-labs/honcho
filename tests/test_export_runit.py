import collections

import pytest
from mock import patch

from honcho.export.runit import Export

FakeProcess = collections.namedtuple('FakeProcess', 'name env')

FIX_1PROC = [FakeProcess('web.1', {'FOO': 'bar'})]


class TestExportRunit(object):
    @pytest.fixture(autouse=True)
    def exporter(self, request):
        self.export = Export()

        get_template_patcher = patch.object(Export, 'get_template')
        self.get_template = get_template_patcher.start()
        request.addfinalizer(get_template_patcher.stop)

    def test_render(self):
        out = list(self.export.render(FIX_1PROC, {'app': 'elephant'}))

        run = _files_named('elephant-web-1/run', out)
        log = _files_named('elephant-web-1/log/run', out)
        assert len(run) == 1
        assert run[0].executable
        assert len(log) == 1
        assert log[0].executable

    def test_render_env(self):
        out = list(self.export.render(FIX_1PROC, {'app': 'elephant'}))

        env = _files_named('elephant-web-1/env/FOO', out)
        assert len(env) == 1
        assert env[0].content == 'bar'


def _files_named(name, lst):
    return [f for f in lst if f.name == name]
