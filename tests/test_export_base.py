# -*- coding: utf-8 -*-

import pytest
from mock import Mock, patch

from honcho.export.base import BaseExport, dashrepl, percentescape


class GiraffeExport(BaseExport):
    def get_template_loader(self):
        return 'longneck'


class TestBaseExport():
    @patch('jinja2.Environment')
    def test_env_default(self, env_mock):
        GiraffeExport()
        env_mock.assert_called_with(loader='longneck')

    @patch('jinja2.Environment')
    @patch('jinja2.FileSystemLoader')
    def test_env_template_dir(self, loader_mock, env_mock):
        BaseExport(template_dir='foo/bar')
        loader_mock.assert_called_with(['foo/bar'])
        env_mock.assert_called_with(loader=loader_mock.return_value)

    def test_get_template(self):
        fake_env = Mock()
        export = BaseExport(template_env=fake_env)
        export.get_template('foo/bar.tpl')
        fake_env.get_template.assert_called_with('foo/bar.tpl')


@pytest.mark.parametrize('name_in,name_out', (
    ('foo', 'foo'),
    ('foo.1', 'foo-1'),
    ('foo.bar.baz', 'foo-bar-baz'),
    ('foo_bar_baz', 'foo_bar_baz'),
    ('foo!bar:baz', 'foo-bar-baz'),
    ('καλημέρα.κόσμε', 'καλημέρα-κόσμε'),
))
def test_dashrepl(name_in, name_out):
    assert dashrepl(name_in) == name_out


@pytest.mark.parametrize('val_in,val_out', (
    ('foo', 'foo'),
    ('foo name.%h', 'foo name.%%h'),
    ('%one two% %three', '%%one two%% %%three'),
    ('foo%%bar', 'foo%%%%bar'),
))
def test_percentescape(val_in, val_out):
    assert percentescape(val_in) == val_out
