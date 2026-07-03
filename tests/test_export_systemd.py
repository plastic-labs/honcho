import collections

import pytest
from mock import MagicMock, call, patch

from honcho.export.systemd import Export

FakeProcess = collections.namedtuple("FakeProcess", "name")

FIX_1PROC = [FakeProcess("web.1")]
FIX_NPROC = [FakeProcess("web.1"), FakeProcess("worker.1"), FakeProcess("worker.2")]


class TestExportSystemd(object):
    @pytest.fixture(autouse=True)
    def exporter(self, request):
        self.export = Export()

        self.master = MagicMock()
        self.process_master = MagicMock()
        self.process = MagicMock()

        def _get_template(name):
            if name.endswith("process_master.target"):
                return self.process_master
            elif name.endswith("process.service"):
                return self.process
            elif name.endswith("master.target"):
                return self.master
            else:
                raise RuntimeError("tests don't know about that template")

        get_template_patcher = patch.object(Export, "get_template")
        self.get_template = get_template_patcher.start()
        self.get_template.side_effect = _get_template
        request.addfinalizer(get_template_patcher.stop)

    @pytest.fixture(autouse=True)
    def file(self, request):
        file_patcher = patch("honcho.export.systemd.File")
        self.File = file_patcher.start()
        request.addfinalizer(file_patcher.stop)

    def test_render_master(self):
        out = list(self.export.render(FIX_1PROC, {"app": "elephant"}))

        master = self.File("elephant.target", self.master.render.return_value)
        assert master in out

        expected = {
            "app": "elephant",
            "master_wants": "elephant-web.target",
            "process_groups": [("elephant-web", [("elephant-web-1", FIX_1PROC[0])])],
            "process_master_name": "elephant-web",
            "process_master_wants": "elephant-web-1.service",
            "process": FIX_1PROC[0],
        }

        self.master.render.assert_called_once_with(expected)

    def test_render_process_master(self):
        out = list(self.export.render(FIX_1PROC, {"app": "elephant"}))

        process_master = self.File(
            "elephant-web.target", self.process_master.render.return_value
        )
        assert process_master in out

        expected = {
            "app": "elephant",
            "master_wants": "elephant-web.target",
            "process_groups": [("elephant-web", [("elephant-web-1", FIX_1PROC[0])])],
            "process_master_name": "elephant-web",
            "process_master_wants": "elephant-web-1.service",
            "process": FIX_1PROC[0],
        }
        self.process_master.render.assert_called_once_with(expected)

    def test_render_process(self):
        out = list(self.export.render(FIX_1PROC, {"app": "elephant"}))

        process = self.File("elephant-web-1.service", self.process.render.return_value)
        assert process in out

        expected = {
            "app": "elephant",
            "master_wants": "elephant-web.target",
            "process_groups": [("elephant-web", [("elephant-web-1", FIX_1PROC[0])])],
            "process_master_name": "elephant-web",
            "process_master_wants": "elephant-web-1.service",
            "process": FIX_1PROC[0],
        }
        self.process.render.assert_called_once_with(expected)

    def test_render_multiple_process_groups(self):
        out = list(self.export.render(FIX_NPROC, {"app": "elephant"}))

        assert (
            self.File("elephant-web.target", self.process_master.render.return_value)
            in out
        )
        assert (
            self.File("elephant-worker.target", self.process_master.render.return_value)
            in out
        )

        expected = [
            call(
                {
                    "app": "elephant",
                    "master_wants": "elephant-web.target elephant-worker.target",
                    "process_groups": [
                        ("elephant-web", [("elephant-web-1", FIX_NPROC[0])]),
                        (
                            "elephant-worker",
                            [
                                ("elephant-worker-1", FIX_NPROC[1]),
                                ("elephant-worker-2", FIX_NPROC[2]),
                            ],
                        ),
                    ],
                    "process_master_name": "elephant-worker",
                    "process_master_wants": "elephant-worker-1.service elephant-worker-2.service",
                    "process": FIX_NPROC[2],
                }
            ),
            call(
                {
                    "app": "elephant",
                    "master_wants": "elephant-web.target elephant-worker.target",
                    "process_groups": [
                        ("elephant-web", [("elephant-web-1", FIX_NPROC[0])]),
                        (
                            "elephant-worker",
                            [
                                ("elephant-worker-1", FIX_NPROC[1]),
                                ("elephant-worker-2", FIX_NPROC[2]),
                            ],
                        ),
                    ],
                    "process_master_name": "elephant-worker",
                    "process_master_wants": "elephant-worker-1.service elephant-worker-2.service",
                    "process": FIX_NPROC[2],
                }
            ),
        ]

        assert self.process_master.render.call_args_list == expected

    def test_render_multiple_processes(self):
        out = list(self.export.render(FIX_NPROC, {"app": "elephant"}))

        assert (
            self.File("elephant-web-1.service", self.process.render.return_value) in out
        )
        assert (
            self.File("elephant-worker-1.service", self.process.render.return_value)
            in out
        )
        assert (
            self.File("elephant-worker-2.service", self.process.render.return_value)
            in out
        )

        print(self.process.render.call_args_list)
        expected = call(
            {
                "app": "elephant",
                "master_wants": "elephant-web.target elephant-worker.target",
                "process_groups": [
                    ("elephant-web", [("elephant-web-1", FIX_NPROC[0])]),
                    (
                        "elephant-worker",
                        [
                            ("elephant-worker-1", FIX_NPROC[1]),
                            ("elephant-worker-2", FIX_NPROC[2]),
                        ],
                    ),
                ],
                "process_master_name": "elephant-worker",
                "process_master_wants": "elephant-worker-1.service elephant-worker-2.service",
                "process": FIX_NPROC[2],
            }
        )
        expected = [expected] * 3
        assert self.process.render.call_args_list == expected
