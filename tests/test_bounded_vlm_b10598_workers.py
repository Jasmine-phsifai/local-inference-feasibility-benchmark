import json
from pathlib import Path

import pytest

import scripts.run_bounded_vlm_b10598_quality as runner
from scripts.run_bounded_vlm_b10598_quality import build_request
import workers.hunyuanocr_1_5_b10598_quality_worker as hunyuan_worker
import workers.ovisocr2_b10598_quality_worker as ovis_worker
from workers.hunyuanocr_1_5_b10598_quality_worker import (
    build_b10598_server_command,
)
from workers.ovisocr2_b10598_quality_worker import (
    IDENTITY_TEMPLATE,
    RESPONSE_MARKER,
    build_command,
    build_rendered_prompt,
    extract_completion_tokens,
    extract_prediction,
)


def test_ovis_prompt_and_command_keep_exact_media_marker_and_cpu_only_flags(tmp_path):
    prompt = build_rendered_prompt()
    command = build_command(
        executable=tmp_path / "llama-mtmd-cli.exe",
        model=tmp_path / "model.gguf",
        projector=tmp_path / "mmproj.gguf",
        image=tmp_path / "control.png",
        prompt=prompt,
        log_path=tmp_path / "llama.log",
        threads=24,
        max_new_tokens=4096,
    )

    assert prompt.count("<__media__>") == 1
    assert "<__media__>\nExtract all readable content" in prompt
    assert command[command.index("--chat-template") + 1] == IDENTITY_TEMPLATE
    assert command[command.index("--threads") + 1] == "24"
    assert command[command.index("--predict") + 1] == "4096"
    assert command[command.index("--device") + 1] == "none"
    assert command[command.index("-ngl") + 1] == "0"
    assert "--no-mmproj-offload" in command


def test_ovis_output_parser_requires_one_marker_and_exact_perf_runs():
    stdout = f"ignored prelude\n{RESPONSE_MARKER}\npublic generated output\n"
    log = "llama_perf_context_print: eval time = 12.3 ms / 148 runs"

    assert extract_prediction(stdout) == "public generated output"
    assert extract_prediction(stdout + RESPONSE_MARKER) == ""
    assert extract_completion_tokens(log) == 148
    assert extract_completion_tokens("no perf line") is None


def test_hunyuan_wrapper_changes_only_server_binary_and_keeps_cpu_gate(tmp_path):
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"runtime")
    command = build_b10598_server_command(
        b10598_server=server,
        model_path=tmp_path / "model.gguf",
        projector_path=tmp_path / "projector.gguf",
        port=43123,
        threads=24,
    )

    assert command[0] == str(server)
    assert command[command.index("--device") + 1] == "none"
    assert command[command.index("--n-gpu-layers") + 1] == "0"
    assert command[command.index("--threads") + 1] == "24"
    assert command[command.index("--chat-template") + 1] == "hunyuan-vl"


def test_monitored_runner_request_satisfies_hunyuan_worker_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "repo"
    fixture_root = project_root / "fixtures"
    fixture_root.mkdir(parents=True)
    for name in ("warmup", "code_formula", "dense_table", "negative_diagram"):
        (fixture_root / f"{name}.png").write_bytes(name.encode("ascii"))
    manifest = {
        "schema_version": 1,
        "task": "ocr",
        "workload_class": "generated_quality_control",
        "warmup": {"id": "warmup", "path": "warmup.png", "expected_text": True},
        "items": [
            {"id": name, "path": f"{name}.png", "expected_text": name != "negative_diagram"}
            for name in ("code_formula", "dense_table", "negative_diagram")
        ],
    }
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assets = {
        "fixtures": {
            "manifest": {"path": manifest_path},
            "images": {
                name: {"path": (fixture_root / f"{name}.png").resolve()}
                for name in ("warmup", "code_formula", "dense_table", "negative_diagram")
            },
        },
        "candidate": {
            "sample_ids": ["code_formula", "dense_table", "negative_diagram"]
        },
    }
    run_root = project_root / "results" / "artifacts" / "run"
    request = build_request(
        candidate_id="hunyuanocr_1_5_gguf_cpu",
        assets=assets,
        response_path=run_root / "response.json",
        records_path=run_root / "private-records.jsonl",
    )

    assert request["config"] == {
        "processes": 1,
        "threads_per_process": 24,
        "max_new_tokens": 4096,
        "mode": "doc_parse",
    }
    assert [item["id"] for item in request["workload"]["items"]] == [
        "code_formula",
        "dense_table",
        "negative_diagram",
    ]
    assert request["capture_predictions"] is True
    monkeypatch.setattr(hunyuan_worker, "PROJECT_ROOT", project_root)
    hunyuan_worker._validate_request(request, assets=assets)
    stale_request = {**request, "protocol": "bounded-vlm-b10598-run-v1"}
    with pytest.raises(ValueError, match="request identity changed"):
        hunyuan_worker._validate_request(stale_request, assets=assets)


def test_monitored_runner_request_satisfies_ovis_worker_contract(tmp_path):
    project_root = tmp_path / "repo"
    fixture_root = project_root / "fixtures"
    fixture_root.mkdir(parents=True)
    image_path = fixture_root / "page_008_table_columns.png"
    image_path.write_bytes(b"public generated fixture")
    manifest = {
        "schema_version": 1,
        "task": "ocr",
        "workload_class": "generated_quality_control",
        "warmup_item_id": "page_008_table_columns",
        "items": [
            {
                "id": "page_008_table_columns",
                "path": image_path.name,
                "expected_text": True,
            }
        ],
    }
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assets = {
        "fixtures": {"manifest": {"path": manifest_path}},
        "candidate": {"sample_ids": ["page_008_table_columns"]},
    }
    run_root = project_root / "results" / "artifacts" / "run"

    request = build_request(
        candidate_id="ovisocr2_q8_cpu",
        assets=assets,
        response_path=run_root / "response.json",
        records_path=run_root / "private-records.jsonl",
    )

    ovis_worker._validate_request(request, project_root=project_root)
    stale_request = {**request, "protocol": "bounded-vlm-b10598-run-v1"}
    with pytest.raises(ValueError, match="request identity changed"):
        ovis_worker._validate_request(stale_request, project_root=project_root)


def test_actual_worker_environment_disables_user_site_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path / "repo")
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    monkeypatch.setenv("PYTHONPATH", "inherited-pythonpath")

    environment = runner._worker_environment()

    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["CI"] == "true"
    assert environment["PYTHONPATH"].split(runner.os.pathsep) == [
        str(runner.PROJECT_ROOT / "src"),
        str(runner.PROJECT_ROOT),
        "inherited-pythonpath",
    ]


def test_bounded_runner_preflight_failure_does_not_claim_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "repo"
    allowed_root = project_root / "results" / "artifacts"
    allowed_root.mkdir(parents=True)
    output_dir = allowed_root / "attempt"
    monkeypatch.setattr(runner, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        runner,
        "load_and_verify_candidate_assets",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("preflight failed")),
    )

    with pytest.raises(RuntimeError, match="preflight failed"):
        runner.run_bounded_quality_gate(
            candidate_id="ovisocr2_q8_cpu",
            output_dir=output_dir,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("host_error", "stop_reason", "process_error", "expected_failure"),
    (
        ("counter_process_exit", None, None, "monitor_failure"),
        (None, None, "sample_write_failed", "monitor_failure"),
        (None, "available_memory_below_4_gib", None, "safety_stop"),
    ),
)
def test_wait_for_worker_honors_monitor_state_before_clean_exit(
    host_error: str | None,
    stop_reason: str | None,
    process_error: str | None,
    expected_failure: str,
) -> None:
    events: list[str] = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            events.append("poll")
            return 0

    class FakeHostMonitor:
        pass

    class FakeProcessMonitor:
        pass

    class FakeJob:
        @staticmethod
        def close():
            events.append("close")

    host_monitor = FakeHostMonitor()
    host_monitor.start_error = host_error
    host_monitor.stop_reason = stop_reason
    process_monitor = FakeProcessMonitor()
    process_monitor.monitor_error = process_error
    assert runner._wait_for_worker(
        FakeProcess(),
        host_monitor,
        process_monitor,
        FakeJob(),
        timeout_seconds=60,
    ) == (0, expected_failure)
    assert events[0] == "close"


def test_wait_for_worker_surfaces_job_close_failure() -> None:
    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    class FakeHostMonitor:
        start_error = "counter_process_exit"
        stop_reason = None

    class FakeProcessMonitor:
        monitor_error = None

    class FakeJob:
        @staticmethod
        def close():
            raise RuntimeError("injected Job close failure")

    assert runner._wait_for_worker(
        FakeProcess(),
        FakeHostMonitor(),
        FakeProcessMonitor(),
        FakeJob(),
        timeout_seconds=60,
    ) == (-2, "termination_failure")


def test_wait_for_worker_closes_job_on_timeout() -> None:
    closed: list[bool] = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    class FakeHostMonitor:
        start_error = None
        stop_reason = None

    class FakeProcessMonitor:
        monitor_error = None

    class FakeJob:
        @staticmethod
        def close():
            closed.append(True)

    assert runner._wait_for_worker(
        FakeProcess(),
        FakeHostMonitor(),
        FakeProcessMonitor(),
        FakeJob(),
        timeout_seconds=0,
    ) == (-1, "timeout")
    assert closed == [True]


def _patch_bounded_runner_preflight(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[str]]:
    project_root = tmp_path / "repo"
    allowed_root = project_root / "results" / "artifacts"
    allowed_root.mkdir(parents=True)
    events: list[str] = []
    monkeypatch.setattr(runner, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        runner,
        "load_and_verify_candidate_assets",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(runner, "_verify_controller_environment", lambda: None)
    monkeypatch.setattr(
        runner,
        "_controller_environment_fingerprint",
        lambda: "controller-fingerprint",
    )
    monkeypatch.setattr(runner, "build_request", lambda **_kwargs: {"request": True})
    monkeypatch.setattr(runner, "_producer_hashes", lambda _candidate: {})
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})
    monkeypatch.setattr(
        runner,
        "_validate_completed_run",
        lambda **_kwargs: "succeeded",
    )
    monkeypatch.setattr(
        runner,
        "build_run_provenance",
        lambda **_kwargs: {"provenance": True},
    )
    return allowed_root / "attempt", events


def test_bounded_runner_assigns_suspended_worker_before_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, events = _patch_bounded_runner_preflight(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    creation_flags: list[int] = []
    no_window = 0x08000000

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return 0

    class FakeJob:
        def __init__(self):
            events.append("job-created")

        @staticmethod
        def assign(_process):
            events.append("assigned")

        @staticmethod
        def resume(_process):
            events.append("resumed")

        @staticmethod
        def close():
            events.append("job-closed")

    class FakeHostMonitor:
        start_error = None
        stop_reason = None
        samples = [{}]

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            events.append("host-started")

        @staticmethod
        def stop():
            events.append("host-stopped")
            return {"status": "observed", "sample_count": 2}

    class FakeProcessMonitor:
        monitor_error = None

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            events.append("process-monitor-started")

        @staticmethod
        def stop():
            events.append("process-monitor-stopped")
            return {"sample_count": 2}

    def fake_popen(*_args, **kwargs):
        events.append("popen")
        creation_flags.append(kwargs["creationflags"])
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "CREATE_NO_WINDOW", no_window, raising=False)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "WindowsKillOnCloseJob", FakeJob)
    monkeypatch.setattr(runner, "WindowsHostMonitor", FakeHostMonitor)
    monkeypatch.setattr(runner, "ProcessTreeMonitor", FakeProcessMonitor)

    result = runner.run_bounded_quality_gate(
        candidate_id="ovisocr2_q8_cpu",
        output_dir=output_dir,
    )

    assert result["status"] == "succeeded"
    assert creation_flags == [runner.CREATE_SUSPENDED | no_window]
    assert events.index("job-created") < events.index("popen")
    assert events.index("popen") < events.index("assigned")
    assert events.index("assigned") < events.index("resumed")
    assert events.index("resumed") < events.index("process-monitor-started")
    assert events.index("job-closed") < events.index("process-monitor-stopped")
    assert events.index("process-monitor-stopped") < events.index("host-stopped")


def test_controller_error_is_preserved_when_job_close_fails_and_monitors_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, events = _patch_bounded_runner_preflight(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    class FakeJob:
        @staticmethod
        def assign(_process):
            pass

        @staticmethod
        def resume(_process):
            pass

        @staticmethod
        def close():
            events.append("job-close-failed")
            raise RuntimeError("injected Job close failure")

    class FakeHostMonitor:
        start_error = None
        stop_reason = None
        samples = [{}]

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            pass

        @staticmethod
        def stop():
            events.append("host-stopped")
            return {"status": "observed", "sample_count": 1}

    class FakeProcessMonitor:
        monitor_error = None

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            raise RuntimeError("primary controller failure")

        @staticmethod
        def stop():
            events.append("process-monitor-stopped")
            return {"sample_count": 0}

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(runner, "WindowsKillOnCloseJob", FakeJob)
    monkeypatch.setattr(runner, "WindowsHostMonitor", FakeHostMonitor)
    monkeypatch.setattr(runner, "ProcessTreeMonitor", FakeProcessMonitor)
    monkeypatch.setattr(
        runner,
        "terminate_process_tree",
        lambda _pid: {"surviving": 0, "error_count": 0},
    )

    with pytest.raises(RuntimeError, match="primary controller failure") as raised:
        runner.run_bounded_quality_gate(
            candidate_id="ovisocr2_q8_cpu",
            output_dir=output_dir,
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "termination could not be verified" in str(raised.value.__cause__)
    assert events == [
        "job-close-failed",
        "process-monitor-stopped",
        "host-stopped",
    ]


@pytest.mark.parametrize(
    "termination_result",
    (
        None,
        {},
        {"surviving": False, "error_count": 0},
        {"surviving": 0, "error_count": False},
        {"surviving": 1, "error_count": 0},
        {"surviving": 0, "error_count": 1},
    ),
)
def test_unassigned_worker_fallback_requires_exact_termination_verification(
    monkeypatch: pytest.MonkeyPatch,
    termination_result: object,
) -> None:
    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        runner,
        "terminate_process_tree",
        lambda _pid: termination_result,
    )

    error = runner._close_worker_containment(
        process=FakeProcess(),
        process_job=None,
        job_assigned=False,
    )

    assert isinstance(error, RuntimeError)
    assert "termination could not be verified" in str(error)
