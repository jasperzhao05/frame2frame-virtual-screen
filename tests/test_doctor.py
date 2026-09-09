from types import SimpleNamespace

from frame2frame import _doctor


def _fake_import(name):
    if name == "mediapipe.tasks.python":
        return SimpleNamespace(BaseOptions=object())
    if name == "mediapipe.tasks.python.vision":
        return SimpleNamespace(
            FaceLandmarker=object(),
            FaceLandmarkerOptions=object(),
            RunningMode=object(),
        )
    return SimpleNamespace(__version__=f"test-{name}")


def _fake_version(name):
    return "0.10.35" if name == "mediapipe" else f"test-{name}"


def _fake_dependencies(monkeypatch):
    monkeypatch.setattr(
        _doctor,
        "import_module",
        _fake_import,
    )
    monkeypatch.setattr(_doctor.metadata, "version", _fake_version)


def test_doctor_separates_optional_tools_from_blocking_dependencies(monkeypatch, tmp_path):
    _fake_dependencies(monkeypatch)
    monkeypatch.setenv("FRAME2FRAME_CACHE", str(tmp_path))
    monkeypatch.setattr(_doctor.shutil, "which", lambda name: None)

    checks = _doctor.collect_doctor_checks()
    by_name = {check.name: check for check in checks}

    assert by_name["numpy"].level == "ok"
    assert by_name["mediapipe-model"].level == "warning"
    assert "3,758,596" in by_name["mediapipe-model"].detail
    assert by_name["ffmpeg"].level == "warning"
    assert all(check.level != "error" for check in checks)


def test_doctor_reports_a_required_import_failure(monkeypatch, tmp_path):
    def import_module(name):
        if name == "cv2":
            raise ImportError("binary extension unavailable")
        return _fake_import(name)

    monkeypatch.setattr(_doctor, "import_module", import_module)
    monkeypatch.setattr(_doctor.metadata, "version", _fake_version)
    monkeypatch.setenv("FRAME2FRAME_CACHE", str(tmp_path))
    monkeypatch.setattr(_doctor.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    checks = _doctor.collect_doctor_checks()

    assert any(
        check.name == "opencv"
        and check.level == "error"
        and "binary extension unavailable" in check.detail
        for check in checks
    )


def test_doctor_reports_a_missing_required_distribution(monkeypatch, tmp_path):
    _fake_dependencies(monkeypatch)

    def distribution_version(name):
        if name == "mediapipe":
            raise _doctor.metadata.PackageNotFoundError(name)
        return "test"

    monkeypatch.setattr(_doctor.metadata, "version", distribution_version)
    monkeypatch.setenv("FRAME2FRAME_CACHE", str(tmp_path))

    checks = _doctor.collect_doctor_checks()

    assert next(check for check in checks if check.name == "mediapipe").level == "error"


def test_doctor_reports_a_missing_tasks_api_symbol(monkeypatch, tmp_path):
    def import_module(name):
        module = _fake_import(name)
        if name == "mediapipe.tasks.python.vision":
            del module.RunningMode
        return module

    monkeypatch.setattr(_doctor, "import_module", import_module)
    monkeypatch.setattr(_doctor.metadata, "version", _fake_version)
    monkeypatch.setenv("FRAME2FRAME_CACHE", str(tmp_path))

    checks = _doctor.collect_doctor_checks()
    mediapipe = next(check for check in checks if check.name == "mediapipe")

    assert mediapipe.level == "error"
    assert "RunningMode" in mediapipe.detail


def test_doctor_rejects_an_unvalidated_mediapipe_major(monkeypatch, tmp_path):
    _fake_dependencies(monkeypatch)
    monkeypatch.setattr(
        _doctor.metadata,
        "version",
        lambda name: "1.0.1" if name == "mediapipe" else f"test-{name}",
    )
    monkeypatch.setenv("FRAME2FRAME_CACHE", str(tmp_path))

    checks = _doctor.collect_doctor_checks()
    mediapipe = next(check for check in checks if check.name == "mediapipe")

    assert mediapipe.level == "error"
    assert ">=0.10.9,<1" in mediapipe.detail


def test_doctor_imports_mediapipe_tasks_but_not_matplotlib(monkeypatch, tmp_path):
    imported = []

    def import_module(name):
        imported.append(name)
        return _fake_import(name)

    monkeypatch.setattr(_doctor, "import_module", import_module)
    monkeypatch.setattr(_doctor.metadata, "version", _fake_version)
    monkeypatch.setenv("FRAME2FRAME_CACHE", str(tmp_path))

    checks = _doctor.collect_doctor_checks()

    assert {"mediapipe.tasks.python", "mediapipe.tasks.python.vision"} <= set(imported)
    assert "matplotlib" not in imported
    assert next(check for check in checks if check.name == "matplotlib").level == "ok"


def test_doctor_report_has_a_stable_summary():
    checks = (
        _doctor._DoctorCheck("required", "ok", "ready"),
        _doctor._DoctorCheck("optional", "warning", "missing"),
        _doctor._DoctorCheck("broken", "error", "failed"),
    )

    report = _doctor.format_doctor_report(checks)

    assert report.startswith(f"frame2frame doctor {_doctor.__version__}\n")
    assert "[warning] optional: missing" in report
    assert report.endswith("summary: 1 ok, 1 warning, 1 error")


def test_doctor_keeps_reporting_when_one_probe_raises(monkeypatch, tmp_path):
    _fake_dependencies(monkeypatch)
    monkeypatch.setenv("FRAME2FRAME_CACHE", str(tmp_path))
    monkeypatch.setattr(
        _doctor,
        "_model_check",
        lambda cache: (_ for _ in ()).throw(RuntimeError("unreadable metadata")),
    )

    checks = _doctor.collect_doctor_checks()
    by_name = {check.name: check for check in checks}

    assert by_name["mediapipe-model"].level == "error"
    assert by_name["mediapipe-model"].detail == "check failed: unreadable metadata"
    assert by_name["ffmpeg"].name == "ffmpeg"
