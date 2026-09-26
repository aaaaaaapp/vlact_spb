import starVLA.model.framework as framework


def test_optional_framework_import_failure_does_not_stop_registration(monkeypatch):
    discovered_modules = [
        (None, "optional_framework", False),
        (None, "target_framework", False),
    ]
    imported_modules = []
    warnings = []

    monkeypatch.setattr(framework.pkgutil, "iter_modules", lambda _: discovered_modules)

    def fake_import_module(module_name):
        imported_modules.append(module_name)
        if module_name.endswith("optional_framework"):
            raise ModuleNotFoundError("missing optional dependency")

    monkeypatch.setattr(framework.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(framework.logger, "warning", lambda *args: warnings.append(args))

    framework._auto_import_framework_modules(["unused"])

    assert imported_modules == [
        "starVLA.model.framework.optional_framework",
        "starVLA.model.framework.target_framework",
    ]
    assert len(warnings) == 1
    assert warnings[0][1] == "optional_framework"
