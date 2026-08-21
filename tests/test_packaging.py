import re
from importlib import resources
from importlib.metadata import distribution
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

import frame2frame

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _requirements():
    return [Requirement(raw) for raw in distribution("frame2frame-virtual-screen").requires]


def _names_for_extra(extra=None):
    environment = {"extra": extra or ""}
    return {
        requirement.name.lower()
        for requirement in _requirements()
        if requirement.marker is None or requirement.marker.evaluate(environment)
    }


def test_default_install_contains_default_backend_and_one_opencv_provider():
    names = _names_for_extra()

    assert "mediapipe" in names
    assert "opencv-contrib-python" in names
    assert "opencv-python" not in names
    assert "opencv-python-headless" not in names
    assert "opencv-contrib-python-headless" not in names


def test_every_mediapipe_install_path_excludes_the_unvalidated_major_release():
    requirements = [item for item in _requirements() if item.name.lower() == "mediapipe"]

    assert len(requirements) == 3
    assert all(requirement.specifier == SpecifierSet(">=0.10.9,<1") for requirement in requirements)


def test_no_published_requirement_adds_a_second_cv2_provider():
    names = {requirement.name.lower() for requirement in _requirements()}

    assert names.isdisjoint(
        {
            "opencv-python",
            "opencv-python-headless",
            "opencv-contrib-python-headless",
        }
    )


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        pytest.param("mediapipe", {"mediapipe"}, id="mediapipe"),
        pytest.param(
            "hopenet",
            {"mediapipe", "torch", "torchvision"},
            id="hopenet",
        ),
    ],
)
def test_backend_extras_are_dependency_closed(extra, expected):
    assert expected <= _names_for_extra(extra)


def test_conflicting_sixdrepnet_extra_is_not_advertised():
    installed = distribution("frame2frame-virtual-screen")

    assert "sixdrepnet" not in (installed.metadata.get_all("Provides-Extra") or [])
    assert "sixdrepnet" not in _names_for_extra()


def test_hopenet_does_not_depend_on_unverified_downloader():
    assert "gdown" not in _names_for_extra("hopenet")


def test_distribution_metadata_matches_importable_package():
    installed = distribution("frame2frame-virtual-screen")

    assert installed.version == frame2frame.__version__
    assert SpecifierSet(installed.metadata["Requires-Python"]) == SpecifierSet(">=3.9,<3.14")
    assert Path(frame2frame.__file__).with_name("py.typed").is_file()
    assert (
        resources.files("frame2frame.pose")
        .joinpath("data", "canonical_face_vertices.txt")
        .is_file()
    )
    assert any(
        entry_point.group == "console_scripts"
        and entry_point.name == "frame2frame"
        and entry_point.value == "frame2frame.cli:main"
        for entry_point in installed.entry_points
    )


def test_readme_links_are_absolute_or_existing_repository_assets():
    readme = (_REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)]+)\)", readme)
    relative = [
        target
        for target in targets
        if not target.startswith(("https://", "http://", "mailto:", "#"))
    ]

    assert relative
    for target in relative:
        asset = (_REPOSITORY_ROOT / target).resolve()
        assert asset.is_relative_to(_REPOSITORY_ROOT.resolve())
        assert asset.is_file(), target
