"""Build and execute the packaged native profile FFM solver."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from avazu_ctr.profile_ffm.config import NativeExecutor, ProfileFFMConfig
from avazu_ctr.profile_ffm.contracts import NativeSolverEvidence, sha256_file
from avazu_ctr.profile_ffm.hashing import hash_token


@dataclass(frozen=True)
class SolverBuild:
    binary: Path
    evidence: NativeSolverEvidence


@dataclass(frozen=True)
class SolverJob:
    name: str
    training: Path
    scoring: Path
    output: Path
    publisher_mask_basis_points: int = 0
    score_cold_publisher: bool = False


def solver_source_path() -> Path:
    return Path(__file__).with_name("native") / "solver.cpp"


def resolve_executor(requested: NativeExecutor) -> NativeExecutor:
    if requested is NativeExecutor.AUTO:
        return NativeExecutor.WSL if platform.system() == "Windows" else NativeExecutor.NATIVE
    return requested


def _wsl_path(path: Path) -> str:
    executable = shutil.which("wsl.exe")
    if executable is None:
        raise RuntimeError("WSL execution requires wsl.exe")
    result = subprocess.run(
        [executable, "--", "wslpath", "-a", path.resolve().as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"wslpath failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _execution_path(path: Path, executor: NativeExecutor) -> str:
    if executor is NativeExecutor.WSL:
        return _wsl_path(path)
    return str(path.resolve())


def build_solver(
    config: ProfileFFMConfig,
    destination: Path,
) -> SolverBuild:
    source = solver_source_path()
    if not source.is_file():
        raise RuntimeError(f"packaged solver source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    executor = resolve_executor(config.training.executor)
    compiler = "g++"
    if executor is NativeExecutor.NATIVE:
        resolved_compiler = shutil.which("g++") or shutil.which("c++")
        if resolved_compiler is None:
            raise RuntimeError("native profile FFM execution requires g++ or c++")
        compiler = resolved_compiler
        prefix: list[str] = []
    else:
        wsl = shutil.which("wsl.exe")
        if wsl is None:
            raise RuntimeError("profile FFM WSL execution requires wsl.exe")
        prefix = [wsl, "--"]
    command = [
        *prefix,
        compiler,
        "-Wall",
        "-Wextra",
        "-Wconversion",
        "-O3",
        "-fPIC",
        "-std=c++20",
        "-march=native",
        "-msse3",
        "-fopenmp",
        "-o",
        _execution_path(destination, executor),
        _execution_path(source, executor),
    ]
    version_result = subprocess.run(
        [*prefix, compiler, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if version_result.returncode:
        raise RuntimeError(
            "profile FFM compiler version probe failed:\n"
            f"{version_result.stdout}\n{version_result.stderr}"
        )
    compiler_version = version_result.stdout.splitlines()[0].strip()
    if not compiler_version:
        raise RuntimeError("profile FFM compiler returned an empty version")
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"profile FFM solver compilation failed:\n{result.stdout}\n{result.stderr}"
        )
    if not destination.is_file():
        raise RuntimeError("profile FFM solver compiler produced no binary")
    return SolverBuild(
        binary=destination,
        evidence=NativeSolverEvidence(
            executor=executor,
            compiler_version=compiler_version,
            source_sha256=sha256_file(source),
            binary_sha256=sha256_file(destination),
            build_command=tuple(command),
        ),
    )


def run_solver_job(
    build: SolverBuild,
    job: SolverJob,
    config: ProfileFFMConfig,
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[str, ...]:
    executor = build.evidence.executor
    cold_token = hash_token(
        config.cold_publisher.token,
        bins=config.features.hash_bins,
    )
    command = [
        _execution_path(build.binary, executor),
        "--train",
        _execution_path(job.training, executor),
        "--score",
        _execution_path(job.scoring, executor),
        "--output",
        _execution_path(job.output, executor),
        "--learning-rate",
        str(config.training.learning_rate),
        "--l2",
        str(config.training.l2),
        "--rank",
        str(config.training.rank),
        "--epochs",
        str(config.training.epochs),
        "--publisher-mask-bp",
        str(job.publisher_mask_basis_points),
        "--cold-publisher-token",
        str(cold_token),
    ]
    if job.score_cold_publisher:
        command.append("--score-cold-publisher")
    if executor is NativeExecutor.WSL:
        wsl = shutil.which("wsl.exe")
        if wsl is None:
            raise RuntimeError("profile FFM WSL execution requires wsl.exe")
        command = [wsl, "--", *command]
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    job.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            command,
            check=False,
            stdout=stdout,
            stderr=stderr,
        )
    if result.returncode:
        raise RuntimeError(
            f"profile FFM job {job.name!r} failed with exit code "
            f"{result.returncode}; see {stderr_path}"
        )
    if not job.output.is_file():
        raise RuntimeError(f"profile FFM job {job.name!r} produced no predictions")
    return tuple(command)
