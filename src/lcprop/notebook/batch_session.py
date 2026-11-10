"""Kernel-loop ownership for one bounded notebook execution."""
import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from queue import Empty, Full, Queue
from time import monotonic
from uuid import uuid4
import warnings as runtime_warnings

from lcprop.core.execution import CancellationToken
from lcprop.notebook.batch_desktop import desktop_metadata, update_presentation
from lcprop.notebook.batch_files import MAX_UPLOAD_BYTES, PACKAGE_FILES
from lcprop.notebook.batch_requests import ROUTES, preset, validate
from lcprop.persistence.experiments import (
    ExperimentCodecRegistry, read_experiment_file, write_experiment_file,
)
from lcprop.runners.local import LocalRunner
from lcprop.transport.codecs import TransportCodecRegistry
from lcprop.transport.io import (
    read_request_package, read_result_package, write_request_package, write_result_package,
)


@dataclass(frozen=True)
class Record:
    run_id: str
    route: str
    request: object
    request_json: str
    digest: str
    result: object
    products: object
    provenance: dict | None = None
    package_files: tuple | None = None


def identity(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return text, hashlib.sha256(text.encode()).hexdigest()


def environment_identity():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("lcprop", "numpy", "scipy"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "source checkout"
    return versions


class BatchSession:
    def __init__(self, *, runner=None, validator=validate, notify=None, fixed_route=None):
        self.runner = LocalRunner() if runner is None else runner
        self.validator = validator
        self.notify = notify or (lambda: None)
        self.fixed_route = fixed_route
        self.route = fixed_route or "lc"
        self.draft = preset(self.route)
        self.presentation = {}
        self.execution_intent = None
        self.completed = self.partial = None
        self.status = "idle"
        self.message = ""
        self.progress = None
        self.active_id = self.last_run_id = None
        self.elapsed = 0.0
        self.task = None
        self.disposed = False
        self.generation = 0
        self.experiments = ExperimentCodecRegistry()
        self.transport = TransportCodecRegistry()
        for _, experiment, transport in ROUTES.values():
            self.experiments.register(experiment)
            self.transport.register(transport)

    def _notify(self):
        try:
            self.notify()
        except Exception as error:
            runtime_warnings.warn(f"Notebook view update failed: {error}", RuntimeWarning)

    @property
    def busy(self):
        return self.active_id is not None

    def _editable(self):
        if self.busy or self.disposed:
            raise RuntimeError("Draft controls are unavailable while running or disposed")

    def set_draft(self, route, request, generation):
        if generation != self.generation or self.busy or self.disposed:
            return False
        if self.fixed_route is not None and route != self.fixed_route:
            raise ValueError(f"This notebook only accepts {self.fixed_route.upper()} drafts")
        presentation = self.presentation if route == self.route else {}
        intent = self.execution_intent if route == self.route else None
        if presentation:
            # Desktop derives interference from explicit groups and retains
            # the canonical legacy flag. Preserve that flag when regrouping.
            beams = replace(request.beams, coherence=self.draft.beams.coherence)
            request = replace(request, beams=beams)
            if beams != self.draft.beams:
                presentation = update_presentation(presentation, beams)
        self.route, self.draft = route, deepcopy(request)
        self.presentation, self.execution_intent = presentation, intent
        return True

    def start(self, generation):
        if self.busy or self.disposed or generation != self.generation:
            return False
        loop = asyncio.get_running_loop()
        route, request = self.route, deepcopy(self.draft)
        payload, warnings = self.validator(route, request)
        encoded, digest = identity(payload)
        run_id = uuid4().hex
        self.token = CancellationToken()
        self.active_id = self.last_run_id = run_id
        self.elapsed = 0.0
        self.generation += 1
        self.status, self.message = "running", "\n".join(warnings)
        self.progress = self.partial = None
        self.task = loop.create_task(self._execute(
            route, request, run_id, encoded, digest, self.token))
        self._notify()
        return True

    async def _execute(self, route, request, run_id, encoded, digest, token):
        mailbox = Queue(maxsize=1)
        limit = 120 if route == "lc" else 60
        started = monotonic()

        def progress(event):
            if monotonic() - started >= limit:
                token.cancel()
            stage = getattr(event, "stage", None) or event.message
            if not stage:
                stage = "LC optical propagation" if route == "lc" else "PR material evolution"
            item = (run_id, event.completed_units, event.total_units, stage)
            try:
                mailbox.get_nowait()
            except Empty:
                pass
            try:
                mailbox.put_nowait(item)
            except Full:
                pass

        operation = ROUTES[route][0]
        worker = asyncio.create_task(asyncio.to_thread(
            self.runner.run_operation, operation, deepcopy(request),
            cancellation_token=token, progress_callback=progress))
        try:
            while not worker.done():
                await asyncio.sleep(0.2)
                if monotonic() - started >= limit and self.status == "running":
                    token.cancel()
                    self.status = "stopping"
                    self.message = "Elapsed budget reached. Waiting for a safe boundary."
                try:
                    item = mailbox.get_nowait()
                except Empty:
                    item = None
                if not self.disposed and self.active_id == run_id:
                    self.elapsed = monotonic() - started
                    if item is not None and item[0] == run_id:
                        self.progress = item[1:]
                    self._notify()
            output = await asyncio.shield(worker)
            if self.disposed or self.active_id != run_id:
                return
            self.elapsed = monotonic() - started
            try:
                item = mailbox.get_nowait()
            except Empty:
                pass
            else:
                if item[0] == run_id:
                    self.progress = item[1:]
            result = output.result
            status = getattr(result, "status", "completed")
            if status in {"completed", "stopped", "cancelled"}:
                owned_result, products = deepcopy((result, output.run_data))
                record = Record(run_id, route, request, encoded, digest, owned_result, products,
                                provenance=environment_identity())
                if status == "completed" and not token.is_cancelled():
                    self.completed = record
                    self.status = "completed"
                else:
                    self.partial = record
                    self.status = "stopped"
                self.message = (
                    "Stop was requested before result admission. Kept as a separate attempt."
                    if status == "completed" and token.is_cancelled() else output.message
                )
            else:
                self.status, self.message = "failed", f"Workflow returned {status}"
        except asyncio.CancelledError:
            token.cancel()
            await asyncio.shield(worker)
            raise
        except Exception as error:
            if not self.disposed and self.active_id == run_id:
                self.elapsed = monotonic() - started
                self.status, self.message = "failed", f"Execution: {type(error).__name__}: {error}"
        finally:
            if self.active_id == run_id:
                self.active_id = None
                self.generation += 1
                if not self.disposed:
                    self._notify()

    def stop(self):
        if self.busy and not self.disposed:
            self.token.cancel()
            self.status, self.message = "stopping", "Waiting for a safe boundary."
            self._notify()

    async def dispose(self):
        self.disposed = True
        self.notify = lambda: None
        if self.busy:
            self.token.cancel()
            await asyncio.shield(self.task)

    def save_draft(self, path):
        self._editable()
        self.validator(self.route, self.draft)
        operation = ROUTES[self.route][0]
        return write_experiment_file(path, self.draft, material_id=operation.material_id,
                                     workflow_id=operation.workflow_id, registry=self.experiments,
                                     presentation_payload=self.presentation,
                                     execution_intent=self.execution_intent)

    def load_draft(self, path):
        self._editable()
        loaded = read_experiment_file(path, registry=self.experiments)
        if self.fixed_route is not None and loaded.material_id != self.fixed_route:
            raise ValueError(f"This notebook only accepts {self.fixed_route.upper()} drafts")
        presentation, intent = desktop_metadata(loaded)
        self.validator(loaded.material_id, loaded.request)
        self.route, self.draft = loaded.material_id, loaded.request
        self.presentation, self.execution_intent = presentation, intent
        self.generation += 1

    def export_completed(self, path):
        record = self.completed
        if record is None:
            raise ValueError("No completed result")
        path = Path(path)
        path.mkdir()
        if record.package_files is not None:
            for name, contents in record.package_files:
                target = path / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(contents)
            return path
        operation, _, codec = ROUTES[record.route]
        write_request_package(path, registry=self.transport, material_id=operation.material_id,
                              workflow_id=operation.workflow_id, request=record.request,
                              run_id=record.run_id, execution_target="local",
                              provenance={"notebook_request_sha256": record.digest,
                                          "environment": record.provenance or {}})
        request = read_request_package(path, registry=self.transport)
        write_result_package(path, codec=codec, result=record.result,
                             request_envelope=request.envelope)
        return path

    def reopen(self, path):
        self._editable()
        request = read_request_package(path, registry=self.transport)
        output = read_result_package(path, registry=self.transport)
        if output.envelope.cancelled or output.envelope.scientific_status != "completed":
            raise ValueError("Only completed packages can replace the completed view")
        route = request.envelope.material_id
        if self.fixed_route is not None and route != self.fixed_route:
            raise ValueError(f"This notebook only accepts {self.fixed_route.upper()} results")
        operation, experiment, _ = ROUTES[route]
        # Transport packages can contain requests outside the editable subset.
        # Use their verified envelope as identity if the draft codec rejects it.
        try:
            payload = experiment.encode_request(request.request)
        except (TypeError, ValueError):
            payload = request.envelope.request_payload
        encoded, digest = identity(payload)
        declared = request.envelope.provenance.get("notebook_request_sha256")
        if declared is not None and declared != digest:
            raise ValueError("Saved notebook request hash does not match the decoded request")
        products = operation.to_run_data(output.result)
        # Keep original manifests, policy and provenance when re-exporting.
        path = Path(path)
        files = [(name, path / name) for name in sorted(PACKAGE_FILES)
                 if (path / name).is_file()]
        if sum(file.stat().st_size for _, file in files) > MAX_UPLOAD_BYTES:
            raise ValueError("Completed package exceeds 128 MiB")
        package = tuple((name, file.read_bytes()) for name, file in files)
        self.completed = Record(request.envelope.run_id, route, request.request,
                                encoded, digest, output.result, products,
                                provenance=dict(request.envelope.provenance),
                                package_files=package)
        self._notify()
