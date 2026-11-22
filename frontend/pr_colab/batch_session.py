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
from lcprop.core.grid import grid_metadata
from pr_colab.batch_beams import DISABLED_BEAMS, edited_stack, objects, validate_objects
from lcprop.notebook.batch_desktop import update_presentation
from .metadata import experiment_metadata, remap_elements
from lcprop.notebook.batch_device import RunDevice, resolve_device
from pr_colab.batch_files import MAX_UPLOAD_BYTES, PACKAGE_FILES, package_sizes
from pr_colab.batch_requests import (
    ROUTES, TRANSVERSE_ROUTES, material_for, preset, validate,
)
from pr_colab.batch_resources import (
    available_memory, estimate,
)
from lcprop.persistence.execution_intent import ExecutionIntent
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
    device: RunDevice | None = None
    retained_bytes: int = 0
    export_bytes: int | None = None
    archive_bytes: int = 0


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
    def __init__(self, *, runner=None, validator=validate, notify=None, fixed_route=None,
                 gpu_probe=None, memory_available=None, device_policy=None, runtime_policy=None, time_limit=600):
        from .runtime_policy import RuntimePolicy
        self.runtime_policy = runtime_policy or RuntimePolicy()
        self.time_limit = time_limit
        self.result_policy = "full"
        self.continuation_checkpoint = None
        self._continuation_operation = None
        self.runner = LocalRunner() if runner is None else runner
        self.validator = validator
        self.memory_available = memory_available or available_memory
        self.route_state = {}
        self.route_device_intents = {}
        self.route_result_policies = {}
        self.selected_upload_bytes = 0
        self.gpu_probe = gpu_probe
        from .runtime_policy import CompatibleDevices
        self.device_policy = device_policy or CompatibleDevices()
        self.device_intent = "numpy"
        self.device_decision = None
        self.last_device = None
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
            if transport is not None:
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

    def _check_material(self, route):
        if self.fixed_route is not None and material_for(route) != self.fixed_route:
            raise ValueError(f"This notebook only accepts {self.fixed_route.upper()} workflows")

    def switch(self, route):
        self._editable()
        self._check_material(route)
        if route == self.route:
            return
        self.route_device_intents[self.route] = self.device_intent
        self.route_result_policies[self.route] = self.result_policy
        self.route_state[self.route] = (self.draft, self.presentation, self.execution_intent,
                                        self.completed, self.partial)
        self.route = route
        self.device_intent = self.route_device_intents.get(route,"numpy")
        self.result_policy = self.route_result_policies.get(route,"full")
        self.device_decision = None
        saved = self.route_state.pop(route, None)
        self.draft, self.presentation, self.execution_intent, self.completed, self.partial = (
            saved if saved is not None else (preset(route), {}, None, None, None)
        )
        self.last_run_id = None
        self.last_device = None
        self.status, self.message, self.progress, self.elapsed = "idle", "", None, 0.0
        self.generation += 1

    def copy_grid(self, source):
        self._editable()
        if source == self.route or material_for(source) != material_for(self.route):
            raise ValueError("Choose another workflow for the same material")
        saved = self.route_state.get(source)
        grid = (saved[0] if saved else preset(source)).grid
        draft = replace(self.draft, grid=grid)
        self.validator(self.route, draft)
        self.draft = draft
        self.generation += 1

    def release_result(self):
        self._editable()
        self.completed = self.partial = None
        self.generation += 1

    def resident_storage(self):
        records = [self.completed, self.partial]
        for state in self.route_state.values():
            records.extend(state[3:])
        present = [record for record in records if record is not None]
        return (sum(record.retained_bytes for record in present),
                sum(record.archive_bytes for record in present))

    def resident_bytes(self):
        return sum(self.resident_storage()) + self.selected_upload_bytes

    def export_unavailable(self):
        if self.route == "lc_td":
            return "LC time-dependent completed ZIP export is unavailable in this notebook."
        if self.completed is None:
            return "No completed result"
        if self.route == "lc" and self.completed.device is not None and self.completed.device.gpu is not None:
            return "LC Static GPU completed ZIP export is pending GPU-aware transport."
        size = self.export_size(self.completed)
        if size is None or size > MAX_UPLOAD_BYTES:
            return "Completed ZIP export estimate is unavailable or exceeds this process address space"
        return ""

    @staticmethod
    def export_size(record):
        if record.export_bytes is not None:
            return record.export_bytes
        return estimate(record.route, record.request).export

    def validate_intent(self,route,request):
        if self.device_intent != "auto":return self.validator(route,request)
        try:
            return self.validator(route,replace(request,backend=replace(request.backend,backend="numpy")))
        except ValueError as cpu_error:
            try:
                payload,warnings=self.validator(route,replace(request,backend=replace(request.backend,backend="cupy")))
            except ValueError:raise cpu_error
            return payload,(*warnings,"Auto: CPU/model combination is incompatible; GPU must pass admission, otherwise no dispatch.")

    def preflight(self, route, request):
        payload, warnings = self.validate_intent(route, request)
        charge = estimate(route, request)
        available=self.memory_available()
        if available is not None and request.grid.Nx * request.grid.Ny * (16 if request.backend.precision=="float64" else 8) > available:
            raise ValueError("A required optical plane cannot fit in available host memory")
        self.require_admission(charge)
        self.device_policy.working = charge.working
        self.device_policy.request = request
        return payload, warnings, charge

    def require_admission(self, charge):
        self.runtime_policy.check(charge.working, self.memory_available(), label="working memory")

    def set_draft(self, route, request, generation, *, reset_presentation=False):
        if generation != self.generation or self.busy or self.disposed:
            return False
        self._check_material(route)
        if reset_presentation:
            self.device_intent = request.backend.backend
            self.device_decision = None
        preserve = route == self.route and not reset_presentation
        if route != self.route:
            self.switch(route)
        presentation = self.presentation if preserve else {}
        intent = self.execution_intent if preserve else None
        if "beam_editor" in presentation:
            # Desktop derives interference from explicit groups and retains
            # the canonical legacy flag. Preserve that flag when regrouping.
            beams = replace(request.beams, coherence=self.draft.beams.coherence)
            request = replace(request, beams=beams)
            if beams != self.draft.beams:
                presentation = {**presentation, **update_presentation(presentation, beams)}
        if intent is not None and material_for(route) == "pr":
            intent = replace(intent, requested_backend=request.backend.backend)
        self.route, self.draft = route, deepcopy(request)
        self.presentation, self.execution_intent = presentation, intent
        return True

    def edit_beams(self, items, generation, coherence=None):
        if generation != self.generation or self.busy or self.disposed:
            return False
        beams, presentation = edited_stack(self.draft.beams, items, self.presentation, coherence)
        beams.validate()
        launch_elements, presentation = remap_elements(self.draft, items, self.presentation, presentation)
        request = replace(self.draft, beams=beams, launch_elements=launch_elements)
        self.draft = request
        self.presentation = presentation
        return True

    def device_choice(self, route, request):
        if material_for(route) == "lc":
            return self.device_intent, request.runtime.precision
        return self.device_intent, request.backend.precision

    @staticmethod
    def qualification_digest(route, request):
        if material_for(route) == "pr":
            request = replace(request, backend=replace(request.backend, backend="numpy"))
        return identity(ROUTES[route][1].encode_request(request))[1]

    def start(self, generation, *, review_auto=None, _request=None, _device_intent=None):
        if self.busy or self.disposed or generation != self.generation:
            return False
        loop = asyncio.get_running_loop()
        route, request = self.route, deepcopy(self.draft if _request is None else _request)
        payload, warnings, charge = self.preflight(route, request)
        _, requested_digest = identity(payload)
        if self.device_policy is None:
            device = resolve_device(route, request, self.gpu_probe)
        else:
            intent, precision = self.device_choice(route, request)
            if _device_intent is not None:intent=_device_intent
            if material_for(route) == "lc" and intent != "numpy":
                _, requested_digest = identity({"request": payload, "device": intent})
            qualification_digest = (
                self.qualification_digest(route, request) if intent != "numpy" else requested_digest
            )
            self.device_decision = None
            device = self.device_policy.resolve(route, qualification_digest, intent, precision,
                                                (request.grid.Nx, request.grid.Ny))
            self.device_decision = device
        if device.requested == "auto":
            if review_auto is None and self.device_policy is not None:
                raise ValueError("Review the resolved Auto device before dispatch")
            if review_auto is not None and not review_auto(requested_digest, device):
                return False
        if material_for(route) == "pr":
            request = replace(request, backend=replace(request.backend, backend=device.resolved))
            validate(route,request)  # Resolved backend/model compatibility before task creation.
            payload = ROUTES[route][1].encode_request(request)
        encoded, digest = identity(payload)
        run_id = uuid4().hex
        self.token = CancellationToken()
        self.active_id = self.last_run_id = run_id
        self.last_device = device
        self.elapsed = 0.0
        self.generation += 1
        self.status, self.message = "running", "\n".join(
            (*warnings, device.fallback_reason) if device.fallback_reason else warnings)
        self.progress = self.partial = None
        self.task = loop.create_task(self._execute(
            route, request, run_id, encoded, digest, self.token, device, charge, requested_digest))
        self._notify()
        return True

    def start_continuation(self,route,request,checkpoint):
        from .continuation import operation_for, describe
        self._editable()
        if route != self.route:raise ValueError('Choose the checkpoint workflow first')
        describe(route,request,checkpoint)
        self._continuation_operation=operation_for(route,checkpoint)
        try:
            return self.start(self.generation,_request=request,_device_intent="numpy")
        except BaseException:
            self._continuation_operation=None
            raise

    async def _execute(self, route, request, run_id, encoded, digest, token, device, charge, requested_digest):
        mailbox = Queue(maxsize=1)
        limit = self.time_limit
        started = monotonic()

        def progress(event):
            if monotonic() - started >= limit:
                token.cancel()
            stage = getattr(event, "stage", None) or event.message
            if not stage:
                stage = "LC optical propagation" if material_for(route) == "lc" else "PR material evolution"
            item = (run_id, event.completed_units, event.total_units, stage)
            try:
                mailbox.get_nowait()
            except Empty:
                pass
            try:
                mailbox.put_nowait(item)
            except Full:
                pass

        operation = self._continuation_operation or ROUTES[route][0]
        self._continuation_operation = None

        def run_on_device():
            self.runtime_policy.check(charge.working, self.memory_available(), label="worker working memory")
            validate(route, request)
            with device.context():
                if device.gpu is not None:
                    device.verify_gpu_memory((request.grid.Nx, request.grid.Ny))
                options = {"execution_backend": device.resolved} if (
                    material_for(route) == "lc" and device.resolved == "cupy"
                ) else {}
                return self.runner.run_operation(
                    operation, deepcopy(request), cancellation_token=token,
                    progress_callback=progress,
                    **({"_result_policy": self.result_policy} if operation.supports_result_policy else {}), **options,
                )

        worker = asyncio.create_task(asyncio.to_thread(run_on_device))
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
            complete = status == "completed" or (
                route == "pr_static" and status in {"converged", "not_converged"}
                and getattr(result, "completed_slices", 0) == grid_metadata(request.grid).Nz
            ) or (route == "pr_xy_static" and status == "converged" and result.converged)
            if complete or status in {"stopped", "cancelled", "not_converged"}:
                if material_for(route) == "pr":
                    backend = (result.backend_summary if route == "pr_static" or route in TRANSVERSE_ROUTES else
                               getattr(result, "diagnostics", {}).get("backend", {}))
                    if route in {"pr_unified","pr_published"}:
                        backend={"backend":result.backend,"real_dtype":device.precision,
                                 "complex_dtype":"complex128" if device.precision=="float64" else "complex64"}
                    expected = device.metadata()
                    if (backend.get("backend"), backend.get("real_dtype"), backend.get("complex_dtype")) != (
                        device.resolved, expected["real_dtype"], expected["complex_dtype"],
                    ):
                        raise RuntimeError("Result backend or precision differs from the admitted device")
                if material_for(route) == "lc" and device.gpu is not None:
                    backend = (result.provenance or {}).get("backend", {})
                    if (backend.get("backend"), backend.get("real_dtype"), backend.get("complex_dtype")) != (
                        "cupy", "float64", "complex128",
                    ):
                        raise RuntimeError("LC result backend or precision differs from the admitted device")
                owned_result, products = deepcopy((result, output.run_data))
                record = Record(run_id, route, request, encoded, digest, owned_result, products,
                                provenance={**environment_identity(), "requested_digest": requested_digest},
                                device=device,
                                retained_bytes=charge.retained, export_bytes=charge.export)
                if complete and not token.is_cancelled():
                    self.completed = record
                    self.status = "completed"
                else:
                    self.partial = record
                    self.status = "stopped" if token.is_cancelled() or status != "not_converged" else "failed"
                self.message = (
                    "Stop was requested before result admission. Kept as a separate attempt."
                    if complete and token.is_cancelled() else
                    "All slices completed, but the static solve did not converge."
                    if complete and status == "not_converged" else output.message
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
                cleanup = getattr(error, "cuda_cleanup_error", None)
                if cleanup is not None:
                    self.message += (
                        f" · CUDA context cleanup also failed ({type(cleanup).__name__}: {cleanup}). "
                        "Reconnect the runtime."
                    )
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
        self.validate_intent(self.route, self.draft)
        operation = ROUTES[self.route][0]
        snapshot=replace(self.draft,backend=replace(self.draft.backend,backend=self.device_intent))
        # Native codecs that require an explicit backend cannot persist Auto.
        # Fail with an explanation rather than silently saving a different intent.
        try:ROUTES[self.route][1].encode_request(snapshot)
        except ValueError as error:
            if self.device_intent == "auto":
                raise ValueError("This native experiment codec requires CPU or GPU explicitly for saving. Choose one before Download draft; no request changed.") from error
            raise
        intent=ExecutionIntent("local",None,None,self.device_intent,snapshot.backend.precision,self.result_policy)
        return write_experiment_file(path, snapshot, material_id=operation.material_id,
                                     workflow_id=operation.workflow_id, registry=self.experiments,
                                     presentation_payload=self.presentation,
                                     execution_intent=intent)

    def load_draft(self, path, *, upload_bytes=0):
        self._editable()
        path = Path(path)
        size = path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            raise ValueError("Draft JSON exceeds this process address space")
        # Allow for file staging, decoded text, parsed objects and codec copies.
        self.runtime_policy.check(32 * size + upload_bytes, self.memory_available(), label="draft decoding")
        loaded = read_experiment_file(path, registry=self.experiments)
        if self.fixed_route is not None and loaded.material_id != self.fixed_route:
            raise ValueError(f"This notebook only accepts {self.fixed_route.upper()} drafts")
        route = self.route_for(loaded.material_id, loaded.workflow_id)
        presentation, intent, policy = experiment_metadata(loaded, ROUTES[route][0])
        if DISABLED_BEAMS in presentation and "beam_editor" in presentation:
            raise ValueError("Conflicting beam editor metadata")
        validate_objects(objects(loaded.request.beams, presentation))
        route = self.route_for(loaded.material_id, loaded.workflow_id)
        self.validator(route, loaded.request)
        self.switch(route)
        self.draft = loaded.request
        self.device_intent = loaded.request.backend.backend
        self.device_decision = None
        self.presentation, self.execution_intent = presentation, intent
        self.result_policy = policy
        self.generation += 1

    @staticmethod
    def route_for(material, workflow):
        for key, (operation, _, _) in ROUTES.items():
            if operation.key == (material, workflow):
                return key
        raise ValueError(f"Unsupported notebook workflow: {material}/{workflow}")

    def export_completed(self, path):
        self._editable()
        reason = self.export_unavailable()
        if reason:
            raise ValueError(reason)
        record = self.completed
        self.runtime_policy.check(2 * self.export_size(record), self.memory_available(), label="export")
        path = Path(path)
        path.mkdir()
        if record.package_files is not None:
            for name, contents in record.package_files:
                target = path / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(contents)
            return path
        operation, _, codec = ROUTES[record.route]
        provenance = {"notebook_request_sha256": record.digest,
                      "environment": record.provenance or {}}
        if record.device is not None:
            provenance["device"] = record.device.metadata()
        write_request_package(path, registry=self.transport, material_id=operation.material_id,
                              workflow_id=operation.workflow_id, request=record.request,
                              run_id=record.run_id, execution_target="local",
                              provenance=provenance)
        request = read_request_package(path, registry=self.transport)
        write_result_package(path, codec=codec, result=record.result,
                             request_envelope=request.envelope)
        return path

    def reopen(self, path, *, upload_bytes=0):
        self._editable()
        path = Path(path)
        files = [(name, path / name) for name in sorted(PACKAGE_FILES)
                 if (path / name).is_file()]
        packed_bytes, expanded_bytes = package_sizes([file for _, file in files])
        # Verification and codec decoding can overlap independent copies.
        decode_bytes = 2 * expanded_bytes + 2 * packed_bytes
        resident = self.resident_bytes() + upload_bytes
        self.runtime_policy.check(decode_bytes, self.memory_available(), label="ZIP decoding")
        request = read_request_package(path, registry=self.transport)
        route = self.route_for(request.envelope.material_id, request.envelope.workflow_id)
        self._check_material(route)
        charge = estimate(route, request.request)
        self.runtime_policy.check(decode_bytes + charge.retained, self.memory_available(), label="ZIP reopen")
        output = read_result_package(path, registry=self.transport)
        completed_status = output.envelope.scientific_status == "completed" or (
            route == "pr_static" and output.envelope.scientific_status in {"converged", "not_converged"}
            and output.result.completed_slices == grid_metadata(request.request.grid).Nz
        ) or (route == "pr_xy_static" and output.envelope.scientific_status == "converged"
              and output.result.converged)
        if output.envelope.cancelled or not completed_status:
            raise ValueError("Only completed packages can replace the completed view")
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
        # Keep original manifests, policy and provenance for exact re-export.
        package = tuple((name, file.read_bytes()) for name, file in files)
        self.switch(route)
        self.completed = Record(request.envelope.run_id, route, request.request,
                                encoded, digest, output.result, products,
                                provenance=dict(request.envelope.provenance),
                                package_files=package, retained_bytes=charge.retained,
                                export_bytes=packed_bytes, archive_bytes=packed_bytes)
        self._notify()
