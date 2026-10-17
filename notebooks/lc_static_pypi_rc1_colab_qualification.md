# LCProp 0.4.0rc1 hosted Colab qualification

Open `lc_static_pypi_rc1_colab_qualification.ipynb` in Colab with a fresh CPU runtime. Run setup, then initialize the interface. The notebook downloads the exact hash-verified TestPyPI RC1 wheel; dependencies resolve from production PyPI. No checkout or manual file upload is needed.

Setup verifies version 0.4.0rc1, both published artifact hashes, downloaded wheel bytes, all 269 installed runtime files, import ownership and dependency consistency. It stops on discrepancies. Source identity: `2e195579dc3d76f6477bf6c885fd537043acdf9d`.

Interactive initialization enables Colab's custom widget manager and the Matplotlib widget backend. Repeating initialization reuses the controller. If interactive rendering fails, record the failure, export any needed records, restart the runtime, set `MODE = "static"`, and rerun setup and initialization. Static mode preserves numeric editing and the canonical launch preview; it does not qualify mouse interaction.

Follow the notebook checklist for first/repeated Beam 2 activation, Move/Tilt, numeric edits, stable rendering and controller reuse. Record browser observations and environment output. No cell automatically executes science. Do not press Run for graphics testing; completed-run isolation remains pending a separately authorized bounded calculation.

Stage 3C records preparation before publication; its 404 and notebook hash are historical. Stage 3E confirms the subsequent exact publication. Stage 3F changes only the notebook's publication-status prose and adds these instructions; setup code and artifact pins are unchanged. The production 0.3.0 notebook remains unchanged.

## Hosted presentation correction

After the first installation, choose **Runtime → Restart session**, then rerun setup/verification before opening the interface. Do not disconnect/delete the runtime. No automatic restart or module reload is performed; ordinary verified setup and initialization reruns do not require another restart. Export needed results before any restart.

The notebook displays the existing Configure contents, Run and Results in one retained container, bypassing Colab's missing accordion view. Container/HTML layouts expand vertically, with a bounded overall width; canvas and output internals and Beam 2 visibility remain controller-owned. Repeated initialization replaces only its own cell output and reuses both controller and presentation. Avoid displaying diagnostic copies in other cells.

User-reported graphical acceptance covers Beam 2 toggling, Move/Tilt and numeric edits in the directly displayed view. Fresh hosted acceptance of this normal initialization and scrollbar behavior is still required. Completed-run and export checks remain pending separately authorized science.
