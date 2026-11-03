"""Exercise calculation and persistence without optional front ends or runners."""
from pathlib import Path
import subprocess
import sys
import textwrap

import lcprop


def test_headless_calculation_and_persistence_without_frontends(tmp_path):
    code = textwrap.dedent('''
        import importlib.abc
        import sys
        from pathlib import Path

        sys.path.insert(0, sys.argv[1])

        class NoFrontends(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (fullname.split('.')[0] in
                        {'PySide6', 'PyQt6', 'launchplane', 'matplotlib', 'IPython', 'ipywidgets', 'cupy'}
                        or fullname.startswith('lcprop.gui')
                        or fullname.startswith('lcprop.notebook')
                        or fullname in {'lcprop.runners.local', 'lcprop.runners.slurm',
                                        'lcprop.runners.source_deployment', 'lcprop.runners.cluster_profiles'}
                        or '.gui' in fullname):
                    raise AssertionError('Engine imported an optional adapter: ' + fullname)

        sys.meta_path.insert(0, NoFrontends())
        import numpy as np
        from lcprop.core.context import GridSpec
        from lcprop.core.beams import BeamChannel, BeamStack
        from lcprop.pr.specs import PRMaterialSpec
        from lcprop.pr.unified.specs import PRUnifiedSpatialSpec, PRElectricalClosureSpec, UNBIASED
        from lcprop.pr.unified.workflow import UnifiedStaticRequest, run_unified_static
        from lcprop.pr.unified.codec import encode_request, decode_request
        from lcprop.pr.static_workflow import PRStaticRunRequest, PR_STATIC_WORKFLOW
        from lcprop.persistence import save_experiment, load_experiment

        grid = GridSpec(Nx=8, Ny=4, x_aperture_um=16., y_aperture_um=8.,
                        z_length_um=2., dz_um=2.)
        request = UnifiedStaticRequest(
            grid, PRUnifiedSpatialSpec((8,), (16.,), batch_axes=('y',), batch_shape=(4,)),
            PRElectricalClosureSpec(UNBIASED, 1, (0.,)),
            np.ones((1, 8, 4), dtype=np.complex128),
            material=PRMaterialSpec(gain_length_product=.1,
                                   characteristic_wavenumber_per_um_override=1.))
        restored = decode_request(encode_request(request)).materialize()
        result = run_unified_static(restored)
        assert result.status == 'completed', result.failure
        assert result.completed_cells == 1
        assert np.isfinite(result.boundary_field).all()
        np.testing.assert_allclose(np.abs(result.boundary_field), 1., atol=1e-12, rtol=1e-12)

        legacy = PRStaticRunRequest(grid=grid, beams=BeamStack(channels=(BeamChannel(),)))
        path = Path(sys.argv[2]) / 'legacy.lcprop.json'
        save_experiment(legacy, path, material_id='pr', workflow_id=PR_STATIC_WORKFLOW)
        loaded = load_experiment(path, expected_material_id='pr')
        assert loaded.workflow_id == PR_STATIC_WORKFLOW
        assert loaded.request == legacy

        from lcprop.pr.transverse.static import PRTransverseStaticMaterialSolverOptions
        from lcprop.pr.transverse.marching_static import (
            PRTransverseMarchingStaticOptions, PRTransverseMarchingStaticRunRequest)
        from lcprop.pr.transverse.streaming import run_pr_transverse_streaming
        assert PRTransverseStaticMaterialSolverOptions().potential_space == 'derivative_resolved_v1'
        corrected = PRTransverseMarchingStaticRunRequest(
            grid=GridSpec(Nx=8, Ny=8, x_aperture_um=40., y_aperture_um=40.,
                          dz_um=5., z_length_um=15.),
            beams=BeamStack(channels=(BeamChannel(coherence_group='engine'),)),
            material=PRMaterialSpec(dark_intensity=.4, uniform_background_intensity=.1,
                                   gain_length_product=1e-3,
                                   characteristic_wavenumber_per_um_override=.1),
            solver=PRTransverseMarchingStaticOptions(material_solver=
                PRTransverseStaticMaterialSolverOptions(potential_space='constant_gauge_v1')),
            initial_A=(np.exp(.01 * (-1.) ** np.arange(8)[:, None])
                       * np.ones((1, 8)))[None].astype(complex))
        replay = run_pr_transverse_streaming(corrected, stop_after_accepted_cells=2)
        assert replay.converged
        assert all(replay.diagnostics['replay_checks'].values())
        assert replay.first_pass.resolved_profile['solver']['material_solver']['potential_space'] == 'constant_gauge_v1'
        print('Headless calculation, codec and legacy persistence passed')
    ''')
    result = subprocess.run(
        [sys.executable, '-I', '-B', '-c', code,
         str(Path(lcprop.__file__).resolve().parent.parent), str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
