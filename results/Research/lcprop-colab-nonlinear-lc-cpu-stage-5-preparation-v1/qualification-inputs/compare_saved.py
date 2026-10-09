"""Read saved outputs only; never execute LCProp science. See PROCEDURE.md criteria."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(inputs, output):
    ref = inputs / 'reference'
    for directory in (ref, output):
        manifest = json.loads((directory / 'sha256.json').read_text())
        for name, sha in manifest.items():
            assert Path(name).name == name
            assert digest(directory / name) == sha, str(directory / name)
    expected = json.loads((inputs / 'expected-arrays.json').read_text())
    rows = {}
    for mode in ('nonlinear', 'fixed'):
        assert (output / f'{mode}.lcprop.json').read_bytes() == (ref / f'{mode}.lcprop.json').read_bytes(), 'request mismatch'
        with np.load(ref / f'{mode}.npz', allow_pickle=False) as a, np.load(output / f'{mode}.npz', allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                x, y = a[key], b[key]
                spec = expected[f'{mode}/{key}']
                assert x.shape == y.shape == tuple(spec['shape'])
                assert str(x.dtype) == str(y.dtype) == spec['dtype']
                assert np.isfinite(x).all() and np.isfinite(y).all()
                hx = hashlib.sha256(x.tobytes(order='C')).hexdigest()
                hy = hashlib.sha256(y.tobytes(order='C')).hexdigest()
                assert hx == spec['sha256']
                delta = np.abs(y-x); scale = float(np.max(np.abs(x)))
                floor = max(1e-300, 1e-12*scale); mask = np.abs(x) >= floor
                norm = float(np.linalg.norm(x.ravel()))
                rel = float(np.linalg.norm((y-x).ravel())/norm) if norm else None
                # Provisional reporting screen, not a solver tolerance or blanket qualification.
                passed = float(delta.max()) <= 1e-12 + 1e-9*scale and (rel is None or rel <= 1e-9)
                if key in ('x_um', 'y_um', 'z_um'):
                    passed = hx == hy
                rows[f'{mode}/{key}'] = dict(reference_sha256=hx, hosted_sha256=hy,
                    bitwise=hx==hy, max_abs=float(delta.max()), rms_abs=float(np.sqrt(np.mean(delta**2))),
                    relative_l2=rel, max_relative=float(np.max(delta[mask]/np.abs(x[mask]))) if mask.any() else None,
                    relative_denominator_threshold=floor, differing_elements=int(np.count_nonzero(x!=y)),
                    provisional_screen_pass=passed)
    assert set(rows) == set(expected) and len(rows) == 18
    s = json.loads((output/'summary.json').read_text())
    reference = json.loads((ref/'summary.json').read_text())
    n = s['nonlinear']; f = s['fixed']
    assert n['status'] == f['status'] == 'completed'
    assert n['completed_slices'] == n['total_slices'] == f['completed_slices'] == f['total_slices'] == 100
    assert n['all_slices_converged'] is True
    assert n['residual_rms_max'] <= .005 and n['residual_max'] <= .02
    assert len(n['slice_summaries']) == 100
    for i, row in enumerate(n['slice_summaries']):
        assert row['z_index'] == i and row['converged'] is True
        assert row['final_residual_rms'] <= .005 and row['final_residual_max'] <= .02
    # Compare every scientific scalar/history entry; do not hide iteration changes.
    differences = []
    def walk(a, b, path):
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                differences.append((path, 'membership')); return
            for key in a:
                if key not in ('environment', 'elapsed_seconds_including_products'):
                    walk(a[key], b[key], path+'/'+key)
        elif isinstance(a, list):
            if not isinstance(b,list) or len(a) != len(b):
                differences.append((path,'length')); return
            for i, (x,y) in enumerate(zip(a,b)): walk(x,y,path+'/'+str(i))
        elif isinstance(a,float):
            if not isinstance(b,(int,float)) or not np.isfinite(b) or abs(a-b)>1e-12+1e-9*abs(a):
                differences.append((path, 'numeric'))
        elif a != b:
            differences.append((path,'exact'))
    walk(reference, s, 'summary')
    checks = {}
    for mode in ('nonlinear','fixed'):
        with np.load(output/f'{mode}.npz',allow_pickle=False) as z:
            dx=float(z['x_um'][1]-z['x_um'][0]); dy=float(z['y_um'][1]-z['y_um'][0])
            checks[mode] = {'normalized_initial_from_array':float(np.sum(np.abs(z['A_initial'])**2)*dx*dy),
                'normalized_final_from_array':float(np.sum(np.abs(z['A_final'])**2)*dx*dy)}
            for phase in ('initial','final'):
                assert np.isclose(checks[mode]['normalized_'+phase+'_from_array'],s[mode]['normalized_power_'+phase],rtol=1e-12,atol=1e-12)
            # Product's public RMS diagnostic, applied to saved planes only.
            from lcprop.core.grid import make_grid
            from lcprop.persistence import load_experiment
            from lcprop.products.diagnostics import rms_widths
            request=load_experiment(output/f'{mode}.lcprop.json').request
            grid=make_grid(request.grid,real_dtype=np.float64)
            w=np.asarray([rms_widths(plane,grid) for plane in z['intensity_stack']])
            assert np.allclose(w,z['widths_um'],rtol=1e-12,atol=1e-12)
            assert np.allclose(w[-1],s['last_slice_rms_width_um'][mode],rtol=1e-12,atol=1e-12)
            if mode=='nonlinear':
                checks['max_reorientation_rad']=float(np.max(np.abs(z['theta']-z['theta_bias'][None])))
                assert np.isclose(checks['max_reorientation_rad'],s['max_reorientation_rad'],rtol=1e-12,atol=1e-12)
            else:
                assert np.array_equal(z['theta'],z['theta_bias'])
    return dict(arrays=rows, diagnostic_differences=differences, consistency=checks,
        physical_convergence_gates_pass=True,
        provisional_numerical_screen_pass=all(r['provisional_screen_pass'] for r in rows.values()) and not differences,
        decision='Human review of provenance, plots and differences required; not automatic scientific qualification')

if __name__ == '__main__':
    result=compare(Path(sys.argv[1]),Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('18 arrays compared; provisional numerical screen:',result['provisional_numerical_screen_pass'])
