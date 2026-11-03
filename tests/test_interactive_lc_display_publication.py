"""Real IOPub regression; requires an environment with ipykernel and ipympl."""
import ast
import importlib.util
import os
from pathlib import Path
import sys

import pytest


def test_persistent_canvas_edits_publish_no_additional_displays(tmp_path):
    for package in ('ipympl', 'ipykernel'):
        if importlib.util.find_spec(package) is None:
            pytest.skip(f'{package} is required for real-kernel publication coverage')
    jupyter = pytest.importorskip('jupyter_client')
    km = jupyter.KernelManager()
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    root = Path(__file__).resolve().parents[1]
    km.start_kernel(cwd=str(root), env={**os.environ, 'PYTHONPATH': str(root/'src'),
                                      'MPLCONFIGDIR': str(tmp_path/'mpl')})
    client = km.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=45)
        def run(code, expressions=None):
            msg_id = client.execute(code, user_expressions=expressions or {})
            published = []
            while True:
                msg = client.get_iopub_msg(timeout=45)
                kind, content = msg['header']['msg_type'], msg['content']
                # Include comm callback publications, not just this execute request.
                if kind == 'error':
                    pytest.fail(content['ename'] + ': ' + content['evalue'])
                if kind in ('display_data', 'update_display_data', 'execute_result', 'clear_output'):
                    published.append((kind, tuple(content.get('data', {}))))
                if (kind == 'status' and content['execution_state'] == 'idle'
                        and msg['parent_header'].get('msg_id') == msg_id):
                    break
            while True:
                reply = client.get_shell_msg(timeout=45)
                if reply['parent_header'].get('msg_id') == msg_id:
                    assert reply['content']['status'] == 'ok'
                    return published, reply['content'].get('user_expressions', {})

        initial, expression = run("""%matplotlib widget
from IPython.display import display
from lcprop.notebook import InteractiveLC
import lcprop.notebook.interactive_lc as ui
from ipympl.backend_nbagg import Canvas
from PIL import Image
import io
binary_frames = []
original_send = Canvas.send
def capture_send(self, content, buffers=None):
    if buffers:
        raw = memoryview(buffers[0]).tobytes()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            assert image.format == 'PNG'
            binary_frames.append((len(raw), image.size))
    return original_send(self, content, buffers=buffers)
Canvas.send = capture_send
ui.execute = lambda *a, **k: (_ for _ in ()).throw(AssertionError('No solver'))
demo = InteractiveLC()
canvas = demo._position_editor.canvas
assert not demo._position_editor._frontend_initialized
assert binary_frames == []
# Committed edits before the handshake must also defer transmission.
demo.second_enabled.value = True
demo.beam2_controls['w1_um'].value = 12
assert binary_frames == []
display(demo.widget)
""", {'widget': "demo.controls['x'].model_id", 'canvas': 'canvas.model_id'})
        assert len(initial) == 1 and 'application/vnd.jupyter.widget-view+json' in initial[0][1]
        comm_id = ast.literal_eval(expression['widget']['data']['text/plain'])
        canvas_id = ast.literal_eval(expression['canvas']['data']['text/plain'])
        def canvas_message(content):
            client.shell_channel.send(client.session.msg('comm_msg', content={
                'comm_id': canvas_id, 'data': {'method': 'custom', 'content': content}}))
        # Exercise the real ipympl message dispatcher, including a pre-init resize.
        canvas_message({'type': 'resize', 'width': 375, 'height': 375})
        outputs, _ = run('assert binary_frames == []')
        assert outputs == []
        for kind in ('refresh', 'send_image_mode', 'initialized'):
            canvas_message({'type': kind})
        outputs, _ = run("assert demo._position_editor._frontend_initialized\nassert binary_frames == []")
        assert outputs == []
        canvas_message({'type': 'draw'})
        outputs, _ = run("""assert binary_frames and binary_frames[0][1] == (375, 375)
last_frame_count = len(binary_frames)
initial_layout = tuple(tuple(ax.get_position().bounds) for ax in canvas.figure.axes)
""")
        assert outputs == []
        for browser_style in (False, True):
            for enabled in (-19., -18., -17., -16.):
                if browser_style:
                    client.shell_channel.send(client.session.msg('comm_msg', content={
                        'comm_id': comm_id, 'data': {'method': 'update',
                        'state': {'value': enabled}, 'buffer_paths': []}}))
                    code = f"assert demo.controls['x'].value == {enabled}"
                else:
                    code = f"demo.controls['x'].value = {enabled}"
                outputs, _ = run(code + f"\nassert demo.controls['x'].value == {enabled}"
                    + '\nassert demo._position_editor.canvas is canvas'
                    + '\nassert tuple(tuple(ax.get_position().bounds) for ax in canvas.figure.axes) == initial_layout'
                    + '\nassert len(demo.preview_host.children) == 1'
                    + '\nassert len(binary_frames) == last_frame_count')
                assert outputs == []
                canvas_message({'type': 'draw'})
                outputs, _ = run('assert len(binary_frames) > last_frame_count\nlast_frame_count = len(binary_frames)')
                assert outputs == [], (browser_style, enabled, outputs)
        for enabled in (False, True):
            outputs, _ = run(f"demo.second_enabled.value = {enabled}\n"
                "assert len(binary_frames) == last_frame_count\n"
                "assert demo._position_editor.canvas is canvas")
            assert outputs == []
            canvas_message({'type': 'draw'})
            outputs, _ = run('assert len(binary_frames) > last_frame_count\nlast_frame_count = len(binary_frames)')
            assert outputs == []
        outputs, _ = run("get_ipython().events.trigger('post_execute')\nassert demo.completed is None\ndemo.close()")
        assert outputs == []
    finally:
        client.stop_channels()
        km.shutdown_kernel(now=True)
