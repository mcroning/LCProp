"""Bounded retained optical movie samples; no propagation or material evaluation."""
import base64
import zlib
import numpy as np
from lcprop.products.data_model import ArtifactData

IDENTITY='pr_accepted_optical_preview_frames_v1'
MAX_BYTES=36*128*128*4


def pack_frames(stack):
    a=np.asarray(stack,dtype='<f4')
    if a.ndim!=3 or not 0<len(a)<=36 or max(a.shape[1:])>128 or not np.isfinite(a).all():
        raise ValueError('invalid bounded trajectory frames')
    return {'identity':IDENTITY,'shape':list(a.shape),'dtype':'<f4',
            'zlib_base64':base64.b64encode(zlib.compress(a.tobytes())).decode('ascii')}


def unpack_frames(value):
    if value['identity']!=IDENTITY or value['dtype']!='<f4':raise ValueError('unknown trajectory identity')
    shape=tuple(value['shape'])
    if len(shape)!=3 or any(type(n) is not int or n<1 for n in shape) or shape[0]>36 or max(shape[1:])>128:
        raise ValueError('invalid trajectory shape')
    stream=zlib.decompressobj();raw=stream.decompress(base64.b64decode(value['zlib_base64'],validate=True),MAX_BYTES+1)
    if not stream.eof or stream.unused_data or len(raw)!=int(np.prod(shape))*4:raise ValueError('invalid trajectory payload')
    a=np.frombuffer(raw,dtype='<f4').reshape(shape)
    if not np.isfinite(a).all():raise ValueError('nonfinite trajectory')
    return a


def movie_artifacts(result):
    metadata=getattr(result,'td_preview_movie_metadata',None)
    if not isinstance(metadata,dict):return {}
    out={};meta={k:v for k,v in metadata.items() if k not in ('interactive_frames','interactive_material_frames','additional_movies')}
    movie=getattr(result,'td_preview_movie',None)
    if movie is not None:
        out['td_preview_movie']=ArtifactData('td_preview_movie','Downsampled TD Preview (MP4)',
            np.asarray(movie,dtype=np.uint8),'video/mp4','pr_td_preview.mp4',meta)
    if 'interactive_frames' not in metadata:return out
    frames=unpack_frames(metadata['interactive_frames'])
    times=np.asarray(metadata['material_times_normalized'],dtype=float)
    if times.shape!=(len(frames),) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('trajectory times must be strictly increasing accepted times')
    profile=getattr(result,'resolved_profile',{})
    segment=profile.get('continuation_segment')
    offset=0. if segment is None else float(segment['source_cumulative_time'])
    grid=metadata['original_spatial_grid'];coords={}
    for axis in ('x','y'):
        n=int(grid['N'+axis]);count=frames.shape[1 if axis=='x' else 2]
        edges=np.linspace(0,n,count+1,dtype=int)
        nodes=(np.arange(n)-n/2+.5)*float(grid['d'+axis+'_um'])
        coords[axis]=[float(nodes[a:b].mean()) for a,b in zip(edges[:-1],edges[1:])]
    lineage=getattr(result,'diagnostics',{}).get('segment_lineage',())
    start=(float(lineage[-1]['source_cumulative_time']) if lineage else offset)
    meta.update(times=(times+offset).tolist(),coordinates=coords,segment_start=start,
        availability='Bounded accepted-time observations only; unretained quantities are unavailable.',
        playback='frame_uniform; labels are actual accepted characteristic times')
    out['td_trajectory']=ArtifactData('td_trajectory','Accepted optical trajectory',frames,
        'application/x-lcprop-trajectory','td_trajectory.npy',meta)
    if 'interactive_material_frames' in metadata:
        material=unpack_frames(metadata['interactive_material_frames'])
        if material.shape!=frames.shape:raise ValueError('material trajectory layout mismatch')
        sample=metadata['material_sample']
        material_meta=dict(meta,value_unit='1',quantity=sample['quantity'],
            label=sample['label']+' — plane '+str(sample['material_plane_index']))
        out['td_trajectory_material']=ArtifactData('td_trajectory_material',material_meta['label'],material,
            'application/x-lcprop-trajectory','td_material_trajectory.npy',material_meta)
    for key, record in metadata.get('additional_movies', {}).items():
        if key not in ('xz', 'yz', 'far_field'):
            raise ValueError('unknown additional TD movie')
        samples = unpack_frames(record['frames'])
        if len(samples) != len(frames):
            raise ValueError('trajectory sample count mismatch')
        axes = tuple(record['axes'])
        expected = {'xz': ('x', 'z'), 'yz': ('y', 'z'), 'far_field': ('s_x', 's_y')}[key]
        if axes != expected:
            raise ValueError('invalid trajectory axes')
        for axis, size in zip(axes, samples.shape[1:]):
            values = np.asarray(record['coordinates'][axis], dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
                raise ValueError('invalid trajectory coordinates')
        m = dict(meta, **{k: v for k, v in record.items() if k != 'frames'})
        artifact_key = 'td_trajectory_' + key
        out[artifact_key] = ArtifactData(artifact_key, record['display_name'], samples,
            'application/x-lcprop-trajectory', artifact_key + '.npy', m)
    return out
