"""M1 metadata only: no material solve, optical propagation, or dispatch."""
from dataclasses import FrozenInstanceError, replace
import pytest
from lcprop.pr.unified.specs import (
    A7_CURRENT, DOUBLE_PRECISION, FIXED_FIELD, MIXED_PRECISION, OPEN_TRANSVERSE,
    PRESCRIBED_CURRENT, UNBIASED, PRElectricalClosureSpec, PRMaterialPrecisionSpec,
    PRUnifiedSpatialSpec,
)


@pytest.mark.parametrize('axes,shape,lengths,batch_axes,batch,field,harmonic,reductions', [
    (('x',),(4,),(2.,),(),(),(4,),(1,),(0,)),
    (('x',),(4,),(2.,),('y',),(3,),(4,3),(3,1),(0,)),
    (('x','y'),(4,3),(2.,5.),(),(),(4,3),(2,),(0,1)),
])
def test_spatial_layout(axes,shape,lengths,batch_axes,batch,field,harmonic,reductions):
    s=PRUnifiedSpatialSpec(shape,lengths,axes,batch,batch_axes)
    assert s.field_shape==field and s.harmonic_shape==harmonic
    assert s.reduction_axes==reductions


@pytest.mark.parametrize('changes', [
    {'active_axes':('y',)}, {'active_axes':('x','z')}, {'active_shape':(1,)},
    {'active_shape':(True,)}, {'active_shape':(4.,)}, {'active_shape':(4,3)},
    {'normalized_lengths':(0.,)}, {'normalized_lengths':(float('nan'),)},
    {'normalized_lengths':(float('inf'),)}, {'normalized_lengths':(True,)},
    {'normalized_lengths':(2.,3.)}, {'batch_shape':(3,)}, {'batch_axes':('z',),'batch_shape':(3,)},
    {'batch_axes':('y',),'batch_shape':(0,)}, {'batch_axes':('y',),'batch_shape':(2,3)},
    {'spatial_id':'unknown_v2'}, {'units':'micrometers'}, {'boundary':'open'}, {'transport_profile':'anisotropic'},
])
def test_invalid_spatial(changes):
    with pytest.raises(ValueError): PRUnifiedSpatialSpec(**dict({'active_shape':(4,), 'normalized_lengths':(2.,)},**changes))


def test_full_xy_cannot_have_y_batches():
    with pytest.raises(ValueError): PRUnifiedSpatialSpec((4,3),(2.,3.),('x','y'),(2,),('y',))


def test_metadata_detached_and_frozen():
    shape=[4];lengths=[2.];target=[.2]
    s=PRUnifiedSpatialSpec(shape,lengths);c=PRElectricalClosureSpec(FIXED_FIELD,1,target)
    shape[0]=8;lengths[0]=9;target[0]=10
    assert s.active_shape==(4,) and s.normalized_lengths==(2.,) and c.target==(.2,)
    with pytest.raises(FrozenInstanceError):s.boundary='open'


@pytest.mark.parametrize('dimension',[1,2])
@pytest.mark.parametrize('identity',[UNBIASED,FIXED_FIELD,PRESCRIBED_CURRENT])
def test_named_closures(identity,dimension):
    c=PRElectricalClosureSpec(identity,dimension,(0.,)*dimension)
    s=PRUnifiedSpatialSpec((4,)*dimension,(2.,)*dimension,('x',) if dimension==1 else ('x','y'))
    c.validate_spatial(s)


def test_column_closure_is_per_active_domain():
    s=PRUnifiedSpatialSpec((4,),(2.,),batch_axes=('y',),batch_shape=(3,))
    PRElectricalClosureSpec(FIXED_FIELD,1,(.2,)).validate_spatial(s)
    with pytest.raises(ValueError):PRElectricalClosureSpec(FIXED_FIELD,2,(.2,0)).validate_spatial(s)


def test_open_current_is_not_zero_transverse_field():
    c=PRElectricalClosureSpec(OPEN_TRANSVERSE,2,(.3,0.))
    assert c.identity==OPEN_TRANSVERSE and c.target==(.3,0.)
    assert c != PRElectricalClosureSpec(FIXED_FIELD,2,(.3,0.))


@pytest.mark.parametrize('identity,dimension,target,extra', [
    ('unknown',1,(0,),{}), ('illustrative_normalized_linear_load_v1',1,(0,),{}),
    (UNBIASED,1,(.1,),{}), (FIXED_FIELD,True,(0,),{}), (FIXED_FIELD,3,(0,0,0),{}),
    (FIXED_FIELD,2,(0,),{}), (FIXED_FIELD,1,(float('nan'),),{}),
    (FIXED_FIELD,1,(True,),{}), (FIXED_FIELD,1,(0,),{'reservoir_field':0.}),
    (OPEN_TRANSVERSE,1,(0,),{}), (OPEN_TRANSVERSE,2,(.3,.1),{}),
    (A7_CURRENT,2,(.1,0),{'reservoir_field':.5,'background_intensity':.2}),
    (A7_CURRENT,1,(.5,),{'reservoir_field':.5,'background_intensity':.2}),
    (A7_CURRENT,1,(0,),{}), (A7_CURRENT,1,(0,),{'reservoir_field':0.,'background_intensity':-.1}),
])
def test_invalid_closures(identity,dimension,target,extra):
    with pytest.raises(ValueError):PRElectricalClosureSpec(identity,dimension,target,**extra)


def test_a7_keeps_reservoir_background_and_current():
    c=PRElectricalClosureSpec.a7(.4,.2)
    assert c.reservoir_field==.4 and c.background_intensity==.2
    assert c.target==(.4*.2,) and c.target!=(c.reservoir_field,)
    assert PRElectricalClosureSpec.a7(-.4,.2).target==(-.4*.2,)
    assert PRElectricalClosureSpec.a7(.4,0.).target==(0.,)
    with pytest.raises(ValueError):PRElectricalClosureSpec.a7(1e308,1e308)
    with pytest.raises(TypeError):PRElectricalClosureSpec(FIXED_FIELD,1,(0,),U=[[1]])


@pytest.mark.parametrize('identity,dtype',[(DOUBLE_PRECISION,'float64'),(MIXED_PRECISION,'float32')])
def test_precision(identity,dtype):
    p=PRMaterialPrecisionSpec(identity=identity,state_dtype=dtype,output_dtype=dtype)
    assert p.linear_dtype==p.bernoulli_dtype=='float64'
    for change in [{'identity':'unknown_v2'},{'linear_dtype':'float32'},{'bernoulli_dtype':'float32'},
                   {'state_dtype':'complex128'},{'output_dtype':'complex64'}]:
        with pytest.raises(ValueError):replace(p,**change)
    with pytest.raises(ValueError):PRMaterialPrecisionSpec(MIXED_PRECISION)
