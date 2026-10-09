"""Reference export and convergence setup, with optional compiler integration."""
import importlib.util
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    path=ROOT/'numerical_reference/swe'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result

solver=module('generate')
study=module('convergence')

def test_staggered_reference_export(tmp_path,monkeypatch):
    def run(command,cwd,check):
        cells=5000
        u=np.arange(3*(cells+1),dtype=np.float32).reshape(3,cells+1)
        zeta=np.arange(3*cells,dtype=np.float32).reshape(3,cells)
        u.tofile(Path(cwd)/'u.bin'); zeta.tofile(Path(cwd)/'zeta.bin')
    monkeypatch.setattr(solver.subprocess,'run',run)
    output=solver.generate(tmp_path/'reference.nc',duration=120.,executable='solver')
    with xr.open_dataset(output) as ds:
        assert ds.u_ref.shape==(201,3)
        assert ds.zeta_ref.shape==(200,3)
        np.testing.assert_array_equal(ds.time,[0.,60.,120.])
        np.testing.assert_array_equal(ds.x_u,np.linspace(-1e6,1e6,201))
        np.testing.assert_array_equal(ds.x_zeta,np.arange(-995000.,1e6,10000.))
        assert ds.zeta_ref[0,0]==12
    assert not list(tmp_path.glob('solver-*'))
    with pytest.raises(FileExistsError): solver.generate(output)

@pytest.mark.parametrize('viscosity',[0.,50000.])
@pytest.mark.parametrize('kind',['dx','dt'])
def test_study_choices_are_stable(viscosity,kind):
    grid=study.cases(kind,viscosity,270000.,60.)
    assert grid
    for dx,dt in grid: solver.validate(dx,dt,viscosity,270000.,60.)
    if kind=='dx': assert len({dt for dx,dt in grid})==1

@pytest.mark.skipif(not shutil.which('gfortran'),reason='gfortran not installed')
def test_compiled_solver_closed_boundaries(tmp_path):
    output=solver.generate(tmp_path/'short.nc',duration=120.)
    with xr.open_dataset(output) as ds:
        assert np.isfinite(ds.u_ref).all()
        np.testing.assert_array_equal(ds.u_ref.isel(x_u=[0,-1]),0)
        np.testing.assert_allclose(ds.zeta_ref.isel(time=0),2.5/np.sqrt(2*np.pi)*np.exp(-.5*(ds.x_zeta/100000.)**2),rtol=2e-5,atol=1e-7)
