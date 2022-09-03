# mesh_grid

`mesh_grid` is an accelerated library converts discrete triangular mesh to piece-wise smooth signed distance function (SDF).

## Requirement
`mesh_grid` has been tested its compatiblity on pytorch==[1.1.0-1.9.0].

## Install
It is recommend to installl `mesh_grid` via pip directly.
```bash
pip install git+https://github.com/generalizable-neural-performer/gnr.git@mesh_grid
```

## Usage
```python
from mesh_grid_searcher import MeshGridSearcher
import trimesh

## load mesh
mesh = trimesh.load('yourmesh.obj')
vertices = torch.from_numpy(mesh.vertices).float().cuda()
faces = torch.from_numpy(faces).int().cuda()

# create a searcher
mgs = MeshGridSearcher(vertices, faces)
points = torch.rand([100, 3]).float().cuda()

# compute the signed distance function of points
sdf = mgs.signed_distance_function(points)
```

## Citation
If you find this library is useful to your work, please cite the following paper below:

```
@article{cheng2022generalizable,
    title={Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    author={Cheng, Wei and Xu, Su and Piao, Jingtan and Qian, Chen and Wu, Wayne and Lin, Kwan-Yee and Li, Hongsheng},
    journal={arXiv preprint arXiv:2204.11798},
    year={2022}
}
```

