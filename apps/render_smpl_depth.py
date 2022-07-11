from tqdm import tqdm
import numpy as np
import argparse
import struct
import sys
import cv2
import os
import re
from multiprocessing import Queue, Lock, Process

from trimesh import load_mesh
base_dir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type = str, required=True)
parser.add_argument('--outdir', type = str, default = '')
parser.add_argument('--annotdir', type = str, default = '')
parser.add_argument('--workers', type = int, default = 8)

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False, with_texture_image=False):
	vertex_data = []
	norm_data = []
	uv_data = []

	face_data = []
	face_norm_data = []
	face_uv_data = []

	if isinstance(mesh_file, str):
		f = open(mesh_file, "r")
	else:
		f = mesh_file
	for line in f:
		if isinstance(line, bytes):
			line = line.decode("utf-8")
		if line.startswith('#'):
			continue
		values = line.split()
		if not values:
			continue

		if values[0] == 'v':
			v = list(map(float, values[1:4]))
			vertex_data.append(v)
		elif values[0] == 'vn':
			vn = list(map(float, values[1:4]))
			norm_data.append(vn)
		elif values[0] == 'vt':
			vt = list(map(float, values[1:3]))
			uv_data.append(vt)

		elif values[0] == 'f':
			# quad mesh
			if len(values) > 4:
				f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
				face_data.append(f)
				f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
				face_data.append(f)
			# tri mesh
			else:
				f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
				face_data.append(f)
			
			# deal with texture
			if len(values[1].split('/')) >= 2:
				# quad mesh
				if len(values) > 4:
					f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
					face_uv_data.append(f)
					f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
					face_uv_data.append(f)
				# tri mesh
				elif len(values[1].split('/')[1]) != 0:
					f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
					face_uv_data.append(f)
			# deal with normal
			if len(values[1].split('/')) == 3:
				# quad mesh
				if len(values) > 4:
					f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
					face_norm_data.append(f)
					f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
					face_norm_data.append(f)
				# tri mesh
				elif len(values[1].split('/')[2]) != 0:
					f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
					face_norm_data.append(f)
		elif 'mtllib' in line.split():
			mtlname = line.split()[-1]
			mtlfile = os.path.join(os.path.dirname(mesh_file), mtlname)
			with open(mtlfile, 'r') as fmtl:
				mtllines = fmtl.readlines()
				for mtlline in mtllines:
					# if mtlline.startswith('map_Kd'):
					if 'map_Kd' in mtlline.split():
						texname = mtlline.split()[-1]
						texfile = os.path.join(os.path.dirname(mesh_file), texname)
						texture_image = cv2.imread(texfile)
						texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
						break

	vertices = np.array(vertex_data)
	faces = np.array(face_data) - 1

	if with_texture and with_normal:
		uvs = np.array(uv_data)
		face_uvs = np.array(face_uv_data) - 1
		norms = np.array(norm_data)
		if norms.shape[0] == 0:
			norms = compute_normal(vertices, faces)
			face_normals = faces
		else:
			norms = normalize_v3(norms)
			face_normals = np.array(face_norm_data) - 1
		if with_texture_image:
			return vertices, faces, norms, face_normals, uvs, face_uvs, texture_image
		else:
			return vertices, faces, norms, face_normals, uvs, face_uvs

	if with_texture:
		uvs = np.array(uv_data)
		face_uvs = np.array(face_uv_data) - 1
		return vertices, faces, uvs, face_uvs

	if with_normal:
		# norms = np.array(norm_data)
		# norms = normalize_v3(norms)
		# face_normals = np.array(face_norm_data) - 1
		norms = np.array(norm_data)
		if norms.shape[0] == 0:
			norms = compute_normal(vertices, faces)
			face_normals = faces
		else:
			norms = normalize_v3(norms)
			face_normals = np.array(face_norm_data) - 1
		return vertices, faces, norms, face_normals

	return vertices, faces

def extract_float(text):
	flts = []
	for c in re.findall('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]+)',text):
		if c != '':
			try:
				flts.append(float(c))
			except ValueError as e:
				continue
	return flts

def natural_sort(files):
	return	sorted(files, key = lambda text: \
		extract_float(os.path.basename(text)) \
		if len(extract_float(os.path.basename(text))) > 0 else \
		[float(ord(c)) for c in os.path.basename(text)])
		
def load_ply(file_name):
	v = []; tri = []
	try:
		fid = open(file_name, 'r')
		head = fid.readline().strip()
		readl= lambda f: f.readline().strip()
	except UnicodeDecodeError as e:
		fid = open(file_name, 'rb')
		readl =	(lambda f: str(f.readline().strip())[2:-1]) \
			if sys.version_info[0] == 3 else \
			(lambda f: str(f.readline().strip()))
		head = readl(fid)
	if head.lower() != 'ply':
		return	v, tri
	form = readl(fid).split(' ')[1]
	line = readl(fid)
	vshape = fshape = [0]
	while line != 'end_header':
		s = [i for i in line.split(' ') if len(i) > 0]
		if len(s) > 2 and s[0] == 'element' and s[1] == 'vertex':
			vshape = [int(s[2])]
			line = readl(fid)
			s = [i for i in line.split(' ') if len(i) > 0]
			while s[0] == 'property' or s[0][0] == '#':
				if s[0][0] != '#':
					vshape += [s[1]]
				line = readl(fid)
				s = [i for i in line.split(' ') if len(i) > 0]
		elif len(s) > 2 and s[0] == 'element' and s[1] == 'face':
			fshape = [int(s[2])]
			line = readl(fid)
			s = [i for i in line.split(' ') if len(i) > 0]
			while s[0] == 'property' or s[0][0] == '#':
				if s[0][0] != '#':
					fshape = [fshape[0],s[2],s[3]]
				line = readl(fid)
				s = [i for i in line.split(' ') if len(i) > 0]
		else:
			line = readl(fid)
	if form.lower() == 'ascii':
		for i in range(vshape[0]):
			s = [i for i in readl(fid).split(' ') if len(i) > 0]
			if len(s) > 0 and s[0][0] != '#':
				v += [[float(i) for i in s]]
		v = np.array(v, np.float32)
		for i in range(fshape[0]):
			s = [i for i in readl(fid).split(' ') if len(i) > 0]
			if len(s) > 0 and s[0][0] != '#':
				tri += [[int(s[1]),int(s[i-1]),int(s[i])] \
					for i in range(3,len(s))]
		tri = np.array(tri, np.int64)
	else:
		maps = {'float': ('f',4), 'double':('d',8), \
			'uint':  ('I',4), 'int':   ('i',4), \
			'ushort':('H',2), 'short': ('h',2), \
			'uchar': ('B',1), 'char':  ('b',1)}
		if 'little' in form.lower():
			fmt = '<' + ''.join([maps[i][0] for i in vshape[1:]]*vshape[0])
		else:
			fmt = '>' + ''.join([maps[i][0] for i in vshape[1:]]*vshape[0])
		l = sum([maps[i][1] for i in vshape[1:]]) * vshape[0]
		v = struct.unpack(fmt, fid.read(l))
		v = np.array(v).reshape(vshape[0],-1).astype(np.float32)
		v = v[:,:3]
		tri = []
		for i in range(fshape[0]):
			l = struct.unpack(fmt[0]+maps[fshape[1]][0], \
				fid.read(maps[fshape[1]][1]))[0]
			f = struct.unpack(fmt[0]+maps[fshape[2]][0]*l, \
				fid.read(l*maps[fshape[2]][1]))
			tri += [[f[0],f[i-1],f[i]] for i in range(2,len(f))]
		tri = np.array(tri).reshape(fshape[0],-1).astype(np.int64)
	fid.close()
	return	v, tri

def distortPoints(p, dist):
	dist = np.reshape(dist,-1) \
		if dist is not None else []
	k1 = dist[0] if len(dist) > 0 else 0
	k2 = dist[1] if len(dist) > 1 else 0
	p1 = dist[2] if len(dist) > 2 else 0
	p2 = dist[3] if len(dist) > 3 else 0
	k3 = dist[4] if len(dist) > 4 else 0
	k4 = dist[5] if len(dist) > 5 else 0
	k5 = dist[6] if len(dist) > 6 else 0
	k6 = dist[7] if len(dist) > 7 else 0
	x, y = p[...,0], p[...,1]
	x2 = x * x; y2 = y * y; xy = x * y
	r2 = x2 + x2
	c =	(1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / \
		(1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
	x_ = c*x + p1*2*xy + p2*(r2+2*x2)
	y_ = c*y + p2*2*xy + p1*(r2+2*y2)
	p[...,0] = x_
	p[...,1] = y_
	return p

def rasterize(v, tri, size, K = np.identity(3), \
		dist = None, persp = True, eps = 1e-6):
	h, w = size
	zbuf = np.ones([h, w], v.dtype) * float('inf')
	if dist is not None:
		valid = np.where(v[:,2] >= eps)[0] \
			if persp else np.arange(len(v))
		v_proj = v[valid,:2] / v[valid,2:]
		v_proj = distortPoints(v_proj, dist)
		v[valid,:2]= v_proj * v[valid,2:]
	v_proj = v.dot(K.T)[:,:2] / np.maximum(v[:,2:], eps) \
		if persp else v.dot(K.T)[:,:2]
	va = v_proj[tri[:,0],:2]
	vb = v_proj[tri[:,1],:2]
	vc = v_proj[tri[:,2],:2]
	front = np.cross(vc - va, vb - va)
	umin = np.maximum(np.ceil (np.vstack((va[:,0],vb[:,0],vc[:,0])).min(0)), 0)
	umax = np.minimum(np.floor(np.vstack((va[:,0],vb[:,0],vc[:,0])).max(0)),w-1)
	vmin = np.maximum(np.ceil (np.vstack((va[:,1],vb[:,1],vc[:,1])).min(0)), 0)
	vmax = np.minimum(np.floor(np.vstack((va[:,1],vb[:,1],vc[:,1])).max(0)),h-1)
	umin = umin.astype(np.int32)
	umax = umax.astype(np.int32)
	vmin = vmin.astype(np.int32)
	vmax = vmax.astype(np.int32)
	front = np.where(np.logical_and(np.logical_and( \
			umin <= umax, vmin <= vmax), front > 0))[0]
	for t in front:
		A = np.concatenate((vb[t:t+1]-va[t:t+1], vc[t:t+1]-va[t:t+1]),0)
		x, y = np.meshgrid(	range(umin[t],umax[t]+1), \
					range(vmin[t],vmax[t]+1))
		u = np.vstack((x.reshape(-1),y.reshape(-1))).T
		coeff = (u.astype(v.dtype) - va[t:t+1,:]).dot(np.linalg.pinv(A))
		coeff = np.concatenate((1-coeff.sum(1).reshape(-1,1),coeff),1)
		if persp:
			z = coeff.dot(v[tri[t], 2])
		else:
			z = 1 / np.maximum((coeff/v[tri[t],2:3].T).sum(1), eps)
		for i, (x, y) in enumerate(u):
			if  coeff[i,0] >= -eps \
			and coeff[i,1] >= -eps \
			and coeff[i,2] >= -eps \
			and zbuf[y,x] > z[i]:
				zbuf[y,x] = z[i]
	return	zbuf

def render_view(intri, dists, c2ws, meshes, view, i):
	K = intri[i]
	dist = dists[i]
	# c2w  = np.concatenate([c2ws[i],[[0,0,0,1]]], 0)
	w2c = np.linalg.inv(c2ws[i])
	# w2c = c2w
	out = os.path.join(args.outdir, os.path.basename(view))
	if not os.path.isdir(out):
		os.makedirs(out, exist_ok=True)
	imgs = [os.path.join(view, f) for f in os.listdir(view) \
		if f[-4:].lower() in ['.jpg','.png']]
	imgs = sorted(imgs) if len(imgs) > 1 else imgs
	for i in tqdm(range(len(imgs))):
		img = cv2.imread(imgs[i])
		try:
			if i < len(meshes) and meshes[i][-4:] == '.ply':
				v, tri = load_ply(meshes[i])
			elif i < len(meshes) and meshes[i][-4:] == '.npy':
				v = np.load(meshes[i])
			else:
				v, tri = load_obj_mesh(meshes[i])
		except:
			continue
		v_= v.dot(w2c[:3,:3].T) + w2c[:3,3:].T
		z = rasterize(v_, tri, img.shape[:2], K, dist)
		height = v_[:,1].max() - v_[:,1].min()
		z[z == float('inf')] = 0
		z = np.clip(np.round(z * 1000), 0, 65535).astype(np.uint16)
		cv2.imwrite(os.path.join(out, \
			'smpl_'+os.path.basename(imgs[i][:-4])+'.png'), z) 

class Worker(Process):

	def __init__(self, queue, lock):
		super(Worker, self).__init__()
		self.queue = queue
		self.lock = lock

	def run(self):
		while True:
			self.lock.acquire()
			if self.queue.empty():
				self.lock.release()
				break
			else:
				kwargs = self.queue.get()
				queue_len = self.queue.qsize()
				self.lock.release()
				print("started {}, {} jobs left".format(kwargs["view"], queue_len))
				render_view(**kwargs)

if __name__ == '__main__':
	args  = parser.parse_args()
	views = [os.path.join(args.datadir, 'image', f) \
			for f in os.listdir(os.path.join(args.datadir, 'image'))]
	if args.annotdir == '':
		args.annotdir = os.path.join(args.datadir, 'annots.npy')
	annot = np.load(os.path.join(args.annotdir), allow_pickle = True).item()['cams']
	intri = np.array([annot[view]['K'] for view in annot.keys()], np.float32)
	dists = np.array([annot[view]['D'] for view in annot.keys()], intri.dtype)
	c2ws  = np.array([annot[view]['c2w'] for view in annot.keys()]).astype(intri.dtype)
	if args.outdir == '': 
		args.outdir = os.path.join(args.datadir, 'smpl_depth')
	if not os.path.isdir(args.outdir):
		os.makedirs(args.outdir)
	if os.path.exists(os.path.join(args.datadir,'new_smpl')):
		meshes = [os.path.join(args.datadir,'new_smpl',f) \
			for f in os.listdir(os.path.join(args.datadir,'new_smpl')) \
			if f[-4:] == '.ply' or f[-4:] == '.obj']
		meshes = natural_sort(meshes)
	elif os.path.exists(os.path.join(args.datadir,'smpl')):
		meshes = [os.path.join(args.datadir,'smpl',f) \
			for f in os.listdir(os.path.join(args.datadir,'smpl')) \
			if f[-4:] == '.ply' or f[-4:] == '.obj']
		meshes = natural_sort(meshes)
	elif os.path.exists(os.path.join(args.datadir,'new_vertices')):
		meshes = [os.path.join(args.datadir,'new_vertices',f) \
			for f in os.listdir(os.path.join(args.datadir,'new_vertices')) \
			if f[-4:] == '.npy']
		tri = np.loadtxt(os.path.join(base_dir,'tri.txt')).astype(np.int64)
		meshes = natural_sort(meshes)
	elif os.path.exists(os.path.join(args.datadir,'vertices')):
		meshes = [os.path.join(args.datadir,'vertices',f) \
			for f in os.listdir(os.path.join(args.datadir,'vertices')) \
			if f[-4:] == '.npy']
		_, tri = load_obj_mesh('./smpl_t_pose/smplx.obj')
		tri = tri.astype(np.int64)
		meshes = natural_sort(meshes)

	queue = Queue()
	lock = Lock()

	for i, view in enumerate(natural_sort(views)):

		queue.put({
			'intri': intri, 
			'dists': dists, 
			'c2ws': c2ws, 
			'meshes': meshes, 
			'view': view, 
			'i': i,
		})

	print("num of workers", args.workers, flush=True)
	pool = [Worker(queue, lock) for _ in range(args.workers)]
	for worker in pool:  worker.start()
	for worker in pool:  worker.join()
