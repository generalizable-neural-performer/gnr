import numpy as np
import struct
import sys, os, re
import cv2
if sys.version_info[0] == 3:
    from functools import reduce
# type: (regularExpression, structPack, stringFormat, numpyType, isFloat)

decode_map = {
    'bool': ('([01])', '?', '%d', 1, np.dtype('bool'), False),
    'uchar':('([0-9]{1,3})', 'B', '%d', 1, np.uint8, False, 1),
    'uint8':('([0-9]{1,3})', 'B', '%d', 1, np.uint8, False),
    'byte': ('([0-9]{1,3})', 'B', '%d', 1, np.uint8, False),
    'unsigned char':('([0-9]{1,3})', 'B', '%d', 1, np.uint8, False),
    'char': ('(-?[0-9]{1,3})', 'b', '%d', 1, np.int8, False, 1),
    'int8': ('(-?[0-9]{1,3})', 'b', '%d', 1, np.int8, False),

    'ushort': ('([0-9]{1,5})', 'H', '%d', 2, np.uint16, False, 1),
    'uint16': ('([0-9]{1,5})', 'H', '%d', 2, np.uint16, False),
    'unsigned short': ('([0-9]{1,5})', 'H', '%d', 2, np.uint16, False),
    'short': ('(-?[0-9]{1,5})', 'h', '%d', 2, np.int16, False, 1),
    'int16': ('(-?[0-9]{1,5})', 'h', '%d', 2, np.int16, False),
    'half': ('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]*)', 'e', '%f', 2, np.float16, True, 1),
    'float16':('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]*)', 'e', '%f', 2, np.float16, True),

    'uint':  ('([0-9]{1,10})', 'I', '%u', 4, np.uint32, False, 1),
    'uint32':('([0-9]{1,10})', 'I', '%u', 4, np.uint32, False),
    'ulong': ('([0-9]{1,10})', 'I', '%u', 4, np.uint32, False),
    'unsigned':('([0-9]{1,10})', 'I', '%u', 4, np.uint32, False),
    'unsigned int':('([0-9]{1,10})', 'I', '%u', 4, np.uint32, False),
    'unsigned long':('([0-9]{1,10})', 'I', '%u', 4, np.uint32, False),
    'int':  ('(-?[0-9]{1,10})', 'i', '%d', 4, np.int32, False, 1),
    'long': ('(-?[0-9]{1,10})', 'i', '%d', 4, np.int32, False),
    'int32':('(-?[0-9]{1,10})', 'i', '%d', 4, np.int32, False),
    'float':('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]*)', 'f', '%f', 4, np.float32, True, 1),
    'single':('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]*)', 'f', '%f', 4, np.float32, True),
    'float32':('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]*)', 'f', '%f', 4, np.float32, True),

    'uint64':('([0-9]{1,20})', 'Q', '%lu', 8, np.uint64, False, 1),
    'ullong':('([0-9]{1,20})', 'Q', '%lu', 8, np.uint64, False),
    'unsigned long long':('([0-9]{1,20})', 'Q', '%lu', 8, np.uint64, False),
    'int64':('(-?[0-9]{1,19})', 'q', '%ld', 8, np.int64, False, 1),
    'llong':('(-?[0-9]{1,19})', 'q', '%ld', 8, np.int64, False),
    'long long':('(-?[0-9]{1,19})', 'q', 8, np.int64, False),
    'double':('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]*)', 'd', '%f', 8, np.float64, True, 1),
    'float64':('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]*)', 'd', '%f', 8, np.float64, True),
}

def max_precision(type1, type2):
    if decode_map[type1][5]:
        if decode_map[type2][5]:
            if decode_map[type1][3] < decode_map[type2][3]:
                return type2
            else:
                return type1
        else:
            if decode_map[type1][3] < decode_map[type2][3]:
                for t, c in decode_map.items():
                    if c[5] and c[3] >= decode_map[type2][3]:
                        return t
            else:
                return type1
    elif decode_map[type2][5]:
        if decode_map[type2][3] < decode_map[type1][3]:
            for t, c in decode_map.items():
                if c[5] and c[3] >= decode_map[type1][3]:
                    return t
        else:
            return type2
    else:
        if decode_map[type2][3] < decode_map[type1][3]:
            return type1
        elif decode_map[type2][3] < decode_map[type1][3]:
            return type2
        elif decode_map[type2][4] == decode_map[type2][4]:
            return type1
        else:
            for t, c in decode_map.items():
                if not c[5] and c[3] > decode_map[type1][3]:
                    return t
            return 'unsigned long long'

def decode(content, structure, num, form):
    if form.lower() == 'ascii':
        l = 0; d = []; lines = content.split('\n')
        for i in range(num):
            s = [j for j in lines[i].split(' ') if len(j) > 0]
            l += len(lines[i]) + 1; k = []; j = 0
            while j < len(s) and len(k) < len(structure):
                t = structure[len(k)]
                if t[:4] == 'list':
                    n = int(s[j]); t = t.split(':')[-1]
                    k += [[float(s[i]) if decode_map[t][5] else int(s[i]) \
                        for i in range(j+1,j+n+1)]]
                    j += n + 1
                else:
                    k += [float(s[j]) if decode_map[t][5] else int(s[j])]
                    j += 1
            d += [k]
    else:
        if form.lower() == 'binary_little_endian':
            c = '<'
        elif form.lower() == 'binary_big_endian':
            c = '>'
        l = 0; d = []
        for i in range(num):
            k = []
            while len(k) < len(structure) and l < len(content):
                t = structure[len(k)]
                if t[:4] == 'list':
                    t = t.split(':')
                    n = struct.unpack(c+decode_map[t[1]][1], \
                        content[l:l+decode_map[t[1]][3]])[0]
                    l += decode_map[t[1]][3]
                    k += [struct.unpack(c+decode_map[t[2]][1]*n, \
                        content[l:l+decode_map[t[2]][3]*n])]
                    l += decode_map[t[2]][3]*n
                else:
                    k += [struct.unpack(c+decode_map[t][1], \
                        content[l:l+decode_map[t][3]])[0]]
                    l += decode_map[t][3]
            d += [k]
    try:
        t = reduce(max_precision, [t if t[:4] != 'list' \
            else t.split(':')[-1] for t in structure])
        d = np.array(d, dtype = decode_map[t][4])
    except ValueError:
        print('Warning: Not in Matrix')
    return d, content[l:]

def load_ply(file_name):
    try:
        with open(file_name, 'r') as f:
            head = f.readline().strip()
            if head.lower() != 'ply':
                raise('Error: Not a valid PLY file')
            content = f.read()
            i = content.find('end_header\n')
            if i < 0:
                raise('Error: Not a valid PLY file')
            info = [[l for l in line.split(' ') if len(l) > 0] \
                for line in content[:i].split('\n')]
            content = content[i+11:]
    except UnicodeDecodeError as e:
        with open(file_name, 'rb') as f:
            head = f.readline().strip()
            if sys.version_info[0] == 3:
                head = str(head)[2:-1]
            else:
                head = str(head)
            if head.lower() != 'ply':
                raise('Error: Not a valid PLY file')
            content = f.read()
            i = content.find(b'end_header\n')
            if i < 0:
                raise('Error: Not a valid PLY file')
            if sys.version_info[0] == 3:
                cnt = str(content[:i])[2:-1].replace('\\n', '\n')
            else:
                cnt = str(content[:i])
            info = [[l for l in line.split(' ') if len(l) > 0] \
                for line in cnt.split('\n')]
            content = content[i+11:]
    form = 'ascii'
    elem_names = []
    elem = {}
    for i in info:
        if len(i) >= 2 and i[0] == 'format':
            form = i[1]
        elif len(i) >= 3 and i[0] == 'element':
            if len(elem_names) > 0:
                elem[elem_names[-1]] = (structure_name, structure)
            elem_names += [(i[1], int(i[2]))]
            structure_name = []
            structure = []
        elif len(i) >= 3 and i[0] == 'property' and len(elem_names) > 0:
            structure_name += [i[-1]]
            if i[1] == 'list' and len(i) >= 5:
                structure += [i[1] + ':' + i[2] + ':' + ' '.join(i[3:-1])]
            else:
                structure += [' '.join(i[1:-1])]
    if len(elem_names) > 0:
        elem[elem_names[-1]] = (structure_name, structure)
    elem_ = {}
    for k in elem_names:
        d, content = decode(content, elem[k][1], k[1], form)
        if 'face' in k[0] and isinstance(d, np.ndarray):
            d = d.reshape((k[1], -1))
        elem_[k[0]] = d # elem[k] = (elem[k][0], d)
    return elem_

def save_ply(file_name, elems, _type = 'binary_little_endian', comments = []):
    _type = _type.lower()
    types = {}
    if isinstance(comments, str):
        comments = [comments]
    comments = [c for l in comments for c in l.split('\n')]
    with open(file_name, 'w') as f:
        f.write('ply\nformat %s 1.0\n' % _type)
        for comment in comments:
            f.write('comment %s\n' % comment)
        for key, elem in elems.items():
            f.write('element %s %d\n' % (key, len(elem)))
            if isinstance(elem, np.ndarray):
                for e, c in decode_map.items():
                    if len(c) > 6 and c[4] == elem.dtype:
                        c = (e, c[1], c[2]); break
            else:
                c = ('int', 'i', '%d')
                for e in elem:
                    if hasattr(e, '__len__'):
                        for i in range(len(e)):
                            if int(e[i]) != e[i]:
                                c = ('float','f','%f'); break
                        if i != len(e): break
                    elif int(e) != e:
                        c = ('float','f','%f'); break
            if 'face' in key:
                tag = 'vertex_index'
                max_num = max([len(e) for e in elem])
                if max_num < 256:
                    l = ('uchar', 'B', '%d')
                elif max_num < 65536:
                    l = ('ushort', 'H', '%d')
                elif max_num < 4294967296:
                    l = ('uint', 'I', '%d')
                else:
                    l = ('uint64', 'Q', '%d')
                f.write('property list %s %s %s\n' % \
                    (l[0], c[0], tag))
                types[key] = (l, c)
            else:
                if 'vert' in key:
                    if len(elem) > 0:
                        if len(elem[0]) > 4:
                            tag = ['x', 'y', 'z', 'red', 'green', 'blue', 'alpha']
                        else:
                            tag = ['x', 'y', 'z', 'w']
                elif 'norm' or 'texcoord' in key:
                    tag = ['x', 'y', 'z', 'w']
                elif 'color' in key:
                    tag = ['red', 'green', 'blue', 'alpha']
                else:
                    tag = []
                if len(elem) > 0:
                    for j in range(len(elem[0])):
                        f.write('property %s %s\n' % (c[0], tag[j] \
                            if len(tag) > j else 'k%d'%(j-len(tag))))
                types[key] = c
        f.write('end_header\n')
    with open(file_name, 'ab' if 'binary' in _type else 'a') as f:
        for key, elem in elems.items():
            c = types[key]
            if len(c) == 2:
                for e in elem:
                    l = len(e)
                    if 'ascii' in _type:
                        f.write((c[0][2]+(' '+c[1][2])*l+'\n') % \
                            tuple([l]+list(e)))
                    elif 'little' in _type:
                        f.write(struct.pack('<'+c[0][1], l))
                        f.write(struct.pack('<'+c[1][1]*l, *e))
                    elif 'big' in _type:
                        f.write(struct.pack('>'+c[0][1], l))
                        f.write(struct.pack('>'+c[1][1]*l, *e))
            else:
                seg = len(elem[0]) if len(elem) > 0 else 1
                elem = [i for l in elem for i in l] \
                    if isinstance(elem, np.ndarray) else elem.reshape(-1)
                if 'ascii' in _type:
                    for i in range(len(elem)):
                        f.write((c[2] + '\n' if (i + 1) % seg == 0 \
                            else c[2] + ' ') % elem[i])
                elif 'little' in _type:
                    f.write(struct.pack('<'+c[1]*len(elem), *elem))
                elif 'big' in _type:
                    f.write(struct.pack('>'+c[1]*len(elem), *elem))


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

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
