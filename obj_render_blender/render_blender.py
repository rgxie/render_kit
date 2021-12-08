# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Tested with Blender 2.9
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os, math, re
import bpy
import numpy as np
from glob import glob

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=24,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--resolution', type=int, default=600,
                    help='Resolution of the images.')
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])
    # print(param[0],param[1], theta, phi, param[6])

    camZ = param[2]*np.cos(phi) * param[3]
    temp = param[2]*np.sin(phi) * param[3]
    camX = temp * np.cos(theta)
    camY = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    print(camX, camY, camZ)
    return camX, camY , camZ


def normalize_obj(obj):
    maxP = [-9999999,-999999,-999999]
    minP= [9999999,999999,999999]
    for vert in obj.data.vertices:
        p=vert.co
        minP[0] = p[0] if p[0] < minP[0] else minP[0]
        minP[1] = p[1] if p[1] < minP[1] else minP[1]
        minP[2]= p[2] if p[2] < minP[2] else minP[2]
        maxP[0]= p[0] if p[0] > maxP[0] else maxP[0]
        maxP[1] = p[1] if p[1] > maxP[1] else maxP[1]
        maxP[2]= p[2] if p[2] > maxP[2] else maxP[2]
    box_len = maxP[0] - minP[0]
    if box_len < (maxP[1] - minP[1]):
        box_len = maxP[1] - minP[1]
    if box_len < (maxP[2] - minP[2]):
        box_len = maxP[2] - minP[2]
    for vert in obj.data.vertices:
        vert.co[0] = (vert.co[0] - minP[0]) * 2 / box_len - 1
        vert.co[1] = (vert.co[1] - minP[1]) * 2 / box_len - 1
        vert.co[2] = (vert.co[2] - minP[2]) * 2 / box_len - 1

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = args.color_depth # ('8', '16')
render.image_settings.file_format = args.format # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
render.film_transparent = True

scene.use_nodes = True
scene.view_layers["ViewLayer"].use_pass_normal = True
scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
scene.view_layers["ViewLayer"].use_pass_object_index = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')

# Create albedo output nodes
alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
albedo_file_output.base_path = ''
albedo_file_output.file_slots[0].use_node_format = True
albedo_file_output.format.file_format = args.format
albedo_file_output.format.color_mode = 'RGBA'
albedo_file_output.format.color_depth = args.color_depth
links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

# Delete default cube
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')


file_extension=os.path.split(args.obj)[1].split('.')[-1]
if file_extension == 'fbx':
    bpy.ops.import_scene.fbx(filepath=args.obj)
elif file_extension == 'obj':
    bpy.ops.import_scene.obj(filepath=args.obj)
else:
    print('obj file type don\'t supported!')
    sys.exit(0)



obj = bpy.context.selected_objects[0]

context.view_layer.objects.active = obj


# max_dimension=max(obj.dimensions[0],obj.dimensions[1],obj.dimensions[2])
# scale_factor=2/max_dimension
# obj.scale = obj.scale*scale_factor
# obj.location = (0,0,0)

# for vert in obj.data.vertices:
#     print(vert.co)

normalize_obj(obj)
obj.scale = (1,1,1)
obj.location = (0,0,0)

# for vert in obj.data.vertices:
#     print(vert.co)

# Possibly disable specular shading
for slot in obj.material_slots:
    node = slot.material.node_tree.nodes['Principled BSDF']
    node.inputs['Specular'].default_value = 0.05

if args.scale != 1:
    bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
    bpy.ops.object.transform_apply(scale=True)
if args.remove_doubles:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode='OBJECT')
if args.edge_split:
    bpy.ops.object.modifier_add(type='EDGE_SPLIT')
    context.object.modifiers["EdgeSplit"].split_angle = 1.32645
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

# Set objekt IDs
obj.pass_index = 1

# Make light just directional, disable shadows.
light = bpy.data.lights['Light']
light.type = 'SUN'
light.use_shadow = True
# Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 10.0


#Place camera
cam = scene.objects['Camera']

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty
scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
#always target location (0,0,0)
cam_constraint.target = cam_empty

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'

#get the dir with objs 
model_identifier = os.path.split(os.path.split(args.obj)[0])[1]

file_name=args.obj.split('\\')[-1]
file_name=file_name.split('.')[0]

fp = os.path.join(os.path.abspath(args.output_folder), file_name)

current_rot_value = 0
metastring = ""
for i in range(0, args.views):
    angle_rand = np.random.rand(3)
    theta = current_rot_value
    phi = 45
    dist = 6
    param = [theta, phi , dist, 1]
    camX, camY, camZ = camera_info(param)
    cam.location = (camX, camY, camZ)
    print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

    render_file_path = fp +"\\shade\\"+'shade_r_{0:03d}'.format(int(i * stepsize))

    scene.render.filepath = render_file_path
    albedo_file_output.file_slots[0].path = fp +"\\albedo\\"+ 'albedo_r_{0:03d}'.format(int(i * stepsize))

    bpy.ops.render.render(write_still=True)  # render still

    metastring = metastring + "{} {} {} {} {} \n" \
                     .format(theta, phi, 0, dist, 35,)
    current_rot_value += stepsize
#meta_data_path = os.path.join(os.path.abspath(args.output_folder), file_name)                 
with open(fp+"\\rendering_metadata.txt", "w") as f:
        f.write(metastring)
