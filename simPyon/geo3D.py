from matplotlib import cm
import numpy as np

def stl2mesh3d(stl_mesh,ymin = -5):

    vecs = stl_mesh.vectors[stl_mesh.vectors[:,2,1]>=ymin]
    vecs = vecs[vecs[:,1,1]>=ymin]
    vecs = vecs[vecs[:,0,1]>=ymin]
    
    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points) 
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    p, q, r = vecs.shape #(p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(vecs.reshape(p*q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])
    return vertices, I, J, K

def mesh_stl(stl,origin = np.zeros(3)):
        from stl import mesh
        import numpy as np
        your_mesh = mesh.Mesh.from_file(stl)
        for shift,dim in zip(origin,[your_mesh.x,your_mesh.y,your_mesh.z]):
            dim+=shift
       
        return(your_mesh)
    
def shape_stl(your_mesh,geo):
    import plotly.graph_objects as go
    vertices, I, J, K = stl2mesh3d(your_mesh)
    x, y, z = vertices.T
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z, 
        i=I, 
        j=J, 
        k=K, 
#             flatshading=True,
#             colorscale=colorscale, 
#             intensity=z, 
#             name='AT&T',
#             showscale=False
    )
    return(mesh)

def mplcm_to_plotly(col):
    from matplotlib import colors
    return('rgb%s'%str(colors.colorConverter.to_rgb(col)))


def sim_shapes3D(sim):
    import plotly.graph_objects as go
    shapes = []
    for pa,info in zip(sim.pa,sim.pa_info):
        mesh = mesh_stl(pa.replace('.pa','.stl'),origin = info['pa_offset_position'])
        shapes.append(shape_stl(mesh,sim.geo))

    for labs,vecs in sim.geo.verts.items():
        col = mplcm_to_plotly(cm.viridis(labs/10))
        for v in vecs:
            shapes.append(go.Scatter3d(x = v[:,0],y = np.zeros(len(v)), z = v[:,1],mode = 'lines',
                                    legendgroup = labs,hoverinfo = 'skip',showlegend = False,
                                       line = {'color':col}
                                               ))
            shapes.append(go.Scatter3d(x = v[:,0],y = np.zeros(len(v)), z = -v[:,1],mode = 'lines',
                                    legendgroup = labs,hoverinfo = 'skip',showlegend = False,
                                       line = {'color':col}
                                               ))
    return(shapes)

def show_shapes3D(shapes):
    import plotly.graph_objects as go
    fig = go.Figure(data = shapes)
    return(fig)

def sim_traj3D_shapes(sim,cmap = cm.plasma,eng_cmap = True):
    import plotly.graph_objects as go
    def traj_pltr_3d(traj,cmp,norm):
        if eng_cmap:
            cval = traj['ke'].values[0]/norm
        else:
            cval = np.random()
        col = mplcm_to_plotly(cmp(cval))
        return(go.Scatter3d(x = traj['x'],y = traj['z'],z = traj['y'],mode = 'lines',line = {'color':col}))

    norm = np.max(sim.source['ke'].dist_out)

    return(list(sim.traj_data.groupby('n').apply(traj_pltr_3d,cmp = cmap,norm = norm).values))
    


