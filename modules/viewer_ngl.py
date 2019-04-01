from bokeh.models import TapTool, CustomJS, ColumnDataSource, HoverTool, ColumnDataSource
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from ipywidgets import  HBox, VBox,  Label
import nglview as nv

class Viewer(object):
    
    def __init__(self, show_geos=True):
        self.struc_indices = [None, None]
        self.geo_counter = -1
        self.show_geos = show_geos
        

    def show_map(self, df, df_D, indices_selected, is_show=True):

        hover = HoverTool(
                tooltips="""
                <div>
                    <div>
                        <img
                            src="@imgs" height="42" alt="@imgs" width="42"
                            style="float: left; margin: 0px 15px 15px 0px;"
                            border="2"
                        ></img>
                    </div>
                    <div>
                        <span style="font-size: 17px; font-weight: bold;">@desc</span>
                        <span style="font-size: 15px; color: #966;">[$index]</span>
                    </div>
                    <div>
                        <span style="font-size: 15px;">Location</span>
                        <span style="font-size: 10px; color: #696;">($x, $y)</span>
                    </div>
                </div>
                """
            )
        code = """var idx_point = cb_data.source['selected']['1d'].indices;
                  if (!Number.isInteger(idx_point)){idx_point=idx_point[0]}
                  var cell = Jupyter.notebook.get_selected_cell(); 
                  var curr_cell_idx = Jupyter.notebook.find_cell_index(cell)
                  next_cell_idx = curr_cell_idx + 1;
                  Jupyter.notebook.kernel.execute(`idx_struc=${idx_point}`);
                  Jupyter.notebook.kernel.execute(`idx=${curr_cell_idx}`);
                  Jupyter.notebook.execute_cells([next_cell_idx])"""
        
        structures =  sorted(set(df['min_struc_type']))
        ref_colors = ['blue', 'red']
        
        chemical_formulas = df.index.tolist() 
        min_structures = df['min_struc_type'].tolist()
        colors = [ref_colors[i] for min_struc in min_structures for i, ref_struc in enumerate(structures) if ref_struc==min_struc]
        self.atoms_list = df['struc_obj_min'].tolist()
        
        #for struc in structures:
        source = ColumnDataSource(
                data=dict(
                    x=df_D.values[:, indices_selected[0]],
                    y=df_D.values[:, indices_selected[1]],
                    desc=chemical_formulas,
                    color = colors,
                    legend = min_structures,
                    imgs = ['data/Thumbnail_%s_%s.png' % (min_structures[i], chemical_formulas[i])  for i in range(df.shape[0])]
                )
            )

        p = figure(plot_width=600, plot_height=300, tools=[hover,"tap", "box_zoom", "pan", "reset"], 
                   x_axis_label=df_D.columns[indices_selected[0]],  y_axis_label=df_D.columns[indices_selected[1]])
        p.circle('x', 'y', color='color', size=20, source=source, legend='legend')
        if self.show_geos:
            taptool = p.select(type=TapTool)
            taptool.callback = CustomJS(args=dict(source=source), code=code)
        
        # if test.py is run is_show is False
        if is_show:
            show(p)
    def show_geometries(self, idx_struc):
        self.geo_counter += 1
        idx_box = self.geo_counter %2
        self.struc_indices[idx_box] = idx_struc

        view1 = nv.show_ase(self.atoms_list[self.struc_indices[0]] * [3, 3, 3])
        view1._remote_call('setSize', target='Widget', args=['%dpx' % (400,), '%dpx' % (400,)])
        if self.struc_indices[1] is None:
            display(view1)
        else:
            view2 = nv.show_ase(self.atoms_list[self.struc_indices[1]] * [3, 3, 3])
            view2._remote_call('setSize', target='Widget', args=['%dpx' % (400,), '%dpx' % (400,)])
            sidebyside = HBox([view1, view2])
            display(sidebyside)
