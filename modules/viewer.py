from IPython.display import display_html
from bokeh.models import TapTool, CustomJS, ColumnDataSource, HoverTool, ColumnDataSource
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.embed import components
import ase.io

output_notebook()

def show_structure(structure, filename='tmp/example_structure.in'):
    ase.io.write(filename, structure, format='aims')
    
    page = """<!DOCTYPE html>
        <html>
            <body>
        <!-- load JSmol; relative path works, absolute fails, might be access rights related -->
        <script type="text/javascript" src="JSmol.min.js"></script> 
        <script type="text/javascript">
        
            Jmol.setDocument(false);
            
            
            var jmolApplet0;
            var info = {
                width: 400,
                height: 300,
                use: "HTML5",
                j2spath: "./j2s",
                script: 'load %s'
            };
 
            $(document).ready(function() {
                $("#appdiv").html(Jmol.getAppletHtml("jmolApplet0", info))
            });


        </script>
        
        <div style="display: table; margin: 0 auto;" id="appdiv"></div> 
            </body>
        </html>
    """ %filename
    
    display_html(page, raw=True)






def show_map(df, D_selcted, P_predict, features):

    hover = HoverTool(
            tooltips="""
            <div>
                <div>
                    <img
                        src="@imgs" height="56" alt="@imgs" width="56"
                        style="display: table; margin: 0 auto;"
                        border="2"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 15px; font-weight: bold;">@chemical_formulas</span>
                </div>
                
                <div >
                    <span style="font-size: 10px;">Ref. &#916E = @Ref eV/atom</span><br>
                    <span style="font-size: 10px;">Pred. &#916E = @Pred eV/atom</span>
                </div>
                
                <div>
                    <span style="font-size: 10px;">Location</span>
                    <span style="font-size:  8px; color: #696;">($x, $y)</span>
                </div>
            </div>
            """
        )
    code = """var idx_struc = cb_data.source['selected']['1d'].indices;
              if (!Number.isInteger(idx_struc)){idx_struc=idx_struc[0]}
              var data = source.data;
              var geo_file = data['geo_files'][idx_struc]
              var chemical_formula = data['chemical_formulas'][idx_struc]
              var structure_type = data['legend'][idx_struc]

              change_geo(geo_file, i_click);
              document.getElementById("chemical_formula" + i_click%2).innerHTML = String(chemical_formula)
              document.getElementById("structure_type" + i_click%2).innerHTML = String(structure_type)

              i_click = i_click + 1;

              """
    
    # data to lists
    structures =  sorted(set(df['min_struc_type']))
    ref_colors = ['blue', 'red', 'green', 'orange', 'black']

    chemical_formulas = df.index.tolist() 
    P, min_structures, atoms_list = zip(*df[['energy_diff', 'min_struc_type', 'struc_obj_min']].values)
    colors = [ref_colors[i] for min_struc in min_structures for i, ref_struc in enumerate(structures) if ref_struc==min_struc]
    imgs = ['tmp/Thumbnail_%s_%s.png' % (min_structures[i], chemical_formulas[i])  for i in range(len(atoms_list))]
    geo_files = ['tmp/Geo_%s_%s.in' % (min_structures[i], chemical_formulas[i])  for i in range(len(atoms_list))]

    # make thumbnails and geometry files out of structure supercells
    for i, atoms in enumerate(atoms_list):
        atoms_super = atoms * [3, 3, 3]
        atoms_super.write(imgs[i], format='png', 
                    rotation='10z,-80x', radii=0.5, scale=100)
        atoms_super.write(geo_files[i], format='aims')

    source = ColumnDataSource(
            data=dict(
                x=D_selcted[:, 0],
                y=D_selcted[:, 1],
                chemical_formulas=chemical_formulas,
                color=colors,
                legend=min_structures,
                imgs=imgs,
                geo_files=geo_files,
                Ref=P,
                Pred=P_predict
            )
        )

    p = figure(plot_width=600, plot_height=300, tools=[hover,"tap", "box_zoom", "pan", "reset"], 
               x_axis_label=features[0],  y_axis_label=features[1])
    p.circle('x', 'y', color='color', size=20, source=source, legend='legend')
    
    taptool = p.select(type=TapTool)
    taptool.callback = CustomJS(args=dict(source=source), code=code)


    script, div = components(p)

    page = """<!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Molecule #0</title>

    <!-- load JSmol; relative path works, absolute fails, might be access rights related -->
    <script type="text/javascript" src="JSmol.min.js"></script> 
    <script type="text/javascript">
        var i_click = 0;
        Jmol.setDocument(false);


        jsmolReady = function(app) {
            Jmol.evaluateVar(app, "appletdiv");

        };

        <!-- see http://wiki.jmol.org/index.php/Jmol_JavaScript_Object for functionality -->
        var info = {
            width: 400,
            height: 300,
            use: "HTML5",
            j2spath: "./j2s",

        };

        get_info = function(fil) {
        var info_here = {
            width: 400,
            height: 300,
            use: "HTML5",
            j2spath: "./j2s",
            script: 'load ' + fil
        };
        return info_here;
        }



        $(document).ready(function() {
            $("#appdiv0").html(Jmol.getAppletHtml("jsmol0", info))
            $("#appdiv1").html(Jmol.getAppletHtml("jsmol1", info))
        });

        change_geo = function(filename, i_click_){
        var i_geo_box = i_click %s 2;
        $(document).ready(function() {

            $("#appdiv" + i_geo_box).html(Jmol.getAppletHtml("jsmol" + i_geo_box, get_info(filename)))
        });
        };

    </script>
    %s
</head>
<body>
<br>
<br>
    %s
    <table >
          <tr>
                    <td align="center">
                        <table id="jsmol_table">
                        <tr>
                            <th>Chemical Formula</th>
                            <th>Structure Type</th>
                        </tr>
                        <tr>
                            <td> <div id="chemical_formula0"> &nbsp; </div> </td>    
                            <td> <div id="structure_type0">  </td>  
                        </tr>
                        <tr>
                            <td colspan=2 class="none"> 
                                <div id="appdiv0"></div>
                            </td>  
                        </tr>
                        </table>
                    </td>


                    <td align="center">
                        <table id="jsmol_table">
                        <tr>
                            <th>Chemical Formula</th>
                            <th>Structure Type</th>
                        </tr>
                        <tr>
                            <td> <div id="chemical_formula1"> &nbsp; </div> </td> 
                            <td> <div id="structure_type1">  </td>  
                        </tr>
                        <tr>
                            <td colspan=2 class="none"> 
                                <div id="appdiv1"></div>
                            </td>  
                        </tr>
                        </table>
                    </td>
    </table>
        </body>
    </html>
    """ %('%', script, div)

    display_html(page, raw=True)
