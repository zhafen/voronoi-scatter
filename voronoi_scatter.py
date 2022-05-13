import copy
import numpy as np
import scipy
import scipy.spatial
import tqdm
import warnings
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

########################################################################

def scatter(
    x,
    y,
    labels = None,
    fontsize = 10,
    label_kwargs = {},
    offset_magnitude = 5,
    ax = None,
    xlim = None,
    ylim = None,
    plot_scatter = True,
    plot_cells = False,
    cell_kwargs = {},
    colors = None,
    color_default = 'none',
    edgecolors = None,
    edgecolor_default = 'k',
    cmap = 'cubehelix',
    norm = None,
    vmin = None,
    vmax = None,
    hatching = None,
    plot_label_box = False,
    qhull_options = 'Qbb Qc Qz',
    **scatter_kwargs
):
    '''Matplotlib scatter plot with labels placed in voronoi cells
    if there is space. The calculations used to make this possible
    rely on converting to-and-from display vs data units. As such,
    adjustments to axis limits or scale should be performed *prior*
    to calling this function, or as part of this function
    (w/ e.g. xlim).

    Args:
    x, y: float or array-like, shape (n, )
        The data positions.

    labels: array-like of strs, shape (n, )
        Labels for the plots.

    fontsize: int
        Fontsize for labels.
    
    label_kwargs: dict
        Keyword arguments passed to ax.annotate for the labels.

    offset_magnitude: int
        Distance in points from the scatter point to the label.

    ax: None or axis object
        If provided, the axis to plot on.
        If not provided, a new figure will be created.

    xlim, ylim: None or array-like, shape (2, )
        x- and y-limits. Set automatically if not.
        *Do not use set_xlim or set_ylim independently of these
        arguments, nor set_aspect_ratio, or the labels may break.*

    plot_scatter: bool
        If True, call ax.scatter using the inputs and arguments to
        **kwargs

    plot_cells: bool
        If True, plot the voronoi cells the scatter points correspond to.

    cell_kwargs: dict
        Keyword arguments passed to descartes.patch.PolygonPatch

    colors: None or array-like, shape (n, )
        Colors for each scatter point or cell. Transformed into actual colors with cmap.

    color_default: str
        Default color for points and cells

    edgecolors: None or array-like, shape (n, )
        Edgecolors for each scatter point or cell. Transformed into actual colors with cmap.

    edgecolor_default: str
        Default edgecolor for poitns and cells

    cmap: str or colormap instance
        Colormap to use

    norm: None or matplotlib.colors normalization object.
        Normalization for the colors.

    vmin, vmax: None or float
        Color limits. Overriden by norm, if passed.

    hatching: None or array-like of strs, shape (n, )
        Hatching to use for cells.

    plot_label_box: bool
        If True plot a box around each label. Useful for debugging.

    qhull_options: str
        Additional options to pass to Qhull via scipy.spatial.Voronoi.
        See Qhull manual for details.

    **scatter_kwargs
        Additional arguments passed to ax.scatter
    '''

    # Format data
    points = np.array([ x, y ]).transpose()
    
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        
    # Limits
    if xlim is None:
        if ax.get_xscale() == 'log':
            x = np.log10( x )
        xmin = np.nanmin( x )
        xmax = np.nanmax( x )
        xwidth = xmax - xmin
        xlim = [ xmin - 0.1 * xwidth, xmax + 0.1 * xwidth ]
        if ax.get_xscale() == 'log':
            xlim = 10.**np.array( xlim )
    if ylim is None:
        # Auto limits depends on if logscale or not
        if ax.get_yscale() == 'log':
            y = np.log10( y )
        ymin = np.nanmin( y )
        ymax = np.nanmax( y )
        ywidth = ymax - ymin
        ylim = [ ymin - 0.1 * ywidth, ymax + 0.1 * ywidth ]
        if ax.get_yscale() == 'log':
            ylim = 10.**np.array( ylim )
    if ( vmin is None ) and ( colors is not None ):
        vmin = np.nanmin( colors )
    if ( vmax is None ) and ( colors is not None ):
        vmax = np.nanmax( colors )
        
    ax.set_xlim( xlim )
    ax.set_ylim( ylim )

    # Convert to colors arrays
    if ( norm is None ) and ( colors is not None ):
        norm = matplotlib.colors.Normalize( vmin=vmin, vmax=vmax )
    if isinstance( cmap, str ):
        cmap = matplotlib.cm.get_cmap( cmap )

    # Duplicate coordinates are not handled well
    points, unique_inds = np.unique( points, axis=0, return_index=True )
    if labels is not None:
        labels = np.array( labels )[unique_inds]
    if colors is not None:
        colors = np.array( colors )[unique_inds]
        colors = cmap( norm( colors ) )
    if edgecolors is not None:
        try:
            edgecolors = np.array( edgecolors )[unique_inds]
            edgecolors = cmap( norm( edgecolors ) )
        except IndexError:
            warnings.warn( 'edgecolors could not be converted to an array' )
    if hatching is not None:
        hatching = hatching[unique_inds]
        if ( edgecolor_default == 'none' ) or ( edgecolors is not None ):
            warnings.warn( 'Hatchcolor and edgecolor are the same in matplotlib.' )

    # Matplotlib scatter plot
    if plot_scatter:

        used_scatter_kwargs = {}
        if colors is None:
            used_scatter_kwargs['color'] = color_default
        if edgecolors is None:
            used_scatter_kwargs['edgecolors'] = edgecolor_default
        else:
            used_scatter_kwargs['edgecolors'] = edgecolors
        used_scatter_kwargs.update( scatter_kwargs )

        x, y = points.transpose()
        ax.scatter(
            x, y,
            c = colors,
            **used_scatter_kwargs
        )

    # Work in log space if log axes
    if ax.get_xscale() == 'log':
        points[:,0] = np.log10( points[:,0] )
    if ax.get_yscale() == 'log':
        points[:,1] = np.log10( points[:,1] )

    vor = scipy.spatial.Voronoi( points, qhull_options=qhull_options )
    
    ptp_bound = vor.points.ptp( axis=0 )
    center = vor.points.mean( axis=0 )

    for i, point in enumerate( tqdm.tqdm( points ) ):
        
        # Get data for this point
        i_region = vor.point_region[i]
        region = np.array( vor.regions[i_region] )
        is_neg = region == -1
        is_on_edge = is_neg.sum() > 0
        region = region[np.invert(is_neg)]
        vertices = vor.vertices[region]
        
        # Add additional points to the vertices for the regions
        # that are on the edge. This is taken from scipy's source code for the most part.
        if is_on_edge:
            add_vertices = []
            for j, pointidx in enumerate( vor.ridge_points ):
                simplex = np.array( vor.ridge_vertices[j] )
                if ( i in pointidx ) and ( -1 in vor.ridge_vertices[j] ):

                    ii = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                    t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[pointidx].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    if (vor.furthest_site):
                        direction = -direction
                    far_point = vor.vertices[ii] + direction * ptp_bound.max()

                    add_vertices.append( far_point )
            # If found a vertex, add it on
            if len( add_vertices ) > 0:
                vertices = np.concatenate( [ vertices, add_vertices ], axis=0 )
            else:
                raise Exception( 'No vertex found, crashing.' )

        # Convert back to non-logspace prior to making vertices
        # This is only done for the polygon that will be plotted
        vertices_for_plot = copy.copy( vertices )
        if ax.get_xscale() == 'log':
            vertices_for_plot[:,0] = 10.**vertices_for_plot[:,0]
        if ax.get_yscale() == 'log':
            vertices_for_plot[:,1] = 10.**vertices_for_plot[:,1]
            
        # Construct a shapely polygon for the region
        region_polygon = Polygon( vertices ).convex_hull
        region_polygon_for_plot = Polygon( vertices_for_plot ).convex_hull
        
        # Plot the cell
        if plot_cells:
            used_cell_kwargs = {
                'alpha': 1.,
            }
            used_cell_kwargs.update( cell_kwargs )
            if colors is not None:
                facecolor = colors[i]
            else:
                facecolor = color_default
            if edgecolors is not None:
                edgecolor = edgecolors[i]
            else:
                edgecolor = edgecolor_default
            if hatching is not None:
                used_cell_kwargs['hatch'] = hatching[i]
            patch = PolygonPatch(
                region_polygon_for_plot,
                facecolor = facecolor,
                edgecolor = edgecolor,
                **used_cell_kwargs
            )
            ax.add_patch( patch )
            
        # Add a label, trying a few orientations
        if labels is not None:
            has = [ 'left', 'center', 'right' ]
            vas = [ 'bottom', 'center', 'top' ]
            offsets = offset_magnitude * np.array([ 1, 0, -1 ])
            break_out = False
            for iii in [ 0, 1, 2 ]:
                for jjj in [ 0, 1, 2 ]:

                    xytext = np.array([ offsets[iii], offsets[jjj] ])

                    # Handle diagonals
                    if ( iii != 1 ) & ( jjj != 1 ):
                        xytext = xytext / np.sqrt( 2. )

                    used_kwargs = dict(
                        xycoords = 'data',
                        xytext = xytext,
                        textcoords = 'offset points',
                        ha = has[iii],
                        va = vas[jjj],
                        fontsize = fontsize,
                    )
                    used_kwargs.update( label_kwargs )

                    # Place points, accounting for scale
                    if ax.get_xscale() == 'log':
                        x_point = 10.**point[0]
                    else:
                        x_point = point[0]
                    if ax.get_yscale() == 'log':
                        y_point = 10.**point[1]
                    else:
                        y_point = point[1]

                    text = ax.annotate(
                        text = labels[i],
                        xy = ( x_point, y_point ),
                        **used_kwargs
                    )
                    text.set_path_effects([
                        path_effects.Stroke( linewidth=text.get_fontsize() / 5., foreground='w' ),
                        path_effects.Normal()
                    ])

                    # Create a polygon for the label
                    bbox_text = text.get_window_extent( ax.figure.canvas.get_renderer() )
                    display_to_data = ax.transData.inverted()
                    text_data_corners = display_to_data.transform( bbox_text.corners() )
                    text_data_corners = text_data_corners[[0,1,3,2],:] # Reformat

                    # Create two versions, one for logspace calcs and one for plotting
                    text_data_corners_for_calc = copy.copy( text_data_corners )
                    if ax.get_xscale() == 'log':
                        text_data_corners_for_calc[:,0] = np.log10( text_data_corners_for_calc[:,0] )
                    if ax.get_yscale() == 'log':
                        text_data_corners_for_calc[:,1] = np.log10( text_data_corners_for_calc[:,1] )
                    text_polygon = Polygon( text_data_corners_for_calc )
                    text_polygon_for_plot = Polygon( text_data_corners )
                    
                    text.set_visible( False )
                    
                    # We'll never fit it in if it's just too large
                    if text_polygon.area > region_polygon.area:
                        break_out = True
                        break
                        
                    # If it doesn't fit in the region try again
                    if not region_polygon.contains( text_polygon ):
                        continue
                        
                    # If it doesn't fit in the bounds try again
                    if text_polygon_for_plot.bounds[0] < xlim[0]:
                        continue
                    if text_polygon_for_plot.bounds[1] < ylim[0]:
                        continue
                    if text_polygon_for_plot.bounds[2] > xlim[1]:
                        continue
                    if text_polygon_for_plot.bounds[3] > ylim[1]:
                        continue

                    # If we find a good option stop iterating
                    text.set_visible( True )
                    if plot_label_box:
                        patch = PolygonPatch(
                            text_polygon_for_plot,
                            facecolor = 'none',
                            edgecolor = 'k',
                        )
                        ax.add_patch( patch )
                    break_out = True
                    break
                if break_out:
                    break

    return ax, vor
