import numpy as np
import os
import unittest

import matplotlib
import matplotlib.pyplot as plt

import voronoi_scatter

########################################################################

n = 50

########################################################################

class TestVoronoiScatter( unittest.TestCase ):

    def setUp( self ):

        self.output_dir = './test_output'
        os.makedirs( self.output_dir, exist_ok=True )

        np.random.seed( 1234 )

    ########################################################################

    def tearDown( self ):
        '''Save the resultant plot.'''

        filename = self._testMethodName + '.pdf'
        plt.gcf().savefig(
            os.path.join( self.output_dir, filename ), 
        )

    ########################################################################

    def test_runs( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )

        ax, vor = voronoi_scatter.scatter(
            x,
            y,
        )

    ########################################################################

    def test_labels( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
        )

    ########################################################################

    def test_label_kwargs( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            label_kwargs = { 'fontsize': 5, }
        )

    ########################################################################

    def test_fontsize( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            fontsize = 5,
        )

    ########################################################################
    def test_offset_magnitudes( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            offset_magnitude = 1,
        )

    ########################################################################

    def test_provide_axis( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        fig = plt.figure()
        ax = plt.gca()

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            ax = ax,
        )

    ########################################################################

    def test_set_lim( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            xlim = [ 0, 0.5 ],
            ylim = [ 0, 0.5 ],
        )

    ########################################################################

    def test_cells( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            plot_cells = True,
        )

    ########################################################################

    def test_scatter_naive_usage( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        z = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            color = 'k',
            edgecolors = 'b',
        )

    ########################################################################

    def test_scatter_colors( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        z = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            colors = z,
        )

    ########################################################################

    def test_scatter_edgecolors( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        z = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            colors = z,
            edgecolors = z / 2.,
        )

    ########################################################################

    def test_cell_colors( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        z = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            plot_cells = True,
            colors = z,
        )

    ########################################################################

    def test_cell_colors_vlim( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        z = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            plot_cells = True,
            colors = z,
            vmin = 0.5,
            vmax = 0.8,
        )

    ########################################################################

    def test_cell_colors_norm( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        z = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            plot_cells = True,
            colors = z,
            norm = matplotlib.colors.LogNorm( vmin=z.min(), vmax=1 ),
        )

    ########################################################################

    def test_cell_kwargs( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        z = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            plot_cells = True,
            colors = z,
            cell_kwargs = { 'alpha': 0.5 },
        )

    ########################################################################

    def test_hatching( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        z = np.random.uniform( size=n )
        hatching = np.array([ '/', '//', None ])[np.random.randint( 0, 3, n )]
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            plot_cells = True,
            colors = z,
            hatching = hatching,
        )

    ########################################################################

    def test_plot_label_box( self ):

        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            plot_label_box = True,
        )

    ########################################################################

    def test_logscale( self ):

        x = 10.**np.random.uniform( size=n )
        y = 10.**np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        fig = plt.figure()
        ax = plt.gca()

        ax.set_xscale( 'log' )
        ax.set_yscale( 'log' )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            ax = ax,
            labels = labels,
            plot_cells = True,
            plot_label_box = True,
        )

