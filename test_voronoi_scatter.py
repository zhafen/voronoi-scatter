import numpy as np
import os
import unittest

import matplotlib
import matplotlib.pyplot as plt

import voronoi_scatter

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

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

        n = 20
        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )

        ax, vor = voronoi_scatter.scatter(
            x,
            y,
        )

    ########################################################################

    def test_labels( self ):

        n = 20
        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
        )

    ########################################################################

    def test_label_kwargs( self ):

        n = 20
        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            label_kwargs = { 'fontsize': 5, }
        )

    ########################################################################
    def test_offset_magnitudes( self ):

        n = 20
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

        n = 20
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

        n = 20
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

        n = 20
        x = np.random.uniform( size=n )
        y = np.random.uniform( size=n )
        labels = np.arange( n ).astype( str )

        ax, vor = voronoi_scatter.scatter(
            x, y,
            labels = labels,
            plot_cells = True,
        )

    ########################################################################

    def test_cell_colors( self ):

        n = 20
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

    def test_cell_colors_norm( self ):

        n = 20
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

        n = 20
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