import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8

mpl.rcParams['font.family'] = ['Helvetica LT Std']
mpl.rcParams['font.sans-serif'] = ['Helvetica LT Std']
mpl.rcParams['text.usetex'] = False

mpl.rcParams['svg.fonttype'] = 'none'

mpl.rcParams['lines.markersize'] = 4

# colorcycle from http://jfly.iam.u-tokyo.ac.jp/color/#pallet as suggested by eLife
colorcycle = sp.array([[0, 0, 0], [230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178], [213, 94, 0], [204, 121, 167]]) / 255.0
colorcycle = map(lambda x: tuple(x), list(colorcycle))
mpl.rcParams['axes.color_cycle'] = colorcycle