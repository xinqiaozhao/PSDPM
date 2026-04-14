def create_pascal_label_colormap():
    colormap = 255*np.ones((256,3),dtype=np.uint8)
    colormap[0] = [0,0,0]
