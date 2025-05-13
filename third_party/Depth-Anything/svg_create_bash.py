import os

video_path = '/mnt/sda/dp/SVG/Videos/'

datasets = ['lumiere', 'real', 'walt', 'zeroscope']
f = open('svg_depth_estimation.sh', 'w')
for dataset in datasets:
    dataset_root = '%s/%s/' % (video_path, dataset)
    folders = [x[0] for x in os.walk(dataset_root)]
    for folder in folders:
        if folder.endswith('320x576'):
            print(folder)
            depth_folder = folder + '/depths/'
            image_folder = folder + '/images/'
            f.write('python svg_run.py --encoder vitl --img-path %s --outdir %s\n' % (image_folder, depth_folder))
        else:
            continue
        
f.close()
