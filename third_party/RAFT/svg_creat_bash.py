import os, glob

dataset = 'real'
dataset_root = '/mnt/sda/dp/SVG/Videos/%s/' % dataset
subsample = 1

folders = [x[0] for x in os.walk(dataset_root)]
f = open('%s.sh' % dataset, 'w')

for folder in folders:
    if folder.endswith('320x576'):
        print(folder)
        if subsample > 1:
            depth_folder = folder + '/subsample%d/depths/' % subsample
            image_folder = folder + '/subsample%d/images/' % subsample
        else:
            depth_folder = folder + '/depths/'
            image_folder = folder + '/images/'

        out_dir = depth_folder.replace('/depths', '/depths_refined')
    else:
        continue
    
    f.write('python svg_enhance_depth.py --path=%s  --depth_path=%s --out_dir=%s\n' % (image_folder, depth_folder, out_dir))

f.close()

        
