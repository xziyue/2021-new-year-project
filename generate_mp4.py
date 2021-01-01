import os
import subprocess

this_dir, _ = os.path.split(os.path.abspath(__file__))
frame_dir = os.path.join(this_dir, 'frames')

os.chdir(frame_dir)


for i in range(144):
    in_filename = 'frame_{}.pdf'.format(i+1)
    out_filename = 'frame_{}.png'.format(i+1)
    subprocess.run(['convert', '-density', '500', in_filename, out_filename])


os.system('ffmpeg -i frame_%d.png -c:v libx264 -crf 1 -profile:v main -vf fps=24,format=yuv420p all.mp4')
