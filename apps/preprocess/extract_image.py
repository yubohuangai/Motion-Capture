'''
  @ Date: 2021-08-19 22:06:13
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-12 13:39:25
  @ FilePath: /EasyMocapPublic/apps/preprocess/extract_image.py
'''
# extract image from videos
import os
from os.path import join
from glob import glob

extensions = ['.mp4', '.webm', '.flv', '.MP4', '.MOV', '.mov', '.avi']


def run(cmd):
    print(cmd)
    os.system(cmd)


def extract_images(path, ffmpeg, image):
    videos = sorted(sum([
        glob(join(path, 'videos', '*'+ext)) for ext in extensions
        ], [])
    )
    for videoname in videos:
        sub = '.'.join(os.path.basename(videoname).split('.')[:-1])
        sub = sub.replace(args.strip, '')
        outpath = join(path, image, sub)
        if os.path.exists(outpath) and (len(os.listdir(outpath)) > 10 or len(os.listdir(outpath)) == args.num) and not args.restart:
            continue
        os.makedirs(outpath, exist_ok=True)
        other_cmd = ''
        vf_filters = []

        if args.start > 0 or args.end != -1:
            if args.end != -1:
                vf_filters.append(f"select='between(n,{args.start},{args.end})'")
            else:
                vf_filters.append(f"select='gte(n,{args.start})'")
        if args.scale != 1:
            vf_filters.append(f"scale=iw/{args.scale}:ih/{args.scale}")
        if args.transpose != -1:
            vf_filters.append(f"transpose={args.transpose}")

        if vf_filters:
            other_cmd += ' -vf "{}"'.format(','.join(vf_filters))

        cmd = '{} -i {} {} -vsync 0 -q:v 1 -start_number 0 {}/%06d.jpg'.format(
            ffmpeg, videoname, other_cmd, outpath
        )
        if not args.debug:
            cmd += ' -loglevel quiet'
        run(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--strip', type=str, default='')
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--start', type=int, default=0, help='start frame index')
    parser.add_argument('--end', type=int, default=-1, help='end frame index (inclusive)')
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--transpose', type=int, default=-1)
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    extract_images(args.path, args.ffmpeg, args.image)