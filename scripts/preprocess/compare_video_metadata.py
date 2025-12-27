import subprocess, json

def probe(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate",
        "-show_entries",
        "format=duration,bit_rate",
        "-of", "json",
        path
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    return json.loads(out)

# v1 = probe("/Users/yubo/data/s2/0/output_1.mkv")
v2 = probe("/Users/yubo/data/s2/seq1/01.mp4")

# print("MKV:", v1)
print("MP4:", v2)