"""Batch crop: python batch_tile_160.py (Pillow and numpy required)."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw


def axis(length, size, overlap):
    starts = list(range(0, max(0, length-size)+1, size-overlap))
    last = max(0, length-size)
    if starts[-1] != last:
        starts.append(last)
    ends = [min(s+size, length) for s in starts]
    edges = [0]+[(ends[i]+starts[i+1])//2 for i in range(len(starts)-1)]+[length]
    return [(s,e,edges[i],edges[i+1]) for i,(s,e) in enumerate(zip(starts,ends))]


def process(source, root, output, size, overlap, filtering):
    relative = source.relative_to(root).as_posix()
    key = source.stem+'_'+hashlib.sha256(relative.encode()).hexdigest()[:8]
    folder = output/key
    folder.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as raw:
        im = ImageOps.exif_transpose(raw).convert('RGB')
    w,h = im.size
    mw,mh = (w+7)//8,(h+7)//8
    small = im.resize((mw,mh), Image.Resampling.BOX)
    rgb = np.asarray(small).astype(np.float32)
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    # Broad blue-green leaf cue, not a trained broccoli/head detector.
    seed = (g > r*1.06) & (b > r*1.01) & (g > b*.90) & (g > 25)
    mask = Image.fromarray((seed*255).astype('uint8')).filter(ImageFilter.MaxFilter(17))
    if filtering == 'none':
        mask = Image.new('L',(mw,mh),255)
    xs,ys = axis(w,size,overlap),axis(h,size,overlap)
    records=[]
    overview=im.copy();overview.thumbnail((1200,800))
    draw=ImageDraw.Draw(overview);sx=overview.width/w;sy=overview.height/h
    for row,(y,y2,cy,cy2) in enumerate(ys):
        for col,(x,x2,cx,cx2) in enumerate(xs):
            mb=(x*mw//w,y*mh//h,min(mw,(x2*mw+w-1)//w),min(mh,(y2*mh+h-1)//h))
            keep=mask.crop(mb).getbbox() is not None
            if not keep:
                draw.rectangle((int(x*sx),int(y*sy),int((x2-1)*sx),int((y2-1)*sy)),outline='#ff5959',width=1)
                continue
            name=f'{source.stem}_r{row:03d}_c{col:03d}.png'
            crop=im.crop((x,y,x2,y2))
            if crop.size != (size,size):
                padded=Image.new('RGB',(size,size));padded.paste(crop,(0,0));crop=padded
            crop.save(folder/name,compress_level=1)
            records.append({'tile':name,'box_xyxy':[x,y,x2,y2],
                            'core_local_xyxy':[cx-x,cy-y,cx2-x,cy2-y],
                            'valid_size':[x2-x,y2-y]})
    manifest={'source':relative,'source_size':[w,h],'tile_size':size,'overlap':overlap,
              'filter':filtering,'coordinates':'EXIF-oriented pixels, right/bottom exclusive',
              'tiles':records}
    manifest_path=folder/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        retained={t['tile'] for t in records}
        for t in old['tiles']:
            if t['tile'] not in retained:
                target=(folder/t['tile']).resolve()
                if target.parent != folder.resolve() or target.suffix != '.png':
                    raise ValueError(f'Unsafe obsolete path: {target}')
                target.unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    overview.save(output/'overviews'/f'{key}.jpg',quality=90)
    return {'source':relative,'folder':key,'kept':len(records),'excluded':len(xs)*len(ys)-len(records),'source_size':[w,h]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--size',type=int,default=160)
    parser.add_argument('--overlap',type=int,default=40)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--filter',choices=['conservative','none'],default='conservative')
    args=parser.parse_args()
    if not 0 <= args.overlap < args.size or args.workers < 1:
        parser.error('Require size > overlap >= 0 and workers >= 1')
    root=args.input.resolve();output=(args.output or root/'annotation_tiles_160').resolve()
    if output==root:
        parser.error('Output must be a separate subdirectory')
    sources=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'} and not p.is_relative_to(output))
    if not sources:
        parser.error('No source images found')
    (output/'overviews').mkdir(parents=True,exist_ok=True)
    summary=[];errors=[];start=time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(process,p,root,output,args.size,args.overlap,args.filter):p for p in sources}
        for i,future in enumerate(as_completed(jobs),1):
            try:
                result=future.result();summary.append(result)
                print(f"[{i}/{len(sources)}] {result['source']}: {result['kept']} kept, {result['excluded']} excluded",flush=True)
            except Exception as exc:
                errors.append({'source':str(jobs[future]),'error':str(exc)})
                print(f'ERROR: {jobs[future]}: {exc}',flush=True)
    report={'tile_size':args.size,'overlap':args.overlap,'filter':args.filter,
            'images':sorted(summary,key=lambda v:v['source']),'errors':errors,
            'total_tiles':sum(v['kept'] for v in summary),'elapsed_seconds':round(time.time()-start,1)}
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('images','errors')}),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
