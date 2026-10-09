"""Read-only browser viewer for loss journals written during training."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path


def loss_records(path, max_points=2000):
    """Ignore incomplete writes and limit data sent to the browser."""
    if max_points < 2:
        raise ValueError("max_points must be at least two")
    rows = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue
                try:
                    record = json.loads(line)
                    epoch = record["epoch"]
                    if not isinstance(epoch, (int, float)) or not math.isfinite(epoch):
                        continue
                    values = {key: value for key, value in record.items()
                              if key.startswith("loss_") and isinstance(value, (int, float))
                              and math.isfinite(value)}
                    if values:
                        rows.append(dict(epoch=epoch, **values))
                except (ValueError, KeyError, TypeError):
                    continue
    except (FileNotFoundError, PermissionError):
        return []
    if len(rows) <= max_points:
        return rows
    # Preserve the first and latest report while sampling the full epoch range.
    return [rows[round(index * (len(rows) - 1) / (max_points - 1))]
            for index in range(max_points)]


def loss_payload(run):
    run = Path(run)
    directories = sorted(path for path in (run / "segments").glob("*") if path.is_dir())
    if not directories:
        directories = [run]
    return {"run": run.name, "series": [
        {"label": directory.name if directory != run else "Training",
         "records": loss_records(directory / "history.jsonl")}
        for directory in directories]}


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PINN training losses</title>
<style>
body{font:16px system-ui;background:#fafafa;color:#222;max-width:1100px;margin:32px auto;padding:0 20px}
canvas{width:100%;height:450px;background:white;border:1px solid #ddd}select{padding:6px}
#legend{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}#status{color:#555}
</style>
<h1>Training losses</h1><p id="run"></p><label>Time segment <select id="segment"></select></label>
<div id="legend"></div><canvas id="plot"></canvas><p id="status">Waiting for loss reports…</p>
<p>Vertical axis: logarithmic loss. Horizontal axis: completed epochs.
Updates follow training's report period; zero losses are omitted from the log plot.</p>
<script>
const canvas=document.querySelector('#plot'), select=document.querySelector('#segment');
const colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666'];
let data={series:[]};
function draw(){
 const series=data.series.find(s=>s.label===select.value),rows=series?.records||[];
 const keys=[...new Set(rows.flatMap(row=>Object.keys(row).filter(k=>k.startsWith('loss_'))))].sort();
 const legend=document.querySelector('#legend');legend.replaceChildren();
 keys.forEach((key,i)=>{let item=document.createElement('span');item.style.color=colors[i%colors.length];
 item.textContent=key.replaceAll('_',' ');legend.append(item)});
 const width=canvas.clientWidth,height=450,ratio=window.devicePixelRatio||1;
 canvas.width=width*ratio;canvas.height=height*ratio;let ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
 ctx.clearRect(0,0,width,height);ctx.font='12px system-ui';
 const vals=rows.flatMap(row=>keys.map(key=>row[key]).filter(v=>v>0));
 if(!vals.length){ctx.fillText('Waiting for positive loss reports…',70,70);return;}
 let low=Infinity,high=-Infinity;for(const value of vals){const log=Math.log10(value);low=Math.min(low,log);high=Math.max(high,log)}
 low=Math.floor(low);high=Math.ceil(high);if(low===high)high++;
 const xmin=rows[0].epoch,xmax=Math.max(xmin+1,rows.at(-1).epoch),left=85,right=width-20,top=25,bottom=395;
 const x=e=>left+(e-xmin)/(xmax-xmin)*(right-left),y=v=>bottom-(Math.log10(v)-low)/(high-low)*(bottom-top);
 ctx.strokeStyle='#ddd';ctx.fillStyle='#444';
 const step=Math.max(1,Math.ceil((high-low)/8));
 for(let tick=low;tick<=high;tick+=step){let py=y(10**tick);ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(right,py);ctx.stroke();ctx.fillText('10^'+tick,12,py+4)}
 for(let i=0;i<=5;i++){let epoch=xmin+(xmax-xmin)*i/5;ctx.fillText(Math.round(epoch),x(epoch)-12,bottom+23)}
 ctx.fillText('Epoch',(left+right)/2,bottom+45);
 keys.forEach((key,i)=>{ctx.strokeStyle=colors[i%colors.length];ctx.lineWidth=2;ctx.beginPath();let started=false;
 for(const row of rows){if(!(row[key]>0)){started=false;continue}const px=x(row.epoch),py=y(row[key]);
 if(started)ctx.lineTo(px,py);else ctx.moveTo(px,py);started=true;}ctx.stroke();
 const last=rows.at(-1);if(last[key]>0){ctx.beginPath();ctx.arc(x(last.epoch),y(last[key]),3,0,2*Math.PI);ctx.fillStyle=colors[i%colors.length];ctx.fill()}});
}
async function refresh(){try{
 const response=await fetch('/losses',{cache:'no-store'});if(!response.ok)throw Error('Could not read losses');
 data=await response.json();document.querySelector('#run').textContent=data.run;
 const previous=select.value;select.replaceChildren();for(const series of data.series){let option=document.createElement('option');option.value=option.textContent=series.label;select.append(option)}
 if(data.series.some(s=>s.label===previous))select.value=previous;
 draw();const rows=data.series.find(s=>s.label===select.value)?.records||[];
 document.querySelector('#status').textContent=(rows.length?'Latest reported epoch: '+rows.at(-1).epoch:'Waiting for loss reports. Keep output.stream_history enabled.')+' · Refreshed '+new Date().toLocaleTimeString();
 }catch(error){document.querySelector('#status').textContent='Connection unavailable; retrying…'}
 setTimeout(refresh,3000);
}
select.addEventListener('change',draw);window.addEventListener('resize',draw);refresh();
</script></html>'''


def serve_losses(run, port=8765):
    run = Path(run).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run}")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body, kind = PAGE.encode(), "text/html; charset=utf-8"
            elif self.path == "/losses":
                body, kind = json.dumps(loss_payload(run), allow_nan=False).encode(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Live losses: http://127.0.0.1:{port}/ (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
