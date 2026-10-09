# Live loss tracking

Keep `output.stream_history: true` (the default). In a separate terminal:

```bash
python -m swe_pinn monitor --run-dir training_results/<case>/<run-id>
```

Open `http://127.0.0.1:8765` in your browser. The viewer refreshes every three
seconds and plots all recorded loss terms on a logarithmic axis. SWE time
segments have a selector. New points arrive at `training.report_period`;
monitoring does not change training or compute extra diagnostics.
Zero losses are omitted from the logarithmic plot. The viewer samples long
histories for display while keeping the saved journal intact.

For a cluster run, launch the viewer on a host that can read the run directory,
using that host's allowed interactive/job mechanism. Forward its local port:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@MONITOR_HOST
```

Then open the same local browser address. If the viewer is on a compute node,
use your cluster's SSH jump-host routing to that node. The viewer listens only
on localhost. Use `--port 8766` if the default port is occupied.
Stop the viewer with Ctrl+C; training continues independently.
