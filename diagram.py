import numpy as np
import plotly.express as px
import plotly.io as pio
import pandas as pd

pio.kaleido.scope.default_width = 2500
pio.kaleido.scope.default_height = 1000

df = pd.read_csv("./results.csv")

df['time'] = df['time'] * 1e-9

fig = px.line(
    df,
    title="Transformation Time Per Number of Threads",
    x="threads",
    y="time",
    labels={"time": "Duration (seconds)", "threads": "Number of Threads"},
    range_x=[2, 1024],
    range_y=[0, 30],
    facet_row_spacing=0.1
)
fig.update_layout(font_size=22)
fig.update_xaxes(tickvals=np.arange(2, 1025, 14))
fig.update_yaxes(tickvals=np.arange(0, 31, 2))
fig.write_image("./benchmark.png")
