"""Read-only native curve, diagnostic and artifact inspection."""
from html import escape
import json
import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np


def attach(app):
    app.curve_choice=W.Dropdown(description='Saved curve',layout=W.Layout(width='98%'))
    app.curve_output=W.Output();app.product_details=W.HTML()
    def draw(_=None):
        app.curve_output.outputs=()
        _,_,record=app._products()
        if record is None or not app.curve_choice.value:return
        curve=record.products.curves.get(app.curve_choice.value)
        if curve is None:return
        with plt.ioff():fig,ax=plt.subplots(figsize=(4,2.8),layout='constrained')
        ax.plot(np.asarray(curve.x),np.asarray(curve.y));ax.set_xlabel(curve.x_label);ax.set_ylabel(curve.y_label)
        ax.set_title(curve.display_name,fontsize=10,wrap=True);ax.set_yscale(curve.y_scale)
        if curve.series_labels:ax.legend(curve.series_labels)
        from IPython.display import display
        with app.curve_output:display(fig)
        plt.close(fig)
    app.curve_choice.observe(draw,names='value')
    def update():
        _,_,record=app._products()
        curves=[] if record is None else [(c.display_name,k) for k,c in record.products.curves.items()]
        old=app.curve_choice.value;app.curve_choice.options=[('No saved curve',''),*curves]
        if old in dict(curves).values():app.curve_choice.value=old
        diagnostics={} if record is None else {k:d.values for k,d in record.products.diagnostics.items()}
        artifacts={} if record is None else {k:dict(filename=a.filename,media_type=a.media_type,metadata=a.metadata)
                                            for k,a in record.products.artifacts.items()}
        app.product_details.value='<pre>'+escape(json.dumps(dict(diagnostics=diagnostics,artifacts=artifacts),indent=2,default=str))+'</pre>'
    app._show_extra_results=update
    return W.VBox((app.curve_choice,app.curve_output,app.product_details))
