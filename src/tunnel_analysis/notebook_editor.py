"""Optional notebook widgets. Processing and persistence live in editor.py."""
import builtins
from dataclasses import replace
import importlib.metadata
import math
import types

import numpy as np

from .editor_geometry import Boundaries, Ellipse


def _cells(function):
    return dict(zip(function.__code__.co_freevars, function.__closure__ or ()))


def _close_widgets(widget, seen=None):
    seen = set() if seen is None else seen
    if id(widget) in seen:
        return
    seen.add(id(widget))
    for child in getattr(widget, 'children', ()):
        _close_widgets(child, seen)
    widget.close()


class StackviewBrush:
    """Isolated adapter for stackview 0.19.1; never patches global modules.

    Reuse annotate's canvas, stroke interpolation, and brush controls. Override
    its unclipped disk write and capture pointer boundaries for precise undo.
    The pinned version's private closure interface is covered by event tests.
    """
    def __init__(self, crop, zoom=1, on_change=lambda: None):
        import stackview
        import ipywidgets as W
        from ipyevents import Event
        if importlib.metadata.version('stackview') != '0.19.1':
            raise RuntimeError('This adapter requires stackview==0.19.1')
        if zoom not in (1, 2, 4):
            raise ValueError('Zoom must be 1, 2 or 4 native display pixels per image pixel')
        if max(crop.labels.shape)*zoom > 1024:
            raise ValueError('For this zoom, choose a smaller crop; the display is limited to 1024 pixels per side')
        self.crop, self.events = crop, []
        adapter = self

        class BrushEvents(Event):
            def __init__(self, *args, **kwargs):
                kwargs['watched_events'] = ['mousedown', 'mousemove', 'mouseup', 'mouseleave']
                kwargs['prevent_default_action'] = True
                super().__init__(*args, **kwargs)
                adapter.events.append(self)

            def on_dom_event(self, callback, remove=False):
                self.original_callback = callback
                super().on_dom_event(self.handle, remove=remove)

            def handle(self, event):
                try:
                    crop.check()
                    kind = event['type']
                    if kind in ('mouseup', 'mouseleave') or not (event.get('buttons', 0) & 1):
                        self.original_callback({**event, 'buttons': 0})
                        crop.end()
                    else:
                        if kind == 'mousedown':
                            self.original_callback({**event, 'buttons': 0})
                        self.original_callback(event)
                    adapter.error = None
                except Exception as exc:
                    adapter.error = str(exc)
                on_change()

        def local_import(name, *args, **kwargs):
            if name == 'ipyevents':
                return types.SimpleNamespace(Event=BrushEvents)
            return builtins.__import__(name, *args, **kwargs)

        def draw(x, y, z, axis, radius, value, image):
            crop.paint(x, y, radius=radius, value=int(value))

        scope = dict(stackview.annotate.__globals__)
        scope.update(draw_circle=draw, __builtins__={**vars(builtins), '__import__': local_import})
        annotate = types.FunctionType(stackview.annotate.__code__, scope,
                                      argdefs=stackview.annotate.__defaults__)
        self.widget = annotate(crop.gray, crop.labels, zoom_factor=zoom, zoom_spline_order=0,
                               alpha=.5, display_min=0, display_max=1)
        cells = _cells(self.widget.update)
        self.view = cells['view'].cell_contents
        original_rgb = cells['_img_to_rgb'].cell_contents

        def rgb(array, **kwargs):
            if array.dtype in (np.uint32, np.uint64):
                gray = (np.clip(crop.gray, 0, 1)*255).astype(np.uint8)
                out = np.repeat(gray[..., None], 3, axis=2)
                out[array > 0] = (40, 220, 80)
                out[~crop.allowed] = (70, 130, 255)
                return out
            return original_rgb(array, **kwargs)

        cells['_img_to_rgb'].cell_contents = rgb
        self.alpha_cell = cells['alpha']
        draw_cells = _cells(self.events[0].original_callback)
        self.label = draw_cells['label_id_slider'].cell_contents
        self.radius = draw_cells['radius_slider'].cell_contents
        self.eraser_radius = draw_cells['radius_eraser_slider'].cell_contents
        # Hide multiclass controls; only binary editing is meaningful here.
        def hide_label_controls(widget):
            children = getattr(widget, 'children', ())
            if any(child is self.label for child in children):
                widget.layout.display = 'none'
            else:
                for child in children:
                    hide_label_controls(child)
        hide_label_controls(self.widget)
        self.error = None
        self.controls = W.HBox()
        self.mode = W.ToggleButtons(options=['Paint', 'Erase'], value='Paint')
        self.size = W.BoundedIntText(value=2, min=1, max=64, description='Radius (px)')
        self.visible = W.Checkbox(value=True, description='Show mask')
        self.controls.children = (self.mode, self.size, self.visible)
        def settings(_=None):
            crop.end()
            self.label.value = 1 if self.mode.value == 'Paint' else 0
            self.radius.value = self.eraser_radius.value = self.size.value
            self.alpha_cell.cell_contents = .5 if self.visible.value else 0.
            self.widget.update()
        for control in self.controls.children:
            control.observe(settings, names='value')
        settings()

    def close(self):
        for event in self.events:
            event.close()
        _close_widgets(self.widget)
        _close_widgets(self.controls)


class BoundaryEditor:
    """Three independent ellipses, using Matplotlib's stock interactive selector."""
    def __init__(self, session):
        import ipywidgets as W
        import matplotlib.pyplot as plt
        from matplotlib.widgets import EllipseSelector
        self.session = session
        self.values = {'arena': session.boundaries.arena, 'outer': session.boundaries.outer,
                       'inner': session.boundaries.inner or replace(session.boundaries.arena,
                           radius_x=session.boundaries.arena.radius_x*.2,
                           radius_y=session.boundaries.arena.radius_y*.2)}
        self.circle_constraints = {name: False for name in self.values}
        self.loading = False
        self.valid = True
        self.kind = W.Dropdown(options=[('Full arena (denominator)', 'arena'),
                                        ('Outer cutoff', 'outer'), ('Inner exclusion', 'inner')])
        self.enabled = W.Checkbox(value=session.boundaries.inner is not None, description='Exclude inner ellipse')
        self.circle = W.Checkbox(value=False, description='Constrain to circle')
        self.fields = {key: W.FloatText(description=label, layout=W.Layout(width='210px')) for key, label in
                       [('center_x', 'Centre x'), ('center_y', 'Centre y'), ('radius_x', 'Radius x'),
                        ('radius_y', 'Radius y'), ('angle', 'Angle (°)')]}
        self.message = W.HTML(value='Adjust all boundaries, then Confirm. Blue areas are excluded.')
        self.confirm = W.Button(description='Confirm boundaries', button_style='success')
        self.detail = W.Button(description='Inspect selected edge')
        self.overview = W.Button(description='Whole image')
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
        h, w = session.image.shape
        self.preview = session.image.preview(1100)
        self.image_artist = self.ax.imshow(self.preview, cmap='gray', vmin=0, vmax=1,
                       extent=(-.5, w-.5, h-.5, -.5), interpolation='nearest')
        self.ax.set(xlim=(-.5, w-.5), ylim=(h-.5, -.5), title='Native pixel coordinates')
        self.selector = EllipseSelector(self.ax, self._selected, interactive=True,
                                        useblit=False, drag_from_anywhere=True,
                                        props={'edgecolor': 'yellow', 'facecolor': 'none', 'linewidth': 1})
        self.artists = []
        self.detail_artist = None
        self.kind.observe(lambda _: self._load(), names='value')
        for field in self.fields.values():
            field.observe(self._typed, names='value')
        self.circle.observe(self._typed, names='value')
        self.enabled.observe(self._typed, names='value')
        self.confirm.on_click(self._confirm)
        self.detail.on_click(self._detail)
        self.overview.on_click(self._overview)
        self.widget = W.VBox([W.HTML(value='<b>1. Confirm arena and exclusions</b>'), self.kind,
                              W.HBox([self.enabled, self.circle]),
                              W.HBox(list(self.fields.values())[:3]), W.HBox(list(self.fields.values())[3:]),
                              W.HBox([self.detail, self.overview, self.confirm]), self.message, self.fig.canvas])
        self._load()

    def close(self):
        import matplotlib.pyplot as plt
        self.selector.disconnect_events()
        _close_widgets(self.widget)
        plt.close(self.fig)

    def _load(self):
        self.loading = True
        self.valid = True
        e = self.values[self.kind.value]
        self.circle.value = self.circle_constraints[self.kind.value]
        for key, field in self.fields.items():
            field.value = math.degrees(e.angle) if key == 'angle' else getattr(e, key)
        self.loading = False
        self._draw()

    def _typed(self, _=None):
        if self.loading:
            return
        if self.session.active_crop and self.session.active_crop.dirty:
            self.valid = False
            self.message.value = 'Apply or discard tunnel strokes before changing boundaries.'
            return
        self.session.confirmed = False
        try:
            self.circle_constraints[self.kind.value] = self.circle.value
            data = {k: w.value for k, w in self.fields.items()}
            data['angle'] = math.radians(data['angle'])
            if self.circle.value:
                data['radius_y'] = data['radius_x']
            self.values[self.kind.value] = Ellipse(**data)
            self.valid = True
            self.message.value = 'Boundary changes are not confirmed yet.'
            self._load()
        except ValueError as exc:
            self.valid = False
            self.message.value = str(exc)

    def _selected(self, press, release):
        if self.session.active_crop and self.session.active_crop.dirty:
            self.message.value = 'Apply or discard tunnel strokes before changing boundaries.'
            self._draw()
            return
        x0, x1, y0, y1 = self.selector.extents
        a, b = (x1-x0)/2, (y1-y0)/2
        if min(a, b) <= 0:
            return
        self.values[self.kind.value] = Ellipse((x0+x1)/2, (y0+y1)/2, a,
                                               a if self.circle.value else b,
                                               math.radians(self.selector.rotation))
        self.session.confirmed = False
        self._load()
        self.message.value = 'Boundary changes are not confirmed yet.'

    def _draw(self):
        from matplotlib.patches import Ellipse as Patch
        from .editor_geometry import ellipse_mask
        h,w = self.session.image.shape
        ph,pw = self.preview.shape
        yy,xx = np.ogrid[:ph,:pw]
        xx,yy = (xx+.5)*w/pw-.5,(yy+.5)*h/ph-.5
        allowed = ellipse_mask(self.values['arena'],xx,yy) & ellipse_mask(self.values['outer'],xx,yy)
        if self.enabled.value:
            allowed &= ~ellipse_mask(self.values['inner'],xx,yy)
        rgb = np.repeat(self.preview[...,None],3,axis=2)
        rgb[~allowed] = rgb[~allowed]*.65 + np.array([70,130,255])/255*.35
        self.image_artist.set_data(rgb)
        for artist in self.artists:
            artist.remove()
        self.artists.clear()
        for name, color in [('arena', 'white'), ('outer', '#71b5ff'), ('inner', '#71b5ff')]:
            if name == 'inner' and not self.enabled.value:
                continue
            e = self.values[name]
            patch = Patch((e.center_x, e.center_y), 2*e.radius_x, 2*e.radius_y,
                          angle=math.degrees(e.angle), fill=False, edgecolor=color, linewidth=1)
            self.ax.add_patch(patch); self.artists.append(patch)
        e = self.values[self.kind.value]
        self.selector.rotation = 0
        self.selector.extents = (e.center_x-e.radius_x, e.center_x+e.radius_x,
                                 e.center_y-e.radius_y, e.center_y+e.radius_y)
        self.selector.rotation = math.degrees(e.angle)
        self.fig.canvas.draw_idle()

    def _confirm(self, _):
        try:
            if not self.valid:
                raise ValueError('Correct invalid boundary coordinates before confirming')
            self.session.confirm_boundaries(Boundaries(self.values['arena'], self.values['outer'],
                                                       self.values['inner'] if self.enabled.value else None))
            self.message.value = 'Boundaries confirmed. Run the analysis cell next.'
        except ValueError as exc:
            self.message.value = str(exc)

    def _detail(self, _):
        if self.session.closed:
            self.message.value = 'Session closed; use the current boundary editor.'
            return
        # Magnify the selected point nearest the current axes centre, so pan
        # the overview to inspect any portion of an ellipse, then click this.
        e = self.values[self.kind.value]
        theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
        x = e.center_x + e.radius_x*np.cos(theta)*np.cos(e.angle)-e.radius_y*np.sin(theta)*np.sin(e.angle)
        y = e.center_y + e.radius_x*np.cos(theta)*np.sin(e.angle)+e.radius_y*np.sin(theta)*np.cos(e.angle)
        j = np.argmin((x-np.mean(self.ax.get_xlim()))**2+(y-np.mean(self.ax.get_ylim()))**2)
        h, w = self.session.image.shape
        x0 = int(np.clip(round(x[j])-256, 0, max(0, w-512)))
        y0 = int(np.clip(round(y[j])-256, 0, max(0, h-512)))
        x1, y1 = min(w, x0+512), min(h, y0+512)
        if self.detail_artist is not None:
            self.detail_artist.remove()
        self.detail_artist = self.ax.imshow(self.session.image.gray(slice(y0,y1), slice(x0,x1)),
            extent=(x0-.5,x1-.5,y1-.5,y0-.5), cmap='gray', vmin=0, vmax=1, interpolation='nearest', zorder=.5)
        self.ax.set(xlim=(x0-.5,x1-.5), ylim=(y1-.5,y0-.5))
        self.fig.canvas.draw_idle()

    def _overview(self, _):
        if self.detail_artist is not None:
            self.detail_artist.remove(); self.detail_artist = None
        h, w = self.session.image.shape
        self.ax.set(xlim=(-.5,w-.5), ylim=(h-.5,-.5))
        self.fig.canvas.draw_idle()


class MaskEditor:
    def __init__(self, session, results_parent, reviewer=''):
        import ipywidgets as W
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector
        from .report import overlay, small_mask
        from .io import resize_mask
        if not session.ready:
            raise ValueError('Run automatic analysis before opening the editor')
        self.session, self.brush = session, None
        self.results_parent, self.reviewer = results_parent, reviewer
        self.x = W.BoundedIntText(value=0, min=0, max=session.image.shape[1]-1, description='Crop x')
        self.y = W.BoundedIntText(value=0, min=0, max=session.image.shape[0]-1, description='Crop y')
        self.size = W.Dropdown(options=[256,512,1024], value=512, description='Crop size')
        self.zoom = W.Dropdown(options=[1,2,4], value=1, description='Display zoom')
        self.message = W.HTML(value='Select a region in the overview, then Open crop. Radius 1 edits one pixel.')
        self.holder = W.VBox()
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.preview = session.image.preview(900)
        self.overview_image = self.ax.imshow(overlay(self.preview,
            small_mask(session.arrays['corrected'], self.preview.shape),
            small_mask(session.arrays['candidates'], self.preview.shape),
            resize_mask(session.arrays['permitted'], self.preview.shape)),
            extent=(-.5,session.image.shape[1]-.5,session.image.shape[0]-.5,-.5), interpolation='nearest')
        def select(a, b):
            if a.xdata is not None and b.xdata is not None:
                self.x.value = int(np.clip(round(min(a.xdata,b.xdata)),0,self.x.max))
                self.y.value = int(np.clip(round(min(a.ydata,b.ydata)),0,self.y.max))
        self.selector = RectangleSelector(self.ax, select, interactive=True, useblit=False)
        actions = [('Open crop', self.open_crop), ('Undo stroke', self.undo), ('Apply to Drive', self.apply),
                   ('Discard strokes', self.discard), ('Save reviewed result', self.save)]
        buttons = []
        for label, function in actions:
            button = W.Button(description=label)
            button.on_click(self._guard(function)); buttons.append(button)
        self.widget = W.VBox([W.HTML(value='<b>2. Review tunnels</b>'), self.fig.canvas,
            W.HBox([self.x,self.y,self.size,self.zoom]), W.HBox(buttons), self.message, self.holder])

    def close(self):
        import matplotlib.pyplot as plt
        if self.brush:
            self.brush.close()
        self.selector.disconnect_events()
        _close_widgets(self.widget)
        plt.close(self.fig)

    def _guard(self, callback):
        def call(_):
            try:
                callback()
            except Exception as exc:
                self.message.value = f'Action failed: {exc}. Applied checkpoints remain saved.'
        return call

    def _status(self):
        if self.brush:
            self.message.value = (self.brush.error or ('Unsaved strokes — Apply or Discard before switching crops.'
                                  if self.brush.crop.dirty else 'Crop matches the saved draft.'))

    def open_crop(self):
        crop = self.session.crop(self.x.value, self.y.value, self.size.value)
        if self.brush:
            self.brush.close()
        self.brush = StackviewBrush(crop, self.zoom.value, self._status)
        self.holder.children = (self.brush.controls, self.brush.widget)
        self._status()

    def undo(self):
        if self.brush:
            self.brush.crop.undo(); self.brush.widget.update(); self._status()

    def discard(self):
        if self.brush:
            self.brush.crop.discard(); self.brush.widget.update(); self._status()

    def apply(self):
        if self.brush:
            self.brush.crop.apply()
            # Update only the affected overview rectangle; no full-mask redraw.
            x0,y0,x1,y1 = self.brush.crop.bbox
            h,w = self.session.image.shape
            ph,pw = self.preview.shape
            a,b = max(0,int(y0*ph/h)-1), min(ph,int(np.ceil(y1*ph/h))+1)
            c,d = max(0,int(x0*pw/w)-1), min(pw,int(np.ceil(x1*pw/w))+1)
            yy = np.clip(((np.arange(a,b)+.5)*h/ph).astype(int),0,h-1)
            xx = np.clip(((np.arange(c,d)+.5)*w/pw).astype(int),0,w-1)
            from .report import overlay
            small = np.array(self.overview_image.get_array())
            ix = np.ix_(yy,xx)
            small[a:b,c:d] = overlay(self.preview[a:b,c:d], self.session.arrays['corrected'][ix]>0,
                                    self.session.arrays['candidates'][ix]>0, self.session.arrays['permitted'][ix]>0)
            self.overview_image.set_data(small); self.fig.canvas.draw_idle()
            self.message.value = 'Applied edits verified on Drive. You can open another crop.'

    def save(self):
        path = self.session.save_reviewed(self.results_parent, self.reviewer)
        self.message.value = f'Reviewed result saved: {path}'


def brush_demo(image):
    """Cheap Colab feasibility gate using a real native crop; never saves data."""
    import ipywidgets as W
    from .editor import Crop
    from types import SimpleNamespace
    h,w = image.shape
    y,x = max(0,h//2-128),max(0,w//2-128)
    gray = image.gray(slice(y,min(y+256,h)),slice(x,min(x+256,w)))
    shape = gray.shape
    demo = SimpleNamespace(ready=True, arrays={'permitted':np.ones(shape,np.uint8),
                                               'corrected':(gray>.2).astype(np.uint8)},
                           image=SimpleNamespace(gray=lambda *args:gray))
    crop = Crop(demo,0,0,shape[1],shape[0])
    message = W.HTML(value='Demo only. Try paint, erase and radius 1. These changes are never saved.')
    brush = StackviewBrush(crop,zoom=2,on_change=lambda:setattr(message,'value',
        f'Demo only: {np.count_nonzero(crop.labels != crop.original)} pixels changed; {brush.error or "no callback error"}'))
    undo = W.Button(description='Undo demo stroke')
    def undo_demo(_):
        crop.undo();brush.widget.update()
        message.value = f'Demo only: {np.count_nonzero(crop.labels != crop.original)} pixels changed.'
    undo.on_click(undo_demo)
    return W.VBox([message,brush.controls,undo,brush.widget]), brush
