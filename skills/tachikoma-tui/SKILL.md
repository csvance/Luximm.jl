---
name: tachikoma-tui
description: >-
  Bootstrap guide for building a Tachikoma.jl TUI app. Load when asked to create,
  scaffold, or extend a terminal UI in Julia using Tachikoma, including any work
  that touches Model/update!/view, the app() loop, layouts, widgets, or KeyEvent
  / MouseEvent handling. Covers the minimal app skeleton and the sharp edges
  that cost the most round trips.
---

# Tachikoma.jl App Scaffolding

A reference for building a TUI in Tachikoma.jl. Covers the protocol every app implements, the layout primitives, the small set of widgets you reach for first, and the sharp edges that cost time when you hit them blind.

For anything not covered here, the canonical references are the published docs at [kahliburke.github.io/Tachikoma.jl/dev](https://kahliburke.github.io/Tachikoma.jl/dev/) and the bundled demos in the `Tachikoma.jl` repo. Treat this skill as the runway, not the manual.

## 1. Minimum viable app

Every Tachikoma app does three things: define a mutable `Model`, implement `update!` to react to events, and implement `view` to draw each frame. `app(m)` runs the loop.

```julia
using Tachikoma
@tachikoma_app

@kwdef mutable struct Counter <: Model
    quit::Bool = false
    tick::Int = 0
    n::Int = 0
end

should_quit(m::Counter) = m.quit

function update!(m::Counter, evt::KeyEvent)
    evt.key == :char && evt.char == 'q' && (m.quit = true)
    evt.key == :escape                  && (m.quit = true)
    evt.key == :char && evt.char == '+' && (m.n += 1)
    evt.key == :char && evt.char == '-' && (m.n -= 1)
end

function view(m::Counter, f::Frame)
    m.tick += 1
    inner = render(Block(title="Counter"), f.area, f.buffer)
    set_string!(f.buffer, inner.x + 1, inner.y + 1,
        "n = $(m.n)   [+/-]change [q]uit", tstyle(:primary))
end

app(Counter())
```

Three things to notice:

- The struct **must** be `mutable` and subtype `Model`. `update!` mutates it in place.
- `@tachikoma_app` brings the callback names (`view`, `update!`, `should_quit`, `init!`, `cleanup!`) into your namespace so your method definitions extend Tachikoma's, instead of shadowing them in a fresh namespace.
- `app(m)` enters the alt screen, raw mode, and mouse mode, then runs the 60fps loop. Ctrl+C exits regardless of what `update!` does.

## 2. The protocol

All callbacks dispatch on your model type. Only `view` and `update!` (or `should_quit` flag) are typically needed; the rest exist for less common cases.

| Callback                                | Required | Purpose                                                  |
|:----------------------------------------|:---------|:---------------------------------------------------------|
| `view(m, f::Frame)`                     | yes      | Draw into `f.buffer` within `f.area`. Called each frame. |
| `update!(m, e::Event)`                  | yes      | React to input. Dispatch on `KeyEvent`, `MouseEvent`, `TaskEvent`. |
| `should_quit(m)`                        | yes      | Return `true` to exit the loop cleanly.                  |
| `init!(m, t::Terminal)`                 | no       | One-shot setup; terminal size is known here.             |
| `cleanup!(m)`                           | no       | Teardown after the loop exits (terminal already restored). |
| `pre_render!(m)` / `post_render!(m)`    | no       | Hooks immediately before/after `view`.                   |
| `task_queue(m)`                         | no       | Return a `TaskQueue` to enable `spawn_task!` / `TaskEvent`. |
| `recording_enabled(m)`                  | no       | Return `false` to disable Ctrl+R recording for this app. |

See [architecture](https://kahliburke.github.io/Tachikoma.jl/dev/architecture) for the full lifecycle.

## 3. Events

`update!` is dispatched on three concrete event types. Define one method per type you care about.

**`KeyEvent`** — fields `key::Symbol`, `char::Char`, `action::KeyAction`.

Common `key` values: `:char`, `:escape`, `:enter`, `:backspace`, `:tab`, `:backtab`, `:up`, `:down`, `:left`, `:right`, `:home`, `:end_key`, `:pageup`, `:pagedown`, `:delete`, `:insert`, `:f1` through `:f12`, `:ctrl`, `:ctrl_c`. For printable characters, `key == :char` and `char` holds the value. Match.jl is idiomatic for non-trivial dispatch:

```julia
using Match
function update!(m::MyApp, e::KeyEvent)
    @match (e.key, e.char) begin
        (:char, 'q') || (:escape, _) => (m.quit = true)
        (:char, 'r') || (:enter, _)  => reset!(m)
        _                            => nothing
    end
end
```

**`MouseEvent`** — fields `x::Int`, `y::Int` (1-based screen coords), `button`, `action`, plus `shift`/`alt`/`ctrl` flags. Buttons include `mouse_left`, `mouse_middle`, `mouse_right`, `mouse_scroll_up`, `mouse_scroll_down`. Actions are `mouse_press`, `mouse_release`, `mouse_drag`, `mouse_move`. Store widget rects on the model and use `contains(rect, e.x, e.y)` for hit testing.

**`TaskEvent`** — fields `id::Symbol`, `value::Any`. Delivered when background work spawned via `spawn_task!`/`spawn_timer!` completes. Requires a `task_queue(m)` method. See the [async docs](https://kahliburke.github.io/Tachikoma.jl/dev/async).

## 4. Layout

Layouts are constraint-based and explicit. You divide a `Rect` into child rects using a vector of constraints.

| Constraint     | Meaning                                          |
|:---------------|:-------------------------------------------------|
| `Fixed(n)`     | Exactly `n` cells.                               |
| `Min(n)`       | At least `n` cells; grows if room remains.       |
| `Max(n)`       | Up to `n` cells.                                 |
| `Percent(p)`   | `p`% of the parent rect (integer 0-100).         |
| `Fill(w=1)`    | Takes remaining space; `w` is the relative weight. |
| `Ratio(a, b)`  | `a/b` of the parent rect.                        |

```julia
rows = split_layout(
    Layout(Vertical, [Fixed(1), Fixed(5), Fill(), Fixed(1)]),
    f.area)
length(rows) < 4 && return   # terminal may be too small

cols = split_layout(Layout(Horizontal, [Percent(30), Fill()]), rows[3])
```

Layouts nest freely: split horizontally, then split each column vertically. Use `Layout(Vertical, [...]; spacing=1)` to leave a row between children.

Geometry helpers worth knowing (all exported): `inner(r)` shrinks by 1 on every side, `center(parent, w, h)` returns a `Rect` of size `w×h` centered in `parent`, `shrink(r, n)` applies a uniform margin, `bottom(r)` and `right(r)` give the last row/column, `anchor(parent, w, h; h=:center, v=:center)` for corner placement.

## 5. Rendering primitives

Two ways to put characters on screen, both writing to `f.buffer`:

```julia
set_char!(buf, x, y, '█', tstyle(:primary))
set_string!(buf, x, y, "hello", tstyle(:accent))
```

Coordinates are **1-based**: `set_char!(buf, 1, 1, ...)` writes the top-left cell. `f.area.x` and `f.area.y` are also 1-based.

For anything more complex, call a widget's `render`:

```julia
inner = render(Block(title="Stats"), f.area, buf)
# `inner` is the rect inside the border; draw widgets into it
render(Gauge(0.42; filled_style=tstyle(:primary)), inner, buf)
```

`render(widget, rect, buf)` returns the **inner rect** the widget consumed. For container widgets like `Block`, that's the space inside the border; for content widgets, it's usually the input rect unchanged. Construct stateless widgets fresh each frame with current data. Persist stateful widgets (see Section 7) on the model.

## 6. Styling

Use `tstyle(name; bold=false, dim=false, italic=false, underline=false)` keyed by a theme slot:

```julia
tstyle(:primary, bold=true)
tstyle(:text_dim)
tstyle(:error)
```

Available slots include `:primary`, `:accent`, `:success`, `:warning`, `:error`, `:border`, `:title`, `:text`, `:text_dim`, `:background`. The active theme controls the actual colors, so an app written with `tstyle` automatically adapts to all 24 bundled themes and to the user's Ctrl+\ theme switch. Avoid hard-coded `Color256`/`ColorRGB` unless drawing pure-aesthetic content.

## 7. The widgets you'll reach for first

Tachikoma ships 30+ widgets. Five cover most app scaffolds; for the rest see the [widgets reference](https://kahliburke.github.io/Tachikoma.jl/dev/widgets).

**`Block(title=..., border_style=...)`** — bordered container. Returns the inner `Rect`.
```julia
inner = render(Block(title="Logs", border_style=tstyle(:border)), area, buf)
```

**`StatusBar(left=[Span(...)], right=[Span(...)])`** — single-row strip with left/right-aligned spans. Render into a `Fixed(1)` row, typically `rows[end]` or pinned to `bottom(f.area)`.
```julia
render(StatusBar(
    left=[Span("  [r]oll  ", tstyle(:accent))],
    right=[Span("[q]uit ", tstyle(:text_dim))],
), Rect(f.area.x, bottom(f.area), f.area.width, 1), buf)
```

**`Gauge(ratio; filled_style, empty_style, label="")`** — horizontal progress bar; `ratio` is `0.0..1.0`. Reconstruct each frame.

**`SelectableList(items; selected=1, focused=true)`** — keyboard-navigable list. **Stateful**: store on the model. Use `handle_key!(list, evt)` to delegate arrow keys; read `value(list)` for the current item.

**`Form([FormField("Name", TextInput(...); required=true), ...]; tick)`** — labeled fields, Tab/Shift-Tab navigation, submit button, validation. **Stateful**: store on the model. Sync `m.form.tick = m.tick` each frame so cursors blink. `handle_key!(m.form, evt)` drives navigation; `valid(m.form)` and `value(m.form)` read the result.

For a fuller catalog: `MarkdownPane`, `DataTable`, `Chart`, `BarChart`, `Sparkline`, `Calendar`, `Modal`, `TreeView`, `Checkbox`, `RadioGroup`, `DropDown`, `Button`, `TabBar`, `ScrollPane`, `Paragraph`, `BigText`, `CodeEditor`, `Canvas`, `BlockCanvas`, `PixelImage`, `FloatingWindow`, `WindowManager`, `TerminalWidget`, `REPLWidget`, `Separator`.

## 8. Default key bindings

`app(m)` installs these automatically. Pass `default_bindings=false` to disable when an app needs the Ctrl+letter range for its own use.

| Key      | Action                                 |
|:---------|:---------------------------------------|
| `Ctrl+C` | Quit (always active, can't be disabled) |
| `Ctrl+G` | Toggle mouse mode                      |
| `Ctrl+\` | Open theme selector                    |
| `Ctrl+A` | Toggle animations                      |
| `Ctrl+S` | Settings overlay                       |
| `Ctrl+?` | Help overlay                           |
| `Ctrl+R` | Start/stop `.tach` recording           |
| `Ctrl+Y` | Copy pane to clipboard                 |

## 9. Sharp edges

These are the mistakes that cost the most round trips. Most are not visible until they bite.

- **Struct must be `mutable`.** `update!` mutates the model in place. An immutable struct fails with `setfield! immutable struct` the first time a key arrives.
- **`@tachikoma_app` (or explicit `import Tachikoma: view, update!, should_quit`) is required.** Without it, `function view(::MyModel, ...)` defines a fresh `view` in your module instead of extending Tachikoma's, and the loop never sees it. Symptom: blank screen, no errors.
- **Reconstruct stateless widgets each frame, persist stateful ones.** `Block`, `Gauge`, `Sparkline`, `BarChart`, `StatusBar`, `Paragraph`, `BigText`, `Modal`, `Separator` are stateless: build them fresh in `view`. `TextInput`, `TextArea`, `CodeEditor`, `DataTable`, `Form`, `SelectableList`, `TreeView`, `DropDown`, `Checkbox`, `RadioGroup`, `Button`, `ScrollPane`, `REPLWidget`, `TerminalWidget` carry internal state (cursor, scroll, selection, focus); rebuilding them each frame resets it. Store these on the model.
- **Coordinates are 1-based.** `set_char!(buf, 1, 1, ...)` is the top-left. So are `f.area.x` and `f.area.y`. Off-by-one bugs at the top edge usually trace to assuming 0-based.
- **`split_layout` returns fewer rects than constraints if the area is too small.** Always guard: `length(rows) < N && return` before indexing. This happens routinely at app startup when the terminal hasn't reported a size yet, and when the user shrinks the window.
- **Stateful widgets carrying a `tick` need it synced before render.** Pattern: `m.form.tick = m.tick` (and same for any list/button with cursor animations) at the top of `view`. Without it, cursors and spinners freeze on the first value.
- **`println` and `print` from inside `view`/`update!` corrupt the screen.** They write directly to the terminal while Tachikoma is composing the frame. Use `@info`/`@debug`/`@warn` (Tachikoma redirects logging through the recording channel) or wire `on_stdout` / `on_stderr` callbacks via `app(m; on_stdout=..., on_stderr=...)`.
- **Don't `Pkg.activate` inside `init!`.** Activate the project before launching the app. Activating during `init!` causes Pkg side effects to compete with the render loop for the terminal.
- **Wrap `app(m)` in a function.** A bare `app(m)` at top level blocks the REPL with no way to re-enter cleanly after exit. Standard pattern: `function myapp(); app(MyModel()); end`, then call `myapp()`.
- **`TerminalWidget` and `REPLWidget` own a subprocess or PTY.** Close them in `cleanup!(m)` (call `close!(w)` or `pty_close!`) or wrap your `app(...)` call in `try/finally`. Otherwise the child process leaks past app exit.
- **`Frame.area` may not equal terminal size.** It's the area the framework gave this `view` call; on resize events you'll see a different rect mid-frame. Always layout from `f.area`, never from a cached terminal size.

## 10. Where to look next

Reference docs (published at [kahliburke.github.io/Tachikoma.jl/dev](https://kahliburke.github.io/Tachikoma.jl/dev/)):

| Page                                                                               | Topic                                          |
|:-----------------------------------------------------------------------------------|:-----------------------------------------------|
| [getting-started](https://kahliburke.github.io/Tachikoma.jl/dev/getting-started)   | Build-your-first-app walkthrough               |
| [architecture](https://kahliburke.github.io/Tachikoma.jl/dev/architecture)         | Full lifecycle, `Frame`/`Buffer` internals     |
| [layout](https://kahliburke.github.io/Tachikoma.jl/dev/layout)                     | Constraint semantics, `ResizableLayout`        |
| [widgets](https://kahliburke.github.io/Tachikoma.jl/dev/widgets)                   | Every widget with constructor signatures       |
| [events](https://kahliburke.github.io/Tachikoma.jl/dev/events)                     | Event types and dispatch in depth              |
| [styling](https://kahliburke.github.io/Tachikoma.jl/dev/styling)                   | Themes, palettes, color helpers                |
| [async](https://kahliburke.github.io/Tachikoma.jl/dev/async)                       | `TaskQueue`, `spawn_task!`, `spawn_timer!`     |
| [animation](https://kahliburke.github.io/Tachikoma.jl/dev/animation)               | Tweens, springs, organic effects               |
| [canvas](https://kahliburke.github.io/Tachikoma.jl/dev/canvas)                     | Braille, block, and pixel rendering            |
| [testing](https://kahliburke.github.io/Tachikoma.jl/dev/testing)                   | `TestBackend` for headless widget tests        |
| [recording](https://kahliburke.github.io/Tachikoma.jl/dev/recording)               | `.tach` format, `record_app`, SVG/GIF export   |

Seed templates in the [Tachikoma.jl repo](https://github.com/kahliburke/Tachikoma.jl), under `demos/TachikomaDemos/src/`:

- [`clock.jl`](https://github.com/kahliburke/Tachikoma.jl/blob/main/demos/TachikomaDemos/src/clock.jl) — minimal animation loop with `BigText`, `Calendar`, `StatusBar`. The cleanest "shape of a view" reference.
- [`form_demo.jl`](https://github.com/kahliburke/Tachikoma.jl/blob/main/demos/TachikomaDemos/src/form_demo.jl) — `Form` with validators, `FocusRing`, submit detection.
- [`dashboard.jl`](https://github.com/kahliburke/Tachikoma.jl/blob/main/demos/TachikomaDemos/src/dashboard.jl) — multi-widget layout, simulated data, nested `split_layout`.
- [`life.jl`](https://github.com/kahliburke/Tachikoma.jl/blob/main/demos/TachikomaDemos/src/life.jl) and [`mouse_demo.jl`](https://github.com/kahliburke/Tachikoma.jl/blob/main/demos/TachikomaDemos/src/mouse_demo.jl) — `init!`-based canvas setup, mouse hit testing with `contains`.
- [`snake.jl`](https://github.com/kahliburke/Tachikoma.jl/blob/main/demos/TachikomaDemos/src/snake.jl) — game loop with `tick`-driven simulation.

To run the demos locally, clone Tachikoma.jl, then from its root:

```julia
using Pkg
Pkg.activate("demos/TachikomaDemos")
Pkg.instantiate()
using TachikomaDemos
launcher()
```
