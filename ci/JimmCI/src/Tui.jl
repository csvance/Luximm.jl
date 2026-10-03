module Tui

using Dates
using Tachikoma
import Tachikoma:
    view, update!, should_quit, task_queue, set_wake!, has_pending_output, init!, cleanup!

using ..ConfigMod
using ..GitHubAppMod
using ..PathFilter
using ..Jobs
using ..BuilderMod
using ..SkipMarker

export TuiModel, run_tui

# ── View modes ───────────────────────────────────────────────────────

@enum ViewMode VIEW_LIST VIEW_RUNNING VIEW_CONFIRM_CANCEL VIEW_CONFIRM_FORK

mutable struct RunningJob
    job::Job
    cancel::BuildCancel
    started_at::DateTime
    current_family::String
end

# ── Model ────────────────────────────────────────────────────────────

@kwdef mutable struct TuiModel <: Model
    cfg::Config
    gh::GitHubApp
    builder::Builder

    jobs::Vector{Job} = Job[]
    selected::Int = 1
    list_offset::Int = 0
    mode::ViewMode = VIEW_LIST

    running::Union{Nothing, RunningJob} = nothing
    queue::Vector{Job} = Job[]            # back-to-back master queue
    pending_fork::Union{Nothing, Job} = nothing  # job awaiting fork-confirm

    # Live-output plumbing
    log_pane::ScrollPane = ScrollPane(
        String[];
        block = Block(
            title = " build output ",
            border_style = tstyle(:border),
            title_style = tstyle(:title, bold = true),
        ),
    )

    # Background tasks (Tachikoma)
    tq::TaskQueue = TaskQueue()

    # Thread-safe log pipeline: background threads put! here; view() drains into log_pane.
    log_channel::Channel{String} = Channel{String}(Inf)
    _wake::Union{Nothing, Function} = nothing

    refreshing::Bool = false
    status::String = "press r to refresh, ↑/↓ to select, Enter to run"
    tick::Int = 0
    quit::Bool = false
end

should_quit(m::TuiModel) = m.quit
task_queue(m::TuiModel) = m.tq
set_wake!(m::TuiModel, f::Function) = (m._wake = f; nothing)
has_pending_output(m::TuiModel) = isready(m.log_channel)

# Spawn the initial discovery from `init!` rather than blocking before
# `app(model)` runs — the user sees the spinner instead of a frozen
# terminal while the GitHub API call is in flight.
init!(m::TuiModel, ::Tachikoma.Terminal) = (_spawn_refresh!(m); nothing)

# Signal cancellation for any running job when the app exits — so
# pressing `q` (or Ctrl+C) during a build still tears down the
# subprocess group instead of orphaning it.
function cleanup!(m::TuiModel)
    rj = m.running
    rj === nothing && return
    BuilderMod.request_cancel!(rj.cancel)
    return
end

# ── Helpers ──────────────────────────────────────────────────────────

function _short(sha::AbstractString, n::Int = 8)
    n = min(n, length(sha))
    return SubString(sha, 1, n)
end

function _row_text(j::Job)
    when = Dates.format(j.created_at, dateformat"yyyy-mm-dd HH:MM")
    kind = j.kind == Jobs.PR_JOB ? "PR    " : "master"
    fams = join(j.families, ",")
    glyph = j.is_fork ? "⚠ " : "  "
    title = if j.kind == Jobs.PR_JOB
        repo = j.head_repo === nothing ? "?" : j.head_repo
        body = j.pr_title === nothing ? "" : j.pr_title
        "#$(j.pr_number) [$repo] $body"
    else
        "master @ $(_short(j.head_sha))"
    end
    return "$glyph$when  $kind  $(rpad(title, 60))  [$fams]"
end

function _push_log!(m::TuiModel, line::AbstractString)
    put!(m.log_channel, String(line))
    return m._wake !== nothing && m._wake()
end

# Matches the Builder's per-family marker:
#   "==> [check-run] jimm-ci / <family> ... → in_progress (...)"
# so the header can show which family of the sweep is currently running.
const _FAMILY_MARKER_RE = r"^==> \[check-run\] jimm-ci / (\w+).* → in_progress"

function _log_callback(m::TuiModel)
    return function (line)
        rj = m.running
        if rj !== nothing
            mt = match(_FAMILY_MARKER_RE, line)
            mt === nothing || (rj.current_family = String(mt.captures[1]))
        end
        return _push_log!(m, line)
    end
end

_current_master_index_or_nothing(m::TuiModel) =
    1 <= m.selected <= length(m.jobs) && m.jobs[m.selected].kind == Jobs.MASTER_JOB ?
    m.selected : nothing

# ── Background work: discovery (refresh) ─────────────────────────────

function _spawn_refresh!(m::TuiModel)
    m.refreshing && return
    m.refreshing = true
    m.status = "refreshing…"
    spawn_task!(m.tq, :refresh) do
        return discover_jobs(m.cfg, m.gh)
    end
    return nothing
end

# ── Background work: running one job ─────────────────────────────────

function _spawn_run!(m::TuiModel, job::Job)
    cancel = BuildCancel()
    m.running =
        RunningJob(job, cancel, now(UTC), isempty(job.families) ? "" : first(job.families))
    m.mode = VIEW_RUNNING
    # Clear the log pane between jobs so the user isn't confused by
    # leftover output from the previous build.
    empty!(m.log_pane.content::Vector{String})
    m.log_pane.block = Block(
        title = " build output  $(job.label) ",
        border_style = tstyle(:border),
        title_style = tstyle(:title, bold = true),
    )

    on_line = _log_callback(m)
    spawn_task!(m.tq, :run) do
        try
            BuilderMod.run_job(m.builder, job; on_line = on_line, token = cancel)
            return :ok
        catch e
            if e isa InterruptException || BuilderMod.is_cancelled(cancel)
                return :cancelled
            end
            @error "run_job failed" exception = (e, catch_backtrace())
            return e
        end
    end
    return nothing
end

function _start_next_in_queue!(m::TuiModel)
    isempty(m.queue) && return false
    next_job = popfirst!(m.queue)
    _spawn_run!(m, next_job)
    return true
end

# ── Background work: skip a single job ───────────────────────────────

function _spawn_skip!(m::TuiModel, job::Job)
    spawn_task!(m.tq, :skip) do
        try
            mark_skipped(
                m.gh,
                repo_fullname(m.cfg),
                job.head_sha,
                job.families;
                source = :run,
            )
            return job.head_sha
        catch e
            bt = catch_backtrace()
            @error "mark_skipped failed" job = job.label exception = (e, bt)
            # CapturedException bundles the backtrace so the TaskEvent
            # handler's `sprint(showerror, …)` renders something useful
            # instead of just the bare error type.
            return CapturedException(e, bt)
        end
    end
    m.status = "skipping $(job.label)…"
    return nothing
end

# ── Key handling ─────────────────────────────────────────────────────

function update!(m::TuiModel, evt::KeyEvent)
    if m.mode == VIEW_CONFIRM_CANCEL
        return _on_key_confirm_cancel!(m, evt)
    elseif m.mode == VIEW_CONFIRM_FORK
        return _on_key_confirm_fork!(m, evt)
    elseif m.mode == VIEW_RUNNING
        return _on_key_running!(m, evt)
    else
        return _on_key_list!(m, evt)
    end
end

function _on_key_list!(m::TuiModel, evt::KeyEvent)
    return if evt.key == :char
        c = evt.char
        c == 'q' && (m.quit = true; return)
        c == 'j' && (_move_selection!(m, 1); return)
        c == 'k' && (_move_selection!(m, -1); return)
        c == 'r' && (_spawn_refresh!(m); return)
        c == 's' && (_skip_selected!(m); return)
        c == 'A' && (_run_all_master!(m); return)
        c == 'y' && (_run_selected!(m); return)
    elseif evt.key == :up
        _move_selection!(m, -1)
    elseif evt.key == :down
        _move_selection!(m, 1)
    elseif evt.key == :enter || evt.key == :return
        _run_selected!(m)
    elseif evt.key == :escape
        m.quit = true
    end
end

function _on_key_running!(m::TuiModel, evt::KeyEvent)
    if evt.key == :char
        c = evt.char
        c == 'c' && (m.mode = VIEW_CONFIRM_CANCEL; return)
        c == 'C' && (_cancel_all!(m); return)
        # Quitting during a build signals the cancel via `cleanup!`, so
        # the subprocess group goes down rather than orphaning grandchildren.
        (c == 'q' || c == 'Q') && (m.quit = true; return)
    elseif evt.key == :escape
        m.mode = VIEW_CONFIRM_CANCEL
        return
    end
    return handle_key!(m.log_pane, evt)
end

function _on_key_confirm_cancel!(m::TuiModel, evt::KeyEvent)
    return if evt.key == :char
        c = evt.char
        if c == 'y' || c == 'Y'
            _cancel_current!(m)
        elseif c == 'a' || c == 'A'
            _cancel_all!(m)
        elseif c == 'n' || c == 'N'
            m.mode = VIEW_RUNNING
        end
    elseif evt.key == :escape
        m.mode = VIEW_RUNNING
    end
end

function _on_key_confirm_fork!(m::TuiModel, evt::KeyEvent)
    return if evt.key == :char
        c = evt.char
        if c == 'y' || c == 'Y'
            _confirm_fork_run!(m)
        elseif c == 'n' || c == 'N'
            _cancel_fork_modal!(m)
        end
    elseif evt.key == :escape
        _cancel_fork_modal!(m)
    end
end

function _move_selection!(m::TuiModel, delta::Int)
    isempty(m.jobs) && return
    return m.selected = clamp(m.selected + delta, 1, length(m.jobs))
end

function _run_selected!(m::TuiModel)
    isempty(m.jobs) && return
    job = m.jobs[m.selected]
    if job.is_fork
        # Two-step confirm: fork PRs run contributor code on the CI VM, so
        # the maintainer has to press y twice to approve.
        m.pending_fork = job
        m.mode = VIEW_CONFIRM_FORK
        return
    end
    # Remove from list — it's now in flight.
    deleteat!(m.jobs, m.selected)
    m.selected = clamp(m.selected, 1, max(length(m.jobs), 1))
    return _spawn_run!(m, job)
end

function _confirm_fork_run!(m::TuiModel)
    job = m.pending_fork
    m.pending_fork = nothing
    if job === nothing
        m.mode = VIEW_LIST
        return
    end
    # A background refresh may have replaced m.jobs while the modal was up;
    # `===` matches the stashed object regardless of position.
    idx = findfirst(j -> j === job, m.jobs)
    if idx !== nothing
        deleteat!(m.jobs, idx)
        m.selected = clamp(m.selected, 1, max(length(m.jobs), 1))
    end
    return _spawn_run!(m, job)
end

function _cancel_fork_modal!(m::TuiModel)
    m.pending_fork = nothing
    m.mode = VIEW_LIST
    return m.status = "fork run cancelled"
end

function _run_all_master!(m::TuiModel)
    isempty(m.jobs) && return
    idx = _current_master_index_or_nothing(m)
    if idx === nothing
        m.status = "[A] only enqueues master commits (selection is a PR)"
        return
    end
    # Build queue oldest-first: anchor + every older master commit.
    anchor = m.jobs[idx]
    masters_older = [j for (i, j) in pairs(m.jobs) if i > idx && j.kind == Jobs.MASTER_JOB]
    sort!(masters_older; by = j -> j.created_at)   # oldest first
    queue = vcat(masters_older, [anchor])

    # Drop them from the visible list.
    keep_idx = Set{Int}()
    for (i, j) in pairs(m.jobs)
        (j in queue) || push!(keep_idx, i)
    end
    m.jobs = m.jobs[sort(collect(keep_idx))]
    m.selected = clamp(m.selected, 1, max(length(m.jobs), 1))

    isempty(queue) && return
    first_job = popfirst!(queue)
    m.queue = queue
    return _spawn_run!(m, first_job)
end

function _skip_selected!(m::TuiModel)
    isempty(m.jobs) && return
    job = m.jobs[m.selected]
    deleteat!(m.jobs, m.selected)
    m.selected = clamp(m.selected, 1, max(length(m.jobs), 1))
    return _spawn_skip!(m, job)
end

function _cancel_current!(m::TuiModel)
    rj = m.running
    rj === nothing && return
    request_cancel!(rj.cancel)
    m.status = "cancellation requested for $(rj.job.label)…"
    return m.mode = VIEW_RUNNING
end

function _cancel_all!(m::TuiModel)
    empty!(m.queue)
    _cancel_current!(m)
    return m.status = "cancelling current + clearing queue"
end

# ── TaskEvent ingestion ──────────────────────────────────────────────

function update!(m::TuiModel, evt::TaskEvent)
    return if evt.id == :refresh
        m.refreshing = false
        if evt.value isa Exception
            m.status = "refresh failed: $(sprint(showerror, evt.value))"
        else
            m.jobs = evt.value::Vector{Job}
            m.selected = clamp(m.selected, 1, max(length(m.jobs), 1))
            m.status = "$(length(m.jobs)) job(s) pending"
        end
    elseif evt.id == :run
        # Job task finished — clean up `running`, drain the queue if any.
        rj = m.running
        m.running = nothing
        if rj !== nothing
            label = rj.job.label
            outcome = if evt.value isa Exception
                "errored: $(sprint(showerror, evt.value))"
            elseif evt.value === :cancelled
                "cancelled"
            else
                "completed"
            end
            m.status = "$(label) $(outcome)"
        end
        if !_start_next_in_queue!(m)
            m.mode = VIEW_LIST
        end
    elseif evt.id == :skip
        if evt.value isa Exception
            m.status = "skip failed: $(sprint(showerror, evt.value))"
        else
            m.status = "skipped $(_short(String(evt.value)))"
        end
    end
end

# Catch-all so other Tachikoma events (resize, mouse, etc.) don't error.
update!(m::TuiModel, evt::Event) = nothing

# ── View ─────────────────────────────────────────────────────────────

function view(m::TuiModel, f::Frame)
    m.tick += 1
    while isready(m.log_channel)
        push_line!(m.log_pane, take!(m.log_channel))
    end
    buf = f.buffer

    outer = Block(
        title = " jimm-ci ",
        border_style = tstyle(:border),
        title_style = tstyle(:title, bold = true),
    )
    inner = render(outer, f.area, buf)

    rows = split_layout(Layout(Vertical, [Fixed(1), Fill(), Fixed(1)]), inner)
    length(rows) < 3 && return
    header_area, body_area, status_area = rows[1], rows[2], rows[3]

    _render_header!(m, buf, header_area)

    if m.mode == VIEW_LIST
        _render_list!(m, buf, body_area)
    elseif m.mode == VIEW_RUNNING
        _render_running!(m, buf, body_area)
    elseif m.mode == VIEW_CONFIRM_CANCEL
        _render_running!(m, buf, body_area)
        _render_cancel_modal!(m, buf, body_area)
    elseif m.mode == VIEW_CONFIRM_FORK
        _render_list!(m, buf, body_area)
        _render_fork_modal!(m, buf, body_area)
    end

    return _render_status!(m, buf, status_area)
end

function _render_header!(m::TuiModel, buf, area)
    queue_n = length(m.queue)
    extra = queue_n > 0 ? "  queue: $queue_n" : ""
    spinner = if m.refreshing || m.running !== nothing
        si = mod1(m.tick ÷ 3, length(SPINNER_BRAILLE))
        string(SPINNER_BRAILLE[si], " ")
    else
        ""
    end
    title = "$(spinner)$(length(m.jobs)) pending$(extra)"
    return set_string!(buf, area.x + 1, area.y, title, tstyle(:text); max_x = right(area))
end

function _render_list!(m::TuiModel, buf, area)
    if isempty(m.jobs)
        msg = "no jobs to run — press r to refresh"
        set_string!(
            buf,
            area.x + max(0, (area.width - length(msg)) ÷ 2),
            area.y + area.height ÷ 2,
            msg,
            tstyle(:text_dim);
            max_x = right(area),
        )
        return
    end
    items = [
        ListItem(
            _row_text(j),
            j.is_fork ? tstyle(:warning) :
                j.kind == Jobs.PR_JOB ? tstyle(:primary) : tstyle(:secondary),
        ) for j in m.jobs
    ]
    return render(
        SelectableList(
            items;
            selected = m.selected,
            offset = m.list_offset,
            block = Block(
                title = " queue ",
                border_style = tstyle(:border),
                title_style = tstyle(:title),
            ),
            highlight_style = tstyle(:accent, bold = true),
            tick = m.tick,
        ),
        area,
        buf,
    )
end

function _render_running!(m::TuiModel, buf, area)
    rj = m.running
    if rj === nothing
        # Job just finished but we haven't returned to LIST yet.
        render(m.log_pane, area, buf)
        return
    end
    rows = split_layout(Layout(Vertical, [Fixed(2), Fill()]), area)
    length(rows) < 2 && return
    info_area = rows[1]
    pane_area = rows[2]

    elapsed = Dates.value(now(UTC) - rj.started_at) ÷ 1000
    current = isempty(rj.current_family) ? "(starting)" : rj.current_family
    info =
        "running $(rj.job.label)  elapsed $(elapsed)s  on $(current) " *
        "[$(length(rj.job.families)) families: $(join(rj.job.families, ","))]"
    set_string!(
        buf,
        info_area.x + 1,
        info_area.y,
        info,
        tstyle(:primary, bold = true);
        max_x = right(info_area),
    )
    queue_n = length(m.queue)
    if queue_n > 0
        set_string!(
            buf,
            info_area.x + 1,
            info_area.y + 1,
            "back-to-back queue: $queue_n more",
            tstyle(:text_dim);
            max_x = right(info_area),
        )
    end
    return render(m.log_pane, pane_area, buf)
end

function _render_cancel_modal!(m::TuiModel, buf, area)
    w = min(60, area.width - 4)
    h = 7
    (w < 10 || h > area.height) && return
    rect = center(area, w, h)
    for cy in rect.y:bottom(rect), cx in rect.x:right(rect)
        in_bounds(buf, cx, cy) && set_char!(buf, cx, cy, ' ', tstyle(:text))
    end
    inner = render(
        Block(
            title = " cancel? ",
            border_style = tstyle(:warning, bold = true),
            title_style = tstyle(:warning, bold = true),
        ),
        rect,
        buf,
    )
    set_string!(
        buf,
        inner.x + 2,
        inner.y + 1,
        "cancel the running build?",
        tstyle(:text);
        max_x = right(inner),
    )
    return set_string!(
        buf,
        inner.x + 2,
        inner.y + 3,
        "[y] cancel current   [a] cancel current + queue   [n] keep running",
        tstyle(:text_dim);
        max_x = right(inner),
    )
end

function _render_fork_modal!(m::TuiModel, buf, area)
    job = m.pending_fork
    job === nothing && return
    w = min(80, area.width - 4)
    h = 9
    (w < 20 || h > area.height) && return
    rect = center(area, w, h)
    for cy in rect.y:bottom(rect), cx in rect.x:right(rect)
        in_bounds(buf, cx, cy) && set_char!(buf, cx, cy, ' ', tstyle(:text))
    end
    inner = render(
        Block(
            title = " ⚠ fork PR — confirm ",
            border_style = tstyle(:warning, bold = true),
            title_style = tstyle(:warning, bold = true),
        ),
        rect,
        buf,
    )
    repo = job.head_repo === nothing ? "?" : job.head_repo
    title = job.pr_title === nothing ? "" : job.pr_title
    set_string!(
        buf,
        inner.x + 2,
        inner.y + 1,
        "PR #$(job.pr_number) from $(repo)",
        tstyle(:text);
        max_x = right(inner),
    )
    set_string!(
        buf,
        inner.x + 2,
        inner.y + 2,
        "head $(_short(job.head_sha, 12))  $(title)",
        tstyle(:text_dim);
        max_x = right(inner),
    )
    set_string!(
        buf,
        inner.x + 2,
        inner.y + 4,
        "Running this executes contributor code on the CI VM.",
        tstyle(:text);
        max_x = right(inner),
    )
    return set_string!(
        buf,
        inner.x + 2,
        inner.y + 6,
        "[y] run   [n/Esc] cancel",
        tstyle(:text_dim);
        max_x = right(inner),
    )
end

function _render_status!(m::TuiModel, buf, area)
    hint = if m.mode == VIEW_LIST
        "[↑↓/jk] move  [Enter/y] run  [A] run all master  [s] skip  [r] refresh  [q] quit"
    elseif m.mode == VIEW_RUNNING
        "[c] cancel  [C] cancel all  [q] quit (signals cancel)  [PgUp/PgDn] scroll"
    elseif m.mode == VIEW_CONFIRM_FORK
        "[y] run fork PR   [n/Esc] cancel"
    else
        "[y] yes  [a] cancel queue too  [n/Esc] keep running"
    end
    return render(
        StatusBar(
            left = [Span("  $(m.status)  ", tstyle(:text_dim))],
            right = [Span(hint * " ", tstyle(:text_dim))],
        ),
        area,
        buf,
    )
end

# ── Entry point ──────────────────────────────────────────────────────

"""
    run_tui(cfg, gh)

Launch the interactive TUI. Performs an initial discovery before the
first frame so the list is populated when the user sees it.
"""
function run_tui(cfg::Config, gh::GitHubApp)
    builder = Builder(cfg, gh)
    model = TuiModel(cfg = cfg, gh = gh, builder = builder)
    # Discovery is spawned from `init!` so the spinner shows while the
    # GitHub API call is in flight instead of a frozen terminal. The
    # spinner ticks every 3 frames; 15 fps is plenty for a CI dashboard
    # and halves render cost on a long-idle daemon.
    app(model; fps = 15)
    return nothing
end

end # module
