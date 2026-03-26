import * as pako from 'https://cdn.jsdelivr.net/npm/pako@2.0.4/+esm';

async function fetchData(url) {
    const response = await fetch(url);
    return await response.json();
}
async function fetchAndDecompressData(url) {
    const response = await fetch(url);
    const compressedData = await response.arrayBuffer();
    const decompressedData = pako.ungzip(new Uint8Array(compressedData), { to: 'string' });
    return JSON.parse(decompressedData);
}

function getFileExtension(path) {
    return path.split('.').pop();
}

function fallbackTerminationInfo(parameters, stepData) {
    const termination = parameters?.mdp?.termination;
    if (!termination || !termination.enabled || !stepData?.state) {
        return {
            terminated: Boolean(stepData?.terminated),
            enabled: Boolean(termination?.enabled),
            reason: termination?.enabled ? "not_available" : "termination_disabled",
            violations: [],
            thresholds: termination || {}
        };
    }
    const violations = [];
    const metrics = [
        { key: "position", values: stepData.state.position, threshold: termination.position_threshold },
        { key: "linear_velocity", values: stepData.state.linear_velocity, threshold: termination.linear_velocity_threshold },
        { key: "angular_velocity", values: stepData.state.angular_velocity, threshold: termination.angular_velocity_threshold }
    ];
    metrics.forEach(({ key, values, threshold }) => {
        if (!Array.isArray(values)) {
            return;
        }
        values.slice(0, 3).forEach((value, axis) => {
            const absValue = Math.abs(value);
            if (absValue > threshold) {
                violations.push({ metric: key, axis, value: absValue, threshold });
            }
        });
    });
    return {
        terminated: Boolean(stepData?.terminated),
        enabled: true,
        reason: violations.length > 0 ? "threshold_exceeded" : "within_thresholds",
        violations,
        thresholds: {
            position: termination.position_threshold,
            linear_velocity: termination.linear_velocity_threshold,
            angular_velocity: termination.angular_velocity_threshold
        }
    };
}

// ── Telemetry / Chart helpers ──────────────────────────────────────────────

async function loadChartJS() {
    return new Promise((resolve, reject) => {
        if (window.Chart) { resolve(window.Chart); return; }
        const s = document.createElement('script');
        s.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
        s.onload = () => resolve(window.Chart);
        s.onerror = () => reject(new Error('Failed to load Chart.js'));
        document.head.appendChild(s);
    });
}

// MATLAB classic color order
const MATLAB_COLORS = ['#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'];

const STEP_LINE_PLUGIN = {
    id: 'stepLine',
    afterDraw(chart) {
        if (chart._stepPos === undefined || !chart.data.labels.length) return;
        const xScale = chart.scales.x;
        if (!xScale) return;
        const x = xScale.getPixelForValue(chart._stepPos);
        if (!isFinite(x)) return;
        const ctx = chart.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x, chart.chartArea.top);
        ctx.lineTo(x, chart.chartArea.bottom);
        ctx.strokeStyle = 'rgba(180, 0, 0, 0.75)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 3]);
        ctx.stroke();
        ctx.restore();
    }
};

const DROP_LINE_PLUGIN = {
    id: 'dropLine',
    afterDraw(chart) {
        if (chart._dropStep === undefined || !chart.data.labels.length) return;
        const xScale = chart.scales.x;
        if (!xScale) return;
        const x = xScale.getPixelForValue(chart._dropStep);
        if (!isFinite(x)) return;
        const ctx = chart.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x, chart.chartArea.top);
        ctx.lineTo(x, chart.chartArea.bottom);
        ctx.strokeStyle = 'rgba(255, 140, 0, 0.85)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        // Small label at top
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255, 140, 0, 0.9)';
        ctx.font = '8px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('drop', x, chart.chartArea.top + 8);
        ctx.restore();
    }
};

const CHART_BG_PLUGIN = {
    id: 'chartBg',
    beforeDraw(chart) {
        const ctx = chart.ctx;
        ctx.save();
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, chart.width, chart.height);
        ctx.restore();
    }
};

const ROTOR_COLORS = MATLAB_COLORS;

function buildChart(container, datasets, title, height = 110) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('telem-chart-wrapper');
    wrapper.style.height = height + 'px';
    container.appendChild(wrapper);
    const canvas = document.createElement('canvas');
    wrapper.appendChild(canvas);
    return new window.Chart(canvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: datasets.map(d => ({
                label: d.label, data: [],
                borderColor: d.color,
                backgroundColor: 'transparent',
                borderWidth: 1.5, pointRadius: 0, tension: 0, fill: false
            }))
        },
        options: {
            animation: false, responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#000000',
                        font: { size: 9, family: 'Helvetica Neue, Arial, sans-serif' },
                        boxWidth: 20, boxHeight: 2, padding: 6,
                        usePointStyle: false
                    }
                },
                title: {
                    display: true, text: title, color: '#000000',
                    font: { size: 10, family: 'Helvetica Neue, Arial, sans-serif', weight: 'bold' },
                    padding: { top: 5, bottom: 0 }
                },
                tooltip: { enabled: false }
            },
            scales: {
                x: { display: true,
                     ticks: { display: false },
                     grid: { color: '#e0e0e0', drawBorder: true, borderColor: '#666' },
                     border: { color: '#666666' }
                },
                y: {
                    ticks: { color: '#333333',
                             font: { size: 8, family: 'Helvetica Neue, Arial, sans-serif' },
                             maxTicksLimit: 4,
                             callback: v => Number(v).toFixed(2) },
                    grid: { color: '#e0e0e0' },
                    border: { color: '#666666' }
                }
            },
            layout: { padding: { left: 2, right: 6, top: 0, bottom: 4 } }
        },
        plugins: [CHART_BG_PLUGIN, DROP_LINE_PLUGIN, STEP_LINE_PLUGIN]
    });
}

// ── Main class ─────────────────────────────────────────────────────────────

export class TrajectoryPlayer{
    constructor(ui_path, size, options) {
        // Support legacy call: new TrajectoryPlayer(path, {verbose:true}) — options in size slot
        this.size = (typeof size === 'number') ? size : null;
        this.options = (typeof size === 'object' && size !== null ? size : options) || {};

        this.ui = import(ui_path)
        this.container = document.createElement('div');
        this.container.classList.add("trajectory-player-container")

        // Left column: 3-D canvas + controls
        this.left_col = document.createElement('div');
        this.left_col.classList.add("trajectory-player-left-col");

        this.canvas_container = document.createElement('div');
        this.canvas_container.classList.add("trajectory-player-canvas-container")
        this.left_col.appendChild(this.canvas_container);

        this.loading_text = document.createElement('p');
        this.loading_text.style.display = "none";
        this.left_col.appendChild(this.loading_text);

        this.controls_container = document.createElement('div');
        this.controls_container.classList.add("trajectory-player-controls-container");
        this.left_col.appendChild(this.controls_container);

        this.container.appendChild(this.left_col);

        // Telemetry panel (opt-in via options.telemetry)
        if (this.options.telemetry) {
            this.container.classList.add('telem-active');
            this.telemetry_panel = document.createElement('div');
            this.telemetry_panel.classList.add('telem-panel');
            this.container.appendChild(this.telemetry_panel);
            this.telem_initialized = false;
            this.telem_uav_index = 0;
            this.telem_header_label = null;
            this.charts = {};
        }

        this.canvas = null
    }
    getCanvas(){
        return this.container;
    }

    // ── Telemetry ──────────────────────────────────────────────────────────

    async initTelemetry() {
        await loadChartJS();
        const panel = this.telemetry_panel;

        // Header
        const header = document.createElement('div');
        header.classList.add('telem-header');
        const headerLeft = document.createElement('span');
        headerLeft.innerHTML = '<span class="telem-dot"></span>DRONE TELEMETRY';
        headerLeft.style.display = 'flex';
        headerLeft.style.alignItems = 'center';
        headerLeft.style.gap = '7px';
        this.telem_header_label = document.createElement('span');
        this.telem_header_label.classList.add('telem-uav-badge');
        this.telem_header_label.textContent = 'UAV 0';
        header.appendChild(headerLeft);
        header.appendChild(this.telem_header_label);
        panel.appendChild(header);

        // State cards
        const cards = document.createElement('div');
        cards.classList.add('telem-cards');

        const makeCard = (label, fields) => {
            const card = document.createElement('div');
            card.classList.add('telem-card');
            const lbl = document.createElement('div');
            lbl.classList.add('telem-card-label');
            lbl.textContent = label;
            card.appendChild(lbl);
            const vals = {};
            fields.forEach(([name, color]) => {
                const row = document.createElement('div');
                row.classList.add('telem-card-row');
                const sn = document.createElement('span');
                sn.classList.add('telem-field-name');
                sn.textContent = name;
                if (color) sn.style.color = color;
                const sv = document.createElement('span');
                sv.classList.add('telem-field-val');
                sv.textContent = '—';
                row.appendChild(sn); row.appendChild(sv);
                card.appendChild(row);
                vals[name] = sv;
            });
            return { card, vals };
        };

        const posCard  = makeCard('POSITION (m)',       [['x','#0072BD'],['y','#D95319'],['z','#77AC30']]);
        const velCard  = makeCard('LIN. VEL (m/s)',      [['vx','#0072BD'],['vy','#D95319'],['vz','#77AC30']]);
        const angCard  = makeCard('ANG. VEL (rad/s)',    [['wx','#7E2F8E'],['wy','#4DBEEE'],['wz','#A2142F']]);
        const infoCard = makeCard('STEP INFO',           [['step',null],['reward','#D95319'],['return','#0072BD']]);

        [posCard, velCard, angCard, infoCard].forEach(c => cards.appendChild(c.card));
        panel.appendChild(cards);
        this.telem_cards = { posCard, velCard, angCard, infoCard };

        // Charts
        const cs = document.createElement('div');
        cs.classList.add('telem-charts');
        panel.appendChild(cs);

        this.charts.position = buildChart(cs, [
            { label: 'x', color: '#0072BD' }, { label: 'y', color: '#D95319' }, { label: 'z', color: '#77AC30' }
        ], 'POSITION');
        this.charts.velocity = buildChart(cs, [
            { label: 'vx', color: '#0072BD' }, { label: 'vy', color: '#D95319' }, { label: 'vz', color: '#77AC30' }
        ], 'LINEAR VELOCITY');
        this.charts.angular = buildChart(cs, [
            { label: 'wx', color: '#7E2F8E' }, { label: 'wy', color: '#4DBEEE' }, { label: 'wz', color: '#A2142F' }
        ], 'ANGULAR VELOCITY');
        this.charts.actions = buildChart(cs,
            Array.from({ length: 6 }, (_, i) => ({ label: `r${i+1}`, color: MATLAB_COLORS[i] })),
            'ROTOR COMMANDS', 95);
        this.charts.reward = buildChart(cs, [
            { label: 'reward', color: '#D95319' }, { label: 'cumulative', color: '#0072BD' }
        ], 'REWARD', 95);

        this.telem_initialized = true;
    }

    loadEpisodeData(episode) {
        if (!this.telem_initialized) return;
        const traj = episode?.trajectory;
        if (!traj || !traj.length) return;
        const drop_step = episode?.parameters?.payload?.drop_step;
        this._current_drop_step = (drop_step !== undefined) ? drop_step : undefined;

        const labels = traj.map((_, i) => i);
        const field = (key, idx) => traj.map(s => {
            const v = s?.state?.[key];
            return Array.isArray(v) ? (v[idx] ?? 0) : 0;
        });

        this._loadChart(this.charts.position, labels,
            field('position', 0), field('position', 1), field('position', 2));
        this._loadChart(this.charts.velocity, labels,
            field('linear_velocity', 0), field('linear_velocity', 1), field('linear_velocity', 2));
        this._loadChart(this.charts.angular, labels,
            field('angular_velocity', 0), field('angular_velocity', 1), field('angular_velocity', 2));

        const nAct = Array.isArray(traj[0]?.action) ? traj[0].action.length : 0;
        if (nAct > 0) {
            const chart = this.charts.actions;
            while (chart.data.datasets.length < nAct) {
                const i = chart.data.datasets.length;
                chart.data.datasets.push({ label: `r${i+1}`, data: [],
                    borderColor: ROTOR_COLORS[i % ROTOR_COLORS.length], borderWidth: 1.5, pointRadius: 0, tension: 0.2 });
            }
            this._loadChart(chart, labels,
                ...Array.from({ length: Math.min(nAct, 6) }, (_, i) => traj.map(s => s?.action?.[i] ?? 0)));
        }

        const rewards = traj.map(s => s?.reward ?? 0);
        let cum = 0;
        this._loadChart(this.charts.reward, labels, rewards, rewards.map(r => (cum += r)));
    }

    _loadChart(chart, labels, ...series) {
        chart.data.labels = labels;
        chart.data.datasets.forEach((ds, i) => { ds.data = series[i] || []; });
        chart._stepPos = 0;
        if (this._current_drop_step !== undefined) chart._dropStep = this._current_drop_step;
        chart.update('none');
    }

    updateTelemetry(state, _action, reward, step, totalReward) {
        if (!this.telem_initialized) return;

        const pos = state?.position         || [0, 0, 0];
        const vel = state?.linear_velocity  || [0, 0, 0];
        const ang = state?.angular_velocity || [0, 0, 0];
        const fmt = v => (typeof v === 'number') ? v.toFixed(4) : '—';

        const { posCard, velCard, angCard, infoCard } = this.telem_cards;
        posCard.vals['x'].textContent  = fmt(pos[0]);
        posCard.vals['y'].textContent  = fmt(pos[1]);
        posCard.vals['z'].textContent  = fmt(pos[2]);
        velCard.vals['vx'].textContent = fmt(vel[0]);
        velCard.vals['vy'].textContent = fmt(vel[1]);
        velCard.vals['vz'].textContent = fmt(vel[2]);
        angCard.vals['wx'].textContent = fmt(ang[0]);
        angCard.vals['wy'].textContent = fmt(ang[1]);
        angCard.vals['wz'].textContent = fmt(ang[2]);
        infoCard.vals['step'].textContent   = step;
        infoCard.vals['reward'].textContent = fmt(reward);
        infoCard.vals['return'].textContent = fmt(totalReward);

        Object.values(this.charts).forEach(chart => {
            chart._stepPos = step;
            chart.update('none');
        });
    }

    async playTrajectories(path) {
        this.loading_text.innerHTML = `Loading Trajectory Data from ${path}`
        this.loading_text.style.display = "inline";

        const trajectoryData = getFileExtension(path) === "json" ? await fetchData(path) : await fetchAndDecompressData(path)
        this.loading_text.style.display = "none";

        if (this.options.telemetry) {
            await this.initTelemetry();
        }
        let currentEpisode = 0;
        let currentStep = 0;
        let currentEpisodeLength = 0;
        let currentEpisodeReturn = 0;
        let current_episode_return_multi = 0;

        const ui = await this.ui;

        const episode_info = document.createElement('div');
        episode_info.classList.add("trajectory-player-episode-info");
        const selection_info = document.createElement('div');
        selection_info.classList.add("trajectory-player-selection-info");
        selection_info.style.whiteSpace = "pre-wrap";
        selection_info.style.maxWidth = "900px";
        selection_info.style.margin = "8px auto";
        selection_info.style.textAlign = "left";

        this.controls_container.replaceChildren(episode_info, selection_info);
        const previous_button = document.createElement('button');
        previous_button.innerHTML = "Previous Episode";
        this.controls_container.appendChild(previous_button);
        const restart_button = document.createElement('button');
        restart_button.innerHTML = "Restart Episode";
        this.controls_container.appendChild(restart_button);
        const pause_button = document.createElement('button');
        pause_button.innerHTML = "Pause";
        this.controls_container.appendChild(pause_button);
        const skip_button = document.createElement('button');
        skip_button.innerHTML = "Next Episode";
        this.controls_container.appendChild(skip_button);

        var ratio = window.devicePixelRatio || 1;


        const onResize = () => {
            if(this.canvas){
                let size;
                if(this.size){
                    size = this.size;
                }
                else{
                    size = Math.min(this.canvas_container.clientWidth, this.canvas_container.clientHeight);
                }
                this.canvas.width = size * ratio;
                this.canvas.height = size * ratio;

                this.canvas.style.width = size + 'px';
                this.canvas.style.height = size + 'px';
            }
        }
        const resizeListener = new ResizeObserver(onResize);
        resizeListener.observe(this.canvas_container);
        onResize();
        window.addEventListener('resize', onResize);

        const dt = trajectoryData[0].trajectory[0].dt;
        const single = !Boolean(ui.render_multi)
        if(!single){
            previous_button.style.display = "none";
            skip_button.style.display = "none";
        }
        let current_parameters = null
        let ui_state = null
        let current_state = null
        let current_action = null
        let render_loop = {id: 0}
        let render_loops_running = {}
        let current_parameters_multi = null
        let current_step_data_multi = null
        let current_step_data_single = null
        let current_render_step_multi = 0
        let current_render_step_single = 0
        let fallback_click_listener_attached = false
        let isPaused = false
        const renderSelectionInfo = (lines) => {
            selection_info.textContent = lines.join("\n");
        };
        const formatViolationLines = (termInfo, prefix = "") => {
            const violations = termInfo?.violations || [];
            if (violations.length === 0) {
                return [`${prefix}violations=0`];
            }
            return [
                `${prefix}violations=${violations.length}`,
                ...violations.map((violation, index) => {
                    const value = Number(violation.value);
                    const threshold = Number(violation.threshold);
                    const exceededBy = value - threshold;
                    return `${prefix}v${index + 1}: metric=${violation.metric}, axis=${violation.axis}, value=${value.toFixed(6)}, threshold=${threshold.toFixed(6)}, exceeded_by=${exceededBy.toFixed(6)}`;
                })
            ];
        };
        const logSelectedUAV = (droneIndex) => {
            if (single) {
                if (!trajectoryData[currentEpisode] || !current_step_data_single) {
                    return;
                }
                const selected_step_data = current_step_data_single;
                const selected_parameters = trajectoryData[currentEpisode].parameters;
                const termInfo = selected_step_data.termination_info || fallbackTerminationInfo(selected_parameters, selected_step_data);
                console.log(`[UAV 0] episode=${currentEpisode} step=${current_render_step_single} terminated=${Boolean(selected_step_data.terminated)}`);
                console.log(`[UAV 0] termination details`, termInfo);
                renderSelectionInfo([
                    `[UAV 0] episode=${currentEpisode} step=${current_render_step_single}`,
                    `terminated=${Boolean(selected_step_data.terminated)}`,
                    `reason=${termInfo.reason || "unknown"}`,
                    ...formatViolationLines(termInfo)
                ]);
                return;
            }
            if (!Number.isInteger(droneIndex) || droneIndex < 0 || droneIndex >= trajectoryData.length) {
                console.log(`Clicked UAV index ${droneIndex} is out of range.`);
                return;
            }
            const episode = trajectoryData[droneIndex];
            const selected_step_data = episode?.trajectory?.[current_render_step_multi];
            if (!selected_step_data) {
                return;
            }
            const firstTerminatedStep = episode.trajectory.findIndex(step => step.terminated);
            const everTerminated = firstTerminatedStep >= 0;
            const firstTerminatedStepData = everTerminated ? episode.trajectory[firstTerminatedStep] : null;
            const currentStepTermInfo = selected_step_data.termination_info || fallbackTerminationInfo(episode.parameters, selected_step_data);
            const firstTerminatedTermInfo = firstTerminatedStepData ? (firstTerminatedStepData.termination_info || fallbackTerminationInfo(episode.parameters, firstTerminatedStepData)) : null;
            console.log(`[UAV ${droneIndex}] step=${current_render_step_multi} terminated_now=${Boolean(selected_step_data.terminated)} terminated_in_episode=${everTerminated}`);
            if (everTerminated) {
                console.log(`[UAV ${droneIndex}] first terminated step=${firstTerminatedStep}`);
            }
            console.log(`[UAV ${droneIndex}] termination details (current step)`, currentStepTermInfo);
            if (firstTerminatedTermInfo) {
                console.log(`[UAV ${droneIndex}] termination details (first terminated step ${firstTerminatedStep})`, firstTerminatedTermInfo);
            }
            const infoLines = [
                `[UAV ${droneIndex}] step=${current_render_step_multi}`,
                `terminated_now=${Boolean(selected_step_data.terminated)}`,
                `terminated_in_episode=${everTerminated}`,
                `first_terminated_step=${everTerminated ? firstTerminatedStep : "none"}`,
                `reason_current_step=${currentStepTermInfo.reason || "unknown"}`,
                ...formatViolationLines(currentStepTermInfo, "current_")
            ];
            if (firstTerminatedTermInfo) {
                infoLines.push(`reason_first_terminated_step=${firstTerminatedTermInfo.reason || "unknown"}`);
                infoLines.push(...formatViolationLines(firstTerminatedTermInfo, "first_terminated_"));
            }
            renderSelectionInfo(infoLines);
            // Switch telemetry to this drone
            if (this.options.telemetry && this.telem_initialized) {
                this.telem_uav_index = droneIndex;
                this.loadEpisodeData(trajectoryData[droneIndex]);
                if (this.telem_header_label) {
                    this.telem_header_label.textContent = `UAV ${droneIndex} / ${trajectoryData.length}`;
                }
            }
        }
        const attachFallbackClickHandler = () => {
            if (fallback_click_listener_attached || !this.canvas || !ui_state || !ui_state.THREE || !ui_state.camera) {
                return;
            }
            fallback_click_listener_attached = true;
            const raycaster = new ui_state.THREE.Raycaster();
            const pointer = new ui_state.THREE.Vector2();
            this.canvas.addEventListener("click", (event) => {
                const roots = [];
                if (Array.isArray(ui_state.clickable_drones) && ui_state.clickable_drones.length > 0) {
                    roots.push(...ui_state.clickable_drones);
                }
                else if (Array.isArray(ui_state.drones) && ui_state.drones.length > 0) {
                    roots.push(...ui_state.drones.map(drone => drone.get()).filter(Boolean));
                }
                else if (ui_state.drone && ui_state.drone.get) {
                    roots.push(ui_state.drone.get());
                }
                if (roots.length === 0) {
                    return;
                }
                const rect = this.canvas.getBoundingClientRect();
                pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                raycaster.setFromCamera(pointer, ui_state.camera);
                const hits = raycaster.intersectObjects(roots, true);
                if (hits.length === 0) {
                    return;
                }
                if (single) {
                    logSelectedUAV(0);
                    return;
                }
                const hitRoot = roots.findIndex(root => {
                    let node = hits[0].object;
                    while (node) {
                        if (node === root) {
                            return true;
                        }
                        node = node.parent;
                    }
                    return false;
                });
                if (hitRoot >= 0) {
                    logSelectedUAV(hitRoot);
                }
            });
        };

        const render = async () => {
            if (ui.render && ui.init){
                if(!ui_state){
                    if(current_parameters || current_parameters_multi){
                        this.canvas_container.innerHTML = "";
                        this.canvas = document.createElement('canvas');
                        this.canvas.classList.add("trajectory-player-canvas")
                        this.canvas_container.appendChild(this.canvas);
                        onResize()
                        console.log("Init Environment UI")
                        ui_state = await ui.init(this.canvas, {
                            devicePixelRatio: window.devicePixelRatio,
                            on_drone_click: (droneIndex) => logSelectedUAV(droneIndex)
                        })
                        if(ui_state.cursor_grab){
                            this.canvas.style.cursor = "grab"
                        }
                        attachFallbackClickHandler()
                    }
                    else{
                        throw new Error('Init but parameters not set')
                    }
                }
                if(ui_state){
                    render_loop.id += 1;
                    await new Promise((resolve) => {
                        let interval_id = null
                        interval_id = setInterval(() => {
                            if(Object.keys(render_loops_running).length === 0){
                                clearInterval(interval_id)
                                resolve();
                            }
                        })
                    })
                    const current_id = render_loop.id;
                    render_loops_running[current_id] = true;
                    const loop = async () => {
                        if(render_loop.id === current_id){
                            if(single){
                                if(current_parameters && current_state && current_action){
                                    await ui.render(ui_state, current_parameters, current_state, current_action)
                                    requestAnimationFrame(loop);
                                    return
                                }
                            }
                            else{
                                if(current_parameters_multi && current_step_data_multi){
                                    await ui.render_multi(ui_state, current_parameters_multi, current_step_data_multi.map(step => step.state), current_step_data_multi.map(step => step.action))
                                    requestAnimationFrame(loop);
                                    return
                                }
                            }
                        }
                        console.log("Stopping render loop: ", current_id)
                        delete render_loops_running[current_id];
                    }
                    if(ui.episode_init){
                        if(single){
                            await ui.episode_init(ui_state, current_parameters)
                        }
                        else{
                            await ui.episode_init_multi(ui_state, current_parameters_multi)
                        }
                    }
                    requestAnimationFrame(loop)
                }
                else{
                    throw new Error('init function not successfull')
                }
            }
        }
        const step = () => {
            if(isPaused){
                return;
            }
            // const size = Math.min(this.canvas_container.clientWidth, this.canvas_container.clientHeight);
            // this.canvas.width = size;
            // this.canvas.height = size;
            if(this.options["verbose"]){
                episode_info.innerHTML = `Path: ${path}</br>`;
            }
            else{
                episode_info.innerHTML = "";
            }
            if(single){
                episode_info.innerHTML += `Episode: ${currentEpisode+1}/${trajectoryData.length}, Step: ${currentStep}, Return: ${currentEpisodeReturn.toFixed(2)}`;
                if (currentEpisode < trajectoryData.length) {
                    const { parameters, trajectory } = trajectoryData[currentEpisode];
                    current_parameters = parameters;
                    let skip = false;
                    if (currentStep < trajectory.length) {
                        const { state, action, reward, terminated } = trajectory[currentStep];
                        current_step_data_single = trajectory[currentStep];
                        current_render_step_single = currentStep;
                        current_state = state;
                        current_action = action;
                        currentEpisodeLength++;
                        currentEpisodeReturn += reward;
                        if(currentStep === 0){
                            render()
                            if (this.options.telemetry && this.telem_initialized) {
                                this.loadEpisodeData(trajectoryData[currentEpisode]);
                            }
                        }
                        if (this.options.telemetry && this.telem_initialized) {
                            this.updateTelemetry(state, action, reward, currentStep, currentEpisodeReturn);
                        }
                        currentStep++;
                        if(terminated){
                            skip = true
                        }
                    } else {
                        skip = true
                    }
                    if(skip){
                        currentStep = 0;
                        currentEpisode++;
                        if(currentEpisode >= trajectoryData.length){
                            currentEpisode = 0;
                        }
                        console.log(`Episode ${currentEpisode} finished. Return = ${currentEpisodeReturn}, Length = ${currentEpisodeLength}`);
                        currentEpisodeLength = 0;
                        currentEpisodeReturn = 0;
                    }
                }
            }
            else{
                episode_info.innerHTML += `Step: ${currentStep}, Return (avg): ${current_episode_return_multi.toFixed(2)}`;
                current_parameters_multi = trajectoryData.map((episode) => episode.parameters);
                current_step_data_multi = trajectoryData.map((episode) => episode.trajectory[currentStep]);
                current_render_step_multi = currentStep;
                if(currentStep == 0) {
                    render()
                    if (this.options.telemetry && this.telem_initialized) {
                        const idx = Math.min(this.telem_uav_index, trajectoryData.length - 1);
                        this.loadEpisodeData(trajectoryData[idx]);
                        if (this.telem_header_label) {
                            this.telem_header_label.textContent = `UAV ${idx} / ${trajectoryData.length}`;
                        }
                    }
                }
                current_episode_return_multi += trajectoryData.reduce((acc, episode) => acc + episode.trajectory[currentStep].reward, 0) / trajectoryData.length;
                if (this.options.telemetry && this.telem_initialized) {
                    const idx = Math.min(this.telem_uav_index, trajectoryData.length - 1);
                    const s = trajectoryData[idx].trajectory[currentStep];
                    this.updateTelemetry(s.state, s.action, s.reward, currentStep, current_episode_return_multi);
                }
                currentStep++;
                if(currentStep >= trajectoryData[0].trajectory.length) {
                    currentStep = 0;
                    current_episode_return_multi = 0;
                }
            }
        }

        function previousEpisode() {
            currentStep = 0;
            currentEpisode--;
            if (currentEpisode < 0) {
                currentEpisode = trajectoryData.length - 1;
            }
            currentEpisodeLength = 0;
            currentEpisodeReturn = 0;
            console.log(`Skipped to Episode ${currentEpisode}`);
        }
        function restartEpisode() {
            currentStep = 0;
            currentEpisodeLength = 0;
            currentEpisodeReturn = 0;
            console.log(`Restarted Episode ${currentEpisode}`);
        }
        function skipEpisode() {
            currentStep = 0;
            currentEpisode++;
            if (currentEpisode >= trajectoryData.length) {
                currentEpisode = 0;
            }
            currentEpisodeLength = 0;
            currentEpisodeReturn = 0;
            console.log(`Skipped to Episode ${currentEpisode}`);
        }
        function togglePause() {
            isPaused = !isPaused;
            pause_button.innerHTML = isPaused ? "Resume" : "Pause";
            console.log(isPaused ? "Playback paused" : "Playback resumed");
        }

        previous_button.addEventListener('click', previousEpisode);
        restart_button.addEventListener('click', restartEpisode);
        pause_button.addEventListener('click', togglePause);
        skip_button.addEventListener('click', skipEpisode);


        setInterval(step, dt * 1000);
    }

}
