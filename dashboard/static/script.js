let currentDb = "";
let domainChart = null;

document.addEventListener("DOMContentLoaded", () => {
    init();
});

async function init() {
    await loadDatabases();
    setupEventListeners();
    if (currentDb) {
        await refreshData();
    }
}

async function loadDatabases() {
    try {
        const response = await fetch("/api/dbs");
        const data = await response.json();
        const select = document.getElementById("db_select");
        select.innerHTML = "";
        
        if (data.databases.length === 0) {
            select.innerHTML = '<option value="">No DBs found</option>';
            return;
        }

        data.databases.forEach(db => {
            const option = document.createElement("option");
            option.value = db;
            option.textContent = db;
            select.appendChild(option);
        });

        currentDb = data.databases[0];
        document.getElementById("current_db_display").textContent = `Database: ${currentDb}`;
    } catch (error) {
        console.error("Error loading databases:", error);
    }
}

function setupEventListeners() {
    // DB Selection
    document.getElementById("db_select").addEventListener("change", (e) => {
        currentDb = e.target.value;
        document.getElementById("current_db_display").textContent = `Database: ${currentDb}`;
        refreshData();
    });

    // Refresh Button
    document.getElementById("refresh_btn").addEventListener("click", refreshData);

    // Navigation
    document.querySelectorAll(".nav-links li").forEach(li => {
        li.addEventListener("click", () => {
            const view = li.dataset.view;
            switchView(view);
            document.querySelectorAll(".nav-links li").forEach(l => l.classList.remove("active"));
            li.classList.add("active");
        });
    });

    // Run Search
    document.getElementById("run_search").addEventListener("input", (e) => {
        filterRuns(e.target.value);
    });
}

function switchView(viewId) {
    document.querySelectorAll(".view").forEach(v => v.classList.add("hidden"));
    document.getElementById(`view_${viewId}`).classList.remove("hidden");
    document.getElementById("view_title").textContent = viewId.charAt(0).toUpperCase() + viewId.slice(1);
}

async function refreshData() {
    if (!currentDb) return;
    
    // Show loading states
    document.getElementById("stat_total_runs").textContent = "...";
    
    try {
        const statsResponse = await fetch(`/api/stats/${currentDb}`);
        const stats = await statsResponse.json();
        
        updateOverview(stats);
        await loadRuns();
    } catch (error) {
        console.error("Error refreshing data:", error);
    }
}

function updateOverview(stats) {
    document.getElementById("stat_total_runs").textContent = stats.total_runs.toLocaleString();
    document.getElementById("stat_total_readings").textContent = stats.total_readings.toLocaleString();
    document.getElementById("stat_pass_rate").textContent = `${stats.pass_rate.toFixed(1)}%`;

    updateDomainChart(stats.domain_distribution);
    updateLatestRuns(stats.latest_runs);
}

function updateDomainChart(dist) {
    const ctx = document.getElementById("domainChart").getContext("2d");
    
    if (domainChart) {
        domainChart.destroy();
    }

    const labels = Object.keys(dist);
    const data = Object.values(dist);

    domainChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: ['#38bdf8', '#818cf8', '#6366f1'],
                borderWidth: 0,
                hoverOffset: 10
            }]
        },
        options: {
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { family: 'Inter' } }
                }
            },
            cutout: '70%'
        }
    });
}

function updateLatestRuns(runs) {
    const list = document.getElementById("latest_runs_list");
    list.innerHTML = "";
    
    runs.forEach(run => {
        const item = document.createElement("div");
        item.className = "mini-item";
        const status = run.fault_count === 0 ? "PASSED" : `${run.fault_count} FAULTS`;
        const color = run.fault_count === 0 ? "#4ade80" : "#f87171";
        
        item.innerHTML = `
            <div class="mini-item-info">
                <strong>${run.scenario}</strong>
                <span>${run.run_id.substring(0, 8)}...</span>
            </div>
            <span style="color: ${color}; font-size: 0.75rem; font-weight: 700;">${status}</span>
        `;
        list.appendChild(item);
    });
}

async function loadRuns() {
    try {
        const response = await fetch(`/api/runs/${currentDb}`);
        const data = await response.json();
        renderRunsTable(data.runs);
    } catch (error) {
        console.error("Error loading runs:", error);
    }
}

function renderRunsTable(runs) {
    const tbody = document.querySelector("#runs_table tbody");
    tbody.innerHTML = "";
    
    runs.forEach(run => {
        const tr = document.createElement("tr");
        const statusClass = run.fault_count === 0 ? "badge-pass" : "badge-fail";
        const statusText = run.fault_count === 0 ? "PASS" : "FAIL";
        
        tr.innerHTML = `
            <td style="font-family: monospace; font-size: 0.8rem;">${run.run_id.substring(0, 12)}...</td>
            <td>${run.domain}</td>
            <td>${run.scenario}</td>
            <td>${run.total_readings.toLocaleString()}</td>
            <td>${run.fault_count}</td>
            <td><span class="status-badge ${statusClass}">${statusText}</span></td>
            <td><button class="premium-button" style="padding: 0.2rem 0.6rem; font-size: 0.7rem;" onclick="viewRunDetails('${run.run_id}')">VIEW</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function filterRuns(query) {
    const rows = document.querySelectorAll("#runs_table tbody tr");
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query.toLowerCase()) ? "" : "none";
    });
}

async function viewRunDetails(runId) {
    const modal = document.getElementById("run_modal");
    modal.classList.remove("hidden");
    
    document.getElementById("modal_run_id").textContent = `Run: ${runId.substring(0, 16)}...`;
    
    try {
        const response = await fetch(`/api/run_details/${currentDb}/${runId}`);
        const data = await response.json();
        
        document.getElementById("modal_meta_domain").textContent = `Domain: ${data.metadata.domain}`;
        document.getElementById("modal_meta_scenario").textContent = `Scenario: ${data.metadata.scenario}`;
        
        // Show a simple chart of values for the first sensor found
        if (data.logs.length > 0) {
            updateRunDetailChart(data.logs);
        }
    } catch (error) {
        console.error("Error loading run details:", error);
    }
}

let detailChart = null;
function updateRunDetailChart(logs) {
    const ctx = document.getElementById("runDetailChart").getContext("2d");
    if (detailChart) detailChart.destroy();
    
    // Group by timestamp for chart
    const labels = logs.map(l => new Date(l.timestamp * 1000).toLocaleTimeString());
    const values = logs.map(l => l.value);

    detailChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sensor Value (Sample)',
                data: values,
                borderColor: '#38bdf8',
                tension: 0.3,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, border: { display: false } },
                x: { display: false }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// Modal Close
document.getElementById("close_modal").onclick = () => {
    document.getElementById("run_modal").classList.add("hidden");
};
window.onclick = (event) => {
    const modal = document.getElementById("run_modal");
    if (event.target == modal) {
        modal.classList.add("hidden");
    }
};
