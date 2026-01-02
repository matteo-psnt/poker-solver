import { useEffect, useMemo, useState } from "react";

const META_URL = "/api/meta";
const chartUrl = (position, situation) =>
  `/api/chart?position=${position}&situation=${encodeURIComponent(situation)}`;

const buildPalette = (actions) => {
  const palette = {};
  const aggressive = actions.filter((action) => action.kind === "aggressive");
  const aggressiveCount = aggressive.length;

  aggressive.forEach((action, idx) => {
    const t = aggressiveCount > 1 ? idx / (aggressiveCount - 1) : 0.5;
    const hue = 170 - 45 * t;
    const sat = 62 + 8 * t;
    const light = 38 + 12 * t;
    palette[action.id] = `hsl(${hue.toFixed(0)}, ${sat.toFixed(0)}%, ${light.toFixed(
      0
    )}%)`;
  });

  actions.forEach((action) => {
    if (action.kind === "call") {
      palette[action.id] = "hsl(205, 64%, 50%)";
    }
    if (action.kind === "fold") {
      palette[action.id] = "hsl(6, 70%, 52%)";
    }
    if (action.kind === "all-in") {
      palette[action.id] = "hsl(31, 82%, 52%)";
    }
  });

  return palette;
};

const orderActions = (actions) => {
  const aggressive = actions
    .filter((action) => action.kind === "aggressive")
    .sort((a, b) => (a.sizeBb ?? 0) - (b.sizeBb ?? 0));
  const allIn = actions.filter((action) => action.kind === "all-in");
  const call = actions.filter((action) => action.kind === "call");
  const fold = actions.filter((action) => action.kind === "fold");
  return [...aggressive, ...allIn, ...call, ...fold];
};

const formatPct = (pct) => `${pct.toFixed(1)}%`;

const getActionSummary = (actions, metaById) => {
  if (!actions.length) return "No data";
  return actions
    .map((action) => `${metaById[action.id]?.label ?? action.id}: ${formatPct(action.pct)}`)
    .join(" | ");
};

export default function App() {
  const [meta, setMeta] = useState(null);
  const [chart, setChart] = useState(null);
  const [position, setPosition] = useState(0);
  const [situation, setSituation] = useState("first_to_act");
  const [selectedHand, setSelectedHand] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    fetch(META_URL)
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        setMeta(data);
        setPosition(data.defaultPosition ?? 0);
        setSituation(data.defaultSituation ?? "first_to_act");
      })
      .catch(() => {
        if (!active) return;
        setError("Unable to load chart metadata.");
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!meta) return;
    let active = true;
    setLoading(true);
    setError("");
    fetch(chartUrl(position, situation))
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        setChart(data);
        setSelectedHand(null);
        setLoading(false);
      })
      .catch(() => {
        if (!active) return;
        setLoading(false);
        setError("Unable to load chart data.");
      });
    return () => {
      active = false;
    };
  }, [meta, position, situation]);

  const orderedActions = useMemo(() => {
    if (!chart?.actions) return [];
    return orderActions(chart.actions);
  }, [chart]);

  const palette = useMemo(() => {
    if (!chart?.actions) return {};
    return buildPalette(chart.actions);
  }, [chart]);

  const actionMetaById = useMemo(() => {
    if (!chart?.actions) return {};
    return chart.actions.reduce((acc, action) => {
      acc[action.id] = action;
      return acc;
    }, {});
  }, [chart]);

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div>
          <p className="kicker">Poker Solver Viewer</p>
          <h1>Preflop Strategy Matrix</h1>
        </div>
        <div className="run-chip">
          <span>Run</span>
          <strong>{meta?.runId ?? "Loading..."}</strong>
        </div>
      </header>

      <section className="panel-grid">
        <aside className="panel panel-controls">
          <div className="panel-header">
            <h2>Scenario</h2>
            <p>Swap positions and situations without leaving the viewer.</p>
          </div>
          <label className="field">
            <span>Position</span>
            <select
              value={position}
              onChange={(event) => setPosition(Number(event.target.value))}
            >
              {meta?.positions?.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Situation</span>
            <select value={situation} onChange={(event) => setSituation(event.target.value)}>
              {meta?.situations?.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          {loading ? <div className="status-text">Loading chart...</div> : null}
        </aside>

        <main className="panel panel-chart">
          <div className="panel-header">
            <div>
              <h2>{chart?.positionLabel ?? "Loading..."}</h2>
              <p>{chart?.situationLabel ?? " "} • 200bb depth</p>
            </div>
            <div className="legend">
              {orderedActions.map((action) => (
                <div key={action.id} className="legend-item">
                  <span
                    className="legend-swatch"
                    style={{ background: palette[action.id] }}
                  />
                  <span>{action.label}</span>
                </div>
              ))}
              <div className="legend-item">
                <span className="legend-swatch muted" />
                <span>No data</span>
              </div>
            </div>
          </div>

          {error && <div className="error-banner">{error}</div>}

          <div className={`chart-wrapper ${loading ? "is-loading" : ""}`}>
            {chart?.grid ? (
              <div
                className="chart-grid"
                style={{
                  "--cols": chart.ranks.length + 1,
                }}
              >
                <div className="grid-corner" />
                {chart.ranks.split("").map((rank) => (
                  <div key={`col-${rank}`} className="grid-label">
                    {rank}
                  </div>
                ))}
                {chart.grid.map((row, rowIndex) => (
                  <div key={`row-${rowIndex}`} className="grid-row">
                    <div className="grid-label">{chart.ranks[rowIndex]}</div>
                    {row.map((cell) => {
                      const cellActions = cell.actions || [];
                      let cursor = 0;
                      const stops = orderedActions.flatMap((action) => {
                        const cellAction = cellActions.find((item) => item.id === action.id);
                        if (!cellAction || cellAction.pct <= 0) return [];
                        const start = cursor;
                        const end = cursor + cellAction.pct;
                        cursor = end;
                        const color = palette[action.id] ?? "#8e8e8e";
                        return [`${color} ${start}%`, `${color} ${end}%`];
                      });
                      const background = stops.length
                        ? `linear-gradient(90deg, ${stops.join(", ")})`
                        : "repeating-linear-gradient(135deg, #8b8b8b 0 6px, #5f5f5f 6px 12px)";

                      return (
                        <button
                          key={cell.hand}
                          className="hand-cell"
                          style={{ background }}
                          onClick={() => setSelectedHand(cell)}
                          title={getActionSummary(cellActions, actionMetaById)}
                        >
                          <span>{cell.hand}</span>
                        </button>
                      );
                    })}
                  </div>
                ))}
              </div>
            ) : (
              <div className="loading-state">Waiting on chart data…</div>
            )}
          </div>
        </main>

        <aside className="panel panel-details">
          <div className="panel-header">
            <h2>Hand Detail</h2>
            <p>Click any hand to lock the breakdown.</p>
          </div>
          {selectedHand ? (
            <div className="hand-detail">
              <div className="hand-header">
                <span>{selectedHand.hand}</span>
                <strong>{selectedHand.actions.length ? "Strategy Mix" : "No data"}</strong>
              </div>
              {selectedHand.actions.length ? (
                <div className="action-list">
                  {selectedHand.actions.map((action) => (
                    <div key={action.id} className="action-row">
                      <div className="action-label">
                        <span
                          className="legend-swatch"
                          style={{ background: palette[action.id] }}
                        />
                        <span>{actionMetaById[action.id]?.label ?? action.id}</span>
                      </div>
                      <div className="action-bar">
                        <div
                          className="action-bar-fill"
                          style={{
                            width: `${action.pct}%`,
                            background: palette[action.id],
                          }}
                        />
                      </div>
                      <strong>{formatPct(action.pct)}</strong>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state">No strategy stored for this combo.</div>
              )}
            </div>
          ) : (
            <div className="empty-state">Select a hand to inspect its mix.</div>
          )}
        </aside>
      </section>
    </div>
  );
}
