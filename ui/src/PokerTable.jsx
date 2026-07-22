import { useCallback, useEffect, useState } from "react";

const SUIT = {
  s: { symbol: "♠", color: "spade" },
  h: { symbol: "♥", color: "heart" },
  d: { symbol: "♦", color: "diamond" },
  c: { symbol: "♣", color: "club" },
};

function Card({ code, hidden }) {
  if (hidden) {
    return <div className="card card-back" aria-label="face-down card" />;
  }
  if (!code) {
    return <div className="card card-empty" />;
  }
  const rank = code.slice(0, -1);
  const suit = code.slice(-1);
  const meta = SUIT[suit] ?? { symbol: suit, color: "spade" };
  return (
    <div className={`card card-${meta.color}`}>
      <span className="card-rank">{rank}</span>
      <span className="card-suit">{meta.symbol}</span>
    </div>
  );
}

function Seat({ label, cards, hidden, stack, isTurn, isHuman }) {
  return (
    <div className={`seat ${isHuman ? "seat-human" : "seat-bot"} ${isTurn ? "seat-active" : ""}`}>
      <div className="seat-header">
        <span className="seat-name">{label}</span>
        <span className="seat-stack">{stack} chips</span>
      </div>
      <div className="seat-cards">
        {(cards ?? [null, null]).map((code, idx) => (
          <Card key={idx} code={code} hidden={hidden} />
        ))}
      </div>
    </div>
  );
}

const streetLabel = (street) => (street ? street[0].toUpperCase() + street.slice(1) : "");

function ActionLog({ log }) {
  if (!log?.length) return null;
  return (
    <div className="action-log">
      <h3>Action</h3>
      <ol>
        {log.map((event, idx) => {
          const who = event.actor === "human" ? "You" : "Bot";
          const verb = event.actionType;
          const amount = event.amount > 0 ? ` ${event.amount}` : "";
          return (
            <li key={idx} className={event.actor}>
              <span className="log-street">{streetLabel(event.street)}</span>
              <span className="log-text">
                {who} {verb}
                {amount}
              </span>
              {event.untrained ? (
                <span className="log-untrained" title="Bot had no trained strategy here — random fallback">
                  untrained
                </span>
              ) : null}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export default function PokerTable() {
  const [view, setView] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const deal = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/game/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) throw new Error(`deal failed (${res.status})`);
      setView(await res.json());
    } catch (err) {
      setError(String(err.message ?? err));
    } finally {
      setBusy(false);
    }
  }, []);

  const act = useCallback(
    async (actionId) => {
      if (!view?.sessionId) return;
      setBusy(true);
      setError("");
      try {
        const res = await fetch(`/api/game/${view.sessionId}/action`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ actionId }),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail ?? `action failed (${res.status})`);
        }
        setView(await res.json());
      } catch (err) {
        setError(String(err.message ?? err));
      } finally {
        setBusy(false);
      }
    },
    [view]
  );

  useEffect(() => {
    deal();
  }, [deal]);

  if (!view) {
    return (
      <div className="table-shell">
        {error ? <div className="error-banner">{error}</div> : <div className="loading-state">Dealing…</div>}
      </div>
    );
  }

  const youAreButton = view.humanSeat === view.button;
  const botSeatIsButton = !youAreButton;
  const result = view.result;
  const untrainedNote =
    view.botDecisions > 0
      ? `${view.botUntrainedDecisions}/${view.botDecisions} bot decisions were untrained`
      : null;

  return (
    <div className="table-shell">
      <header className="table-bar">
        <div>
          <p className="kicker">Play the Blueprint</p>
          <h1>Heads-Up No-Limit</h1>
        </div>
        <div className="run-chip">
          <span>Run</span>
          <strong>{view.runId}</strong>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="felt">
        <Seat
          label={`Bot ${botSeatIsButton ? "(BTN)" : "(BB)"}`}
          cards={view.botHole}
          hidden={!view.botHole}
          stack={view.stacks[1 - view.humanSeat]}
          isTurn={!view.isOver && view.currentPlayer !== view.humanSeat}
          isHuman={false}
        />

        <div className="board-row">
          <div className="pot-pill">Pot {view.pot}</div>
          <div className="board">
            {[0, 1, 2, 3, 4].map((idx) => (
              <Card key={idx} code={view.board[idx]} />
            ))}
          </div>
          <div className="street-pill">{streetLabel(view.street)}</div>
        </div>

        <Seat
          label={`You ${youAreButton ? "(BTN)" : "(BB)"}`}
          cards={view.yourHole}
          hidden={false}
          stack={view.stacks[view.humanSeat]}
          isTurn={!view.isOver && view.currentPlayer === view.humanSeat}
          isHuman
        />
      </div>

      <div className="controls">
        {result ? (
          <div className={`result-banner result-${result.outcome}`}>
            <strong>
              {result.outcome === "win" ? "You win" : result.outcome === "loss" ? "You lose" : "Split pot"}
            </strong>
            <span>
              {result.humanPayoff >= 0 ? "+" : ""}
              {result.humanPayoff} chips · {result.terminal}
            </span>
            <button className="primary-btn" onClick={deal} disabled={busy}>
              Next hand
            </button>
          </div>
        ) : view.yourTurn ? (
          <div className="action-bar-row">
            {view.legalActions.map((action) => (
              <button
                key={action.id}
                className={`action-btn action-${action.kind}`}
                onClick={() => act(action.id)}
                disabled={busy}
              >
                {action.label}
              </button>
            ))}
          </div>
        ) : (
          <div className="waiting">{busy ? "…" : "Bot is thinking…"}</div>
        )}
      </div>

      {untrainedNote ? <div className="untrained-note">{untrainedNote}</div> : null}
      <ActionLog log={view.log} />
    </div>
  );
}
