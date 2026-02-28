import { useState } from "react";
import App from "./App.jsx";
import PokerTable from "./PokerTable.jsx";

const TABS = [
  { id: "play", label: "Play" },
  { id: "charts", label: "Charts" },
];

export default function Root() {
  const [tab, setTab] = useState("play");

  return (
    <div className="root-shell">
      <nav className="tab-nav">
        {TABS.map((item) => (
          <button
            key={item.id}
            className={`tab-btn ${tab === item.id ? "is-active" : ""}`}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      {tab === "play" ? <PokerTable /> : <App />}
    </div>
  );
}
