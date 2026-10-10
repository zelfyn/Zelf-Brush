import { cancelJob } from "../actions";
import { store, useApp } from "../store";

export function QueuePanel() {
  const jobs = useApp((s) => s.jobs);
  const list = Object.values(jobs).filter((j) => j.lane === "export").reverse();
  return (
    <div className="queue" role="region" aria-label="Export queue">
      <h4>Export queue <button style={{ padding: "0 8px" }} onClick={() => store.set({ queueOpen: false })}>Close</button></h4>
      {list.length === 0 && <div className="qrow"><span className="st">Nothing queued.</span></div>}
      {list.map((j) => (
        <div className="qrow" key={j.id}>
          <span>{j.title}</span>
          <span className={`st ${j.status}`}>
            {j.status === "running" ? `${Math.round(j.progress * 100)}%` : j.status === "waiting" ? "Waiting" : j.status === "completed" ? "Completed" : j.status === "failed" ? "Failed" : "Cancelled"}
            {(j.status === "running" || j.status === "waiting") && <> <button style={{ padding: "0 6px" }} onClick={() => void cancelJob(j.id)}>Cancel</button></>}
          </span>
          {j.status === "running" && <span className="bar"><i style={{ width: `${j.progress * 100}%` }} /></span>}
          {j.status === "running" && <span className="st" style={{ gridColumn: "1 / -1" }}>{j.message}</span>}
          {j.status === "failed" && j.error && <span className="st failed" style={{ gridColumn: "1 / -1" }}>{j.error.message}</span>}
        </div>
      ))}
    </div>
  );
}
