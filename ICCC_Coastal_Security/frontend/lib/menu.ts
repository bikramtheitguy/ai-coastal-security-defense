/* The ten primary menu groups (§6). `perm` gates visibility; the backend enforces the same rules. */
export type Sub = { v: string; label: string; perm?: string };
export type Group = { n: string; key: string; path: string; label: string; perm: string[]; subs: Sub[] };

export const MENU: Group[] = [
  {
    n: "01", key: "cop", path: "/cop", label: "Common Operating Picture", perm: ["COP_VIEW"],
    subs: [
      { v: "map", label: "Live Nautical Map" }, { v: "chart", label: "Static Nautical Chart" },
      { v: "layers", label: "Operational Layers" }, { v: "vessels", label: "Vessel Picture" },
      { v: "dark", label: "Dark / Unidentified Targets" }, { v: "toi", label: "Targets of Interest", perm: "INTEL_VIEW" },
      { v: "patrols", label: "Patrol Activity" }, { v: "drones", label: "Drone Activity" },
      { v: "incidents", label: "Incidents", perm: "INCIDENT_VIEW" }, { v: "weather", label: "Weather / Hazards" },
    ],
  },
  {
    n: "02", key: "readiness", path: "/readiness", label: "Readiness", perm: ["READINESS_VIEW"],
    subs: [
      { v: "state", label: "State Readiness" }, { v: "district", label: "District Readiness" },
      { v: "station", label: "MPS Readiness" }, { v: "personnel", label: "Personnel Readiness" },
      { v: "assets", label: "Asset Readiness" }, { v: "comms", label: "Communications Readiness" },
      { v: "surveillance", label: "Surveillance Readiness" }, { v: "maintenance", label: "Maintenance Deficiencies" },
      { v: "training", label: "Training Readiness" },
    ],
  },
  {
    n: "03", key: "personnel", path: "/personnel", label: "Personnel & Force", perm: ["PERSONNEL_VIEW"],
    subs: [
      { v: "directory", label: "Personnel Directory" }, { v: "deployment", label: "Current Deployment" },
      { v: "duty", label: "Duty Status" }, { v: "quals", label: "Qualifications" }, { v: "training", label: "Training" },
      { v: "sea", label: "Sea-Ready Personnel" }, { v: "crew", label: "Boat Crew Availability" },
      { v: "uav", label: "UAV Operators" }, { v: "sar", label: "SAR Qualifications" },
      { v: "deficiencies", label: "Personnel Deficiencies" },
    ],
  },
  {
    n: "04", key: "assets", path: "/assets", label: "Assets & Operations", perm: ["ASSET_VIEW"],
    subs: [
      { v: "boats", label: "Boats" }, { v: "uav", label: "UAVs" }, { v: "vehicles", label: "Vehicles" },
      { v: "comms", label: "Communications" }, { v: "surveillance", label: "Surveillance Assets" },
      { v: "planning", label: "Patrol Planning", perm: "PATROL_PLAN" }, { v: "patrols", label: "Active Patrols" },
      { v: "history", label: "Patrol History" }, { v: "fuel", label: "Fuel & Utilisation" },
      { v: "maintenance", label: "Maintenance" }, { v: "defects", label: "Defects" }, { v: "availability", label: "Asset Availability" },
    ],
  },
  {
    n: "05", key: "incidents", path: "/incidents", label: "Incident Command", perm: ["INCIDENT_VIEW"],
    subs: [
      { v: "alerts", label: "Active Alerts" }, { v: "incidents", label: "Incidents" }, { v: "queue", label: "Verification Queue" },
      { v: "tasking", label: "Tasking" }, { v: "dispatch", label: "Dispatch" }, { v: "tracking", label: "Response Tracking" },
      { v: "sar", label: "Search and Rescue" }, { v: "evidence", label: "Evidence" }, { v: "closure", label: "Closure" },
      { v: "aar", label: "After-Action Review" },
    ],
  },
  {
    n: "06", key: "intel", path: "/intel", label: "Maritime Intelligence", perm: ["INTEL_VIEW"],
    subs: [
      { v: "vessels", label: "Vessel Intelligence" }, { v: "search", label: "Vessel Search" }, { v: "history", label: "Vessel History" },
      { v: "behaviour", label: "Behaviour Analytics" }, { v: "AIS_LOST", label: "AIS Anomalies" },
      { v: "DARK_VESSEL", label: "Dark Vessels" }, { v: "IDENTITY_MISMATCH", label: "Identity Mismatch" },
      { v: "LOITERING", label: "Loitering" }, { v: "RENDEZVOUS", label: "Rendezvous" },
      { v: "ROUTE_DEVIATION", label: "Route Deviation" }, { v: "REPEATED_VISITS", label: "Repeated Visits" },
      { v: "toi", label: "Targets of Interest" }, { v: "watchlists", label: "Watch Lists" },
      { v: "community", label: "Community Intelligence" }, { v: "fusion", label: "Multi-Source Correlation" },
    ],
  },
  {
    n: "07", key: "assistant", path: "/assistant", label: "AI Maritime Public Assistant", perm: ["CHAT_VIEW"],
    subs: [
      { v: "conversations", label: "Citizen Conversations" }, { v: "distress", label: "Distress Queue" },
      { v: "reporting", label: "Incident Reporting" }, { v: "missing_vessel", label: "Missing Vessel" },
      { v: "missing_fisherman", label: "Missing Fisherman" }, { v: "suspicious", label: "Suspicious Activity" },
      { v: "assistance", label: "Public Assistance" }, { v: "review", label: "Conversation Review" },
      { v: "takeover", label: "Human Operator Takeover" }, { v: "handoff", label: "MRCC/MRSC Handoff" },
      { v: "analytics", label: "Chatbot Analytics" }, { v: "kb", label: "Knowledge Base" },
    ],
  },
  {
    n: "08", key: "command", path: "/command", label: "Command & Tasking", perm: ["ORDERS_ISSUE", "TASK_ASSETS", "ORDERS_FIELD"],
    subs: [
      { v: "desk", label: "Command Desk", perm: "ORDERS_ISSUE" }, { v: "alerts", label: "Operational Alerts" },
      { v: "movement", label: "Asset Movement Orders" }, { v: "personnel", label: "Personnel Tasking" },
      { v: "response", label: "Incident Response Orders" }, { v: "acks", label: "Command Acknowledgements" },
      { v: "active", label: "Active Orders" }, { v: "completed", label: "Completed Orders" },
      { v: "field", label: "Field Unit Console", perm: "ORDERS_FIELD" }, { v: "uav", label: "UAV Mission Console", perm: "ORDERS_FIELD" },
    ],
  },
  {
    n: "09", key: "analytics", path: "/analytics", label: "Analytics & System Health",
    perm: ["ANALYTICS_VIEW", "CYBER_VIEW", "AUDIT_VIEW"],
    subs: [
      { v: "leadership", label: "Leadership Dashboard", perm: "ANALYTICS_VIEW" }, { v: "patrols", label: "Patrol Effectiveness", perm: "ANALYTICS_VIEW" },
      { v: "response", label: "Response Times", perm: "ANALYTICS_VIEW" }, { v: "trends", label: "Incident Trends", perm: "ANALYTICS_VIEW" },
      { v: "behaviour", label: "Vessel Behaviour", perm: "ANALYTICS_VIEW" }, { v: "utilisation", label: "Resource Utilisation", perm: "ANALYTICS_VIEW" },
      { v: "cyber", label: "Cybersecurity", perm: "CYBER_VIEW" }, { v: "network", label: "Network Health", perm: "CYBER_VIEW" },
      { v: "sources", label: "Data Source Health" }, { v: "audit", label: "Audit Logs", perm: "AUDIT_VIEW" },
      { v: "backup", label: "Backup / Disaster Recovery", perm: "BACKUP" }, { v: "scenarios", label: "Exercise Scenarios", perm: "SCENARIO_RUN" },
    ],
  },
  {
    n: "10", key: "admin", path: "/admin", label: "Administration", perm: ["ADMIN_MASTER", "ADMIN_USERS", "ADMIN_CONFIG"],
    subs: [
      { v: "personnel", label: "Personnel Master", perm: "PERSONNEL_EDIT" }, { v: "assets", label: "Asset Master", perm: "ASSET_EDIT" },
      { v: "stations", label: "Marine Police Stations" }, { v: "ranks", label: "Ranks" }, { v: "roles", label: "Roles" },
      { v: "qualifications", label: "Qualifications" }, { v: "courses", label: "Training Courses" },
      { v: "zones", label: "Geographic Zones" }, { v: "sources", label: "Data Sources" },
      { v: "users", label: "Users", perm: "ADMIN_USERS" }, { v: "access", label: "Access Control", perm: "ADMIN_USERS" },
      { v: "alertrules", label: "Alert Rules", perm: "ADMIN_CONFIG" }, { v: "riskweights", label: "Risk Weights", perm: "ADMIN_CONFIG" },
      { v: "config", label: "System Configuration", perm: "ADMIN_CONFIG" }, { v: "audit", label: "Audit", perm: "AUDIT_VIEW" },
    ],
  },
];
