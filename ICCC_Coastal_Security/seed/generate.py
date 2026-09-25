"""Deterministic synthetic dataset generator — ALL DATA IS SIMULATED / POC DATA.

Generates relational demo data: 6 districts, 18 illustrative Marine Police Stations,
~170 personnel with qualifications and training, boats / trawlers / RWCs / UAVs /
vehicles / comms / sensors with crews, maintenance and defects, map reference places,
simulated zones, ~150 vessels with tracks, alerts, incident history, patrol history,
weather, data-source registry, knowledge base, configuration and demo user accounts.

No real personnel, credentials, deployments or intelligence are represented.
"""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from app.models import (Alert, Asset, ConfigItem, CrewAssignment, CyberEvent, DataSource, Defect, District, EventFeed,
                        Incident, IncidentEvent, KnowledgeArticle, MaintenanceRecord, Mission, Personnel,
                        PersonnelQualification, Place, QualificationType, Rank, Station, TrainingCourse,
                        TrainingRecord, User, Vessel, VesselTrackPoint, WatchListEntry, WeatherReport, Zone, utcnow)
from app.security import hash_password
from app.services.common import default_config_items
from app.services.geo import bearing_deg, move

from .geography import COASTLINE, DISTRICTS, PORTS, SEAWARD, STATIONS

FIRST = ["Ramesh", "Suresh", "Prasanta", "Bijay", "Sanjay", "Debasis", "Manoj", "Pradeep", "Ashok", "Sarat", "Bikash",
         "Ranjan", "Subrat", "Sasmita", "Pragyan", "Itishree", "Rashmita", "Sujata", "Laxmidhar", "Gopinath",
         "Jagannath", "Niranjan", "Satya", "Tapan", "Dillip", "Akshaya", "Chittaranjan", "Kishore", "Rajendra",
         "Santosh", "Biswajit", "Soumya", "Anita", "Mamata", "Puspanjali", "Madhusmita", "Hemanta", "Sukanta",
         "Pabitra", "Umakanta", "Bhagaban", "Kalandi", "Sibaram", "Trilochan", "Nirmala", "Jyotirmayee", "Amiya",
         "Deepak", "Rabindra", "Sudhir"]
LAST = ["Mohanty", "Das", "Sahoo", "Behera", "Nayak", "Pradhan", "Swain", "Rout", "Panda", "Mishra", "Jena", "Parida",
        "Sethi", "Biswal", "Mallick", "Samal", "Barik", "Senapati", "Dash", "Tripathy", "Pattnaik", "Sahu", "Muduli",
        "Bhoi", "Majhi", "Lenka", "Khuntia", "Mahapatra"]
BOAT_NAMES = ["Maa Tarini", "Jay Jagannath", "Maa Mangala", "Sagar Kanya", "Maa Kalijai", "Baba Lokanath",
              "Maa Samaleswari", "Sri Ganesh", "Maa Bhagabati", "Jay Maa Durga", "Sagar Rani", "Maa Ramachandi",
              "Hanuman", "Maa Biraja", "Nilachakra", "Maa Harachandi", "Sagar Deep", "Maa Sarala", "Jaya Laxmi",
              "Om Sai", "Maa Chandi", "Sri Krishna", "Maa Tara", "Bhai Bhai", "Sagar Mitra", "Samudra Devi",
              "Maa Khambeswari", "Gangadevi", "Maa Kichakeswari", "Jay Hanuman"]
MERCHANT = ["MV Ocean Pearl (SIM)", "MV Eastern Star (SIM)", "MT Coastal Spirit (SIM)", "MV Bay Trader (SIM)",
            "MV Kalinga Bulk (SIM)", "MT Sagar Flame (SIM)", "MV Delta Carrier (SIM)", "MV Indigo Wave (SIM)"]

RANKS = [("DGP", "Director General of Police", 12), ("ADGP", "Additional Director General of Police", 11),
         ("IG", "Inspector General of Police", 10), ("DIG", "Deputy Inspector General of Police", 9),
         ("SP", "Superintendent of Police", 8), ("ADDL_SP", "Additional Superintendent of Police", 7),
         ("DSP", "Deputy Superintendent of Police", 6), ("INSP", "Inspector of Police", 5),
         ("SI", "Sub-Inspector of Police", 4), ("ASI", "Assistant Sub-Inspector of Police", 3),
         ("HAV", "Havildar", 2), ("CONST", "Constable", 1), ("CIV", "Civilian Staff", 0)]
QUALS = [("BOAT_CREW", "Boat crew qualification", 36), ("NAVIGATION", "Navigation / boat master", 36),
         ("MARINE_VHF", "Marine VHF operator", 60), ("UAV_PILOT", "UAV remote pilot", 24),
         ("SWIMMING", "Swimming proficiency", 12), ("SEA_SURVIVAL", "Personal survival at sea", 24),
         ("SAR", "Search and Rescue", 24), ("FIRST_AID", "First aid / basic life support", 24),
         ("NIGHT_OPS", "Night maritime operations", 24), ("WEAPONS", "Weapons handling", 12),
         ("CYBER_IT", "Cyber / IT systems", 36)]
COURSES = [("CRS-BCT", "Basic Coastal Security Training", "BOAT_CREW", 21, 36),
           ("CRS-NAV", "Boat Master / Coastal Navigation Course", "NAVIGATION", 28, 36),
           ("CRS-VHF", "Marine VHF Radio Operator Course", "MARINE_VHF", 5, 60),
           ("CRS-UAV", "UAV Remote Pilot Training", "UAV_PILOT", 10, 24),
           ("CRS-SWM", "Swimming & Water Confidence", "SWIMMING", 5, 12),
           ("CRS-PSS", "Personal Survival at Sea", "SEA_SURVIVAL", 5, 24),
           ("CRS-SAR", "Maritime Search and Rescue", "SAR", 10, 24),
           ("CRS-FA", "First Aid & BLS", "FIRST_AID", 3, 24),
           ("CRS-NOP", "Night Maritime Operations", "NIGHT_OPS", 7, 24),
           ("CRS-WPN", "Weapons Refresher", "WEAPONS", 3, 12),
           ("CRS-CYB", "Cyber Hygiene & ICCC Systems", "CYBER_IT", 3, 36)]


class Gen:
    def __init__(self, db: Session, seed: int, demo_password: str, pbkdf2_iterations: int | None = None):
        self.db, self.r = db, random.Random(seed)
        self.now = utcnow()
        self.today = date.today()
        self.pw = demo_password
        self.iters = pbkdf2_iterations
        self.st: dict[str, Station] = {}
        self.dist: dict[str, District] = {}
        self.people_by_station: dict[int, list[Personnel]] = {}
        self.pid_n = 0

    # ------------------------------------------------------------ helpers
    def name(self) -> str:
        return f"{self.r.choice(FIRST)} {self.r.choice(LAST)}"

    def sea_point(self, st: Station, dmin: float, dmax: float, spread: float = 35):
        code = next(k for k, v in self.dist.items() if v.id == st.district_id)
        brg = SEAWARD[code] + self.r.uniform(-spread, spread)
        return move(st.lat, st.lon, brg, self.r.uniform(dmin, dmax))

    def add(self, obj):
        self.db.add(obj)
        return obj

    # ------------------------------------------------------------ build
    def run(self) -> dict:
        self.reference()
        self.geography()
        self.personnel()
        self.assets()
        self.vessels()
        self.missions()
        self.incident_history()
        self.system()
        self.users()
        self.db.flush()
        return {"stations": len(self.st), "personnel": self.pid_n}

    def reference(self):
        for c, n, lvl in RANKS:
            self.add(Rank(code=c, name=n, level=lvl))
        for c, n, m in QUALS:
            self.add(QualificationType(code=c, name=n, validity_months=m))
        for c, n, g, d, rm in COURSES:
            self.add(TrainingCourse(code=c, name=n, grants_qualification=g, duration_days=d, refresher_months=rm))
        for k, v in default_config_items().items():
            cat = "RISK_WEIGHT" if k.startswith("risk") or k.startswith("recommend") else \
                "READINESS" if k.startswith("readiness") else "ALERT_RULE" if k.startswith("alert") else "SYSTEM"
            self.add(ConfigItem(key=k, value=v, category=cat, description=f"Default {k} (POC)", updated_by="seed"))

    def geography(self):
        for code, name, lat, lon in DISTRICTS:
            self.dist[code] = self.add(District(code=code, name=name, lat=lat, lon=lon))
        self.db.flush()
        for i, (code, name, dcode, lat, lon, aliases) in enumerate(STATIONS):
            st = self.add(Station(code=code, name=name, district_id=self.dist[dcode].id, lat=lat, lon=lon,
                                  phone=f"0000-{600100 + i:06d} (SIM)", min_sea_ready=4, min_boats_ready=1,
                                  backup_vhf_last_test=self.today - timedelta(days=self.r.randint(3, 25)),
                                  backup_vhf_test_interval_days=30))
            self.st[name] = st
        self.db.flush()
        self.st["Dhamra"].backup_vhf_last_test = self.today - timedelta(days=41)  # overdue (explainability demo)
        self.st["Kharinashi"].network_backup = "OFFLINE"
        self.st["Sonapur"].vhf_base_status = "DEGRADED"
        prov = dict(source="SIMULATED", classification="RESTRICTED (SIMULATED)", verification="SYSTEM")
        n = 0
        for code, name, dcode, lat, lon, aliases in STATIONS:
            st = self.st[name]
            flat, flon = move(lat, lon, 200, 0.6)
            self.add(Place(code=f"FLC-{code[4:]}", name=f"{name} Fish Landing Centre", place_type="FISH_LANDING_CENTRE",
                           district_id=st.district_id, station_id=st.id, lat=flat, lon=flon,
                           attributes={"aliases": aliases + [name], "boats_registered": self.r.randint(60, 400)}, **prov))
            self.add(Place(code=f"CCTV-{code[4:]}", name=f"{name} FLC CCTV", place_type="CCTV",
                           district_id=st.district_id, station_id=st.id, lat=flat + 0.002, lon=flon + 0.002,
                           attributes={"cameras": self.r.randint(2, 6)}, **prov))
            jlat, jlon = move(lat, lon, 20, 0.5)
            self.add(Place(code=f"JTY-{code[4:]}", name=f"{name} Jetty", place_type="JETTY", district_id=st.district_id,
                           station_id=st.id, lat=jlat, lon=jlon, attributes={}, **prov))
            vlat, vlon = move(lat, lon, self.r.choice([30, 210]), self.r.uniform(3, 7))
            n += 1
            self.add(Place(code=f"VLP-{n:02d}", name=f"Vulnerable Landing Point VLP-{n:02d} (SIMULATED)",
                           place_type="VULNERABLE_LANDING", district_id=st.district_id, station_id=st.id,
                           lat=vlat, lon=vlon, attributes={"note": "Generic placeholder, not a real assessment"}, **prov))
            clat, clon = move(lat, lon, 270, 1.5)
            self.add(Place(code=f"MCS-{code[4:]}", name=f"Cyclone Shelter near {name} (SIMULATED)", place_type="CYCLONE_SHELTER",
                           district_id=st.district_id, station_id=st.id, lat=clat, lon=clon,
                           attributes={"capacity": self.r.choice([500, 800, 1000, 1500])}, **prov))
            tlat, tlon = move(lat, lon, 300, 0.8)
            self.add(Place(code=f"CT-{code[4:]}", name=f"{name} Communication Tower", place_type="COMM_TOWER",
                           district_id=st.district_id, station_id=st.id, lat=tlat, lon=tlon, attributes={}, **prov))
        for i, (code, name, dcode) in enumerate([("ST-1", "Surveillance Tower ST-1", "BLS"), ("ST-2", "Surveillance Tower ST-2", "KDP"),
                                                 ("ST-3", "Surveillance Tower ST-3", "PRI"), ("ST-4", "Surveillance Tower ST-4", "GJM")]):
            ref = [s for s in self.st.values() if s.district_id == self.dist[dcode].id][0]
            la, lo = move(ref.lat, ref.lon, 225, 2)
            self.add(Place(code=code, name=f"{name} (SIMULATED)", place_type="SURVEILLANCE_TOWER",
                           district_id=ref.district_id, station_id=ref.id, lat=la, lon=lo, attributes={}, **prov))
        for i, (dcode, sname) in enumerate([("BLS", "Chandipur"), ("BDK", "Dhamra"), ("JSP", "Paradip"),
                                            ("PRI", "Konark"), ("GJM", "Gopalpur"), ("KDP", "Jambu")]):
            ref = self.st[sname]
            la, lo = move(ref.lat, ref.lon, 250, 1.2)
            self.add(Place(code=f"RDR-{i + 1}", name=f"Coastal Radar Site R-{i + 1} (SIMULATED)", place_type="RADAR_SITE",
                           district_id=ref.district_id, station_id=ref.id, lat=la, lon=lo,
                           attributes={"range_nm": 25}, **prov))
        for code, name, ptype, dcode, lat, lon, aliases in PORTS:
            ref = min(self.st.values(), key=lambda s: (s.lat - lat) ** 2 + (s.lon - lon) ** 2)
            self.add(Place(code=code, name=name, place_type=ptype, district_id=self.dist[dcode].id, station_id=ref.id,
                           lat=lat, lon=lon, attributes={"aliases": aliases}, source="PUBLIC NAME / APPROX POSITION",
                           classification="UNCLASSIFIED", verification="UNVERIFIED"))
        sens = [("SI-A", "Paradip", 150, 1.8), ("SI-B", "Chandipur", 160, 2.5), ("SI-C", "Gopalpur", 100, 1.5)]
        for code, sname, brg, d in sens:
            ref = self.st[sname]
            la, lo = move(ref.lat, ref.lon, brg, d)
            self.add(Place(code=code, name=f"Sensitive Installation {code} (SIMULATED PLACEHOLDER)",
                           place_type="SENSITIVE_INSTALLATION", district_id=ref.district_id, station_id=ref.id,
                           lat=la, lon=lo, attributes={"note": "Generic placeholder; not a real facility"}, **prov))
            ring = [list(reversed(move(la, lo, b, 2.2))) for b in range(0, 360, 45)]
            ring.append(ring[0])
            ring = [[p[0], p[1]] for p in ring]
            self.add(Zone(code=f"RZ-{code[-1]}", name=f"Restricted Zone RZ-{code[-1]} around {code} (SIMULATED)",
                          zone_type="RESTRICTED", polygon=ring, notes="Illustrative geofence, not an official zone"))
        # SAR sectors: six offshore boxes, one per district
        for dcode, name, lat, lon in DISTRICTS:
            st = [s for s in self.st.values() if s.district_id == self.dist[dcode].id]
            la = sum(s.lat for s in st) / len(st)
            lo = sum(s.lon for s in st) / len(st)
            sb = SEAWARD[dcode]
            a = list(reversed(move(la, lo, sb - 60, 4)))
            b = list(reversed(move(la, lo, sb + 60, 4)))
            c = list(reversed(move(*move(la, lo, sb + 60, 4), sb, 22)))
            d = list(reversed(move(*move(la, lo, sb - 60, 4), sb, 22)))
            self.add(Zone(code=f"SAR-{dcode}", name=f"SAR Sector {name} (SIMULATED)", zone_type="SAR_SECTOR",
                          polygon=[a, b, c, d, a], notes="Illustrative sector for exercise use"))
        # Watch zones near vulnerable landing points of two stations
        for sname in ("Talchua", "Astaranga"):
            s = self.st[sname]
            ring = [list(reversed(move(s.lat, s.lon, b, 5))) for b in range(40, 220, 30)]
            ring.append(ring[0])
            self.add(Zone(code=f"WZ-{s.code[4:]}", name=f"Watch Zone {sname} approaches (SIMULATED)", zone_type="WATCH",
                          polygon=ring, notes="Illustrative watch zone"))
        self.db.flush()

    def _qual(self, p: Personnel, code: str, expired: bool = False):
        issued = self.today - timedelta(days=self.r.randint(60, 900))
        months = dict((c, m) for c, _, m in QUALS)[code]
        valid = issued + timedelta(days=months * 30)
        if expired:
            valid = self.today - timedelta(days=self.r.randint(5, 120))
        elif valid < self.today:
            valid = self.today + timedelta(days=self.r.randint(20, 400))
        p.qualifications.append(PersonnelQualification(qual_code=code, issued_on=issued, valid_until=valid))
        course = next(c for c in COURSES if c[2] == code)
        p.training.append(TrainingRecord(course_code=course[0], completed_on=issued, due_on=valid,
                                         status="OVERDUE" if valid < self.today else "COMPLETED"))

    def _person(self, st: Station | None, rank: str, role: str, designation: str, duty: str = "ON_DUTY",
                quals: list[str] = (), expired: list[str] = (), name: str | None = None) -> Personnel:
        self.pid_n += 1
        p = Personnel(pid=f"OPCS-{self.pid_n:04d}", name=name or self.name(), rank=rank, designation=designation,
                      role=role, station_id=st.id if st else None,
                      posting=f"{st.name} Marine PS" if st else "ICCC / Coastal Security HQ (SIM)",
                      current_duty="Station duty" if duty in {"ON_DUTY", "STANDBY"} else duty.replace("_", " ").title(),
                      duty_status=duty, shift=self.r.choice(["A", "B", "C"]), medical_fit=True,
                      medical_valid_until=self.today + timedelta(days=self.r.randint(30, 700)),
                      mobile=f"90000{self.r.randint(10000, 99999)}", source="HRMS (SIMULATED)", verification="SYSTEM")
        self.db.add(p)
        for q in quals:
            self._qual(p, q, expired=q in expired)
        if st:
            self.people_by_station.setdefault(st.id, []).append(p)
        return p

    def personnel(self):
        r = self.r
        for idx, st in enumerate(self.st.values()):
            self._person(st, "INSP", "IIC", "Inspector-in-Charge", "ON_DUTY",
                         ["BOAT_CREW", "NAVIGATION", "SWIMMING", "SEA_SURVIVAL", "MARINE_VHF", "WEAPONS", "NIGHT_OPS"])
            for m in range(2):  # boat masters
                exp = ["SEA_SURVIVAL"] if (st.name == "Chandbali" and m == 1) else []
                q = ["BOAT_CREW", "NAVIGATION", "SWIMMING", "SEA_SURVIVAL", "MARINE_VHF", "NIGHT_OPS", "FIRST_AID"]
                if r.random() < 0.6:
                    q.append("SAR")
                self._person(st, r.choice(["SI", "ASI"]), "BOAT_MASTER", "Boat Master", "ON_DUTY", q, exp)
            n_crew = 10 if st.name in {"Dhamra", "Paradip", "Gopalpur", "Chandipur", "Puri", "Astaranga", "Talchua", "Bahabalpur"} else 7
            for c in range(n_crew):
                q = ["BOAT_CREW", "SWIMMING", "SEA_SURVIVAL"]
                for extra, p in (("MARINE_VHF", 0.5), ("SAR", 0.35), ("FIRST_AID", 0.5), ("NIGHT_OPS", 0.4),
                                 ("WEAPONS", 0.7), ("CYBER_IT", 0.08)):
                    if r.random() < p:
                        q.append(extra)
                exp = ["SWIMMING"] if r.random() < 0.08 else []
                duty = r.choices(["ON_DUTY", "STANDBY", "OFF_DUTY", "LEAVE", "TRAINING", "MEDICAL_LEAVE"],
                                 [55, 20, 10, 8, 4, 3])[0]
                self._person(st, r.choice(["HAV", "CONST", "CONST", "ASI"]), "CREW", "Marine Crew", duty, q, exp)
            for _ in range({"Talasari": 1, "Dhamra": 2, "Jambu": 1, "Siali": 1, "Puri": 2, "Arjyapalli": 1}.get(st.name, 0)):
                self._person(st, "CONST", "UAV_PILOT", "UAV Remote Pilot", "ON_DUTY",
                             ["UAV_PILOT", "SWIMMING", "CYBER_IT", "FIRST_AID"])
            self._person(st, "CONST", "DRIVER", "Driver", r.choice(["ON_DUTY", "STANDBY"]), ["FIRST_AID"])
        # Dhamra: two sea-ready crew unavailable (explainability demo)
        dh = self.people_by_station[self.st["Dhamra"].id]
        for p in [x for x in dh if x.role == "CREW"][:2]:
            p.duty_status, p.current_duty = "LEAVE", "Leave"
        # Astaranga: make night-ops qualification interesting
        # ICCC / HQ staff (not station-posted)
        for rank, role, desig in [("DSP", "ICCC_SUPERVISOR", "ICCC Duty Officer"), ("INSP", "ICCC_SUPERVISOR", "ICCC Shift Supervisor"),
                                  ("SI", "ICCC_OPERATOR", "ICCC Operator"), ("ASI", "ICCC_OPERATOR", "ICCC Operator"),
                                  ("HAV", "ICCC_OPERATOR", "ICCC Operator"), ("INSP", "INTEL_OFFICER", "Intelligence Officer"),
                                  ("SI", "CYBER_ADMIN", "Cybersecurity Administrator"), ("CIV", "SYSTEM_ADMIN", "System Administrator"),
                                  ("ADGP", "ADGP", "ADGP, Coastal Security (SIMULATED ROLE)"), ("SP", "SP", "SP Coastal Security (SIMULATED)"),
                                  ("CONST", "UAV_PILOT", "UAV Remote Pilot (HQ pool)"), ("SI", "AUDITOR", "Internal Auditor")]:
            q = ["CYBER_IT"] if role in {"CYBER_ADMIN", "SYSTEM_ADMIN"} else (["UAV_PILOT", "SWIMMING"] if role == "UAV_PILOT" else [])
            self._person(None, rank, role, desig, "ON_DUTY", q)
        self.db.flush()

    def assets(self):
        r = self.r
        today = self.today
        fib_n = trawler_n = rwc_n = uav_n = veh_n = 0
        stations = list(self.st.values())
        big = {"Dhamra", "Paradip", "Gopalpur", "Chandipur", "Puri", "Astaranga", "Talchua", "Bahabalpur"}
        for st in stations:
            people = self.people_by_station[st.id]
            masters = [p for p in people if p.role == "BOAT_MASTER"]
            crew = [p for p in people if p.role in {"CREW", "IIC"}]
            nboats = 2 if st.name in big else 1
            for b in range(nboats):
                fib_n += 1
                twelve = st.name in big and b == 0 or r.random() < 0.4
                code = f"FIB-12T-{fib_n:02d}" if twelve else f"FIB-5T-{fib_n:02d}"
                a = Asset(asset_code=code, asset_type="BOAT", subtype="Fast Interceptor Boat 12T" if twelve else "Fast Interceptor Boat 5T",
                          station_id=st.id, manufacturer="Simulated Shipyard Ltd", model="FIB-12" if twelve else "FIB-5",
                          lat=st.lat, lon=st.lon, cruise_speed_kn=22 if twelve else 18, endurance_nm=180 if twelve else 120,
                          fuel_pct=r.randint(55, 100), operating_hours=r.randint(400, 3000), crew_required=6 if twelve else 4,
                          last_maintenance=today - timedelta(days=r.randint(10, 80)),
                          next_maintenance=today + timedelta(days=r.randint(5, 90)),
                          certification_valid_until=today + timedelta(days=r.randint(30, 400)),
                          amc_valid_until=today + timedelta(days=r.randint(30, 500)),
                          last_inspection=today - timedelta(days=r.randint(5, 60)), gps_status="OPERATIONAL",
                          ais_status="OPERATIONAL", vhf_status="OPERATIONAL", radar_status="OPERATIONAL" if twelve else "N/A",
                          source="ASSET TELEMETRY (SIMULATED)", verification="SYSTEM")
                self.db.add(a)
                self.db.flush()
                self.db.add(CrewAssignment(asset_id=a.id, personnel_id=masters[b % len(masters)].id, crew_role="MASTER"))
                for p in r.sample(crew, min(len(crew), a.crew_required + 1)):
                    self.db.add(CrewAssignment(asset_id=a.id, personnel_id=p.id, crew_role="CREW"))
                self.db.add(MaintenanceRecord(asset_id=a.id, kind="SCHEDULED", description="Periodic engine service",
                                              started_at=self.now - timedelta(days=40), completed_at=self.now - timedelta(days=38),
                                              performed_by="AMC vendor (SIM)"))
            if r.random() < 0.35:
                rwc_n += 1
                a = self.add(Asset(asset_code=f"RWC-{rwc_n:02d}", asset_type="RWC", subtype="Rescue Water Craft", station_id=st.id,
                                   manufacturer="Simulated Marine", model="RWC-1100", lat=st.lat, lon=st.lon, cruise_speed_kn=30,
                                   endurance_nm=45, fuel_pct=r.randint(60, 100), crew_required=2,
                                   last_maintenance=today - timedelta(days=30), next_maintenance=today + timedelta(days=60),
                                   certification_valid_until=today + timedelta(days=200), source="ASSET TELEMETRY (SIMULATED)"))
                self.db.flush()
                for p in r.sample(crew, 3):
                    self.db.add(CrewAssignment(asset_id=a.id, personnel_id=p.id, crew_role="CREW"))
                self.db.add(CrewAssignment(asset_id=a.id, personnel_id=masters[0].id, crew_role="MASTER"))
            veh_n += 1
            v = self.add(Asset(asset_code=f"VEH-{veh_n:02d}", asset_type="VEHICLE", subtype="Patrol Vehicle (4x4)", station_id=st.id,
                               manufacturer="Simulated Motors", model="PV-4", lat=st.lat - 0.004, lon=st.lon - 0.006,
                               cruise_speed_kn=25, endurance_nm=300, fuel_pct=r.randint(40, 100), crew_required=1,
                               vhf_status="OPERATIONAL", source="ASSET REGISTER (SIMULATED)"))
            self.db.flush()
            drv = [p for p in people if p.role == "DRIVER"]
            if drv:
                self.db.add(CrewAssignment(asset_id=v.id, personnel_id=drv[0].id, crew_role="DRIVER"))
            self.add(Asset(asset_code=f"VHF-{st.code[4:]}", asset_type="COMMS", subtype="VHF base station", station_id=st.id,
                           manufacturer="Simulated Radio", model="VB-25", lat=st.lat + 0.002, lon=st.lon + 0.001,
                           operational_status="OPERATIONAL" if st.vhf_base_status == "OPERATIONAL" else "DEGRADED",
                           source="NMS (SIMULATED)"))
            if st.name in big:
                self.add(Asset(asset_code=f"EOIR-{st.code[4:]}", asset_type="SENSOR", subtype="EO/IR camera", station_id=st.id,
                               manufacturer="Simulated Optronics", model="EO-12", lat=st.lat + 0.003, lon=st.lon - 0.002,
                               operational_status="OPERATIONAL", source="NMS (SIMULATED)"))
        # hired trawlers
        for sname in ["Paradip", "Dhamra", "Puri", "Gopalpur", "Chandipur"]:
            st = self.st[sname]
            trawler_n += 1
            people = self.people_by_station[st.id]
            a = self.add(Asset(asset_code=f"Trawler-{trawler_n:02d}", asset_type="TRAWLER", subtype="Hired fishing trawler",
                               station_id=st.id, manufacturer="Local builder (SIM)", model="Wooden trawler 15m",
                               lat=st.lat, lon=st.lon, cruise_speed_kn=9, endurance_nm=250, fuel_pct=r.randint(50, 95),
                               crew_required=4, certification_valid_until=today + timedelta(days=r.randint(20, 300)),
                               next_maintenance=today + timedelta(days=r.randint(10, 60)), source="ASSET TELEMETRY (SIMULATED)"))
            self.db.flush()
            self.db.add(CrewAssignment(asset_id=a.id, personnel_id=[p for p in people if p.role == "BOAT_MASTER"][1].id, crew_role="MASTER"))
            for p in r.sample([p for p in people if p.role == "CREW"], 4):
                self.db.add(CrewAssignment(asset_id=a.id, personnel_id=p.id, crew_role="CREW"))
        # UAVs
        pilots = self.db.query(Personnel).filter(Personnel.role == "UAV_PILOT").all()
        for sname in ["Talasari", "Dhamra", "Jambu", "Siali", "Puri", "Arjyapalli"]:
            st = self.st[sname]
            uav_n += 1
            a = self.add(Asset(asset_code=f"UAV-{uav_n:02d}", asset_type="UAV", subtype="Multirotor surveillance UAV",
                               station_id=st.id, manufacturer="Simulated Aero", model="MR-8 EO/IR", lat=st.lat + 0.001,
                               lon=st.lon + 0.001, cruise_speed_kn=35, endurance_nm=25, fuel_pct=r.randint(70, 100),
                               crew_required=1, vhf_status="OPERATIONAL", gps_status="OPERATIONAL",
                               certification_valid_until=today + timedelta(days=r.randint(60, 300)),
                               source="GCS TELEMETRY (SIMULATED)"))
            self.db.flush()
            local = [p for p in pilots if p.station_id == st.id]
            pilot = local[0] if local else pilots[-1]
            self.db.add(CrewAssignment(asset_id=a.id, personnel_id=pilot.id, crew_role="UAV_PILOT"))
            if sname == "Dhamra":
                self.db.add(Defect(asset_id=a.id, description="Battery pack replacement due (cycle limit reached)",
                                   severity="MAJOR", reported_by="seed", reported_at=self.now - timedelta(days=3)))
                a.fuel_pct = 45
        self.db.flush()
        # Explainability demo state at Dhamra: FIB-12T-03 under maintenance; FIB-12T-04 patrolling (spec example)
        dh = self.st["Dhamra"]
        dboats = self.db.query(Asset).filter(Asset.station_id == dh.id, Asset.asset_type == "BOAT").order_by(Asset.id).all()
        self._rename(dboats[0], "FIB-12T-04")
        self._rename(dboats[1], "FIB-12T-03", subtype="Fast Interceptor Boat 12T", crew=6)
        dboats[1].operational_status, dboats[1].availability = "UNDER_MAINTENANCE", "MAINTENANCE"
        self.db.add(MaintenanceRecord(asset_id=dboats[1].id, kind="BREAKDOWN", description="Gearbox overhaul",
                                      started_at=self.now - timedelta(days=4), performed_by="AMC vendor (SIM)"))
        b4 = dboats[0]
        b4.fuel_pct, b4.cruise_speed_kn, b4.crew_required = 68, 18, 6
        # Other degraded examples
        others = self.db.query(Asset).filter(Asset.asset_type == "BOAT", Asset.station_id != dh.id).order_by(Asset.id).all()
        others[3].fuel_pct = 22
        self.db.add(Defect(asset_id=others[5].id, description="Port engine fuel injector failure", severity="CRITICAL",
                           reported_by="seed", reported_at=self.now - timedelta(days=1)))
        others[5].operational_status, others[5].availability = "DEFECTIVE", "DEFECTIVE"
        self.db.add(Defect(asset_id=others[7].id, description="Search light intermittent", severity="MINOR",
                           reported_by="seed", reported_at=self.now - timedelta(days=6)))
        others[9].vhf_status = "DEGRADED"
        others[11].availability = "RESERVE"
        others[12].safety_equipment_ok = False
        self.db.add(Defect(asset_id=others[12].id, description="Two life rafts overdue for servicing", severity="MAJOR",
                           reported_by="seed", reported_at=self.now - timedelta(days=9)))
        cams = self.db.query(Place).filter(Place.place_type == "CCTV").order_by(Place.id).all()
        cams[8].status = "DEGRADED"
        self.db.flush()

    def _rename(self, a: Asset, code: str, subtype: str | None = None, crew: int | None = None):
        clash = self.db.query(Asset).filter(Asset.asset_code == code).first()
        if clash and clash.id != a.id:  # swap codes so numbering stays contiguous
            mine = a.asset_code
            a.asset_code = "__swap__"
            self.db.flush()
            clash.asset_code = mine
            self.db.flush()
        a.asset_code = code
        if subtype:
            a.subtype, a.model = subtype, "FIB-12"
        if crew:
            a.crew_required = crew

    def vessels(self):
        r = self.r
        stations = list(self.st.values())
        flcs = {p.station_id: p for p in self.db.query(Place).filter(Place.place_type == "FISH_LANDING_CENTRE")}
        n = 0
        for i in range(140):
            st = stations[i % len(stations)]
            dcode = next(k for k, v in self.dist.items() if v.id == st.district_id)
            n += 1
            vtype = r.choices(["FISHING_TRAWLER", "GILLNETTER", "MOTORISED_BOAT", "NON_MOTORISED"], [35, 30, 25, 10])[0]
            lat, lon = self.sea_point(st, 1.5, 22 if vtype in {"FISHING_TRAWLER", "GILLNETTER"} else 9)
            name = f"{r.choice(BOAT_NAMES)}-{r.randint(1, 99)}"
            has_ais = vtype == "FISHING_TRAWLER" or (vtype == "GILLNETTER" and r.random() < 0.3)
            transponder = r.choices(["NABHMITRA", "VCSS", "NONE"], [40, 20, 40])[0] if vtype != "NON_MOTORISED" else "NONE"
            beh = r.choices(["FISHING", "TRANSIT", "MOORED"], [60, 30, 10])[0]
            tl, to = self.sea_point(st, 2, 20)
            v = Vessel(vessel_code=f"VSL-{n:04d}", name=name, vessel_type=vtype,
                       registration=f"OD-SIM-{dcode}-{r.randint(1000, 9999)}",
                       mmsi=f"SIM{419000000 + r.randint(1000, 99999)}" if has_ais else None,
                       ais_name=name if has_ais else None, owner_name=self.name(),
                       owner_mobile=f"90000{r.randint(10000, 99999)}", home_flc_id=flcs[st.id].id, station_id=st.id,
                       length_m=round(r.uniform(7, 18), 1) if vtype != "NON_MOTORISED" else round(r.uniform(5, 8), 1),
                       crew_count=r.randint(2, 12), transponder=transponder,
                       safety_equipment={"life_jackets": r.randint(2, 12), "life_buoy": r.random() < 0.7,
                                         "vhf": has_ais or r.random() < 0.3, "first_aid": r.random() < 0.6},
                       emergency_contact=f"{self.name()} 90000{r.randint(10000, 99999)}",
                       expected_return=self.now + timedelta(hours=r.randint(2, 60)),
                       lat=lat, lon=lon, course=bearing_deg(lat, lon, tl, to),
                       speed_kn=r.uniform(1, 3) if beh == "FISHING" or vtype == "NON_MOTORISED" else r.uniform(5, 8),
                       ais_active=has_ais, last_ais_ts=self.now if has_ais else None,
                       track_source="AIS" if has_ais else ("TRANSPONDER" if transponder != "NONE" else "REPORTED"),
                       identity_status="IDENTIFIED", behaviour=beh, target_lat=tl, target_lon=to, registered_citizen=True,
                       source="AIS (SIMULATED)" if has_ais else "TRANSPONDER/RADAR (SIMULATED)", verification="SYSTEM",
                       confidence=0.9 if has_ais else 0.6)
            if not has_ais and transponder == "NONE":
                v.lat, v.lon = self.sea_point(st, 1, 6)
                v.course = bearing_deg(v.lat, v.lon, tl, to)
            self.db.add(v)
        # merchant traffic near ports
        ports = {p.code: p for p in self.db.query(Place).filter(Place.place_type == "PORT")}
        for i, nm in enumerate(MERCHANT):
            port = list(ports.values())[i % len(ports)]
            lat, lon = move(port.lat, port.lon, 110 + r.uniform(-20, 20), r.uniform(8, 25))
            n += 1
            self.db.add(Vessel(vessel_code=f"VSL-{n:04d}", name=nm, vessel_type=r.choice(["CARGO", "TANKER", "CARGO"]),
                               mmsi=f"SIM{470000000 + r.randint(1000, 99999)}", ais_name=nm, flag=r.choice(["IN", "SG", "PA", "LR"]),
                               length_m=r.randint(120, 250), lat=lat, lon=lon, course=bearing_deg(lat, lon, port.lat, port.lon + 0.03), speed_kn=r.uniform(8, 13),
                               ais_active=True, last_ais_ts=self.now, track_source="AIS", identity_status="IDENTIFIED",
                               behaviour="TRANSIT", target_lat=port.lat, target_lon=port.lon + 0.03,
                               source="AIS (SIMULATED)", verification="SYSTEM"))
        # Pre-set intelligence picture: 1 identity mismatch, 1 AIS gap, 2 TOIs, 1 dark contact
        self.db.flush()
        vs = self.db.query(Vessel).filter(Vessel.vessel_type == "FISHING_TRAWLER").order_by(Vessel.id).all()
        vs[2].ais_name = "SAGAR SAMRAT"
        vs[2].identity_status = "MISMATCH"
        vs[5].ais_active, vs[5].last_ais_ts, vs[5].track_source = False, self.now - timedelta(minutes=48), "RADAR"
        vs[8].is_toi, vs[8].toi_reason = True, "SIMULATED: repeated night-time movements near VLP-05 (exercise intelligence)"
        vs[11].is_toi, vs[11].toi_reason = True, "SIMULATED: correlated with earlier unverified community report"
        self.db.add(WatchListEntry(list_name="Exercise Watch List (SIMULATED)", vessel_id=vs[8].id, reason="Exercise entry",
                                   added_by="intel.officer"))
        self.db.add(WatchListEntry(list_name="Exercise Watch List (SIMULATED)", identifier="SAGAR SAMRAT",
                                   reason="Exercise entry: name reported in simulated tip-off", added_by="intel.officer"))
        st = self.st["Talchua"]
        la, lo = self.sea_point(st, 6, 9, 10)
        vlp = self.db.query(Place).filter(Place.place_type == "VULNERABLE_LANDING", Place.station_id == st.id).first()
        self.db.add(Vessel(vessel_code="TGT-0001", vessel_type="UNKNOWN", lat=la, lon=lo, course=270, speed_kn=6,
                           ais_active=False, track_source="RADAR", identity_status="UNIDENTIFIED", behaviour="INBOUND",
                           target_lat=vlp.lat, target_lon=vlp.lon, source="COASTAL RADAR (SIMULATED)", confidence=0.6,
                           verification="UNVERIFIED"))
        self.db.flush()
        # 2 h of track history per vessel (every 10 min) so history/analytics have context
        for v in self.db.query(Vessel).all():
            la, lo = v.lat, v.lon
            back = (v.course + 180) % 360
            for k in range(12, 0, -1):
                pla, plo = move(la, lo, back + r.uniform(-15, 15), (v.speed_kn or 1) * (k / 6) * 0.9)
                self.db.add(VesselTrackPoint(vessel_id=v.id, ts=self.now - timedelta(minutes=10 * k), lat=pla, lon=plo,
                                             speed_kn=v.speed_kn, course=v.course,
                                             source="AIS" if v.ais_active else v.track_source))

    def missions(self):
        r = self.r
        year = self.now.year
        boats = self.db.query(Asset).filter(Asset.asset_type.in_(["BOAT", "TRAWLER"])).order_by(Asset.id).all()
        n = 0
        for d in range(30, 0, -1):
            for b in r.sample(boats, 3):
                n += 1
                st = self.db.get(Station, b.station_id)
                start = self.now - timedelta(days=d, hours=r.randint(0, 10))
                dist = r.uniform(18, 60)
                self.db.add(Mission(code=f"PATROL-{year}-{n:04d}", mission_type="BOAT_PATROL", station_id=st.id,
                                    asset_id=b.id, status="COMPLETED", objective="Routine coastal patrol",
                                    route=[], planned_start=start, started_at=start,
                                    ended_at=start + timedelta(hours=r.uniform(3, 7)), distance_nm=round(dist, 1),
                                    fuel_used_pct=round(dist / (b.endurance_nm or 120) * 100, 1),
                                    sightings=r.randint(0, 25), boardings=r.randint(0, 6), created_by="seed"))
        n = 97
        dh_boat = self.db.query(Asset).filter(Asset.asset_code == "FIB-12T-04").first()
        active = [dh_boat] + [b for b in boats if b.asset_type == "BOAT" and b.id != dh_boat.id
                              and b.operational_status == "OPERATIONAL" and b.availability == "AVAILABLE"
                              and b.fuel_pct >= 45][::4][:5]
        for b in active:
            n += 1
            st = self.db.get(Station, b.station_id)
            route = []
            for k in range(5):
                la, lo = self.sea_point(st, 3, 11, 50)
                route.append([lo, la])
            start = self.now.replace(hour=1, minute=10) if b is dh_boat else self.now - timedelta(hours=r.uniform(0.5, 3))
            m = Mission(code=f"PATROL-{year}-{n:04d}", mission_type="BOAT_PATROL", station_id=st.id, asset_id=b.id,
                        status="ACTIVE", route=route, route_index=1, objective="Coastal patrol — sector sweep",
                        planned_start=start, started_at=start, created_by="seed")
            self.db.add(m)
            self.db.flush()
            b.mission_status, b.availability, b.current_mission_id = "PATROLLING", "DEPLOYED", m.id
            b.lat, b.lon = route[0][1], route[0][0]
            b.speed_kn = 18 if b is dh_boat else 14
            # Crew the patrol with sea-ready qualified members (recalling off-duty staff if needed).
            fit = []
            for c in b.crew:
                p = c.personnel
                q = {x.qual_code for x in p.qualifications if x.valid_until is None or x.valid_until >= self.today}
                if c.active and p.duty_status in {"ON_DUTY", "STANDBY", "OFF_DUTY"} and {"SWIMMING", "SEA_SURVIVAL", "BOAT_CREW"} <= q:
                    fit.append(c)
            if b is dh_boat and len(fit) < b.crew_required:
                pool = [p for p in self.people_by_station[st.id] if p.duty_status in {"ON_DUTY", "STANDBY", "OFF_DUTY"}
                        and p.id not in {c.personnel_id for c in b.crew}
                        and {"SWIMMING", "SEA_SURVIVAL", "BOAT_CREW"} <= {x.qual_code for x in p.qualifications
                                                                         if x.valid_until and x.valid_until >= self.today}]
                for p in pool[: b.crew_required - len(fit)]:
                    ca = CrewAssignment(asset_id=b.id, personnel_id=p.id, crew_role="CREW")
                    self.db.add(ca)
                    self.db.flush()
                    fit.append(ca)
            fit.sort(key=lambda c: c.crew_role != "MASTER")
            crew = fit[: b.crew_required]
            for c in crew:
                c.personnel.duty_status = "DEPLOYED"
                c.personnel.current_duty = f"{m.code} on {b.asset_code}"
            m.crew = [c.personnel_id for c in crew]
            self.db.add(EventFeed(ts=start, category="PATROL", message=f"{st.name} MPS — Patrol launched ({b.asset_code})",
                                  station_id=st.id, ref_type="mission", ref_id=m.id))
        # Ensure FIB-12T-04 has a qualified master and full sea-ready crew aboard (6/6)
        for c in dh_boat.crew:
            p = c.personnel
            if p.duty_status == "LEAVE":
                c.active = False
        self.db.flush()

    def incident_history(self):
        r = self.r
        fams = [("ENGINE_FAILURE", "L2", "Engine failure / drifting"), ("MISSING_BOAT", "L2", "Missing fishing boat"),
                ("MEDICAL_EMERGENCY", "L2", "Medical emergency at sea"), ("SUSPICIOUS_VESSEL", "L3", "Possible suspicious vessel"),
                ("CAPSIZING", "L1", "Capsized vessel"), ("ILLEGAL_FISHING", "L3", "Possible illegal fishing"),
                ("FLOATING_OBSTRUCTION", "L3", "Floating obstruction"), ("BEACH_MISSING_PERSON", "L1", "Missing person at beach"),
                ("FUEL_SHORTAGE", "L2", "Fuel shortage at sea"), ("COLLISION", "L2", "Collision")]
        stations = list(self.st.values())
        boats = self.db.query(Asset).filter(Asset.asset_type.in_(["BOAT", "RWC", "TRAWLER"])).all()
        year = self.now.year
        for i in range(46):
            fam, prio, label = r.choice(fams)
            st = r.choice(stations)
            lat, lon = self.sea_point(st, 2, 14)
            det = self.now - timedelta(days=r.uniform(1, 60), hours=r.uniform(0, 12))
            ver = det + timedelta(minutes=r.uniform(1.5, 9))
            disp = ver + timedelta(minutes=r.uniform(3, 18))
            arr = disp + timedelta(minutes=r.uniform(12, 70))
            false = r.random() < 0.12
            inc = Incident(code=f"INC-{year}-{i + 1:04d}", title=label, family=fam, priority=prio,
                           status="C8" if false else "C7", classification=label, lat=lat, lon=lon,
                           location_desc=f"{r.uniform(2, 14):.0f} NM off {st.name}", location_confidence=r.choice(["GPS", "APPROXIMATE", "REPORTED"]),
                           persons_onboard=r.randint(1, 10) if fam not in {"FLOATING_OBSTRUCTION", "ILLEGAL_FISHING"} else None,
                           station_id=st.id, detected_at=det, alert_at=det + timedelta(seconds=30), verified_at=ver,
                           verified_by="operator.iccc", human_verified=True, owner_user="operator.iccc",
                           source=r.choice(["CHATBOT/WEB", "CHATBOT/WHATSAPP_SIM", "PHONE (SIM)", "PATROL", "ALERT"]),
                           created_at=det, updated_at=arr, closure_at=arr + timedelta(hours=1), closed_by="supervisor.iccc",
                           verification="HUMAN_VERIFIED", source_ts=det, received_ts=det)
            if false:
                inc.false_reason = "Duplicate of an earlier report (SIMULATED)"
            else:
                b = r.choice([x for x in boats if x.station_id == st.id] or boats)
                inc.assigned_asset_id = b.id
                inc.dispatch_at, inc.launch_at, inc.arrival_at = disp, disp + timedelta(minutes=r.uniform(2, 8)), arr
                inc.outcome = r.choice(["All persons safe; vessel towed to harbour", "Persons rescued and handed over to family",
                                        "Area searched; no vessel found; referred to MRCC", "Vessel located and escorted",
                                        "Medical evacuation completed; patient handed to ambulance"])
            self.db.add(inc)
            self.db.flush()
            for ts, et, d in [(det, "C0", "Intake"), (det + timedelta(seconds=30), "C1", "Provisional alert"),
                              (ver, "C2", "Operator acknowledged"),
                              (disp, "C5", "Dispatch confirmed") if not false else (ver + timedelta(minutes=5), "C8", "Duplicate"),
                              (arr + timedelta(hours=1), "C7", "Closed")][: 4 if false else 5]:
                self.db.add(IncidentEvent(incident_id=inc.id, ts=ts, event_type=et, actor="seed", detail=d))
        # An open unassigned L3 incident for the queue
        st = self.st["Konark"]
        lat, lon = self.sea_point(st, 3, 5)
        inc = Incident(code=f"INC-{year}-{47:04d}", title="Floating obstruction — drifting container reported", family="FLOATING_OBSTRUCTION",
                       priority="L3", status="C1", classification="Floating obstruction", lat=lat, lon=lon,
                       location_desc="about 4 NM off Konark (reported)", location_confidence="REPORTED", station_id=st.id,
                       detected_at=self.now - timedelta(minutes=22), alert_at=self.now - timedelta(minutes=21),
                       source="PATROL", created_at=self.now - timedelta(minutes=22), updated_at=self.now - timedelta(minutes=21))
        self.db.add(inc)
        self.db.flush()
        self.db.add(IncidentEvent(incident_id=inc.id, ts=inc.detected_at, event_type="C0", actor="FIB crew", detail="Reported by patrol"))
        self.db.add(IncidentEvent(incident_id=inc.id, ts=inc.alert_at, event_type="C1", actor="system", detail="Provisional alert"))
        # historical alerts (resolved) for analytics
        types = ["AIS_LOST", "LOITERING", "ABNORMAL_SPEED", "IDENTITY_MISMATCH", "RESTRICTED_ZONE", "RADAR_NO_AIS", "RENDEZVOUS"]
        vs = self.db.query(Vessel).order_by(Vessel.id).all()
        for i in range(60):
            v = r.choice(vs)
            t = r.choice(types)
            ts = self.now - timedelta(days=r.uniform(1, 30))
            self.db.add(Alert(code=f"ALR-{year}-{i + 1:05d}", alert_type=t, severity=r.choice(["LOW", "MEDIUM", "HIGH"]),
                              title=f"{t.replace('_', ' ').title()}: {v.name or v.vessel_code}", description="Historical (SIMULATED)",
                              lat=v.lat, lon=v.lon, vessel_id=v.id, detected_at=ts, status=r.choice(["RESOLVED", "DISMISSED", "RESOLVED"]),
                              risk=0.3, source="ANALYTICS (SIMULATED)", source_ts=ts, received_ts=ts, confidence=0.6))

    def system(self):
        now = self.now
        rows = [
            ("AIS", "AIS feed (terrestrial)", "AIS", "SIMULATED", "SIMULATED", "Radar tracks; last known positions", "Synthetic AIS generated by simulator"),
            ("RADAR", "Coastal radar chain", "RADAR", "SIMULATED", "SIMULATED", "AIS + patrol reports", "Synthetic radar tracks; no live radar integration"),
            ("UAV", "UAV video / telemetry", "UAV", "SIMULATED", "SIMULATED", "Boat patrol visual confirmation", "Synthetic UAV observations"),
            ("CCTV", "FLC CCTV network", "CCTV", "SIMULATED", "SIMULATED", "Station staff visual checks", "Status only; no video"),
            ("SATELLITE", "Satellite imagery / SAR cues", "SATELLITE", "NOT_INTEGRATED", "NOT_INTEGRATED", "None", "Architecture-ready; not integrated"),
            ("NABHMITRA", "NABHMITRA / VCSS fishing-vessel positions", "TRANSPONDER", "NOT_INTEGRATED", "NOT_INTEGRATED",
             "Citizen-shared location; AIS", "Real system exists (ISRO / Dept of Fisheries); NOT connected - positions simulated"),
            ("REGISTRY", "Vessel registry", "REGISTRY", "SIMULATED", "SIMULATED", "Local FLC registers", "Synthetic registry; national registry not integrated"),
            ("WEATHER", "Weather & sea state", "WEATHER", "SIMULATED", "SIMULATED", "Manual IMD bulletin entry", "Synthetic; IMD not integrated"),
            ("OSM_TILES", "OpenStreetMap tiles (public)", "MAP", "ONLINE", "LIVE", "Offline schematic coastline", "Public internet service; may be unavailable"),
            ("OPENSEAMAP", "OpenSeaMap seamarks (public)", "MAP", "ONLINE", "LIVE", "Static chart reference", "Public internet service; NOT FOR NAVIGATION"),
            ("ENC", "Authorised ENC charts", "MAP", "NOT_INTEGRATED", "NOT_INTEGRATED", "Static chart", "Architecture-ready layer slot"),
            ("MRCC_LINK", "MRCC / MRSC coordination link", "COORDINATION", "NOT_INTEGRATED", "NOT_INTEGRATED", "Telephone / VHF", "Handoffs are recorded only"),
            ("WHATSAPP", "WhatsApp Business channel", "CHANNEL", "NOT_INTEGRATED", "NOT_INTEGRATED", "Web chat", "Simulated channel endpoint only"),
            ("STATION_NETWORK", "Station WAN links", "NETWORK", "SIMULATED", "SIMULATED", "Backup link / VHF", "Synthetic link status"),
        ]
        for code, name, kind, status, integ, fb, notes in rows:
            self.add(DataSource(code=code, name=name, kind=kind, status=status, integration=integ, fallback=fb, notes=notes,
                                last_success=now if status in {"SIMULATED", "ONLINE"} else None, last_attempt=now,
                                latency_ms=self.r.randint(20, 400), owner="Coastal Security Wing (POC)"))
        for d in self.dist.values():
            self.add(WeatherReport(district_id=d.id, wind_kn=self.r.randint(8, 16), wind_dir=self.r.randint(150, 230),
                                   wave_m=round(self.r.uniform(0.8, 1.6), 1), sea_state=3, visibility_km=self.r.randint(6, 12),
                                   condition=self.r.choice(["Partly cloudy", "Clear", "Hazy"]), source="WEATHER (SIMULATED, not IMD)",
                                   verification="UNVERIFIED", classification="UNCLASSIFIED (SIMULATED)"))
        for i in range(6):
            self.add(CyberEvent(ts=now - timedelta(hours=self.r.randint(2, 90)), event_type=self.r.choice(
                ["FAILED_LOGIN", "PORT_SCAN_BLOCKED", "AV_SIGNATURE_UPDATE", "PATCH_PENDING"]), severity=self.r.choice(["LOW", "INFO"]),
                source_ip=f"198.51.100.{self.r.randint(2, 250)}", target="edge-fw", detail="Baseline synthetic event", status="CLOSED"))
        kb = [("SEA_SAFETY", "Sea safety checklist", "Carry life jackets for all, charged phone, VHF/transponder, extra fuel and water. Inform family of return time."),
              ("DISTRESS", "What to do in distress", "Stay with the boat, wear life jackets, call 112 or Coast Guard 1554, VHF Channel 16."),
              ("REPORTING", "How to report suspicious activity", "Report what you observe: time, place, boat description, persons, direction. Never approach."),
              ("WEATHER", "Weather warnings", "Follow official IMD and Fisheries Department warnings. This POC shows simulated weather only."),
              ("LOCATION", "Sharing live location", "Use the Share location button, or WhatsApp attach > Location > Send current location.")]
        for topic, title, body in kb:
            self.add(KnowledgeArticle(topic=topic, language="en", title=title, body=body))
        feed = [(95, "SYSTEM", "Shift handover completed — ICCC (SIMULATED)"), (70, "PATROL", "Paradip MPS — Patrol launched"),
                (52, "ALERT", "AIS anomaly detected — vessel off Chandipur"), (48, "INCIDENT", "Operator verification started"),
                (35, "ORDER", "UAV-02 tasked for area search"), (21, "INCIDENT", "Floating obstruction reported off Konark")]
        for mins, cat, msg in feed:
            self.add(EventFeed(ts=now - timedelta(minutes=mins), category=cat, message=msg))

    def users(self):
        pw = hash_password(self.pw, self.iters)
        P = self.db.query(Personnel)
        dh, ast = self.st["Dhamra"], self.st["Astaranga"]
        fib04 = self.db.query(Asset).filter(Asset.asset_code == "FIB-12T-04").first()
        master = next(c.personnel for c in fib04.crew if c.crew_role == "MASTER")
        uav = self.db.query(Asset).filter(Asset.asset_code == "UAV-02").first()
        pilot = next(c.personnel for c in uav.crew if c.crew_role == "UAV_PILOT")
        iic_dh = P.filter(Personnel.station_id == dh.id, Personnel.role == "IIC").first()
        iic_ast = P.filter(Personnel.station_id == ast.id, Personnel.role == "IIC").first()
        hq = {p.role: p for p in P.filter(Personnel.station_id.is_(None))}
        bls = self.dist["BLS"]
        rows = [
            ("dgp.demo", "DGP (Demo Account)", "DGP", "DGP", None, None, None, "STATE", True, None),
            ("adgp.demo", "ADGP Coastal Security (Demo)", "ADGP", "ADGP", hq.get("ADGP"), None, None, "STATE", True, None),
            ("sp.balasore", "SP Balasore (Demo)", "SP", "SP", None, None, bls.id, "DISTRICT", True, None),
            ("dsp.iccc", "DSP ICCC (Demo)", "DSP", "DSP", None, None, None, "STATE", True, None),
            ("supervisor.iccc", "ICCC Supervisor (Demo)", "INSP", "ICCC_SUPERVISOR", hq.get("ICCC_SUPERVISOR"), None, None, "STATE", True, None),
            ("operator.iccc", "ICCC Operator (Demo)", "SI", "ICCC_OPERATOR", hq.get("ICCC_OPERATOR"), None, None, "STATE", False, None),
            ("iic.dhamra", f"IIC Dhamra — {iic_dh.name}", "INSP", "IIC", iic_dh, dh.id, dh.district_id, "STATION", False, None),
            ("iic.astaranga", f"IIC Astaranga — {iic_ast.name}", "INSP", "IIC", iic_ast, ast.id, ast.district_id, "STATION", False, None),
            ("master.fib04", f"Boat Master FIB-12T-04 — {master.name}", master.rank, "BOAT_MASTER", master, dh.id, dh.district_id, "UNIT", False, fib04.id),
            ("uav.op1", f"UAV Operator UAV-02 — {pilot.name}", pilot.rank, "UAV_OPERATOR", pilot, uav.station_id, None, "UNIT", False, uav.id),
            ("mpo.dhamra", "Marine Police Officer, Dhamra (Demo)", "CONST", "MARINE_POLICE_OFFICER", None, dh.id, dh.district_id, "STATION", False, None),
            ("intel.officer", "Intelligence Officer (Demo)", "INSP", "INTEL_OFFICER", hq.get("INTEL_OFFICER"), None, None, "STATE", True, None),
            ("cyber.admin", "Cybersecurity Administrator (Demo)", "SI", "CYBER_ADMIN", hq.get("CYBER_ADMIN"), None, None, "STATE", False, None),
            ("sys.admin", "System Administrator (Demo)", "CIV", "SYSTEM_ADMIN", hq.get("SYSTEM_ADMIN"), None, None, "STATE", False, None),
            ("auditor", "Auditor (Demo)", "SI", "AUDITOR", hq.get("AUDITOR"), None, None, "STATE", False, None),
        ]
        for u, dn, rank, role, person, st_id, d_id, jur, intel, asset_id in rows:
            self.add(User(username=u, password_hash=pw, display_name=dn, rank=rank, role=role,
                          personnel_id=person.id if person else None, station_id=st_id, district_id=d_id,
                          jurisdiction=jur, intel_access=intel, assigned_asset_id=asset_id, is_demo=True))


def seed_if_empty(db: Session, seed: int, demo_password: str, pbkdf2_iterations: int | None = None) -> dict | None:
    if db.query(Station).first() is not None:
        return None
    out = Gen(db, seed, demo_password, pbkdf2_iterations).run()
    db.commit()
    from app.services.analytics import fuse, run_detectors
    run_detectors(db)
    fuse(db)
    db.commit()
    return out
