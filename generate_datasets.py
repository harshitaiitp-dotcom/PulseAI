"""
data/generate_datasets.py
Simulates realistic enterprise software project records and Jira/Git tickets.

Background & Motivation:
Real-world software telemetry is often proprietary or NDA-protected. To benchmark
different ML & DL approaches, we generate synthetic data modeled after distributions
observed in empirical software engineering literature (e.g., Boehm's COCOMO II,
Menzies et al. on defect density, and agile velocity metrics from Jira studies).
"""

import os
import random
import numpy as np
import pandas as pd

# Fix seed so benchmark results are reproducible across runs
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample project names to make records look like real Jira/GitHub repositories
PROJECT_NAMES = [
    "checkout-service-v2", "auth-oauth2-migration", "billing-core-engine",
    "k8s-cluster-upgrade", "search-indexer-es", "mobile-ios-app-refresh",
    "analytics-event-bus", "customer-portal-react", "data-warehouse-etl",
    "notification-hub", "inventory-sync-service", "fraud-detection-api"
]

def generate_projects(n: int = 1500) -> pd.DataFrame:
    """
    Generates tabular project metrics.
    Features follow typical distributions found in empirical SE research:
    - Team size: typically 5-25 engineers (Scrum two-pizza teams)
    - Seniority: beta distribution centered around ~40-60% senior
    - Scope creep: gamma distribution (right-skewed, most projects have modest creep, some severe)
    - Bug density: defects per 1,000 lines of code (KLOC)
    """
    print(f"[data-gen] Synthesizing {n} software project records...")

    # --- 1. Team & Team Dynamics ---
    # Most teams are 5 to 30 people
    team_size = np.random.randint(4, 35, size=n)
    
    # Seniority ratio (% of senior/lead engineers in the team)
    seniority_ratio = np.clip(np.random.beta(a=3.0, b=3.0, size=n), 0.15, 0.85)
    
    # Annualized turnover rate (exponential distribution, avg ~10-15%)
    team_turnover_rate = np.clip(np.random.exponential(scale=0.10, size=n), 0.0, 0.45)
    
    # Sprint velocity stability: std / mean velocity ratio (1.0 = very stable)
    velocity_stability = np.clip(np.random.normal(loc=0.74, scale=0.16, size=n), 0.25, 1.0)

    # --- 2. Scope & Planning ---
    # Projects usually run in increments of 8 to 48 weeks
    planned_weeks = np.random.choice([8, 12, 16, 24, 32, 48], size=n, p=[0.15, 0.25, 0.30, 0.15, 0.10, 0.05])
    
    # Average cost per dev-week ~ $2.5k - $3.8k
    planned_budget_k = np.round(planned_weeks * team_size * np.random.uniform(2.4, 3.6, size=n), 1)
    
    # Story points committed at kickoff (roughly 8-11 points per dev per 2-week sprint)
    story_points = np.round(planned_weeks * (team_size * np.random.uniform(7.5, 11.5, size=n)), 0).astype(int)
    
    # Scope creep % (unplanned user stories injected during development)
    # Right-skewed: most projects have 10-25% creep, but some blow up to 60%+
    scope_creep_pct = np.clip(np.random.gamma(shape=2.4, scale=9.0, size=n), 0.0, 75.0)
    
    # Requirements volatility: frequency of scope changes after sprint backlog lock (0 to 1)
    req_volatility = np.clip(np.random.beta(a=2.0, b=3.2, size=n), 0.05, 0.95)

    # --- 3. Engineering Quality & Codebase Health ---
    # Technical debt ratio: estimated refactoring debt vs active code
    tech_debt = np.clip(np.random.beta(a=2.2, b=2.8, size=n), 0.05, 0.88)
    
    # Test coverage percentage
    test_cov = np.clip(np.random.normal(loc=67.0, scale=17.0, size=n), 15.0, 98.0)
    
    # Defect density per KLOC (bugs found in staging / pre-release)
    bug_density = np.clip(np.random.gamma(shape=2.8, scale=2.6, size=n), 1.0, 26.0)
    
    # Code churn rate: fraction of lines touched multiple times in same release
    code_churn = np.clip(np.random.beta(a=2.0, b=4.0, size=n), 0.05, 0.80)
    
    # CI/CD build failure rate (flaky tests, broken trunk)
    ci_failure_rate = np.clip(np.random.beta(a=1.5, b=4.5, size=n), 0.02, 0.55)
    
    # Weekly meeting overhead (hours/week per dev)
    meeting_hrs = np.clip(np.random.normal(loc=11.5, scale=4.5, size=n), 3.0, 25.0)
    
    # Team sentiment index from retrospectives / commit comments (-1.0 to +1.0)
    sentiment = np.clip(np.random.normal(loc=0.18, scale=0.42, size=n), -0.90, 0.90)

    # --- 4. Synthesize Realistic Outcome Labels (Delay & Cost Variance) ---
    # Non-linear interaction based on Brooks' Law and agile empirical research:
    # Adding people late or high turnover + high scope creep compounds delays quadratically.
    risk_stress = (
        (scope_creep_pct / 22.0) * 1.6 +
        (req_volatility * 2.3) +
        (team_turnover_rate * 3.8) +
        ((1.0 - seniority_ratio) * 1.9) +
        (tech_debt * 2.1) +
        ((100.0 - test_cov) / 40.0) * 1.1 +
        (bug_density / 9.0) * 1.3 +
        (ci_failure_rate * 2.0) +
        ((meeting_hrs - 8.0) / 10.0) * 0.7 -
        (sentiment * 1.6) -
        (velocity_stability * 2.0)
    )

    # Add realistic environmental noise (market shifts, vendor delays, unexpected outages)
    noise = np.random.normal(0, 0.6, size=n)
    stress_total = risk_stress + noise

    # Target 1: Schedule Slip in Days
    base_slip = np.maximum(0.0, stress_total - 1.0)
    schedule_slip_days = np.round(np.clip(
        (base_slip ** 1.32) * (planned_weeks * 0.85) + np.random.normal(0, 3.0, size=n),
        0.0, 160.0
    ), 1)

    # Target 2: Cost Overrun % (relative to planned budget)
    cost_overrun_pct = np.round(np.clip(
        (schedule_slip_days / (planned_weeks * 7.0)) * 70.0 +
        (scope_creep_pct * 0.42) +
        (team_turnover_rate * 16.0) +
        np.random.normal(0, 3.5, size=n),
        0.0, 110.0
    ), 1)

    # Target 3: Delivery Risk Category (Low, Medium, High)
    risk_category = []
    for slip, cost in zip(schedule_slip_days, cost_overrun_pct):
        if slip <= 8.0 and cost <= 10.0:
            risk_category.append("Low")
        elif slip <= 26.0 and cost <= 30.0:
            risk_category.append("Medium")
        else:
            risk_category.append("High")

    # Pick project name prefixes and IDs
    project_ids = [f"PRJ-{1001 + i}" for i in range(n)]
    names = [f"{random.choice(PROJECT_NAMES)}-{100 + (i % 899)}" for i in range(n)]
    methodology = np.random.choice(["Scrum", "Kanban", "Scrumban"], size=n, p=[0.65, 0.20, 0.15])

    df = pd.DataFrame({
        "project_id": project_ids,
        "project_name": names,
        "methodology": methodology,
        "team_size": team_size,
        "seniority_ratio": np.round(seniority_ratio, 2),
        "team_turnover_rate": np.round(team_turnover_rate, 3),
        "sprint_velocity_stability": np.round(velocity_stability, 2),
        "planned_duration_weeks": planned_weeks,
        "planned_budget_k": planned_budget_k,
        "initial_story_points": story_points,
        "scope_creep_pct": np.round(scope_creep_pct, 1),
        "requirements_volatility": np.round(req_volatility, 2),
        "technical_debt_ratio": np.round(tech_debt, 2),
        "test_coverage_pct": np.round(test_cov, 1),
        "bug_density_per_kloc": np.round(bug_density, 1),
        "code_churn_rate": np.round(code_churn, 2),
        "ci_cd_failure_rate": np.round(ci_failure_rate, 2),
        "meeting_overhead_hrs": np.round(meeting_hrs, 1),
        "sentiment_score": np.round(sentiment, 2),
        # Target variables:
        "schedule_slip_days": schedule_slip_days,
        "cost_overrun_pct": cost_overrun_pct,
        "delivery_risk_category": risk_category
    })

    return df

def generate_tickets(n: int = 4000) -> pd.DataFrame:
    """
    Generates realistic Jira/GitHub issue tickets for NLP triage analysis.
    """
    print(f"[data-gen] Synthesizing {n} sprint issue tickets...")

    # Real-world engineering issues taken from public bug trackers & postmortems
    critical_templates = [
        "Memory leak in {service} pod causing continuous crash loops during load test",
        "OAuth2 token refresh fails silently; checkout flow blocked for all mobile users",
        "Data race condition in {service} leading to inconsistent balance calculations",
        "Downstream payment webhook timeout (>15s); sprint release candidate blocked",
        "Database migration deadlock in staging on orders table alter statement",
        "Unhandled null pointer in {service} gateway payload validation crashing container",
        "CI/CD deployment pipeline broken for 36 hours; release tag blocked",
        "TLS certificate expiration warning on {service} internal RPC endpoint"
    ]

    medium_templates = [
        "Intermittent HTTP 504 gateway timeout in {service} batch sync endpoint",
        "Slow SQL query in admin analytics dashboard taking over 4.2 seconds",
        "Safari iOS CSS layout shift on checkout confirmation screen",
        "Flaky integration test in {service} causing random CI pipeline retries",
        "Missing redis cache invalidation when user updates account profile settings",
        "Logging level set to DEBUG in production {service}, spiking CloudWatch bill",
        "Deprecated API call in {service} client SDK needs upgrade before next sprint",
        "Upstream swagger docs out of sync with {service} REST responses"
    ]

    low_templates = [
        "Update developer setup instructions and local docker-compose guide in README",
        "Fix typo in error message modal for invalid postal code input",
        "Bump patch versions in package.json for minor security advisories",
        "Remove unused CSS helper utilities and legacy SVG icons in {service}",
        "Add onboarding tooltip for new user dashboard filter bar",
        "Reorder navigation dropdown menu items as per UX design feedback",
        "Run automated linter fixes across {service} codebase",
        "Add helper comment explaining regex pattern in {service} validation util"
    ]

    services = [
        "PaymentGateway", "AuthService", "BillingWorker", "SearchIndexer",
        "OrderProcessor", "NotificationService", "UserRegistry", "InventorySync"
    ]

    records = []
    for i in range(n):
        roll = random.random()
        svc = random.choice(services)
        
        if roll < 0.28:
            # Critical / Blocker
            summary = random.choice(critical_templates).format(service=svc)
            urgency = "Critical"
            sentiment = round(random.uniform(-0.95, -0.35), 2)
            blocker = 1 if random.random() < 0.85 else 0
            reopen_count = random.choice([1, 2, 3])
            comments = random.randint(8, 30)
            res_days = round(random.uniform(5.0, 20.0), 1)
        elif roll < 0.68:
            # Medium / Normal
            summary = random.choice(medium_templates).format(service=svc)
            urgency = "Medium"
            sentiment = round(random.uniform(-0.30, 0.25), 2)
            blocker = 1 if random.random() < 0.20 else 0
            reopen_count = random.choice([0, 1])
            comments = random.randint(3, 12)
            res_days = round(random.uniform(2.0, 8.0), 1)
        else:
            # Low
            summary = random.choice(low_templates).format(service=svc)
            urgency = "Low"
            sentiment = round(random.uniform(0.15, 0.80), 2)
            blocker = 0
            reopen_count = 0
            comments = random.randint(1, 4)
            res_days = round(random.uniform(0.5, 3.0), 1)

        records.append({
            "ticket_id": f"DEV-{20000 + i}",
            "service": svc,
            "summary": summary,
            "sentiment_score": sentiment,
            "is_blocker": blocker,
            "reopen_count": reopen_count,
            "comments_count": comments,
            "resolution_days": res_days,
            "urgency": urgency
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df_proj = generate_projects(1500)
    p_path = os.path.join(OUTPUT_DIR, "projects_dataset.csv")
    df_proj.to_csv(p_path, index=False)
    print(f"[OK] Generated {len(df_proj)} projects -> {p_path}")

    df_tick = generate_tickets(4000)
    t_path = os.path.join(OUTPUT_DIR, "tickets_dataset.csv")
    df_tick.to_csv(t_path, index=False)
    print(f"[OK] Generated {len(df_tick)} tickets -> {t_path}")
