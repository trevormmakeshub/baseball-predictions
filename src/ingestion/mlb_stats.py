# src/ingestion/mlb_stats.py
"""Pull schedules, game results, and probable pitchers from the MLB Stats API."""

import statsapi
import pandas as pd
import requests
from datetime import datetime, timedelta
from time import sleep

from .config import config

# Completed games only. The schedule request stops on this date.
COMPLETED_STATUSES = {"Final", "Game Over", "Completed Early"}
PULL_THROUGH = "2026-09-30"


def _month_windows(start: str, end: str):
    """Yield [start, end] pairs split on month boundaries."""
    cursor = datetime.strptime(start, "%Y-%m-%d").date()
    last = datetime.strptime(end, "%Y-%m-%d").date()
    while cursor <= last:
        if cursor.month == 12:
            nxt = cursor.replace(year=cursor.year + 1, month=1, day=1)
        else:
            nxt = cursor.replace(month=cursor.month + 1, day=1)
        chunk_end = min(last, nxt - timedelta(days=1))
        yield cursor.isoformat(), chunk_end.isoformat()
        cursor = chunk_end + timedelta(days=1)


def _fetch_range(start: str, end: str, depth: int = 0) -> list:
    """Fetch one schedule window. Split it if the MLB Stats API times out."""
    try:
        games = statsapi.schedule(start_date=start, end_date=end, sportId=1)
        sleep(0.25)
        return games
    except requests.HTTPError as exc:
        start_d = datetime.strptime(start, "%Y-%m-%d").date()
        end_d = datetime.strptime(end, "%Y-%m-%d").date()
        if depth >= 6 or start_d >= end_d:
            raise
        mid = start_d + (end_d - start_d) // 2
        print(f"  split {start}..{end} after HTTP {exc.response.status_code}")
        sleep(1.0)
        left = _fetch_range(start, mid.isoformat(), depth + 1)
        right = _fetch_range((mid + timedelta(days=1)).isoformat(), end, depth + 1)
        return left + right


def fetch_season_schedule(year: int) -> pd.DataFrame:
    """Fetch completed games for a season through 2026-09-30.

    Returns DataFrame with: game_id, date, away_team, home_team,
    away_score, home_score, status, venue, away_pitcher, home_pitcher.
    """
    start = f"{year}-02-20"  # Spring training start
    end = min(f"{year}-11-05", PULL_THROUGH)

    print(f"Fetching {year} schedule through {end}...")
    games = []
    for chunk_start, chunk_end in _month_windows(start, end):
        games.extend(_fetch_range(chunk_start, chunk_end))

    rows = []
    for g in games:
        rows.append({
            "game_id": g["game_id"],
            "date": g["game_date"],
            "away_team": g["away_name"],
            "home_team": g["home_name"],
            "away_score": g.get("away_score"),
            "home_score": g.get("home_score"),
            "status": g["status"],
            "venue": g.get("venue_name", ""),
            "away_probable_pitcher": g.get("away_probable_pitcher", "TBD"),
            "home_probable_pitcher": g.get("home_probable_pitcher", "TBD"),
            "series_description": g.get("series_description", ""),
            "game_type": g.get("game_type", "R"),  # R=regular, P=postseason
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["date"] <= PULL_THROUGH]
    df = df[df["status"].isin(COMPLETED_STATUSES)]
    df = df[df["away_score"].notna() & df["home_score"].notna()]
    df = df.drop_duplicates(subset=["game_id"])
    return df.reset_index(drop=True)


def fetch_all_schedules() -> pd.DataFrame:
    """Fetch completed games for all configured years and save them."""
    all_dfs: list[pd.DataFrame] = []
    for year in range(config.start_year, min(config.end_year, 2026) + 1):
        df = fetch_season_schedule(year)
        if df.empty:
            print(f"  {year}: no completed games")
            sleep(config.request_delay_sec)
            continue
        # Filter to regular season + postseason only
        df = df[df["game_type"].isin(["R", "F", "D", "L", "W"])]
        outpath = config.raw_dir / "gamelogs" / f"schedule_{year}.csv"
        df.to_csv(outpath, index=False)
        print(f"  Saved {len(df)} games → {outpath}")
        all_dfs.append(df)
        sleep(config.request_delay_sec)

    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(config.raw_dir / "gamelogs" / "schedule_all.csv", index=False)
    tracked = config.processed_dir / "completed_games.parquet"
    combined.to_parquet(tracked, index=False)
    print(f"min date {combined['date'].min()}")
    print(f"max date {combined['date'].max()}")
    print(f"rows {len(combined)}")
    return combined


def fetch_todays_probable_pitchers() -> pd.DataFrame:
    """Fetch today's games with probable pitchers (for daily picks)."""
    today = datetime.now().strftime("%Y-%m-%d")
    games = statsapi.schedule(date=today)
    rows = []
    for g in games:
        rows.append({
            "game_id": g["game_id"],
            "date": today,
            "away_team": g["away_name"],
            "home_team": g["home_name"],
            "away_probable_pitcher": g.get("away_probable_pitcher", "TBD"),
            "home_probable_pitcher": g.get("home_probable_pitcher", "TBD"),
            "venue": g.get("venue_name", ""),
            "game_time": g.get("game_datetime", ""),
        })
    return pd.DataFrame(rows)


def fetch_game_pace(year: int) -> pd.DataFrame:
    """Fetch game-pace metrics for a season from the MLB Stats API.

    Includes average innings pitched, game duration, pitches per plate appearance,
    and total runs per game — useful context for totals model calibration.

    Args:
        year: The MLB season.

    Returns:
        DataFrame with columns: season, league, games, avg_game_duration_min,
        avg_innings, runs_per_game, pitches_per_pa.
        Returns an empty DataFrame if the endpoint is unavailable.
    """
    try:
        data = statsapi.get(
            "schedule_games_pace",
            {"season": year, "sportId": 1},
        )
        items = data.get("gamesPaced", [])
        rows = []
        for item in items:
            rows.append({
                "season": year,
                "league": item.get("leagueAbbreviation", "MLB"),
                "games": item.get("gamesPlayed"),
                "avg_game_duration_min": item.get("avgGameDurationMinutes"),
                "avg_innings": item.get("avgInningsPlayed"),
                "runs_per_game": item.get("runsPerGame"),
                "pitches_per_pa": item.get("pitchesPerPlateAppearance"),
            })
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "fetch_game_pace failed for %d: %s", year, exc
        )
        return pd.DataFrame()


def fetch_streaks(year: int, streak_type: str = "wins", threshold: int = 4) -> pd.DataFrame:
    """Fetch current hot/cold streaks for teams via the MLB Stats API.

    Uses the ``/stats/streaks`` endpoint to identify teams on notable
    win or loss streaks — a signal for short-term momentum in moneyline models.

    Args:
        year:        The MLB season.
        streak_type: "wins" or "losses".
        threshold:   Minimum streak length to include.

    Returns:
        DataFrame with columns: team, streak_type, streak_length, season.
        Returns an empty DataFrame if the endpoint is unavailable.
    """
    try:
        stat_type = "wins" if streak_type.lower() == "wins" else "losses"
        data = statsapi.get(
            "stats_streaks",
            {
                "season": year,
                "sportId": 1,
                "streakType": stat_type,
                "streakSpan": "career",
                "gameType": "R",
                "limit": 50,
            },
        )
        rows = []
        for entry in data.get("streaks", []):
            length = entry.get("streakLength", 0)
            if length >= threshold:
                team_info = entry.get("team", {}) or entry.get("player", {})
                rows.append({
                    "team": team_info.get("name", ""),
                    "streak_type": stat_type,
                    "streak_length": length,
                    "season": year,
                })
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "fetch_streaks failed for %d/%s: %s", year, streak_type, exc
        )
        return pd.DataFrame()


if __name__ == "__main__":
    fetch_all_schedules()
