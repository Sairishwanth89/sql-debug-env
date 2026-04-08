"""
graders/sql_grader.py — SQLGrader class for OpenEnv Phase 2 validation.
Called by the OpenEnv validator to score each task submission.
Score must be strictly between 0 and 1 (never 0.0 or 1.0).
"""


class SQLGrader:
    """
    Grader for all SQL Debug tasks.
    Evaluates a fixed SQL submission and returns a score in (0, 1).
    """

    # Per-task solution keywords — presence indicates a correct fix
    TASK_SIGNALS = {
        "task_1_easy": [","],
        "task_2_medium": ["GROUP BY"],
        "task_3_hard": ["PARTITION BY"],
        "task_4_expert": ["2024-12", "12-01"],
        "task_5_optimization": ["INNER JOIN", "JOIN"],
        "task_6_migration": ["INSERT INTO", "DROP"],
        "task_7_chaos": ["UNIQUE", "COALESCE"],
    }

    def grade(self, task_id: str, fixed_sql: str, **kwargs) -> float:
        """
        Grade a SQL submission.

        Args:
            task_id: The task identifier (e.g. 'task_1_easy')
            fixed_sql: The agent's submitted SQL fix

        Returns:
            float strictly in (0, 1)
        """
        signals = self.TASK_SIGNALS.get(task_id, [])
        sql_upper = (fixed_sql or "").upper()

        if not signals:
            return 0.5  # Unknown task — neutral score

        hits = sum(1 for s in signals if s.upper() in sql_upper)
        raw = hits / len(signals)

        # Map to (0.1, 0.9) — never touches 0.0 or 1.0
        score = 0.1 + raw * 0.8
        return round(max(0.01, min(0.99, score)), 4)
