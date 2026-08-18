# Verify

A reusable verification graph. The main agent reads this file and spawns children. There is no graph runtime.

1. Split the claim into independent sub-facts.
2. Spawn one `explore` child per sub-fact.
3. Stay pending until every child Observation is back.
4. Read each `outcome`. Write one report: succeeded, failed, or partial.
5. Do not spawn a separate merge agent. The parent writes the report.
